"""
Variant Prioritization Service - Research-backed pathogenicity scoring.

Implements CPU-friendly, evidence-grounded variant pathogenicity ranking using
a model stack based on Hugging Face artifacts:
- Primary: XGBoost with precomputed missense features (GPN-MSA, CADD, phyloP, ESM-1b, etc.)
- Fallback 1: Random Forest for protein/AA-change features
- Fallback 2: Logistic Regression ultra-light model

Scientific foundations from:
- Frazer et al. 2021 (ESM-1b embeddings)
- Cheng et al. 2023 (AlphaMissense, precomputed conservation)
- Notin et al. 2022 (Tranception)
- Landrum et al. 2018 (ClinVar ground truth)

All models run CPU-only, lazy-load with in-process caching, and gracefully
degrade on artifact unavailability.
"""

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports - will only be imported when actually needed
_hf_hub = None
_joblib = None
_xgb = None
_sklearn = None

logger = logging.getLogger(__name__)


# Tier definitions for pathogenicity scores
TIER_THRESHOLDS = {
    1: 0.90,  # Tier 1: Very high pathogenicity confidence
    2: 0.70,  # Tier 2: High pathogenicity confidence
    3: 0.50,  # Tier 3: Moderate pathogenicity confidence
}

# Default Hugging Face repository settings
DEFAULT_HF_REPO_ID = "omshrivastava/omnibimol-variant-priority"
DEFAULT_HF_REVISION = "main"
DEFAULT_CACHE_PATH = "./cache/hf_artifacts"

# Precomputed missense feature names (order matters for schema validation)
PRE_COMPUTED_FEATURES = [
    "gpn_msa_score",
    "cadd_raw",
    "cadd_phred",
    "phyloP100way_vertebrate",
    "phyloP241way_mammalian",
    "phastCons100way_vertebrate",
    "phastCons241way_mammalian",
    "esm1b_embedding_mean",
    "esm1b_embedding_max",
    "esm1b_embedding_norm",
    "nt_score",
    "hyena_dna_embedding_mean",
]

# Protein-level / AA-change feature names
PROTEIN_FEATURES = [
    "aa_position",
    "aa_change_type",
    "domain_score",
    "conservation_score",
    "blosum62_score",
    "grantham_distance",
    "sift_score",
    "polyphen_score",
]


def _lazy_import_hf_hub():
    """Lazy import of huggingface_hub to avoid unnecessary dependencies."""
    global _hf_hub
    if _hf_hub is None:
        try:
            from huggingface_hub import snapshot_download
            _hf_hub = snapshot_download
        except ImportError:
            _hf_hub = False  # Mark as unavailable
    return _hf_hub if _hf_hub else None


def _lazy_import_joblib():
    """Lazy import of joblib for model loading."""
    global _joblib
    if _joblib is None:
        try:
            import joblib
            _joblib = joblib
        except ImportError:
            _joblib = False
    return _joblib if _joblib else None


def _lazy_import_xgboost():
    """Lazy import of xgboost."""
    global _xgb
    if _xgb is None:
        try:
            import xgboost as xgb
            _xgb = xgb
        except ImportError:
            _xgb = False
    return _xgb if _xgb else None


def _lazy_import_sklearn():
    """Lazy import of sklearn."""
    global _sklearn
    if _sklearn is None:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            _sklearn = {
                "RandomForestClassifier": RandomForestClassifier,
                "LogisticRegression": LogisticRegression,
                "StandardScaler": StandardScaler,
            }
        except ImportError:
            _sklearn = False
    return _sklearn if _sklearn else None


class VariantPrioritizer:
    """
    Research-backed variant pathogenicity prioritization service.
    
    Implements a three-tier model stack for missense variant scoring:
    
    1. Primary: XGBoost with comprehensive precomputed features
       - Requires: Full precomputed missense feature set
       - Features: GPN-MSA, CADD, phyloP, phastCons, ESM-1b, NT, HyenaDNA
       - Output: Pathogenicity probability [0, 1]
    
    2. Fallback 1: Random Forest for protein-level features
       - Triggered when: Precomputed features missing but AA-change features available
       - Features: Position, conservation, physicochemical properties
       - Output: Pathogenicity probability [0, 1]
    
    3. Fallback 2: Logistic Regression ultra-light model
       - Triggered when: Only minimal features available
       - Output: Pathogenicity probability [0, 1]
    
    All models use pre-trained imputers and scalers for deterministic handling
    of missing values. Feature schema is strictly versioned to prevent drift.
    
    Args:
        hf_repo_id: Hugging Face repository ID for model artifacts
        hf_revision: Git revision (branch/tag/commit) to use
        cache_path: Local directory for cached artifacts
        enable_remote_download: Whether to download from Hugging Face if not cached
        strict_schema: Whether to enforce strict feature schema validation
    """
    
    def __init__(
        self,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_revision: str = DEFAULT_HF_REVISION,
        cache_path: str = DEFAULT_CACHE_PATH,
        enable_remote_download: bool = True,
        strict_schema: bool = True,
    ):
        self.hf_repo_id = hf_repo_id
        self.hf_revision = hf_revision
        self.cache_path = Path(cache_path)
        self.enable_remote_download = enable_remote_download
        self.strict_schema = strict_schema
        
        # Model state
        self._xgb_model = None
        self._xgb_imputer = None
        self._xgb_scaler = None
        self._rf_model = None
        self._rf_imputer = None
        self._rf_scaler = None
        self._lr_model = None
        self._lr_imputer = None
        self._lr_scaler = None
        self._metrics_summary = None
        
        self._model_loaded = False
        self._load_lock = False  # Simple lock for lazy loading
        
        # Version info for reproducibility
        self.artifact_versions = {
            "xgb_precomputed": None,
            "rf_protein": None,
            "lr_protein": None,
            "research_memo": None,
        }
    
    def _ensure_models_loaded(self) -> bool:
        """Lazy-load model artifacts on first use."""
        if self._model_loaded:
            return True
        if self._load_lock:
            # Another thread/process is loading; skip to avoid recursion
            return False
        
        self._load_lock = True
        try:
            return self._load_model_artifacts()
        finally:
            self._load_lock = False
    
    def _load_model_artifacts(self) -> bool:
        """Load model artifacts from Hugging Face with local caching."""
        repo_path = self._get_or_download_artifacts()
        if repo_path is None:
            logger.warning(
                "VariantPrioritizer: Could not load model artifacts. "
                "Remote download may be disabled or repo unavailable."
            )
            return False
        
        # Load XGBoost stack
        xgb_loaded = self._load_xgboost_models(repo_path)
        
        # Load Random Forest fallback
        rf_loaded = self._load_rf_models(repo_path)
        
        # Load Logistic Regression fallback
        lr_loaded = self._load_lr_models(repo_path)
        
        # Load metrics summary for explainability
        self._load_metrics_summary(repo_path)
        
        # Load version metadata
        self._load_version_info(repo_path)
        
        success = xgb_loaded or rf_loaded or lr_loaded
        if success:
            self._model_loaded = True
            logger.info(
                f"VariantPrioritizer: Loaded models from {repo_path}. "
                f"XGB={xgb_loaded}, RF={rf_loaded}, LR={lr_loaded}"
            )
        
        return success
    
    def _get_or_download_artifacts(self) -> Optional[Path]:
        """Get cached artifacts or download from Hugging Face."""
        # Check local cache first
        cached_repo = self.cache_path / self.hf_repo_id.replace("/", "--")
        cached_repo.mkdir(parents=True, exist_ok=True)
        if self._has_artifacts(cached_repo):
            return cached_repo
        
        if not self.enable_remote_download:
            logger.warning(
                "VariantPrioritizer: Remote download disabled and no cached artifacts found."
            )
            return None
        
        # Download from Hugging Face
        snapshot_download = _lazy_import_hf_hub()
        if snapshot_download is None:
            logger.warning(
                "VariantPrioritizer: huggingface_hub not installed. "
                "Install with: pip install huggingface_hub"
            )
            return None
        
        try:
            logger.info(
                f"Downloading model artifacts from {self.hf_repo_id} "
                f"(revision: {self.hf_revision})..."
            )
            repo_path = snapshot_download(
                repo_id=self.hf_repo_id,
                revision=self.hf_revision,
                cache_dir=str(self.cache_path.parent),
                local_dir_use_symlinks=False,
                local_dir=str(cached_repo),
                allow_patterns=[
                    "xgb_precomputed.json",
                    "xgb_precomputed.pkl",
                    "xgb_precomputed_imp.pkl",
                    "xgb_precomputed_scaler.pkl",
                    "rf_protein.pkl",
                    "rf_protein_imp.pkl",
                    "rf_protein_scaler.pkl",
                    "lr_protein.pkl",
                    "lr_protein_imp.pkl",
                    "lr_protein_scaler.pkl",
                    "metrics_summary.json",
                    "research_memo.md",
                ],
            )
            downloaded_repo = Path(repo_path)
            return downloaded_repo if self._has_artifacts(downloaded_repo) else None
        except Exception as e:
            logger.error(f"Failed to download model artifacts: {e}")
            return cached_repo if self._has_artifacts(cached_repo) else None

    @staticmethod
    def _has_artifacts(repo_path: Path) -> bool:
        """Check whether a cache directory contains at least one expected artifact."""
        expected_files = [
            "xgb_precomputed.json",
            "xgb_precomputed.pkl",
            "xgb_precomputed_imp.pkl",
            "xgb_precomputed_scaler.pkl",
            "rf_protein.pkl",
            "rf_protein_imp.pkl",
            "rf_protein_scaler.pkl",
            "lr_protein.pkl",
            "lr_protein_imp.pkl",
            "lr_protein_scaler.pkl",
            "metrics_summary.json",
            "research_memo.md",
        ]
        return any((repo_path / filename).exists() for filename in expected_files)
    
    def _load_xgboost_models(self, repo_path: Path) -> bool:
        """Load XGBoost model and preprocessing artifacts."""
        joblib = _lazy_import_joblib()
        xgb = _lazy_import_xgboost()
        
        if joblib is None or xgb is False:
            return False
        
        try:
            # Load model config
            model_config_path = repo_path / "xgb_precomputed.json"
            if model_config_path.exists():
                with open(model_config_path) as f:
                    model_config = json.load(f)
            else:
                return False
            
            # Load model (try pickle first, then try to construct from config)
            model_path = repo_path / "xgb_precomputed.pkl"
            if model_path.exists():
                self._xgb_model = joblib.load(str(model_path))
            else:
                # Try loading as JSON model config
                import json
                self._xgb_model = xgb.XGBClassifier()
                # Note: In practice, you'd need the actual model binary
                # This is a fallback
                logger.warning("XGBoost model binary not found, using config only")
            
            # Load imputer
            imputer_path = repo_path / "xgb_precomputed_imp.pkl"
            if imputer_path.exists():
                self._xgb_imputer = joblib.load(str(imputer_path))
            
            # Load scaler
            scaler_path = repo_path / "xgb_precomputed_scaler.pkl"
            if scaler_path.exists():
                self._xgb_scaler = joblib.load(str(scaler_path))
            
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost models: {e}")
            self._xgb_model = None
            self._xgb_imputer = None
            self._xgb_scaler = None
            return False
    
    def _load_rf_models(self, repo_path: Path) -> bool:
        """Load Random Forest fallback models."""
        joblib = _lazy_import_joblib()
        sklearn = _lazy_import_sklearn()
        
        if joblib is None or sklearn is False:
            return False
        
        try:
            model_path = repo_path / "rf_protein.pkl"
            if not model_path.exists():
                return False
            
            self._rf_model = joblib.load(str(model_path))
            
            imputer_path = repo_path / "rf_protein_imp.pkl"
            if imputer_path.exists():
                self._rf_imputer = joblib.load(str(imputer_path))
            
            scaler_path = repo_path / "rf_protein_scaler.pkl"
            if scaler_path.exists():
                self._rf_scaler = joblib.load(str(scaler_path))
            
            return True
        except Exception as e:
            logger.error(f"Failed to load Random Forest models: {e}")
            self._rf_model = None
            self._rf_imputer = None
            self._rf_scaler = None
            return False
    
    def _load_lr_models(self, repo_path: Path) -> bool:
        """Load Logistic Regression fallback models."""
        joblib = _lazy_import_joblib()
        sklearn = _lazy_import_sklearn()
        
        if joblib is None or sklearn is False:
            return False
        
        try:
            model_path = repo_path / "lr_protein.pkl"
            if not model_path.exists():
                return False
            
            self._lr_model = joblib.load(str(model_path))
            
            imputer_path = repo_path / "lr_protein_imp.pkl"
            if imputer_path.exists():
                self._lr_imputer = joblib.load(str(imputer_path))
            
            scaler_path = repo_path / "lr_protein_scaler.pkl"
            if scaler_path.exists():
                self._lr_scaler = joblib.load(str(scaler_path))
            
            return True
        except Exception as e:
            logger.error(f"Failed to load Logistic Regression models: {e}")
            self._lr_model = None
            self._lr_imputer = None
            self._lr_scaler = None
            return False
    
    def _load_metrics_summary(self, repo_path: Path):
        """Load metrics summary for explainability."""
        metrics_path = repo_path / "metrics_summary.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    self._metrics_summary = json.load(f)
                logger.info(f"Loaded metrics summary with keys: {list(self._metrics_summary.keys())}")
            except Exception as e:
                logger.error(f"Failed to load metrics summary: {e}")
    
    def _load_version_info(self, repo_path: Path):
        """Load version information for reproducibility."""
        for key, filename in self.artifact_versions.items():
            file_path = repo_path / f"{key}.json"
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        metadata = json.load(f)
                        self.artifact_versions[key] = metadata.get("version", "unknown")
                except Exception:
                    pass
    
    def predict_pathogenicity(
        self,
        features: Dict[str, Any],
        feature_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Predict variant pathogenicity using the appropriate model tier.
        
        Args:
            features: Dictionary of variant features. Expected keys depend on feature_type:
                - For 'precomputed' (XGBoost): All PRE_COMPUTED_FEATURES keys
                - For 'protein' (Random Forest): PROTEIN_FEATURES keys
                - For 'auto': Automatically detect best available feature set
            feature_type: One of 'auto', 'precomputed', 'protein', or 'minimal'
        
        Returns:
            Dictionary with prediction results:
            {
                "score": float,  # Pathogenicity probability [0, 1]
                "model_used": str,  # e.g., "xgb_precomputed", "rf_protein", "lr_protein"
                "confidence_label": str,  # "high", "medium", "low"
                "evidence_features_used": List[str],
                "missing_features": List[str],
                "fallback_reason": Optional[str],
                "tier": int,  # 1, 2, or 3 (or None if score unavailable)
                "metadata": Dict  # Model confidence intervals, etc.
            }
        """
        if not self._ensure_models_loaded():
            return {
                "score": None,
                "model_used": None,
                "confidence_label": "low",
                "evidence_features_used": [],
                "missing_features": list(features.keys()),
                "fallback_reason": "models_not_available",
                "tier": None,
                "metadata": {"warning": "Model artifacts not loaded. Scoring unavailable."},
            }
        
        # Auto-detect feature type if needed
        if feature_type == "auto":
            feature_type = self._detect_feature_type(features)
        elif feature_type in {"precomputed", "protein"}:
            validation = self.validate_features(features, feature_type)
            if not validation["valid"]:
                feature_type = self._detect_feature_type(features)
        
        # Route to appropriate scoring method
        if feature_type == "precomputed" and self._xgb_model is not None:
            result = self._score_xgboost(features)
        elif feature_type == "protein" and self._rf_model is not None:
            result = self._score_random_forest(features)
        else:
            # Fallback to LR or whatever is available
            result = self._score_logistic_regression(features)
        
        # Add tier information
        result["tier"] = self._assign_tier(result["score"])
        
        return result
    
    def _detect_feature_type(self, features: Dict[str, Any]) -> str:
        """Auto-detect which feature set is available."""
        if self.validate_features(features, "precomputed")["valid"]:
            return "precomputed"
        if self.validate_features(features, "protein")["valid"]:
            return "protein"
        return "minimal"
    
    def _score_xgboost(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score using XGBoost with precomputed missense features."""
        # Extract and validate features
        feature_vector, missing = self._extract_features(
            features, PRE_COMPUTED_FEATURES, "precomputed"
        )
        
        if len(feature_vector) == 0:
            # No features available; fallback
            return {
                "score": 0.5,
                "model_used": "xgb_precomputed",
                "confidence_label": "low",
                "evidence_features_used": [],
                "missing_features": PRE_COMPUTED_FEATURES,
                "fallback_reason": "no_features_available",
                "metadata": {},
            }
        
        # Impute missing values
        if self._xgb_imputer is not None:
            try:
                feature_vector = self._xgb_imputer.transform([feature_vector])[0]
            except Exception as e:
                logger.warning(f"XGBoost imputation failed: {e}")
        
        # Scale if scaler available
        if self._xgb_scaler is not None:
            try:
                feature_vector = self._xgb_scaler.transform([feature_vector])[0]
            except Exception as e:
                logger.warning(f"XGBoost scaling failed: {e}")
        
        # Predict
        if self._xgb_model is not None:
            try:
                # Handle both sklearn-like and native xgboost models
                if hasattr(self._xgb_model, "predict_proba"):
                    proba = self._xgb_model.predict_proba([feature_vector])[0]
                    score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    # Native xgboost
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix([feature_vector])
                    pred = self._xgb_model.predict(dmatrix)
                    score = float(pred[0])
                
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                
                return {
                    "score": score,
                    "model_used": "xgb_precomputed",
                    "confidence_label": "high",
                    "evidence_features_used": [
                        f for f in PRE_COMPUTED_FEATURES if f not in missing
                    ],
                    "missing_features": missing,
                    "fallback_reason": None,
                    "metadata": self._get_metadata("xgb_precomputed"),
                }
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
        
        # Fallback
        return self._score_random_forest(features)
    
    def _score_random_forest(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score using Random Forest fallback."""
        feature_vector, missing = self._extract_features(
            features, PROTEIN_FEATURES, "protein"
        )
        
        # Impute and scale
        if self._rf_imputer is not None:
            try:
                feature_vector = self._rf_imputer.transform([feature_vector])[0]
            except Exception:
                pass
        
        if self._rf_scaler is not None:
            try:
                feature_vector = self._rf_scaler.transform([feature_vector])[0]
            except Exception:
                pass
        
        # Predict
        if self._rf_model is not None:
            try:
                if hasattr(self._rf_model, "predict_proba"):
                    proba = self._rf_model.predict_proba([feature_vector])[0]
                    score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    score = float(self._rf_model.predict([feature_vector])[0])
                
                score = max(0.0, min(1.0, score))
                
                return {
                    "score": score,
                    "model_used": "rf_protein",
                    "confidence_label": "medium",
                    "evidence_features_used": [
                        f for f in PROTEIN_FEATURES if f not in missing
                    ],
                    "missing_features": missing,
                    "fallback_reason": "precomputed_features_missing" if missing else None,
                    "metadata": self._get_metadata("rf_protein"),
                }
            except Exception as e:
                logger.error(f"Random Forest prediction failed: {e}")
        
        # Fallback to LR
        return self._score_logistic_regression(features)
    
    def _score_logistic_regression(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score using Logistic Regression ultra-light fallback."""
        # Use whatever features are available
        available_features = list(features.keys())
        feature_vector = [float(features.get(f, 0)) for f in available_features]
        
        # Ensure we have something to work with
        if len(feature_vector) == 0:
            feature_vector = [0.0]
            available_features = ["intercept"]
        
        # Predict
        if self._lr_model is not None:
            try:
                # Impute and scale
                if self._lr_imputer is not None:
                    feature_vector = self._lr_imputer.transform([feature_vector])[0]
                if self._lr_scaler is not None:
                    feature_vector = self._lr_scaler.transform([feature_vector])[0]
                
                if hasattr(self._lr_model, "predict_proba"):
                    proba = self._lr_model.predict_proba([feature_vector])[0]
                    score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    raw = float(self._lr_model.predict([feature_vector])[0])
                    score = 1.0 / (1.0 + np.exp(-raw))  # Sigmoid
                
                score = max(0.0, min(1.0, score))
                
                return {
                    "score": score,
                    "model_used": "lr_protein",
                    "confidence_label": "low",
                    "evidence_features_used": available_features,
                    "missing_features": [],
                    "fallback_reason": "minimal_features_available",
                    "metadata": self._get_metadata("lr_protein"),
                }
            except Exception as e:
                logger.error(f"Logistic Regression prediction failed: {e}")
        
        # Ultimate fallback: return neutral score
        return {
            "score": 0.5,
            "model_used": None,
            "confidence_label": "low",
            "evidence_features_used": available_features,
            "missing_features": [],
            "fallback_reason": "all_models_failed",
            "metadata": {"warning": "All models failed; returning default"},
        }
    
    def _extract_features(
        self,
        features: Dict[str, Any],
        expected_features: List[str],
        feature_set_name: str,
    ) -> Tuple[List[float], List[str]]:
        """Extract feature values in expected order, tracking missing values."""
        values = []
        missing = []
        
        for feat in expected_features:
            if feat in features and features[feat] is not None:
                try:
                    val = float(features[feat])
                    values.append(val)
                except (TypeError, ValueError):
                    values.append(0.0)
                    missing.append(feat)
                    if self.strict_schema:
                        logger.warning(
                            f"Feature '{feat}' in '{feature_set_name}' "
                            f"has non-numeric value: {features[feat]}"
                        )
            else:
                values.append(0.0)
                missing.append(feat)
        
        return values, missing
    
    def _assign_tier(self, score: Optional[float]) -> Optional[int]:
        """Assign tier based on pathogenicity score."""
        if score is None:
            return None
        # Sort by threshold value descending to check highest thresholds first
        for tier, threshold in sorted(TIER_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return tier
        return 3  # Below 0.5 is still Tier 3
    
    def _get_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata for explainability."""
        metadata = {
            "model_name": model_name,
            "artifact_versions": self.artifact_versions,
            "feature_sets": {
                "precomputed": PRE_COMPUTED_FEATURES,
                "protein": PROTEIN_FEATURES,
            },
        }
        
        if self._metrics_summary:
            # Add model performance metrics if available
            for key in ["xgb_precomputed", "rf_protein", "lr_protein"]:
                if key in self._metrics_summary:
                    metadata[f"{key}_metrics"] = self._metrics_summary[key]
        
        return metadata
    
    def batch_predict(
        self,
        variants: List[Dict[str, Any]],
        feature_type: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Score multiple variants in batch.
        
        Args:
            variants: List of variant feature dictionaries
            feature_type: Feature type for all variants
        
        Returns:
            List of prediction results (one per variant)
        """
        if not self._ensure_models_loaded():
            return [{
                "score": None,
                "model_used": None,
                "confidence_label": "low",
                "evidence_features_used": [],
                "missing_features": [],
                "fallback_reason": "models_not_available",
                "tier": None,
                "metadata": {"warning": "Models not available"},
            } for _ in variants]
        
        results = []
        for variant in variants:
            try:
                result = self.predict_pathogenicity(variant, feature_type=feature_type)
            except Exception as e:
                logger.error(f"Batch prediction failed for variant: {e}")
                result = {
                    "score": None,
                    "model_used": None,
                    "confidence_label": "low",
                    "evidence_features_used": [],
                    "missing_features": [],
                    "fallback_reason": f"prediction_error: {e}",
                    "tier": None,
                    "metadata": {},
                }
            results.append(result)
        
        return results
    
    def validate_features(
        self,
        features: Dict[str, Any],
        feature_type: str = "precomputed",
    ) -> Dict[str, Any]:
        """
        Validate feature dictionary against expected schema.
        
        Args:
            features: Feature dictionary to validate
            feature_type: Expected feature type
        
        Returns:
            Dictionary with validation results
        """
        if feature_type == "precomputed":
            expected = PRE_COMPUTED_FEATURES
        elif feature_type == "protein":
            expected = PROTEIN_FEATURES
        else:
            expected = []
        
        present = []
        missing = []
        invalid = []
        
        for feat in expected:
            if feat not in features:
                missing.append(feat)
            elif features[feat] is None:
                missing.append(feat)
            else:
                try:
                    value = float(features[feat])
                    if math.isfinite(value):
                        present.append(feat)
                    else:
                        invalid.append(feat)
                except (TypeError, ValueError):
                    invalid.append(feat)
        
        completeness = len(present) / len(expected) if expected else 0
        
        return {
            "valid": len(invalid) == 0 and completeness >= 0.8,
            "completeness": round(completeness, 4),
            "present": present,
            "missing": missing,
            "invalid": invalid,
            "expected_features": expected,
        }