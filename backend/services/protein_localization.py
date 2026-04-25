"""CPU-friendly protein subcellular localization inference service."""

from __future__ import annotations

import logging
import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from backend.core.config import Settings, get_settings


LOGGER = logging.getLogger(__name__)

LOCALIZATION_LABELS = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_BASE_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
DEFAULT_CACHE_DIR = Path("cache") / "hf_artifacts"


class LocalizationBackend(Protocol):
    def predict_proba(self, sequence: str) -> dict[str, Any]:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def health_snapshot(self) -> dict[str, Any]:
        ...


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _lazy_import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download
    except ImportError:
        return None


def _lazy_import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _lazy_import_transformers():
    try:
        from transformers import EsmModel, EsmTokenizer

        return EsmModel, EsmTokenizer
    except ImportError:
        return None, None


def clean_protein_sequence(sequence: str, *, max_seq_len: int | None = None) -> str:
    """Normalize and validate a protein sequence for localization inference."""

    if not isinstance(sequence, str):
        raise ValueError("Protein sequence must be a string")

    text = sequence.replace("\r", "").strip().upper()
    if text.startswith(">"):
        text = "".join(line.strip() for line in text.splitlines()[1:])
    else:
        text = "".join(text.split())

    if not text:
        raise ValueError("Protein sequence cannot be empty")

    invalid_chars = set(text) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        raise ValueError(
            "Protein localization only accepts standard amino acids (ACDEFGHIKLMNPQRSTVWY). "
            f"Invalid characters: {', '.join(sorted(invalid_chars))}"
        )

    if len(text) < 20:
        raise ValueError("Protein sequence must be at least 20 amino acids long")

    if max_seq_len is not None and len(text) > max_seq_len:
        raise ValueError(f"Protein sequence exceeds maximum length of {max_seq_len} amino acids")

    return text


def compute_length_factor(sequence_length: int) -> float:
    """Deterministic length preference centered around ~300 aa."""

    return _clamp(1.0 - abs(sequence_length - 300) / 1000.0, 0.5, 1.0)


def compute_wetlab_prioritization_score(
    *,
    non_membrane_probability: float,
    confidence: float,
    sequence_length: int,
) -> float:
    """Compute a 0-100 wet-lab prioritization score.

    Formula: 100 * non_membrane_probability * confidence * length_factor.
    The length factor is centered around ~300 aa and clamped to [0.5, 1.0].
    """

    length_factor = compute_length_factor(sequence_length)
    score = 100.0 * _clamp(non_membrane_probability) * _clamp(confidence) * length_factor
    return round(_clamp(score, 0.0, 100.0), 1)


def recommend_assay(*, membrane_risk: float, confidence: float, confidence_threshold: float) -> str:
    """Recommend the next wet-lab assay based on localization evidence."""

    if membrane_risk > 0.5:
        return "Cell-free / Detergent solubilization / Nanodiscs"
    if confidence >= max(0.75, confidence_threshold):
        return "Standard E. coli expression + soluble tag"
    return "Confirmatory localization assay first (e.g. immunofluorescence or tag fusion)"


@dataclass(frozen=True)
class LocalizationArtifactBundle:
    tokenizer: Any | None
    esm_model: Any | None
    esm_lr: Any | None
    heuristic_lr: Any | None
    binary_lr: Any | None
    esm_rf: Any | None
    heuristic_rf: Any | None
    metadata: dict[str, Any]


class RuleBasedLocalizationBackend:
    """Fallback backend when HF artifacts are unavailable."""

    def __init__(self, *, metadata: dict[str, Any]):
        self._metadata = metadata

    def predict_proba(self, sequence: str) -> dict[str, Any]:
        length = len(sequence)
        composition = {aa: sequence.count(aa) / length for aa in STANDARD_AMINO_ACIDS}
        hydrophobic = sum(sequence.count(aa) for aa in set("VLIMFWYC")) / length
        charged_pos = sum(sequence.count(aa) for aa in set("RKH")) / length
        charged_neg = sum(sequence.count(aa) for aa in set("DE")) / length
        polar = sum(sequence.count(aa) for aa in set("STNQ")) / length
        aromatic = sum(sequence.count(aa) for aa in set("FWY")) / length

        membrane_risk = _clamp(0.45 * hydrophobic + 0.20 * aromatic + 0.15 * polar + 0.10 * charged_pos + 0.10 * charged_neg)
        confidence = _clamp(0.55 + 0.25 * abs(hydrophobic - polar) + 0.10 * aromatic)
        non_membrane = 1.0 - membrane_risk

        raw_scores = [
            0.22 * composition["A"] + 0.18 * composition["G"] + 0.10 * composition["S"],
            0.22 * composition["K"] + 0.20 * composition["R"] + 0.12 * composition["H"],
            0.18 * composition["E"] + 0.18 * composition["D"],
            membrane_risk,
            0.15 * composition["M"] + 0.12 * composition["I"] + 0.10 * composition["L"],
            0.12 * composition["C"] + 0.10 * composition["W"],
            0.16 * composition["Q"] + 0.14 * composition["N"] + 0.10 * composition["T"],
            0.08 * composition["P"] + 0.08 * composition["Y"],
            0.12 * composition["F"] + 0.10 * composition["V"],
            0.15 * charged_neg + 0.08 * hydrophobic,
        ]
        probabilities = np.array(raw_scores, dtype=np.float64)
        probabilities = probabilities / max(probabilities.sum(), 1e-8)
        return {
            "localization_probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(LOCALIZATION_LABELS)},
            "confidence": float(confidence),
            "membrane_risk": float(membrane_risk),
            "non_membrane_probability": float(non_membrane),
            "backend": "rule-based",
        }

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def health_snapshot(self) -> dict[str, Any]:
        return {**self._metadata, "loaded": True, "fallback": True}


class ProteinLocalizationService:
    """Lazy-loading CPU-only protein localization inference service."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.repo_id = self.settings.localizer_repo_id
        self.confidence_threshold = self.settings.localizer_confidence_threshold
        self.max_seq_len = self.settings.localizer_max_seq_len
        self.enabled = self.settings.localizer_enabled
        self.cache_dir = Path(os.getenv("OMNIBIMOL_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
        self.base_model_id = DEFAULT_BASE_MODEL_ID
        self._lock = threading.Lock()
        self._bundle: LocalizationArtifactBundle | None = None
        self._engine: LocalizationBackend | None = None
        self._load_error: str | None = None

    def _artifact_root(self) -> Path:
        return self.cache_dir / self.repo_id.replace("/", "--")

    def _download_artifacts(self) -> Path | None:
        snapshot_download = _lazy_import_snapshot_download()
        if snapshot_download is None:
            self._load_error = "huggingface_hub is not installed"
            return None

        local_dir = self._artifact_root()
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            repo_path = snapshot_download(
                repo_id=self.repo_id,
                revision="main",
                cache_dir=str(self.cache_dir),
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "esm_lr.pkl",
                    "heuristic_lr.pkl",
                    "binary_esm_lr.pkl",
                    "esm_rf.pkl",
                    "heuristic_rf.pkl",
                    "config.json",
                    "final_results.json",
                    "README.md",
                    "RESEARCH_MEMO.md",
                    "inference.py",
                ],
            )
            return Path(repo_path)
        except Exception as exc:
            self._load_error = str(exc)
            LOGGER.warning("Failed to download localization artifacts from %s: %s", self.repo_id, exc)
            return local_dir if local_dir.exists() else None

    def _load_pickle(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception as exc:
            LOGGER.warning("Could not load localization artifact %s: %s", path.name, exc)
            return None

    def _build_bundle(self) -> LocalizationArtifactBundle | None:
        artifact_root = self._download_artifacts()
        if artifact_root is None:
            return None

        esm_lr = self._load_pickle(artifact_root / "esm_lr.pkl")
        heuristic_lr = self._load_pickle(artifact_root / "heuristic_lr.pkl")
        binary_lr = self._load_pickle(artifact_root / "binary_esm_lr.pkl")
        esm_rf = self._load_pickle(artifact_root / "esm_rf.pkl")
        heuristic_rf = self._load_pickle(artifact_root / "heuristic_rf.pkl")

        EsmModel, EsmTokenizer = _lazy_import_transformers()
        torch = _lazy_import_torch()
        tokenizer = None
        esm_model = None
        if EsmModel is not None and EsmTokenizer is not None and torch is not None:
            try:
                tokenizer = EsmTokenizer.from_pretrained(self.base_model_id)
                esm_model = EsmModel.from_pretrained(self.base_model_id)
                esm_model.eval()
                for parameter in esm_model.parameters():
                    parameter.requires_grad = False
            except Exception as exc:
                LOGGER.warning("Localization base model unavailable: %s", exc)

        metadata = {
            "repo_id": self.repo_id,
            "base_model_id": self.base_model_id,
            "artifact_root": str(artifact_root),
            "loaded": any(model is not None for model in (esm_lr, heuristic_lr, binary_lr, esm_rf, heuristic_rf)),
            "fallback": False,
            "load_error": self._load_error,
        }
        if not metadata["loaded"]:
            return None

        return LocalizationArtifactBundle(
            tokenizer=tokenizer,
            esm_model=esm_model,
            esm_lr=esm_lr,
            heuristic_lr=heuristic_lr,
            binary_lr=binary_lr,
            esm_rf=esm_rf,
            heuristic_rf=heuristic_rf,
            metadata=metadata,
        )

    def _ensure_loaded(self) -> bool:
        if self._engine is not None:
            return True
        if not self.enabled:
            self._load_error = "protein localization service disabled"
            return False

        with self._lock:
            if self._engine is not None:
                return True

            bundle = self._build_bundle()
            if bundle is None:
                self._engine = RuleBasedLocalizationBackend(
                    metadata={
                        "repo_id": self.repo_id,
                        "base_model_id": self.base_model_id,
                        "loaded": False,
                        "fallback": True,
                        "load_error": self._load_error,
                    }
                )
                self._bundle = None
                return True

            self._bundle = bundle
            self._engine = _LocalizationEngine(bundle=bundle)
            self._load_error = None
            LOGGER.info("Loaded localization artifacts from %s", bundle.metadata.get("artifact_root"))
            return True

    def predict(self, sequence: str, *, confidence_threshold: float | None = None) -> dict[str, Any]:
        cleaned = clean_protein_sequence(sequence, max_seq_len=self.max_seq_len)
        if not self._ensure_loaded():
            raise RuntimeError(self._load_error or "protein localization model unavailable")

        assert self._engine is not None
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        raw = self._engine.predict_proba(cleaned)
        probabilities = raw.get("localization_probabilities", {})
        confidence = _safe_float(raw.get("confidence"), 0.0)
        membrane_risk = _safe_float(raw.get("membrane_risk"), 0.0)
        non_membrane_probability = _safe_float(raw.get("non_membrane_probability"), 1.0 - membrane_risk)

        top_label = max(probabilities.items(), key=lambda item: item[1])[0] if probabilities else LOCALIZATION_LABELS[0]
        wetlab_score = compute_wetlab_prioritization_score(
            non_membrane_probability=non_membrane_probability,
            confidence=confidence,
            sequence_length=len(cleaned),
        )
        evidence_passed = confidence >= threshold
        recommended_assay = recommend_assay(
            membrane_risk=membrane_risk,
            confidence=confidence,
            confidence_threshold=threshold,
        )

        result = {
            "localization": top_label,
            "confidence": round(confidence, 4),
            "membrane_risk": round(_clamp(membrane_risk), 4),
            "wetlab_prioritization_score": wetlab_score,
            "recommended_assay": recommended_assay,
            "evidence_passed": evidence_passed,
            "sequence_length": len(cleaned),
            "all_probabilities": {label: round(_safe_float(probabilities.get(label), 0.0), 4) for label in LOCALIZATION_LABELS},
            "non_membrane_probability": round(non_membrane_probability, 4),
            "confidence_threshold": round(threshold, 4),
            "model_metadata": self.health_snapshot(),
        }
        LOGGER.info(
            "protein_localization_predicted repo_id=%s length=%s localization=%s confidence=%.3f membrane_risk=%.3f evidence_passed=%s",
            self.repo_id,
            len(cleaned),
            top_label,
            confidence,
            membrane_risk,
            evidence_passed,
        )
        return result

    def health_snapshot(self) -> dict[str, Any]:
        loaded = self._engine is not None
        metadata = self._bundle.metadata if self._bundle is not None else {
            "repo_id": self.repo_id,
            "base_model_id": self.base_model_id,
            "artifact_root": str(self._artifact_root()),
            "loaded": loaded,
            "fallback": isinstance(self._engine, RuleBasedLocalizationBackend),
            "load_error": self._load_error,
        }
        return {
            **metadata,
            "enabled": self.enabled,
            "loaded": loaded,
            "confidence_threshold": self.confidence_threshold,
            "max_seq_len": self.max_seq_len,
            "load_error": self._load_error,
        }


class _LocalizationEngine:
    def __init__(self, *, bundle: LocalizationArtifactBundle):
        self.bundle = bundle
        self.torch = _lazy_import_torch()
        if self.torch is None:
            raise RuntimeError("torch is not installed")

    def _embed(self, sequence: str) -> np.ndarray:
        if self.bundle.tokenizer is None or self.bundle.esm_model is None:
            raise RuntimeError("ESM backbone is unavailable")
        inputs = self.bundle.tokenizer(sequence, return_tensors="pt", max_length=1024, truncation=True)
        with self.torch.no_grad():
            outputs = self.bundle.esm_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    def _heuristic_features(self, sequence: str) -> np.ndarray:
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        hydrophobic = set("VLIMFWYC")
        charged_pos = set("RKH")
        charged_neg = set("DE")
        polar = set("STNQ")
        aromatic = set("FWY")
        length = len(sequence)
        freq = [sequence.count(aa) / length for aa in amino_acids]
        features = freq + [
            sum(1 for aa in sequence if aa in hydrophobic) / length,
            sum(1 for aa in sequence if aa in charged_pos) / length,
            sum(1 for aa in sequence if aa in charged_neg) / length,
            sum(1 for aa in sequence if aa in polar) / length,
            sum(1 for aa in sequence if aa in aromatic) / length,
            float(length),
            sum(1 for aa in sequence[:30] if aa in hydrophobic) / max(1, min(30, length)),
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)

    @staticmethod
    def _predictor_proba(model: Any, features: np.ndarray) -> np.ndarray | None:
        if model is None:
            return None
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(features))[0]
        if hasattr(model, "predict"):
            predicted = np.asarray(model.predict(features))
            if predicted.ndim == 1 and predicted.size == len(LOCALIZATION_LABELS):
                return predicted.astype(float)
        return None

    def predict_proba(self, sequence: str) -> dict[str, Any]:
        esm_probs = None
        heuristic_probs = None
        binary_probs = None

        if self.bundle.esm_lr is not None or self.bundle.esm_rf is not None:
            embedding = self._embed(sequence)
            esm_probs = self._predictor_proba(self.bundle.esm_lr, embedding)
            if esm_probs is None:
                esm_probs = self._predictor_proba(self.bundle.esm_rf, embedding)
            binary_probs = self._predictor_proba(self.bundle.binary_lr, embedding)

        if self.bundle.heuristic_lr is not None or self.bundle.heuristic_rf is not None:
            heuristic_features = self._heuristic_features(sequence)
            heuristic_probs = self._predictor_proba(self.bundle.heuristic_lr, heuristic_features)
            if heuristic_probs is None:
                heuristic_probs = self._predictor_proba(self.bundle.heuristic_rf, heuristic_features)

        probability_sets = [probs for probs in (esm_probs, heuristic_probs) if probs is not None]
        if probability_sets:
            combined = np.mean(np.vstack(probability_sets), axis=0)
        else:
            combined = np.ones(len(LOCALIZATION_LABELS), dtype=np.float64) / len(LOCALIZATION_LABELS)

        combined = np.clip(combined, 1e-8, None)
        combined = combined / combined.sum()
        membrane_risk = float(binary_probs[1]) if binary_probs is not None and len(binary_probs) > 1 else float(combined[3])
        confidence = float(combined.max())
        return {
            "localization_probabilities": {label: float(combined[idx]) for idx, label in enumerate(LOCALIZATION_LABELS)},
            "confidence": confidence,
            "membrane_risk": membrane_risk,
            "non_membrane_probability": float(1.0 - membrane_risk),
            "backend": "hf_ensemble",
            "components": {
                "esm_available": esm_probs is not None,
                "heuristic_available": heuristic_probs is not None,
                "binary_available": binary_probs is not None,
            },
        }

    def metadata(self) -> dict[str, Any]:
        return dict(self.bundle.metadata)

    def health_snapshot(self) -> dict[str, Any]:
        return {**self.bundle.metadata, "loaded": True, "fallback": False}


_SERVICE: ProteinLocalizationService | None = None


def get_protein_localization_service() -> ProteinLocalizationService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ProteinLocalizationService()
    return _SERVICE