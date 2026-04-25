"""
Ligand Binding Prediction Module
Predicts binding affinity and binding likelihood for drug molecules (SMILES format)
"""

import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional, Tuple
import re
import warnings
import logging
import os
from pathlib import Path
import joblib
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
HF_MODEL_ID = "omshrivastava/omnibimol-binding-affinity-chemberta"

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None
    rdMolDescriptors = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


class SMILESValidator:
    """Validates and preprocesses SMILES strings"""
    
    @staticmethod
    def is_valid_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SMILES string
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not isinstance(smiles, str):
            return False, "SMILES must be a non-empty string"
        
        smiles = smiles.strip()
        if len(smiles) == 0:
            return False, "SMILES string is empty"
        
        if not RDKIT_AVAILABLE:
            # Basic validation without RDKit
            if len(smiles) < 1:
                return False, "SMILES too short"
            # Check for basic SMILES characters
            valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#+-.@/\\')
            if not all(c in valid_chars or c.isspace() for c in smiles):
                return False, "SMILES contains invalid characters"
            return True, None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES: Cannot parse molecule"
            
            # Check if molecule has atoms
            if mol.GetNumAtoms() == 0:
                return False, "Invalid SMILES: Molecule has no atoms"
            
            return True, None
            
        except Exception as e:
            return False, f"SMILES validation error: {str(e)}"
    
    @staticmethod
    def canonicalize_smiles(smiles: str) -> Optional[str]:
        """
        Canonicalize SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            Canonicalized SMILES or None if invalid
        """
        if not RDKIT_AVAILABLE:
            return smiles.strip()
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None
    
    @staticmethod
    def preprocess_smiles(smiles: str) -> Dict:
        """
        Preprocess SMILES and extract basic information
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with preprocessed data
        """
        result = {
            "original_smiles": smiles.strip(),
            "canonical_smiles": None,
            "is_valid": False,
            "error": None,
            "atom_count": 0,
            "bond_count": 0
        }
        
        is_valid, error = SMILESValidator.is_valid_smiles(smiles)
        result["is_valid"] = is_valid
        result["error"] = error
        
        if not is_valid:
            return result
        
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    result["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
                    result["atom_count"] = mol.GetNumAtoms()
                    result["bond_count"] = mol.GetNumBonds()
            except Exception:
                pass
        
        return result


class MolecularDescriptorCalculator:
    """Calculates molecular descriptors for ML models"""
    
    @staticmethod
    def calculate_descriptors(smiles: str) -> Dict:
        """
        Calculate molecular descriptors from SMILES
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of molecular descriptors
        """
        descriptors = {
            "molecular_weight": 0.0,
            "logp": 0.0,
            "hbd": 0,  # Hydrogen bond donors
            "hba": 0,  # Hydrogen bond acceptors
            "rotatable_bonds": 0,
            "aromatic_rings": 0,
            "tpsa": 0.0,  # Topological polar surface area
            "num_atoms": 0,
            "num_heavy_atoms": 0,
            "num_rings": 0,
            "fraction_csp3": 0.0,
            "num_heteroatoms": 0,
            "num_aromatic_atoms": 0
        }
        
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Try fallback calculation
                    return MolecularDescriptorCalculator._calculate_basic_descriptors(smiles)
                
                descriptors["molecular_weight"] = Descriptors.MolWt(mol)
                descriptors["logp"] = Descriptors.MolLogP(mol)
                descriptors["hbd"] = Descriptors.NumHDonors(mol)
                descriptors["hba"] = Descriptors.NumHAcceptors(mol)
                descriptors["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
                descriptors["aromatic_rings"] = Descriptors.NumAromaticRings(mol)
                descriptors["tpsa"] = Descriptors.TPSA(mol)
                descriptors["num_atoms"] = mol.GetNumAtoms()
                descriptors["num_heavy_atoms"] = Descriptors.HeavyAtomCount(mol)
                descriptors["num_rings"] = Descriptors.RingCount(mol)
                descriptors["fraction_csp3"] = Descriptors.FractionCSP3(mol)
                descriptors["num_heteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
                descriptors["num_aromatic_atoms"] = rdMolDescriptors.CalcNumAromaticAtoms(mol)
                
            except Exception as e:
                print(f"Error calculating descriptors with RDKit: {e}")
                # Fallback to basic calculation
                return MolecularDescriptorCalculator._calculate_basic_descriptors(smiles)
        else:
            # Use fallback calculation when RDKit is not available
            return MolecularDescriptorCalculator._calculate_basic_descriptors(smiles)
        
        return descriptors
    
    @staticmethod
    def _calculate_basic_descriptors(smiles: str) -> Dict:
        """
        Calculate basic molecular descriptors without RDKit
        Uses simple heuristics and atom counting
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of basic molecular descriptors
        """
        descriptors = {
            "molecular_weight": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "rotatable_bonds": 0,
            "aromatic_rings": 0,
            "tpsa": 0.0,
            "num_atoms": 0,
            "num_heavy_atoms": 0,
            "num_rings": 0,
            "fraction_csp3": 0.0,
            "num_heteroatoms": 0,
            "num_aromatic_atoms": 0
        }
        
        try:
            # Atomic weights (approximate)
            atomic_weights = {
                'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'P': 30.97,
                'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90,
                'H': 1.008, 'B': 10.81, 'Si': 28.09
            }
            
            # Count atoms (simple heuristic)
            mw = 0.0
            heavy_atoms = 0
            h_count = 0
            o_count = 0
            n_count = 0
            c_count = 0
            aromatic_count = 0
            
            # Simple parsing - count elements
            i = 0
            while i < len(smiles):
                char = smiles[i]
                
                # Check for two-letter elements
                if i < len(smiles) - 1:
                    two_char = char + smiles[i+1]
                    if two_char in ['Cl', 'Br']:
                        mw += atomic_weights.get(two_char[0], 0) + atomic_weights.get(two_char[1], 0)
                        heavy_atoms += 1
                        i += 2
                        continue
                
                # Single character elements
                if char == 'C':
                    mw += atomic_weights['C']
                    c_count += 1
                    heavy_atoms += 1
                elif char == 'N':
                    mw += atomic_weights['N']
                    n_count += 1
                    heavy_atoms += 1
                elif char == 'O':
                    mw += atomic_weights['O']
                    o_count += 1
                    heavy_atoms += 1
                elif char == 'S':
                    mw += atomic_weights['S']
                    heavy_atoms += 1
                elif char == 'P':
                    mw += atomic_weights['P']
                    heavy_atoms += 1
                elif char == 'F':
                    mw += atomic_weights['F']
                    heavy_atoms += 1
                elif char == 'H':
                    mw += atomic_weights['H']
                    h_count += 1
                elif char in ['[', ']', '(', ')', '=', '#', '-', '+', '.', '@', '/', '\\']:
                    # Skip SMILES syntax characters
                    pass
                elif char.isdigit():
                    # Skip numbers (they indicate ring closures or atom counts)
                    pass
                elif char.lower() == char and char.isalpha():
                    # Lowercase might indicate aromatic (simplified)
                    aromatic_count += 1
                    mw += atomic_weights.get(char.upper(), 12.01)  # Default to C
                    heavy_atoms += 1
                
                i += 1
            
            # Estimate hydrogen count (add implicit hydrogens)
            # Very rough estimate: assume valency satisfaction
            estimated_h = max(0, (c_count * 4 + n_count * 3 + o_count * 2) - (heavy_atoms * 2))
            mw += estimated_h * atomic_weights['H']
            
            descriptors["molecular_weight"] = round(mw, 2)
            descriptors["num_atoms"] = heavy_atoms + h_count + estimated_h
            descriptors["num_heavy_atoms"] = heavy_atoms
            
            # Estimate LogP (very rough: based on O, N count and MW)
            # More O/N = more polar = lower LogP
            # Higher MW = higher LogP (generally)
            polar_atoms = o_count + n_count
            if heavy_atoms > 0:
                descriptors["logp"] = round(2.0 + (mw / 100.0) - (polar_atoms * 0.5), 2)
            else:
                descriptors["logp"] = 0.0
            
            # Estimate H-bond donors (O-H, N-H groups)
            descriptors["hbd"] = min(o_count + n_count, 10)  # Rough estimate
            
            # Estimate H-bond acceptors (O, N atoms)
            descriptors["hba"] = min(o_count + n_count, 15)
            
            # Estimate aromatic rings (count lowercase letters / 6)
            descriptors["aromatic_rings"] = max(0, aromatic_count // 6)
            descriptors["num_aromatic_atoms"] = aromatic_count
            
            # Estimate rings (count parentheses and numbers)
            ring_indicators = smiles.count('1') + smiles.count('2') + smiles.count('3') + \
                            smiles.count('4') + smiles.count('5') + smiles.count('6') + \
                            smiles.count('7') + smiles.count('8') + smiles.count('9')
            descriptors["num_rings"] = max(0, ring_indicators // 2)  # Rough estimate
            
            # Estimate rotatable bonds (rough: based on chain length)
            if heavy_atoms > 0:
                descriptors["rotatable_bonds"] = max(0, heavy_atoms - descriptors["num_rings"] * 6 - 3)
            
            # Estimate TPSA (rough: based on polar atoms)
            descriptors["tpsa"] = round((o_count * 20.23) + (n_count * 12.03), 2)
            
            # Estimate heteroatoms
            descriptors["num_heteroatoms"] = o_count + n_count + (heavy_atoms - c_count - o_count - n_count)
            
            # Estimate fraction Csp3 (very rough)
            if heavy_atoms > 0:
                descriptors["fraction_csp3"] = round(max(0, (c_count - aromatic_count) / max(1, c_count)), 2)
            
        except Exception as e:
            print(f"Error in basic descriptor calculation: {e}")
        
        return descriptors
    
    @staticmethod
    def calculate_lipinski_violations(descriptors: Dict) -> int:
        """
        Calculate Lipinski's Rule of Five violations
        
        Args:
            descriptors: Dictionary of molecular descriptors
            
        Returns:
            Number of violations (0-4)
        """
        violations = 0
        
        # Rule 1: MW <= 500
        if descriptors["molecular_weight"] > 500:
            violations += 1
        
        # Rule 2: LogP <= 5
        if descriptors["logp"] > 5:
            violations += 1
        
        # Rule 3: HBD <= 5
        if descriptors["hbd"] > 5:
            violations += 1
        
        # Rule 4: HBA <= 10
        if descriptors["hba"] > 10:
            violations += 1
        
        return violations


class HFChemBERTaBindingPredictor:
    """Loads and runs HF ChemBERTa pKd inference for SMILES batches."""

    def __init__(self, model_id: str = HF_MODEL_ID, batch_size: int = 32):
        self.model_id = model_id
        self.batch_size = max(1, batch_size)
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.local_files_only = os.getenv("OMNIBIMOL_HF_LOCAL_ONLY", "0") == "1"

    def _ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True

        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            self.load_error = "transformers/torch not available"
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.load_error = None
            logger.info("Loaded HF binding model %s on %s", self.model_id, self.device)
            return True
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning(
                "Unable to load HF binding model %s. Falling back to local predictor. Error: %s",
                self.model_id,
                exc,
            )
            return False

    def predict_batch(self, smiles_batch: List[str]) -> Optional[List[float]]:
        if not smiles_batch:
            return []
        if not self._ensure_loaded():
            return None

        try:
            predictions: List[float] = []
            for i in range(0, len(smiles_batch), self.batch_size):
                batch = smiles_batch[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.dim() == 1:
                    batch_preds = logits.detach().cpu().numpy().tolist()
                else:
                    batch_preds = logits[:, 0].detach().cpu().numpy().tolist()
                predictions.extend(float(x) for x in batch_preds)
            return predictions
        except Exception as exc:
            logger.warning(
                "HF batch inference failed. Falling back to local predictor. Error: %s",
                exc,
            )
            return None

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source": "huggingface",
            "device": self.device,
            "loaded": self.is_loaded,
            "load_error": self.load_error,
        }

class BindingAffinityPredictor:
    """ML-based binding affinity predictor — now loads REAL trained model"""
    
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        self.model_dir = Path(__file__).resolve().parent / "models"
        hf_batch_size = int(os.getenv("OMNIBIMOL_HF_BATCH_SIZE", "32"))
        self.hf_predictor = HFChemBERTaBindingPredictor(batch_size=hf_batch_size)
        self._load_trained_model()

    def _artifact_path(self, filename: str) -> Path:
        """Resolve model artifact paths relative to this module."""
        return self.model_dir / filename

    def _initialize_model(self):
        """Safe fallback initializer when persisted artifacts are unavailable."""
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.is_trained = False

    def _rule_based_prediction(self, descriptors: Dict) -> Dict:
        """Rule-based fallback prediction when ML artifacts are unavailable."""
        mw = descriptors.get("molecular_weight", 0)
        logp = descriptors.get("logp", 0)
        hbd = descriptors.get("hbd", 0)
        hba = descriptors.get("hba", 0)
        rot_bonds = descriptors.get("rotatable_bonds", 0)
        aromatic_rings = descriptors.get("aromatic_rings", 0)

        score = 0
        if 200 <= mw <= 500:
            score += 30
        elif 100 <= mw <= 600:
            score += 20
        else:
            score += 10

        if 0 <= logp <= 5:
            score += 25
        elif -2 <= logp <= 7:
            score += 15
        else:
            score += 5

        score += 20 if hbd <= 5 else 5
        score += 20 if hba <= 10 else 5
        if rot_bonds <= 10:
            score += 5
        if aromatic_rings > 0:
            score += 5

        # Map heuristic score to pAffinity-like scale where higher implies stronger binding.
        affinity = 4.0 + (score / 100) * 6.0
        probability = min(max(score / 100.0, 0.0), 1.0)

        return {
            "binding_affinity": float(affinity),
            "binding_affinity_units": "pAffinity (-log10(M))",
            "binding_likelihood": float(probability * 100),
            "binding_probability": float(probability),
            "prediction_method": "Rule-based fallback",
            "confidence": "low",
            "model_metadata": {
                "source": "local_fallback",
                "reason": "heuristic_rule_based",
            },
        }

    @staticmethod
    def _probability_from_affinity(affinity: float) -> float:
        # Logistic calibration around pAffinity 6.0 keeps probabilities smooth and bounded.
        return float(1.0 / (1.0 + np.exp(-(affinity - 6.0))))

    def _hf_prediction_from_affinity(self, affinity: float) -> Dict:
        probability = self._probability_from_affinity(affinity)
        return {
            "binding_affinity": float(affinity),
            "binding_affinity_units": "pAffinity (-log10(M))",
            "binding_likelihood": float(probability * 100.0),
            "binding_probability": float(probability),
            "prediction_method": "HF ChemBERTa (SMILES->pKd)",
            "confidence": "high",
            "model_metadata": self.hf_predictor.metadata(),
        }
    
    def _load_trained_model(self):
        """Load persisted real model instead of synthetic training"""
        regressor_path = self._artifact_path("binding_predictor.pkl")
        classifier_path = self._artifact_path("binding_classifier.pkl")
        scaler_path = self._artifact_path("scaler.pkl")

        missing = [
            str(path) for path in (regressor_path, classifier_path, scaler_path)
            if not path.exists()
        ]
        if missing:
            logger.warning(
                "Binding model artifact(s) missing. Falling back to rule-based prediction. Missing: %s",
                ", ".join(missing),
            )
            self._initialize_model()
            return

        try:
            self.regressor = joblib.load(regressor_path)
            self.classifier = joblib.load(classifier_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            logger.info("Loaded real ChEMBL-trained binding model artifacts from %s", self.model_dir)
        except Exception as e:
            logger.warning(
                "Failed to load trained model artifacts. Falling back to rule-based prediction. Error: %s",
                e,
            )
            self._initialize_model()
    
    def predict(self, descriptors: Dict) -> Dict:
        """Same public API as before — but now uses real model"""
        if not self.is_trained or self.regressor is None:
            return self._rule_based_prediction(descriptors)
        
        try:
            features = np.array([[
                descriptors.get("molecular_weight", 0),
                descriptors.get("logp", 0),
                descriptors.get("hbd", 0),
                descriptors.get("hba", 0),
                descriptors.get("rotatable_bonds", 0),
                descriptors.get("aromatic_rings", 0),
                descriptors.get("tpsa", 0),
                descriptors.get("num_atoms", 0),
                descriptors.get("num_heavy_atoms", 0),
                descriptors.get("num_rings", 0),
                descriptors.get("fraction_csp3", 0),
                descriptors.get("num_heteroatoms", 0),
                descriptors.get("num_aromatic_atoms", 0)
            ]])
            
            features_scaled = self.scaler.transform(features)
            
            affinity = self.regressor.predict(features_scaled)[0]
            binding_prob = self.classifier.predict_proba(features_scaled)[0][1]
            
            return {
                "binding_affinity": float(affinity),           # now in pAffinity units (higher = stronger)
                "binding_affinity_units": "pAffinity (-log10(M))",
                "binding_likelihood": float(binding_prob * 100),
                "binding_probability": float(binding_prob),
                "prediction_method": "ML (Random Forest — real ChEMBL data)",
                "confidence": "high",
                "model_metadata": {
                    "source": "local_ml",
                    "model_family": "random_forest",
                },
            }
        except Exception as e:
            logger.warning("ML prediction error. Falling back to rule-based prediction. Error: %s", e)
            return self._rule_based_prediction(descriptors)

    def predict_smiles_batch(
        self,
        smiles_batch: List[str],
        descriptors_batch: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Primary fast path: HF SMILES->pKd. Fallback: local descriptor predictor."""
        descriptors_batch = descriptors_batch or []

        hf_affinities = self.hf_predictor.predict_batch(smiles_batch)
        if hf_affinities is not None and len(hf_affinities) == len(smiles_batch):
            return [self._hf_prediction_from_affinity(value) for value in hf_affinities]

        fallback_predictions: List[Dict] = []
        for idx, _smiles in enumerate(smiles_batch):
            descriptors = descriptors_batch[idx] if idx < len(descriptors_batch) else {}
            fallback_predictions.append(self.predict(descriptors))
        return fallback_predictions

    def predict_smiles(self, smiles: str, descriptors: Optional[Dict] = None) -> Dict:
        batch_predictions = self.predict_smiles_batch([smiles], [descriptors or {}])
        return batch_predictions[0] if batch_predictions else self._rule_based_prediction(descriptors or {})


class LigandBindingPredictor:
    """Main class for ligand binding prediction"""
    
    def __init__(self):
        self.validator = SMILESValidator()
        self.descriptor_calc = MolecularDescriptorCalculator()
        self.affinity_predictor = BindingAffinityPredictor()
    
    def predict_single(self, smiles: str, molecule_name: str = None) -> Dict:
        """
        Predict binding for a single molecule
        
        Args:
            smiles: SMILES string
            molecule_name: Optional name for the molecule
            
        Returns:
            Dictionary with prediction results
        """
        return self.predict_batch([smiles], [molecule_name])[0]
    
    def predict_batch(self, smiles_list: List[str], molecule_names: List[str] = None) -> List[Dict]:
        """
        Predict binding for multiple molecules
        
        Args:
            smiles_list: List of SMILES strings
            molecule_names: Optional list of molecule names
            
        Returns:
            List of prediction result dictionaries
        """
        if molecule_names is None:
            molecule_names = [None] * len(smiles_list)

        if len(molecule_names) < len(smiles_list):
            molecule_names = molecule_names + [None] * (len(smiles_list) - len(molecule_names))

        results: List[Dict] = []
        valid_smiles: List[str] = []
        valid_descriptors: List[Dict] = []
        valid_indices: List[int] = []

        for idx, (smiles, name) in enumerate(zip(smiles_list, molecule_names)):
            result = {
                "molecule_name": name or "Unknown",
                "smiles": smiles,
                "is_valid": False,
                "error": None,
                "descriptors": {},
                "prediction": {},
                "lipinski_violations": 0,
            }

            is_valid, error = self.validator.is_valid_smiles(smiles)
            result["is_valid"] = is_valid
            result["error"] = error

            if not is_valid:
                results.append(result)
                continue

            preprocessed = self.validator.preprocess_smiles(smiles)
            canonical_smiles = preprocessed.get("canonical_smiles") or smiles.strip()
            result["canonical_smiles"] = canonical_smiles

            try:
                descriptors = self.descriptor_calc.calculate_descriptors(canonical_smiles)
                if descriptors.get("molecular_weight", 0) == 0 and descriptors.get("num_atoms", 0) == 0:
                    descriptors = self.descriptor_calc._calculate_basic_descriptors(canonical_smiles)
            except Exception as exc:
                logger.warning(
                    "Descriptor calculation failed for SMILES '%s'. Falling back to basic descriptors. Error: %s",
                    canonical_smiles,
                    exc,
                )
                descriptors = self.descriptor_calc._calculate_basic_descriptors(canonical_smiles)

            result["descriptors"] = descriptors
            result["lipinski_violations"] = self.descriptor_calc.calculate_lipinski_violations(descriptors)

            results.append(result)
            valid_indices.append(idx)
            valid_smiles.append(canonical_smiles)
            valid_descriptors.append(descriptors)

        if valid_smiles:
            predictions = self.affinity_predictor.predict_smiles_batch(valid_smiles, valid_descriptors)
            for idx, prediction in zip(valid_indices, predictions):
                results[idx]["prediction"] = prediction

        return results
    
    def rank_molecules(self, predictions: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Rank molecules by binding likelihood and affinity
        
        Args:
            predictions: List of prediction result dictionaries
            top_n: Number of top candidates to return
            
        Returns:
            Ranked list of predictions
        """
        # Filter valid predictions
        valid_predictions = [p for p in predictions if p.get("is_valid", False)]
        
        if not valid_predictions:
            return []

        def affinity_sort_key(pred: Dict) -> float:
            prediction = pred.get("prediction", {})
            affinity = prediction.get("binding_affinity", 0)
            units = str(prediction.get("binding_affinity_units", "")).lower()
            # For pAffinity, higher values imply stronger binding; for legacy kcal/mol, lower is better.
            return -affinity if "paffinity" in units or "-log10" in units else affinity
        
        # Sort by binding likelihood (descending), then by affinity semantic tie-break.
        ranked = sorted(
            valid_predictions,
            key=lambda x: (
                -x.get("prediction", {}).get("binding_likelihood", 0),
                affinity_sort_key(x)
            )
        )
        
        # Add rank
        for i, pred in enumerate(ranked):
            pred["rank"] = i + 1
        
        return ranked[:top_n]
    
    def recommend_top_candidates(self, predictions: List[Dict], n: int = 5) -> Dict:
        """
        Recommend top N candidates with summary statistics
        
        Args:
            predictions: List of prediction result dictionaries
            n: Number of top candidates to recommend
            
        Returns:
            Dictionary with recommendations and statistics
        """
        ranked = self.rank_molecules(predictions, top_n=n)
        
        if not ranked:
            return {
                "top_candidates": [],
                "total_molecules": len(predictions),
                "valid_molecules": 0,
                "average_affinity": None,
                "average_likelihood": None
            }
        
        valid_predictions = [p for p in predictions if p.get("is_valid", False)]
        
        avg_affinity = np.mean([
            p.get("prediction", {}).get("binding_affinity", 0)
            for p in valid_predictions
        ]) if valid_predictions else None
        
        avg_likelihood = np.mean([
            p.get("prediction", {}).get("binding_likelihood", 0)
            for p in valid_predictions
        ]) if valid_predictions else None
        
        return {
            "top_candidates": ranked,
            "total_molecules": len(predictions),
            "valid_molecules": len(valid_predictions),
            "average_affinity": float(avg_affinity) if avg_affinity is not None else None,
            "average_likelihood": float(avg_likelihood) if avg_likelihood is not None else None,
            "recommendation_count": len(ranked)
        }
