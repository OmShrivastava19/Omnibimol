"""
Production-grade binding affinity model training pipeline.

Implements chemistry-aware splitting, multiple regressor comparison,
comprehensive evaluation, and artifact persistence.

Usage:
    python -m backend.training.train_binding_model \\
        --input data/processed/binding_data_clean.csv \\
        --output models/ \\
        --seed 42 \\
        --split-strategy scaffold
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature order must match runtime predictor exactly
FEATURE_ORDER = [
    'molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings',
    'tpsa', 'num_atoms', 'num_heavy_atoms', 'num_rings', 'fraction_csp3',
    'num_heteroatoms', 'num_aromatic_atoms'
]

# Try to import optional chemistry libraries for better feature generation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Will use fallback feature generation and random splitting.")


class DataValidator:
    """Validate and clean training data."""

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
        """Check that all required columns exist."""
        missing = [col for col in required_cols if col not in df.columns]
        return len(missing) == 0, missing

    @staticmethod
    def validate_label(df: pd.DataFrame, label_col: str = 'pAffinity') -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and filter label values.
        
        Returns:
            - Cleaned DataFrame
            - Statistics dict with before/after counts
        """
        stats = {
            'before': len(df),
            'null_removed': 0,
            'invalid_removed': 0,
            'after': 0
        }

        # Remove nulls
        before = len(df)
        df = df.dropna(subset=[label_col])
        stats['null_removed'] = before - len(df)

        # Remove NaN/inf values
        before = len(df)
        df = df[~(df[label_col].isna() | np.isinf(df[label_col]))]
        stats['invalid_removed'] = before - len(df)

        stats['after'] = len(df)
        return df, stats

    @staticmethod
    def validate_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate feature columns.
        
        Check for constant features, all-zeros, and NaN values.
        Returns cleaned data and statistics.
        """
        stats = {
            'before': len(df),
            'rows_with_null_features': 0,
            'constant_features': [],
            'all_zero_features': [],
            'after': 0
        }

        # Remove rows with any null feature values
        before = len(df)
        df = df.dropna(subset=feature_cols)
        stats['rows_with_null_features'] = before - len(df)

        # Identify problematic features
        for col in feature_cols:
            if df[col].nunique() == 1:
                stats['constant_features'].append(col)
            if (df[col] == 0).all():
                stats['all_zero_features'].append(col)

        stats['after'] = len(df)
        return df, stats


class FeatureGenerator:
    """Generate molecular descriptors, handling fallback if RDKit unavailable."""

    @staticmethod
    def calculate_descriptors_from_smiles(smiles: str) -> Optional[Dict]:
        """
        Calculate molecular descriptors from SMILES string.
        
        Returns dictionary in FEATURE_ORDER or None if SMILES invalid.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return None

        if RDKIT_AVAILABLE:
            return FeatureGenerator._rdkit_descriptors(smiles)
        else:
            return FeatureGenerator._fallback_descriptors(smiles)

    @staticmethod
    def _rdkit_descriptors(smiles: str) -> Optional[Dict]:
        """Calculate descriptors using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None

            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'num_rings': Descriptors.RingCount(mol),
                'fraction_csp3': Descriptors.FractionCSP3(mol),
                'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                'num_aromatic_atoms': rdMolDescriptors.CalcNumAromaticAtoms(mol),
            }
            return descriptors
        except Exception as e:
            logger.debug(f"RDKit failed for SMILES '{smiles}': {e}")
            return None

    @staticmethod
    def _fallback_descriptors(smiles: str) -> Optional[Dict]:
        """
        Fallback descriptor calculation using pattern matching.
        Much less accurate than RDKit but better than zeros.
        """
        try:
            smiles = smiles.strip()
            if not smiles:
                return None

            # Simple atom counting heuristic
            atomic_weights = {
                'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'P': 30.97,
                'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90, 'H': 1.008
            }

            mw = 0.0
            heavy_atoms = 0
            c_count = 0
            n_count = 0
            o_count = 0

            i = 0
            while i < len(smiles):
                if i < len(smiles) - 1 and smiles[i:i+2] in ['Cl', 'Br']:
                    mw += atomic_weights.get(smiles[i:i+2][0], 0) + atomic_weights.get(smiles[i:i+2][1], 0)
                    heavy_atoms += 1
                    i += 2
                    continue

                char = smiles[i]
                if char in atomic_weights:
                    mw += atomic_weights[char]
                    heavy_atoms += 1
                    if char == 'C':
                        c_count += 1
                    elif char == 'N':
                        n_count += 1
                    elif char == 'O':
                        o_count += 1

                i += 1

            # Add implicit hydrogens estimate
            estimated_h = max(0, (c_count * 4 + n_count * 3 + o_count * 2) - (heavy_atoms * 2))
            mw += estimated_h * atomic_weights['H']

            descriptors = {
                'molecular_weight': round(mw, 2),
                'logp': round(2.0 + (mw / 100.0) - ((o_count + n_count) * 0.5), 2),
                'hbd': min(o_count + n_count, 10),
                'hba': min(o_count + n_count, 15),
                'rotatable_bonds': max(0, heavy_atoms - 3),
                'aromatic_rings': max(0, smiles.count('c')),
                'tpsa': round((o_count * 20.23) + (n_count * 12.03), 2),
                'num_atoms': heavy_atoms + estimated_h,
                'num_heavy_atoms': heavy_atoms,
                'num_rings': max(0, smiles.count('1') // 2),
                'fraction_csp3': round(max(0, (c_count - smiles.count('c')) / max(1, c_count)), 2) if c_count > 0 else 0.0,
                'num_heteroatoms': o_count + n_count,
                'num_aromatic_atoms': smiles.count('c'),
            }
            return descriptors
        except Exception as e:
            logger.debug(f"Fallback descriptor generation failed for '{smiles}': {e}")
            return None


class ChemistryAwareSplitter:
    """Implement chemistry-aware train/val/test splitting."""

    @staticmethod
    def get_murcko_scaffold(smiles: str) -> Optional[str]:
        """Extract Murcko scaffold for chemistry-aware splitting."""
        if not RDKIT_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except Exception:
            return None

    @staticmethod
    def scaffold_split(
        df: pd.DataFrame,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        smiles_col: str = 'SMILES'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by Murcko scaffold (or fallback to random).
        
        Returns train, val, test DataFrames.
        """
        np.random.seed(random_state)

        # Try to extract scaffolds
        if RDKIT_AVAILABLE:
            logger.info("Extracting Murcko scaffolds for chemistry-aware splitting...")
            scaffolds = df[smiles_col].apply(ChemistryAwareSplitter.get_murcko_scaffold)
            unique_scaffolds = scaffolds.nunique()
            logger.info(f"Found {unique_scaffolds} unique scaffolds from {len(df)} molecules")

            if unique_scaffolds > 1:
                # Group by scaffold and split at scaffold level
                df_with_scaffold = df.copy()
                df_with_scaffold['_scaffold'] = scaffolds

                scaffold_groups = df_with_scaffold.groupby('_scaffold').size().reset_index(name='count')
                scaffold_groups = scaffold_groups.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

                # Assign scaffolds to splits
                n_test_scaffolds = max(1, int(len(scaffold_groups) * test_size / len(df)))
                n_val_scaffolds = max(1, int(len(scaffold_groups) * val_size / len(df)))

                test_scaffolds = set(scaffold_groups.iloc[:n_test_scaffolds]['_scaffold'])
                val_scaffolds = set(scaffold_groups.iloc[n_test_scaffolds:n_test_scaffolds + n_val_scaffolds]['_scaffold'])

                test_idx = df_with_scaffold[df_with_scaffold['_scaffold'].isin(test_scaffolds)].index
                val_idx = df_with_scaffold[df_with_scaffold['_scaffold'].isin(val_scaffolds)].index
                train_idx = df_with_scaffold[~df_with_scaffold['_scaffold'].isin(test_scaffolds | val_scaffolds)].index

                return (
                    df.loc[train_idx].reset_index(drop=True),
                    df.loc[val_idx].reset_index(drop=True),
                    df.loc[test_idx].reset_index(drop=True)
                )

        # Fallback: random split
        logger.warning("Using random split (chemistry-aware splitting unavailable)")
        remaining_size = 1.0 - test_size
        val_size_in_remaining = val_size / remaining_size

        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train, test_size=val_size_in_remaining, random_state=random_state)

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class ModelTrainer:
    """Train and evaluate multiple regression models."""

    @staticmethod
    def train_regressor(
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = 'random_forest'
    ) -> object:
        """Train a regressor model."""
        logger.info(f"Training {model_name}...")

        if model_name == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=0
            )
        elif model_name == 'linear_regression':
            model = LinearRegression(n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        logger.info(f"✓ {model_name} training complete")
        return model

    @staticmethod
    def evaluate_regressor(
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = 'model'
    ) -> Dict:
        """Evaluate regressor on train/val/test splits."""
        results = {'model_name': model_name}

        for split_name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)

            results[f'{split_name}_mae'] = float(mae)
            results[f'{split_name}_rmse'] = float(rmse)
            results[f'{split_name}_r2'] = float(r2)

            logger.info(f"  {split_name.upper()}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

        # Compute residuals and distribution
        pred_test = model.predict(X_test)
        residuals = y_test - pred_test
        results['residual_mean'] = float(np.mean(residuals))
        results['residual_std'] = float(np.std(residuals))
        results['residual_min'] = float(np.min(residuals))
        results['residual_max'] = float(np.max(residuals))

        # Overfitting indicator
        results['train_val_gap'] = float(results['train_mae'] - results['val_mae'])

        return results

    @staticmethod
    def evaluate_by_affinity_bins(
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        bins: List[float] = None
    ) -> Dict:
        """Evaluate performance by affinity difficulty bins."""
        if bins is None:
            bins = [3, 6, 9, 12]

        pred = model.predict(X_test)
        bin_results = {}

        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            mask = (y_test >= lower) & (y_test < upper)
            if mask.sum() == 0:
                continue

            y_bin = y_test[mask]
            pred_bin = pred[mask]

            mae = mean_absolute_error(y_bin, pred_bin)
            rmse = np.sqrt(mean_squared_error(y_bin, pred_bin))
            r2 = r2_score(y_bin, pred_bin)

            bin_name = f"pAffinity_{lower}-{upper}"
            bin_results[bin_name] = {
                'samples': int(mask.sum()),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }
            logger.info(f"  Bin {lower}-{upper}: n={mask.sum()}, MAE={mae:.3f}, R²={r2:.3f}")

        return bin_results

    @staticmethod
    def train_classifier(
        X_train: np.ndarray,
        y_train: np.ndarray,
        threshold: float = 6.5
    ) -> Tuple[object, float]:
        """
        Train binary classifier (binder vs non-binder).
        
        Returns classifier and optimal threshold.
        """
        logger.info(f"Training classifier with threshold={threshold}...")
        y_binary = (y_train >= threshold).astype(int)

        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        classifier.fit(X_train, y_binary)
        logger.info("✓ Classifier training complete")

        return classifier, threshold

    @staticmethod
    def evaluate_classifier(
        classifier: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 6.5
    ) -> Dict:
        """Evaluate binary classifier."""
        results = {}

        y_binary = (y_test >= threshold).astype(int)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:, 1]

        results['threshold'] = float(threshold)
        results['roc_auc'] = float(roc_auc_score(y_binary, y_prob))
        results['f1_score'] = float(f1_score(y_binary, y_pred))

        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_binary, y_prob)
        results['pr_auc'] = float(auc(recall, precision))

        logger.info(f"  ROC-AUC: {results['roc_auc']:.3f}, PR-AUC: {results['pr_auc']:.3f}, F1: {results['f1_score']:.3f}")

        return results


def regenerate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regenerate molecular descriptors from SMILES.
    
    Checks if features are all zero and recalculates if needed.
    """
    feature_cols = FEATURE_ORDER
    
    # Check if features are already present and non-zero
    if all(col in df.columns for col in feature_cols):
        if not (df[feature_cols] == 0).all().all():
            logger.info("Features already present and non-zero, skipping regeneration")
            return df
    
    logger.warning("Features are zero-valued or missing. Regenerating from SMILES...")
    
    if 'SMILES' not in df.columns:
        raise ValueError("SMILES column not found in data")
    
    # Generate descriptors
    df_copy = df.copy()
    descriptors_list = []
    
    for idx, smiles in enumerate(df_copy['SMILES']):
        if idx % 5000 == 0:
            logger.info(f"  Processed {idx}/{len(df_copy)} molecules...")
        
        desc = FeatureGenerator.calculate_descriptors_from_smiles(smiles)
        if desc is None:
            desc = {col: 0.0 for col in feature_cols}
        
        descriptors_list.append(desc)
    
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # Replace or add columns
    for col in feature_cols:
        df_copy[col] = descriptors_df[col]
    
    logger.info(f"✓ Feature regeneration complete")
    return df_copy


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train binding affinity predictor')
    parser.add_argument('--input', type=str, default='data/processed/binding_data_clean.csv',
                        help='Input data CSV path')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for artifacts')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--split-strategy', type=str, default='scaffold',
                        choices=['scaffold', 'random'],
                        help='Data splitting strategy')
    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("BINDING AFFINITY MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    # 1. Load data
    logger.info(f"\n1. Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Validate columns
    logger.info("\n2. Validating data...")
    required_cols = ['SMILES', 'pAffinity'] + FEATURE_ORDER
    valid, missing = DataValidator.validate_columns(df, required_cols)
    if not valid:
        logger.warning(f"   Missing columns: {missing}. Will regenerate features...")

    # 3. Regenerate features if necessary
    logger.info("\n3. Feature validation and regeneration...")
    df = regenerate_features(df)
    logger.info(f"   Features ready: {FEATURE_ORDER}")

    # 4. Clean labels
    logger.info("\n4. Cleaning labels...")
    df, label_stats = DataValidator.validate_label(df)
    logger.info(f"   Before: {label_stats['before']}, After: {label_stats['after']}")
    logger.info(f"   Removed: {label_stats['null_removed']} nulls, {label_stats['invalid_removed']} invalid")

    # 5. Clean features
    logger.info("\n5. Validating features...")
    df, feature_stats = DataValidator.validate_features(df, FEATURE_ORDER)
    logger.info(f"   Rows after cleanup: {feature_stats['after']}")
    if feature_stats['constant_features']:
        logger.warning(f"   Constant features: {feature_stats['constant_features']}")
    if feature_stats['all_zero_features']:
        logger.warning(f"   All-zero features: {feature_stats['all_zero_features']}")

    if len(df) < 100:
        logger.error(f"   ERROR: Insufficient data after cleaning ({len(df)} rows). Abort.")
        return

    logger.info(f"   ✓ Final dataset: {len(df)} samples")
    logger.info(f"   Label range: {df['pAffinity'].min():.2f} - {df['pAffinity'].max():.2f}")
    logger.info(f"   Label mean: {df['pAffinity'].mean():.2f}, std: {df['pAffinity'].std():.2f}")

    # 6. Split data
    logger.info(f"\n6. Splitting data (strategy: {args.split_strategy})...")
    train_df, val_df, test_df = ChemistryAwareSplitter.scaffold_split(
        df,
        test_size=0.1,
        val_size=0.1,
        random_state=args.seed,
        smiles_col='SMILES'
    )
    logger.info(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 7. Prepare features and labels
    X_train = train_df[FEATURE_ORDER].values
    y_train = train_df['pAffinity'].values
    X_val = val_df[FEATURE_ORDER].values
    y_val = val_df['pAffinity'].values
    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df['pAffinity'].values

    # 8. Feature scaling
    logger.info("\n7. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info("   ✓ Scaler fitted on training data")

    # 9. Train regressors
    logger.info("\n8. Training regressors...")
    models = {}
    results = {}

    for model_name in ['random_forest', 'gradient_boosting', 'linear_regression']:
        logger.info(f"\n   {model_name.upper()}")
        model = ModelTrainer.train_regressor(X_train_scaled, y_train, model_name)
        models[model_name] = model

        eval_result = ModelTrainer.evaluate_regressor(
            model, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            model_name
        )
        results[model_name] = eval_result

    # 10. Select best model
    logger.info("\n9. Selecting best model (by validation R²)...")
    best_model_name = max(results.keys(), key=lambda k: results[k]['val_r2'])
    best_model = models[best_model_name]
    best_results = results[best_model_name]
    logger.info(f"   ✓ Selected: {best_model_name} (val_r2={best_results['val_r2']:.3f})")

    # 11. Performance by affinity bins
    logger.info("\n10. Evaluating best model by affinity bins...")
    bin_results = ModelTrainer.evaluate_by_affinity_bins(best_model, X_test_scaled, y_test)
    best_results['performance_by_bin'] = bin_results

    # 12. Train classifier (optional)
    logger.info("\n11. Training binary classifier...")
    classifier, threshold = ModelTrainer.train_classifier(X_train_scaled, y_train, threshold=6.5)
    classifier_results = ModelTrainer.evaluate_classifier(classifier, X_test_scaled, y_test, threshold)
    best_results['classifier_metrics'] = classifier_results

    # 13. Save artifacts
    logger.info("\n12. Saving artifacts...")
    regressor_path = output_dir / 'binding_predictor.pkl'
    classifier_path = output_dir / 'binding_classifier.pkl'
    scaler_path = output_dir / 'scaler.pkl'

    joblib.dump(best_model, regressor_path)
    joblib.dump(classifier, classifier_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"   ✓ Regressor: {regressor_path}")
    logger.info(f"   ✓ Classifier: {classifier_path}")
    logger.info(f"   ✓ Scaler: {scaler_path}")

    # 14. Generate metadata
    logger.info("\n13. Generating metadata...")
    metadata = {
        'trained_date': datetime.now().isoformat(),
        'model_version': '2.0',
        'data_source': 'ChEMBL (high-confidence binding assays)',
        'n_samples': len(df),
        'split_strategy': args.split_strategy,
        'split_sizes': {
            'train': int(len(train_df)),
            'validation': int(len(val_df)),
            'test': int(len(test_df))
        },
        'feature_order': FEATURE_ORDER,
        'label_definition': 'pAffinity (-log10(M))',
        'label_range': {
            'min': float(df['pAffinity'].min()),
            'max': float(df['pAffinity'].max()),
            'mean': float(df['pAffinity'].mean()),
            'std': float(df['pAffinity'].std())
        },
        'best_model': {
            'name': best_model_name,
            'test_mae': best_results['test_mae'],
            'test_rmse': best_results['test_rmse'],
            'test_r2': best_results['test_r2'],
            'validation_r2': best_results['val_r2']
        },
        'model_comparison': {name: {
            'test_mae': results[name]['test_mae'],
            'test_rmse': results[name]['test_rmse'],
            'test_r2': results[name]['test_r2']
        } for name in results},
        'performance_by_bin': bin_results,
        'classifier_metrics': classifier_results,
        'residual_analysis': {
            'mean': best_results['residual_mean'],
            'std': best_results['residual_std'],
            'min': best_results['residual_min'],
            'max': best_results['residual_max']
        },
        'overfitting_check': {
            'train_mae': best_results['train_mae'],
            'validation_mae': best_results['val_mae'],
            'train_val_gap': best_results['train_val_gap']
        },
        'dependencies': {
            'scikit-learn': '1.0+',
            'numpy': '1.20+',
            'pandas': '1.3+',
            'rdkit': 'available' if RDKIT_AVAILABLE else 'not available'
        }
    }

    metadata_path = output_dir / 'binding_predictor_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   ✓ Metadata: {metadata_path}")

    # 15. Summary report
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test R²: {best_results['test_r2']:.4f}")
    logger.info(f"Test MAE: {best_results['test_mae']:.4f}")
    logger.info(f"Test RMSE: {best_results['test_rmse']:.4f}")
    logger.info(f"Classifier ROC-AUC: {classifier_results['roc_auc']:.4f}")
    logger.info("=" * 80)

    return metadata


if __name__ == '__main__':
    main()
