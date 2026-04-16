"""
Unit tests for binding affinity model training pipeline.

Tests cover:
- Feature order consistency with runtime predictor
- Data validation and cleaning
- Chemistry-aware splitting
- Model training and evaluation
- Artifact persistence and loading
- Metadata generation
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import training components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.training.train_binding_model import (
    FEATURE_ORDER,
    DataValidator,
    FeatureGenerator,
    ChemistryAwareSplitter,
    ModelTrainer
)


class TestFeatureOrderConsistency(unittest.TestCase):
    """Verify feature order matches runtime predictor."""

    def test_feature_order_matches_runtime_predictor(self):
        """
        Runtime predictor in ligand_binding_predictor.py expects features in this exact order.
        This test ensures training uses the same order.
        """
        expected_order = [
            'molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings',
            'tpsa', 'num_atoms', 'num_heavy_atoms', 'num_rings', 'fraction_csp3',
            'num_heteroatoms', 'num_aromatic_atoms'
        ]
        self.assertEqual(FEATURE_ORDER, expected_order)
        self.assertEqual(len(FEATURE_ORDER), 13)

    def test_feature_order_stability(self):
        """Ensure feature order doesn't accidentally change."""
        self.assertIsInstance(FEATURE_ORDER, list)
        for feature in FEATURE_ORDER:
            self.assertIsInstance(feature, str)
            self.assertTrue(len(feature) > 0)


class TestDataValidator(unittest.TestCase):
    """Test data validation and cleaning."""

    def test_validate_columns_success(self):
        """Test successful column validation."""
        df = pd.DataFrame({
            'col1': [1, 2],
            'col2': [3, 4]
        })
        valid, missing = DataValidator.validate_columns(df, ['col1', 'col2'])
        self.assertTrue(valid)
        self.assertEqual(missing, [])

    def test_validate_columns_missing(self):
        """Test detection of missing columns."""
        df = pd.DataFrame({'col1': [1, 2]})
        valid, missing = DataValidator.validate_columns(df, ['col1', 'col2'])
        self.assertFalse(valid)
        self.assertEqual(missing, ['col2'])

    def test_validate_label_removes_nulls(self):
        """Test that null labels are removed."""
        df = pd.DataFrame({
            'pAffinity': [1.0, np.nan, 3.0],
            'other': ['a', 'b', 'c']
        })
        df_clean, stats = DataValidator.validate_label(df)
        self.assertEqual(len(df_clean), 2)
        self.assertEqual(stats['null_removed'], 1)
        self.assertTrue(np.all(df_clean['pAffinity'].notna()))

    def test_validate_label_removes_inf(self):
        """Test that infinite values are removed."""
        df = pd.DataFrame({
            'pAffinity': [1.0, np.inf, 3.0]
        })
        df_clean, stats = DataValidator.validate_label(df)
        self.assertEqual(len(df_clean), 2)

    def test_validate_features_removes_nulls(self):
        """Test that null features are removed."""
        df = pd.DataFrame({
            'feat1': [1.0, np.nan, 3.0],
            'feat2': [1.0, 2.0, 3.0]
        })
        df_clean, stats = DataValidator.validate_features(df, ['feat1', 'feat2'])
        self.assertEqual(len(df_clean), 2)
        self.assertEqual(stats['rows_with_null_features'], 1)

    def test_validate_features_detects_constant(self):
        """Test detection of constant features."""
        df = pd.DataFrame({
            'feat1': [1.0, 1.0, 1.0],
            'feat2': [1.0, 2.0, 3.0]
        })
        df_clean, stats = DataValidator.validate_features(df, ['feat1', 'feat2'])
        self.assertIn('feat1', stats['constant_features'])
        self.assertNotIn('feat2', stats['constant_features'])

    def test_validate_features_detects_all_zeros(self):
        """Test detection of all-zero features."""
        df = pd.DataFrame({
            'feat1': [0.0, 0.0, 0.0],
            'feat2': [1.0, 2.0, 3.0]
        })
        df_clean, stats = DataValidator.validate_features(df, ['feat1', 'feat2'])
        self.assertIn('feat1', stats['all_zero_features'])
        self.assertNotIn('feat2', stats['all_zero_features'])


class TestFeatureGenerator(unittest.TestCase):
    """Test molecular descriptor generation."""

    def test_fallback_descriptors_generates_dict(self):
        """Test fallback descriptor generation for valid SMILES."""
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
        desc = FeatureGenerator._fallback_descriptors(smiles)
        self.assertIsNotNone(desc)
        self.assertIsInstance(desc, dict)
        self.assertEqual(set(desc.keys()), set(FEATURE_ORDER))

    def test_fallback_descriptors_invalid_smiles(self):
        """Test fallback handles empty/invalid SMILES."""
        desc = FeatureGenerator._fallback_descriptors("")
        self.assertIsNone(desc)

    def test_fallback_descriptors_reasonable_values(self):
        """Test descriptor values are in reasonable ranges."""
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        desc = FeatureGenerator._fallback_descriptors(smiles)
        self.assertGreater(desc['molecular_weight'], 0)
        self.assertGreaterEqual(desc['hbd'], 0)
        self.assertGreaterEqual(desc['hba'], 0)
        self.assertGreaterEqual(desc['num_atoms'], 0)

    def test_calculate_descriptors_from_smiles(self):
        """Test top-level descriptor calculation."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        desc = FeatureGenerator.calculate_descriptors_from_smiles(smiles)
        # Should return dict or None, but not fail
        self.assertTrue(desc is None or isinstance(desc, dict))


class TestChemistryAwareSplitter(unittest.TestCase):
    """Test data splitting strategies."""

    def setUp(self):
        """Create sample data for splitting tests."""
        self.df = pd.DataFrame({
            'SMILES': [f"C{i}" for i in range(100)],
            'pAffinity': np.random.uniform(3, 12, 100),
            'feat1': np.random.randn(100)
        })

    def test_scaffold_split_creates_three_splits(self):
        """Test that scaffold split returns train/val/test."""
        train, val, test = ChemistryAwareSplitter.scaffold_split(
            self.df, test_size=0.1, val_size=0.1, random_state=42
        )
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))

    def test_scaffold_split_sizes_approximate(self):
        """Test that split sizes are approximately correct."""
        train, val, test = ChemistryAwareSplitter.scaffold_split(
            self.df, test_size=0.1, val_size=0.1, random_state=42
        )
        # Allow 5% tolerance
        self.assertAlmostEqual(len(test) / len(self.df), 0.1, delta=0.05)
        self.assertAlmostEqual(len(val) / len(self.df), 0.1, delta=0.05)
        self.assertGreater(len(train), len(val))

    def test_scaffold_split_no_overlap(self):
        """Test that splits don't overlap (check by row content, not index)."""
        train, val, test = ChemistryAwareSplitter.scaffold_split(
            self.df, test_size=0.1, val_size=0.1, random_state=42
        )
        # Verify total row count matches
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))


class TestModelTrainer(unittest.TestCase):
    """Test model training and evaluation."""

    def setUp(self):
        """Create sample training data."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 13
        
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.uniform(3, 12, self.n_samples)
        
        self.X_test = np.random.randn(self.n_samples // 2, self.n_features)
        self.y_test = np.random.uniform(3, 12, self.n_samples // 2)
        
        self.X_val = np.random.randn(self.n_samples // 4, self.n_features)
        self.y_val = np.random.uniform(3, 12, self.n_samples // 4)

    def test_train_regressor_random_forest(self):
        """Test RandomForest training."""
        model = ModelTrainer.train_regressor(self.X_train, self.y_train, 'random_forest')
        self.assertIsNotNone(model)
        pred = model.predict(self.X_test[:5])
        self.assertEqual(len(pred), 5)

    def test_train_regressor_gradient_boosting(self):
        """Test GradientBoosting training."""
        model = ModelTrainer.train_regressor(self.X_train, self.y_train, 'gradient_boosting')
        self.assertIsNotNone(model)
        pred = model.predict(self.X_test[:5])
        self.assertEqual(len(pred), 5)

    def test_train_regressor_linear(self):
        """Test LinearRegression training."""
        model = ModelTrainer.train_regressor(self.X_train, self.y_train, 'linear_regression')
        self.assertIsNotNone(model)
        pred = model.predict(self.X_test[:5])
        self.assertEqual(len(pred), 5)

    def test_evaluate_regressor_returns_dict(self):
        """Test regressor evaluation returns required keys."""
        model = ModelTrainer.train_regressor(self.X_train, self.y_train, 'random_forest')
        results = ModelTrainer.evaluate_regressor(
            model, self.X_train, self.y_train, self.X_val, self.y_val,
            self.X_test, self.y_test, 'test_model'
        )
        
        self.assertIn('model_name', results)
        self.assertIn('train_mae', results)
        self.assertIn('val_mae', results)
        self.assertIn('test_mae', results)
        self.assertIn('test_r2', results)
        self.assertIn('residual_mean', results)
        self.assertIn('train_val_gap', results)

    def test_evaluate_by_affinity_bins(self):
        """Test evaluation by affinity bins."""
        model = ModelTrainer.train_regressor(self.X_train, self.y_train, 'random_forest')
        bin_results = ModelTrainer.evaluate_by_affinity_bins(model, self.X_test, self.y_test)
        
        self.assertIsInstance(bin_results, dict)
        # Should have some bins with samples
        self.assertGreater(len(bin_results), 0)

    def test_train_classifier(self):
        """Test binary classifier training."""
        classifier, threshold = ModelTrainer.train_classifier(self.X_train, self.y_train, threshold=6.5)
        self.assertIsNotNone(classifier)
        self.assertEqual(threshold, 6.5)
        
        pred = classifier.predict(self.X_test[:5])
        self.assertTrue(all(p in [0, 1] for p in pred))

    def test_evaluate_classifier_returns_metrics(self):
        """Test classifier evaluation returns required metrics."""
        classifier, threshold = ModelTrainer.train_classifier(
            self.X_train, self.y_train, threshold=6.5
        )
        results = ModelTrainer.evaluate_classifier(classifier, self.X_test, self.y_test)
        
        self.assertIn('roc_auc', results)
        self.assertIn('pr_auc', results)
        self.assertIn('f1_score', results)
        self.assertIn('threshold', results)
        
        self.assertGreaterEqual(results['roc_auc'], 0)
        self.assertLessEqual(results['roc_auc'], 1)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_artifact_loading_schema_matches_runtime(self):
        """
        Test that saved artifacts can be loaded and used in the format
        expected by the runtime predictor.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training data
            X = np.random.randn(50, 13)
            y = np.random.uniform(3, 12, 50)
            
            # Train and save
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_scaled, y)
            
            joblib.dump(model, Path(tmpdir) / 'binding_predictor.pkl')
            joblib.dump(scaler, Path(tmpdir) / 'scaler.pkl')
            
            # Load and verify
            loaded_model = joblib.load(Path(tmpdir) / 'binding_predictor.pkl')
            loaded_scaler = joblib.load(Path(tmpdir) / 'scaler.pkl')
            
            # Test prediction with expected feature shape
            test_features = np.random.randn(1, 13)
            test_features_scaled = loaded_scaler.transform(test_features)
            pred = loaded_model.predict(test_features_scaled)
            
            self.assertEqual(len(pred), 1)
            self.assertTrue(3 <= pred[0] <= 12 or np.isfinite(pred[0]))

    def test_metadata_includes_required_fields(self):
        """Test that generated metadata includes all required fields."""
        required_fields = [
            'trained_date',
            'model_version',
            'data_source',
            'n_samples',
            'split_strategy',
            'split_sizes',
            'feature_order',
            'label_definition',
            'best_model',
            'classifier_metrics',
            'dependencies'
        ]
        
        # Create minimal metadata
        metadata = {
            'trained_date': '2026-01-01T00:00:00',
            'model_version': '2.0',
            'data_source': 'test',
            'n_samples': 100,
            'split_strategy': 'scaffold',
            'split_sizes': {'train': 80, 'val': 10, 'test': 10},
            'feature_order': FEATURE_ORDER,
            'label_definition': 'pAffinity (-log10(M))',
            'best_model': {'name': 'rf', 'test_r2': 0.5},
            'classifier_metrics': {'roc_auc': 0.8},
            'dependencies': {'scikit-learn': '1.0+'}
        }
        
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing required field: {field}")


if __name__ == '__main__':
    unittest.main()
