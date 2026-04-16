import unittest
from unittest.mock import patch

from ligand_binding_predictor import BindingAffinityPredictor, LigandBindingPredictor


class TestBindingAffinityPredictor(unittest.TestCase):
    @patch("ligand_binding_predictor.joblib.load")
    @patch("ligand_binding_predictor.Path.exists")
    def test_model_files_present_load_successfully(self, mock_exists, mock_load):
        mock_exists.return_value = True
        mock_load.side_effect = [object(), object(), object()]

        predictor = BindingAffinityPredictor()

        self.assertTrue(predictor.is_trained)
        self.assertIsNotNone(predictor.regressor)
        self.assertIsNotNone(predictor.classifier)
        self.assertIsNotNone(predictor.scaler)
        self.assertEqual(mock_load.call_count, 3)

    @patch("ligand_binding_predictor.Path.exists")
    def test_missing_model_file_triggers_safe_fallback(self, mock_exists):
        # Called in order: regressor, classifier, scaler.
        mock_exists.side_effect = [True, False, True]

        predictor = BindingAffinityPredictor()

        self.assertFalse(predictor.is_trained)
        self.assertIsNone(predictor.regressor)
        self.assertIsNone(predictor.classifier)
        self.assertIsNone(predictor.scaler)

        prediction = predictor.predict({})
        self.assertEqual(prediction["prediction_method"], "Rule-based fallback")

    @patch("ligand_binding_predictor.Path.exists", return_value=False)
    def test_predict_returns_stable_schema(self, _mock_exists):
        predictor = BindingAffinityPredictor()
        prediction = predictor.predict({"molecular_weight": 320, "logp": 2.1, "hbd": 1, "hba": 4})

        expected_keys = {
            "binding_affinity",
            "binding_affinity_units",
            "binding_likelihood",
            "binding_probability",
            "prediction_method",
            "confidence",
        }
        self.assertTrue(expected_keys.issubset(prediction.keys()))


class TestLigandBindingRanking(unittest.TestCase):
    def test_ranking_order_correct_for_paffinity(self):
        predictor = object.__new__(LigandBindingPredictor)

        predictions = [
            {
                "molecule_name": "A",
                "is_valid": True,
                "prediction": {
                    "binding_likelihood": 80.0,
                    "binding_affinity": 7.2,
                    "binding_affinity_units": "pAffinity (-log10(M))",
                },
            },
            {
                "molecule_name": "B",
                "is_valid": True,
                "prediction": {
                    "binding_likelihood": 80.0,
                    "binding_affinity": 8.1,
                    "binding_affinity_units": "pAffinity (-log10(M))",
                },
            },
        ]

        ranked = LigandBindingPredictor.rank_molecules(predictor, predictions, top_n=2)

        self.assertEqual(ranked[0]["molecule_name"], "B")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["molecule_name"], "A")
        self.assertEqual(ranked[1]["rank"], 2)


if __name__ == "__main__":
    unittest.main()
