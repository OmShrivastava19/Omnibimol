import unittest

from target_prioritization_engine import TargetPrioritizationEngine


class TestTargetPrioritizationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TargetPrioritizationEngine()
        self.payload = {
            "target_id": "EGFR",
            "expression_data": {
                "tissues": [
                    {"tissue": "Lung", "level_numeric": 3},
                    {"tissue": "Skin", "level_numeric": 2},
                    {"tissue": "Brain", "level_numeric": 1},
                ],
                "disease_tissues": ["Lung"],
            },
            "pathway_data": {
                "available": True,
                "pathways": [{"name": "MAPK"}, {"name": "PI3K"}, {"name": "EGFR signaling"}],
                "graph_centrality": 0.7,
            },
            "ppi_data": {
                "available": True,
                "interactions": [
                    {"combined_score": 920},
                    {"combined_score": 830},
                    {"combined_score": 770},
                ],
            },
            "genetic_data": {
                "mutation_analysis": {"total_variants": 4, "high_risk_variants": 2},
                "biomarker_detection": {"therapeutic_targets": 2},
                "disease_associations": {"high_confidence": 2, "moderate_confidence": 1},
            },
            "ligandability_data": {
                "chembl": {"available": True, "ligands": [{"id": "L1"}, {"id": "L2"}]},
                "docking": {"binding_affinity": -8.6, "simulated": True},
                "binding_prediction": {"available": True, "ranked_molecules": [{"binding_likelihood": 72}]},
            },
            "trial_data": {
                "available": True,
                "clinical_trials": [
                    {"phase": "PHASE3", "status": "COMPLETED"},
                    {"phase": "PHASE2", "status": "RECRUITING"},
                ],
            },
        }

    def test_component_scores_in_range(self):
        scores = self.engine.compute_component_scores(self.payload)
        for key, value in scores.items():
            if value["available"]:
                self.assertGreaterEqual(value["score"], 0.0)
                self.assertLessEqual(value["score"], 100.0)

    def test_missing_data_renormalization(self):
        partial_scores = {
            "expression": {"available": True, "score": 80, "source_quality": 0.8},
            "pathway": {"available": False, "score": 0, "source_quality": 0.0},
            "ppi": {"available": True, "score": 60, "source_quality": 0.7},
            "genetic": {"available": False, "score": 0, "source_quality": 0.0},
            "ligandability": {"available": True, "score": 50, "source_quality": 0.6},
            "trials": {"available": False, "score": 0, "source_quality": 0.0},
        }
        result = self.engine.compute_composite_score(partial_scores)
        self.assertEqual(set(result["renormalized_weights"].keys()), {"expression", "ppi", "ligandability"})
        self.assertAlmostEqual(sum(result["renormalized_weights"].values()), 1.0, places=3)

    def test_deterministic_output(self):
        first = self.engine.rank_targets([self.payload])[0]
        second = self.engine.rank_targets([self.payload])[0]
        self.assertEqual(first["composite_score"], second["composite_score"])
        self.assertEqual(first["confidence_score"], second["confidence_score"])

    def test_ranking_stability_for_ties(self):
        payload_a = dict(self.payload)
        payload_b = dict(self.payload)
        payload_a["target_id"] = "AAA1"
        payload_b["target_id"] = "BBB1"
        ranked = self.engine.rank_targets([payload_b, payload_a])
        self.assertEqual(ranked[0]["target_id"], "AAA1")
        self.assertEqual(ranked[1]["target_id"], "BBB1")

    def test_explainability_schema(self):
        ranked = self.engine.rank_targets([self.payload])[0]
        explain = ranked["explainability"]
        self.assertIn("breakdown", explain)
        self.assertIn("top_positive_drivers", explain)
        self.assertIn("top_risk_flags", explain)
        self.assertIn("improvement_suggestions", explain)
        self.assertIn("rationale", explain)
        self.assertLessEqual(len(explain["top_positive_drivers"]), 3)
        self.assertLessEqual(len(explain["top_risk_flags"]), 3)

    def test_integration_known_payload(self):
        ranked = self.engine.rank_targets([self.payload])
        self.assertEqual(len(ranked), 1)
        row = ranked[0]
        self.assertGreater(row["composite_score"], 0)
        self.assertGreater(row["confidence_score"], 0)
        self.assertIsInstance(row["missing_components"], list)


if __name__ == "__main__":
    unittest.main()
