import unittest

from wet_lab_handoff_engine import WetLabHandoffEngine


class TestWetLabHandoffEngine(unittest.TestCase):
    def setUp(self):
        self.engine = WetLabHandoffEngine()
        self.sequence = ("ATGCGGCTAGCTAGGCTTACG" * 20) + "GGGCCCGGATCCGATCGATCGGATCC"
        self.context = self.engine.build_experiment_context(
            target_data={"gene_name": "EGFR", "uniprot_id": "P00533", "protein_name": "Epidermal growth factor receptor"},
            sequence_data={"sequence": self.sequence, "gene_name": "EGFR"},
            pathway_data={"pathways": [{"name": "MAPK signaling"}, {"name": "PI3K-Akt signaling"}]},
            variant_data={"annotated": [{"predicted_effect_class": "high"}, {"predicted_effect_class": "moderate"}]},
        )

    def test_assay_ranking_logic(self):
        lab_profile = {
            "available_instruments": ["qPCR", "western", "plate reader"],
            "throughput_preference": "medium",
            "budget_tier": "medium",
        }
        assays = self.engine.suggest_assays(self.context, objective="target validation", lab_profile=lab_profile)
        self.assertGreaterEqual(len(assays), 6)
        self.assertGreaterEqual(assays[0]["rank_score"], assays[-1]["rank_score"])
        required_fields = {
            "assay_name",
            "purpose",
            "why_it_fits_this_target",
            "required_materials",
            "readout_type",
            "positive_negative_controls",
            "expected_signal_pattern",
            "turnaround_estimate",
            "cost_tier",
            "risk_flags",
            "confidence",
            "references_evidence_tags",
            "top_3_drivers",
            "top_3_risks_assumptions",
            "missing_inputs_reducing_uncertainty",
        }
        self.assertTrue(required_fields.issubset(set(assays[0].keys())))

    def test_crispr_candidate_scoring_deterministic(self):
        first = self.engine.suggest_crispr_targets(self.sequence, "EGFR", genome_build="GRCh38")
        second = self.engine.suggest_crispr_targets(self.sequence, "EGFR", genome_build="GRCh38")
        self.assertTrue(first)
        self.assertEqual(first[0]["spacer_sequence"], second[0]["spacer_sequence"])
        self.assertEqual(first[0]["heuristic_score"], second[0]["heuristic_score"])
        self.assertIn("preliminary heuristic only", first[0]["notes_for_manual_review"].lower())

    def test_primer_constraints_and_filtering(self):
        crispr = self.engine.suggest_crispr_targets(self.sequence, "EGFR", genome_build="GRCh38")
        regions = crispr[:3]
        constraints = {
            "intended_use": "qPCR",
            "min_len": 18,
            "max_len": 24,
            "min_gc": 40.0,
            "max_gc": 62.0,
            "min_tm": 57.0,
            "max_tm": 65.0,
            "amplicon_min": 80,
            "amplicon_max": 220,
        }
        primers = self.engine.suggest_primers(self.sequence, regions, constraints)
        self.assertTrue(primers)
        for row in primers[:10]:
            self.assertGreaterEqual(row["expected_amplicon_size"], 80)
            self.assertLessEqual(row["expected_amplicon_size"], 220)
            self.assertIn("quality_score", row)
            self.assertIn("caveats", row)

    def test_checklist_schema_and_completeness(self):
        plan = {
            "context": self.context,
            "assays": [{"assay_name": "qPCR", "risk_flags": []}],
            "crispr_candidates": [],
            "primers": [],
        }
        checklist = self.engine.generate_validation_checklist(plan, objective="knockout")
        self.assertGreaterEqual(len(checklist), 21)
        for item in checklist:
            self.assertIn("section", item)
            self.assertIn("item_text", item)
            self.assertIn("severity_if_skipped", item)
            self.assertIn("owner_role_suggestion", item)
            self.assertEqual(item["completion_status"], "unchecked")

    def test_confidence_readiness_computation(self):
        plan = {
            "context": self.context,
            "assays": [{"confidence": "High", "risk_flags": []}],
            "crispr_candidates": [{"heuristic_score": 82.0, "off_target_risk_level": "Low"}],
            "primers": [{"quality_score": 78.0, "caveats": ["heuristic only"]}],
        }
        conf = self.engine.compute_plan_confidence(plan)
        self.assertIn("plan_confidence_score", conf)
        self.assertIn("data_completeness", conf)
        self.assertIn(conf["readiness_label"], {"red", "yellow", "green"})
        self.assertLessEqual(conf["plan_confidence_score"], 100.0)

    def test_export_schema_validation(self):
        plan = {
            "objective": "target validation",
            "context": self.context,
            "assays": self.engine.suggest_assays(
                self.context,
                objective="target validation",
                lab_profile={"available_instruments": ["qPCR", "western", "plate reader"], "throughput_preference": "medium", "budget_tier": "medium"},
            )[:3],
            "crispr_candidates": self.engine.suggest_crispr_targets(self.sequence, "EGFR", genome_build="GRCh38")[:5],
            "primers": [],
            "validation_checklist": [],
        }
        plan["confidence"] = self.engine.compute_plan_confidence(plan)
        plan["validation_checklist"] = self.engine.generate_validation_checklist(plan, objective="target validation")
        exported = self.engine.export_wet_lab_package(plan, format="json")
        self.assertIn("content", exported)
        content = exported["content"]
        self.assertIn("json", content)
        self.assertIn("csv", content)
        self.assertIn("markdown_brief", content)
        self.assertIn("txt_brief", content)
        self.assertIn("assays", content["csv"])
        self.assertIn("disclaimer", content)


if __name__ == "__main__":
    unittest.main()
