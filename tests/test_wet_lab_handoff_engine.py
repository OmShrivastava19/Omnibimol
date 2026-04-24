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

    def test_crispr_off_target_returns_specificity_and_ranked_sites(self):
        guide = "GCTAGCTAGGCTTACGATCG"
        patient_genome = {
            "sequence": (
                "TTTTT" + guide + "AGG" + "A" * 24 + "GCTAGCTAGGCTTTCGATCGAGG" + "C" * 24
            ),
            "annotations": [
                {
                    "start": 4,
                    "end": 28,
                    "gene_name": "BRCA2",
                    "is_coding": True,
                    "is_cancer_gene": True,
                    "is_essential_gene": True,
                    "is_regulatory_hotspot": False,
                },
                {
                    "start": 52,
                    "end": 80,
                    "gene_name": "BCL6",
                    "is_coding": True,
                    "is_cancer_gene": False,
                    "is_essential_gene": False,
                    "is_regulatory_hotspot": True,
                },
            ],
            "variants": [{"position": 18, "pathogenic": True}],
        }
        result = self.engine.analyze_crispr_off_targets(guide, patient_genome)
        self.assertIn("specificity_score_pct", result)
        self.assertIn("ranked_off_targets", result)
        self.assertTrue(result["ranked_off_targets"])
        self.assertIn("summary_text", result)
        self.assertIn("specificity", result["summary_text"].lower())

    def test_crispr_off_target_deterministic_for_same_inputs(self):
        guide = "GCTAGCTAGGCTTACGATCG"
        patient_genome = {
            "sequence": "AAAAA" + guide + "AGG" + "TTTTT" + "GCTAGCTAGGCTTTCGATCGAGG" + "GGGGG",
            "annotations": [{"start": 4, "end": 28, "gene_name": "BRCA2", "is_coding": True, "is_cancer_gene": True}],
            "variants": [],
        }
        first = self.engine.analyze_crispr_off_targets(guide, patient_genome)
        second = self.engine.analyze_crispr_off_targets(guide, patient_genome)
        self.assertEqual(first["specificity_score_pct"], second["specificity_score_pct"])
        self.assertEqual(first["ranked_off_targets"], second["ranked_off_targets"])

    def test_crispr_off_target_candidate_fields_present(self):
        guide = "GCTAGCTAGGCTTACGATCG"
        patient_genome = {
            "sequence": "AAAAA" + guide + "AGG" + "TTTTT" + "GCTAGCTAGGCTTTCGATCGAGG" + "GGGGG",
            "annotations": [{"start": 4, "end": 28, "gene_name": "BRCA2", "is_coding": True, "is_cancer_gene": True}],
            "variants": [],
        }
        result = self.engine.analyze_crispr_off_targets(guide, patient_genome)
        expected_fields = {
            "site_position",
            "site_end",
            "strand",
            "pam",
            "candidate_sequence",
            "mismatches",
            "bulges",
            "seed_mismatches",
            "alignment_label",
            "cleavage_probability",
            "impact_weight",
            "impact_flags",
            "gene_name",
            "risk_score",
            "tier_label",
            "rank_explanation",
        }
        self.assertTrue(expected_fields.issubset(set(result["ranked_off_targets"][0].keys())))

    def test_crispr_off_target_tiering_high_impact_vs_lower_impact(self):
        guide = "GCTAGCTAGGCTTACGATCG"
        patient_genome = {
            "sequence": (
                "TTTTT" + guide + "AGG" + "A" * 24 + "GCTAGCTAGGCTTTCGATCGAGG" + "C" * 24
            ),
            "annotations": [
                {
                    "start": 4,
                    "end": 28,
                    "gene_name": "BRCA2",
                    "is_coding": True,
                    "is_cancer_gene": True,
                    "is_essential_gene": True,
                },
                {
                    "start": 52,
                    "end": 80,
                    "gene_name": "GENE_X",
                    "is_coding": False,
                    "is_cancer_gene": False,
                    "is_essential_gene": False,
                },
            ],
            "variants": [],
        }
        result = self.engine.analyze_crispr_off_targets(guide, patient_genome)
        tiers = {row["gene_name"]: row["tier_label"] for row in result["ranked_off_targets"]}
        scores = {row["gene_name"]: row["risk_score"] for row in result["ranked_off_targets"]}
        self.assertEqual(tiers.get("BRCA2"), "Tier 1")
        self.assertGreater(scores.get("BRCA2", 0.0), scores.get("GENE_X", 0.0))

    def test_crispr_off_target_missing_annotations_are_conservative(self):
        guide = "GCTAGCTAGGCTTACGATCG"
        patient_genome = {"sequence": "AAAAA" + guide + "AGG" + "TTTTT" + "GCTAGCTAGGCTTTCGATCGAGG" + "GGGGG"}
        result = self.engine.analyze_crispr_off_targets(guide, patient_genome)
        self.assertIn("missing_inputs_reducing_uncertainty", result)
        joined = " ".join(result["missing_inputs_reducing_uncertainty"]).lower()
        self.assertIn("annotations", joined)
        self.assertIn("uncertainty notes", result["summary_text"].lower())

    def test_export_includes_optional_off_target_analysis(self):
        guide = "GCTAGCTAGGCTTACGATCG"
        off_target = self.engine.analyze_crispr_off_targets(
            guide,
            {
                "sequence": "AAAAA" + guide + "AGG" + "TTTTT" + "GCTAGCTAGGCTTTCGATCGAGG" + "GGGGG",
                "annotations": [{"start": 4, "end": 28, "gene_name": "BRCA2", "is_coding": True, "is_cancer_gene": True}],
            },
        )
        plan = {
            "objective": "target validation",
            "context": self.context,
            "assays": [],
            "crispr_candidates": [],
            "primers": [],
            "validation_checklist": [],
            "crispr_off_target_analysis": off_target,
            "confidence": self.engine.compute_plan_confidence(
                {"context": self.context, "assays": [], "crispr_candidates": [], "primers": []}
            ),
        }
        exported = self.engine.export_wet_lab_package(plan, format="json")
        self.assertIn("crispr_off_targets", exported["content"]["csv"])
        self.assertIn("CRISPR Off-Target Risk", exported["content"]["markdown_brief"])


if __name__ == "__main__":
    unittest.main()
