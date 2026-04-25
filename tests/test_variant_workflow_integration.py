import unittest
from unittest.mock import MagicMock

from genome_analysis_engine import GenomeAnalysisEngine, Variant
from variant_therapy_engine import VariantTherapyEngine


class StubPrioritizer:
    PRECOMPUTED = [
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
    PROTEIN = [
        "aa_position",
        "aa_change_type",
        "domain_score",
        "conservation_score",
        "blosum62_score",
        "grantham_distance",
        "sift_score",
        "polyphen_score",
    ]

    def validate_features(self, features, feature_type):
        expected = self.PRECOMPUTED if feature_type == "precomputed" else self.PROTEIN if feature_type == "protein" else []
        present = [name for name in expected if name in features]
        return {
            "valid": len(present) == len(expected),
            "completeness": round(len(present) / len(expected), 4) if expected else 0.0,
            "present": present,
            "missing": [name for name in expected if name not in features],
            "invalid": [],
            "expected_features": expected,
        }

    def predict_pathogenicity(self, features, feature_type="auto"):
        if feature_type == "precomputed":
            return {
                "score": 0.93,
                "model_used": "xgb_precomputed",
                "confidence_label": "high",
                "evidence_features_used": [name for name in self.PRECOMPUTED if name in features],
                "missing_features": [name for name in self.PRECOMPUTED if name not in features],
                "fallback_reason": None,
                "tier": 1,
                "metadata": {"source": "stub"},
            }
        if feature_type == "protein":
            return {
                "score": 0.71,
                "model_used": "rf_protein",
                "confidence_label": "medium",
                "evidence_features_used": [name for name in self.PROTEIN if name in features],
                "missing_features": [name for name in self.PROTEIN if name not in features],
                "fallback_reason": "precomputed_features_missing",
                "tier": 2,
                "metadata": {"source": "stub"},
            }
        return {
            "score": 0.58,
            "model_used": "lr_protein",
            "confidence_label": "low",
            "evidence_features_used": list(features.keys()),
            "missing_features": [],
            "fallback_reason": "minimal_features_available",
            "tier": 3,
            "metadata": {"source": "stub"},
        }


class TestVariantWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        self.prioritizer = StubPrioritizer()
        self.therapy_engine = VariantTherapyEngine(prioritizer=self.prioritizer)

    def test_variant_therapy_pipeline_propagates_pathogenicity_fields(self):
        variant = {
            "variant_key": "1:123:A>T",
            "gene": "TP53",
            "ref": "A",
            "alt": "T",
            "consequence": "missense_variant",
            "impact_score": 0.64,
            "confidence": "High",
            "genotype": "0/1",
            "filter": "PASS",
            "qual": 99.0,
            "info": {
                "gpn_msa_score": 0.87,
                "CADD": 29.1,
                "phyloP": 2.3,
                "phastCons": 0.8,
                "esm1b_embedding_mean": 0.42,
                "esm1b_embedding_max": 0.91,
                "esm1b_embedding_norm": 0.55,
                "nt_score": 0.73,
                "hyena_dna_embedding_mean": 0.38,
            },
        }

        scored = self.therapy_engine.score_variant_pathogenicity([variant])
        scored_variant = scored[0]

        self.assertEqual(scored_variant["pathogenicity_method"], "xgb_precomputed")
        self.assertEqual(scored_variant["pathogenicity_tier"], 1)
        self.assertEqual(scored_variant["model_confidence"], "High")
        self.assertIn("xgb_precomputed", scored_variant["evidence_summary"])

        gene_impact = self.therapy_engine.aggregate_gene_impact(scored)
        tp53 = gene_impact["genes"]["TP53"]
        self.assertIn("mean_pathogenicity_score", tp53)
        self.assertIn("pathogenicity_score", tp53["top_driving_variants"][0])
        self.assertIn("pathogenicity_method", tp53["top_driving_variants"][0])

        explainability = self.therapy_engine.build_explainability_payload(
            ranked_candidates=[
                {
                    "drug_name": "Example Drug",
                    "target_gene_match": 0.9,
                    "pathway_relevance": 0.7,
                    "evidence_quality": 0.8,
                    "clinical_maturity": 0.6,
                    "safety_risk_penalty": 0.1,
                }
            ],
            gene_impact=gene_impact,
            pathway_impact={"pathways": []},
            parsing_stats={"parsed": 1},
        )
        self.assertIn("pathogenicity_context", explainability["global"])

    def test_genome_analysis_accepts_scored_variant_payloads(self):
        mock_therapy = MagicMock()
        mock_therapy.score_variant_pathogenicity.return_value = [
            {
                "gene": "TP53",
                "variant_key": "1:123:A>T",
                "pathogenicity_score": 0.9,
                "pathogenicity_tier": 1,
                "pathogenicity_method": "xgb_precomputed",
                "model_confidence": "High",
                "evidence_summary": "stub",
            }
        ]

        genome_engine = GenomeAnalysisEngine(variant_therapy_engine=mock_therapy)
        result = genome_engine.analyze_genome(
            "ATGCGTATGCGTATGCGTATGCGTATGCGT",
            annotated_variants=[{"gene": "TP53", "consequence": "missense_variant"}],
        )

        self.assertIn("variant_prioritization", result)
        self.assertEqual(result["variant_prioritization"]["total_variants"], 1)
        self.assertEqual(result["variant_prioritization"]["scored_variants"][0]["pathogenicity_score"], 0.9)
        mock_therapy.score_variant_pathogenicity.assert_called_once()

    def test_risk_score_prefers_pathogenicity_signal(self):
        engine = GenomeAnalysisEngine(variant_therapy_engine=self.therapy_engine)
        high_pathogenicity = Variant(
            gene="TP53",
            variant_id="v1",
            type="Missense",
            description="test",
            confidence=0.2,
            pathogenicity_score=0.95,
        )
        no_pathogenicity = Variant(
            gene="TP53",
            variant_id="v2",
            type="Missense",
            description="test",
            confidence=0.2,
            pathogenicity_score=None,
        )

        with_path = engine.mutation_analyzer.calculate_risk_score([high_pathogenicity])
        without_path = engine.mutation_analyzer.calculate_risk_score([no_pathogenicity])

        self.assertGreater(with_path, without_path)


if __name__ == "__main__":
    unittest.main()