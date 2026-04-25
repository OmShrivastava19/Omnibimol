from __future__ import annotations

import asyncio

from backend.services.chemprot import ChemProtInteractionService, extract_evidence_features


class CountingBackend:
    def __init__(self, *, lora_prob: float = 0.81, ensemble_prob: float | None = None):
        self.lora_prob = lora_prob
        self.ensemble_prob = ensemble_prob
        self.lora_calls = 0
        self.ensemble_calls = 0

    def predict_lora_proba(self, records):
        self.lora_calls += 1
        return [self.lora_prob for _ in records]

    def predict_ensemble_proba(self, records):
        self.ensemble_calls += 1
        if self.ensemble_prob is None:
            return None
        return [self.ensemble_prob for _ in records]

    def metadata(self):
        return {"backend": "fake", "loaded": True, "ensemble_loaded": self.ensemble_prob is not None}

    def health_snapshot(self):
        return {"backend": "fake", "loaded": True, "degraded": self.ensemble_prob is None}


def test_extract_evidence_features_detects_mentions_and_keywords() -> None:
    features = extract_evidence_features(
        chemical="imatinib",
        protein="ABL1",
        abstract="Imatinib inhibits ABL1 and shows strong binding interaction in leukemia studies.",
        pmid="12345",
        disease_context="leukemia",
    )

    assert features["cooccurrence"] == 1
    assert features["chem_in_context"] == 1
    assert features["prot_in_context"] == 1
    assert features["keyword_count"] >= 2
    assert features["disease_context_hit"] == 1
    assert "ABL1" in features["snippet"] or "imatinib" in features["snippet"]


def test_lazy_loading_caches_backend_singleton() -> None:
    load_count = {"value": 0}

    def backend_factory(_settings):
        load_count["value"] += 1
        return CountingBackend()

    service = ChemProtInteractionService(backend_factory=backend_factory)
    payload = {
        "chemical": "imatinib",
        "candidate_proteins": ["ABL1"],
        "abstracts": [
            {
                "pmid": "12345",
                "title": "Example",
                "abstract": "Imatinib inhibits ABL1 in chronic myeloid leukemia.",
            }
        ],
    }

    first = asyncio.run(service.score_request(**payload))
    second = asyncio.run(service.score_request(**payload))

    assert load_count["value"] == 1
    assert first["ranked_targets"][0]["protein"] == "ABL1"
    assert first["ranked_targets"][0]["final_score"] == second["ranked_targets"][0]["final_score"]


def test_score_request_returns_expected_schema_and_range() -> None:
    service = ChemProtInteractionService(backend_factory=lambda _settings: CountingBackend())
    response = asyncio.run(
        service.score_request(
            chemical="imatinib",
            candidate_proteins=["ABL1"],
            abstracts=[
                {
                    "pmid": "12345",
                    "title": "Example",
                    "abstract": "Imatinib interacts with ABL1 in a leukemia model.",
                }
            ],
        )
    )

    assert 0.0 <= response["interaction_probability"] <= 1.0
    assert 0.0 <= response["final_score"] <= 1.0
    assert response["reranker_used"] is False
    assert response["ranked_targets"]
    top = response["ranked_targets"][0]
    assert top["protein"] == "ABL1"
    assert "evidence" in top
    assert "model_metadata" in top


def test_ensemble_fallback_when_artifact_missing() -> None:
    service = ChemProtInteractionService(
        backend_factory=lambda _settings: CountingBackend(lora_prob=0.73, ensemble_prob=None)
    )
    response = asyncio.run(
        service.score_request(
            chemical="imatinib",
            candidate_proteins=["ABL1"],
            abstracts=[
                {
                    "pmid": "12345",
                    "title": "Example",
                    "abstract": "Imatinib interacts with ABL1 in a leukemia model.",
                }
            ],
        )
    )

    assert response["reranker_used"] is False
    assert response["degraded_mode"] is True
    assert response["final_score"] == response["interaction_probability"]
