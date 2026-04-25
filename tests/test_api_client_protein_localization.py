from __future__ import annotations

from api_client import ProteinAPIClient


class _DummyCache:
    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None


def test_predict_protein_localization_posts_expected_contract(monkeypatch) -> None:
    client = ProteinAPIClient(_DummyCache(), backend_api_url="http://localhost:8000")
    captured = {}

    def fake_request(method, path, *, json_body=None, timeout=60.0):  # noqa: ANN001
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = json_body
        return {
            "body": {
                "localization": "Nucleus",
                "confidence": 0.78,
                "membrane_risk": 0.12,
                "wetlab_prioritization_score": 70.4,
                "recommended_assay": "Standard E. coli expression + soluble tag",
                "evidence_passed": True,
            },
            "status_code": 200,
            "headers": {"x-request-id": "loc-req-1"},
        }

    monkeypatch.setattr(client, "_request_backend_json_with_metadata", fake_request)
    out = client.predict_protein_localization(sequence="ACDEFGHIKLMNPQRSTVWY" * 15, confidence_threshold=0.6)
    assert out["localization"] == "Nucleus"
    assert out["_client_meta"]["request_id"] == "loc-req-1"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/protein-localization/predict"
    assert captured["json_body"]["confidence_threshold"] == 0.6