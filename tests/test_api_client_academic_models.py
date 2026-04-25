from __future__ import annotations

import httpx

from api_client import ProteinAPIClient


class _DummyCache:
    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None


def test_get_academic_models_uses_expected_endpoint(monkeypatch) -> None:
    client = ProteinAPIClient(_DummyCache(), backend_api_url="http://localhost:8000")
    captured = {}

    def fake_request(method, path, *, json_body=None, timeout=60.0):  # noqa: ANN001
        captured["method"] = method
        captured["path"] = path
        captured["timeout"] = timeout
        return [{"model_name": "flexpose"}]

    monkeypatch.setattr(client, "_request_backend_json", fake_request)
    out = client.get_academic_models()
    assert out[0]["model_name"] == "flexpose"
    assert captured["method"] == "GET"
    assert captured["path"] == "/api/v1/academic-models/models"


def test_predict_academic_model_posts_contract(monkeypatch) -> None:
    client = ProteinAPIClient(_DummyCache(), backend_api_url="http://localhost:8000")
    captured = {}

    def fake_request(method, path, *, json_body=None, timeout=60.0):  # noqa: ANN001
        captured["method"] = method
        captured["path"] = path
        captured["json_body"] = json_body
        return {"body": {"status": "success", "provenance": {"request_hash": "abc"}}, "status_code": 200, "headers": {"x-request-id": "req-123"}}

    monkeypatch.setattr(client, "_request_backend_json_with_metadata", fake_request)
    out = client.predict_academic_model(model_name="flexpose", payload={"a": 1}, timeout=120.0)
    assert out["status"] == "success"
    assert out["provenance"]["request_id"] == "req-123"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/academic-models/predict"
    assert captured["json_body"]["model_name"] == "flexpose"


def test_predict_academic_model_retries_on_transient(monkeypatch) -> None:
    client = ProteinAPIClient(_DummyCache(), backend_api_url="http://localhost:8000")
    calls = {"count": 0}

    def fake_request(method, path, *, json_body=None, timeout=60.0):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("temporary connection error")
        return {"body": {"status": "success", "provenance": {}}, "status_code": 200, "headers": {}}

    monkeypatch.setattr(client, "_request_backend_json_with_metadata", fake_request)
    out = client.predict_academic_model(model_name="deepathnet", payload={"runtime_mode": "native"})
    assert out["status"] == "success"
    assert calls["count"] == 2


def test_predict_academic_model_maps_timeout_to_error(monkeypatch) -> None:
    client = ProteinAPIClient(_DummyCache(), backend_api_url="http://localhost:8000")

    def fake_request(method, path, *, json_body=None, timeout=60.0):  # noqa: ANN001
        raise httpx.ReadTimeout("read timeout")

    monkeypatch.setattr(client, "_request_backend_json_with_metadata", fake_request)
    out = client.predict_academic_model(model_name="crispr-dipoff", payload={"runtime_mode": "native"}, retry_once_on_network_error=False)
    assert out["status"] == "error"
    assert out["errors"][0]["code"] == "UPSTREAM_TIMEOUT"


def test_predict_academic_model_retries_once_on_timeout(monkeypatch) -> None:
    client = ProteinAPIClient(_DummyCache(), backend_api_url="http://localhost:8000")
    calls = {"count": 0}

    def fake_request(method, path, *, json_body=None, timeout=60.0):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("timeout")
        return {"body": {"status": "success", "provenance": {}}, "status_code": 200, "headers": {"x-request-id": "req-timeout"}}

    monkeypatch.setattr(client, "_request_backend_json_with_metadata", fake_request)
    out = client.predict_academic_model(model_name="deepdtagen", payload={"runtime_mode": "native"}, retry_once_on_network_error=True)
    assert out["status"] == "success"
    assert out["provenance"]["request_id"] == "req-timeout"
    assert calls["count"] == 2


def test_backend_url_normalizes_huggingface_spaces_runtime_domain() -> None:
    client = ProteinAPIClient(
        _DummyCache(),
        backend_api_url="https://huggingface.co/spaces/omshrivastava/Omnibimol",
    )
    assert client.backend_api_url == "https://omshrivastava-omnibimol.hf.space"


def test_backend_headers_include_hf_token_for_hf_space(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test_token")
    client = ProteinAPIClient(_DummyCache(), backend_api_url="https://omshrivastava-omnibimol.hf.space")
    headers = client._build_backend_headers()
    assert headers["Authorization"] == "Bearer hf_test_token"
