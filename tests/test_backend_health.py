from backend.main import create_app
from fastapi.testclient import TestClient


def test_healthz_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_service_metadata() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/readyz")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["database_configured"] is True
    assert payload["redis_configured"] is True
    assert "timestamp_utc" in payload


def test_request_id_header_echoed() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/healthz", headers={"x-request-id": "req-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
