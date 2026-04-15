import httpx
import pytest
from backend.core.config import get_settings
from backend.core.errors import UpstreamUnavailableError
from backend.db.base import Base
from backend.db.session import get_db
from backend.integrations.resilient_http import ResilientHttpClient
from backend.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def app_client() -> TestClient:
    get_settings.cache_clear()
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)

    def override_get_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_invalid_upload_returns_standard_error_schema(app_client: TestClient) -> None:
    response = app_client.post(
        "/api/v1/reliability/validate-upload",
        json={"filename": "sample.fasta", "content": "INVALID"},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_input"
    assert "request_id" in payload["error"]


def test_project_not_found_uses_standard_error_schema(app_client: TestClient) -> None:
    response = app_client.get("/api/v1/projects/99999")
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "http_404"
    assert payload["error"]["message"] == "Project not found"


def test_upstream_failure_gracefully_degrades(
    monkeypatch: pytest.MonkeyPatch,
    app_client: TestClient,
) -> None:
    def fail_request(_self, _url):
        raise UpstreamUnavailableError(message="simulated upstream outage")

    monkeypatch.setattr(ResilientHttpClient, "get_json", fail_request)
    response = app_client.get("/api/v1/reliability/upstream-status", params={"url": "https://example.org"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["error"]["code"] == "upstream_unavailable"


def test_circuit_breaker_opens_after_repeated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ResilientHttpClient(max_retries=0, failure_threshold=2, reset_timeout_seconds=60)

    def failing_get(_self, url):
        request = httpx.Request("GET", url)
        raise httpx.RequestError("boom", request=request)

    monkeypatch.setattr(httpx.Client, "get", failing_get)

    with pytest.raises(UpstreamUnavailableError):
        client.get_json("https://api.example.org/resource")
    with pytest.raises(UpstreamUnavailableError):
        client.get_json("https://api.example.org/resource")

    with pytest.raises(UpstreamUnavailableError) as exc:
        client.get_json("https://api.example.org/resource")
    assert exc.value.details["host"] == "api.example.org"
