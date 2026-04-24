from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from academic_model_hub.adapters.deepathnet_adapter import DeePathNetAdapter
from academic_model_hub.adapters.flexpose_adapter import FlexPoseAdapter
from backend.db.base import Base
from backend.db.session import get_db
from backend.main import create_app

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "academic_model_hub"


def test_runtime_compatibility_failure_maps_to_error() -> None:
    adapter = FlexPoseAdapter()
    payload = {
        "protein_path": str(FIXTURE_DIR / "tiny_protein.pdb"),
        "ligand": "CCO",
        "ref_pocket_center": str(FIXTURE_DIR / "ref_ligand.mol2"),
        "runtime_mode": "native",
    }
    result = adapter.predict(payload)
    assert result["status"] == "error"
    assert result["errors"][0]["code"] == "INCOMPATIBLE_RUNTIME"


def test_deepathnet_container_command_build(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd=None, timeout_seconds=120, env=None):  # noqa: ANN001
        captured["command"] = command
        out = FIXTURE_DIR / "out" / "deepathnet_predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("sample_id,response_score\nS1,0.5\n", encoding="utf-8")
        return None

    monkeypatch.setattr("academic_model_hub.adapters.deepathnet_adapter.run_command_allow_timeout", fake_run)
    monkeypatch.setenv("DEEPATHNET_REPO_PATH", str(FIXTURE_DIR))
    monkeypatch.setattr(DeePathNetAdapter, "runtime_precheck", lambda self, payload, mode: {"mode": "container"})
    adapter = DeePathNetAdapter()
    payload = {
        "input_table_path": str(FIXTURE_DIR / "mini_omics.csv"),
        "task": "drug_response",
        "pretrained_model_path": str(FIXTURE_DIR / "deepathnet_weights.pth"),
        "config_path": str(FIXTURE_DIR / "deepathnet_config.json"),
        "output_dir": str(FIXTURE_DIR / "out"),
        "runtime_mode": "container",
    }
    result = adapter.predict(payload)
    assert result["status"] == "success"
    assert str(captured["command"][0]) == "docker"


def test_provenance_manifest_and_run_summary_created() -> None:
    adapter = FlexPoseAdapter()
    payload = {
        "protein_path": str(FIXTURE_DIR / "tiny_protein.pdb"),
        "ligand": "CCO",
        "ref_pocket_center": str(FIXTURE_DIR / "ref_ligand.mol2"),
        "output_dir": str(FIXTURE_DIR / "out"),
        "mock_outputs": True,
    }
    result = adapter.predict(payload)
    manifest = Path(result["artifacts"]["provenance_manifest"])
    summary = manifest.parent / "run_summary.json"
    assert manifest.exists()
    assert summary.exists()


@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _):  # noqa: ANN001
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


def test_api_contract_models_health_predict(api_client: TestClient) -> None:
    models = api_client.get("/api/v1/academic-models/models")
    assert models.status_code == 200
    health = api_client.get("/api/v1/academic-models/health?depth=shallow")
    assert health.status_code == 200
    unknown = api_client.post("/api/v1/academic-models/predict", json={"model_name": "unknown", "payload": {}})
    assert unknown.status_code == 200
    body = unknown.json()
    assert body["status"] == "error"
    assert "prediction" in body and "errors" in body
