import pytest
from backend.core.config import get_settings
from backend.db.base import Base
from backend.db.session import get_db
from backend.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def _build_payload(include_transcriptomics: bool = False, include_proteomics: bool = False) -> dict:
    sample_omics = {
        "genomics": {
            "mutations": {"BRCA1": 1.0, "PIK3CA": 0.8, "TP53": 0.7},
            "cnv": {"BRCA1": 0.4, "PIK3CA": 0.3},
        }
    }
    if include_transcriptomics:
        sample_omics["transcriptomics"] = {"BRCA1": 0.9, "PIK3CA": 0.7, "TP53": 0.8}
    if include_proteomics:
        sample_omics["proteomics"] = {"BRCA1": 0.6, "PIK3CA": 0.5, "TP53": 0.4}

    return {
        "drug": {
            "name": "Olaparib",
            "smiles": "CCN(CC)CCOC(=O)N1CCC(CC1)C2=NC=NC3=CC=CC=C23",
            "descriptors": {"molecular_weight": 434.5, "logp": 2.9},
        },
        "sample_omics": sample_omics,
    }


@pytest.fixture()
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("MULTIOMICS_CALIBRATION_A", raising=False)
    monkeypatch.delenv("MULTIOMICS_CALIBRATION_B", raising=False)
    monkeypatch.setenv("MULTIOMICS_ENABLED", "true")
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


def test_predict_multiomics_genomics_only(app_client: TestClient) -> None:
    response = app_client.post("/api/v1/multiomics/predict", json=_build_payload())
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["predicted_response_probability"] <= 1.0
    assert 0.0 <= payload["predicted_sensitivity_score"] <= 1.0
    assert 0.0 <= payload["uncertainty"] <= 1.0
    assert payload["modality_usage_summary"]["used_modalities"] == ["genomics"]
    assert "transcriptomics" in payload["modality_usage_summary"]["missing_modalities"]
    assert "proteomics" in payload["modality_usage_summary"]["missing_modalities"]


def test_predict_multiomics_with_all_modalities(app_client: TestClient) -> None:
    response = app_client.post(
        "/api/v1/multiomics/predict",
        json=_build_payload(include_transcriptomics=True, include_proteomics=True),
    )
    assert response.status_code == 200
    payload = response.json()
    used = payload["modality_usage_summary"]["used_modalities"]
    assert "genomics" in used
    assert "transcriptomics" in used
    assert "proteomics" in used
    assert payload["top_pathways"]
    assert payload["top_features"]


def test_missing_modality_fallback_for_cohort(app_client: TestClient) -> None:
    payload = _build_payload(include_transcriptomics=True)
    payload.pop("sample_omics")
    payload["cohort_omics"] = [
        {"genomics": {"mutations": {"BRCA1": 1.0}, "cnv": {"BRCA1": 0.1}}},
        {
            "genomics": {"mutations": {"PIK3CA": 0.8}, "cnv": {"PIK3CA": 0.4}},
            "transcriptomics": {"PIK3CA": 0.7},
            "proteomics": {"PIK3CA": 0.5},
        },
    ]

    response = app_client.post("/api/v1/multiomics/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["modality_usage_summary"]["samples_evaluated"] == 2
    assert "genomics" in body["modality_usage_summary"]["used_modalities"]
    assert body["predicted_response_probability"] >= 0.0


def test_response_calibration_settings_affect_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    request_payload = _build_payload(include_transcriptomics=True)

    def build_client(a: str, b: str) -> TestClient:
        monkeypatch.setenv("MULTIOMICS_ENABLED", "true")
        monkeypatch.setenv("MULTIOMICS_CALIBRATION_A", a)
        monkeypatch.setenv("MULTIOMICS_CALIBRATION_B", b)
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

    baseline_client = build_client("1.0", "0.0")
    baseline = baseline_client.post("/api/v1/multiomics/predict", json=request_payload)
    boosted_client = build_client("4.0", "0.0")
    boosted = boosted_client.post("/api/v1/multiomics/predict", json=request_payload)
    assert baseline.status_code == 200
    assert boosted.status_code == 200
    assert boosted.json()["predicted_response_probability"] > baseline.json()["predicted_response_probability"]


def test_explanation_is_pathway_focused(app_client: TestClient) -> None:
    response = app_client.post(
        "/api/v1/multiomics/predict",
        json=_build_payload(include_transcriptomics=True),
    )
    assert response.status_code == 200
    payload = response.json()
    assert "pathway" in payload["top_pathways"][0]
    assert "contribution" in payload["explanation_text"].lower() or "pathway" in payload["explanation_text"].lower()
