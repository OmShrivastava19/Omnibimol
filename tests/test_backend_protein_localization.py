from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from backend.api.routes import protein_localization as localization_routes
from backend.db.base import Base
from backend.db.session import get_db
from backend.main import create_app
from backend.services import protein_localization as localization_service
from backend.services.protein_localization import (
    ProteinLocalizationService,
    clean_protein_sequence,
    compute_wetlab_prioritization_score,
    recommend_assay,
)
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


class FakeLocalizationEngine:
    def predict_proba(self, sequence: str):
        probabilities = {label: 0.0 for label in localization_service.LOCALIZATION_LABELS}
        probabilities["Nucleus"] = 0.78
        probabilities["Cytoplasm"] = 0.08
        probabilities["Cell membrane"] = 0.04
        probabilities["Extracellular"] = 0.03
        probabilities["Endoplasmic reticulum"] = 0.02
        probabilities["Mitochondrion"] = 0.02
        probabilities["Golgi apparatus"] = 0.01
        probabilities["Peroxisome"] = 0.01
        probabilities["Plastid"] = 0.01
        probabilities["Lysosome/Vacuole"] = 0.00
        return {
            "localization_probabilities": probabilities,
            "confidence": 0.8,
            "membrane_risk": 0.12,
            "non_membrane_probability": 0.88,
            "backend": "fake",
        }

    def metadata(self):
        return {"backend": "fake", "loaded": True}

    def health_snapshot(self):
        return {"backend": "fake", "loaded": True, "fallback": False}


@pytest.fixture()
def client() -> TestClient:
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

    def override_get_db() -> Generator[Session, None, None]:
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_clean_sequence_rejects_invalid_residues() -> None:
    with pytest.raises(ValueError, match="standard amino acids"):
        clean_protein_sequence("MKWVTFISLLFLFSSAYSRGVFZX")


def test_wetlab_score_formula_prefers_non_membrane_high_confidence() -> None:
    score = compute_wetlab_prioritization_score(
        non_membrane_probability=0.88,
        confidence=0.8,
        sequence_length=300,
    )
    assert score == 70.4


def test_assay_recommendation_rules() -> None:
    assert recommend_assay(membrane_risk=0.8, confidence=0.95, confidence_threshold=0.6) == "Cell-free / Detergent solubilization / Nanodiscs"
    assert recommend_assay(membrane_risk=0.2, confidence=0.9, confidence_threshold=0.6) == "Standard E. coli expression + soluble tag"
    assert recommend_assay(membrane_risk=0.2, confidence=0.62, confidence_threshold=0.6).startswith("Confirmatory localization assay")


def test_localization_service_uses_mocked_engine() -> None:
    service = ProteinLocalizationService()
    service._engine = FakeLocalizationEngine()  # type: ignore[attr-defined]
    service._bundle = localization_service.LocalizationArtifactBundle(  # type: ignore[attr-defined]
        tokenizer=None,
        esm_model=None,
        esm_lr=None,
        heuristic_lr=None,
        binary_lr=None,
        esm_rf=None,
        heuristic_rf=None,
        metadata={"backend": "fake", "loaded": True},
    )
    service._load_error = None  # type: ignore[attr-defined]

    sequence = "ACDEFGHIKLMNPQRSTVWY" * 15
    result = service.predict(sequence, confidence_threshold=0.6)
    assert result["localization"] == "Nucleus"
    assert result["confidence"] == 0.8
    assert result["membrane_risk"] == 0.12
    assert result["evidence_passed"] is True
    assert result["recommended_assay"] == "Standard E. coli expression + soluble tag"
    assert result["wetlab_prioritization_score"] == 70.4


def test_localization_endpoint_happy_path(client: TestClient) -> None:
    service = ProteinLocalizationService()
    service._engine = FakeLocalizationEngine()  # type: ignore[attr-defined]
    service._bundle = localization_service.LocalizationArtifactBundle(  # type: ignore[attr-defined]
        tokenizer=None,
        esm_model=None,
        esm_lr=None,
        heuristic_lr=None,
        binary_lr=None,
        esm_rf=None,
        heuristic_rf=None,
        metadata={"backend": "fake", "loaded": True},
    )
    service._load_error = None  # type: ignore[attr-defined]
    client.app.dependency_overrides[localization_routes.get_protein_localization_service] = lambda: service

    response = client.post(
        "/api/v1/protein-localization/predict",
        json={"sequence": "ACDEFGHIKLMNPQRSTVWY" * 15},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["localization"] == "Nucleus"
    assert body["evidence_passed"] is True
    assert body["wetlab_prioritization_score"] == 70.4


def test_localization_endpoint_rejects_short_sequence(client: TestClient) -> None:
    response = client.post(
        "/api/v1/protein-localization/predict",
        json={"sequence": "ACDEFGHIKLMNPQRSTV"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "request_validation_error"
    assert "at least 20" in str(payload["error"]["details"]).lower()