from __future__ import annotations

import pytest
from backend.api.routes import chemprot as chemprot_routes
from backend.db.base import Base
from backend.db.session import get_db
from backend.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.services.chemprot import ChemProtInteractionService
from target_prioritization_engine import TargetPrioritizationEngine


class ProteinAwareBackend:
    def predict_lora_proba(self, records):
        values = []
        for record in records:
            protein = str(record.get("protein", ""))
            values.append(0.92 if protein == "ABL1" else 0.15)
        return values

    def predict_ensemble_proba(self, records):
        values = []
        for record in records:
            protein = str(record.get("protein", ""))
            values.append(0.97 if protein == "ABL1" else 0.10)
        return values

    def metadata(self):
        return {"backend": "fake", "loaded": True, "ensemble_loaded": True}

    def health_snapshot(self):
        return {"backend": "fake", "loaded": True, "degraded": False}


class NoEnsembleBackend(ProteinAwareBackend):
    def predict_ensemble_proba(self, records):
        return None

    def metadata(self):
        return {"backend": "fake", "loaded": True, "ensemble_loaded": False}

    def health_snapshot(self):
        return {"backend": "fake", "loaded": True, "degraded": True}


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

    def override_get_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_chemprot_scoring_endpoint_ranks_targets(client: TestClient) -> None:
    service = ChemProtInteractionService(backend_factory=lambda _settings: ProteinAwareBackend())
    app = client.app
    app.dependency_overrides[chemprot_routes.get_chemprot_service] = lambda: service

    response = client.post(
        "/api/v1/chemprot/score",
        json={
            "chemical": "imatinib",
            "candidate_proteins": ["ABL1", "EGFR"],
            "abstracts": [
                {
                    "pmid": "11111111",
                    "title": "Imatinib and ABL1",
                    "abstract": "Imatinib interacts with ABL1 in chronic myeloid leukemia.",
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ranked_targets"][0]["protein"] == "ABL1"
    assert body["ranked_targets"][0]["reranker_used"] is True
    assert body["ranked_targets"][0]["evidence"]
    assert body["ranked_targets"][0]["model_metadata"]["cpu_only"] is True


def test_chemprot_scoring_endpoint_handles_missing_ensemble(client: TestClient) -> None:
    service = ChemProtInteractionService(backend_factory=lambda _settings: NoEnsembleBackend())
    app = client.app
    app.dependency_overrides[chemprot_routes.get_chemprot_service] = lambda: service

    response = client.post(
        "/api/v1/chemprot/score",
        json={
            "chemical": "imatinib",
            "candidate_proteins": ["ABL1"],
            "abstracts": [
                {
                    "pmid": "11111111",
                    "title": "Imatinib and ABL1",
                    "abstract": "Imatinib interacts with ABL1 in chronic myeloid leukemia.",
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reranker_used"] is False
    assert body["degraded_mode"] is True
    assert body["ranked_targets"][0]["final_score"] == body["ranked_targets"][0]["interaction_probability"]


def test_target_prioritization_chemprot_toggle_preserves_default_behavior() -> None:
    payload = {
        "target_id": "ABL1",
        "expression_data": {
            "tissues": [{"tissue": "Blood", "level_numeric": 3}],
            "disease_tissues": [],
        },
        "pathway_data": {"available": False},
        "ppi_data": {"available": False},
        "genetic_data": {},
        "ligandability_data": {},
        "trial_data": {},
    }

    baseline_engine = TargetPrioritizationEngine()
    baseline = baseline_engine.rank_targets([payload])[0]

    enabled_engine = TargetPrioritizationEngine(chemprot_enabled=True)
    enriched_payload = dict(payload)
    enriched_payload["interaction_data"] = {
        "final_score": 0.94,
        "interaction_probability": 0.92,
        "reranker_used": True,
        "evidence": [{"pmid": "1"}],
        "ranked_targets": [{"protein": "ABL1"}],
    }
    enriched = enabled_engine.rank_targets([enriched_payload])[0]

    assert baseline["composite_score"] == enabled_engine.rank_targets([payload])[0]["composite_score"]
    assert enriched["composite_score"] > baseline["composite_score"]
    assert any(item["component"] == "interaction" for item in enriched["explainability"]["breakdown"])
