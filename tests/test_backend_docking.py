from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from backend.auth.token_verifier import Auth0TokenVerifier
from backend.core.config import Settings, get_settings
from backend.db.base import Base
from backend.db.job_repository import JobRepository
from backend.db.models import Membership, Role, Tenant, User
from backend.db.session import get_db
from backend.main import create_app
from backend.services.docking import DockingCache
from backend.workers.docking_worker import DockingWorker
from api_client import ProteinAPIClient
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

TEST_SECRET = "this-is-a-very-long-phase-9-docking-test-secret-key"


def _token_payload(*, sub: str, tenant_slug: str) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "sub": sub,
        "email": f"{sub.replace('|', '_')}@example.com",
        "name": "Docking User",
        "tenant_slug": tenant_slug,
        "iss": "https://example.auth0.com/",
        "aud": "https://api.omnibimol.local",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }


@pytest.fixture()
def auth_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple[TestClient, sessionmaker]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH0_DOMAIN", "example.auth0.com")
    monkeypatch.setenv("AUTH0_AUDIENCE", "https://api.omnibimol.local")
    monkeypatch.setenv("AUTH_JWT_ALGORITHMS", '["HS256"]')
    monkeypatch.setenv("DOCKING_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    monkeypatch.setattr(Auth0TokenVerifier, "_resolve_signing_key", lambda *_: TEST_SECRET)

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
    return TestClient(app), session_local


def _sync(client: TestClient, token: str) -> None:
    response = client.post("/api/v1/auth/sync", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def _set_role(
    session_local: sessionmaker,
    *,
    tenant_slug: str,
    subject: str,
    role_name: str,
) -> None:
    db = session_local()
    try:
        tenant = db.scalar(select(Tenant).where(Tenant.slug == tenant_slug))
        user = db.scalar(select(User).where(User.auth0_subject == subject))
        role = db.scalar(select(Role).where(Role.tenant_id == tenant.id, Role.name == role_name))
        memberships = list(
            db.scalars(
                select(Membership).where(
                    Membership.tenant_id == tenant.id,
                    Membership.user_id == user.id,
                )
            )
        )
        for membership in memberships:
            db.delete(membership)
        db.flush()
        db.add(Membership(tenant_id=tenant.id, user_id=user.id, role_id=role.id))
        db.commit()
    finally:
        db.close()


def test_receptor_cache_hits_after_initial_write(tmp_path) -> None:
    cache = DockingCache(str(tmp_path / "receptor-cache"))
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
ATOM      2  CA  GLY A   1      12.560  13.107  14.497  1.00 20.00           C
END
"""

    first = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )
    second = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.receptor_pdbqt_path.exists()
    assert second.receptor_pdbqt_path.exists()


def test_worker_completes_claimed_docking_job(monkeypatch: pytest.MonkeyPatch, auth_client: tuple[TestClient, sessionmaker], tmp_path) -> None:
    _client, session_local = auth_client
    monkeypatch.setattr("backend.workers.docking_worker.get_session_local", lambda: session_local)
    monkeypatch.setattr(
        "backend.workers.docking_worker.get_settings",
        lambda: Settings(
            docking_cache_dir=str(tmp_path / "worker-cache"),
            docking_timeout_seconds=30,
            docking_worker_concurrency=1,
        ),
    )
    monkeypatch.setattr(
        "backend.services.docking.DockingProcessor.process_job_payload",
        lambda self, payload: {
            "available": True,
            "mode": "real",
            "simulated": False,
            "engine": "vina",
            "status": "completed",
            "binding_affinity": -8.4,
            "modes": [{"mode": 1, "affinity": -8.4, "center": {"x": 1.0, "y": 2.0, "z": 3.0}}],
            "best_mode": {"mode": 1, "affinity": -8.4, "center": {"x": 1.0, "y": 2.0, "z": 3.0}},
            "has_coordinates": True,
            "receptor_cache_hit": True,
            "receptor_cache_key": "abc123",
        },
    )

    db = session_local()
    try:
        tenant = Tenant(name="Dock Tenant", slug="dock-tenant")
        db.add(tenant)
        db.flush()
        user = User(
            tenant_id=tenant.id,
            auth0_subject="auth0|dock-user",
            email="dock@example.com",
            display_name="Dock User",
        )
        role = Role(tenant_id=tenant.id, name="scientist", description="Scientist")
        db.add_all([user, role])
        db.flush()
        db.add(Membership(tenant_id=tenant.id, user_id=user.id, role_id=role.id))
        db.commit()

        repo = JobRepository(db)
        job = repo.create_job(
            tenant_id=tenant.id,
            requested_by_user_id=user.id,
            job_type="docking.vina",
            input_payload={
                "protein": {"structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb"},
                "ligand": {"name": "Ligand", "smiles": "CC"},
                "parameters": {"exhaustiveness": 4, "num_modes": 3, "energy_range": 2},
            },
        )
    finally:
        db.close()

    worker = DockingWorker()
    assert worker.run_once() is True

    db = session_local()
    try:
        updated_job = db.get(type(job), job.id)
        assert updated_job.status == "completed"
        assert updated_job.result_payload["binding_affinity"] == -8.4
        assert updated_job.result_payload["available"] is True
    finally:
        db.close()


class _FakeCache:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value):
        self.values[key] = value


def test_real_mode_falls_back_to_simulation_when_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "true")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "real")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:9")

    monkeypatch.setattr(
        ProteinAPIClient,
        "submit_real_docking_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend unavailable")),
    )

    result = client.run_docking_workflow(
        protein_prep={"available": True, "structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb", "sequence_length": 100},
        ligand_data={"name": "Ligand", "smiles": "CC", "molecular_weight": 44.0},
        ligand_name="Ligand",
        protein_length=100,
        ligand_mw=44.0,
        mode="real",
    )

    assert result["simulated"] is True
    assert result["mode"] == "simulation"
    assert result["available"] is True
    assert "fallback_reason" in result
    assert "binding_affinity" in result


def test_docking_result_normalization_keeps_schema_compatible() -> None:
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:9")
    payload = client.normalize_docking_result(
        {
            "available": True,
            "mode": "real",
            "simulated": False,
            "binding_affinity": -8.1,
            "modes": [{"mode": 1, "affinity": -8.1, "center": {"x": 1.0, "y": 2.0, "z": 3.0}}],
            "best_mode": {"mode": 1, "affinity": -8.1, "center": {"x": 1.0, "y": 2.0, "z": 3.0}},
            "has_coordinates": True,
        }
    )

    assert payload["available"] is True
    assert payload["mode"] == "real"
    assert payload["binding_affinity"] == -8.1
    assert payload["modes"][0]["center"]["x"] == 1.0