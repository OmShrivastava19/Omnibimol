from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import jwt
import pytest
from backend.auth.token_verifier import Auth0TokenVerifier
from backend.core.config import Settings, get_settings
from backend.db.base import Base
from backend.db.job_repository import JobRepository
from backend.db.models import Membership, Role, Tenant, User
from backend.db.session import get_db
from backend.main import create_app
from backend.services.docking import (
    RECEPTOR_CONVERTER_VERSION,
    RECEPTOR_PDBQT_FORMAT_VERSION,
    DockingConversionError,
    DockingCache,
    _format_pdbqt_atom_line,
    convert_pdb_to_pdbqt,
    validate_receptor_pdbqt_text,
)
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


def test_convert_pdb_to_pdbqt_produces_parseable_coordinates() -> None:
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
ATOM      2  CA  GLY A   1      12.560  13.107  14.497  1.00 20.00           C
END
"""

    pdbqt_text = convert_pdb_to_pdbqt(pdb_text, structure_id="AF-P12345-F1")
    atom_lines = [line for line in pdbqt_text.splitlines() if line.startswith("ATOM")]

    assert len(atom_lines) == 2
    for line in atom_lines:
        assert len(line) >= 80
        assert line[0:6] == "ATOM  "
        assert line[29] == " "
        assert isinstance(float(line[30:38]), float)
        assert isinstance(float(line[38:46]), float)
        assert isinstance(float(line[46:54]), float)
        assert float(line[68:76]) == 0.0
        assert line[77:79].strip() in {"N", "C", "OA", "SA", "P", "H"}


def test_receptor_pdbqt_validation_rejects_malformed_coordinate_field() -> None:
    malformed = "\n".join(
        [
            "REMARK  RECEPTOR malformed",
            "ATOM      1    N RES A   1     12.685  10.100  11.200  1.00  0.00    0.000 N ",
        ]
    )
    malformed = malformed.replace("12.685", "12A685")

    with pytest.raises(DockingConversionError):
        validate_receptor_pdbqt_text(malformed)


def test_receptor_cache_rejects_stale_format_versions(tmp_path) -> None:
    cache = DockingCache(str(tmp_path / "receptor-cache"))
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
END
"""

    record = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )
    metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = "pdbqt-v2"
    record.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    assert cache.load(record.cache_key) is None


def test_receptor_cache_regenerates_after_format_version_mismatch(tmp_path) -> None:
    cache = DockingCache(str(tmp_path / "receptor-cache"))
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
END
"""

    first = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = "pdbqt-v2"
    first.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    regenerated = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )

    assert regenerated.cache_hit is False
    regenerated_metadata = json.loads(regenerated.metadata_path.read_text(encoding="utf-8"))
    assert regenerated_metadata["format_version"] == RECEPTOR_PDBQT_FORMAT_VERSION


def test_receptor_cache_regenerates_after_converter_version_mismatch(tmp_path) -> None:
    cache = DockingCache(str(tmp_path / "receptor-cache"))
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
END
"""

    first = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    metadata["converter_version"] = "legacy-converter"
    first.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    regenerated = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )

    assert regenerated.cache_hit is False
    regenerated_metadata = json.loads(regenerated.metadata_path.read_text(encoding="utf-8"))
    assert regenerated_metadata["converter_version"] == RECEPTOR_CONVERTER_VERSION


def test_receptor_cache_regenerates_when_cached_pdbqt_is_malformed(tmp_path) -> None:
    cache = DockingCache(str(tmp_path / "receptor-cache"))
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
END
"""

    first = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )

    broken_line = "ATOM      1    N RES A   1     11X104  13.207  14.497  1.00 20.00    0.000 N "
    first.receptor_pdbqt_path.write_text(f"REMARK  RECEPTOR broken\n{broken_line}\n", encoding="utf-8")

    regenerated = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )

    assert regenerated.cache_hit is False
    regenerated_text = regenerated.receptor_pdbqt_path.read_text(encoding="utf-8")
    validate_receptor_pdbqt_text(regenerated_text)


def test_ligand_hetatm_line_uses_fixed_columns() -> None:
    line = _format_pdbqt_atom_line(
        record_name="HETATM",
        atom_index=7,
        atom_name="C",
        res_name="LIG",
        chain_id="A",
        res_seq=1,
        x=12.685,
        y=-3.25,
        z=101.0,
        occupancy=1.0,
        temp_factor=0.0,
        charge=-0.123,
        atom_type="c",
    )

    assert len(line) >= 80
    assert line[0:6] == "HETATM"
    assert line[29] == " "
    assert float(line[30:38]) == pytest.approx(12.685, rel=0, abs=1e-3)
    assert float(line[38:46]) == pytest.approx(-3.25, rel=0, abs=1e-3)
    assert float(line[46:54]) == pytest.approx(101.0, rel=0, abs=1e-3)
    assert float(line[68:76]) == pytest.approx(-0.123, rel=0, abs=1e-3)


def test_generated_receptor_file_is_vina_set_receptor_parseable(tmp_path) -> None:
    cache = DockingCache(str(tmp_path / "receptor-cache"))
    pdb_text = """ATOM      1  N   GLY A   1      11.104  13.207  14.497  1.00 20.00           N
ATOM      2  CA  GLY A   1      12.560  13.107  14.497  1.00 20.00           C
END
"""
    record = cache.get_or_create(
        structure_id="AF-P12345-F1",
        source_url="https://example.org/receptor.pdb",
        pdb_text=pdb_text,
    )

    # Mimic Vina coordinate parsing strictness on canonical fixed columns.
    for line in Path(record.receptor_pdbqt_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        float(line[30:38])
        float(line[38:46])
        float(line[46:54])


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


def _fake_request_backend_json_factory(*, job_status: str | None = None, poll_error: Exception | None = None):
    calls: list[str] = []

    def fake_request(self, method: str, path: str, *, json_body=None, timeout: float = 60.0):
        calls.append(path)
        if path == "/api/v1/readyz":
            return {"status": "ready", "database_configured": True, "redis_configured": True}
        if path == "/api/v1/jobs":
            return {"id": 41, "job_type": "docking.vina", "status": "queued"}
        if path == "/api/v1/jobs/41":
            if poll_error is not None:
                raise poll_error
            payload: dict[str, object] = {"id": 41, "status": job_status or "queued"}
            if job_status == "completed":
                payload["result_payload"] = {
                    "available": True,
                    "mode": "real",
                    "simulated": False,
                    "engine": "vina",
                    "status": "completed",
                    "binding_affinity": -8.4,
                    "modes": [{"mode": 1, "affinity": -8.4, "center": {"x": 1.0, "y": 2.0, "z": 3.0}}],
                    "best_mode": {"mode": 1, "affinity": -8.4, "center": {"x": 1.0, "y": 2.0, "z": 3.0}},
                    "has_coordinates": True,
                }
            elif job_status == "failed":
                payload["error_message"] = "Worker failed to complete the docking job"
                payload["result_payload"] = {"available": False, "mode": "real", "simulated": False, "status": "failed"}
            return payload
        raise AssertionError(f"Unexpected path: {path}")

    return fake_request, calls


def test_real_mode_reports_clean_failure_when_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "true")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "real")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:9")

    monkeypatch.setattr(
        ProteinAPIClient,
        "_request_backend_json",
        lambda self, method, path, *, json_body=None, timeout=60.0: (_ for _ in ()).throw(
            httpx.ConnectError(
                "[WinError 10061] No connection could be made because the target machine actively refused it",
                request=httpx.Request(method, f"{self.backend_api_url}{path}"),
            )
        ) if path == "/api/v1/readyz" else {"id": 41, "job_type": "docking.vina", "status": "queued"},
    )

    result = client.run_docking_workflow(
        protein_prep={"available": True, "structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb", "sequence_length": 100},
        ligand_data={"name": "Ligand", "smiles": "CC", "molecular_weight": 44.0},
        ligand_name="Ligand",
        protein_length=100,
        ligand_mw=44.0,
        mode="real",
    )

    assert result["simulated"] is False
    assert result["mode"] == "real"
    assert result["available"] is False
    assert result["status"] == "failed"
    assert "fallback_reason" in result
    assert result["error_message"] == result["fallback_reason"]
    assert result["binding_affinity"] is None
    assert "WinError" not in result["error_message"]


def test_real_mode_reports_clean_failure_when_poll_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "true")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "real")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:9")

    poll_error = httpx.ConnectError(
        "[WinError 10061] No connection could be made because the target machine actively refused it",
        request=httpx.Request("GET", "http://127.0.0.1:9/api/v1/jobs/41"),
    )
    fake_request, calls = _fake_request_backend_json_factory(poll_error=poll_error)
    monkeypatch.setattr(ProteinAPIClient, "_request_backend_json", fake_request)

    result = client.run_docking_workflow(
        protein_prep={"available": True, "structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb", "sequence_length": 100},
        ligand_data={"name": "Ligand", "smiles": "CC", "molecular_weight": 44.0},
        ligand_name="Ligand",
        protein_length=100,
        ligand_mw=44.0,
        mode="real",
    )

    assert calls == ["/api/v1/readyz", "/api/v1/jobs", "/api/v1/jobs/41"]
    assert result["simulated"] is False
    assert result["mode"] == "real"
    assert result["available"] is False
    assert result["status"] == "failed"
    assert result["job_id"] == 41
    assert result["error_message"] == result["fallback_reason"]
    assert "WinError" not in result["error_message"]


@pytest.mark.parametrize(
    "job_status, expected_status",
    [
        ("queued", "queued"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
    ],
)
def test_real_mode_handles_job_states(monkeypatch: pytest.MonkeyPatch, job_status: str, expected_status: str) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "true")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "real")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:8000")

    fake_request, calls = _fake_request_backend_json_factory(job_status=job_status)
    monkeypatch.setattr(ProteinAPIClient, "_request_backend_json", fake_request)

    result = client.run_docking_workflow(
        protein_prep={"available": True, "structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb", "sequence_length": 100},
        ligand_data={"name": "Ligand", "smiles": "CC", "molecular_weight": 44.0},
        ligand_name="Ligand",
        protein_length=100,
        ligand_mw=44.0,
        mode="real",
    )

    assert calls[0] == "/api/v1/readyz"
    assert calls[1] == "/api/v1/jobs"
    assert calls[2] == "/api/v1/jobs/41"
    assert result["simulated"] is False
    assert result["mode"] == "real"
    assert result["status"] == expected_status
    assert result["job_status"] == expected_status

    if job_status == "completed":
        assert result["available"] is True
        assert result["binding_affinity"] == -8.4
        assert result["modes"][0]["affinity"] == -8.4
    elif job_status in {"queued", "running"}:
        assert result["available"] is False
        assert result["queued_for_worker"] is True
        assert "worker" in result["fallback_reason"].lower()
    else:
        assert result["available"] is False
        assert result["binding_affinity"] is None
        assert result["error_message"] == result["fallback_reason"]


def test_simulation_mode_still_works_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "true")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "simulation")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:9")

    result = client.run_docking_workflow(
        protein_prep={"available": True, "structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb", "sequence_length": 100},
        ligand_data={"name": "Ligand", "smiles": "CC", "molecular_weight": 44.0},
        ligand_name="Ligand",
        protein_length=100,
        ligand_mw=44.0,
        mode="simulation",
    )

    assert result["simulated"] is True
    assert result["mode"] == "simulation"
    assert result["available"] is True
    assert result["status"] == "completed"
    assert result["has_coordinates"] is True


def test_real_mode_disabled_reports_explicit_simulation_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "false")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "real")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:9")

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
    assert "fallback_reason" in result
    assert "DOCKING_ENABLED" in result["fallback_reason"]
    assert result["error_message"] == result["fallback_reason"]


def test_real_mode_failed_job_propagates_result_payload_failure_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKING_ENABLED", "true")
    monkeypatch.setenv("DOCKING_MODE_DEFAULT", "real")
    client = ProteinAPIClient(_FakeCache(), backend_api_url="http://127.0.0.1:8000")

    failure_reason = (
        "PDBQT parsing error before Vina.set_receptor: invalid coordinate field at line 12: '12A685  '"
    )

    def fake_request(self, method: str, path: str, *, json_body=None, timeout: float = 60.0):
        if path == "/api/v1/readyz":
            return {"status": "ready", "database_configured": True, "redis_configured": True}
        if path == "/api/v1/jobs":
            return {"id": 41, "job_type": "docking.vina", "status": "queued"}
        if path == "/api/v1/jobs/41":
            return {
                "id": 41,
                "status": "failed",
                "result_payload": {
                    "available": False,
                    "mode": "real",
                    "simulated": False,
                    "status": "failed",
                    "fallback_reason": failure_reason,
                },
            }
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(ProteinAPIClient, "_request_backend_json", fake_request)

    result = client.run_docking_workflow(
        protein_prep={"available": True, "structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb", "sequence_length": 100},
        ligand_data={"name": "Ligand", "smiles": "CC", "molecular_weight": 44.0},
        ligand_name="Ligand",
        protein_length=100,
        ligand_mw=44.0,
        mode="real",
    )

    assert result["simulated"] is False
    assert result["mode"] == "real"
    assert result["status"] == "failed"
    assert result["fallback_reason"] == failure_reason
    assert result["error_message"] == failure_reason


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