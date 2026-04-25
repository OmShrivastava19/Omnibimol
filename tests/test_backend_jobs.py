from datetime import UTC, datetime, timedelta

import jwt
import pytest
from backend.auth.streamlit_integration import build_job_status_url
from backend.api.routes.jobs import AnnotatedVariantPayload, DrugRepurposingJobPayload, GenomeAnalysisJobPayload
from backend.auth.token_verifier import Auth0TokenVerifier
from backend.core.config import get_settings
from backend.db.base import Base
from backend.db.models import Membership, Role, Tenant, User
from backend.db.session import get_db
from backend.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

TEST_SECRET = "this-is-a-very-long-phase-8-job-test-secret-key"


def _token_payload(*, sub: str, tenant_slug: str) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "sub": sub,
        "email": f"{sub.replace('|', '_')}@example.com",
        "name": "Job User",
        "tenant_slug": tenant_slug,
        "iss": "https://example.auth0.com/",
        "aud": "https://api.omnibimol.local",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }


@pytest.fixture()
def auth_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, sessionmaker]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH0_DOMAIN", "example.auth0.com")
    monkeypatch.setenv("AUTH0_AUDIENCE", "https://api.omnibimol.local")
    monkeypatch.setenv("AUTH_JWT_ALGORITHMS", '["HS256"]')
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


def test_enqueue_and_status_transitions(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token = jwt.encode(
        _token_payload(sub="auth0|job-user", tenant_slug="jobs-tenant"),
        TEST_SECRET,
        algorithm="HS256",
    )
    _sync(client, token)
    _set_role(
        session_local,
        tenant_slug="jobs-tenant",
        subject="auth0|job-user",
        role_name="scientist",
    )

    enqueue = client.post(
        "/api/v1/jobs",
        headers={"Authorization": f"Bearer {token}"},
        json={"job_type": "report_generation", "payload": {"items": [1, 2, 3]}},
    )
    assert enqueue.status_code == 200
    job_id = enqueue.json()["id"]
    assert enqueue.json()["status"] == "queued"

    first_poll = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
    second_poll = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
    assert first_poll.status_code == 200
    assert second_poll.status_code == 200
    assert first_poll.json()["status"] in {"running", "completed"}
    assert second_poll.json()["status"] == "completed"


def test_job_failure_transition(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token = jwt.encode(
        _token_payload(sub="auth0|job-fail", tenant_slug="jobs-tenant-fail"),
        TEST_SECRET,
        algorithm="HS256",
    )
    _sync(client, token)
    _set_role(
        session_local,
        tenant_slug="jobs-tenant-fail",
        subject="auth0|job-fail",
        role_name="scientist",
    )

    enqueue = client.post(
        "/api/v1/jobs",
        headers={"Authorization": f"Bearer {token}"},
        json={"job_type": "fail_demo", "payload": {}},
    )
    assert enqueue.status_code == 200
    job_id = enqueue.json()["id"]

    client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
    final_poll = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
    assert final_poll.status_code == 200
    assert final_poll.json()["status"] == "failed"
    assert "Simulated worker failure" in final_poll.json()["error_message"]


def test_docking_job_remains_queued_until_worker_processes_it(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token = jwt.encode(
        _token_payload(sub="auth0|dock-user", tenant_slug="jobs-tenant-dock"),
        TEST_SECRET,
        algorithm="HS256",
    )
    _sync(client, token)
    _set_role(
        session_local,
        tenant_slug="jobs-tenant-dock",
        subject="auth0|dock-user",
        role_name="scientist",
    )

    enqueue = client.post(
        "/api/v1/jobs",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "job_type": "docking.vina",
            "payload": {
                "protein": {"structure_id": "AF-P12345-F1", "pdb_url": "https://example.org/receptor.pdb"},
                "ligand": {"name": "Ligand", "smiles": "CC"},
                "parameters": {"exhaustiveness": 4, "num_modes": 3, "energy_range": 2},
            },
        },
    )
    assert enqueue.status_code == 200
    job_id = enqueue.json()["id"]

    poll = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
    assert poll.status_code == 200
    assert poll.json()["status"] == "queued"


def test_job_status_tenant_isolated(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token_a = jwt.encode(
        _token_payload(sub="auth0|job-a", tenant_slug="jobs-a"),
        TEST_SECRET,
        algorithm="HS256",
    )
    token_b = jwt.encode(
        _token_payload(sub="auth0|job-b", tenant_slug="jobs-b"),
        TEST_SECRET,
        algorithm="HS256",
    )
    _sync(client, token_a)
    _sync(client, token_b)
    _set_role(session_local, tenant_slug="jobs-a", subject="auth0|job-a", role_name="scientist")
    _set_role(session_local, tenant_slug="jobs-b", subject="auth0|job-b", role_name="scientist")

    enqueue = client.post(
        "/api/v1/jobs",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"job_type": "report_generation", "payload": {"items": [42]}},
    )
    assert enqueue.status_code == 200
    job_id = enqueue.json()["id"]

    foreign_status = client.get(
        f"/api/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert foreign_status.status_code == 404


def test_streamlit_job_polling_url_builder() -> None:
    url = build_job_status_url("https://api.omnibimol.local", 123)
    assert url == "https://api.omnibimol.local/api/v1/jobs/123"


def test_genome_analysis_payload_accepts_vcf_and_annotated_variants() -> None:
    payload = GenomeAnalysisJobPayload(
        vcf_text="##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n1\t123\t.\tA\tT\t99\tPASS\tANN=missense_variant|MODERATE|TP53|TP53|transcript|NM_000546|protein_coding|1|c.123A>T|p.Lys41Asn",
        annotated_variants=[
            AnnotatedVariantPayload(
                chrom="1",
                pos=123,
                ref="A",
                alt="T",
                gene="TP53",
                consequence="missense_variant",
                pathogenicity_score=0.91,
                pathogenicity_tier=1,
                pathogenicity_method="xgb_precomputed",
                model_confidence="High",
                evidence_summary="stub",
            )
        ],
    )

    assert payload.vcf_text is not None
    assert payload.annotated_variants[0].pathogenicity_score == 0.91


def test_genome_analysis_payload_requires_a_variant_source() -> None:
    with pytest.raises(ValueError):
        GenomeAnalysisJobPayload()


def test_drug_repurposing_payload_accepts_variant_context() -> None:
    payload = DrugRepurposingJobPayload(
        target_sequence="ACDEFGHIKLMNPQRSTVWY",
        annotated_variants=[
            AnnotatedVariantPayload(
                gene="TP53",
                consequence="missense_variant",
                pathogenicity_score=0.88,
                pathogenicity_tier=1,
            )
        ],
    )

    assert payload.annotated_variants[0].pathogenicity_tier == 1
