from datetime import UTC, datetime, timedelta

import jwt
import pytest
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

TEST_SECRET = "this-is-a-very-long-phase-5-tenant-isolation-test-secret-key"


def _token_payload(*, sub: str, tenant_slug: str) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "sub": sub,
        "email": f"{sub.replace('|', '_')}@example.com",
        "name": "Tenant User",
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
    auth0_subject: str,
    role_name: str,
) -> None:
    db = session_local()
    try:
        tenant = db.scalar(select(Tenant).where(Tenant.slug == tenant_slug))
        user = db.scalar(select(User).where(User.auth0_subject == auth0_subject))
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


def test_cross_tenant_read_is_denied(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, _session_local = auth_client
    token_a = jwt.encode(
        _token_payload(sub="auth0|tenant-a-user", tenant_slug="tenant-a"),
        TEST_SECRET,
        algorithm="HS256",
    )
    token_b = jwt.encode(
        _token_payload(sub="auth0|tenant-b-user", tenant_slug="tenant-b"),
        TEST_SECRET,
        algorithm="HS256",
    )

    _sync(client, token_a)
    _sync(client, token_b)

    create_resp = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"name": "Tenant A Program", "description": "A only"},
    )
    assert create_resp.status_code == 200
    project_id = create_resp.json()["id"]

    foreign_read = client.get(
        f"/api/v1/projects/{project_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert foreign_read.status_code == 404


def test_cross_tenant_write_is_denied(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token_a = jwt.encode(
        _token_payload(sub="auth0|tenant-a-admin", tenant_slug="tenant-a"),
        TEST_SECRET,
        algorithm="HS256",
    )
    token_b = jwt.encode(
        _token_payload(sub="auth0|tenant-b-admin", tenant_slug="tenant-b"),
        TEST_SECRET,
        algorithm="HS256",
    )

    _sync(client, token_a)
    _sync(client, token_b)

    _set_role(
        session_local,
        tenant_slug="tenant-a",
        auth0_subject="auth0|tenant-a-admin",
        role_name="admin",
    )
    _set_role(
        session_local,
        tenant_slug="tenant-b",
        auth0_subject="auth0|tenant-b-admin",
        role_name="admin",
    )

    create_resp = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"name": "Tenant A Restricted", "description": "A write"},
    )
    assert create_resp.status_code == 200
    project_id = create_resp.json()["id"]

    foreign_write = client.patch(
        f"/api/v1/projects/{project_id}",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"description": "malicious overwrite"},
    )
    assert foreign_write.status_code == 404


def test_cross_tenant_export_is_denied(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token_a = jwt.encode(
        _token_payload(sub="auth0|tenant-a-owner", tenant_slug="tenant-a"),
        TEST_SECRET,
        algorithm="HS256",
    )
    token_b = jwt.encode(
        _token_payload(sub="auth0|tenant-b-owner", tenant_slug="tenant-b"),
        TEST_SECRET,
        algorithm="HS256",
    )

    _sync(client, token_a)
    _sync(client, token_b)

    _set_role(
        session_local,
        tenant_slug="tenant-a",
        auth0_subject="auth0|tenant-a-owner",
        role_name="owner",
    )
    _set_role(
        session_local,
        tenant_slug="tenant-b",
        auth0_subject="auth0|tenant-b-owner",
        role_name="owner",
    )

    create_resp = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"name": "Tenant A Export", "description": "A export"},
    )
    assert create_resp.status_code == 200
    project_id = create_resp.json()["id"]

    foreign_export = client.get(
        f"/api/v1/projects/{project_id}/export",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert foreign_export.status_code == 404
