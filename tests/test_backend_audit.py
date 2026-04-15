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

TEST_SECRET = "this-is-a-very-long-phase-6-audit-test-secret-key"


def _token_payload(*, sub: str, tenant_slug: str) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "sub": sub,
        "email": f"{sub.replace('|', '_')}@example.com",
        "name": "Audit User",
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
        existing = list(
            db.scalars(
                select(Membership).where(
                    Membership.tenant_id == tenant.id,
                    Membership.user_id == user.id,
                )
            )
        )
        for membership in existing:
            db.delete(membership)
        db.flush()
        db.add(Membership(tenant_id=tenant.id, user_id=user.id, role_id=role.id))
        db.commit()
    finally:
        db.close()


def test_privileged_actions_emit_audit_events(auth_client: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_client
    token = jwt.encode(
        _token_payload(sub="auth0|audit-admin", tenant_slug="audit-tenant"),
        TEST_SECRET,
        algorithm="HS256",
    )
    _sync(client, token)
    _set_role(
        session_local,
        tenant_slug="audit-tenant",
        subject="auth0|audit-admin",
        role_name="admin",
    )

    created = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "pytest-agent"},
        json={"name": "Audit Project", "description": "seed"},
    )
    assert created.status_code == 200
    project_id = created.json()["id"]

    updated = client.patch(
        f"/api/v1/projects/{project_id}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "pytest-agent"},
        json={"description": "updated"},
    )
    assert updated.status_code == 200

    exported = client.get(
        f"/api/v1/projects/{project_id}/export",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "pytest-agent"},
    )
    assert exported.status_code == 200

    role_assigned = client.post(
        "/api/v1/auth/rbac/assign-role",
        params={"role_name": "owner"},
        headers={"Authorization": f"Bearer {token}", "User-Agent": "pytest-agent"},
    )
    assert role_assigned.status_code == 200

    audit_response = client.get(
        "/api/v1/audit/events",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert audit_response.status_code == 200
    events = audit_response.json()["events"]
    actions = [event["action"] for event in events]

    assert "auth.sync" in actions
    assert "project.created" in actions
    assert "project.updated" in actions
    assert "project.exported" in actions
    assert "rbac.role_assigned" in actions
    assert all(event["tenant_id"] == events[0]["tenant_id"] for event in events)


def test_audit_listing_requires_admin_permission(
    auth_client: tuple[TestClient, sessionmaker],
) -> None:
    client, session_local = auth_client
    token = jwt.encode(
        _token_payload(sub="auth0|audit-viewer", tenant_slug="audit-tenant-2"),
        TEST_SECRET,
        algorithm="HS256",
    )
    _sync(client, token)
    _set_role(
        session_local,
        tenant_slug="audit-tenant-2",
        subject="auth0|audit-viewer",
        role_name="viewer",
    )

    response = client.get("/api/v1/audit/events", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
