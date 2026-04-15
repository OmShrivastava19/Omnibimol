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

TEST_SECRET = "this-is-a-very-long-phase-4-rbac-test-secret-key"


def _token_payload(sub: str = "auth0|rbac-user") -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "sub": sub,
        "email": f"{sub.replace('|', '_')}@example.com",
        "name": "RBAC User",
        "tenant_slug": "rbac-tenant",
        "iss": "https://example.auth0.com/",
        "aud": "https://api.omnibimol.local",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }


def _build_auth_client() -> tuple[TestClient, sessionmaker]:
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


def _sync_user(client: TestClient, token: str) -> None:
    response = client.post("/api/v1/auth/sync", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def _ensure_single_role(session_local: sessionmaker, *, auth0_subject: str, role_name: str) -> None:
    db = session_local()
    try:
        tenant = db.scalar(select(Tenant).where(Tenant.slug == "rbac-tenant"))
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


@pytest.fixture()
def auth_setup(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, sessionmaker]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH0_DOMAIN", "example.auth0.com")
    monkeypatch.setenv("AUTH0_AUDIENCE", "https://api.omnibimol.local")
    monkeypatch.setenv("AUTH_JWT_ALGORITHMS", '["HS256"]')
    get_settings.cache_clear()
    monkeypatch.setattr(Auth0TokenVerifier, "_resolve_signing_key", lambda *_: TEST_SECRET)
    return _build_auth_client()


def test_viewer_can_read_but_cannot_approve(auth_setup: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_setup
    token = jwt.encode(_token_payload("auth0|viewer"), TEST_SECRET, algorithm="HS256")
    _sync_user(client, token)
    _ensure_single_role(session_local, auth0_subject="auth0|viewer", role_name="viewer")

    read_response = client.get(
        "/api/v1/auth/rbac/can-read-projects",
        headers={"Authorization": f"Bearer {token}"},
    )
    approve_response = client.get(
        "/api/v1/auth/rbac/can-approve-portfolio",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert read_response.status_code == 200
    assert approve_response.status_code == 403


def test_scientist_can_write_but_cannot_approve(
    auth_setup: tuple[TestClient, sessionmaker],
) -> None:
    client, session_local = auth_setup
    token = jwt.encode(_token_payload("auth0|scientist"), TEST_SECRET, algorithm="HS256")
    _sync_user(client, token)
    _ensure_single_role(session_local, auth0_subject="auth0|scientist", role_name="scientist")

    read_response = client.get(
        "/api/v1/auth/rbac/can-read-projects",
        headers={"Authorization": f"Bearer {token}"},
    )
    approve_response = client.get(
        "/api/v1/auth/rbac/can-approve-portfolio",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert read_response.status_code == 200
    assert approve_response.status_code == 403


def test_admin_can_approve(auth_setup: tuple[TestClient, sessionmaker]) -> None:
    client, session_local = auth_setup
    token = jwt.encode(_token_payload("auth0|admin"), TEST_SECRET, algorithm="HS256")
    _sync_user(client, token)
    _ensure_single_role(session_local, auth0_subject="auth0|admin", role_name="admin")

    read_response = client.get(
        "/api/v1/auth/rbac/can-read-projects",
        headers={"Authorization": f"Bearer {token}"},
    )
    approve_response = client.get(
        "/api/v1/auth/rbac/can-approve-portfolio",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert read_response.status_code == 200
    assert approve_response.status_code == 200
