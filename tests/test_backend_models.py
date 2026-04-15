from collections.abc import Generator

import pytest
from backend.db.base import Base
from backend.db.models import Membership, Role, Tenant, User
from backend.db.repositories import ProjectRepository
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_unique_role_per_tenant_constraint(db_session: Session) -> None:
    tenant = Tenant(name="Tenant A", slug="tenant-a")
    db_session.add(tenant)
    db_session.flush()

    db_session.add(Role(tenant_id=tenant.id, name="admin", description="admin role"))
    db_session.commit()

    db_session.add(Role(tenant_id=tenant.id, name="admin", description="duplicate role"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_membership_uniqueness_constraint(db_session: Session) -> None:
    tenant = Tenant(name="Tenant B", slug="tenant-b")
    db_session.add(tenant)
    db_session.flush()

    user = User(
        tenant_id=tenant.id,
        auth0_subject="auth0|user-b",
        email="user-b@example.com",
        display_name="User B",
    )
    role = Role(tenant_id=tenant.id, name="scientist", description="Scientist role")
    db_session.add_all([user, role])
    db_session.flush()

    db_session.add(Membership(tenant_id=tenant.id, user_id=user.id, role_id=role.id))
    db_session.commit()

    db_session.add(Membership(tenant_id=tenant.id, user_id=user.id, role_id=role.id))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_project_repository_enforces_tenant_scoping(db_session: Session) -> None:
    tenant_a = Tenant(name="Tenant C", slug="tenant-c")
    tenant_b = Tenant(name="Tenant D", slug="tenant-d")
    db_session.add_all([tenant_a, tenant_b])
    db_session.flush()

    user_a = User(
        tenant_id=tenant_a.id,
        auth0_subject="auth0|tenant-c-admin",
        email="admin-c@example.com",
        display_name="Admin C",
    )
    user_b = User(
        tenant_id=tenant_b.id,
        auth0_subject="auth0|tenant-d-admin",
        email="admin-d@example.com",
        display_name="Admin D",
    )
    db_session.add_all([user_a, user_b])
    db_session.commit()

    repo = ProjectRepository(db_session)
    repo.create_project(tenant_id=tenant_a.id, name="Program A1", created_by_user_id=user_a.id)
    repo.create_project(tenant_id=tenant_b.id, name="Program B1", created_by_user_id=user_b.id)

    tenant_a_projects = repo.list_projects_for_tenant(tenant_a.id)
    tenant_b_projects = repo.list_projects_for_tenant(tenant_b.id)

    assert len(tenant_a_projects) == 1
    assert tenant_a_projects[0].name == "Program A1"
    assert all(project.tenant_id == tenant_a.id for project in tenant_a_projects)

    assert len(tenant_b_projects) == 1
    assert tenant_b_projects[0].name == "Program B1"
    assert all(project.tenant_id == tenant_b.id for project in tenant_b_projects)
