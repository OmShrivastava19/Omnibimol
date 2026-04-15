"""Seed script for local development tenant, roles, and admin user."""

from backend.db.base import Base
from backend.db.models import Membership, Permission, Role, Tenant, User
from backend.db.session import get_engine, get_session_local
from sqlalchemy import select

DEFAULT_PERMISSIONS = [
    ("project.read", "Read project records"),
    ("project.write", "Create and update project records"),
    ("portfolio.approve", "Approve portfolio recommendations"),
    ("export.download", "Download generated exports"),
]


def run_seed() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    db = get_session_local()()
    try:
        tenant = db.scalar(select(Tenant).where(Tenant.slug == "dev-tenant"))
        if tenant is None:
            tenant = Tenant(name="Development Tenant", slug="dev-tenant")
            db.add(tenant)
            db.flush()

        permission_map: dict[str, Permission] = {}
        for code, description in DEFAULT_PERMISSIONS:
            permission = db.scalar(select(Permission).where(Permission.code == code))
            if permission is None:
                permission = Permission(code=code, description=description)
                db.add(permission)
                db.flush()
            permission_map[code] = permission

        admin_role = db.scalar(
            select(Role).where(Role.tenant_id == tenant.id, Role.name == "admin")
        )
        if admin_role is None:
            admin_role = Role(
                tenant_id=tenant.id,
                name="admin",
                description="Tenant admin with elevated privileges",
                permissions=list(permission_map.values()),
            )
            db.add(admin_role)
            db.flush()

        admin_user = db.scalar(select(User).where(User.auth0_subject == "auth0|dev-admin"))
        if admin_user is None:
            admin_user = User(
                tenant_id=tenant.id,
                auth0_subject="auth0|dev-admin",
                email="admin@omnibimol.local",
                display_name="Development Admin",
            )
            db.add(admin_user)
            db.flush()

        existing_membership = db.scalar(
            select(Membership).where(
                Membership.tenant_id == tenant.id,
                Membership.user_id == admin_user.id,
                Membership.role_id == admin_role.id,
            )
        )
        if existing_membership is None:
            db.add(
                Membership(
                    tenant_id=tenant.id,
                    user_id=admin_user.id,
                    role_id=admin_role.id,
                )
            )

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
