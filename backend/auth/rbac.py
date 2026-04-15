"""Role-based access control helpers."""

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_principal
from backend.auth.token_verifier import AuthPrincipal
from backend.db.models import Membership, Permission, Role, Tenant, User
from backend.db.session import get_db

ROLE_PERMISSION_MAP: dict[str, set[str]] = {
    "owner": {"project.read", "project.write", "portfolio.approve", "export.download"},
    "admin": {"project.read", "project.write", "portfolio.approve", "export.download"},
    "scientist": {"project.read", "project.write"},
    "viewer": {"project.read"},
}


@dataclass(frozen=True)
class TenantContext:
    tenant_id: int
    user_id: int
    permissions: set[str]
    principal: AuthPrincipal


def ensure_default_permissions(db: Session) -> None:
    for permission_code in sorted(
        {perm for permission_set in ROLE_PERMISSION_MAP.values() for perm in permission_set}
    ):
        permission = db.scalar(select(Permission).where(Permission.code == permission_code))
        if permission is None:
            db.add(
                Permission(
                    code=permission_code,
                    description=f"Permission for {permission_code}",
                )
            )
    db.flush()


def ensure_role_templates(db: Session, tenant_id: int) -> dict[str, Role]:
    ensure_default_permissions(db)
    permissions_by_code = {
        permission.code: permission for permission in db.scalars(select(Permission)).all()
    }

    roles: dict[str, Role] = {}
    for role_name, permission_codes in ROLE_PERMISSION_MAP.items():
        role = db.scalar(select(Role).where(Role.tenant_id == tenant_id, Role.name == role_name))
        if role is None:
            role = Role(
                tenant_id=tenant_id,
                name=role_name,
                description=f"Default {role_name} role",
            )
            db.add(role)
            db.flush()

        role.permissions = [permissions_by_code[code] for code in sorted(permission_codes)]
        roles[role_name] = role

    db.flush()
    return roles


def get_or_create_user_and_tenant(db: Session, principal: AuthPrincipal) -> tuple[Tenant, User]:
    tenant = db.scalar(select(Tenant).where(Tenant.slug == principal.tenant_slug))
    if tenant is None:
        tenant = Tenant(
            slug=principal.tenant_slug,
            name=principal.tenant_slug.replace("-", " ").title(),
        )
        db.add(tenant)
        db.flush()

    user = db.scalar(select(User).where(User.auth0_subject == principal.sub))
    if user is None:
        user = User(
            tenant_id=tenant.id,
            auth0_subject=principal.sub,
            email=principal.email or f"{principal.sub}@unknown.local",
            display_name=principal.name or principal.sub,
        )
        db.add(user)
        db.flush()
    else:
        user.email = principal.email or user.email
        user.display_name = principal.name or user.display_name
        if user.tenant_id != tenant.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Principal tenant mismatch",
            )

    return tenant, user


def sync_membership(
    db: Session,
    *,
    tenant_id: int,
    user_id: int,
    role_name: str,
) -> Membership:
    role = db.scalar(select(Role).where(Role.tenant_id == tenant_id, Role.name == role_name))
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Role {role_name} not found",
        )
    membership = db.scalar(
        select(Membership).where(
            Membership.tenant_id == tenant_id,
            Membership.user_id == user_id,
            Membership.role_id == role.id,
        )
    )
    if membership is None:
        membership = Membership(tenant_id=tenant_id, user_id=user_id, role_id=role.id)
        db.add(membership)
        db.flush()
    return membership


def get_user_permissions(db: Session, principal: AuthPrincipal) -> set[str]:
    tenant, user = get_or_create_user_and_tenant(db, principal)
    roles = ensure_role_templates(db, tenant.id)

    memberships = list(
        db.scalars(
            select(Membership)
            .where(Membership.tenant_id == tenant.id, Membership.user_id == user.id)
            .join(Role, Membership.role_id == Role.id)
        )
    )

    if not memberships:
        sync_membership(db, tenant_id=tenant.id, user_id=user.id, role_name="scientist")
        memberships = list(
            db.scalars(
                select(Membership)
                .where(Membership.tenant_id == tenant.id, Membership.user_id == user.id)
                .join(Role, Membership.role_id == Role.id)
            )
        )

    role_ids = [membership.role_id for membership in memberships]
    role_rows = list(db.scalars(select(Role).where(Role.id.in_(role_ids))))

    permissions: set[str] = set()
    for role in role_rows:
        if role.name in roles:
            for permission in role.permissions:
                permissions.add(permission.code)
    return permissions


def get_tenant_context(db: Session, principal: AuthPrincipal) -> TenantContext:
    tenant, user = get_or_create_user_and_tenant(db, principal)
    permissions = get_user_permissions(db, principal)
    return TenantContext(
        tenant_id=tenant.id,
        user_id=user.id,
        permissions=permissions,
        principal=principal,
    )


def require_permission(permission_code: str):
    def dependency(
        principal: AuthPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> AuthPrincipal:
        permissions = get_user_permissions(db, principal)
        if permission_code not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission_code}",
            )
        db.commit()
        return principal

    return dependency
