"""Authentication endpoints and identity synchronization."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.audit.service import AuditService
from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import (
    ensure_role_templates,
    get_or_create_user_and_tenant,
    require_permission,
    sync_membership,
)
from backend.auth.token_verifier import AuthPrincipal
from backend.db.models import Membership, Role, User
from backend.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


def _provision_user(db: Session, principal: AuthPrincipal) -> User:
    tenant, user = get_or_create_user_and_tenant(db, principal)
    ensure_role_templates(db, tenant.id)
    sync_membership(db, tenant_id=tenant.id, user_id=user.id, role_name="scientist")

    db.commit()
    db.refresh(user)
    return user


@router.get("/me")
def me(principal: AuthPrincipal = Depends(get_current_principal)) -> dict[str, str]:
    return {
        "sub": principal.sub,
        "email": principal.email,
        "name": principal.name,
        "tenant_slug": principal.tenant_slug,
    }


@router.post("/sync")
def sync_identity(
    request: Request,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    user = _provision_user(db, principal)
    tenant, _ = get_or_create_user_and_tenant(db, principal)
    AuditService(db).log_event(
        tenant_id=tenant.id,
        actor_id=user.id,
        action="auth.sync",
        resource_type="user",
        resource_id=str(user.id),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"auth0_subject": user.auth0_subject},
    )
    db.commit()
    return {
        "status": "synced",
        "user": {
            "id": user.id,
            "auth0_subject": user.auth0_subject,
            "email": user.email,
            "display_name": user.display_name,
            "tenant_id": user.tenant_id,
        },
    }


@router.get("/rbac/can-read-projects")
def can_read_projects(
    principal: AuthPrincipal = Depends(require_permission("project.read")),
) -> dict[str, object]:
    return {"allowed": True, "permission": "project.read", "subject": principal.sub}


@router.get("/rbac/can-approve-portfolio")
def can_approve_portfolio(
    principal: AuthPrincipal = Depends(require_permission("portfolio.approve")),
) -> dict[str, object]:
    return {"allowed": True, "permission": "portfolio.approve", "subject": principal.sub}


@router.post("/rbac/assign-role")
def assign_role(
    request: Request,
    role_name: str,
    _: AuthPrincipal = Depends(require_permission("portfolio.approve")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant, user = get_or_create_user_and_tenant(db, principal)
    templates = ensure_role_templates(db, tenant.id)
    if role_name not in templates:
        return {"status": "error", "message": f"Unknown role: {role_name}"}

    sync_membership(db, tenant_id=tenant.id, user_id=user.id, role_name=role_name)
    AuditService(db).log_event(
        tenant_id=tenant.id,
        actor_id=user.id,
        action="rbac.role_assigned",
        resource_type="membership",
        resource_id=f"{tenant.id}:{user.id}:{role_name}",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"assigned_role": role_name, "subject": principal.sub},
    )
    db.commit()

    role_rows = list(
        db.scalars(
            select(Role)
            .join(Membership, Membership.role_id == Role.id)
            .where(
                Role.tenant_id == tenant.id,
                Membership.user_id == user.id,
                Membership.tenant_id == tenant.id,
            )
        )
    )
    role_names = [role.name for role in role_rows]
    return {"status": "ok", "assigned_role": role_name, "effective_roles": sorted(set(role_names))}
