"""Audit browsing endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.audit.service import AuditService
from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import get_tenant_context, require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.db.session import get_db

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/events")
def list_audit_events(
    limit: int = 50,
    _: AuthPrincipal = Depends(require_permission("portfolio.approve")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    events = AuditService(db).list_events_for_tenant(
        tenant_id=tenant_context.tenant_id,
        limit=limit,
    )
    return {
        "events": [
            {
                "id": event.id,
                "tenant_id": event.tenant_id,
                "actor_id": event.actor_id,
                "action": event.action,
                "resource_type": event.resource_type,
                "resource_id": event.resource_id,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "details": event.details,
                "created_at": event.created_at.isoformat(),
            }
            for event in events
        ]
    }
