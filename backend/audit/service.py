"""Immutable audit event writer and queries."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import AuditEvent


class AuditService:
    """Persist and query tenant-scoped audit events."""

    def __init__(self, db: Session):
        self.db = db

    def log_event(
        self,
        *,
        tenant_id: int,
        actor_id: int | None,
        action: str,
        resource_type: str,
        resource_id: str,
        ip_address: str | None,
        user_agent: str | None,
        details: dict,
        commit: bool = False,
    ) -> AuditEvent:
        event = AuditEvent(
            tenant_id=tenant_id,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
        )
        self.db.add(event)
        self.db.flush()
        if commit:
            self.db.commit()
        return event

    def list_events_for_tenant(self, *, tenant_id: int, limit: int = 100) -> list[AuditEvent]:
        capped_limit = max(1, min(limit, 500))
        stmt = (
            select(AuditEvent)
            .where(AuditEvent.tenant_id == tenant_id)
            .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
            .limit(capped_limit)
        )
        return list(self.db.scalars(stmt))
