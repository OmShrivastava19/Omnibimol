"""Health and readiness endpoints for platform checks."""

from datetime import UTC, datetime

from fastapi import APIRouter

from backend.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness endpoint used by probes and load balancers."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz() -> dict[str, object]:
    """Readiness endpoint with lightweight dependency metadata."""
    settings = get_settings()
    return {
        "status": "ready",
        "service": settings.app_name,
        "environment": settings.environment,
        "database_configured": bool(settings.database_url),
        "redis_configured": bool(settings.redis_url),
        "docking_enabled": settings.docking_enabled,
        "docking_engine": settings.docking_engine,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
