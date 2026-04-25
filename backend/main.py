"""FastAPI backend entrypoint for OmniBioMol SaaS services."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from backend.api.routes.audit import router as audit_router
from backend.api.routes.academic_models import router as academic_models_router
from backend.api.routes.auth import router as auth_router
from backend.api.routes.health import router as health_router
from backend.api.routes.jobs import router as jobs_router
from backend.api.routes.multiomics import router as multiomics_router
from backend.api.routes.projects import router as projects_router
from backend.api.routes.reliability import router as reliability_router
from backend.core.config import get_settings
from backend.core.errors import register_exception_handlers
from backend.core.logging import configure_logging, request_id_ctx_var

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Run startup checks that are safe for phase-1 scaffolding."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be configured")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL must be configured")

    logger.info("Backend startup checks passed")
    yield
    logger.info("Backend shutdown complete")


def create_app() -> FastAPI:
    """App factory for uvicorn and tests."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    register_exception_handlers(app)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request_id_ctx_var.set(request_id)
        response: Response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/")
    def root() -> dict[str, str]:
        """Root endpoint for platform probes and service discovery."""
        return {
            "service": settings.app_name,
            "status": "ok",
            "health": f"{settings.api_prefix}/healthz",
            "ready": f"{settings.api_prefix}/readyz",
            "academic_models": f"{settings.api_prefix}/academic-models/models",
        }

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(projects_router, prefix=settings.api_prefix)
    app.include_router(jobs_router, prefix=settings.api_prefix)
    app.include_router(multiomics_router, prefix=settings.api_prefix)
    app.include_router(academic_models_router, prefix=settings.api_prefix)
    app.include_router(audit_router, prefix=settings.api_prefix)
    app.include_router(reliability_router, prefix=settings.api_prefix)
    return app


app = create_app()
