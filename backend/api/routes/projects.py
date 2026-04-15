"""Tenant-isolated project endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.audit.service import AuditService
from backend.auth.dependencies import get_current_principal
from backend.auth.rbac import get_tenant_context, require_permission
from backend.auth.token_verifier import AuthPrincipal
from backend.db.repositories import ProjectRepository
from backend.db.session import get_db

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None


@router.post("")
def create_project(
    request: Request,
    body: ProjectCreateRequest,
    _: AuthPrincipal = Depends(require_permission("project.write")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = ProjectRepository(db)
    project = repo.create_project(
        tenant_id=tenant_context.tenant_id,
        name=body.name,
        description=body.description,
        created_by_user_id=tenant_context.user_id,
    )
    AuditService(db).log_event(
        tenant_id=tenant_context.tenant_id,
        actor_id=tenant_context.user_id,
        action="project.created",
        resource_type="project",
        resource_id=str(project.id),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"name": project.name},
        commit=True,
    )
    return {
        "id": project.id,
        "tenant_id": project.tenant_id,
        "name": project.name,
        "description": project.description,
    }


@router.get("")
def list_projects(
    _: AuthPrincipal = Depends(require_permission("project.read")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = ProjectRepository(db)
    projects = repo.list_projects_for_tenant(tenant_context.tenant_id)
    return {
        "projects": [
            {
                "id": project.id,
                "tenant_id": project.tenant_id,
                "name": project.name,
                "description": project.description,
            }
            for project in projects
        ]
    }


@router.get("/{project_id}")
def get_project(
    project_id: int,
    _: AuthPrincipal = Depends(require_permission("project.read")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = ProjectRepository(db)
    project = repo.get_project_for_tenant(
        tenant_id=tenant_context.tenant_id,
        project_id=project_id,
    )
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return {
        "id": project.id,
        "tenant_id": project.tenant_id,
        "name": project.name,
        "description": project.description,
    }


@router.patch("/{project_id}")
def update_project(
    request: Request,
    project_id: int,
    body: ProjectUpdateRequest,
    _: AuthPrincipal = Depends(require_permission("project.write")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = ProjectRepository(db)
    project = repo.update_project_for_tenant(
        tenant_id=tenant_context.tenant_id,
        project_id=project_id,
        name=body.name,
        description=body.description,
    )
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    AuditService(db).log_event(
        tenant_id=tenant_context.tenant_id,
        actor_id=tenant_context.user_id,
        action="project.updated",
        resource_type="project",
        resource_id=str(project.id),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"name": project.name},
        commit=True,
    )
    return {
        "id": project.id,
        "tenant_id": project.tenant_id,
        "name": project.name,
        "description": project.description,
    }


@router.get("/{project_id}/export")
def export_project(
    request: Request,
    project_id: int,
    _: AuthPrincipal = Depends(require_permission("export.download")),
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    tenant_context = get_tenant_context(db, principal)
    repo = ProjectRepository(db)
    project = repo.get_project_for_tenant(
        tenant_id=tenant_context.tenant_id,
        project_id=project_id,
    )
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    AuditService(db).log_event(
        tenant_id=tenant_context.tenant_id,
        actor_id=tenant_context.user_id,
        action="project.exported",
        resource_type="project",
        resource_id=str(project.id),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"export_type": "json"},
        commit=True,
    )
    return {
        "project_id": project.id,
        "tenant_id": project.tenant_id,
        "export_type": "json",
        "payload": {
            "name": project.name,
            "description": project.description,
        },
    }
