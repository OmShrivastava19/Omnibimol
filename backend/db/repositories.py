"""Repository helpers with tenant-scoped access."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Project


class ProjectRepository:
    """Tenant-aware project data access abstraction."""

    def __init__(self, db: Session):
        self.db = db

    def create_project(
        self,
        *,
        tenant_id: int,
        name: str,
        created_by_user_id: int,
        description: str = "",
    ) -> Project:
        project = Project(
            tenant_id=tenant_id,
            name=name,
            description=description,
            created_by_user_id=created_by_user_id,
        )
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project

    def list_projects_for_tenant(self, tenant_id: int) -> list[Project]:
        stmt = select(Project).where(Project.tenant_id == tenant_id).order_by(Project.id.asc())
        return list(self.db.scalars(stmt))

    def get_project_for_tenant(self, *, tenant_id: int, project_id: int) -> Project | None:
        stmt = select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id)
        return self.db.scalar(stmt)

    def update_project_for_tenant(
        self,
        *,
        tenant_id: int,
        project_id: int,
        name: str | None = None,
        description: str | None = None,
    ) -> Project | None:
        project = self.get_project_for_tenant(tenant_id=tenant_id, project_id=project_id)
        if project is None:
            return None
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        self.db.commit()
        self.db.refresh(project)
        return project
