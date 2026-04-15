"""Application settings for the backend API service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings with safe defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="OmniBioMol API")
    environment: str = Field(default="development")
    api_prefix: str = Field(default="/api/v1")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    backend_cors_origins: list[str] = Field(default_factory=list)

    auth0_domain: str = Field(default="")
    auth0_audience: str = Field(default="")
    auth_enabled: bool = Field(default=False)
    auth_jwt_algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    auth_tenant_claim: str = Field(default="https://omnibimol.io/tenant_slug")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/omnibimol"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    @property
    def auth0_issuer(self) -> str:
        if not self.auth0_domain:
            return ""
        return f"https://{self.auth0_domain}/"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object to avoid repeated environment parsing."""
    return Settings()
