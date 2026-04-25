"""Application settings for the backend API service."""

from functools import lru_cache
from typing import Literal

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

    docking_enabled: bool = Field(default=True)
    docking_mode_default: Literal["real", "simulation"] = Field(default="real")
    docking_engine: Literal["vina", "quickvina2"] = Field(default="vina")
    docking_cache_dir: str = Field(default="/var/lib/omnibimol/docking-cache")
    docking_timeout_seconds: int = Field(default=900, ge=30)
    docking_vina_binary: str = Field(default="vina")
    docking_worker_concurrency: int = Field(default=1, ge=1, le=16)
    multiomics_enabled: bool = Field(default=True)
    multiomics_calibration_a: float = Field(default=1.0)
    multiomics_calibration_b: float = Field(default=0.0)
    multiomics_transcriptomics_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    multiomics_genomics_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    multiomics_proteomics_weight: float = Field(default=0.20, ge=0.0, le=1.0)

    chemprot_enabled: bool = Field(default=True)
    chemprot_target_prioritization_enabled: bool = Field(default=False)
    chemprot_adapter_repo_id: str = Field(default="omshrivastava/omnibimol-chemprot-lora")
    chemprot_base_model_id: str = Field(default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    chemprot_base_model_revision: str = Field(default="main")
    chemprot_adapter_revision: str = Field(default="main")
    chemprot_max_length: int = Field(default=256, ge=32, le=1024)
    chemprot_batch_size: int = Field(default=4, ge=1, le=32)
    chemprot_enable_ensemble: bool = Field(default=True)
    chemprot_timeout_sec: int = Field(default=30, ge=5, le=120)
    chemprot_local_files_only: bool = Field(default=False)

    backend_api_url: str = Field(default="http://localhost:8000")

    @property
    def auth0_issuer(self) -> str:
        if not self.auth0_domain:
            return ""
        return f"https://{self.auth0_domain}/"

    @property
    def normalized_docking_mode_default(self) -> str:
        return self.docking_mode_default.lower().strip()

    @property
    def normalized_docking_engine(self) -> str:
        return self.docking_engine.lower().strip()


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object to avoid repeated environment parsing."""
    return Settings()
