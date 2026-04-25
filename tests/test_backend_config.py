from backend.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "OmniBiMol API"
    assert settings.api_prefix == "/api/v1"
    assert settings.environment == "development"
    assert settings.database_url
    assert settings.redis_url
    assert settings.docking_enabled is True
    assert settings.docking_mode_default == "real"
    assert settings.docking_engine == "vina"
    assert settings.docking_cache_dir
    assert settings.docking_timeout_seconds >= 30
    assert settings.localizer_enabled is True
    assert settings.localizer_repo_id == "omshrivastava/omnibimol-protein-localization"
    assert settings.localizer_confidence_threshold == 0.6
    assert settings.localizer_max_seq_len == 1024
