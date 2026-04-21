from backend.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "OmniBioMol API"
    assert settings.api_prefix == "/api/v1"
    assert settings.environment == "development"
    assert settings.database_url
    assert settings.redis_url
    assert settings.docking_enabled is True
    assert settings.docking_mode_default == "real"
    assert settings.docking_engine == "vina"
    assert settings.docking_cache_dir
    assert settings.docking_timeout_seconds >= 30
