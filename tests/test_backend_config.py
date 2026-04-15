from backend.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "OmniBioMol API"
    assert settings.api_prefix == "/api/v1"
    assert settings.environment == "development"
    assert settings.database_url
    assert settings.redis_url
