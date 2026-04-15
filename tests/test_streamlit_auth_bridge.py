from backend.auth.streamlit_integration import build_backend_auth_headers


def test_build_backend_auth_headers_uses_preferred_token_key() -> None:
    headers = build_backend_auth_headers({"auth_access_token": "abc123"})
    assert headers["Authorization"] == "Bearer abc123"


def test_build_backend_auth_headers_falls_back_to_access_token() -> None:
    headers = build_backend_auth_headers({"access_token": "fallback-token"})
    assert headers["Authorization"] == "Bearer fallback-token"


def test_build_backend_auth_headers_empty_when_no_token() -> None:
    assert build_backend_auth_headers({}) == {}
