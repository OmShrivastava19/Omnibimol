"""Helpers for bridging Streamlit session auth to backend API calls."""

from typing import Any


def build_backend_auth_headers(session_state: dict[str, Any]) -> dict[str, str]:
    """Build authorization headers from Streamlit session state token fields."""
    token = (
        session_state.get("auth_access_token")
        or session_state.get("access_token")
        or session_state.get("auth0_access_token")
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def build_job_status_url(api_base_url: str, job_id: int) -> str:
    """Return a pollable URL for backend async job status checks."""
    return f"{api_base_url.rstrip('/')}/api/v1/jobs/{job_id}"
