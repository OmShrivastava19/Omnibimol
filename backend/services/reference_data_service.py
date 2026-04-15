"""Service layer for upstream reference data retrieval."""

from typing import Any

from backend.core.errors import AppError, UpstreamUnavailableError
from backend.integrations.resilient_http import ResilientHttpClient


class ReferenceDataService:
    def __init__(self, client: ResilientHttpClient | None = None):
        self.client = client or ResilientHttpClient()

    def fetch_upstream_status(self, url: str) -> dict[str, Any]:
        try:
            payload = self.client.get_json(url)
            return {"status": "ok", "upstream": payload}
        except UpstreamUnavailableError as exc:
            return {
                "status": "degraded",
                "upstream": {},
                "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            }
        except AppError:
            raise
