"""HTTP client with retries, timeout, and simple circuit breaker behavior."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from backend.core.errors import UpstreamUnavailableError


@dataclass
class CircuitState:
    failure_count: int = 0
    opened_until: datetime | None = None


class ResilientHttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
        failure_threshold: int = 3,
        reset_timeout_seconds: int = 30,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self._circuits: dict[str, CircuitState] = {}

    def _host_key(self, url: str) -> str:
        return httpx.URL(url).host or "unknown"

    def _is_open(self, host: str) -> bool:
        state = self._circuits.get(host)
        if state is None or state.opened_until is None:
            return False
        if datetime.now(UTC) >= state.opened_until:
            state.opened_until = None
            state.failure_count = 0
            return False
        return True

    def _record_failure(self, host: str) -> None:
        state = self._circuits.setdefault(host, CircuitState())
        state.failure_count += 1
        if state.failure_count >= self.failure_threshold:
            state.opened_until = datetime.now(UTC) + timedelta(seconds=self.reset_timeout_seconds)

    def _record_success(self, host: str) -> None:
        self._circuits[host] = CircuitState(failure_count=0, opened_until=None)

    def get_json(self, url: str) -> dict[str, Any]:
        host = self._host_key(url)
        if self._is_open(host):
            raise UpstreamUnavailableError(
                message="Circuit is open for upstream host",
                details={"host": host},
            )

        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    payload = response.json()
                self._record_success(host)
                if not isinstance(payload, dict):
                    return {"data": payload}
                return payload
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_error = exc
                self._record_failure(host)

        raise UpstreamUnavailableError(
            message="Failed to fetch upstream response after retries",
            details={"host": host, "reason": str(last_error)},
        )
