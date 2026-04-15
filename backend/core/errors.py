"""Shared error primitives and FastAPI exception handlers."""

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int = HTTPStatus.BAD_REQUEST
    details: dict[str, Any] | None = None


class UpstreamUnavailableError(AppError):
    def __init__(
        self,
        message: str = "Upstream service unavailable",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            code="upstream_unavailable",
            message=message,
            status_code=HTTPStatus.BAD_GATEWAY,
            details=details or {},
        )


class InputValidationError(AppError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="invalid_input",
            message=message,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            details=details or {},
        )


def _error_payload(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None,
    request: Request,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request.headers.get("x-request-id"),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                code=exc.code,
                message=exc.message,
                details=exc.details,
                request=request,
            ),
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        message = str(exc.detail) if exc.detail else HTTPStatus(exc.status_code).phrase
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                code=f"http_{exc.status_code}",
                message=message,
                details={},
                request=request,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content=_error_payload(
                code="request_validation_error",
                message="Request payload validation failed",
                details={"errors": exc.errors()},
                request=request,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled API exception", exc_info=exc)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=_error_payload(
                code="internal_server_error",
                message="Unexpected internal server error",
                details={},
                request=request,
            ),
        )
