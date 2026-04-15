"""Auth0 JWT token verification helpers."""

from dataclasses import dataclass
from typing import Any

import jwt
from jwt import InvalidTokenError, PyJWKClient

from backend.core.config import Settings, get_settings


class AuthError(Exception):
    """Raised when authentication fails."""


@dataclass(frozen=True)
class AuthPrincipal:
    """Normalized identity extracted from a validated JWT."""

    sub: str
    email: str
    name: str
    tenant_slug: str
    raw_claims: dict[str, Any]


class Auth0TokenVerifier:
    """Validate Auth0 access tokens using issuer/audience/JWKS."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.jwks_client: PyJWKClient | None = None

    def _get_jwks_client(self) -> PyJWKClient:
        if self.jwks_client is None:
            if not self.settings.auth0_domain:
                raise AuthError("AUTH0_DOMAIN is not configured")
            jwks_url = f"https://{self.settings.auth0_domain}/.well-known/jwks.json"
            self.jwks_client = PyJWKClient(jwks_url)
        return self.jwks_client

    def _resolve_signing_key(self, token: str) -> Any:
        client = self._get_jwks_client()
        return client.get_signing_key_from_jwt(token).key

    def verify(self, token: str) -> AuthPrincipal:
        if not self.settings.auth0_audience:
            raise AuthError("AUTH0_AUDIENCE is not configured")
        if not self.settings.auth0_issuer:
            raise AuthError("AUTH0_DOMAIN is not configured")

        try:
            signing_key = self._resolve_signing_key(token)
            claims = jwt.decode(
                token,
                key=signing_key,
                algorithms=self.settings.auth_jwt_algorithms,
                audience=self.settings.auth0_audience,
                issuer=self.settings.auth0_issuer,
            )
        except InvalidTokenError as exc:
            raise AuthError("Token validation failed") from exc

        tenant_slug = (
            claims.get(self.settings.auth_tenant_claim)
            or claims.get("tenant_slug")
            or "default-tenant"
        )
        return AuthPrincipal(
            sub=str(claims.get("sub", "")),
            email=str(claims.get("email", "")),
            name=str(claims.get("name", claims.get("nickname", ""))),
            tenant_slug=str(tenant_slug),
            raw_claims=claims,
        )
