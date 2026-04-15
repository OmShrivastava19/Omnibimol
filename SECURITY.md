# Security Policy

## Supported Scope

This repository is intended for research software and SaaS hardening. Security fixes apply to:
- `backend/` API and authn/authz layers
- Multi-tenant data isolation behavior
- CI/CD and deployment configuration
- Secrets management and operational controls

## Reporting a Vulnerability

If you discover a security issue:
1. Do **not** open a public issue with exploit details.
2. Contact the maintainers privately with:
   - affected component/path
   - reproduction steps
   - potential impact
   - suggested mitigation (optional)
3. Allow time for triage and coordinated disclosure.

## Secrets and Credentials

- Never commit credentials, tokens, private keys, or production dumps.
- Use environment variables for all runtime secrets.
- Keep `.env` local-only; use `.env.example` for placeholders.
- Rotate any key immediately if exposure is suspected.

## Authentication and Authorization Baseline

- Authentication: Auth0 JWT validation (`issuer`, `audience`, signature).
- Authorization: RBAC permissions and tenant-scoped resource checks.
- Tenant isolation is enforced at repository/query level, not UI filtering.

## Key Rotation Basics

- Rotate Auth0 client secrets and signing keys on a fixed cadence.
- Update affected environment variables and redeploy API/workers.
- Invalidate old secrets promptly in identity provider consoles.
- Confirm post-rotation health via `/api/v1/healthz` and auth smoke tests.

## Operational Hardening Checklist

- Enforce TLS for all externally exposed services.
- Restrict database and Redis network access to trusted networks.
- Enable least-privilege database roles.
- Ensure CI quality gates pass (`ruff`, `mypy`, `pytest`, coverage threshold).
- Keep container images and Python dependencies patched regularly.
