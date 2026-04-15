# Operations Runbook

## 1. Service Topology

- `streamlit` UI: user-facing research workflows
- `api` FastAPI: auth, RBAC, tenancy, audit, jobs
- `db` Postgres: system of record
- `redis` queue/cache broker for async workloads

## 2. Local Startup

```bash
docker compose up --build
```

Expected ports:
- Streamlit: `8501`
- API: `8000`
- Postgres: `5432`
- Redis: `6379`

## 3. Health Verification

- Liveness: `GET /api/v1/healthz`
- Readiness: `GET /api/v1/readyz`
- Check request IDs are returned in `x-request-id` response header

## 4. Incident Response Basics

1. Acknowledge incident and capture timestamp/scope.
2. Check API health and error-rate symptoms first.
3. Validate DB and Redis connectivity.
4. Check latest deploy/CI changes that landed before incident.
5. If auth failures spike, validate Auth0 issuer/audience/JWKS reachability.
6. Stabilize service (rollback/feature flag off) before deeper root-cause work.

## 5. Frequent Failure Modes

### Auth failures
- Verify `AUTH_ENABLED`, `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`.
- Confirm system clock drift is not causing token expiry issues.

### Tenant access regressions
- Verify repository-level tenant filters are still active.
- Run tenant isolation tests and inspect audit events for access attempts.

### Job processing issues
- Check queued/running/failed status transitions via `/api/v1/jobs/{id}`.
- Validate Redis availability and worker connectivity (if enabled).

### Upstream API instability
- Inspect degraded responses from reliability endpoints.
- Circuit-breaker behavior should prevent cascading failure.

## 6. Key Rotation Procedure (Minimal)

1. Generate new secret/key in identity provider.
2. Update deployment env vars/secrets.
3. Redeploy API and dependent workers.
4. Run auth smoke tests.
5. Revoke old key once new key is verified.

## 7. Recovery Verification Checklist

- API health and readiness are green.
- Auth sync and role-protected endpoints behave correctly.
- Tenant isolation tests pass.
- Audit events are still emitted for privileged actions.
- Async jobs transition correctly (`queued -> running -> terminal`).
