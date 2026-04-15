# Migration Guide: Legacy to Backend-Backed SaaS

This guide covers migration from local-only Streamlit/cache workflows to the backend-backed architecture.

## 1. Migration Goals

- Preserve scientific workflows and UX behavior in `app.py`
- Introduce secure auth/RBAC/tenant boundaries
- Move long-running tasks to managed job lifecycle
- Keep historical local cache data available where useful

## 2. Current Data Sources

- Legacy local cache DB: `omnibiomol_cache.db`
- Optional local portfolio DBs created in previous workflows
- New backend system tables in Postgres (tenants/users/roles/projects/audit/jobs)

## 3. Recommended Migration Steps

1. **Back up local DBs**
   - Copy `omnibiomol_cache.db` and any local project DB files.
2. **Stand up backend dependencies**
   - Start Postgres + API (+ Redis if async workers are enabled).
3. **Run migrations**
   - Apply Alembic revisions in order.
4. **Seed baseline tenant/admin**
   - Run `backend/scripts/seed_dev.py` in non-production environments.
5. **Enable API integration in UI path**
   - Keep Streamlit scientific flows unchanged while routing identity, RBAC, and jobs to API.
6. **Validate tenant access and audit visibility**
   - Execute tenant isolation and audit tests before production cutover.

## 4. Cache and Local Data Strategy

- Keep legacy cache as read-only reference during transition.
- Do not delete legacy DB files until smoke and regression checks pass.
- Prefer writing net-new workflow state to backend-managed stores.

## 5. Validation Checklist

- All CI gates passing (`ruff`, `mypy`, `pytest`, coverage threshold).
- Health/readiness endpoints return expected status.
- Auth sync works and role checks are enforced.
- Cross-tenant access attempts are denied.
- Audit events are generated for privileged operations.
- Async job statuses progress correctly.

## 6. Rollback Plan

If issues occur:
1. Disable new API-integrated paths via configuration.
2. Continue running Streamlit in legacy/local mode.
3. Preserve backend DB for post-incident analysis.
4. Address root cause, then retry staged cutover.
