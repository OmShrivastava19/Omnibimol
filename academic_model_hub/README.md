# Academic Model Hub

`academic_model_hub` is a plugin-style integration layer that wraps published academic models behind one normalized API while keeping upstream invocation paths traceable.

## Unified API

```python
from academic_model_hub import AcademicModelHub

hub = AcademicModelHub()
result = hub.predict(model_name="flexpose", payload={...})
```

All adapters return the normalized shape:

- `status`, `model`, `model_version`, `source_paper`
- `input_schema_version`
- `prediction`, `explanations`, `artifacts`, `confidence`
- `provenance`
- `errors`

## Included Adapters

- `flexpose`
- `deepathnet`
- `crispr-dipoff`
- `deepdtagen`

## Runtime Modes

Each adapter declares runtime expectations:

- `mode`: `native` or `container`
- `python_version`, `torch_version`
- `gpu_optional`, `gpu_required`

Per request, optionally set `runtime_mode`:

- `native` (in-process / local script path)
- `container` (Docker invocation path)

If runtime prechecks fail, adapters return structured `INCOMPATIBLE_RUNTIME` errors with safe setup hints.

## Executable vs conditional

- Executable now: adapter command construction, runtime prechecks, subprocess execution, output parsing, provenance/run manifests, deep/shallow health checks.
- Conditional on environment: upstream repo checkout paths, model checkpoints, and script compatibility for specific upstream commits.

## Design Guardrails

- Wrappers are intentionally thin.
- Adapter-only compatibility shims; no silent semantic rewrites.
- Wrapper heuristics are explicitly labeled as wrapper logic.
- Error responses use structured, safe error details.
- Provenance manifests are written beside output artifacts.
