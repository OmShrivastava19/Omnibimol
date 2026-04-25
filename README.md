# 🧬 OmniBiMol

**Operations on Medicinally Navigated Information for Biological Molecules**

OmniBiMol is a Streamlit-based bioinformatics platform that unifies protein discovery, sequence analytics, 3D structure visualization, docking workflows, drug repurposing insights, and genome-level risk assessment. It aggregates data from leading public resources while keeping results fast with a built-in cache layer.

## Highlights

- End-to-end protein exploration: UniProt, GO terms, tissue expression, and subcellular localization.
- Sequence analysis suite for DNA/RNA/proteins, including alignment, motifs, and phylogenetics.
- 3D structures from PDB and AlphaFold with interactive viewers and confidence plots.
- KEGG pathway mapping with interactive overlays and export.
- STRING PPI networks and interaction tables.
- Ligand binding prediction, docking simulation, and SAR-style insights.
- Drug search, clinical trials lookup, and repurposing recommendations.
- Whole genome analysis for mutations, biomarkers, and risk scoring.
- Academic model hub UI for FlexPose, DeePathNet, CRISPR-DIPOFF, and DeepDTAGen execution.

## What the App Can Do

### 1) Protein Discovery & Annotation
- UniProt search by protein or gene name.
- Detailed protein profile: function summary, sequence length, mass, and GO terms.
- FASTA extraction and formatting for downstream analysis.

### 2) Tissue Expression & Subcellular Localization (HPA)
- Tissue expression bar charts with expression level scoring.
- Subcellular localization heatmaps with reliability grading.
- Summary tables of tissue and localization coverage.

### 3) Protein Sequence Analysis (UniProt + EMBL-EBI)
- Protein FASTA output.
- Amino acid composition analysis.
- BLAST homology search.
- EMBL-EBI features exploration and Needle pairwise alignment.

### 4) 3D Protein Structure
- Experimental PDB structures with NGL viewer.
- AlphaFold predicted structures with pLDDT confidence plots.
- Optional py3Dmol rendering if available.

### 5) Pathway Exploration (KEGG)
- Primary pathway map with interactive KGML overlays.
- Next 5 related pathways and full list of associated pathways.
- Exportable pathway summaries.

### 6) Protein-Protein Interaction Networks (STRING)
- Interactive PPI network graph.
- Ranked interaction table with confidence and evidence.

### 7) Molecular Docking & Ligand Analysis
- Known ligands from ChEMBL.
- Binding predictor with molecular descriptors and Lipinski checks.
- Similar compound search and ranking.
- Custom docking simulation with predicted binding scores.
- 3D protein-ligand complex visualization.
- Docking results export.

### 8) Drug Search, Trials & Repurposing
- Drug profile search with external database links.
- FDA approval status and clinical trial discovery.
- Repurposing engine built on drug-protein-disease-pathway networks.
- Ranked repurposing opportunities with evidence summaries.

### 9) Whole Genome Sequencing Analysis
- FASTA genome input with metadata (age, gender, etc.).
- Mutation analysis and biomarker detection.
- Disease risk assessment with confidence grading.
- Personalized insights and pharmacogenomic notes.
- Predictive risk calculator and recommendations.

### 10) Academic Models UI
- Discover available academic adapters with runtime and health metadata.
- Submit normalized model runs using guided forms (no raw payload authoring required).
- Inspect prediction/explanations/confidence/artifacts/provenance/errors in consistent tabs.
- Track session run history and reopen previous results.
- Export normalized response, provenance manifest, and run summary artifacts.

#### Academic Models launch-readiness checklist
- Model discovery + shallow/deep health checks render from backend endpoints.
- CRISPR candidate entry supports table editor, CSV upload, and advanced JSON fallback.
- Model-specific analytics render for FlexPose, DeePathNet, CRISPR-DIPOFF, and DeepDTAGen.
- Request correlation fields are visible (`request_id`, `request_hash`, runtime mode, status).
- Browser-level UI tests cover submit flows, tab navigation, artifacts, and error hints.
- Non-clinical disclaimer remains visible in prediction context.

### Academic Models test setup (release gate)
- Install test dependencies: `pip install -r requirements.txt`
- Install browser runtime: `python -m playwright install chromium` (local), `python -m playwright install --with-deps chromium` (CI Linux)
- Run focused gate:
  - `python -m pytest tests/test_academic_model_hub.py tests/test_api_client_academic_models.py tests/test_academic_model_ui_helpers.py tests/test_academic_models_accessibility.py -q`
  - `python -m pytest tests/test_academic_models_ui_playwright.py -q`

## Tech Stack

- Streamlit (UI)
- FastAPI (backend API)
- PostgreSQL + SQLAlchemy + Alembic (data and migrations)
- Redis (async job broker/state backend)
- Pandas, NumPy (data processing)
- Plotly + Matplotlib (visualizations)
- httpx (API calls)
- Biopython (sequence analysis)
- NetworkX (graph analytics)
- SQLite (persistent caching)

## Real Docking Startup

Real docking uses the FastAPI backend job queue and the worker process. For local runs, start all of these together:

1. Backend API: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
2. Docking worker: `python -m backend.workers.docking_worker`
3. PostgreSQL with `DATABASE_URL` pointing at the same database for the API and worker
4. Redis with `REDIS_URL` configured for the backend stack

Set `BACKEND_API_URL` in the Streamlit environment to the API base URL, for example `http://localhost:8000`. If that URL is wrong or the API is down, real docking fails with a clean message instead of a raw socket error.

For academic model runtime setup and UI usage, see `docs/ACADEMIC_MODELS_UI.md` and `docs/academic_model_hub_deployment.md`.

## Data & APIs Used

- UniProt (protein annotations, sequences)
- Human Protein Atlas (tissue expression, subcellular localization)
- KEGG (pathways)
- STRING (PPI)
- AlphaFold DB (predicted structures)
- PDB (experimental structures)
- ChEMBL (ligands and targets)
- PubChem (compound structures)
- PubMed (literature summaries)
- NCBI (sequence lookup)
- EMBL-EBI (sequence analysis services)
- ClinicalTrials.gov (trial discovery)

## Project Structure

```
omnibimol/
├── app.py                      # Main Streamlit application
├── backend/                    # FastAPI backend (auth, RBAC, tenancy, audit, jobs)
├── alembic/                    # DB migrations
├── .github/workflows/          # CI quality gates
├── docker/                     # Dockerfiles for streamlit + backend
├── api_client.py               # API integrations and data fetching
├── cache_manager.py            # SQLite + Streamlit caching utilities
├── drug_repurposing_engine.py  # Repurposing network analysis
├── genome_analysis_engine.py   # Genome risk and biomarker analysis
├── ligand_binding_predictor.py # SMILES validation and ML predictors
├── sequence_analysis.py        # Sequence analytics suite
├── visualizations.py           # Plotly and NGL visualization helpers
├── data/
│   ├── normal_tissue.tsv
│   └── subcellular_location.tsv
├── examples/
│   ├── example_dna.fasta
│   └── example_protein.fasta
└── requirements.txt
```

## Architecture (Current)

- **UI:** Streamlit keeps user-facing scientific workflows intact.
- **API:** FastAPI handles identity, RBAC, tenant isolation, audit events, and async job metadata.
- **Data:** PostgreSQL stores multi-tenant system records; SQLite cache remains for existing local workflow acceleration.
- **Async:** Job lifecycle is persisted and polled through backend endpoints (`queued`, `running`, `completed`/`failed`).
- **Observability:** request IDs + structured logs + audit event trail.

## Environment Variables

Core backend variables (see `.env.example`):

- `ENVIRONMENT` - runtime mode (`development` by default)
- `APP_NAME` - API service name
- `API_PREFIX` - path prefix (default `/api/v1`)
- `DEBUG` - FastAPI debug mode
- `LOG_LEVEL` - logging level
- `DATABASE_URL` - Postgres connection string
- `REDIS_URL` - Redis connection string
- `AUTH_ENABLED` - enable strict JWT auth flow
- `AUTH0_DOMAIN` - Auth0 tenant domain
- `AUTH0_AUDIENCE` - expected token audience
- `AUTH_JWT_ALGORITHMS` - accepted JWT algorithm list
- `AUTH_TENANT_CLAIM` - claim key for tenant slug

## Installation

### Prerequisites
- Python 3.9+ recommended
- pip

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install pinned versions:

```bash
pip install -r requirements.lock
```

## Run the App

```bash
streamlit run app.py
```

Then open: http://localhost:8501

## Backend API (FastAPI)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API base URL: `http://localhost:8000/api/v1`

### Useful API groups

- `/api/v1/auth/*` identity + RBAC helpers
- `/api/v1/projects/*` tenant-scoped project access
- `/api/v1/jobs/*` async workload lifecycle
- `/api/v1/audit/events` privileged audit browsing
- `/api/v1/reliability/*` upstream/degradation and upload validation

## Quality Gates

Run local checks before merging:

```bash
python -m ruff check backend alembic tests
python -m mypy backend
python -m pytest --cov=backend --cov-report=term --cov-fail-under=70
```

CI workflow is defined in `.github/workflows/ci.yml` and runs the same gates on pushes/PRs.

## Docker Compose (App + API + DB + Redis)

```bash
docker compose up --build
```

Services:
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- Postgres: localhost:5432
- Redis: localhost:6379

## Operations and Security Docs

- Security policy: `SECURITY.md`
- Incident + operations runbook: `docs/OPERATIONS_RUNBOOK.md`
- Legacy migration guide: `docs/MIGRATION_GUIDE.md`

## Usage Tips

- Start with a protein search (e.g., TP53, BRCA1, EGFR).
- Use the sequence analysis suite for FASTA uploads and alignment.
- For docking, begin with known ligands and switch to custom SMILES when ready.
- Use the genome analysis tool for high-level risk insights and biomarkers.

## Caching

- SQLite cache persists results for 24 hours.
- Streamlit cache accelerates repeated computations within sessions.
- Clear cache from the UI if external data has been updated.

## Known Limitations

- Some docking results are simulated to provide rapid feedback.
- Public APIs may have rate limits or intermittent availability.
- Genome analysis uses curated pattern matching for demo-level insights.

## Scientific Use Notice

- This platform is for research, educational, and exploratory workflows.
- It is **not** intended for clinical diagnosis or patient-care decisions.
- Existing in-app research-use disclaimers are intentionally preserved.

## Roadmap Ideas

- Expand multi-organism support.
- Integrate real docking engines (AutoDock, Vina).
- Add pathway enrichment analytics.
- Enable export bundles (PDF reports, batch CSV).

## License

Provided for research and educational use. See [LICENSE](LICENSE).

## Repository

GitHub: https://github.com/OmShrivastava19/Omnibimol