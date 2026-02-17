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

## Tech Stack

- Streamlit (UI)
- Pandas, NumPy (data processing)
- Plotly + Matplotlib (visualizations)
- httpx (API calls)
- Biopython (sequence analysis)
- NetworkX (graph analytics)
- SQLite (persistent caching)

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

## Run the App

```bash
streamlit run app.py
```

Then open: http://localhost:8501

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

## Roadmap Ideas

- Expand multi-organism support.
- Integrate real docking engines (AutoDock, Vina).
- Add pathway enrichment analytics.
- Enable export bundles (PDF reports, batch CSV).

## License

Provided for research and educational use. See [LICENSE](LICENSE).

## Repository

GitHub: https://github.com/OmShrivastava19/Omnibimol