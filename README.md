# 🧬 OmniBiMol - Protein Analysis Platform

**Operations on Medicinally Navigated Information for Biological Molecules**

## Project Overview

OmniBiMol is a Phase 1 MVP (Minimum Viable Product) Streamlit application designed for comprehensive protein analysis and visualization. It integrates data from multiple bioinformatics sources to provide researchers with a unified platform for exploring protein properties, tissue expression patterns, subcellular localization, and gene ontology information.

## Features

### 1. **Protein Search**
- Search proteins by name or gene symbol
- Integration with UniProt API for protein discovery
- Display multiple matches with UniProt IDs and organism information
- Support for human proteins (Homo sapiens)

### 2. **Protein Information Display**
- **UniProt Data**: Comprehensive protein annotations including:
  - Protein name and gene name
  - UniProt ID and length
  - Protein function and cellular information
  - Gene ontology (GO) terms categorized by:
    - Biological Process
    - Molecular Function
    - Cellular Component

### 3. **Tissue Expression Analysis**
- Interactive bar charts showing protein expression levels across tissues
- Color-coded expression levels: High (red), Medium (orange), Low (yellow), Not detected (gray)
- Data sourced from local TSV files (normal_tissue.tsv)

### 4. **Subcellular Localization**
- Heatmap visualization of protein subcellular locations
- Main and additional location information
- Reliability scoring for predictions
- Data sourced from local TSV files (subcellular_location.tsv)

### 5. **Gene Ontology Visualization**
- Stacked bar charts for GO term distribution
- Categorized by biological process, molecular function, and cellular component

### 6. **Caching System**
- 24-hour cache expiration for API responses
- SQLite-based local database for performance optimization
- Thread-safe caching to prevent concurrent access issues

## Project Structure

```
omnibimol/
├── app.py                      # Main Streamlit application
├── api_client.py               # UniProt API integration and data fetching
├── cache_manager.py            # SQLite caching system
├── data_processor.py           # Data transformation utilities
├── visualizations.py           # Plotly chart generation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/
    ├── normal_tissue.tsv       # Tissue expression data
    └── subcellular_location.tsv # Subcellular localization data
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Instructions

1. **Clone or download the project**:
```bash
cd omnibimol
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Workflow

1. **Search for a Protein**: Enter a protein name or gene symbol (e.g., "TP53", "BRCA1")
2. **Select Protein**: Choose from the search results
3. **View Analysis**: Explore:
   - Protein function and annotations
   - Gene ontology terms
   - Tissue expression patterns
   - Subcellular localization
   - Summary statistics

## Data Sources

### Local Data Files
- **normal_tissue.tsv**: Human Protein Atlas tissue expression data
  - Columns: Gene, Tissue, Level (High/Medium/Low/Not detected), Level_numeric
  
- **subcellular_location.tsv**: Human Protein Atlas subcellular localization data
  - Columns: Gene, Main_location, Additional_location, Reliability

### APIs
- **UniProt API**: Used for protein discovery and comprehensive annotations
  - Base URL: https://rest.uniprot.org
  - Provides UniProt IDs, gene names, protein functions, and GO terms

## Architecture

### Components

1. **ProteinAPIClient** (`api_client.py`)
   - Handles UniProt API communication
   - Loads and filters local TSV data
   - Manages asynchronous API calls
   - Implements caching with thread-safe SQLite

2. **CacheManager** (`cache_manager.py`)
   - SQLite-based caching with 24-hour TTL
   - Thread-safe connection management
   - Automatic cache expiration

3. **DataProcessor** (`data_processor.py`)
   - Transforms raw API/TSV data into visualization-ready formats
   - Filters and sorts tissue expression data
   - Prepares heatmap and chart data

4. **ProteinVisualizer** (`visualizations.py`)
   - Creates interactive Plotly visualizations
   - Generates tissue expression bar charts
   - Builds subcellular localization heatmaps
   - Creates GO term distribution charts

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **httpx**: Asynchronous HTTP client for APIs
- **SQLite3**: Local caching database
- **Python asyncio**: Asynchronous programming

## Performance Considerations

- **Caching**: 24-hour TTL on API responses reduces network calls
- **Local TSV Files**: Eliminates network latency for tissue/localization data
- **Thread-Safe SQLite**: Prevents concurrent access issues in Streamlit
- **Lazy Loading**: Data loaded only when requested

## Error Handling

- Network error handling for API calls
- Graceful fallbacks for missing data
- Input validation for protein searches
- Thread-safe database operations

## Future Enhancements

- [ ] Additional protein databases (NCBI, ENSEMBL)
- [ ] Multiple organism support
- [ ] Advanced filtering options
- [ ] Protein interaction networks
- [ ] Disease association data
- [ ] Literature mining integration

## Troubleshooting

### Common Issues

**Application won't start**:
- Ensure Python 3.8+ is installed
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory

**Data not loading**:
- Verify `data/normal_tissue.tsv` and `data/subcellular_location.tsv` exist
- Ensure TSV files have correct column names
- Check file encoding (should be UTF-8)

**API errors**:
- Check internet connection
- Verify UniProt API is accessible
- Check cache file permissions

**Performance issues**:
- Clear cache: Delete `omnibiomol_cache.db`
- Check system resources
- Restart Streamlit application

## License

This project is provided as-is for research and educational purposes.

## Support

For issues or questions, please check the deployment_instructions.txt file or review the inline code comments for more details on specific components.

---

**Last Updated**: January 27, 2026  
**Version**: 1.0 (Phase 1 MVP)
