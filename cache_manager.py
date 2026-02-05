"""
cache_manager.py - Unified caching system

Combines both SQLite-based persistent caching and Streamlit native caching:
1. CacheManager: SQLite database for cross-session persistence (24h TTL)
2. Streamlit @st.cache_data functions: In-memory session caching
3. clear_app_cache(): Master function to reset all caches and state

All expensive API calls and computations should be cached to improve performance
and reduce redundant external requests.
"""

from typing import Optional, Dict
import sqlite3
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import asyncio
import hashlib


# =============================================================================
# SQLITE PERSISTENT CACHING
# =============================================================================

class CacheManager:
    """Manages SQLite cache for API responses with 24-hour expiration"""
    
    def __init__(self, db_path: str = "omnibiomol_cache.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize cache database with schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            # Database might be corrupted, try to recover
            try:
                import os
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                # Retry initialization
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
                conn.commit()
                conn.close()
            except Exception as e:
                # Log but don't fail - cache is optional
                pass
    
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached data if not expired (24h TTL)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT value, timestamp FROM cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                value, timestamp = result
                # Check if cache is still valid (24 hours)
                if datetime.now().timestamp() - timestamp < 86400:  # 24h in seconds
                    return json.loads(value)
                else:
                    # Expired, delete it
                    self.delete(key)
            return None
        except sqlite3.OperationalError:
            # Table doesn't exist or database is corrupted
            return None
    
    def set(self, key: str, value: Dict):
        """Store data in cache with current timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, timestamp)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.now().timestamp()))
            
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            # Table doesn't exist, reinitialize
            self.init_db()
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO cache (key, value, timestamp)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(value), datetime.now().timestamp()))
                conn.commit()
                conn.close()
            except Exception:
                pass  # Ignore cache write errors
    
    def delete(self, key: str):
        """Remove specific cache entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            pass  # Ignore if table doesn't exist
    
    def clear_expired(self):
        """Remove all expired entries (maintenance)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = datetime.now().timestamp() - 86400
            cursor.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff,))
            
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            pass  # Ignore if table doesn't exist
    
    def clear_all(self):
        """Clear all SQLite cache entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            # Table doesn't exist, try to reinitialize
            self.init_db()
    
    def clear(self):
        """Alias for clear_all() - clear all cache entries"""
        self.clear_all()


def clear_app_cache():
    """
    MASTER CACHE CLEARING FUNCTION
    Comprehensively resets all caches and session state for a clean app restart.
    
    This function performs three critical operations:
    1. Clear Streamlit's native @st.cache_data and @st.cache_resource
    2. Clear SQLite-based custom cache (manual API response caching)
    3. Reset all session_state keys to remove prefilled values and cached results
    
    Call this when user clicks "Clear Cache" button to ensure complete state reset.
    """
    
    # Step 1: Clear Streamlit native caches
    # These store results from @st.cache_data and @st.cache_resource decorated functions
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Step 2: Clear SQLite cache (manual API response caching)
    # This manages UniProt, KEGG, BLAST, and other API responses
    if 'cache_manager' in st.session_state:
        try:
            st.session_state.cache_manager.clear_all()
        except Exception as e:
            # Handle database corruption or missing table gracefully
            try:
                # Try to recreate the database
                st.session_state.cache_manager = CacheManager()
            except Exception:
                pass  # Ignore initialization errors on clear
    
    # Step 3: Clear ALL session_state keys except Streamlit-managed internal keys
    # These keys hold user inputs, computation results, and prefilled values
    keys_to_clear = []
    
    for key in list(st.session_state.keys()):
        # Skip Streamlit's internal keys (they manage widget state internally)
        if key.startswith('_'):
            continue
        
        # Core data results - MUST CLEAR
        if key in [
            'current_data',           # Fetched protein data
            'current_uniprot_id',     # Selected protein ID
            'search_results',         # UniProt search results
            'show_results',           # UI state flag
            'trigger_search',         # Search trigger flag
            'protein_input',          # Prefilled search box
            'fetch_time',             # Timing info
        ]:
            keys_to_clear.append(key)
        
        # Sequence analysis results - MUST CLEAR
        elif key in [
            'sequence_analysis_results',  # BLAST, alignment, phylo results
            'blast_results',              # BLAST homology search results
            'blast_protein_id',           # Protein ID for BLAST cache validation
            'blast_time',                 # BLAST timing
            'embl_features',              # EMBL domain/feature annotations
            'embl_protein_id',            # Protein ID for EMBL cache validation
            'needle_results',             # Pairwise alignment results
            'compare_sequence',           # Comparison sequence for Needle
            'compare_id',                 # ID for comparison sequence
        ]:
            keys_to_clear.append(key)
        
        # Molecular docking results - MUST CLEAR
        elif key in [
            'docking_results',            # AutoDock Vina simulation results
            'docked_ligand_name',         # Name of docked compound
            'docked_ligand_data',         # Structure/properties of docked ligand
            'protein_structure',          # Selected protein structure (PDB/AlphaFold)
            'show_docking_success',       # Docking completion UI flag
            'selected_ligand',            # Ligand selected from binding predictor
            'selected_ligand_name',       # Name of selected ligand
            'selected_ligand_for_docking',# Ligand data for custom docking
            'custom_ligand',              # User-entered custom ligand
        ]:
            keys_to_clear.append(key)
        
        # Binding prediction results - MUST CLEAR
        elif key in [
            'binding_prediction',         # ML binding affinity predictions
            'novel_prediction',           # Novel drug candidate predictions
            'novel_candidates_data',      # Candidate discovery results
            'similar_prediction',         # Similar compound predictions
            'similar_data',               # Similar compound search data
            'similar_auto_run',           # Auto-run flag for similarity search
            'reference_smiles',           # Reference compound for similarity
            'reference_name',             # Reference compound name
            'similar_similarity',         # Similarity threshold
        ]:
            keys_to_clear.append(key)
        
        # UI input fields and settings - SHOULD CLEAR to reset forms
        elif key in [
            'search_input_key',           # Text input widget value (resets form)
            'page_selector',              # Page navigation state
            'current_page',               # Current page selection
            'blast_run_search',           # Button state
            'needle_fetch_sequence',      # Button state
            'needle_run_alignment',       # Button state
            'pubchem_input',              # PubChem search input
            'pubchem_search',             # PubChem search button state
            'smiles_input',               # SMILES text input
            'single_smiles_input',        # Single SMILES input
            'single_molecule_name',       # Molecule name input
            'binding_input_method',       # Input method selection
            'batch_smiles_input',         # Batch SMILES input
            'smiles_file_upload',         # File upload state
            'ligand_source_radio',        # Ligand source selection
            'use_prev_selected_ligand_select',  # Previous ligand selection
            'chembl_select',              # ChEMBL ligand selection
            'exhaustiveness_slider',      # Docking parameters
            'num_modes_slider',           # Docking parameters
            'energy_range_slider',        # Docking parameters
            'run_docking_btn',            # Docking button state
        ]:
            keys_to_clear.append(key)
    
    # Clear identified keys
    for key in keys_to_clear:
        st.session_state.pop(key, None)


# =============================================================================
# STREAMLIT NATIVE CACHING FUNCTIONS
# =============================================================================
# These @st.cache_data functions wrap expensive API calls and computations.
# They must be defined at module level for Streamlit's caching to work.
# =============================================================================

# UNIPROT & PROTEIN DATA CACHING

@st.cache_data(ttl=86400)  # 24 hour TTL
def cached_search_uniprot(protein_name: str, _api_client, max_results: int = 5):
    """Cache UniProt search results for protein name queries."""
    return asyncio.run(_api_client.search_uniprot(protein_name, max_results))


@st.cache_data(ttl=86400)
def cached_fetch_uniprot_data(uniprot_id: str, _api_client) -> Dict:
    """Cache detailed UniProt protein data (sequence, function, annotations)."""
    return asyncio.run(_api_client.fetch_uniprot_data(uniprot_id))


@st.cache_data(ttl=86400)
def cached_fetch_all_data(uniprot_id: str, gene_name: str, _api_client) -> Dict:
    """Cache comprehensive protein data fetch (UniProt + structures + pathways)."""
    return asyncio.run(_api_client.fetch_all_data(uniprot_id, gene_name))


# 3D STRUCTURE CACHING

@st.cache_data(ttl=86400)
def cached_fetch_alphafold_structure(uniprot_id: str, gene_name: Optional[str], _api_client) -> Dict:
    """Cache AlphaFold structure predictions."""
    return asyncio.run(_api_client.fetch_alphafold_structure(uniprot_id, gene_name))


@st.cache_data(ttl=86400)
def cached_fetch_pdb_structure(uniprot_id: str, _api_client) -> Dict:
    """Cache experimental PDB structures."""
    return asyncio.run(_api_client.fetch_pdb_structure(uniprot_id))


# PATHWAY & INTERACTION DATA CACHING

@st.cache_data(ttl=86400)
def cached_fetch_kegg_pathways(gene_name: str, uniprot_id: str, _api_client) -> Dict:
    """Cache KEGG pathway mapping for genes."""
    return asyncio.run(_api_client.fetch_kegg_pathways(gene_name, uniprot_id))


@st.cache_data(ttl=86400)
def cached_fetch_string_ppi(gene_name: str, uniprot_id: str, _api_client, limit: int = 10) -> Dict:
    """Cache STRING protein-protein interaction networks."""
    return asyncio.run(_api_client.fetch_string_ppi(gene_name, uniprot_id, limit))


# LITERATURE & METADATA CACHING

@st.cache_data(ttl=86400)
def cached_fetch_literature_summary(uniprot_id: str, protein_name: str, _api_client) -> dict:
    """Cache literature summary and Wikipedia snippets."""
    return asyncio.run(_api_client.fetch_literature_summary(uniprot_id, protein_name))


# SEQUENCE ANALYSIS CACHING

@st.cache_data(ttl=86400)
def cached_run_blast_search(sequence: str, uniprot_id: str, _api_client) -> Dict:
    """Cache BLAST homology search results."""
    return asyncio.run(_api_client.run_blast_search(sequence, uniprot_id))


@st.cache_data(ttl=86400)
def cached_fetch_embl_sequence(uniprot_id: str, _api_client) -> Dict:
    """Cache EMBL sequence feature annotations."""
    return asyncio.run(_api_client.fetch_embl_sequence(uniprot_id))


@st.cache_data(ttl=86400)
def cached_run_needle_alignment(sequence1: str, sequence2: str, id1: str, id2: str, _api_client) -> Dict:
    """Cache pairwise sequence alignment results."""
    return asyncio.run(_api_client.run_needle_alignment(sequence1, sequence2, id1, id2))


# LIGAND & DRUG DATA CACHING

@st.cache_data(ttl=86400)
def cached_fetch_chembl_ligands(uniprot_id: str, _api_client) -> Dict:
    """Cache ChEMBL ligand/inhibitor data for protein."""
    return asyncio.run(_api_client.fetch_chembl_ligands(uniprot_id))


@st.cache_data(ttl=86400)
def cached_fetch_pubchem_structure(compound_name: str, _api_client) -> Dict:
    """Cache PubChem compound structure data."""
    return asyncio.run(_api_client.fetch_pubchem_structure(compound_name))


@st.cache_data(ttl=86400)
def cached_fetch_similar_compounds(reference_smiles: str, similarity_threshold: float, _api_client) -> Dict:
    """Cache similar compound search results."""
    return asyncio.run(_api_client.fetch_similar_compounds(reference_smiles, similarity_threshold))


@st.cache_data(ttl=86400)
def cached_predict_drug_candidates(protein_sequence: str, gene_name: str, _api_client) -> Dict:
    """Cache drug repurposing and novel candidate predictions."""
    return asyncio.run(_api_client.predict_drug_candidates(protein_sequence, gene_name))


# HELPER DATA CACHING

@st.cache_data(ttl=86400)
def cached_load_hpa_data():
    """Cache Human Protein Atlas tissue and subcellular data."""
    from api_client import load_hpa_data
    return load_hpa_data()


@st.cache_data
def cached_get_tissue_expression(gene_name: str, _api_client) -> pd.DataFrame:
    """Cache tissue expression data for specific gene."""
    return _api_client.get_tissue_expression(gene_name)


@st.cache_data
def cached_get_subcellular_location(gene_name: str, _api_client) -> pd.DataFrame:
    """Cache subcellular localization data for specific gene."""
    return _api_client.get_subcellular_location(gene_name)


# VISUALIZATION & COMPUTATION CACHING

@st.cache_data
def cached_prepare_protein_for_docking(uniprot_data: Dict, pdb_data: Dict, 
                                       alphafold_data: Dict, _api_client) -> Dict:
    """Cache protein preparation for docking."""
    return _api_client.prepare_protein_for_docking(uniprot_data, pdb_data, alphafold_data)


@st.cache_data
def cached_simulate_docking_score(protein_length: int, ligand_mw: float, 
                                  activity_value: Optional[float], smiles: Optional[str],
                                  _api_client) -> Dict:
    """Cache docking simulation scores."""
    return _api_client.simulate_docking_score(protein_length, ligand_mw, activity_value, smiles)


# BATCH DATA PROCESSING CACHING

@st.cache_data
def cached_predict_ligand_binding(smiles_list: tuple, names_list: tuple, _api_client):
    """Cache batch ligand binding predictions."""
    return _api_client.predict_ligand_binding(list(smiles_list), list(names_list))


# UTILITY FUNCTIONS

def get_cache_hash(key: str) -> str:
    """Generate consistent hash for cache invalidation testing"""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def is_cache_valid(key: str, cache_manager: CacheManager) -> bool:
    """Check if custom SQLite cache entry exists and is valid"""
    return cache_manager.get(key) is not None