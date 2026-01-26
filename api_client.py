import httpx
import pandas as pd
import streamlit as st
import sqlite3
import json
import threading
import asyncio
from datetime import datetime
from typing import List, Dict
import os

# Load TSV data at startup
@st.cache_data
def load_hpa_data():
    """Load HPA data from TSV files at app startup"""
    data_dir = "data"
    
    # Load normal tissue data
    tissue_file = os.path.join(data_dir, "normal_tissue.tsv")
    normal_tissue_df = pd.read_csv(tissue_file, sep='\t') if os.path.exists(tissue_file) else pd.DataFrame()
    
    # Load subcellular location data
    subcellular_file = os.path.join(data_dir, "subcellular_location.tsv")
    subcellular_df = pd.read_csv(subcellular_file, sep='\t') if os.path.exists(subcellular_file) else pd.DataFrame()
    
    return normal_tissue_df, subcellular_df

# Thread-safe SQLite connection manager
class ThreadSafeSQLite:
    def __init__(self, db_path):
        self.db_path = db_path
        self.local = threading.local()

    def get_connection(self):
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.local.connection

    def close_connection(self):
        if hasattr(self.local, "connection"):
            self.local.connection.close()
            del self.local.connection

# Cache manager class
class CacheManager:
    """Handles caching of API responses"""
    
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self.conn = ThreadSafeSQLite(db_path).get_connection()
        self.create_tables()
        
    def create_tables(self):
        """Create tables for caching if they don't exist"""
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
    
    def get(self, key: str):
        """Get cached value by key"""
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None
    
    def set(self, key: str, value):
        """Set cache value by key"""
        with self.conn:
            self.conn.execute("""
            INSERT OR REPLACE INTO cache (key, value)
            VALUES (?, ?)
            """, (key, json.dumps(value)))
    
    def clear(self):
        """Clear the cache"""
        with self.conn:
            self.conn.execute("DELETE FROM cache")
    
    def close(self):
        """Close the database connection"""
        self.conn.close()

# api_client.py - UniProt and HPA API integration with error handling
class ProteinAPIClient:
    """Handles all API interactions with UniProt and Human Protein Atlas"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.uniprot_base = "https://rest.uniprot.org"
        self.hpa_base = "https://www.proteinatlas.org/api"
        
        # Replace the existing SQLite connection with the thread-safe connection
        self.db = ThreadSafeSQLite("omnibiomol_cache.db")
        self.conn = self.db.get_connection()
        
    async def search_uniprot(self, protein_name: str, max_results: int = 5) -> List[Dict]:
        """
        Search UniProt for protein by name
        Returns list of matches with UniProt ID, gene name, organism
        """
        cache_key = f"uniprot_search_{protein_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.uniprot_base}/uniprotkb/search"
                params = {
                    "query": f"{protein_name} AND (reviewed:true) AND (organism_id:9606)",
                    "format": "json",
                    "size": max_results,
                    "fields": "accession,gene_names,protein_name,organism_name,length"
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = []
                for entry in data.get("results", []):
                    results.append({
                        "uniprot_id": entry.get("primaryAccession"),
                        "gene_name": entry.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A"),
                        "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A"),
                        "organism": entry.get("organism", {}).get("scientificName", "N/A"),
                        "length": entry.get("sequence", {}).get("length", 0)
                    })
                
                self.cache.set(cache_key, results)
                return results
                
        except Exception as e:
            st.error(f"UniProt search failed: {str(e)}")
            return []
    
    async def fetch_uniprot_data(self, uniprot_id: str) -> Dict:
        """
        Fetch detailed protein data from UniProt
        Returns function summary and GO terms
        """
        cache_key = f"uniprot_data_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.uniprot_base}/uniprotkb/{uniprot_id}.json"
                
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Extract function
                function = ""
                for comment in data.get("comments", []):
                    if comment.get("commentType") == "FUNCTION":
                        function = comment.get("texts", [{}])[0].get("value", "")
                        break
                
                # Extract GO terms
                go_terms = {
                    "Biological Process": [],
                    "Molecular Function": [],
                    "Cellular Component": []
                }
                
                for xref in data.get("uniProtKBCrossReferences", []):
                    if xref.get("database") == "GO":
                        go_id = xref.get("id")
                        properties = xref.get("properties", [])
                        for prop in properties:
                            if prop.get("key") == "GoTerm":
                                term = prop.get("value")
                                # Parse term type (P:, F:, C:)
                                if term.startswith("P:"):
                                    go_terms["Biological Process"].append(term[2:])
                                elif term.startswith("F:"):
                                    go_terms["Molecular Function"].append(term[2:])
                                elif term.startswith("C:"):
                                    go_terms["Cellular Component"].append(term[2:])
                
                result = {
                    "uniprot_id": uniprot_id,
                    "function": function or "No functional annotation available",
                    "go_terms": go_terms,
                    "sequence_length": data.get("sequence", {}).get("length", 0),
                    "mass": data.get("sequence", {}).get("molWeight", 0)
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            st.error(f"UniProt data fetch failed: {str(e)}")
            return {
                "uniprot_id": uniprot_id,
                "function": "Error fetching data",
                "go_terms": {"Biological Process": [], "Molecular Function": [], "Cellular Component": []},
                "sequence_length": 0,
                "mass": 0
            }
    
    def get_tissue_expression(self, gene_name: str) -> pd.DataFrame:
        """
        Get tissue expression data from local TSV file
        Filters by gene name and returns formatted DataFrame
        """
        try:
            normal_tissue_df, _ = load_hpa_data()
            
            if normal_tissue_df.empty or gene_name not in normal_tissue_df['Gene name'].values:
                return pd.DataFrame(columns=["tissue", "level", "level_numeric"])
            
            # Filter by gene name
            filtered_df = normal_tissue_df[normal_tissue_df['Gene name'] == gene_name].copy()
            
            # Map level to numeric values
            level_map = {
                "High": 3,
                "Medium": 2,
                "Low": 1,
                "Not detected": 0
            }
            
            # Transform the data structure
            tissue_data = []
            for _, row in filtered_df.iterrows():
                level = row['Level']
                tissue_data.append({
                    "tissue": row['Tissue'],
                    "level": level,
                    "level_numeric": level_map.get(level, 0)
                })
            
            return pd.DataFrame(tissue_data)
            
        except Exception as e:
            st.error(f"Error loading tissue data: {str(e)}")
            return pd.DataFrame(columns=["tissue", "level", "level_numeric"])
    
    def get_subcellular_location(self, gene_name: str) -> pd.DataFrame:
        """
        Get subcellular location data from local TSV file
        Filters by gene name and returns formatted DataFrame
        """
        try:
            _, subcellular_df = load_hpa_data()
            
            if subcellular_df.empty or gene_name not in subcellular_df['Gene name'].values:
                return pd.DataFrame(columns=["location", "reliability", "reliability_numeric"])
            
            # Filter by gene name
            filtered_df = subcellular_df[subcellular_df['Gene name'] == gene_name].copy()
            
            # Map reliability to numeric values
            reliability_map = {
                "Enhanced": 3,
                "Supported": 2,
                "Approved": 1,
                "Uncertain": 0
            }
            
            # Transform the data structure
            location_data = []
            for _, row in filtered_df.iterrows():
                main_location = row['Main location']
                reliability = row['Reliability']
                
                # Split multiple locations
                if pd.notna(main_location):
                    locations = [loc.strip() for loc in str(main_location).split(';')]
                    for location in locations:
                        location_data.append({
                            "location": location,
                            "reliability": reliability,
                            "reliability_numeric": reliability_map.get(reliability, 0)
                        })
            
            return pd.DataFrame(location_data)
            
        except Exception as e:
            st.error(f"Error loading subcellular data: {str(e)}")
            return pd.DataFrame(columns=["location", "reliability", "reliability_numeric"])
    
    async def fetch_all_data(self, uniprot_id: str, gene_name: str) -> Dict:
        """
        Fetch all data - UniProt from API, HPA from local TSV files, structures and KEGG from APIs
        Returns combined dictionary with all protein information
        """
        # Run all API calls concurrently
        results = await asyncio.gather(
            self.fetch_uniprot_data(uniprot_id),
            self.fetch_alphafold_structure(uniprot_id),
            self.fetch_pdb_structure(uniprot_id),
            self.fetch_kegg_pathways(gene_name, uniprot_id),
            return_exceptions=True
        )
        
        # Get HPA data synchronously from local files
        tissue_expression = self.get_tissue_expression(gene_name)
        subcellular = self.get_subcellular_location(gene_name)
        
        return {
            "uniprot_data": results[0] if not isinstance(results[0], Exception) else {},
            "tissue_expression": tissue_expression,
            "subcellular": subcellular,
            "alphafold_structure": results[1] if not isinstance(results[1], Exception) else {"available": False},
            "pdb_structure": results[2] if not isinstance(results[2], Exception) else {"available": False, "structures": []},
            "kegg_pathways": results[3] if not isinstance(results[3], Exception) else {"available": False, "pathways": []}
        }
    
    async def fetch_alphafold_structure(self, uniprot_id: str) -> Dict:
        """
        Fetch AlphaFold predicted structure for a protein
        Uses the correct AlphaFold EBI API v1
        """
        cache_key = f"alphafold_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Correct AlphaFold API endpoint
                api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
                
                response = await client.get(api_url)
                response.raise_for_status()
                data = response.json()
                
                # AlphaFold returns a list with one entry
                if data and len(data) > 0:
                    entry = data[0]
                    
                    # Extract the entry ID (e.g., AF-P38398-F1)
                    entry_id = entry.get("entryId", f"AF-{uniprot_id}-F1")
                    
                    result = {
                        "available": True,
                        "uniprot_id": uniprot_id,
                        "entry_id": entry_id,
                        "pdb_url": f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v4.pdb",
                        "cif_url": f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v4.cif",
                        "pae_url": f"https://alphafold.ebi.ac.uk/files/{entry_id}-predicted_aligned_error_v4.json",
                        "alphafold_page": f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}",
                        "model_version": entry.get("latestVersion", 4),
                        "gene_name": entry.get("gene", ""),
                        "organism": entry.get("uniprotDescription", "")
                    }
                    
                    self.cache.set(cache_key, result)
                    return result
                else:
                    result = {"available": False, "uniprot_id": uniprot_id}
                    self.cache.set(cache_key, result)
                    return result
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                st.warning(f"No AlphaFold prediction available for {uniprot_id}")
                result = {"available": False, "uniprot_id": uniprot_id}
                self.cache.set(cache_key, result)
                return result
            else:
                st.error(f"AlphaFold API error: {str(e)}")
                return {"available": False, "uniprot_id": uniprot_id}
        except Exception as e:
            st.error(f"AlphaFold fetch error: {str(e)}")
            return {"available": False, "uniprot_id": uniprot_id}

    async def fetch_pdb_structure(self, uniprot_id: str) -> Dict:
        """
        Check if experimental structure exists in RCSB PDB
        Uses RCSB REST API for UniProt mapping
        """
        cache_key = f"pdb_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Correct RCSB PDB API endpoint for UniProt mapping
                api_url = "https://search.rcsb.org/rcsbsearch/v2/query"
                
                # Query JSON for searching by UniProt accession
                query = {
                    "query": {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                            "operator": "exact_match",
                            "value": uniprot_id
                        }
                    },
                    "return_type": "entry",
                    "request_options": {
                        "return_all_hits": True
                    }
                }
                
                response = await client.post(api_url, json=query)
                response.raise_for_status()
                data = response.json()
                
                pdb_structures = []
                
                # Extract PDB IDs from results
                if "result_set" in data:
                    for result in data["result_set"]:
                        pdb_id = result.get("identifier", "").upper()
                        if pdb_id:
                            # Fetch detailed info for each PDB entry
                            try:
                                detail_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                                detail_response = await client.get(detail_url)
                                detail_data = detail_response.json()
                                
                                # Extract method and resolution
                                exptl = detail_data.get("exptl", [{}])[0]
                                method = exptl.get("method", "Unknown")
                                
                                refine = detail_data.get("refine", [{}])[0] if detail_data.get("refine") else {}
                                resolution = refine.get("ls_d_res_high", "N/A")
                                
                                pdb_structures.append({
                                    "pdb_id": pdb_id,
                                    "pdb_url": f"https://files.rcsb.org/download/{pdb_id}.pdb",
                                    "rcsb_page": f"https://www.rcsb.org/structure/{pdb_id}",
                                    "method": method,
                                    "resolution": f"{resolution} Å" if resolution != "N/A" else "N/A"
                                })
                            except:
                                # If detail fetch fails, add basic info
                                pdb_structures.append({
                                    "pdb_id": pdb_id,
                                    "pdb_url": f"https://files.rcsb.org/download/{pdb_id}.pdb",
                                    "rcsb_page": f"https://www.rcsb.org/structure/{pdb_id}",
                                    "method": "Unknown",
                                    "resolution": "N/A"
                                })
                
                result = {
                    "available": len(pdb_structures) > 0,
                    "structures": pdb_structures,
                    "count": len(pdb_structures)
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            # Don't show error for PDB - it's optional
            # st.warning(f"PDB fetch error: {str(e)}")
            return {"available": False, "structures": [], "count": 0}

    async def fetch_kegg_pathways(self, gene_name: str, uniprot_id: str) -> Dict:
        """
        Fetch KEGG pathways for a protein
        Returns pathway information and links
        """
        cache_key = f"kegg_pathways_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Convert gene name to KEGG gene ID (hsa:XXXXX format)
                # First try to find the gene in KEGG using gene name
                find_url = f"https://rest.kegg.jp/find/genes/{gene_name}+human"
                response = await client.get(find_url)
                response.raise_for_status()
                gene_data = response.text.strip()
                # If not found, try using UniProt ID
                if not gene_data:
                    find_url_uniprot = f"https://rest.kegg.jp/conv/genes/uniprot:{uniprot_id}"
                    response_uniprot = await client.get(find_url_uniprot)
                    response_uniprot.raise_for_status()
                    gene_data = response_uniprot.text.strip()
                if not gene_data:
                    return {
                        "available": False,
                        "gene_name": gene_name,
                        "pathways": []
                    }
                # Extract the first matching KEGG gene ID
                lines = gene_data.split('\n')
                kegg_gene_id = None
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            # For gene name search, KEGG gene ID is in parts[0]
                            # For UniProt search, KEGG gene ID is in parts[1]
                            kegg_gene_id = parts[-1].strip()
                            break
                if not kegg_gene_id:
                    return {
                        "available": False,
                        "gene_name": gene_name,
                        "pathways": []
                    }
                
                # Step 2: Get pathways for this gene
                pathway_url = f"https://rest.kegg.jp/link/pathway/{kegg_gene_id}"
                
                pathway_response = await client.get(pathway_url)
                pathway_response.raise_for_status()
                
                pathway_data = pathway_response.text.strip()
                
                if not pathway_data:
                    return {
                        "available": False,
                        "gene_name": gene_name,
                        "kegg_gene_id": kegg_gene_id,
                        "pathways": []
                    }
                
                # Step 3: Parse pathway IDs and fetch details
                pathways = []
                pathway_lines = pathway_data.split('\n')
                
                for line in pathway_lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            pathway_id = parts[1].replace('path:', '')
                            
                            # Fetch pathway details
                            try:
                                detail_url = f"https://rest.kegg.jp/get/{pathway_id}"
                                detail_response = await client.get(detail_url)
                                detail_response.raise_for_status()
                                
                                detail_text = detail_response.text
                                
                                # Parse pathway name from the response
                                pathway_name = "Unknown Pathway"
                                pathway_class = ""
                                
                                for detail_line in detail_text.split('\n'):
                                    if detail_line.startswith('NAME'):
                                        pathway_name = detail_line.replace('NAME', '').strip()
                                        # Remove species suffix if present
                                        if ' - Homo sapiens' in pathway_name:
                                            pathway_name = pathway_name.replace(' - Homo sapiens', '')
                                    elif detail_line.startswith('CLASS'):
                                        pathway_class = detail_line.replace('CLASS', '').strip()
                                
                                pathways.append({
                                    "pathway_id": pathway_id,
                                    "pathway_name": pathway_name,
                                    "pathway_class": pathway_class,
                                    "kegg_url": f"https://www.kegg.jp/pathway/{pathway_id}",
                                    "kegg_image": f"https://www.kegg.jp/kegg/pathway/hsa/{pathway_id}.png",
                                    "highlight_url": f"https://www.kegg.jp/pathway/{pathway_id}+{kegg_gene_id}"
                                })
                                
                            except Exception as e:
                                # If detail fetch fails, add basic info
                                pathways.append({
                                    "pathway_id": pathway_id,
                                    "pathway_name": pathway_id.replace('hsa', 'Human pathway '),
                                    "pathway_class": "",
                                    "kegg_url": f"https://www.kegg.jp/pathway/{pathway_id}",
                                    "kegg_image": f"https://www.kegg.jp/kegg/pathway/hsa/{pathway_id}.png",
                                    "highlight_url": f"https://www.kegg.jp/pathway/{pathway_id}+{kegg_gene_id}"
                                })
                
                result = {
                    "available": len(pathways) > 0,
                    "gene_name": gene_name,
                    "kegg_gene_id": kegg_gene_id,
                    "pathways": pathways,
                    "pathway_count": len(pathways)
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "available": False,
                    "gene_name": gene_name,
                    "pathways": []
                }
            else:
                st.warning(f"KEGG API error: {str(e)}")
                return {
                    "available": False,
                    "gene_name": gene_name,
                    "pathways": []
                }
        except Exception as e:
            st.warning(f"KEGG fetch error: {str(e)}")
            return {
                "available": False,
                "gene_name": gene_name,
                "pathways": []
            }