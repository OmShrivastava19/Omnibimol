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
        Fetch all data - UniProt from API, HPA from local TSV files
        Returns combined dictionary with all protein information
        """
        # Fetch UniProt data asynchronously
        uniprot_data = await self.fetch_uniprot_data(uniprot_id)
        
        # Get HPA data synchronously from local files
        tissue_expression = self.get_tissue_expression(gene_name)
        subcellular = self.get_subcellular_location(gene_name)
        
        return {
            "uniprot_data": uniprot_data if not isinstance(uniprot_data, Exception) else {},
            "tissue_expression": tissue_expression,
            "subcellular": subcellular
        }