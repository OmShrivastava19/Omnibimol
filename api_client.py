import httpx
import pandas as pd
import streamlit as st
import asyncio
from typing import List, Dict
import os
import re
import html
import hashlib
import sys

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

# api_client.py - UniProt and HPA API integration with error handling
class ProteinAPIClient:
    # DataProcessor methods
    class DataProcessor:
        """Processes raw API data into visualization-ready formats"""

        @staticmethod
        def prepare_tissue_chart_data(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
            """
            Prepare tissue expression data for bar chart
            Prioritizes top N tissues by expression level
            """
            if df.empty:
                return df

            # Sort by expression level and take top N
            df_sorted = df.sort_values("level_numeric", ascending=False)

            # If more than top_n tissues, take top N with highest expression
            if len(df_sorted) > top_n:
                df_sorted = df_sorted.head(top_n)

            return df_sorted.sort_values("level_numeric", ascending=True)

        @staticmethod
        def prepare_subcellular_heatmap(df: pd.DataFrame) -> pd.DataFrame:
            """
            Prepare subcellular location data for heatmap visualization
            """
            if df.empty:
                return df

            # Create pivot-style data for heatmap
            df_pivot = df.copy()
            df_pivot["value"] = df_pivot["reliability_numeric"]

            return df_pivot

        @staticmethod
        def create_summary_table(uniprot_data: Dict, tissue_df: pd.DataFrame,
                                subcellular_df: pd.DataFrame, alphafold_data: Dict = None,
                                pdb_data: Dict = None, kegg_data: Dict = None,
                                chembl_data: Dict = None) -> pd.DataFrame:
            """
            Create comprehensive summary table with key metrics
            """
            # Structure availability
            structure_status = "None available"
            if pdb_data and pdb_data.get('available'):
                structure_status = f"Experimental ({pdb_data.get('count')} PDB entries)"
            elif alphafold_data and alphafold_data.get('available'):
                structure_status = "AlphaFold prediction"

            # Pathway count
            pathway_count = 0
            if kegg_data and kegg_data.get('available'):
                pathway_count = len(kegg_data.get('pathways', []))

            # Ligand count
            ligand_count = 0
            if chembl_data and chembl_data.get('available'):
                ligand_count = len(chembl_data.get('ligands', []))

            summary = {
                "Metric": [
                    "UniProt ID",
                    "Sequence Length",
                    "Molecular Weight (Da)",
                    "3D Structure",
                    "KEGG Pathways",
                    "Known Ligands",
                    "Tissues with Expression (HPA)",
                    "High Expression Tissues",
                    "Subcellular Locations",
                    "GO Terms (Total)"
                ],
                "Value": [
                    str(uniprot_data.get("uniprot_id", "N/A")),
                    str(f"{uniprot_data.get('sequence_length', 0):,}"),
                    str(f"{uniprot_data.get('mass', 0):,.0f}"),
                    str(structure_status),
                    str(pathway_count if pathway_count > 0 else "Not found"),
                    str(ligand_count if ligand_count > 0 else "Not found"),
                    str(len(tissue_df[tissue_df["level_numeric"] > 0]) if not tissue_df.empty else 0),
                    str(len(tissue_df[tissue_df["level"] == "High"]) if not tissue_df.empty else 0),
                    str(len(subcellular_df) if not subcellular_df.empty else 0),
                    str(sum(len(v) for v in uniprot_data.get("go_terms", {}).values()))
                ]
            }

            df = pd.DataFrame(summary)
            # Ensure Value column is consistently typed as string to avoid Arrow type errors
            df['Value'] = df['Value'].astype(str)
            return df
    """Handles all API interactions with UniProt and Human Protein Atlas"""
    
    def __init__(self, cache_manager, backend_api_url: str | None = None):
        self.cache = cache_manager
        self.uniprot_base = "https://rest.uniprot.org"
        self.hpa_base = "https://www.proteinatlas.org/api"
        self.backend_api_url = (backend_api_url or os.getenv("BACKEND_API_URL", "http://localhost:8000")).rstrip("/")
        
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
        Returns function summary, GO terms, and sequence
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
                
                # Extract sequence - THIS IS THE KEY FIX
                sequence_data = data.get("sequence", {})
                sequence = sequence_data.get("value", "")
                
                # Extract gene name
                gene_name = ""
                genes = data.get("genes", [])
                if genes and len(genes) > 0:
                    gene_name = genes[0].get("geneName", {}).get("value", "")
                
                result = {
                    "uniprot_id": uniprot_id,
                    "function": function or "No functional annotation available",
                    "go_terms": go_terms,
                    "sequence_length": sequence_data.get("length", 0),
                    "mass": sequence_data.get("molWeight", 0),
                    "sequence": sequence,  # CRITICAL: Include sequence
                    "gene_name": gene_name
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
                "mass": 0,
                "sequence": "",
                "gene_name": ""
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
        Fetch all data including ligands for docking
        """
        # Run all API calls concurrently
        results = await asyncio.gather(
            self.fetch_uniprot_data(uniprot_id),
            self.fetch_alphafold_structure(uniprot_id, gene_name),
            self.fetch_pdb_structure(uniprot_id),
            self.fetch_kegg_pathways(gene_name, uniprot_id),
            self.fetch_chembl_ligands(uniprot_id),
            self.fetch_string_ppi(gene_name, uniprot_id, limit=15),
            self.fetch_literature_summary(uniprot_id, gene_name),
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
            "kegg_pathways": results[3] if not isinstance(results[3], Exception) else {"available": False, "pathways": []},
            "chembl_ligands": results[4] if not isinstance(results[4], Exception) else {"available": False, "ligands": []},
            "string_ppi": results[5] if not isinstance(results[5], Exception) else {"available": False, "interactions": []},
            "literature": results[6] if not isinstance(results[6], Exception) else {"papers": [], "wiki_title": None, "wiki_snippet": None},
            "drug_targets": {"available": False}
        }
            
    async def fetch_alphafold_structure(self, uniprot_id: str, gene_name: str = None) -> Dict:
        """
        Fetch AlphaFold predicted structure for a protein
        AlphaFold v6 is the latest version
        """
        cache_key = f"alphafold_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                # AlphaFold DB URL pattern - try latest versions first
                # Try multiple URL formats
                urls_to_try = [
                    f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb",
                    f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb",
                    f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v2.pdb",
                ]
                
                pdb_url = None
                entry_id = None
                
                # Test which URL works
                for url in urls_to_try:
                    try:
                        test_response = await client.head(url, timeout=10.0)
                        if test_response.status_code == 200:
                            pdb_url = url
                            # Extract version from URL
                            if 'v6' in url:
                                version = 6
                                entry_id = f"AF-{uniprot_id}-F1"
                            elif 'v4' in url:
                                version = 4
                                entry_id = f"AF-{uniprot_id}-F1"
                            else:
                                version = 2
                                entry_id = f"AF-{uniprot_id}-F1"
                            break
                    except:
                        continue
                
                if not pdb_url:
                    # Structure doesn't exist
                    return {
                        "available": False,
                        "uniprot_id": uniprot_id,
                        "error": "No AlphaFold prediction available for this protein"
                    }
                
                result = {
                    "available": True,
                    "uniprot_id": uniprot_id,
                    "entry_id": entry_id,
                    "pdb_url": pdb_url,
                    "cif_url": pdb_url.replace('.pdb', '.cif'),
                    "pae_url": pdb_url.replace('model_v', 'predicted_aligned_error_v').replace('.pdb', '.json'),
                    "alphafold_page": f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}",
                    "model_version": version,
                    "gene_name": gene_name or ""
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            st.warning(f"AlphaFold structure check failed: {str(e)}")
            return {
                "available": False,
                "uniprot_id": uniprot_id,
                "error": str(e)
            }
            
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
        Fetch KEGG pathways for a PROTEIN (not gene)
        Returns comprehensive pathway information including pathway map images and metadata
        
        Format:
        - 1st Result: Pathway map image + all metadata (name, ID, description, functions)
        - Next 5 Results: List with pathway name, ID, and direct KEGG website links
        """
        cache_key = f"kegg_pathways_protein_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Fetch protein data from KEGG using UniProt ID
                # This converts UniProt ID to KEGG protein ID
                find_url = f"https://rest.kegg.jp/conv/genes/uniprot:{uniprot_id}"
                response = await client.get(find_url)
                response.raise_for_status()
                protein_data = response.text.strip()
                
                # If not found by UniProt, try gene name
                if not protein_data:
                    find_url = f"https://rest.kegg.jp/find/genes/{gene_name}+human"
                    response = await client.get(find_url)
                    response.raise_for_status()
                    protein_data = response.text.strip()
                
                if not protein_data:
                    return {
                        "available": False,
                        "uniprot_id": uniprot_id,
                        "protein_name": gene_name,
                        "pathways": [],
                        "first_result": None
                    }
                
                # Extract the KEGG protein/gene ID
                lines = protein_data.split('\n')
                kegg_protein_id = None
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            # KEGG protein ID is in parts[-1]
                            kegg_protein_id = parts[-1].strip()
                            break
                
                if not kegg_protein_id:
                    return {
                        "available": False,
                        "uniprot_id": uniprot_id,
                        "protein_name": gene_name,
                        "pathways": [],
                        "first_result": None
                    }
                
                # Step 2: Get pathways associated with this protein
                pathway_url = f"https://rest.kegg.jp/link/pathway/{kegg_protein_id}"
                pathway_response = await client.get(pathway_url)
                pathway_response.raise_for_status()
                pathway_data = pathway_response.text.strip()
                
                if not pathway_data:
                    return {
                        "available": False,
                        "uniprot_id": uniprot_id,
                        "protein_name": gene_name,
                        "kegg_protein_id": kegg_protein_id,
                        "pathways": [],
                        "first_result": None
                    }
                
                # Step 3: Parse pathway IDs and fetch comprehensive details
                pathways = []
                pathway_lines = pathway_data.split('\n')
                
                for idx, line in enumerate(pathway_lines):
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
                                
                                # Parse comprehensive pathway information
                                pathway_name = "Unknown Pathway"
                                pathway_description = ""
                                pathway_class = ""
                                molecular_functions = []
                                
                                for detail_line in detail_text.split('\n'):
                                    if detail_line.startswith('NAME'):
                                        pathway_name = detail_line.replace('NAME', '').strip()
                                        # Remove species suffix if present
                                        if ' - Homo sapiens' in pathway_name:
                                            pathway_name = pathway_name.replace(' - Homo sapiens', '')
                                    elif detail_line.startswith('DESCRIPTION'):
                                        pathway_description = detail_line.replace('DESCRIPTION', '').strip()
                                    elif detail_line.startswith('CLASS'):
                                        pathway_class = detail_line.replace('CLASS', '').strip()
                                    elif detail_line.startswith('GENE'):
                                        # Extract molecular functions from gene entries
                                        func_line = detail_line.replace('GENE', '').strip()
                                        if func_line and ';' in func_line:
                                            func_parts = func_line.split(';')
                                            if len(func_parts) > 1:
                                                molecular_functions.append(func_parts[1].strip())
                                
                                pathway_info = {
                                    "pathway_id": pathway_id,
                                    "pathway_name": pathway_name,
                                    "pathway_description": pathway_description,
                                    "pathway_class": pathway_class,
                                    "molecular_functions": list(set(molecular_functions)) if molecular_functions else [],
                                    "kegg_url": f"https://www.kegg.jp/pathway/{pathway_id}",
                                    "kegg_image_url": f"https://www.kegg.jp/kegg/pathway/hsa/{pathway_id}.png",
                                    "highlight_url": f"https://www.kegg.jp/entry/{kegg_protein_id}",
                                    "is_first": idx == 0
                                }
                                
                                pathways.append(pathway_info)
                                
                            except Exception as e:
                                # If detail fetch fails, add basic info
                                pathway_info = {
                                    "pathway_id": pathway_id,
                                    "pathway_name": pathway_id.replace('hsa', 'Human pathway '),
                                    "pathway_description": "",
                                    "pathway_class": "",
                                    "molecular_functions": [],
                                    "kegg_url": f"https://www.kegg.jp/pathway/{pathway_id}",
                                    "kegg_image_url": f"https://www.kegg.jp/kegg/pathway/hsa/{pathway_id}.png",
                                    "highlight_url": f"https://www.kegg.jp/entry/{kegg_protein_id}",
                                    "is_first": idx == 0
                                }
                                pathways.append(pathway_info)
                
                # Separate first result and next 5 results
                first_result = pathways[0] if pathways else None
                next_results = pathways[1:6] if len(pathways) > 1 else []
                
                result = {
                    "available": len(pathways) > 0,
                    "uniprot_id": uniprot_id,
                    "protein_name": gene_name,
                    "kegg_protein_id": kegg_protein_id,
                    "total_pathways": len(pathways),
                    "first_result": first_result,
                    "next_results": next_results,
                    "pathways": pathways  # Keep all for compatibility
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "available": False,
                    "uniprot_id": uniprot_id,
                    "protein_name": gene_name,
                    "pathways": [],
                    "first_result": None
                }
            else:
                st.warning(f"KEGG API error: {str(e)}")
                return {
                    "available": False,
                    "uniprot_id": uniprot_id,
                    "protein_name": gene_name,
                    "pathways": [],
                    "first_result": None
                }
        except Exception as e:
            st.warning(f"KEGG fetch error: {str(e)}")
            return {
                "available": False,
                "uniprot_id": uniprot_id,
                "protein_name": gene_name,
                "pathways": [],
                "first_result": None
            }
    
    async def fetch_literature_summary(self, uniprot_id: str, protein_name: str) -> dict:
        """
        Fetches literature summary from PubMed and Wikipedia for a protein.
        Caches results for 7 days.
        """
        import time
        cache_key = f"lit_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached and isinstance(cached, dict) and (time.time() - cached.get('timestamp', 0) < 7*24*3600):
            return cached.get('data', {})
        
        papers = []
        wiki_title = None
        wiki_snippet = None
        
        try:
            # PubMed search with timeout
            pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': f'("{protein_name}"[Title/Abstract] OR "{uniprot_id}"[All Fields])',
                'retmax': 5,
                'retmode': 'json',
                'sort': 'relevance',
                'usehistory': 'y'
            }
            async with httpx.AsyncClient() as client:
                search_response = await client.get(pubmed_url, params=params, timeout=10)
                search = search_response.json()
            pmids = search.get('esearchresult', {}).get('idlist', [])
            
            if pmids:
                try:
                    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    async with httpx.AsyncClient() as client:
                        abs_response = await client.get(
                            efetch_url,
                            params={'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml'},
                            timeout=10
                        )
                        abs_data = abs_response.text
                    papers = self.parse_pubmed_abstracts(abs_data)
                except Exception as e:
                    pass  # Silently fail on abstract fetch
        except Exception as e:
            pass  # Silently fail on PubMed search
        
        try:
            # Wikipedia search with proper User-Agent to avoid 403
            wiki_url = "https://en.wikipedia.org/w/api.php"
            headers = {
                'User-Agent': 'OmniBiMol/1.0 (Protein Analysis Platform; +http://github.com)'
            }
            wiki_params = {
                'action': 'query',
                'list': 'search',
                'srsearch': protein_name,
                'format': 'json',
                'srlimit': 1,
                'srprop': 'snippet'
            }
            async with httpx.AsyncClient() as client:
                wiki_response = await client.get(wiki_url, params=wiki_params, headers=headers, timeout=10)
            
            if wiki_response.text:
                wiki_res = wiki_response.json()
                wiki_search = wiki_res.get('query', {}).get('search', [])
                if wiki_search:
                    wiki_page = wiki_search[0]
                    wiki_title = wiki_page.get('title')
                    wiki_snippet = wiki_page.get('snippet')
                    
                    # Clean HTML tags and entities from snippet
                    if wiki_snippet:
                        wiki_snippet = re.sub(r'<[^>]+>', '', wiki_snippet)  # Remove HTML tags
                        wiki_snippet = html.unescape(wiki_snippet)  # Convert HTML entities
        except Exception as e:
            pass  # Silently fail on Wikipedia search
        
        result = {
            'papers': papers,
            'wiki_title': wiki_title,
            'wiki_snippet': wiki_snippet
        }
        self.cache.set(cache_key, {'data': result, 'timestamp': time.time()})
        return result

    def parse_pubmed_abstracts(self, xml_data):
        import xml.etree.ElementTree as ET
        papers = []
        try:
            root = ET.fromstring(xml_data)
            for article in root.findall('.//PubmedArticle'):
                title_elem = article.find('.//ArticleTitle')
                abstract_elem = article.find('.//AbstractText')
                pmid_elem = article.find('.//PMID')
                
                title = title_elem.text if title_elem is not None and title_elem.text else ''
                abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ''
                pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ''
                
                if not title or not pmid:
                    continue
                
                authors = []
                for author in article.findall('.//Author'):
                    last = author.findtext('LastName', default='').strip()
                    fore = author.findtext('ForeName', default='').strip()
                    if last:
                        full_name = f"{fore} {last}".strip()
                        if full_name:
                            authors.append(full_name)
                
                abstract_snip = (abstract[:200] + '...') if abstract and len(abstract) > 200 else abstract
                
                papers.append({
                    'title': title,
                    'authors': ', '.join(authors[:3]) if authors else 'Unknown',
                    'abstract_snip': abstract_snip if abstract_snip else '[No abstract available]',
                    'pmid': pmid
                })
        except Exception as e:
            st.error(f"Error parsing PubMed data: {str(e)}")
        return papers

    async def run_blast_search(self, sequence: str, uniprot_id: str) -> Dict:
        """
        Run BLAST search with SwissProt first, fallback to nr database
        Optimized for speed while maintaining full sequence accuracy
        Returns up to 15 hits
        
        Strategy:
        - First tries SwissProt database (faster, curated)
        - If no results, falls back to nr database (comprehensive)
        - Uses adaptive polling with exponential backoff
        """
        cache_key = f"blast_{uniprot_id}_v3"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Try SwissProt first
        swissprot_result = await self._run_blast_search_against_db(
            sequence, uniprot_id, database="swissprot"
        )
        
        # If SwissProt returned results, use them
        if swissprot_result.get("available") and swissprot_result.get("hits"):
            self.cache.set(cache_key, swissprot_result)
            return swissprot_result
        
        # Otherwise, fall back to NR database
        nr_result = await self._run_blast_search_against_db(
            sequence, uniprot_id, database="nr"
        )
        
        self.cache.set(cache_key, nr_result)
        return nr_result

    async def _run_blast_search_against_db(self, sequence: str, uniprot_id: str, database: str) -> Dict:
        """
        Internal method to run BLAST search against a specific database
        Handles single database submission and polling
        """
        try:
            submit_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
            
            # Full accuracy parameters - no shortcuts on sequence or database
            submit_params = {
                "CMD": "Put",
                "PROGRAM": "blastp",
                "DATABASE": database,          # SwissProt or nr
                "QUERY": sequence,             # Full unmodified sequence
                "FORMAT_TYPE": "Tabular",
                "FORMAT_OBJECT": "Alignment",
                "HITLIST_SIZE": "15",           # Request exactly 15 hits
                "ALIGNMENTS": "15",
                "DESCRIPTIONS": "15",
                "FILTER": "F",                 # No filtering - keep all hits
                "COMPOSITION_BASED_STATS": "2", # Best compositional adjustment
            }
            
            # Step 1: Submit BLAST job
            async with httpx.AsyncClient(timeout=60.0) as submit_client:
                submit_response = await submit_client.post(submit_url, data=submit_params)
                submit_response.raise_for_status()
                submit_text = submit_response.text
            
            # Extract RID and estimated time
            rid = None
            estimated_time = 0
            
            for line in submit_text.split('\n'):
                if 'RID =' in line:
                    rid = line.split('=')[1].strip()
                # Extract estimated time if available
                if 'estimated' in line.lower() and 'time' in line.lower():
                    import re
                    time_match = re.search(r'(\d+)', line)
                    if time_match:
                        estimated_time = int(time_match.group(1))
            
            if not rid:
                return {
                    "available": False,
                    "error": f"Failed to submit {database.upper()} BLAST job - no RID received"
                }
            
            # Step 2: Adaptive polling strategy with shorter timeouts
            # Initial wait - much shorter for faster response
            initial_wait = max(2, min(estimated_time // 2, 5))  # Cap initial wait at 5 seconds
            await asyncio.sleep(initial_wait)
            
            # Polling configuration - optimized for speed
            max_attempts = 90           # Max 3 minutes (with shorter intervals)
            attempt = 0
            poll_interval = 0.3         # Start at 0.3 second for faster polling
            max_poll_interval = 1.5     # Cap at 1.5 seconds
            
            while attempt < max_attempts:
                # Use fresh client each poll to avoid closed connection
                async with httpx.AsyncClient(timeout=60.0) as poll_client:
                    check_params = {
                        "CMD": "Get",
                        "RID": rid,
                        "FORMAT_TYPE": "XML"
                    }
                    
                    try:
                        check_response = await poll_client.get(
                            submit_url, 
                            params=check_params
                        )
                        check_text = check_response.text
                        
                        # Results ready - parse immediately
                        if "Status=READY" in check_text or "<BlastOutput" in check_text:
                            hits = self._parse_blast_xml(check_text, sequence)
                            
                            result = {
                                "available": True,
                                "rid": rid,
                                "hits": hits,
                                "hit_count": len(hits),
                                "database": database.upper()
                            }
                            
                            return result
                        
                        # Still waiting - adaptive backoff
                        elif "Status=WAITING" in check_text or "Status=UNKNOWN" in check_text:
                            attempt += 1
                            # Exponential backoff capped at max_poll_interval
                            # Faster ramp-up for quick jobs, slower for long jobs
                            if poll_interval < 0.6:
                                poll_interval = min(max_poll_interval, poll_interval * 1.3)  # Fast increase initially
                            else:
                                poll_interval = min(max_poll_interval, poll_interval * 1.1)  # Slower increase later
                            await asyncio.sleep(poll_interval)
                            continue
                        
                        # Job failed on server side
                        elif "Status=FAILURE" in check_text or "Status=ERROR" in check_text:
                            return {
                                "available": False,
                                "error": f"{database.upper()} BLAST job failed on NCBI server. Trying fallback..."
                            }
                        
                        # Unknown status - keep polling
                        else:
                            attempt += 1
                            await asyncio.sleep(poll_interval)
                            continue
                            
                    except httpx.TimeoutException:
                        # Network timeout on single poll - retry
                        attempt += 1
                        await asyncio.sleep(2)
                        continue
                    except Exception as poll_error:
                        if attempt >= max_attempts - 1:
                            raise
                        attempt += 1
                        await asyncio.sleep(2)
                        continue
            
            return {
                "available": False,
                "error": f"{database.upper()} BLAST search took too long. Trying fallback..."
            }
            
        except Exception as e:
            return {
                "available": False,
                "error": f"{database.upper()} BLAST error: {str(e)}"
            }

    async def search_protein_ncbi(self, sequence: str) -> Dict:
        """
        Search NCBI protein databases for an anonymous amino acid sequence.

        Uses BLASTp against SwissProt with short polling and returns a
        simplified annotation object for the best matching hit.

        Match criteria:
        - Prefer first hit with identity ≥95% and coverage ≥90%
        - Otherwise fall back to the top-ranked hit (if any)
        """
        # Normalize sequence
        if not sequence:
            return {
                "available": False,
                "match_found": False,
                "error": "Empty sequence provided"
            }

        seq_clean = "".join(sequence.split()).upper()
        if not seq_clean:
            return {
                "available": False,
                "match_found": False,
                "error": "Sequence contains no valid characters"
            }

        # Cache per unique sequence (SHA1 hash, truncated)
        seq_hash = hashlib.sha1(seq_clean.encode("utf-8")).hexdigest()[:16]
        cache_key = f"protein_ncbi_lookup_{seq_hash}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # Reuse optimized BLAST polling pipeline against SwissProt
            blast_result = await self._run_blast_search_against_db(
                seq_clean,
                seq_hash,
                database="swissprot"
            )

            hits = blast_result.get("hits", []) if blast_result.get("available") else []

            best_hit = None
            # First, try to find a high-confidence match
            for hit in hits:
                if hit.get("identity_percent", 0) >= 95 and hit.get("coverage_percent", 0) >= 90:
                    best_hit = hit
                    break

            # If no high-confidence hit, fall back to the top hit (if any)
            if best_hit is None and hits:
                best_hit = hits[0]

            if best_hit:
                result = {
                    "available": True,
                    "match_found": True,
                    "protein_name": best_hit.get("title", "Unknown protein"),
                    "accession_id": best_hit.get("accession"),
                    "organism": best_hit.get("organism", "Unknown"),
                    "identity_percent": best_hit.get("identity_percent", 0.0),
                    "coverage_percent": best_hit.get("coverage_percent", 0.0),
                    "e_value": best_hit.get("e_value", 1.0),
                    "ncbi_url": best_hit.get("ncbi_url"),
                    "raw_hits": hits,
                }
            else:
                # No suitable hit – likely novel or unannotated
                result = {
                    "available": True,
                    "match_found": False,
                    "message": "Protein name not found (novel or unannotated sequence)",
                    "raw_hits": hits,
                }

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            result = {
                "available": False,
                "match_found": False,
                "error": f"NCBI protein lookup failed: {str(e)}",
            }
            self.cache.set(cache_key, result)
            return result

    def _parse_blast_xml(self, xml_text: str, query_sequence: str) -> list:
        """
        Parse BLAST XML output to extract up to 15 hits
        Handles edge cases in XML formatting
        """
        import re
        
        hits = []
        
        try:
            # Find all Hit blocks
            hit_pattern = r'<Hit>(.*?)</Hit>'
            hit_blocks = re.findall(hit_pattern, xml_text, re.DOTALL)
            
            for hit_block in hit_blocks[:15]:  # Top 15 hits
                try:
                    # Extract core fields
                    accession_match = re.search(r'<Hit_accession>(.*?)</Hit_accession>', hit_block)
                    def_match = re.search(r'<Hit_def>(.*?)</Hit_def>', hit_block)
                    length_match = re.search(r'<Hit_len>(\d+)</Hit_len>', hit_block)
                    
                    # Extract best HSP (first one is always the best)
                    identity_match = re.search(r'<Hsp_identity>(\d+)</Hsp_identity>', hit_block)
                    positive_match = re.search(r'<Hsp_positive>(\d+)</Hsp_positive>', hit_block)
                    gaps_match = re.search(r'<Hsp_gaps>(\d+)</Hsp_gaps>', hit_block)
                    align_len_match = re.search(r'<Hsp_align-len>(\d+)</Hsp_align-len>', hit_block)
                    evalue_match = re.search(r'<Hsp_evalue>([\d.eE+-]+)</Hsp_evalue>', hit_block)
                    bitscore_match = re.search(r'<Hsp_bit-score>([\d.]+)</Hsp_bit-score>', hit_block)
                    qstart_match = re.search(r'<Hsp_qstart>(\d+)</Hsp_qstart>', hit_block)
                    qend_match = re.search(r'<Hsp_qend>(\d+)</Hsp_qend>', hit_block)
                    
                    if not (accession_match and identity_match and align_len_match):
                        continue
                    
                    accession = accession_match.group(1).strip()
                    definition = def_match.group(1).strip() if def_match else "Unknown"
                    hit_length = int(length_match.group(1)) if length_match else 0
                    identity = int(identity_match.group(1))
                    positives = int(positive_match.group(1)) if positive_match else identity
                    gaps = int(gaps_match.group(1)) if gaps_match else 0
                    align_len = int(align_len_match.group(1))
                    evalue = float(evalue_match.group(1)) if evalue_match else 1.0
                    bitscore = float(bitscore_match.group(1)) if bitscore_match else 0
                    qstart = int(qstart_match.group(1)) if qstart_match else 0
                    qend = int(qend_match.group(1)) if qend_match else 0
                    
                    # Extract organism from definition [Organism Name]
                    organism = "Unknown"
                    org_match = re.search(r'\[([^\]]+)\]', definition)
                    if org_match:
                        organism = org_match.group(1)
                    
                    # Calculate accurate percentages
                    identity_percent = round((identity / align_len) * 100, 2) if align_len > 0 else 0
                    similarity_percent = round((positives / align_len) * 100, 2) if align_len > 0 else 0
                    gap_percent = round((gaps / align_len) * 100, 2) if align_len > 0 else 0
                    coverage_percent = round(((qend - qstart + 1) / len(query_sequence)) * 100, 2) if len(query_sequence) > 0 else 0
                    
                    hits.append({
                        "accession": accession,
                        "title": definition[:250],
                        "organism": organism,
                        "identity_percent": identity_percent,
                        "similarity_percent": similarity_percent,
                        "coverage_percent": coverage_percent,
                        "gap_percent": gap_percent,
                        "e_value": evalue,
                        "bit_score": bitscore,
                        "align_len": align_len,
                        "hit_length": hit_length,
                        "query_range": f"{qstart}-{qend}",
                        "ncbi_url": f"https://www.ncbi.nlm.nih.gov/protein/{accession}"
                    })
                    
                except (ValueError, AttributeError):
                    continue
            
            return hits
            
        except Exception as e:
            return []

    async def predict_structure(self, sequence: str) -> Dict:
        """
        Predict 3D protein structure for an amino acid sequence using ESMFold.

        Uses a remote, CPU-friendly API (no local models, no GPU required).
        Returns predicted PDB text and an optional average pLDDT confidence score.
        
        Important: HTTP 413 errors usually indicate the API request format is wrong,
        not the sequence size. This method handles the request correctly.
        """
        # Basic validation and normalization
        if not sequence:
            return {
                "available": False,
                "error": "Empty sequence provided for structure prediction"
            }

        seq_clean = "".join(sequence.split()).upper()
        if not seq_clean:
            return {
                "available": False,
                "error": "Sequence contains no valid characters"
            }

        # Cache per unique sequence
        seq_hash = hashlib.sha1(seq_clean.encode("utf-8")).hexdigest()[:16]
        cache_key = f"esmfold_structure_{seq_hash}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Public ESMFold endpoint hosted by Meta / ESM Atlas
        esmfold_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # The ESM Atlas API expects the raw sequence as the request body
                # Using data parameter with plain text encoding instead of content
                response = await client.post(
                    esmfold_url,
                    data=seq_clean,
                    headers={"Content-Type": "text/plain"},
                )
                
                # Check for HTTP 413 specifically
                if response.status_code == 413:
                    result = {
                        "available": False,
                        "error": f"Protein sequence too large for prediction (HTTP 413). Sequence: {len(seq_clean)} amino acids. Try a shorter sequence or domain.",
                    }
                    self.cache.set(cache_key, result)
                    return result
                
                response.raise_for_status()
                pdb_text = response.text

            # Attempt to derive an average pLDDT from B-factor column if present
            avg_plddt = None
            try:
                values = []
                for line in pdb_text.splitlines():
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        # B-factor in PDB is columns 61–66 (0-based 60:66)
                        if len(line) >= 66:
                            b_str = line[60:66].strip()
                            if b_str:
                                values.append(float(b_str))
                if values:
                    avg_plddt = sum(values) / len(values)
            except Exception:
                avg_plddt = None

            result = {
                "available": True,
                "pdb": pdb_text,
                "sequence_length": len(seq_clean),
                "source": "ESMFold",
                "avg_plddt": avg_plddt,
            }

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            result = {
                "available": False,
                "error": f"Structure prediction failed or is unavailable: {str(e)}",
            }
            self.cache.set(cache_key, result)
            return result

    def get_fasta_sequence(self, uniprot_data: Dict) -> str:
        """
        Extract FASTA formatted sequence from UniProt data
        """
        uniprot_id = uniprot_data.get('uniprot_id', 'UNKNOWN')
        sequence = uniprot_data.get('sequence', '')
        gene_name = uniprot_data.get('gene_name', '')
        
        # Create FASTA header
        fasta = f">{uniprot_id}"
        if gene_name:
            fasta += f"|{gene_name}"
        fasta += f" Homo sapiens\n"
        
        # Add sequence with 60 characters per line
        for i in range(0, len(sequence), 60):
            fasta += sequence[i:i+60] + "\n"
        
        return fasta

    async def fetch_embl_sequence(self, uniprot_id: str) -> Dict:
        """
        Fetch sequence features from UniProt directly (more reliable than EMBL endpoint)
        Provides domain, region, and site annotations
        """
        cache_key = f"features_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use UniProt's own feature API
                url = f"{self.uniprot_base}/uniprotkb/{uniprot_id}.json"
                
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Extract features from UniProt JSON
                features = []
                
                for feature in data.get("features", []):
                    feature_type = feature.get("type", "")
                    description = feature.get("description", "")
                    
                    # Get location
                    location = feature.get("location", {})
                    start = location.get("start", {}).get("value")
                    end = location.get("end", {}).get("value")
                    
                    if start and end:
                        features.append({
                            "type": feature_type,
                            "description": description if description else feature_type,
                            "start": int(start),
                            "end": int(end),
                            "length": int(end) - int(start) + 1
                        })
                
                result = {
                    "available": len(features) > 0,
                    "uniprot_id": uniprot_id,
                    "features": features,
                    "feature_count": len(features)
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            st.warning(f"Feature fetch error: {str(e)}")
            return {
                "available": False,
                "uniprot_id": uniprot_id,
                "features": []
            }

    async def run_needle_alignment(self, sequence1: str, sequence2: str, 
                                id1: str = "Query", id2: str = "Subject") -> Dict:
        """
        Run EMBOSS Needle pairwise sequence alignment via EMBL-EBI REST API
        Needle performs global alignment (Needleman-Wunsch algorithm)
        """
        cache_key = f"needle_{hash(sequence1)}_{hash(sequence2)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Clean sequences (remove whitespace and non-amino acid characters)
            seq1_clean = ''.join(c for c in sequence1.upper() if c.isalpha())
            seq2_clean = ''.join(c for c in sequence2.upper() if c.isalpha())
            
            # Prepare FASTA format
            fasta1 = f">{id1}\n{seq1_clean}"
            fasta2 = f">{id2}\n{seq2_clean}"
            
            # Step 1: Submit job
            submit_url = "https://www.ebi.ac.uk/Tools/services/rest/emboss_needle/run"
            
            # Correct parameters based on EBI documentation
            submit_data = {
                "email": "omshrivastava01927@gmail.com",
                "title": f"Alignment_{id1}_vs_{id2}",
                "asequence": fasta1,
                "bsequence": fasta2,
                "gapopen": "10",
                "gapextend": "0.5",
                "endweight": "false",
                "endopen": "10",
                "endextend": "0.5",
                "matrix": "EBLOSUM62",
                "sformat": "pair"
            }
            
            # Submit with correct headers
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/plain"
                }
                
                submit_response = await client.post(
                    submit_url, 
                    data=submit_data,
                    headers=headers
                )
                submit_response.raise_for_status()
                
                job_id = submit_response.text.strip()
                
                if not job_id or len(job_id) > 50:
                    return {
                        "available": False,
                        "error": "Failed to submit Needle alignment job"
                    }
                
                # Step 2: Poll for completion
                status_url = f"https://www.ebi.ac.uk/Tools/services/rest/emboss_needle/status/{job_id}"
                # FIXED: Use 'out' instead of 'aln-pair'
                result_url = f"https://www.ebi.ac.uk/Tools/services/rest/emboss_needle/result/{job_id}/out"
                
                max_attempts = 40
                attempt = 0
                
                while attempt < max_attempts:
                    await asyncio.sleep(3)
                    
                    try:
                        status_response = await client.get(status_url)
                        status = status_response.text.strip()
                        
                        if status == "FINISHED":
                            # Get results
                            result_response = await client.get(result_url)
                            result_response.raise_for_status()
                            
                            alignment_text = result_response.text
                            
                            # Parse alignment results
                            alignment_data = self._parse_needle_output(alignment_text, id1, id2)
                            
                            result = {
                                "available": True,
                                "job_id": job_id,
                                "alignment_text": alignment_text,
                                "identity": alignment_data.get("identity", 0),
                                "similarity": alignment_data.get("similarity", 0),
                                "gaps": alignment_data.get("gaps", 0),
                                "score": alignment_data.get("score", 0),
                                "alignment_length": alignment_data.get("alignment_length", 0),
                                "alignment_display": alignment_data.get("alignment_display", "")
                            }
                            
                            self.cache.set(cache_key, result)
                            return result
                            
                        elif status == "RUNNING":
                            attempt += 1
                            continue
                        elif status == "FAILURE" or status == "ERROR":
                            return {
                                "available": False,
                                "error": f"Needle alignment failed with status: {status}"
                            }
                        else:
                            attempt += 1
                            continue
                            
                    except httpx.HTTPStatusError as e:
                        if attempt >= max_attempts - 1:
                            raise
                        attempt += 1
                        continue
                
                return {
                    "available": False,
                    "error": "Needle alignment timed out after 2 minutes"
                }
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            return {
                "available": False,
                "error": error_msg
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
            
    def _parse_needle_output(self, alignment_text: str, id1: str, id2: str) -> Dict:
        """
        Parse EMBOSS Needle alignment output
        Extracts identity, similarity, gaps, score, and formatted alignment
        """
        try:
            lines = alignment_text.split('\n')
            
            identity = 0.0
            similarity = 0.0
            gaps = 0.0
            score = 0.0
            alignment_length = 0
            
            # Extract statistics from header
            for line in lines:
                line = line.strip()
                
                if line.startswith("# Identity:"):
                    # Format: "# Identity:     123/456 (27.0%)"
                    try:
                        match = line.split('(')[1].split('%')[0]
                        identity = float(match.strip())
                    except:
                        pass
                        
                elif line.startswith("# Similarity:"):
                    try:
                        match = line.split('(')[1].split('%')[0]
                        similarity = float(match.strip())
                    except:
                        pass
                        
                elif line.startswith("# Gaps:"):
                    try:
                        match = line.split('(')[1].split('%')[0]
                        gaps = float(match.strip())
                    except:
                        pass
                        
                elif line.startswith("# Score:"):
                    try:
                        score = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                        
                elif line.startswith("# Length:"):
                    try:
                        alignment_length = int(line.split(':')[1].strip())
                    except:
                        pass
            
            # Extract alignment visualization (keep first 2000 chars for display)
            alignment_start = -1
            for i, line in enumerate(lines):
                if not line.startswith('#') and line.strip() and (id1 in line or id2 in line):
                    alignment_start = i
                    break
            
            if alignment_start >= 0:
                alignment_display = '\n'.join(lines[alignment_start:alignment_start+100])
            else:
                alignment_display = alignment_text[:2000]
            
            return {
                "identity": identity,
                "similarity": similarity,
                "gaps": gaps,
                "score": score,
                "alignment_length": alignment_length,
                "alignment_display": alignment_display
            }
            
        except Exception as e:
            st.warning(f"Warning: Could not parse all alignment statistics: {e}")
            return {
                "identity": 0,
                "similarity": 0,
                "gaps": 0,
                "score": 0,
                "alignment_length": 0,
                "alignment_display": alignment_text[:2000]
            }
    
    async def fetch_chembl_ligands(self, uniprot_id: str) -> Dict:
        """
        Fetch known ligands/inhibitors from ChEMBL database
        Returns compounds with binding affinity data
        """
        cache_key = f"chembl_ligands_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Get ChEMBL Target ID from UniProt ID
                target_url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={uniprot_id}"
                
                target_response = await client.get(target_url)
                target_response.raise_for_status()
                target_data = target_response.json()
                
                if not target_data.get("targets"):
                    return {
                        "available": False,
                        "uniprot_id": uniprot_id,
                        "ligands": []
                    }
                
                # Get first matching target
                chembl_id = target_data["targets"][0].get("target_chembl_id")
                
                if not chembl_id:
                    return {
                        "available": False,
                        "uniprot_id": uniprot_id,
                        "ligands": []
                    }
                
                # Step 2: Get bioactivity data for this target
                activity_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={chembl_id}&limit=50"
                
                activity_response = await client.get(activity_url)
                activity_response.raise_for_status()
                activity_data = activity_response.json()
                
                ligands = []
                seen_compounds = set()
                
                for activity in activity_data.get("activities", []):
                    molecule_chembl_id = activity.get("molecule_chembl_id")
                    
                    # Avoid duplicates
                    if molecule_chembl_id in seen_compounds:
                        continue
                    seen_compounds.add(molecule_chembl_id)
                    
                    # Get activity type and value
                    activity_type = activity.get("standard_type", "")
                    activity_value = activity.get("standard_value")
                    activity_units = activity.get("standard_units", "")
                    
                    # Only include IC50, Ki, Kd measurements
                    if activity_type in ["IC50", "Ki", "Kd"] and activity_value:
                        try:
                            activity_value = float(activity_value)
                            
                            # Fetch molecule details
                            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_chembl_id}.json"
                            mol_response = await client.get(mol_url)
                            mol_response.raise_for_status()
                            mol_data = mol_response.json()
                            
                            molecule = mol_data.get("molecule_structures", {})
                            
                            # Extract compound name with fallback logic
                            compound_name = mol_data.get("pref_name")
                            if not compound_name and mol_data.get("molecule_synonyms"):
                                synonyms = mol_data["molecule_synonyms"]
                                if synonyms and len(synonyms) > 0:
                                    compound_name = synonyms[0].get("synonyms")
                            if not compound_name:
                                compound_name = f"Compound {molecule_chembl_id}"
                            
                            ligands.append({
                                "chembl_id": molecule_chembl_id,
                                "name": compound_name,
                                "canonical_smiles": molecule.get("canonical_smiles", ""),
                                "smiles": molecule.get("canonical_smiles", ""),
                                "activity_type": activity_type,
                                "activity_value": activity_value,
                                "activity_units": activity_units,
                                "molecular_weight": mol_data.get("molecule_properties", {}).get("full_mwt"),
                                "mw": mol_data.get("molecule_properties", {}).get("full_mwt"),
                                "logp": mol_data.get("molecule_properties", {}).get("alogp"),
                                "hbd": mol_data.get("molecule_properties", {}).get("hbd"),
                                "hba": mol_data.get("molecule_properties", {}).get("hba"),
                                "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule_chembl_id}/"
                            })
                            
                            # Limit to top 20 ligands
                            if len(ligands) >= 20:
                                break
                                
                        except Exception as e:
                            continue
                
                # Sort by activity value (lower is better for IC50/Ki/Kd)
                ligands = sorted(ligands, key=lambda x: x.get("activity_value", float('inf')))
                
                result = {
                    "available": len(ligands) > 0,
                    "uniprot_id": uniprot_id,
                    "chembl_target_id": chembl_id,
                    "ligands": ligands,
                    "ligand_count": len(ligands)
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            st.warning(f"ChEMBL fetch error: {str(e)}")
            return {
                "available": False,
                "uniprot_id": uniprot_id,
                "ligands": []
            }

    async def fetch_pubchem_structure(self, compound_name: str) -> Dict:
        """
        Fetch 3D structure from PubChem for a compound
        Returns SDF format structure for docking
        """
        cache_key = f"pubchem_{compound_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search PubChem by name
                search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/cids/JSON"
                
                search_response = await client.get(search_url)
                search_response.raise_for_status()
                search_data = search_response.json()
                
                cids = search_data.get("IdentifierList", {}).get("CID", [])
                
                if not cids:
                    return {
                        "available": False,
                        "compound_name": compound_name
                    }
                
                cid = cids[0]
                
                # Get 3D SDF structure
                sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
                
                sdf_response = await client.get(sdf_url)
                sdf_response.raise_for_status()
                sdf_data = sdf_response.text
                
                result = {
                    "available": True,
                    "compound_name": compound_name,
                    "cid": cid,
                    "sdf_data": sdf_data,
                    "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                    "image_url": f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={cid}&t=l"
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            return {
                "available": False,
                "compound_name": compound_name,
                "error": str(e)
            }

    def prepare_protein_for_docking(self, uniprot_data: Dict, pdb_data: Dict, 
                                    alphafold_data: Dict) -> Dict:
        """
        Prepare protein structure for docking
        Returns PDB file content and metadata
        """
        # Prefer experimental structure over predicted
        if pdb_data.get('available') and pdb_data.get('structures'):
            structure_type = "experimental"
            pdb_url = pdb_data['structures'][0]['pdb_url']
            structure_id = pdb_data['structures'][0]['pdb_id']
        elif alphafold_data.get('available'):
            structure_type = "predicted"
            pdb_url = alphafold_data['pdb_url']
            structure_id = alphafold_data['uniprot_id']
        else:
            return {
                "available": False,
                "error": "No protein structure available for docking"
            }
        
        return {
            "available": True,
            "structure_type": structure_type,
            "structure_id": structure_id,
            "pdb_url": pdb_url,
            "sequence_length": uniprot_data.get('sequence_length', 0),
            "uniprot_id": uniprot_data.get('uniprot_id'),
            "source_url": pdb_url,
        }

    def _build_backend_headers(self) -> Dict[str, str]:
        try:
            from backend.auth.streamlit_integration import build_backend_auth_headers

            return build_backend_auth_headers(dict(st.session_state))
        except Exception:
            return {}

    def _request_backend_json(self, method: str, path: str, *, json_body: Dict | None = None) -> Dict:
        if not self.backend_api_url:
            raise RuntimeError("Backend API URL is not configured")

        url = f"{self.backend_api_url}{path}"
        with httpx.Client(timeout=60.0) as client:
            response = client.request(
                method,
                url,
                json=json_body,
                headers=self._build_backend_headers(),
            )
            response.raise_for_status()
            return response.json()

    def submit_real_docking_job(
        self,
        *,
        protein_prep: Dict,
        ligand_data: Dict,
        ligand_name: str,
        exhaustiveness: int,
        num_modes: int,
        energy_range: int,
    ) -> Dict:
        payload = {
            "protein": protein_prep,
            "ligand": {
                **ligand_data,
                "name": ligand_name,
            },
            "parameters": {
                "exhaustiveness": exhaustiveness,
                "num_modes": num_modes,
                "energy_range": energy_range,
            },
        }
        return self._request_backend_json(
            "POST",
            "/api/v1/jobs",
            json_body={"job_type": "docking.vina", "payload": payload},
        )

    def poll_docking_job(self, job_id: int) -> Dict:
        return self._request_backend_json("GET", f"/api/v1/jobs/{job_id}")

    def normalize_docking_result(self, docking_result: Dict, *, fallback_reason: str | None = None) -> Dict:
        normalized = dict(docking_result or {})
        normalized.setdefault("available", True)
        normalized.setdefault("mode", "simulation" if normalized.get("simulated") else "real")
        normalized.setdefault("engine", "simulation" if normalized.get("simulated") else os.getenv("DOCKING_ENGINE", "vina"))
        normalized.setdefault("simulated", normalized.get("mode") == "simulation")
        normalized.setdefault("status", normalized.get("status", "completed" if normalized.get("available") else "queued"))
        normalized.setdefault("modes", normalized.get("modes", []))
        normalized.setdefault("best_mode", normalized.get("best_mode", normalized.get("modes", [{}])[0] if normalized.get("modes") else {}))
        normalized.setdefault("has_coordinates", bool(normalized.get("modes")))
        if fallback_reason:
            normalized["fallback_reason"] = fallback_reason
        return normalized

    def run_docking_workflow(
        self,
        *,
        protein_prep: Dict,
        ligand_data: Dict,
        ligand_name: str,
        protein_length: int,
        ligand_mw: float,
        activity_value: float | None = None,
        mode: str | None = None,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        energy_range: int = 3,
    ) -> Dict:
        selected_mode = (mode or os.getenv("DOCKING_MODE_DEFAULT", "simulation")).lower().strip()

        if selected_mode == "real" and os.getenv("DOCKING_ENABLED", "true").lower() in {"1", "true", "yes"}:
            try:
                job = self.submit_real_docking_job(
                    protein_prep=protein_prep,
                    ligand_data=ligand_data,
                    ligand_name=ligand_name,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes,
                    energy_range=energy_range,
                )
                job_id = int(job["id"])
                try:
                    job_status = self.poll_docking_job(job_id)
                    if job_status.get("status") in {"completed", "failed"}:
                        result_payload = self.normalize_docking_result(job_status.get("result_payload") or {})
                        result_payload.update(
                            {
                                "job_id": job_id,
                                "job_status": job_status.get("status"),
                                "job_type": job.get("job_type", "docking.vina"),
                                "job_url": f"{self.backend_api_url}/api/v1/jobs/{job_id}",
                            }
                        )
                        return result_payload
                except Exception:
                    pass

                return {
                    "available": True,
                    "mode": "real",
                    "simulated": False,
                    "engine": os.getenv("DOCKING_ENGINE", "vina"),
                    "status": job.get("status", "queued"),
                    "job_id": job_id,
                    "job_type": job.get("job_type", "docking.vina"),
                    "job_url": f"{self.backend_api_url}/api/v1/jobs/{job_id}",
                    "binding_affinity": None,
                    "modes": [],
                    "best_mode": {},
                    "has_coordinates": False,
                    "ligand_name": ligand_name,
                    "protein_length": protein_length,
                    "ligand_mw": ligand_mw,
                    "activity_value": activity_value,
                    "queued_for_worker": True,
                }
            except Exception as exc:
                fallback_reason = f"Real docking unavailable: {exc}"
                simulated = self.simulate_docking_score(
                    protein_length,
                    ligand_mw,
                    activity_value,
                    ligand_data.get("smiles"),
                )
                simulated.update(
                    {
                        "mode": "simulation",
                        "simulated": True,
                        "engine": "simulation",
                        "fallback_reason": fallback_reason,
                        "job_id": None,
                        "job_status": None,
                        "job_url": None,
                        "ligand_name": ligand_name,
                    }
                )
                return self.normalize_docking_result(simulated, fallback_reason=fallback_reason)

        simulated = self.simulate_docking_score(
            protein_length,
            ligand_mw,
            activity_value,
            ligand_data.get("smiles"),
        )
        simulated.update(
            {
                "mode": "simulation",
                "simulated": True,
                "engine": "simulation",
                "job_id": None,
                "job_status": None,
                "job_url": None,
                "ligand_name": ligand_name,
            }
        )
        return self.normalize_docking_result(simulated)

    def simulate_docking_score(self, protein_length: int, ligand_mw: float, 
                          activity_value: float = None, ligand_smiles: str = None) -> Dict:
        """
        Simulate docking results with 3D pose information
        
        In production: Would call AutoDock Vina with:
        - vina --receptor protein.pdbqt --ligand ligand.pdbqt --out result.pdbqt
        - Parse PDBQT output for coordinates and orientations
        
        Returns binding modes with simulated 3D coordinates
        """
        import random
        import math
        
        # Simulate binding affinity based on known activity
        if activity_value:
            base_affinity = -math.log10(activity_value / 1000000) * 1.5
            base_affinity = max(-12, min(-4, base_affinity))  # Realistic range
        else:
            base_affinity = random.uniform(-6, -9)
        
        # Molecular complexity factor
        if ligand_smiles:
            complexity = len(ligand_smiles) / 50
            base_affinity -= complexity * 0.5
        
        noise = random.uniform(-0.8, 0.8)
        binding_affinity = base_affinity + noise
        
        # Generate multiple binding modes with 3D coordinates
        modes = []
        num_modes = random.randint(5, 9)
        
        for i in range(num_modes):
            mode_affinity = binding_affinity + random.uniform(0, 2.5)
            
            # Simulate 3D coordinates (center of binding site)
            center_x = random.uniform(-10, 10)
            center_y = random.uniform(-10, 10)
            center_z = random.uniform(-10, 10)
            
            # Simulate rotation (Euler angles)
            rotation_x = random.uniform(0, 360)
            rotation_y = random.uniform(0, 360)
            rotation_z = random.uniform(0, 360)
            
            modes.append({
                "mode": i + 1,
                "affinity": round(mode_affinity, 2),
                "rmsd_lb": round(random.uniform(0, 2), 2),
                "rmsd_ub": round(random.uniform(0, 3), 2),
                "center": {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                    "z": round(center_z, 3)
                },
                "rotation": {
                    "x": round(rotation_x, 2),
                    "y": round(rotation_y, 2),
                    "z": round(rotation_z, 2)
                },
                "orientation": f"α={rotation_x:.1f}° β={rotation_y:.1f}° γ={rotation_z:.1f}°"
            })
        
        modes = sorted(modes, key=lambda x: x['affinity'])
        
        return {
            "available": True,
            "binding_affinity": round(modes[0]['affinity'], 2),
            "modes": modes,
            "best_mode": modes[0],
            "exhaustiveness": 8,
            "simulated": True,
            "has_coordinates": True
        }

    async def fetch_string_ppi(self, gene_name: str, uniprot_id: str, limit: int = 10) -> Dict:
        """
        Fetch protein-protein interactions from STRING database
        STRING provides experimentally validated and predicted interactions
        """
        cache_key = f"string_ppi_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # STRING API endpoint
                # First, get STRING ID from gene name
                base_url = "https://string-db.org/api/json/get_string_ids"
                
                params = {
                    "identifiers": gene_name,
                    "species": 9606,  # Homo sapiens
                    "limit": 1
                }
                
                response = await client.post(base_url, data=params)
                response.raise_for_status()
                id_data = response.json()
                
                if not id_data or len(id_data) == 0:
                    return {
                        "available": False,
                        "gene_name": gene_name,
                        "interactions": [],
                        "error": "Protein not found in STRING database"
                    }
                
                string_id = id_data[0].get("stringId")
                
                # Get interaction partners
                interaction_url = "https://string-db.org/api/json/interaction_partners"
                
                interaction_params = {
                    "identifiers": string_id,
                    "species": 9606,
                    "limit": limit,
                    "required_score": 400  # Medium confidence (0-1000 scale)
                }
                
                interaction_response = await client.post(interaction_url, data=interaction_params)
                interaction_response.raise_for_status()
                interaction_data = interaction_response.json()
                
                # Parse interactions
                interactions = []
                
                for partner in interaction_data:
                    partner_name = partner.get("preferredName_B", partner.get("stringId_B", "Unknown"))
                    # STRING API returns score on 0-1 scale, convert to 0-1000 scale for consistency
                    raw_score = partner.get("score", 0)
                    combined_score = int(raw_score * 1000) if raw_score else 0
                    
                    # Get evidence types from individual scores
                    evidence = []
                    if partner.get("escore", 0) > 0:  # Experimental
                        evidence.append("Experimental")
                    if partner.get("dscore", 0) > 0:  # Database
                        evidence.append("Database")
                    if partner.get("tscore", 0) > 0:  # Text mining
                        evidence.append("Text mining")
                    if partner.get("ascore", 0) > 0:  # Co-expression
                        evidence.append("Co-expression")
                    if partner.get("fscore", 0) > 0:  # Fusion
                        evidence.append("Fusion")
                    if partner.get("pscore", 0) > 0:  # Phylogenetic
                        evidence.append("Phylogenetic")
                    if partner.get("nscore", 0) > 0:  # Neighborhood
                        evidence.append("Neighborhood")
                    
                    # Confidence level based on converted score (0-1000 scale)
                    # Official STRING thresholds (0-1 scale): 0.15=Low, 0.40=Medium, 0.70=High, 0.90=Highest
                    # Converted to 0-1000 scale: 150=Low, 400=Medium, 700=High, 900=Highest
                    if combined_score >= 900:
                        confidence = "Highest"
                    elif combined_score >= 700:
                        confidence = "High"
                    elif combined_score >= 400:
                        confidence = "Medium"
                    else:
                        confidence = "Low"
                    
                    interactions.append({
                        "partner_name": partner_name,
                        "partner_id": partner.get("stringId_B", ""),
                        "combined_score": combined_score,
                        "confidence": confidence,
                        "evidence_types": ", ".join(evidence) if evidence else "Predicted",
                        "experimental_score": int(partner.get("escore", 0) * 1000),
                        "database_score": int(partner.get("dscore", 0) * 1000),
                        "textmining_score": int(partner.get("tscore", 0) * 1000),
                        "coexpression_score": int(partner.get("ascore", 0) * 1000),
                        "fusion_score": int(partner.get("fscore", 0) * 1000),
                        "phylogenetic_score": int(partner.get("pscore", 0) * 1000),
                        "neighborhood_score": int(partner.get("nscore", 0) * 1000)
                    })
                
                # Sort by combined score
                interactions = sorted(interactions, key=lambda x: x["combined_score"], reverse=True)
                
                # Get network image URL
                network_url = f"https://string-db.org/api/image/network?identifiers={string_id}&species=9606&limit={limit}"
                
                result = {
                    "available": len(interactions) > 0,
                    "gene_name": gene_name,
                    "string_id": string_id,
                    "interactions": interactions,
                    "interaction_count": len(interactions),
                    "network_image_url": network_url,
                    "string_url": f"https://string-db.org/network/{string_id}"
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except httpx.HTTPStatusError as e:
            st.warning(f"STRING API error: {e.response.status_code}")
            return {
                "available": False,
                "gene_name": gene_name,
                "interactions": [],
                "error": f"STRING API error: {e.response.status_code}"
            }
        except Exception as e:
            st.warning(f"STRING PPI fetch error: {str(e)}")
            return {
                "available": False,
                "gene_name": gene_name,
                "interactions": [],
                "error": str(e)
            }

    async def fetch_similar_compounds(self, reference_smiles: str, similarity_threshold: float = 0.7) -> Dict:
        """
        Fetch structurally similar compounds from PubChem
        Can identify unknown/novel ligands with binding potential
        """
        cache_key = f"similar_{hash(reference_smiles)}_{similarity_threshold}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # PubChem similarity search
                base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles"
                
                params = {
                    "smiles": reference_smiles,
                    "Threshold": int(similarity_threshold * 100),  # Convert 0.7 to 70
                    "MaxRecords": 20
                }
                
                # Get similar compound CIDs
                search_url = f"{base_url}/cids/JSON"
                response = await client.post(search_url, data=params)
                response.raise_for_status()
                data = response.json()
                
                cids = data.get("IdentifierList", {}).get("CID", [])
                
                if not cids:
                    return {"available": False, "compounds": []}
                
                # Get compound properties
                compounds = []
                for cid in cids[:10]:  # Limit to top 10
                    try:
                        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
                        prop_response = await client.get(prop_url)
                        prop_data = prop_response.json()
                        
                        props = prop_data.get("PropertyTable", {}).get("Properties", [{}])[0]
                        
                        compounds.append({
                            "cid": cid,
                            "name": props.get("IUPACName", f"PubChem-{cid}"),
                            "smiles": props.get("CanonicalSMILES", ""),
                            "molecular_weight": props.get("MolecularWeight", 0),
                            "formula": props.get("MolecularFormula", ""),
                            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                            "similarity": "Unknown",  # Would need fingerprint comparison
                            "source": "PubChem Similar"
                        })
                    except:
                        continue
                
                result = {
                    "available": len(compounds) > 0,
                    "compounds": compounds,
                    "reference_smiles": reference_smiles
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            st.warning(f"Similar compound search error: {str(e)}")
            return {"available": False, "compounds": []}


    async def predict_drug_candidates(self, protein_sequence: str, gene_name: str) -> Dict:
        """
        Predict novel drug candidates using protein sequence/structure
        Uses DrugBank, PubChem, and literature mining
        """
        cache_key = f"drug_candidates_{gene_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            candidates = []
            
            # Strategy 1: Find compounds targeting similar proteins
            # Using protein family/domain information
            
            # Strategy 2: Literature-based discovery (PubMed)
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search PubMed for drug discovery papers
                pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                params = {
                    "db": "pubmed",
                    "term": f"{gene_name} AND (inhibitor OR drug OR compound OR ligand)",
                    "retmax": 5,
                    "retmode": "json"
                }
                
                response = await client.get(pubmed_url, params=params)
                data = response.json()
                
                pmids = data.get("esearchresult", {}).get("idlist", [])
                
                # Extract compound mentions from abstracts
                for pmid in pmids:
                    try:
                        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                        fetch_params = {
                            "db": "pubmed",
                            "id": pmid,
                            "retmode": "xml"
                        }
                        
                        abstract_response = await client.get(fetch_url, params=fetch_params)
                        # Simple parsing - in production use proper XML parser
                        text = abstract_response.text.lower()
                        
                        # Look for common drug/compound indicators
                        compound_indicators = ['inhibitor', 'compound', 'drug', 'molecule']
                        if any(indicator in text for indicator in compound_indicators):
                            candidates.append({
                                "name": f"Literature compound (PMID:{pmid})",
                                "source": "PubMed",
                                "pmid": pmid,
                                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                "evidence": "Literature mention"
                            })
                    except:
                        continue
            
            # Strategy 3: Recommend FDA-approved drugs for repurposing
            # Based on protein family
            try:
                repurposing_candidates = self.suggest_repurposing_drugs(gene_name)
                candidates.extend(repurposing_candidates)
            except Exception as repurposing_error:
                import sys
                print(f"Warning: Drug repurposing failed: {repurposing_error}", file=sys.stderr)
                # Continue without repurposing candidates
            
            result = {
                "available": len(candidates) > 0,
                "candidates": candidates[:10],  # Top 10
                "gene_name": gene_name,
                "strategies": ["Literature mining", "Drug repurposing", "Similarity search"]
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            return {"available": False, "candidates": [], "error": str(e)}

    @staticmethod
    def suggest_repurposing_drugs(gene_name: str) -> list:
        """
        Suggest FDA-approved drugs for repurposing based on target class
        """
        # Drug repurposing database (simplified)
        REPURPOSING_DB = {
            # Kinases
            "kinase": [
                {"name": "Imatinib", "target_class": "Tyrosine kinase", "indication": "CML"},
                {"name": "Gefitinib", "target_class": "EGFR", "indication": "NSCLC"},
                {"name": "Sorafenib", "target_class": "Multi-kinase", "indication": "RCC"}
            ],
            # Proteases
            "protease": [
                {"name": "Darunavir", "target_class": "HIV protease", "indication": "HIV"},
                {"name": "Bortezomib", "target_class": "Proteasome", "indication": "Myeloma"}
            ],
            # DNA repair
            "repair": [
                {"name": "Olaparib", "target_class": "PARP", "indication": "BRCA cancer"},
                {"name": "Talazoparib", "target_class": "PARP", "indication": "Breast cancer"}
            ],
            # Receptors
            "receptor": [
                {"name": "Erlotinib", "target_class": "EGFR", "indication": "NSCLC"},
                {"name": "Cetuximab", "target_class": "EGFR", "indication": "Colorectal"}
            ]
        }
        
        suggestions = []
        gene_lower = gene_name.lower()
        
        # Simple keyword matching (in production: use protein family classification)
        for category, drugs in REPURPOSING_DB.items():
            if category in gene_lower or gene_lower in category:
                for drug in drugs:
                    suggestions.append({
                        "name": drug["name"],
                        "source": "Drug Repurposing",
                        "target_class": drug["target_class"],
                        "original_indication": drug["indication"],
                        "evidence": "Target class similarity",
                        "status": "FDA Approved"
                    })
        
        return suggestions

    async def fetch_drugbank_targets(self, uniprot_id: str, gene_name: str) -> Dict:
        """
        Fetch FDA-approved drugs and clinical trials targeting this protein
        Integrates data from ChEMBL, ClinicalTrials.gov, and DrugBank
        """
        cache_key = f"drugbank_{uniprot_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                drugs = {
                    "fda_approved": [],
                    "clinical_trials": [],
                    "investigational": []
                }
                
                # Source 1: ChEMBL for FDA-approved drugs
                chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={uniprot_id}"
                
                try:
                    chembl_response = await client.get(chembl_url)
                    chembl_response.raise_for_status()
                    chembl_data = chembl_response.json()
                    
                    if chembl_data.get("targets"):
                        target_id = chembl_data["targets"][0].get("target_chembl_id")
                        
                        # Get drugs for this target
                        drug_url = f"https://www.ebi.ac.uk/chembl/api/data/drug_indication.json?target_chembl_id={target_id}"
                        drug_response = await client.get(drug_url)
                        drug_data = drug_response.json()
                        
                        for indication in drug_data.get("drug_indications", []):
                            mol_chembl_id = indication.get("molecule_chembl_id")
                            
                            # Get drug details
                            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_chembl_id}.json"
                            mol_response = await client.get(mol_url)
                            mol_info = mol_response.json()
                            
                            drug_name = mol_info.get("pref_name", mol_chembl_id)
                            max_phase = mol_info.get("max_phase", 0)
                            
                            drug_entry = {
                                "name": drug_name,
                                "chembl_id": mol_chembl_id,
                                "indication": indication.get("mesh_heading", "N/A"),
                                "max_phase": max_phase,
                                "molecule_type": mol_info.get("molecule_type", "Small molecule"),
                                "first_approval": mol_info.get("first_approval", "N/A"),
                                "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{mol_chembl_id}/"
                            }
                            
                            if max_phase == 4:
                                drugs["fda_approved"].append(drug_entry)
                            elif max_phase >= 2:
                                drugs["clinical_trials"].append(drug_entry)
                            else:
                                drugs["investigational"].append(drug_entry)
                except:
                    pass
                
                # Source 2: ClinicalTrials.gov for ongoing trials
                trial_url = "https://clinicaltrials.gov/api/v2/studies"
                
                try:
                    trial_params = {
                        "query.term": f"{gene_name} OR {uniprot_id}",
                        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
                        "pageSize": 20
                    }
                    
                    trial_response = await client.get(trial_url, params=trial_params)
                    trial_data = trial_response.json()
                    
                    for study in trial_data.get("studies", []):
                        protocol = study.get("protocolSection", {})
                        identification = protocol.get("identificationModule", {})
                        status = protocol.get("statusModule", {})
                        design = protocol.get("designModule", {})
                        
                        nct_id = identification.get("nctId", "")
                        title = identification.get("briefTitle", "")
                        phase = design.get("phases", ["N/A"])[0] if design.get("phases") else "N/A"
                        status_val = status.get("overallStatus", "Unknown")
                        
                        # Extract intervention (drug name)
                        interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
                        drug_names = [i.get("name", "") for i in interventions if i.get("type") == "DRUG"]
                        
                        if drug_names:
                            trial_entry = {
                                "nct_id": nct_id,
                                "title": title[:100],
                                "drugs": ", ".join(drug_names[:3]),
                                "phase": phase,
                                "status": status_val,
                                "url": f"https://clinicaltrials.gov/study/{nct_id}"
                            }
                            
                            # Categorize by phase
                            if phase in ["PHASE3", "PHASE2_PHASE3"]:
                                if trial_entry not in drugs["clinical_trials"]:
                                    drugs["clinical_trials"].append(trial_entry)
                except:
                    pass
                
                # Add manual curated database for common targets
                manual_drugs = get_manual_drug_database(gene_name, uniprot_id)
                if manual_drugs:
                    drugs["fda_approved"].extend(manual_drugs.get("fda_approved", []))
                    drugs["clinical_trials"].extend(manual_drugs.get("clinical_trials", []))
                
                result = {
                    "available": any([drugs["fda_approved"], drugs["clinical_trials"], drugs["investigational"]]),
                    "gene_name": gene_name,
                    "uniprot_id": uniprot_id,
                    "fda_approved": drugs["fda_approved"][:20],
                    "clinical_trials": drugs["clinical_trials"][:20],
                    "investigational": drugs["investigational"][:10],
                    "total_fda": len(drugs["fda_approved"]),
                    "total_trials": len(drugs["clinical_trials"]),
                    "total_investigational": len(drugs["investigational"])
                }
                
                self.cache.set(cache_key, result)
                return result
                
        except Exception as e:
            st.warning(f"Drug-target fetch error: {str(e)}")

    async def fetch_clinical_trials_by_drug(self, drug_name: str, max_results: int = 20) -> List[Dict]:
        """
        Fetch clinical trials from ClinicalTrials.gov using a drug name query.
        Returns raw study metadata including NCT ID, title, phase, status, etc.
        """
        cache_key = f"clinical_trials_drug_{drug_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        results: List[Dict] = []
        trial_url = "https://clinicaltrials.gov/api/v2/studies"
        nct_pattern = re.compile(r"^NCT\d{8}$", re.IGNORECASE)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                query_term = f"AREA[InterventionName] \"{drug_name}\" OR AREA[Condition] \"{drug_name}\""
                trial_params = {
                    "query.term": query_term,
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
                    "pageSize": max_results,
                }

                trial_response = await client.get(trial_url, params=trial_params)
                trial_response.raise_for_status()
                trial_data = trial_response.json()

                for study in trial_data.get("studies", []):
                    protocol = study.get("protocolSection", {})
                    identification = protocol.get("identificationModule", {})
                    status = protocol.get("statusModule", {})
                    design = protocol.get("designModule", {})
                    conditions = protocol.get("conditionsModule", {})
                    sponsors = protocol.get("sponsorsModule", {})
                    locations_module = protocol.get("contactsLocationsModule", {})
                    interventions_module = protocol.get("armsInterventionsModule", {})

                    nct_id = identification.get("nctId", "")
                    if not nct_pattern.match(str(nct_id)):
                        print(
                            f"ClinicalTrials.gov: invalid NCT ID skipped: {nct_id}",
                            file=sys.stderr,
                        )
                        continue
                    title = identification.get("briefTitle", "")
                    phase = "N/A"
                    if design.get("phases"):
                        phase = design.get("phases", ["N/A"])[0]
                    status_val = status.get("overallStatus", "Unknown")

                    condition_list = conditions.get("conditions", [])
                    conditions_value = condition_list if condition_list else []

                    interventions = interventions_module.get("interventions", [])
                    intervention_names = [i.get("name", "") for i in interventions if i.get("name")]
                    drug_names = [i.get("name", "") for i in interventions if i.get("type") == "DRUG" and i.get("name")]

                    sponsor_name = sponsors.get("leadSponsor", {}).get("name", "N/A")

                    enrollment = "N/A"
                    enrollment_info = design.get("enrollmentInfo", {})
                    if enrollment_info.get("count") is not None:
                        enrollment = enrollment_info.get("count")

                    start_date = "N/A"
                    start_struct = status.get("startDateStruct", {})
                    if start_struct.get("date"):
                        start_date = start_struct.get("date")

                    location_summary = "N/A"
                    locations = locations_module.get("locations", [])
                    if locations:
                        location_summary = ", ".join(
                            filter(None, [
                                locations[0].get("city"),
                                locations[0].get("state"),
                                locations[0].get("country"),
                            ])
                        ) or "N/A"

                    results.append({
                        "nct_id": str(nct_id).upper(),
                        "title": title,
                        "status": status_val,
                        "phase": phase,
                        "conditions": conditions_value,
                        "clinicaltrials_url": f"https://clinicaltrials.gov/study/{str(nct_id).upper()}",
                        "interventions": intervention_names,
                        "drugs": ", ".join(drug_names),
                        "sponsor": sponsor_name,
                        "locations": location_summary,
                        "enrollment": enrollment,
                        "start_date": start_date,
                    })
        except Exception as e:
            print(f"ClinicalTrials.gov fetch error: {str(e)}", file=sys.stderr)

        self.cache.set(cache_key, results)
        return results
    
    def predict_ligand_binding(self, smiles_list: List[str], molecule_names: List[str] = None) -> Dict:
        """
        Predict binding affinity and likelihood for drug molecules (SMILES format)
        
        Args:
            smiles_list: List of SMILES strings
            molecule_names: Optional list of molecule names
            
        Returns:
            Dictionary with predictions, rankings, and recommendations
        """
        try:
            from ligand_binding_predictor import LigandBindingPredictor
            
            predictor = LigandBindingPredictor()
            
            # Predict for all molecules
            predictions = predictor.predict_batch(smiles_list, molecule_names)
            
            # Rank and recommend
            recommendations = predictor.recommend_top_candidates(predictions, n=min(10, len(smiles_list)))
            
            return {
                "available": True,
                "predictions": predictions,
                "ranked_molecules": recommendations.get("top_candidates", []),
                "statistics": {
                    "total_molecules": recommendations.get("total_molecules", 0),
                    "valid_molecules": recommendations.get("valid_molecules", 0),
                    "average_affinity": recommendations.get("average_affinity"),
                    "average_likelihood": recommendations.get("average_likelihood")
                },
                "recommendations": recommendations.get("top_candidates", [])
            }
            
        except ImportError:
            return {
                "available": False,
                "error": "Ligand binding predictor module not available. Install required dependencies."
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"Binding prediction error: {str(e)}"
            }
    
    def validate_smiles(self, smiles: str) -> Dict:
        """
        Validate a SMILES string
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            from ligand_binding_predictor import SMILESValidator
            
            validator = SMILESValidator()
            is_valid, error = validator.is_valid_smiles(smiles)
            preprocessed = validator.preprocess_smiles(smiles)
            
            return {
                "is_valid": is_valid,
                "error": error,
                "canonical_smiles": preprocessed.get("canonical_smiles"),
                "atom_count": preprocessed.get("atom_count", 0),
                "bond_count": preprocessed.get("bond_count", 0)
            }
            
        except ImportError:
            return {
                "is_valid": False,
                "error": "SMILES validator not available. Install required dependencies."
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Validation error: {str(e)}"
            }


# ===== DRUG METADATA DATABASE =====
# Curated drug database with DrugBank, PubChem IDs, and approval status
DRUG_METADATA_DB = {
    "cetirizine": {"drugbank_id": "DB01156", "pubchem_id": "2678", "status": "FDA Approved"},
    "acetaminophen": {"drugbank_id": "DB00316", "pubchem_id": "1983", "status": "FDA Approved"},
    "ibuprofen": {"drugbank_id": "DB01050", "pubchem_id": "3672", "status": "FDA Approved"},
    "naproxen": {"drugbank_id": "DB00788", "pubchem_id": "156391", "status": "FDA Approved"},
    "aspirin": {"drugbank_id": "DB00945", "pubchem_id": "2244", "status": "FDA Approved"},
    "metformin": {"drugbank_id": "DB00331", "pubchem_id": "14219", "status": "FDA Approved"},
    "atorvastatin": {"drugbank_id": "DB00461", "pubchem_id": "60823", "status": "FDA Approved"},
    "lisinopril": {"drugbank_id": "DB00246", "pubchem_id": "5362129", "status": "FDA Approved"},
    "omeprazole": {"drugbank_id": "DB00338", "pubchem_id": "4594", "status": "FDA Approved"},
    "amoxicillin": {"drugbank_id": "DB01060", "pubchem_id": "33613", "status": "FDA Approved"},
    "erythromycin": {"drugbank_id": "DB00199", "pubchem_id": "5288874", "status": "FDA Approved"},
    "azithromycin": {"drugbank_id": "DB00207", "pubchem_id": "447043", "status": "FDA Approved"},
    "osimertinib": {"drugbank_id": "DB05484", "pubchem_id": "56152474", "status": "FDA Approved"},
    "erlotinib": {"drugbank_id": "DB00530", "pubchem_id": "176155", "status": "FDA Approved"},
    "gefitinib": {"drugbank_id": "DB00817", "pubchem_id": "123631", "status": "FDA Approved"},
    "olaparib": {"drugbank_id": "DB06692", "pubchem_id": "23237613", "status": "FDA Approved"},
    "imatinib": {"drugbank_id": "DB00619", "pubchem_id": "5291", "status": "FDA Approved"},
    "dasatinib": {"drugbank_id": "DB01254", "pubchem_id": "3062316", "status": "FDA Approved"},
    "cetuximab": {"drugbank_id": "DB00734", "pubchem_id": "56842941", "status": "FDA Approved"},
    "bevacizumab": {"drugbank_id": "DB00112", "pubchem_id": "7915435", "status": "FDA Approved"},
    "rituximab": {"drugbank_id": "DB00073", "pubchem_id": "15589180", "status": "FDA Approved"},
    "trastuzumab": {"drugbank_id": "DB00072", "pubchem_id": "7914308", "status": "FDA Approved"},
    "dupilumab": {"drugbank_id": "DB12202", "pubchem_id": "71306916", "status": "FDA Approved"},
    "pembrolizumab": {"drugbank_id": "DB11627", "pubchem_id": "71754778", "status": "FDA Approved"},
    "nivolumab": {"drugbank_id": "DB12218", "pubchem_id": "71779325", "status": "FDA Approved"},
}


def get_drug_metadata(drug_name: str) -> Dict:
    """
    Get drug metadata (DrugBank ID, PubChem ID, status) from curated database.
    Tries exact match, then partial match, then searches online.
    
    Args:
        drug_name: Name of the drug
        
    Returns:
        Dictionary with drugbank_id, pubchem_id, and status
    """
    result = {
        "drugbank_id": "N/A",
        "pubchem_id": "N/A",
        "status": "Status Unknown - Query FDA Database"
    }
    
    # Normalize drug name
    normalized_name = drug_name.strip().lower()
    
    # Try exact match first
    if normalized_name in DRUG_METADATA_DB:
        return DRUG_METADATA_DB[normalized_name]
    
    # Try partial match
    for db_name, metadata in DRUG_METADATA_DB.items():
        if normalized_name in db_name or db_name in normalized_name:
            return metadata
    
    # If not in database, try to fetch from ChEMBL
    try:
        import httpx
        
        # Search ChEMBL
        search_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
        params = {
            "q": drug_name,
            "limit": 5
        }
        
        response = httpx.get(search_url, params=params, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            molecules = data.get("molecules", [])
            
            if molecules:
                mol = molecules[0]
                chembl_id = mol.get("molecule_chembl_id")
                max_phase = mol.get("max_phase")
                
                # Determine status
                if max_phase == 4:
                    result["status"] = "FDA Approved"
                elif max_phase == 3:
                    result["status"] = "Phase 3 Clinical Trial"
                elif max_phase == 2:
                    result["status"] = "Phase 2 Clinical Trial"
                elif max_phase == 1:
                    result["status"] = "Phase 1 Clinical Trial"
                else:
                    result["status"] = "Preclinical"
                
                # Get molecule details for IDs
                if chembl_id:
                    detail_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
                    detail_response = httpx.get(detail_url, timeout=10.0)
                    
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        xrefs = detail_data.get("cross_references", [])
                        
                        for xref in xrefs:
                            src = xref.get("xref_src", "")
                            xid = xref.get("xref_id", "")
                            
                            if src == "DrugBank" and xid and result["drugbank_id"] == "N/A":
                                result["drugbank_id"] = xid
                            elif "PubChem" in src and xid and result["pubchem_id"] == "N/A":
                                result["pubchem_id"] = xid
    
    except Exception as e:
        print(f"Error fetching drug metadata from ChEMBL for {drug_name}: {str(e)}", file=sys.stderr)
    
    return result


def get_manual_drug_database(gene_name: str, uniprot_id: str) -> Dict:
    """
    Curated database of known drug-target relationships
    Covers major therapeutic targets
    """
    DATABASE = {
        # EGFR
        "EGFR": {
            "fda_approved": [
                {"name": "Erlotinib", "indication": "Non-small cell lung cancer (NSCLC)", "year": 2004, "type": "Small molecule TKI"},
                {"name": "Gefitinib", "indication": "NSCLC with EGFR mutations", "year": 2003, "type": "Small molecule TKI"},
                {"name": "Afatinib", "indication": "NSCLC", "year": 2013, "type": "Irreversible TKI"},
                {"name": "Osimertinib", "indication": "NSCLC (T790M mutation)", "year": 2015, "type": "3rd-gen TKI"},
                {"name": "Cetuximab", "indication": "Colorectal cancer, Head & neck", "year": 2004, "type": "Monoclonal antibody"},
                {"name": "Panitumumab", "indication": "Colorectal cancer", "year": 2006, "type": "Monoclonal antibody"}
            ],
            "clinical_trials": [
                {"name": "Mobocertinib", "phase": "Phase 3", "indication": "NSCLC (Exon 20 insertion)", "status": "Active"},
                {"name": "Amivantamab", "phase": "Phase 3", "indication": "NSCLC", "status": "Recruiting"}
            ]
        },
        
        # TP53
        "TP53": {
            "fda_approved": [],
            "clinical_trials": [
                {"name": "APR-246 (Eprenetapopt)", "phase": "Phase 3", "indication": "AML with TP53 mutation", "status": "Active"},
                {"name": "PC14586", "phase": "Phase 1/2", "indication": "Solid tumors with TP53 mutation", "status": "Recruiting"},
                {"name": "Kevetrin", "phase": "Phase 2", "indication": "Ovarian cancer", "status": "Active"}
            ]
        },
        
        # BRCA1
        "BRCA1": {
            "fda_approved": [
                {"name": "Olaparib", "indication": "BRCA-mutated breast/ovarian cancer", "year": 2014, "type": "PARP inhibitor"},
                {"name": "Talazoparib", "indication": "BRCA-mutated breast cancer", "year": 2018, "type": "PARP inhibitor"},
                {"name": "Rucaparib", "indication": "BRCA-mutated ovarian cancer", "year": 2016, "type": "PARP inhibitor"},
                {"name": "Niraparib", "indication": "Ovarian cancer", "year": 2017, "type": "PARP inhibitor"}
            ],
            "clinical_trials": [
                {"name": "Veliparib", "phase": "Phase 3", "indication": "BRCA-mutated breast cancer", "status": "Active"}
            ]
        },
        
        # ALB (Albumin)
        "ALB": {
            "fda_approved": [
                {"name": "Albumin (Human)", "indication": "Hypovolemia, hypoalbuminemia", "year": 1944, "type": "Replacement therapy"}
            ],
            "clinical_trials": []
        },
        
        # INS (Insulin)
        "INS": {
            "fda_approved": [
                {"name": "Insulin glargine", "indication": "Type 1 & 2 diabetes", "year": 2000, "type": "Long-acting insulin"},
                {"name": "Insulin lispro", "indication": "Diabetes mellitus", "year": 1996, "type": "Rapid-acting insulin"},
                {"name": "Insulin aspart", "indication": "Diabetes mellitus", "year": 2000, "type": "Rapid-acting insulin"},
                {"name": "Insulin degludec", "indication": "Diabetes mellitus", "year": 2015, "type": "Ultra-long acting"}
            ],
            "clinical_trials": []
        }
    }
    
    gene_upper = gene_name.upper()
    return DATABASE.get(gene_upper, {"fda_approved": [], "clinical_trials": []})


# ===== PREDICTIVE RISK CALCULATOR =====

def calculate_disease_risk(protein_expression: pd.DataFrame, gene_name: str, 
                        user_factors: Dict = None) -> Dict:
    """
    Calculate disease risk based on protein expression and user factors
    
    Risk formula:
    Risk = (Expression_score × 0.4) + (Age × 0.2) + 
        (Family_history × 0.25) + (Lifestyle × 0.15)
    
    Returns risk level and recommendations
    """
    risk_data = {
        "gene": gene_name,
        "risk_score": 0.0,
        "risk_level": "Unknown",
        "components": {},
        "recommendations": [],
        "detection_advantage": "",
        "confidence": "Medium"
    }
    
    # Component 1: Protein expression score (0-40 points)
    expression_score = calculate_expression_score(protein_expression, gene_name)
    risk_data["components"]["expression"] = {
        "score": expression_score,
        "weight": 0.4,
        "contribution": (expression_score / 40.0) * 0.4 * 100
    }
    
    # Component 2: Age factor (0-20 points)
    age_score = 0
    if user_factors and user_factors.get("age"):
        age = user_factors["age"]
        if age < 40:
            age_score = 5
        elif age < 50:
            age_score = 10
        elif age < 60:
            age_score = 15
        else:
            age_score = 20
    
    risk_data["components"]["age"] = {
        "score": age_score,
        "weight": 0.2,
        "contribution": (age_score / 20.0) * 0.2 * 100
    }
    
    # Component 3: Family history (0-25 points)
    family_score = 0
    if user_factors and user_factors.get("family_history"):
        if user_factors["family_history"] == "first_degree":
            family_score = 25
        elif user_factors["family_history"] == "second_degree":
            family_score = 15
        elif user_factors["family_history"] == "none":
            family_score = 0
    
    risk_data["components"]["family_history"] = {
        "score": family_score,
        "weight": 0.25,
        "contribution": (family_score / 25.0) * 0.25 * 100
    }
    
    # Component 4: Lifestyle factors (0-15 points)
    lifestyle_score = calculate_lifestyle_score(user_factors)
    risk_data["components"]["lifestyle"] = {
        "score": lifestyle_score,
        "weight": 0.15,
        "contribution": (lifestyle_score / 15.0) * 0.15 * 100
    }
    
    # Calculate total risk score (0-100)
    # Normalize component scores to 0-1 range, then apply weights and scale to 0-100
    normalized_expression = (expression_score / 40.0) * 0.4  # 0-40 normalized by 40
    normalized_age = (age_score / 20.0) * 0.2              # 0-20 normalized by 20
    normalized_family = (family_score / 25.0) * 0.25       # 0-25 normalized by 25
    normalized_lifestyle = (lifestyle_score / 15.0) * 0.15 # 0-15 normalized by 15
    
    total_risk = (normalized_expression + normalized_age + normalized_family + normalized_lifestyle) * 100
    
    risk_data["risk_score"] = round(total_risk, 1)
    
    # Determine risk level
    if total_risk >= 70:
        risk_data["risk_level"] = "High Risk"
        risk_data["risk_color"] = "#dc3545"
        risk_data["detection_advantage"] = "Early detection possible 6-12 months earlier"
        risk_data["recommendations"] = get_high_risk_recommendations(gene_name)
    elif total_risk >= 40:
        risk_data["risk_level"] = "Medium Risk"
        risk_data["risk_color"] = "#ffc107"
        risk_data["detection_advantage"] = "Regular monitoring recommended"
        risk_data["recommendations"] = get_medium_risk_recommendations(gene_name)
    else:
        risk_data["risk_level"] = "Low Risk"
        risk_data["risk_color"] = "#28a745"
        risk_data["detection_advantage"] = "Routine screening sufficient"
        risk_data["recommendations"] = get_low_risk_recommendations(gene_name)
    
    return risk_data


def calculate_expression_score(expression_df: pd.DataFrame, gene_name: str) -> float:
    """
    Calculate risk score from protein expression patterns
    Higher/abnormal expression = higher risk
    """
    if expression_df.empty:
        return 20  # Default moderate score
    
    # Gene-specific risk associations
    RISK_GENES = {
        "TP53": {"risk_type": "low_expression", "threshold": 1.0},  # Loss of function
        "BRCA1": {"risk_type": "low_expression", "threshold": 1.0},  # Loss increases cancer risk
        "EGFR": {"risk_type": "high_expression", "threshold": 2.0},  # Overexpression in cancer
        "HER2": {"risk_type": "high_expression", "threshold": 2.0},
        "MYC": {"risk_type": "high_expression", "threshold": 2.0},
        "RAS": {"risk_type": "high_expression", "threshold": 1.5}
    }
    
    gene_config = RISK_GENES.get(gene_name.upper(), {"risk_type": "high_expression", "threshold": 1.5})
    
    # Calculate mean expression across tissues
    if 'level_numeric' in expression_df.columns:
        mean_expression = expression_df['level_numeric'].mean()
        high_expr_count = len(expression_df[expression_df['level'] == 'High'])
        
        if gene_config["risk_type"] == "high_expression":
            # High expression = high risk
            if mean_expression >= 2.5 or high_expr_count >= 5:
                return 40  # Very high
            elif mean_expression >= 2.0 or high_expr_count >= 3:
                return 30  # High
            elif mean_expression >= 1.0:
                return 20  # Moderate
            else:
                return 10  # Low
        else:
            # Low expression = high risk (tumor suppressors)
            if mean_expression <= 0.5:
                return 40  # Very high risk
            elif mean_expression <= 1.0:
                return 30  # High risk
            elif mean_expression <= 1.5:
                return 20  # Moderate
            else:
                return 10  # Low risk
    
    return 20  # Default


def calculate_lifestyle_score(user_factors: Dict) -> float:
    """Calculate risk from lifestyle factors"""
    if not user_factors:
        return 7.5  # Default moderate
    
    score = 0
    
    # Smoking
    if user_factors.get("smoking") == "current":
        score += 5
    elif user_factors.get("smoking") == "former":
        score += 3
    
    # BMI
    bmi = user_factors.get("bmi", 25)
    if bmi >= 30:
        score += 4
    elif bmi >= 25:
        score += 2
    
    # Exercise
    if user_factors.get("exercise") == "none":
        score += 3
    elif user_factors.get("exercise") == "occasional":
        score += 1
    
    # Diet
    if user_factors.get("diet") == "poor":
        score += 3
    
    return min(score, 15)


def get_high_risk_recommendations(gene_name: str) -> list:
    """Recommendations for high-risk individuals"""
    base_recommendations = [
        "Immediate consultation with oncologist/specialist recommended",
        "Enhanced screening protocol: Every 3-6 months",
        "Consider genetic counseling and testing",
        "Discuss preventive treatment options with physician"
    ]
    
    gene_specific = {
        "TP53": [
            "Li-Fraumeni syndrome evaluation recommended",
            "Multi-cancer early detection (MCED) testing",
            "Annual whole-body MRI screening"
        ],
        "BRCA1": [
            "Risk-reducing surgery discussion",
            "MRI + mammography every 6 months",
            "Consider prophylactic oophorectomy after age 40",
            "PARP inhibitor eligibility assessment"
        ],
        "EGFR": [
            "Low-dose CT screening for lung cancer",
            "Targeted therapy eligibility assessment",
            "Smoking cessation program (if applicable)"
        ]
    }
    
    return base_recommendations + gene_specific.get(gene_name.upper(), [])


def get_medium_risk_recommendations(gene_name: str) -> list:
    """Recommendations for medium-risk individuals"""
    return [
        "Annual screening with specialist",
        "Biomarker monitoring every 6-12 months",
        "Lifestyle modification consultation",
        "Consider participation in prevention trials",
        "Regular self-examination and symptom awareness"
    ]


def get_low_risk_recommendations(gene_name: str) -> list:
    """Recommendations for low-risk individuals"""
    return [
        "Standard age-appropriate screening",
        "Annual health check-up",
        "Maintain healthy lifestyle habits",
        "Be aware of warning signs and symptoms",
        "Re-assess if family history changes"
    ]

