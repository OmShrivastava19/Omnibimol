import httpx
import pandas as pd
import streamlit as st
import asyncio
from typing import List, Dict
import os
import random
import math
import re
import html

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
    """Handles all API interactions with UniProt and Human Protein Atlas"""
    
    def __init__(self, cache_manager):
        self.cache = cache_manager
        self.uniprot_base = "https://rest.uniprot.org"
        self.hpa_base = "https://www.proteinatlas.org/api"
        
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
            "literature": results[6] if not isinstance(results[6], Exception) else {"papers": [], "wiki_title": None, "wiki_snippet": None}
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
        import requests
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
            search = requests.get(pubmed_url, params=params, timeout=10).json()
            pmids = search.get('esearchresult', {}).get('idlist', [])
            
            if pmids:
                try:
                    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    abs_data = requests.get(
                        efetch_url,
                        params={'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml'},
                        timeout=10
                    ).text
                    papers = self.parse_pubmed_abstracts(abs_data)
                except Exception as e:
                    st.warning(f"Could not fetch PubMed abstracts: {str(e)}")
        except Exception as e:
            st.warning(f"PubMed search error: {str(e)}")
        
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
            wiki_response = requests.get(wiki_url, params=wiki_params, headers=headers, timeout=10)
            wiki_response.raise_for_status()
            
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
            else:
                st.warning("Wikipedia returned empty response")
        except requests.exceptions.RequestException as e:
            st.warning(f"Wikipedia connection error: {str(e)}")
        except ValueError as e:
            st.warning(f"Wikipedia response parsing error: Invalid JSON response")
        except Exception as e:
            st.warning(f"Wikipedia search error: {str(e)}")
        
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
        Run BLAST search against NCBI nr database
        Optimized for speed while maintaining full sequence accuracy
        Returns up to 15 hits
        
        Key optimizations:
        - Uses 'nr' database for accuracy (not swissprot)
        - Parallel submission + immediate polling
        - Adaptive polling with exponential backoff
        - Concurrent status check + result fetch on ready
        """
        cache_key = f"blast_{uniprot_id}_v2"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            submit_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
            
            # Full accuracy parameters - no shortcuts on sequence or database
            submit_params = {
                "CMD": "Put",
                "PROGRAM": "blastp",
                "DATABASE": "nr",              # Full nr database for accuracy
                "QUERY": sequence,             # Full unmodified sequence
                "FORMAT_TYPE": "XML",
                "HITLIST_SIZE": "15",           # Request exactly 15 hits
                "EXPECT": "0.001",             # Strict E-value for quality hits
                "FILTER": "F",                 # No filtering - keep all hits
                "COMPOSITION_BASED_STATS": "2", # Best compositional adjustment
                "ENTREZ_QUERY": "homo sapiens[organism]"  # Limit to human for speed
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
                    "error": "Failed to submit BLAST job - no RID received"
                }
            
            # Step 2: Adaptive polling strategy
            # Initial wait based on estimated time (minimum 5 seconds)
            initial_wait = max(5, estimated_time * 0.5)
            await asyncio.sleep(initial_wait)
            
            # Polling configuration
            max_attempts = 120          # Max 120 attempts
            attempt = 0
            poll_interval = 1.0         # Start at 1 second
            max_poll_interval = 3.0     # Cap at 3 seconds
            
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
                                "database": "nr",
                                "organism_filter": "Homo sapiens"
                            }
                            
                            self.cache.set(cache_key, result)
                            return result
                        
                        # Still waiting - adaptive backoff
                        elif "Status=WAITING" in check_text or "Status=UNKNOWN" in check_text:
                            attempt += 1
                            # Exponential backoff capped at max_poll_interval
                            poll_interval = min(
                                max_poll_interval, 
                                poll_interval * 1.15  # Gentle 15% increase
                            )
                            await asyncio.sleep(poll_interval)
                            continue
                        
                        # Job failed on server side
                        elif "Status=FAILURE" in check_text or "Status=ERROR" in check_text:
                            return {
                                "available": False,
                                "error": "BLAST job failed on NCBI server. Try again."
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
                "error": "BLAST search timed out. NCBI servers may be busy - try again later."
            }
            
        except Exception as e:
            return {
                "available": False,
                "error": f"BLAST error: {str(e)}"
            }

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
                "email": "omnibiomol@example.com",
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
            "sequence_length": uniprot_data.get('sequence_length', 0)
        }

    def simulate_docking_score(self, protein_length: int, ligand_mw: float, 
                          activity_value: float = None) -> Dict:
        """
        Simulate docking results (since running actual AutoDock Vina requires backend server)
        In production, this would call a backend API running AutoDock Vina
        
        NOTE: This is a SIMPLIFIED SIMULATION for demonstration
        Real docking would use: python subprocess to run vina --receptor protein.pdbqt --ligand ligand.pdbqt
        """
        
        # Simulate binding affinity based on known activity data
        if activity_value:
            # Convert IC50/Ki to approximate binding affinity
            # Lower IC50 = better binding = more negative affinity
            base_affinity = -math.log10(activity_value / 1000000) * 1.5
        else:
            # Random score for unknown compounds
            base_affinity = random.uniform(-4, -9)
        
        # Add some variation
        noise = random.uniform(-0.5, 0.5)
        binding_affinity = base_affinity + noise
        
        # Simulate multiple binding modes
        modes = []
        for i in range(min(5, random.randint(3, 7))):
            mode_affinity = binding_affinity + random.uniform(0, 2)
            modes.append({
                "mode": i + 1,
                "affinity": round(mode_affinity, 2),
                "rmsd_lb": round(random.uniform(0, 2), 2),
                "rmsd_ub": round(random.uniform(0, 3), 2)
            })
        
        # Sort by affinity (most negative = best)
        modes = sorted(modes, key=lambda x: x['affinity'])
        
        return {
            "available": True,
            "binding_affinity": round(modes[0]['affinity'], 2),
            "modes": modes,
            "exhaustiveness": 8,
            "simulated": True  # Flag indicating this is simulated
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