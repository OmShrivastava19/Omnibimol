import pandas as pd
from typing import Dict

# data_processor.py - Transform API responses for visualization
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
                            pdb_data: Dict = None, kegg_data: Dict = None) -> pd.DataFrame:
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
        
        summary = {
            "Metric": [
                "UniProt ID",
                "Sequence Length",
                "Molecular Weight (Da)",
                "3D Structure",
                "KEGG Pathways",
                "Tissues with Expression",
                "High Expression Tissues",
                "Subcellular Locations",
                "GO Terms (Total)"
            ],
            "Value": [
                uniprot_data.get("uniprot_id", "N/A"),
                f"{uniprot_data.get('sequence_length', 0):,}",
                f"{uniprot_data.get('mass', 0):,.0f}",
                structure_status,
                pathway_count if pathway_count > 0 else "Not found",
                len(tissue_df[tissue_df["level_numeric"] > 0]) if not tissue_df.empty else 0,
                len(tissue_df[tissue_df["level"] == "High"]) if not tissue_df.empty else 0,
                len(subcellular_df) if not subcellular_df.empty else 0,
                sum(len(v) for v in uniprot_data.get("go_terms", {}).values())
            ]
        }
        
        return pd.DataFrame(summary)