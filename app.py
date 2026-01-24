# app.py - Full imports with purpose annotations
import streamlit as st
import httpx
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from contextlib import asynccontextmanager
from cache_manager import *
from data_processor import *
from visualizations import *
from api_client import *

# app.py - Main Streamlit application
def main():
    """Main OmniBiMol Phase 1 MVP Application"""
    
    # Page configuration
    st.set_page_config(
        page_title="OmniBiMol - Protein Analysis Platform",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional bioinformatics styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .info-card {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            color: #000;
            font-size: 0.95rem;
        }
        .metric-card {
            background-color: #e7f3ff;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
        }
        .go-tag {
            display: inline-block;
            background-color: #e8f4f8;
            color: #1f77b4;
            padding: 0.4rem 0.8rem;
            margin: 0.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🧬 OmniBiMol</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Operations on Medicinally Navigated Information for Biological Molecules</p>', unsafe_allow_html=True)
    
    # Initialize cache and API client
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    if 'api_client' not in st.session_state:
        st.session_state.api_client = ProteinAPIClient(st.session_state.cache_manager)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        **OmniBiMol Phase 1 MVP**
        
        Integrated protein analysis platform combining:
        - UniProt: Protein function & annotations
        - Human Protein Atlas: Expression data
        
        **Features:**
        - Real-time data retrieval
        - Interactive visualizations
        - 24-hour caching
        - Mobile-responsive design
        """)
        
        st.divider()
        
        st.header("🧪 Example Proteins")
        example_proteins = {
            "TP53 (Tumor suppressor)": "TP53",
            "BRCA1 (DNA repair)": "BRCA1",
            "INS (Insulin)": "INS",
            "EGFR (Growth factor receptor)": "EGFR",
            "ALB (Albumin)": "ALB"
        }
        
        selected_example = st.selectbox(
            "Quick load:",
            [""] + list(example_proteins.keys())
        )
        
        if selected_example and st.button("Load Example"):
            st.session_state.protein_input = example_proteins[selected_example]
            st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Clear Cache"):
            st.session_state.cache_manager.clear_expired()
            st.success("Cache cleared!")
    
    # Main input section
    st.header("🔍 Protein Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        protein_input = st.text_input(
            "Enter Protein Name or Gene Symbol:",
            value=st.session_state.get('protein_input', ''),
            placeholder="e.g., TP53, BRCA1, Insulin",
            help="Enter a protein name, gene symbol, or UniProt ID"
        )
    
    with col2:
        search_button = st.button("🔎 Search", type="primary", use_container_width=True)
    
    # Process search
    if search_button and protein_input:
        with st.spinner("🔍 Searching UniProt database..."):
            # Search UniProt
            search_results = asyncio.run(
                st.session_state.api_client.search_uniprot(protein_input)
            )
            
            if not search_results:
                st.error("❌ No results found. Please check your input and try again.")
                st.stop()
            
            # Store results in session state
            st.session_state.search_results = search_results
            st.session_state.show_results = True
    
    # Display search results for confirmation
    if st.session_state.get('show_results') and st.session_state.get('search_results'):
        st.subheader("Select Protein:")
        
        results = st.session_state.search_results
        
        if len(results) == 1:
            st.info(f"✅ Found: **{results[0]['protein_name']}** ({results[0]['gene_name']}) - {results[0]['uniprot_id']}")
            selected_uniprot_id = results[0]['uniprot_id']
            auto_load = True
        else:
            # Multiple results - let user choose
            options = [
                f"{r['protein_name']} ({r['gene_name']}) - {r['uniprot_id']} | {r['organism']}"
                for r in results
            ]
            
            selected_idx = st.radio("Multiple matches found:", range(len(options)), format_func=lambda i: options[i])
            selected_uniprot_id = results[selected_idx]['uniprot_id']
            selected_gene_name = results[selected_idx]['gene_name']
            auto_load = False
        
        # Get gene name for selected protein
        if auto_load:
            selected_gene_name = results[0]['gene_name']
        
        # Confirm and load data
        if auto_load or st.button("✅ Confirm Selection", type="primary"):
            with st.spinner("📊 Fetching protein data..."):
                start_time = time.time()
                
                # Fetch all data - UniProt from API, HPA from local files
                all_data = asyncio.run(
                    st.session_state.api_client.fetch_all_data(selected_uniprot_id, selected_gene_name)
                )
                
                fetch_time = time.time() - start_time
                
                # Store in session state
                st.session_state.current_data = all_data
                st.session_state.current_uniprot_id = selected_uniprot_id
                st.session_state.fetch_time = fetch_time
                st.session_state.show_results = False
                
                st.success(f"✅ Data loaded in {fetch_time:.2f} seconds!")
                st.rerun()
    
    # Display protein data if available
    if st.session_state.get('current_data'):
        data = st.session_state.current_data
        uniprot_data = data['uniprot_data']
        tissue_df = data['tissue_expression']
        subcellular_df = data['subcellular']
        
        st.divider()
        
        # Section 1: Protein Information
        st.header(f"📖 Protein Information: {st.session_state.current_uniprot_id}")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{uniprot_data.get('sequence_length', 0):,}</h3>
                    <p style="margin:0; color:#666;">Amino Acids</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{uniprot_data.get('mass', 0):,.0f}</h3>
                    <p style="margin:0; color:#666;">Molecular Weight (Da)</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            tissue_count = len(tissue_df[tissue_df["level_numeric"] > 0]) if not tissue_df.empty else 0
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{tissue_count}</h3>
                    <p style="margin:0; color:#666;">Expressed Tissues</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            go_count = sum(len(v) for v in uniprot_data.get("go_terms", {}).values())
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#1f77b4;">{go_count}</h3>
                    <p style="margin:0; color:#666;">GO Terms</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Function description
        st.subheader("🔬 Protein Function")
        st.markdown(f"""
            <div class="info-card">
                {uniprot_data.get('function', 'No functional annotation available')}
            </div>
        """, unsafe_allow_html=True)
        # GO Terms
        st.subheader("🏷️ Gene Ontology Terms")
        
        go_terms = uniprot_data.get('go_terms', {})
        
        for category, terms in go_terms.items():
            if terms:
                st.markdown(f"**{category}:**")
                tags_html = "".join([f'<span class="go-tag">{term}</span>' for term in terms[:10]])
                if len(terms) > 10:
                    tags_html += f'<span class="go-tag">+{len(terms)-10} more</span>'
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("")
        
        # GO terms chart
        if go_count > 0:
            fig_go = ProteinVisualizer.create_go_terms_chart(go_terms)
            st.plotly_chart(fig_go, use_container_width=True)
        
        st.divider()
        
        # Section 2: Tissue Expression
        st.header("🧫 Tissue Expression Analysis")
        
        if not tissue_df.empty:
            # Prepare data
            chart_data = DataProcessor.prepare_tissue_chart_data(tissue_df, top_n=20)
            
            # Create and display chart
            fig_tissue = ProteinVisualizer.create_tissue_expression_chart(chart_data)
            st.plotly_chart(fig_tissue, use_container_width=True)
            
            # Expression summary
            high_tissues = tissue_df[tissue_df["level"] == "High"]["tissue"].tolist()
            if high_tissues:
                st.info(f"**High expression detected in:** {', '.join(high_tissues[:5])}" + 
                    (f" and {len(high_tissues)-5} more tissues" if len(high_tissues) > 5 else ""))
        else:
            st.warning("⚠️ No tissue expression data available from Human Protein Atlas")
        
        st.divider()
        
        # Section 3: Subcellular Localization
        st.header("📍 Subcellular Localization")
        
        if not subcellular_df.empty:
            # Create and display heatmap
            fig_subcellular = ProteinVisualizer.create_subcellular_heatmap(subcellular_df)
            st.plotly_chart(fig_subcellular, use_container_width=True)
            
            # Location list
            st.markdown("**Detected Locations:**")
            for idx, row in subcellular_df.iterrows():
                st.markdown(f"- **{row['location']}** ({row['reliability']} confidence)")
        else:
            st.warning("⚠️ No subcellular localization data available from Human Protein Atlas")
        
        st.divider()
        
        # Section 4: Summary Table
        st.header("📊 Data Summary")
        
        summary_df = DataProcessor.create_summary_table(uniprot_data, tissue_df, subcellular_df)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Download options
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not tissue_df.empty:
                csv_tissue = tissue_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Tissue Data",
                    csv_tissue,
                    f"{st.session_state.current_uniprot_id}_tissue_expression.csv",
                    "text/csv"
                )
        
        with col2:
            if not subcellular_df.empty:
                csv_subcellular = subcellular_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Subcellular Data",
                    csv_subcellular,
                    f"{st.session_state.current_uniprot_id}_subcellular.csv",
                    "text/csv"
                )
        
        with col3:
            csv_summary = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download Summary",
                csv_summary,
                f"{st.session_state.current_uniprot_id}_summary.csv",
                "text/csv"
            )
        
        # Footer
        st.divider()
        st.caption(f"⏱️ Data fetched in {st.session_state.get('fetch_time', 0):.2f}s | 💾 Cached for 24 hours | 🔬 Data sources: UniProt, Human Protein Atlas")

if __name__ == "__main__":
    main()