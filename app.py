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
import requests

# app.py - Main Streamlit application
def main():
    """Main OmniBiMol (MVP)"""
    
    # Page configuration
    st.set_page_config(
        page_title="OmniBiMol - Protein Analysis Platform",
        page_icon="icons/Omnibimol_logo.png",
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
    
    # Header with banner
    st.image("icons/Omnibimol_banner.png", width='stretch')
    
    # Initialize cache and API client
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    if 'api_client' not in st.session_state:
        st.session_state.api_client = ProteinAPIClient(st.session_state.cache_manager)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        **OmniBiMol (MVP)**
        
        Integrated protein analysis platform combining:
        - UniProt: Protein function & annotations
        - Human Protein Atlas: Expression data
        - AlphaFold & PDB: Structural information
        - KEGG: Pathway mapping
        - GO: Gene ontology annotations
        - EMBL-EBI: Sequence analysis
        - NCBI BLAST: Homology search
        - EMBOSS Needle: Sequence alignment
        - And more...
        
        **Features:**
        - Real-time data retrieval
        - Interactive visualizations
        - 24-hour caching
        - Mobile-responsive design
        - User-friendly interface
        - Extensible architecture
        - Open-source & free to use

        **Developed by:** Team BhUOm
        """)

        st.divider()

        if st.button("🔄 Clear All Cache & Reload", key="sidebar_clear_cache"):
            # Clear cache database
            st.session_state.cache_manager.clear()
            
            # Clear session state
            for key in list(st.session_state.keys()):
                if key not in ['cache_manager', 'api_client']:
                    del st.session_state[key]
    
            st.success("Cache cleared! Please search for your protein again.")
            st.rerun()
    
    # Main input section
    st.header("🔍 Protein Search")
    
    def trigger_search():
        """Callback to trigger search when Enter is pressed"""
        if st.session_state.get('search_input_key'):
            st.session_state.trigger_search = True
    
    protein_input = st.text_input(
        "Enter Protein Name or Gene Symbol:",
        value=st.session_state.get('protein_input', ''),
        placeholder="e.g., TP53, BRCA1, Insulin (Press Enter to search)",
        help="Enter a protein name, gene symbol, or UniProt ID",
        key="search_input_key",
        on_change=trigger_search
    )
    
    # Process search (triggered by Enter key or button)
    if (st.session_state.get('trigger_search') or st.button("🔎 Search", key="main_search_button", type="primary", width='stretch')) and protein_input:
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
            st.session_state.trigger_search = False
    
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
        if auto_load or st.button("✅ Confirm Selection", key="protein_confirm_selection", type="primary"):
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
                st.markdown(tags_html, unsafe_allow_html=True)

                if len(terms) > 10:
                    with st.expander(f"+{len(terms)-10} more"):
                        extra_tags_html = "".join([f'<span class="go-tag">{term}</span>' for term in terms[10:]])
                        st.markdown(extra_tags_html, unsafe_allow_html=True)

                st.markdown("")
        
        # GO terms chart
        if go_count > 0:
            fig_go = ProteinVisualizer.create_go_terms_chart(go_terms)
            st.plotly_chart(fig_go, width='stretch')
        
        st.divider()

        # Section 2: FASTA Sequence & BLAST Analysis
        st.header("🧬 Protein Sequence Analysis")
        
        # Create tabs
        sequence_tabs = st.tabs(["📄 FASTA Sequence", "🔬 Sequence Composition", "🔍 BLAST Homology Search", "🧬 EMBL Features & Alignment"])
        
        # Tab 1: FASTA Sequence
        with sequence_tabs[0]:
            st.subheader("FASTA Format Sequence")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                # Generate FASTA
                fasta_sequence = st.session_state.api_client.get_fasta_sequence(uniprot_data)
                
                # Display in text area
                st.text_area(
                    "Protein Sequence (FASTA format):",
                    fasta_sequence,
                    height=300,
                    help="Standard FASTA format with 60 characters per line"
                )
                
                # Sequence statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Length", f"{len(sequence)} aa")
                with col2:
                    st.metric("Molecular Weight", f"{uniprot_data.get('mass', 0):,.0f} Da")
                with col3:
                    # Calculate isoelectric point (simplified)
                    basic = sequence.count('K') + sequence.count('R') + sequence.count('H')
                    acidic = sequence.count('D') + sequence.count('E')
                    st.metric("Basic Residues", basic)
                with col4:
                    st.metric("Acidic Residues", acidic)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download FASTA",
                        fasta_sequence,
                        f"{st.session_state.current_uniprot_id}.fasta",
                        "text/plain",
                        help="Download sequence in FASTA format"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Raw Sequence",
                        sequence,
                        f"{st.session_state.current_uniprot_id}_sequence.txt",
                        "text/plain",
                        help="Download sequence without header"
                    )
                
            else:
                st.warning("⚠️ No sequence data available")
        
        # Tab 2: Sequence Composition
        with sequence_tabs[1]:
            st.subheader("Amino Acid Composition Analysis")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                # Analyze composition
                composition = ProteinVisualizer.analyze_sequence_composition(sequence)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color:#ff7f0e;">{composition['hydrophobic_percent']:.1f}%</h3>
                            <p style="margin:0; color:#666;">Hydrophobic</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color:#2ca02c;">{composition['polar_percent']:.1f}%</h3>
                            <p style="margin:0; color:#666;">Polar</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color:#d62728;">{composition['charged_percent']:.1f}%</h3>
                            <p style="margin:0; color:#666;">Charged</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Composition chart
                fig_composition = ProteinVisualizer.create_sequence_composition_chart(composition)
                st.plotly_chart(fig_composition, width='stretch')
                
                st.info("""
                **Color Legend:**
                - 🟠 **Orange**: Hydrophobic amino acids (A, V, I, L, M, F, W, P)
                - 🟢 **Green**: Polar amino acids (S, T, Y, N, Q, C)
                - 🔴 **Red**: Charged amino acids (K, R, H, D, E)
                - ⚫ **Gray**: Other (G)
                """)
                
            else:
                st.warning("⚠️ No sequence data available for analysis")
        
        # Tab 3: BLAST Search
        with sequence_tabs[2]:
            st.subheader("BLAST Homology Search")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                st.info("""
                **About BLAST Search:**
                - Searches for similar proteins across all organisms
                - Shows top 10 matches from NCBI nr database
                - Takes 30-60 seconds to complete
                - Results are cached for 24 hours
                """)
                
                # Check if BLAST results already exist
                if 'blast_results' not in st.session_state or \
                   st.session_state.get('blast_protein_id') != st.session_state.current_uniprot_id:
                    
                    run_blast = st.button("🚀 Run BLAST Search", key="blast_run_search", type="primary")
                    
                    if run_blast:
                        with st.spinner("🔬 Running BLAST search... This may take 30-60 seconds..."):
                            blast_results = asyncio.run(
                                st.session_state.api_client.run_blast_search(
                                    sequence,
                                    st.session_state.current_uniprot_id
                                )
                            )
                            
                            st.session_state.blast_results = blast_results
                            st.session_state.blast_protein_id = st.session_state.current_uniprot_id
                            st.rerun()
                
                # Display BLAST results if available
                if 'blast_results' in st.session_state and \
                   st.session_state.get('blast_protein_id') == st.session_state.current_uniprot_id:
                    
                    blast_data = st.session_state.blast_results
                    
                    if blast_data.get('available') and blast_data.get('hits'):
                        st.success(f"✅ Found {len(blast_data['hits'])} homologous proteins")
                        
                        # Display results table
                        blast_table_html = ProteinVisualizer.create_blast_results_table_html(blast_data['hits'])
                        st.components.v1.html(blast_table_html, height=600, scrolling=True)
                        
                        # Download results
                        blast_df = pd.DataFrame(blast_data['hits'])
                        csv_blast = blast_df.to_csv(index=False)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption("""
                            **Interpretation:**
                            - **Identity**: Percentage of identical amino acids
                            - **Coverage**: Percentage of query sequence aligned
                            - **E-value**: Lower values indicate more significant matches
                            """)
                        with col2:
                            st.download_button(
                                "📥 Download Results",
                                csv_blast,
                                f"{st.session_state.current_uniprot_id}_blast_results.csv",
                                "text/csv"
                            )
                        
                        # Option to run new search
                        if st.button("🔄 Run New BLAST Search", key="blast_run_new_search"):
                            del st.session_state.blast_results
                            del st.session_state.blast_protein_id
                            st.rerun()
                    
                    elif blast_data.get('error'):
                        error_msg = blast_data.get('error')
                        st.error(f"❌ BLAST search failed: {error_msg}")
                        
                        # Provide helpful suggestions
                        if "timed out" in error_msg.lower():
                            st.info("💡 **Tip:** BLAST searches can take 1-2 minutes for long sequences. Try again with a shorter sequence.")
                        elif "closed" in error_msg.lower():
                            st.info("💡 **Tip:** Network connection issue. Please try again.")
                        else:
                            st.info("💡 **Tip:** Try searching with a shorter protein sequence or check your internet connection.")
                        
                        if st.button("🔄 Try Again", key="blast_try_again"):
                            del st.session_state.blast_results
                            del st.session_state.blast_protein_id
                            st.rerun()
                    else:
                        st.warning("⚠️ No significant matches found")
            
            else:
                st.warning("⚠️ No sequence data available for BLAST search")
        
        # Tab 4: EMBL Features & Needle Alignment
        with sequence_tabs[3]:
            st.subheader("EMBL-EBI Sequence Analysis")
            
            sequence = uniprot_data.get('sequence', '')
            
            if sequence:
                # Create sub-tabs
                embl_subtabs = st.tabs(["🗺️ Protein Features", "⚡ Pairwise Alignment (Needle)"])
                
                # Sub-tab 1: Protein Features
                with embl_subtabs[0]:
                    st.markdown("**Protein Domain & Feature Annotations from EMBL-EBI**")
                    
                    # Check if EMBL data exists
                    if 'embl_features' not in st.session_state or \
                    st.session_state.get('embl_protein_id') != st.session_state.current_uniprot_id:
                        
                        with st.spinner("📡 Fetching feature annotations from EMBL-EBI..."):
                            embl_data = asyncio.run(
                                st.session_state.api_client.fetch_embl_sequence(
                                    st.session_state.current_uniprot_id
                                )
                            )
                            
                            st.session_state.embl_features = embl_data
                            st.session_state.embl_protein_id = st.session_state.current_uniprot_id
                    
                    embl_data = st.session_state.embl_features
                    
                    if embl_data.get('available') and embl_data.get('features'):
                        features = embl_data['features']
                        
                        st.success(f"✅ Found {len(features)} annotated features")
                        
                        # Feature statistics
                        col1, col2, col3 = st.columns(3)
                        
                        # Count feature types
                        feature_type_counts = {}
                        for feat in features:
                            ftype = feat.get('type', 'Other')
                            feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
                        
                        with col1:
                            st.metric("Total Features", len(features))
                        with col2:
                            st.metric("Feature Types", len(feature_type_counts))
                        with col3:
                            # Find longest feature
                            max_length = max([f.get('length', 0) for f in features], default=0)
                            st.metric("Longest Feature", f"{max_length} aa")
                        
                        st.markdown("---")
                        
                        # Feature map visualization
                        fig_features = ProteinVisualizer.create_feature_map(
                            features, 
                            uniprot_data.get('sequence_length', len(sequence))
                        )
                        st.plotly_chart(fig_features, width='stretch')
                        
                        # Detailed feature table
                        st.subheader("📋 Feature Details")
                        
                        feature_df = pd.DataFrame([
                            {
                                "Type": f.get('type', 'Unknown'),
                                "Description": f.get('description', 'N/A'),
                                "Start": f.get('start', 0),
                                "End": f.get('end', 0),
                                "Length": f.get('length', 0)
                            }
                            for f in features
                        ])
                        
                        # Add filter
                        feature_type_filter = st.multiselect(
                            "Filter by feature type:",
                            options=list(feature_type_counts.keys()),
                            default=list(feature_type_counts.keys())
                        )
                        
                        filtered_df = feature_df[feature_df['Type'].isin(feature_type_filter)]
                        st.dataframe(filtered_df, width='stretch', hide_index=True)
                        
                        # Download
                        csv_features = filtered_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Features",
                            csv_features,
                            f"{st.session_state.current_uniprot_id}_features.csv",
                            "text/csv"
                        )
                        
                    else:
                        st.info("ℹ️ No additional feature annotations available from EMBL-EBI")
                
                # Sub-tab 2: Needle Alignment
                with embl_subtabs[1]:
                    st.markdown("**EMBOSS Needle - Global Pairwise Sequence Alignment**")
                    
                    st.info("""
                    **About Needle Alignment:**
                    - Uses Needleman-Wunsch algorithm for global alignment
                    - Compares your protein sequence with another sequence
                    - Shows identity, similarity, gaps, and alignment score
                    - Takes ~10-30 seconds to complete
                    """)
                    
                    # Input for second sequence
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        compare_option = st.radio(
                            "Compare with:",
                            ["Paste sequence", "Use UniProt ID"],
                            horizontal=True
                        )
                    
                    sequence2 = ""
                    seq2_id = "Sequence_2"
                    
                    if compare_option == "Paste sequence":
                        sequence2_input = st.text_area(
                            "Enter second sequence (FASTA or plain text):",
                            height=150,
                            placeholder=">Protein_Name\nMKWVTFISLLFLFSSAYS...\n\nOr paste plain sequence:\nMKWVTFISLLFLFSSAYS..."
                        )
                        
                        if sequence2_input:
                            # Clean and parse input
                            sequence2_input = sequence2_input.strip()
                            
                            # Parse if FASTA format
                            if sequence2_input.startswith('>'):
                                lines = sequence2_input.split('\n')
                                seq2_id = lines[0][1:].strip().split()[0]
                                if not seq2_id:
                                    seq2_id = "Sequence_2"
                                sequence2 = ''.join(lines[1:])
                            else:
                                seq2_id = "Pasted_Sequence"
                                sequence2 = sequence2_input
                            
                            # Remove all whitespace, numbers, and non-letter characters
                            sequence2 = ''.join(c for c in sequence2.upper() if c.isalpha())
                            
                            # Validate sequence
                            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                            invalid_chars = set(sequence2) - valid_aa
                            
                            if invalid_chars:
                                st.warning(f"⚠️ Found non-standard amino acids: {', '.join(sorted(invalid_chars))}")
                                st.info("Only standard 20 amino acids will be used for alignment")
                                # Remove invalid characters
                                sequence2 = ''.join(c for c in sequence2 if c in valid_aa)
                            
                            if len(sequence2) < 10:
                                st.error("❌ Sequence too short (minimum 10 amino acids)")
                                sequence2 = ""
                            elif len(sequence2) > 50000:
                                st.error("❌ Sequence too long (maximum 50,000 amino acids)")
                                sequence2 = ""
                            else:
                                st.success(f"✅ Parsed sequence: {len(sequence2)} amino acids (ID: {seq2_id})")
                                st.session_state.compare_sequence = sequence2
                                st.session_state.compare_id = seq2_id
                    else:
                        compare_uniprot = st.text_input(
                            "Enter UniProt ID:",
                            placeholder="e.g., P04637, P38398"
                        )
                        
                        if compare_uniprot and st.button("🔍 Fetch Sequence", key="needle_fetch_sequence"):
                            with st.spinner("Fetching sequence..."):
                                compare_data = asyncio.run(
                                    st.session_state.api_client.fetch_uniprot_data(compare_uniprot)
                                )
                                
                                if compare_data.get('sequence'):
                                    sequence2 = compare_data['sequence']
                                    seq2_id = compare_uniprot
                                    st.success(f"✅ Loaded sequence from {compare_uniprot}")
                                    st.session_state.compare_sequence = sequence2
                                    st.session_state.compare_id = seq2_id
                        
                        # Use stored sequence if available
                        if 'compare_sequence' in st.session_state:
                            sequence2 = st.session_state.compare_sequence
                            seq2_id = st.session_state.compare_id
                    
                    st.markdown("---")
                    
                    # Run alignment
                    if sequence2:
                        run_needle = st.button("⚡ Run Needle Alignment", key="needle_run_alignment", type="primary")
                        
                        if run_needle:
                            with st.spinner("🧬 Running global alignment... This may take 10-30 seconds..."):
                                needle_results = asyncio.run(
                                    st.session_state.api_client.run_needle_alignment(
                                        sequence,
                                        sequence2,
                                        st.session_state.current_uniprot_id,
                                        seq2_id
                                    )
                                )
                                
                                st.session_state.needle_results = needle_results
                                st.rerun()
                    
                    # Display alignment results
                    if 'needle_results' in st.session_state:
                        needle_data = st.session_state.needle_results
                        
                        if needle_data.get('available'):
                            # Show alignment visualization
                            alignment_html = ProteinVisualizer.create_alignment_visualization(needle_data)
                            st.components.v1.html(alignment_html, height=800, scrolling=True)
                            
                            # Interpretation
                            st.subheader("📊 Interpretation")
                            
                            identity = needle_data.get('identity', 0)
                            
                            if identity >= 70:
                                st.success("✅ **High similarity** - Sequences are highly related (likely orthologs or close homologs)")
                            elif identity >= 40:
                                st.warning("⚠️ **Moderate similarity** - Sequences share common ancestry but have diverged")
                            else:
                                st.info("ℹ️ **Low similarity** - Sequences are distantly related or unrelated")
                            
                            # Download alignment
                            st.download_button(
                                "📥 Download Alignment",
                                needle_data.get('alignment_text', ''),
                                f"alignment_{st.session_state.current_uniprot_id}_vs_{seq2_id}.txt",
                                "text/plain"
                            )
                            
                            # Clear results
                            if st.button("🔄 Run New Alignment", key="needle_run_new_alignment"):
                                del st.session_state.needle_results
                                if 'compare_sequence' in st.session_state:
                                    del st.session_state.compare_sequence
                                    del st.session_state.compare_id
                                st.rerun()
                        
                        elif needle_data.get('error'):
                            error_msg = needle_data.get('error')
                            st.error(f"❌ Alignment failed: {error_msg}")
                            
                            # Provide helpful suggestions
                            if "400" in error_msg:
                                st.info("💡 **Tip:** Check that both sequences contain only valid amino acid letters (A-Z).")
                            elif "timed out" in error_msg.lower():
                                st.info("💡 **Tip:** Alignment is taking too long. Try with shorter sequences.")
                            
                            if st.button("🔄 Try Again", key="needle_try_again"):
                                del st.session_state.needle_results
                                st.rerun()                    
                    else:
                        st.info("👆 Enter a second sequence above and click 'Run Needle Alignment'")
            
            else:
                st.warning("⚠️ No sequence data available for EMBL analysis")
                
        st.divider()
        
        # Section 3: 3D Protein Structure
        st.header("🧊 3D Protein Structure")
        
        alphafold_data = data.get('alphafold_structure', {})
        pdb_data = data.get('pdb_structure', {})
        
        # Create tabs for different structure types
        if pdb_data.get('available') and alphafold_data.get('available'):
            structure_tabs = st.tabs(["📊 Experimental (PDB)", "🤖 Predicted (AlphaFold)"])
        elif pdb_data.get('available'):
            structure_tabs = st.tabs(["📊 Experimental (PDB)"])
        elif alphafold_data.get('available'):
            structure_tabs = st.tabs(["🤖 Predicted (AlphaFold)"])
        else:
            st.warning("⚠️ No 3D structure available for this protein")
            structure_tabs = None
        
        if structure_tabs:
            tab_index = 0
            
            # Experimental structure tab
            if pdb_data.get('available'):
                with structure_tabs[tab_index]:
                    st.markdown("**Available Experimental Structures:**")
                    
                    # Show all available PDB structures
                    pdb_structures = pdb_data.get('structures', [])
                    
                    for idx, struct in enumerate(pdb_structures[:5]):  # Show first 5
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        with col1:
                            st.markdown(f"**PDB ID:** [{struct['pdb_id']}]({struct['rcsb_page']})")
                        with col2:
                            st.markdown(f"**Method:** {struct['method']}")
                        with col3:
                            st.markdown(f"**Resolution:** {struct['resolution']}")
                        with col4:
                            if idx == 0:
                                st.markdown("✅ **Displayed below**")
                    
                    if len(pdb_structures) > 5:
                        st.info(f"+ {len(pdb_structures) - 5} more structures available on RCSB PDB")
                    
                    st.markdown("---")
                    
                    # Display 3D viewer for PDB
                    viewer_html = ProteinVisualizer.create_structure_viewer(pdb_data, "pdb")
                    st.components.v1.html(viewer_html, height=600, scrolling=False)
                    
                    # Download option
                    pdb_file_content = None
                    try:
                        pdb_url = pdb_structures[0]['pdb_url']
                        response = requests.get(pdb_url)
                        if response.status_code == 200:
                            pdb_file_content = response.text
                        else:
                            pdb_file_content = f"Could not fetch PDB file. Status code: {response.status_code}"
                    except Exception as e:
                        pdb_file_content = f"Error fetching PDB file: {str(e)}"

                    st.download_button(
                        "📥 Download PDB File",
                        data=pdb_file_content,
                        file_name=f"{pdb_structures[0]['pdb_id']}.pdb",
                        mime="text/plain"
                    )
                
                tab_index += 1
            
            # AlphaFold structure tab
            if alphafold_data.get('available'):
                with structure_tabs[tab_index]:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **AlphaFold Database Entry**
                        - **UniProt ID:** {alphafold_data.get('uniprot_id')}
                        - **Gene:** {alphafold_data.get('gene_name', 'N/A')}
                        - **Model Version:** v{alphafold_data.get('model_version', 4)}
                        - **[View on AlphaFold DB]({alphafold_data.get('alphafold_page')})**
                        - **[Download PDB]({alphafold_data.get('pdb_url')})**
                        """)
                    
                    with col2:
                        st.info("""
                        **Confidence Color Code:**
                        - 🔵 **Dark Blue** (>90): Very high confidence
                        - 🔵 **Light Blue** (70-90): Confident
                        - 🟡 **Yellow** (50-70): Low confidence
                        - 🟠 **Orange** (<50): Very low confidence
                        """)
                    
                    st.markdown("---")
                    
                    # Display 3D viewer
                    viewer_html = ProteinVisualizer.create_structure_viewer(alphafold_data, "alphafold")
                    st.components.v1.html(viewer_html, height=600, scrolling=False)
                    
                    # Confidence plot
                    st.subheader("📈 Prediction Confidence")
                    fig_confidence = ProteinVisualizer.create_confidence_plot(
                        st.session_state.current_uniprot_id,
                        alphafold_data.get('entry_id')
                    )
                    st.plotly_chart(fig_confidence, width='stretch')
                                        
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"[📥 Download PDB File]({alphafold_data.get('pdb_url')})")
                    with col2:
                        st.markdown(f"[📥 Download PAE Data]({alphafold_data.get('pae_url')})")
        
        st.divider()
        
        # Section 4: Tissue Expression
        st.header("🧫 Tissue Expression Analysis")
        
        if not tissue_df.empty:
            # Prepare data
            chart_data = DataProcessor.prepare_tissue_chart_data(tissue_df, top_n=20)
            
            # Create and display chart
            fig_tissue = ProteinVisualizer.create_tissue_expression_chart(chart_data)
            st.plotly_chart(fig_tissue, width='stretch')
            
            # Expression summary
            high_tissues = tissue_df[tissue_df["level"] == "High"]["tissue"].tolist()
            if high_tissues:
                st.info(f"**High expression detected in:** {', '.join(high_tissues[:5])}" + 
                    (f" and {len(high_tissues)-5} more tissues" if len(high_tissues) > 5 else ""))
        else:
            st.warning("⚠️ No tissue expression data available from Human Protein Atlas")
        
        st.divider()
        
        # Section 5: Subcellular Localization
        st.header("📍 Subcellular Localization")
        
        if not subcellular_df.empty:
            # Create and display heatmap
            fig_subcellular = ProteinVisualizer.create_subcellular_heatmap(subcellular_df)
            st.plotly_chart(fig_subcellular, width='stretch')
            
            # Location list
            st.markdown("**Detected Locations:**")
            for idx, row in subcellular_df.iterrows():
                st.markdown(f"- **{row['location']}** ({row['reliability']} confidence)")
        else:
            st.warning("⚠️ No subcellular localization data available from Human Protein Atlas")

        st.divider()
        
        # Section 6: KEGG Pathways for Proteins
        st.header("🧬 KEGG Pathways for Proteins")
        
        kegg_data = data.get('kegg_pathways', {})
        
        if kegg_data.get('available'):
            # Summary metrics
            total_pathways = kegg_data.get('total_pathways', 0)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1f77b4;">{total_pathways}</h3>
                        <p style="margin:0; color:#666;">Total Pathways Found</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1f77b4;">{kegg_data.get('kegg_protein_id', 'N/A')}</h3>
                        <p style="margin:0; color:#666;">KEGG Protein ID</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1f77b4;">{kegg_data.get('protein_name', 'N/A')}</h3>
                        <p style="margin:0; color:#666;">Protein Name</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Create tabs for different display formats
            pathway_tabs = st.tabs(["🖼️ Primary Pathway Map", "📋 Next 5 Pathways", "🔗 All Pathways Links"])
            
            # Tab 1: First Result with Full Details & Pathway Map
            first_result = kegg_data.get('first_result')
            with pathway_tabs[0]:
                if first_result:
                    st.subheader(f"🏆 Primary Pathway: {first_result.get('pathway_name', 'Unknown')}")
                    
                    # Display all metadata
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**Pathway Details:**")
                        st.markdown(f"- **ID:** `{first_result.get('pathway_id', 'N/A')}`")
                        st.markdown(f"- **Name:** {first_result.get('pathway_name', 'N/A')}")
                        
                        if first_result.get('pathway_description'):
                            st.markdown(f"- **Description:** {first_result.get('pathway_description', 'N/A')}")
                        
                        if first_result.get('pathway_class'):
                            st.markdown(f"- **Classification:** {first_result.get('pathway_class', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Molecular Functions:**")
                        functions = first_result.get('molecular_functions', [])
                        if functions:
                            for func in functions[:10]:  # Limit to 10 functions
                                st.markdown(f"• {func}")
                        else:
                            st.markdown("*No specific molecular functions listed*")
                    
                    st.markdown("---")
                    
                    # Display pathway map image
                    st.markdown("**Pathway Map:**")
                    try:
                        st.image(first_result.get('kegg_image_url', ''), 
                                width='stretch',
                                caption=f"{first_result.get('pathway_name')} - Visual representation from KEGG")
                    except Exception as e:
                        st.warning(f"Could not load pathway map image. [View on KEGG Website]({first_result.get('kegg_url', '#')})")
                    
                    st.markdown("---")
                    
                    # Direct links
                    col_link1, col_link2 = st.columns(2)
                    with col_link1:
                        st.markdown(f"**[📌 View on KEGG Website]({first_result.get('kegg_url', '#')})**")
                    with col_link2:
                        st.markdown(f"**[🔗 KEGG Gene Entry Page]({first_result.get('highlight_url', '#')})**")
                else:
                    st.info("No primary pathway data available")
            
            # Tab 2: Next 5 Results
            next_results = kegg_data.get('next_results', [])
            with pathway_tabs[1]:
                if next_results:
                    st.subheader("📊 Next 5 Pathways Associated with Protein")
                    
                    for idx, pathway in enumerate(next_results, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {pathway.get('pathway_name', 'Unknown')}**")
                                if pathway.get('pathway_class'):
                                    st.caption(f"Class: {pathway.get('pathway_class', '')}")
                            
                            with col2:
                                st.markdown(f"`{pathway.get('pathway_id', 'N/A')}`")
                            
                            with col3:
                                st.markdown(f"**[View →]({pathway.get('kegg_url', '#')})**")
                            
                            st.divider()
                else:
                    st.info("Less than 6 pathways found for this protein")
            
            # Tab 3: All Pathways Links
            all_pathways = kegg_data.get('pathways', [])
            with pathway_tabs[2]:
                st.subheader(f"🔗 All {len(all_pathways)} Associated Pathways")
                
                # Add filter and sort options
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input("🔍 Search pathways:", placeholder="e.g., cancer, metabolism, signaling")
                with col2:
                    sort_option = st.selectbox("Sort by:", ["Name", "ID"])
                
                # Filter pathways
                filtered_pathways = all_pathways
                if search_term:
                    search_term = search_term.lower()
                    filtered_pathways = [
                        p for p in all_pathways 
                        if search_term in p.get('pathway_name', '').lower() or 
                           search_term in p.get('pathway_id', '').lower()
                    ]
                
                # Sort pathways
                if sort_option == "Name":
                    filtered_pathways = sorted(filtered_pathways, key=lambda x: x.get('pathway_name', ''))
                elif sort_option == "ID":
                    filtered_pathways = sorted(filtered_pathways, key=lambda x: x.get('pathway_id', ''))
                
                # Display as table
                st.markdown("| # | Pathway Name | ID | KEGG Link |")
                st.markdown("|---|---|---|---|")
                for idx, pathway in enumerate(filtered_pathways, 1):
                    pathway_name = pathway.get('pathway_name', 'Unknown')
                    pathway_id = pathway.get('pathway_id', 'N/A')
                    kegg_url = pathway.get('kegg_url', '#')
                    st.markdown(f"| {idx} | {pathway_name} | `{pathway_id}` | [View Pathway]({kegg_url}) |")
                
                st.caption(f"Showing {len(filtered_pathways)} of {len(all_pathways)} pathways")
            
            # Download pathway data
            st.markdown("---")
            st.subheader("💾 Export Pathway Data")
            
            # Create DataFrame for export
            pathway_df = pd.DataFrame([
                {
                    "Pathway_Name": p['pathway_name'],
                    "Pathway_ID": p['pathway_id'],
                    "Classification": p.get('pathway_class', ''),
                    "Description": p.get('pathway_description', ''),
                    "KEGG_URL": p['kegg_url'],
                    "Highlighted_URL": p['highlight_url']
                }
                for p in all_pathways
            ])
            
            csv_pathways = pathway_df.to_csv(index=False)
            st.download_button(
                "📥 Download Pathway List (CSV)",
                csv_pathways,
                f"{st.session_state.current_uniprot_id}_kegg_pathways.csv",
                "text/csv"
            )
            
        else:
            st.warning(f"⚠️ No KEGG pathway data found for gene: {kegg_data.get('gene_name', 'Unknown')}")
            st.info("""
            **Why might this happen?**
            - Gene name not recognized in KEGG database
            - Protein not associated with metabolic/signaling pathways
            - Limited annotation in KEGG for this specific protein
            
            Try searching directly on [KEGG website](https://www.kegg.jp/)
            """)
        
        st.divider()
        
        # Section 7: Molecular Docking with AutoDock Vina
        st.header("💊 Molecular Docking Analysis")
        
        chembl_data = data.get('chembl_ligands', {})
        
        st.info("""
        **About Molecular Docking:**
        - Predicts how small molecules (ligands/drugs) bind to proteins
        - Uses AutoDock Vina algorithm for binding affinity calculation
        - Negative values indicate favorable binding (more negative = stronger binding)
        - Typical drug-like binding: -7 to -12 kcal/mol
        """)
        
        # Create tabs
        docking_tabs = st.tabs(["📚 Known Ligands", "🧪 Custom Docking", "📊 Docking Results"])
        
        # Tab 1: Known Ligands from ChEMBL
        with docking_tabs[0]:
            st.subheader("Known Inhibitors & Ligands from ChEMBL")
            
            if chembl_data.get('available') and chembl_data.get('ligands'):
                ligands = chembl_data['ligands']
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Ligands", len(ligands))
                with col2:
                    strong_binders = [l for l in ligands if l.get('activity_value', float('inf')) < 100]
                    st.metric("Strong Binders (<100nM)", len(strong_binders))
                with col3:
                    st.metric("ChEMBL Target", chembl_data.get('chembl_target_id', 'N/A'))
                
                st.markdown("---")
                
                # Display ligand table
                ligand_table_html = ProteinVisualizer.create_ligand_table_html(ligands)
                st.components.v1.html(ligand_table_html, height=800, scrolling=True)
                
                # Download ligand data
                ligand_df = pd.DataFrame([
                    {
                        "ChEMBL_ID": l['chembl_id'],
                        "Name": l['name'],
                        "SMILES": l.get('smiles', ''),
                        "Activity_Type": l['activity_type'],
                        "Activity_Value": l['activity_value'],
                        "Units": l['activity_units'],
                        "Molecular_Weight": l.get('molecular_weight', 'N/A')
                    }
                    for l in ligands
                ])
                
                csv_ligands = ligand_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Ligand Data",
                    csv_ligands,
                    f"{st.session_state.current_uniprot_id}_ligands.csv",
                    "text/csv"
                )
                
            else:
                st.warning(f"⚠️ No known ligands found in ChEMBL for {st.session_state.current_uniprot_id}")
                st.info("""
                **Possible reasons:**
                - Protein not yet studied as drug target
                - No bioactivity data available in ChEMBL
                - Protein may not be druggable
                
                You can still try custom docking in the next tab!
                """)
        
        # Tab 2: Custom Docking
        with docking_tabs[1]:
            st.subheader("Run Custom Molecular Docking")
            
            # Check if protein structure is available
            protein_prep = st.session_state.api_client.prepare_protein_for_docking(
                uniprot_data,
                data.get('pdb_structure', {}),
                data.get('alphafold_structure', {})
            )
            
            if not protein_prep.get('available'):
                st.error("❌ No protein structure available for docking. Please ensure 3D structure is loaded.")
            else:
                st.success(f"✅ Using {protein_prep['structure_type']} structure: {protein_prep['structure_id']}")
                
                # Ligand input options
                st.markdown("**Select Ligand Source:**")
                
                ligand_source = st.radio(
                    "",
                    ["Known ligand from ChEMBL", "Custom compound (PubChem)", "Upload SMILES/SDF"],
                    horizontal=False
                )
                
                selected_ligand = None
                ligand_name = None
                
                if ligand_source == "Known ligand from ChEMBL":
                    if chembl_data.get('available') and chembl_data.get('ligands'):
                        ligand_options = {
                            f"{l['name']} ({l['chembl_id']}) - {l['activity_type']}: {l['activity_value']:.1f} {l['activity_units']}": l
                            for l in chembl_data['ligands'][:10]
                        }
                        
                        selected_option = st.selectbox("Choose ligand:", list(ligand_options.keys()))
                        selected_ligand = ligand_options[selected_option]
                        ligand_name = selected_ligand['name']
                    else:
                        st.warning("No ChEMBL ligands available")
                
                elif ligand_source == "Custom compound (PubChem)":
                    compound_name = st.text_input(
                        "Enter compound name:",
                        placeholder="e.g., Aspirin, Ibuprofen, Caffeine"
                    )
                    
                    if compound_name and st.button("🔍 Search PubChem", key="docking_search_pubchem"):
                        with st.spinner("Searching PubChem..."):
                            pubchem_data = asyncio.run(
                                st.session_state.api_client.fetch_pubchem_structure(compound_name)
                            )
                            
                            if pubchem_data.get('available'):
                                st.success(f"✅ Found: {compound_name} (CID: {pubchem_data['cid']})")
                                st.image(pubchem_data['image_url'], width=200)
                                st.session_state.custom_ligand = pubchem_data
                                selected_ligand = pubchem_data
                                ligand_name = compound_name
                            else:
                                st.error(f"❌ Compound '{compound_name}' not found in PubChem")
                    
                    if 'custom_ligand' in st.session_state:
                        selected_ligand = st.session_state.custom_ligand
                        ligand_name = compound_name
                
                else:  # Upload SMILES/SDF
                    smiles_input = st.text_input(
                        "Enter SMILES string:",
                        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O (Aspirin)"
                    )
                    
                    if smiles_input:
                        ligand_name = "Custom_SMILES"
                        selected_ligand = {"smiles": smiles_input, "name": ligand_name}
                
                st.markdown("---")
                
                # Docking parameters
                st.markdown("**Docking Parameters:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    exhaustiveness = st.slider("Exhaustiveness", 1, 16, 8, 
                                              help="Higher = more thorough but slower")
                with col2:
                    num_modes = st.slider("Number of modes", 1, 20, 5,
                                         help="Number of binding poses to generate")
                with col3:
                    energy_range = st.slider("Energy range (kcal/mol)", 1, 5, 3)
                
                # Run docking button
                if selected_ligand:
                    run_docking = st.button("🚀 Run Molecular Docking", key="docking_run_simulation", type="primary")
                    
                    if run_docking:
                        with st.spinner("🧬 Running AutoDock Vina simulation... This may take 30-60 seconds..."):
                            # Simulate docking (in production, this would call backend Vina)
                            time.sleep(2)  # Simulate computation time
                            
                            docking_results = st.session_state.api_client.simulate_docking_score(
                                protein_prep['sequence_length'],
                                selected_ligand.get('molecular_weight', 300),
                                selected_ligand.get('activity_value')
                            )
                            
                            st.session_state.docking_results = docking_results
                            st.session_state.docked_ligand_name = ligand_name
                            st.rerun()
                else:
                    st.info("👆 Please select or enter a ligand above")
        
        # Tab 3: Docking Results
        with docking_tabs[2]:
            st.subheader("Docking Results")
            
            if 'docking_results' in st.session_state:
                results = st.session_state.docking_results
                ligand_name = st.session_state.get('docked_ligand_name', 'Unknown')
                
                if results.get('simulated'):
                    st.warning("⚠️ **Note:** These are simulated results for demonstration. Production version would use actual AutoDock Vina calculations.")
                
                st.markdown(f"### Results for: **{ligand_name}**")
                
                # Best binding affinity
                best_affinity = results['binding_affinity']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Color code by strength
                    if best_affinity < -7:
                        color = "#28a745"
                        strength = "Strong"
                    elif best_affinity < -5:
                        color = "#ffc107"
                        strength = "Moderate"
                    else:
                        color = "#dc3545"
                        strength = "Weak"
                    
                    st.markdown(f"""
                        <div style="background-color:{color}; color:white; padding:20px; border-radius:8px; text-align:center;">
                            <h2 style="margin:0;">{best_affinity} kcal/mol</h2>
                            <p style="margin:5px 0 0 0;">Best Binding Affinity</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Binding Strength", strength)
                with col3:
                    st.metric("Binding Modes", len(results.get('modes', [])))
                
                st.markdown("---")
                
                # Binding modes chart
                fig_docking = ProteinVisualizer.create_docking_results_chart(results)
                st.plotly_chart(fig_docking, width='stretch')
                
                # Detailed modes table
                st.subheader("📋 Binding Mode Details")
                
                modes_df = pd.DataFrame(results.get('modes', []))
                st.dataframe(modes_df, width='stretch', hide_index=True)
                
                # Interpretation
                st.subheader("💡 Interpretation")
                
                if best_affinity < -7:
                    st.success("""
                    **Strong Binding** (< -7 kcal/mol)
                    - Indicates favorable protein-ligand interaction
                    - This compound shows drug-like binding affinity
                    - Worth further experimental validation""")
                elif best_affinity < -5:
                    st.info("""
                    Moderate Binding (-5 to -7 kcal/mol)
                    - Shows some binding potential
                    - May require optimization for better affinity
                    - Consider structural modifications
                    """)
                else:
                    st.warning("""
                    Weak Binding (> -5 kcal/mol)
                    - Limited binding affinity
                    - Unlikely to be effective inhibitor
                    - Significant optimization needed
                    """)

                # Download results
                st.download_button(
                    "📥 Download Docking Results",
                    modes_df.to_csv(index=False),
                    f"docking_{st.session_state.current_uniprot_id}_{ligand_name}.csv",
                    "text/csv"
                )
            
                # Clear results
                if st.button("🔄 Run New Docking", key="docking_run_new_docking"):
                    del st.session_state.docking_results
                    del st.session_state.docked_ligand_name
                    if 'custom_ligand' in st.session_state:
                        del st.session_state.custom_ligand
                    st.rerun()
        
                else:
                    st.info("👈 Run a docking simulation in the 'Custom Docking' tab to see results here")
        
        st.divider()

        # Section 9: Summary Table
        st.header("📊 Data Summary")
        
        summary_df = DataProcessor.create_summary_table(
            uniprot_data, 
            tissue_df, 
            subcellular_df,
            data.get('alphafold_structure'),
            data.get('pdb_structure'),
            data.get('kegg_pathways'),
            data.get('chembl_ligands')
        )

        st.dataframe(summary_df, width='stretch', hide_index=True)
        
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
        
        # Section: Protein Literature Summary
        with st.expander("🔬 Literature & Overview", expanded=False):
            literature = data.get('literature', {})
            # Wikipedia intro
            if literature.get('wiki_title'):
                st.info(f"**Wikipedia**: [{literature['wiki_title']}](https://en.wikipedia.org/wiki/{literature['wiki_title'].replace(' ', '_')})")
                st.caption(literature.get('wiki_snippet', ''))
            # Top papers
            if literature.get('papers'):
                st.subheader("Top 5 Research Papers")
                for i, p in enumerate(literature['papers'], 1):
                    with st.container():
                        st.markdown(f"**{p['title']}**")
                        st.caption(f"{p['authors']} | [PMID: {p['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{p['pmid']})")
                        st.caption(p['abstract_snip'])
                        st.divider()
            else:
                st.warning("No recent papers found; try official gene name.")
        
if __name__ == "__main__":
    main()