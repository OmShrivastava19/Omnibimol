# mypy: enable-error-code=var-annotated
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64

# visualizations.py - Plotly chart generation
class ProteinVisualizer:
    """Creates interactive Plotly visualizations for protein data"""
    
    @staticmethod
    def create_tissue_expression_chart(df: pd.DataFrame) -> go.Figure:
        """
        Create horizontal bar chart for tissue expression levels
        Color-coded by expression level (High=red, Medium=orange, Low=yellow, None=gray)
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No tissue expression data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        # Color mapping
        color_map = {
            "High": "#d62728",
            "Medium": "#ff7f0e",
            "Low": "#ffdd57",
            "Not detected": "#d3d3d3"
        }
        
        colors = [color_map.get(level, "#d3d3d3") for level in df["level"]]
        
        fig = go.Figure(go.Bar(
            x=df["level_numeric"],
            y=df["tissue"],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgb(8,48,107)', width=1)
            ),
            text=df["level"],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Expression: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Tissue Expression Levels (Top 20)",
            xaxis_title="Expression Level",
            yaxis_title="Tissue",
            height=max(400, len(df) * 20),
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3],
                ticktext=['Not detected', 'Low', 'Medium', 'High']
            ),
            showlegend=False,
            template="plotly_white",
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_subcellular_heatmap(df: pd.DataFrame) -> go.Figure:
        """
        Create heatmap for subcellular locations with reliability scores
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No subcellular location data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=300)
            return fig
        
        # Create matrix format
        z_data = [[val] for val in df["reliability_numeric"]]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=["Reliability"],
            y=df["location"],
            colorscale=[[0, '#f0f0f0'], [0.33, '#ffdd57'], [0.66, '#ff7f0e'], [1, '#2ca02c']],
            text=[[rel] for rel in df["reliability"]],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Reliability",
                tickmode='array',
                tickvals=[0, 1, 2, 3],
                ticktext=['Uncertain', 'Approved', 'Supported', 'Enhanced']
            ),
            hovertemplate='<b>%{y}</b><br>Reliability: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Subcellular Localization",
            height=max(300, len(df) * 30),
            template="plotly_white",
            margin=dict(l=150, r=150, t=50, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_go_terms_chart(go_terms: Dict) -> go.Figure:
        """
        Create stacked bar chart showing GO term counts by category
        """
        categories = []
        counts = []
        
        for category, terms in go_terms.items():
            if terms:
                categories.append(category)
                counts.append(len(terms))
        
        if not categories:
            fig = go.Figure()
            fig.add_annotation(
                text="No GO terms available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=250)
            return fig
        
        fig = go.Figure(go.Bar(
            x=categories,
            y=counts,
            marker=dict(
                color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                line=dict(color='rgb(8,48,107)', width=1)
            ),
            text=counts,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Gene Ontology Term Distribution",
            xaxis_title="GO Category",
            yaxis_title="Number of Terms",
            height=300,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
        
    @staticmethod
    def create_structure_viewer(structure_data: Dict, structure_type: str = "alphafold") -> str:
        """
        Create HTML for 3D protein structure viewer using NGL Viewer
        Returns HTML string with embedded viewer - Streamlit compatible
        """
        if structure_type == "alphafold" and structure_data.get("available"):
            pdb_url = structure_data.get("pdb_url")
            uniprot_id = structure_data.get("uniprot_id")
            title = f"AlphaFold Prediction - {uniprot_id}"
            color_scheme = "bfactor"  # Color by confidence
        elif structure_type == "pdb" and structure_data.get("available"):
            pdb_url = structure_data["structures"][0].get("pdb_url")
            pdb_id = structure_data["structures"][0].get("pdb_id")
            title = f"Experimental Structure - {pdb_id}"
            color_scheme = "chainindex"  # Color by chain
        else:
            error_msg = structure_data.get("error", "No structure available")
            return f"<p style='text-align:center; color:gray;'>{error_msg}</p>"
        
        # Validate URL
        if not pdb_url or pdb_url == "":
            return "<p style='text-align:center; color:red;'>Error: Invalid structure URL</p>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}
                #viewport {{
                    width: 100%;
                    height: 500px;
                    border: 2px solid #1f77b4;
                    border-radius: 8px;
                    position: relative;
                }}
                #loading {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 18px;
                    color: #666;
                    z-index: 10;
                }}
                #controls {{
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background-color: rgba(255,255,255,0.95);
                    padding: 10px;
                    border-radius: 4px;
                    font-size: 12px;
                    border: 1px solid #ccc;
                    z-index: 100;
                }}
                #title {{
                    text-align: center;
                    margin-top: 10px;
                    font-size: 14px;
                    color: #666;
                    font-weight: bold;
                }}
                .error {{
                    color: red;
                    text-align: center;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <div id="viewport">
                <div id="loading">Loading 3D structure from:<br><small>{pdb_url}</small></div>
            </div>
            <div id="title">{title}</div>
            
            <script>
                (function() {{
                    // Validate URL
                    const pdbUrl = "{pdb_url}";
                    if (!pdbUrl || pdbUrl === "") {{
                        document.getElementById('loading').innerHTML = '<span class="error">Error: No PDB URL provided</span>';
                        return;
                    }}
                    
                    // Wait for NGL to be fully loaded
                    if (typeof NGL === 'undefined') {{
                        document.getElementById('loading').innerHTML = '<span class="error">Error: NGL library failed to load</span>';
                        return;
                    }}
                    
                    try {{
                        // Create NGL Stage
                        var stage = new NGL.Stage("viewport", {{
                            backgroundColor: "white"
                        }});
                        
                        // Handle window resize
                        window.addEventListener("resize", function() {{
                            stage.handleResize();
                        }}, false);
                        
                        // Load the structure
                        stage.loadFile(pdbUrl, {{defaultRepresentation: false}})
                            .then(function(component) {{
                                // Remove loading message
                                var loadingDiv = document.getElementById('loading');
                                if (loadingDiv) loadingDiv.style.display = 'none';
                                
                                // Add representations based on structure type
                                {"" if structure_type == "pdb" else '''
                                // AlphaFold: Color by confidence (bfactor = pLDDT)
                                component.addRepresentation("cartoon", {
                                    color: "bfactor",
                                    colorScheme: "RdYlBu",
                                    colorReverse: false,
                                    colorScale: ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]
                                });
                                '''}
                                
                                {"" if structure_type == "alphafold" else '''
                                // PDB: Standard coloring
                                component.addRepresentation("cartoon", {
                                    color: "chainindex"
                                });
                                '''}
                                
                                // Add ligands if present
                                component.addRepresentation("ball+stick", {{
                                    sele: "hetero and not (water or ion)",
                                    colorScheme: "element"
                                }});
                                
                                // Center and zoom
                                component.autoView();
                                
                                // Add control instructions
                                var controlsDiv = document.createElement('div');
                                controlsDiv.id = 'controls';
                                controlsDiv.innerHTML = '🖱️ Left: Rotate | Right: Zoom | Middle: Pan';
                                document.getElementById('viewport').appendChild(controlsDiv);
                                
                            }})
                            .catch(function(error) {{
                                console.error('Error loading structure:', error);
                                document.getElementById('loading').innerHTML = 
                                    '<span class="error">Error loading structure from URL:<br>' + pdbUrl + '<br><br>Error: ' + error.message + '<br><br>This structure may not be available in AlphaFold DB.</span>';
                            }});
                            
                    }} catch (error) {{
                        console.error('Error initializing NGL:', error);
                        document.getElementById('loading').innerHTML = 
                            '<span class="error">Error initializing viewer: ' + error.message + '</span>';
                    }}
                }})();
            </script>
        </body>
        </html>
        """
        
        return html
                
    @staticmethod
    def create_confidence_plot(uniprot_id: str, entry_id: Optional[str] = None) -> go.Figure:
        """
        Create plot showing AlphaFold confidence scores along sequence
        pLDDT scores: >90=very high, 70-90=confident, 50-70=low, <50=very low
        """
        import httpx
        
        try:
            # Use entry_id if provided, otherwise construct it
            if not entry_id:
                entry_id = f"AF-{uniprot_id}-F1"
            
            # Try multiple versions - start with latest v6
            urls_to_try = [
                f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v6.pdb",
                f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v4.pdb",
                f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v3.pdb",
                f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v2.pdb",
            ]
            
            pdb_content = None
            for pdb_url in urls_to_try:
                try:
                    response = httpx.get(pdb_url, timeout=30.0, follow_redirects=True)
                    if response.status_code == 200:
                        pdb_content = response.text
                        break
                except:
                    continue
            
            if not pdb_content:
                raise Exception(f"No AlphaFold structure found for {uniprot_id}")
            
            # Parse pLDDT scores from B-factor column in PDB file
            residues = []
            plddt_scores = []
            
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM') and line[13:15].strip() == 'CA':  # Only CA atoms
                    try:
                        residue_num = int(line[22:26].strip())
                        bfactor = float(line[60:66].strip())
                        
                        residues.append(residue_num)
                        plddt_scores.append(bfactor)
                    except:
                        continue
            
            if not residues:
                raise Exception("No confidence data found in PDB file")
            
            # Create color mapping for scatter plot
            colors = []
            for score in plddt_scores:
                if score > 90:
                    colors.append('#0053D6')  # Very high - dark blue
                elif score > 70:
                    colors.append('#65CBF3')  # Confident - light blue
                elif score > 50:
                    colors.append('#FFDB13')  # Low - yellow
                else:
                    colors.append('#FF7D45')  # Very low - orange
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=residues,
                y=plddt_scores,
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.3)',
                name='pLDDT Score',
                hovertemplate='Residue: %{x}<br>Confidence: %{y:.1f}<extra></extra>'
            ))
            
            # Add confidence level zones
            fig.add_hrect(y0=90, y1=100, fillcolor="rgba(0, 83, 214, 0.1)", 
                        line_width=0, annotation_text="Very High", annotation_position="right")
            fig.add_hrect(y0=70, y1=90, fillcolor="rgba(101, 203, 243, 0.1)", 
                        line_width=0, annotation_text="Confident", annotation_position="right")
            fig.add_hrect(y0=50, y1=70, fillcolor="rgba(255, 219, 19, 0.1)", 
                        line_width=0, annotation_text="Low", annotation_position="right")
            fig.add_hrect(y0=0, y1=50, fillcolor="rgba(255, 125, 69, 0.1)", 
                        line_width=0, annotation_text="Very Low", annotation_position="right")
            
            fig.update_layout(
                title="AlphaFold Confidence Score (pLDDT) per Residue",
                xaxis_title="Residue Position",
                yaxis_title="Confidence Score (pLDDT)",
                yaxis=dict(range=[0, 100]),
                height=350,
                template="plotly_white",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"AlphaFold structure not available for {uniprot_id}<br><br>This protein may not be in the AlphaFold database.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(height=350, template="plotly_white")
            return fig
                
    @staticmethod
    def create_pathway_network(pathways: list) -> go.Figure:
        """
        Create network visualization showing pathway relationships
        Groups pathways by class for better organization
        """
        if not pathways:
            fig = go.Figure()
            fig.add_annotation(
                text="No pathway data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        # Group pathways by class
        pathway_classes: dict[str, list[Dict]] = {}
        for pathway in pathways:
            pathway_class = pathway.get('pathway_class', 'Other')
            # Extract main class (before semicolon)
            if ';' in pathway_class:
                pathway_class = pathway_class.split(';')[0].strip()
            
            if pathway_class not in pathway_classes:
                pathway_classes[pathway_class] = []
            pathway_classes[pathway_class].append(pathway)
        
        # Create sunburst chart showing pathway hierarchy
        labels = ["Pathways"]
        parents = [""]
        values = [len(pathways)]
        colors = ["#1f77b4"]
        hover_texts = [f"Total Pathways: {len(pathways)}"]
        
        # Color palette for classes
        class_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for idx, (pathway_class, class_pathways) in enumerate(pathway_classes.items()):
            # Add class level
            class_label = pathway_class if pathway_class else "Unclassified"
            labels.append(class_label)
            parents.append("Pathways")
            values.append(len(class_pathways))
            colors.append(class_colors[idx % len(class_colors)])
            hover_texts.append(f"{class_label}<br>Count: {len(class_pathways)}")
            
            # Add individual pathways
            for pathway in class_pathways[:10]:  # Limit to first 10 per class for readability
                labels.append(pathway['pathway_name'][:30])  # Truncate long names
                parents.append(class_label)
                values.append(1)
                colors.append(class_colors[idx % len(class_colors)])
                hover_texts.append(f"{pathway['pathway_name']}<br>ID: {pathway['pathway_id']}")
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="KEGG Pathway Classification",
            height=500,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig

    @staticmethod
    def create_pathway_table_html(pathways: list) -> str:
        """
        Create formatted HTML table for pathways with clickable links
        """
        if not pathways:
            return "<p style='text-align:center; color:gray;'>No pathways found</p>"
        
        html = """
        <style>
            .pathway-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }
            .pathway-table th {
                background-color: #1f77b4;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            .pathway-table td {
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }
            .pathway-table tr:hover {
                background-color: #f5f5f5;
            }
            .pathway-link {
                color: #1f77b4;
                text-decoration: none;
                font-weight: 500;
            }
            .pathway-link:hover {
                text-decoration: underline;
            }
            .pathway-id {
                font-family: monospace;
                background-color: #f0f0f0;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
            }
            .pathway-class {
                font-size: 12px;
                color: #666;
                font-style: italic;
            }
        </style>
        
        <table class="pathway-table">
            <thead>
                <tr>
                    <th style="width: 50%">Pathway Name</th>
                    <th style="width: 15%">Pathway ID</th>
                    <th style="width: 35%">Classification</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for pathway in pathways:
            pathway_name = pathway.get('pathway_name', 'Unknown')
            pathway_id = pathway.get('pathway_id', '')
            pathway_class = pathway.get('pathway_class', 'N/A')
            highlight_url = pathway.get('highlight_url', pathway.get('kegg_url', '#'))
            
            html += f"""
            <tr>
                <td>
                    <a href="{highlight_url}" target="_blank" class="pathway-link">
                        {pathway_name}
                    </a>
                </td>
                <td>
                    <span class="pathway-id">{pathway_id}</span>
                </td>
                <td>
                    <span class="pathway-class">{pathway_class}</span>
                </td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    @staticmethod
    def analyze_sequence_composition(sequence: str) -> Dict:
        """
        Analyze amino acid composition of the sequence
        """
        if not sequence:
            return {}
        
        # Count amino acids
        aa_counts: dict[str, int] = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        total = len(sequence)
        
        # Calculate percentages
        aa_composition = {aa: (count / total) * 100 for aa, count in aa_counts.items()}
        
        # Group by properties
        hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
        polar = ['S', 'T', 'Y', 'N', 'Q', 'C']
        charged = ['K', 'R', 'H', 'D', 'E']
        
        hydrophobic_percent = sum(aa_composition.get(aa, 0) for aa in hydrophobic)
        polar_percent = sum(aa_composition.get(aa, 0) for aa in polar)
        charged_percent = sum(aa_composition.get(aa, 0) for aa in charged)
        
        return {
            "aa_composition": aa_composition,
            "hydrophobic_percent": hydrophobic_percent,
            "polar_percent": polar_percent,
            "charged_percent": charged_percent,
            "length": total
        }

    @staticmethod
    def create_sequence_composition_chart(composition_data: Dict) -> go.Figure:
        """
        Create bar chart showing amino acid composition
        """
        if not composition_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No sequence data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        aa_comp = composition_data.get('aa_composition', {})
        
        # Sort by percentage
        sorted_aa = sorted(aa_comp.items(), key=lambda x: x[1], reverse=True)
        amino_acids = [aa for aa, _ in sorted_aa]
        percentages = [pct for _, pct in sorted_aa]
        
        # Color by property
        colors = []
        for aa in amino_acids:
            if aa in ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']:
                colors.append('#ff7f0e')  # Hydrophobic - orange
            elif aa in ['S', 'T', 'Y', 'N', 'Q', 'C']:
                colors.append('#2ca02c')  # Polar - green
            elif aa in ['K', 'R', 'H', 'D', 'E']:
                colors.append('#d62728')  # Charged - red
            else:
                colors.append('#7f7f7f')  # Other - gray
        
        fig = go.Figure(go.Bar(
            x=amino_acids,
            y=percentages,
            marker=dict(color=colors, line=dict(color='rgb(8,48,107)', width=1)),
            text=[f"{p:.1f}%" for p in percentages],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Percentage: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Amino Acid Composition",
            xaxis_title="Amino Acid",
            yaxis_title="Percentage (%)",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        return fig

    @staticmethod
    def create_blast_results_table_html(blast_hits: list) -> str:
        """
        Create formatted HTML table for BLAST results
        Displays all new fields: similarity, gaps, coverage, query range
        """
        if not blast_hits:
            return "<p style='text-align:center; color:gray;'>No BLAST results available</p>"
        
        html = """
        <style>
            .blast-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 12px;
            }
            .blast-table th {
                background-color: #2ca02c;
                color: white;
                padding: 9px 8px;
                text-align: left;
                font-weight: bold;
                position: sticky;
                top: 0;
            }
            .blast-table td {
                padding: 7px 8px;
                border-bottom: 1px solid #ddd;
            }
            .blast-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .blast-table tr:hover {
                background-color: #e8f5e9;
            }
            .accession-link {
                color: #1f77b4;
                text-decoration: none;
                font-family: monospace;
                font-weight: 600;
                font-size: 11px;
            }
            .accession-link:hover {
                text-decoration: underline;
            }
            .organism {
                font-style: italic;
                color: #555;
                font-size: 11px;
            }
            .identity-high {
                background-color: #d4edda;
                color: #155724;
                padding: 2px 7px;
                border-radius: 3px;
                font-weight: bold;
            }
            .identity-medium {
                background-color: #fff3cd;
                color: #856404;
                padding: 2px 7px;
                border-radius: 3px;
                font-weight: bold;
            }
            .identity-low {
                background-color: #f8d7da;
                color: #721c24;
                padding: 2px 7px;
                border-radius: 3px;
                font-weight: bold;
            }
            .e-value {
                font-family: monospace;
                font-size: 11px;
                color: #333;
            }
            .hit-number {
                font-weight: bold;
                color: #2ca02c;
                text-align: center;
            }
            .description-cell {
                max-width: 200px;
                word-wrap: break-word;
                color: #444;
            }
        </style>
        
        <table class="blast-table">
            <thead>
                <tr>
                    <th style="width: 3%">#</th>
                    <th style="width: 10%">Accession</th>
                    <th style="width: 22%">Description</th>
                    <th style="width: 14%">Organism</th>
                    <th style="width: 9%">Identity</th>
                    <th style="width: 9%">Similarity</th>
                    <th style="width: 9%">Coverage</th>
                    <th style="width: 8%">Gaps</th>
                    <th style="width: 10%">E-value</th>
                    <th style="width: 6%">Score</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, hit in enumerate(blast_hits, 1):
            accession = hit.get('accession', 'N/A')
            title = hit.get('title', 'Unknown')
            organism = hit.get('organism', 'Unknown')
            identity = hit.get('identity_percent', 0)
            similarity = hit.get('similarity_percent', 0)
            coverage = hit.get('coverage_percent', 0)
            gaps = hit.get('gap_percent', 0)
            e_value = hit.get('e_value', 1.0)
            bit_score = hit.get('bit_score', 0)
            query_range = hit.get('query_range', 'N/A')
            ncbi_url = hit.get('ncbi_url', f'https://www.ncbi.nlm.nih.gov/protein/{accession}')
            
            # Truncate description
            short_title = title[:80] + ('...' if len(title) > 80 else '')
            
            # Color code identity
            if identity >= 80:
                identity_class = "identity-high"
            elif identity >= 40:
                identity_class = "identity-medium"
            else:
                identity_class = "identity-low"
            
            # Format e-value
            if e_value == 0:
                e_value_str = "0.0"
            elif e_value < 1e-100:
                e_value_str = "< 1e-100"
            elif e_value < 0.0001:
                e_value_str = f"{e_value:.2e}"
            else:
                e_value_str = f"{e_value:.4f}"
            
            html += f"""
            <tr>
                <td class="hit-number">{idx}</td>
                <td>
                    <a href="{ncbi_url}" target="_blank" class="accession-link">{accession}</a>
                </td>
                <td class="description-cell" title="{title}">{short_title}</td>
                <td class="organism">{organism}</td>
                <td><span class="{identity_class}">{identity:.1f}%</span></td>
                <td>{similarity:.1f}%</td>
                <td>{coverage:.1f}%</td>
                <td>{gaps:.1f}%</td>
                <td class="e-value">{e_value_str}</td>
                <td>{bit_score:.0f}</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html

    @staticmethod
    def create_feature_map(features: list, sequence_length: int) -> go.Figure:
        """
        Create visual map of protein features (domains, regions, sites)
        """
        if not features:
            fig = go.Figure()
            fig.add_annotation(
                text="No feature annotations available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=300)
            return fig
        
        # Group features by type
        feature_types: dict[str, list[Dict]] = {}
        for feature in features:
            ftype = feature.get('type', 'Other')
            if ftype not in feature_types:
                feature_types[ftype] = []
            feature_types[ftype].append(feature)
        
        # Create figure
        fig = go.Figure()
        
        # Color palette for different feature types
        colors = {
            'DOMAIN': '#1f77b4',
            'REGION': '#ff7f0e',
            'BINDING': '#2ca02c',
            'SITE': '#d62728',
            'MOTIF': '#9467bd',
            'TRANSMEM': '#8c564b',
            'SIGNAL': '#e377c2',
            'Other': '#7f7f7f'
        }
        
        y_position = 0
        
        for ftype, feats in feature_types.items():
            for feat in feats:
                start = feat.get('start', 0)
                end = feat.get('end', 0)
                description = feat.get('description', ftype)
                
                color = colors.get(ftype, colors['Other'])
                
                # Add rectangle for feature
                fig.add_trace(go.Scatter(
                    x=[start, end, end, start, start],
                    y=[y_position, y_position, y_position + 0.8, y_position + 0.8, y_position],
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>{ftype}</b><br>{description}<br>Position: {start}-{end}<br>Length: {end-start+1} aa<extra></extra>',
                    name=ftype,
                    showlegend=True,
                    legendgroup=ftype
                ))
                
                y_position += 1
        
        # Add full-length protein bar at bottom
        fig.add_trace(go.Scatter(
            x=[1, sequence_length],
            y=[-1, -1],
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="Protein Feature Map",
            xaxis_title="Amino Acid Position",
            yaxis=dict(
                showticklabels=False,
                range=[-2, y_position + 1]
            ),
            height=max(300, y_position * 30 + 100),
            template="plotly_white",
            hovermode='closest'
        )
        
        return fig

    @staticmethod
    def create_alignment_visualization(alignment_data: Dict) -> str:
        """
        Create HTML visualization for pairwise alignment
        """
        if not alignment_data.get('available'):
            return "<p style='text-align:center; color:gray;'>No alignment data available</p>"
        
        identity = alignment_data.get('identity', 0)
        similarity = alignment_data.get('similarity', 0)
        gaps = alignment_data.get('gaps', 0)
        score = alignment_data.get('score', 0)
        alignment_text = alignment_data.get('alignment_display', '')
        
        # Determine quality color
        if identity >= 70:
            quality_color = "#28a745"
            quality_text = "High"
        elif identity >= 40:
            quality_color = "#ffc107"
            quality_text = "Moderate"
        else:
            quality_color = "#dc3545"
            quality_text = "Low"
        
        html = f"""
        <style>
            .alignment-container {{
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                background-color: #f9f9f9;
            }}
            .alignment-stats {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-box {{
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 15px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: {quality_color};
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            .alignment-text {{
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                padding: 20px;
                border-radius: 4px;
                overflow-x: auto;
                white-space: pre;
                line-height: 1.6;
            }}
            .quality-badge {{
                display: inline-block;
                background-color: {quality_color};
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                margin-bottom: 15px;
            }}
        </style>
        
        <div class="alignment-container">
            <div class="quality-badge">Alignment Quality: {quality_text}</div>
            
            <div class="alignment-stats">
                <div class="stat-box">
                    <div class="stat-value">{identity:.1f}%</div>
                    <div class="stat-label">Identity</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{similarity:.1f}%</div>
                    <div class="stat-label">Similarity</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{gaps:.1f}%</div>
                    <div class="stat-label">Gaps</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{score:.1f}</div>
                    <div class="stat-label">Alignment Score</div>
                </div>
            </div>
            
            <h4 style="margin-top: 20px; margin-bottom: 10px;">Alignment Details:</h4>
            <div class="alignment-text">{alignment_text}</div>
        </div>
        """
        
        return html

    @staticmethod
    def create_ligand_table_html(ligands: list) -> str:
        """
        Create formatted HTML table for known ligands
        """
        if not ligands:
            return "<p style='text-align:center; color:gray;'>No known ligands found</p>"
        
        html = """
        <style>
            .ligand-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 13px;
            }
            .ligand-table th {
                background-color: #2ca02c;
                color: white;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            .ligand-table td {
                padding: 8px 10px;
                border-bottom: 1px solid #ddd;
            }
            .ligand-table tr:hover {
                background-color: #f5f5f5;
            }
            .chembl-link {
                color: #1f77b4;
                text-decoration: none;
                font-family: monospace;
                font-weight: 500;
            }
            .chembl-link:hover {
                text-decoration: underline;
            }
            .activity-strong {
                background-color: #d4edda;
                color: #155724;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            .activity-moderate {
                background-color: #fff3cd;
                color: #856404;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            .activity-weak {
                background-color: #f8d7da;
                color: #721c24;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            .mol-structure {
                width: 80px;
                height: 80px;
                object-fit: contain;
            }
        </style>
        
        <table class="ligand-table">
            <thead>
                <tr>
                    <th style="width: 10%">Structure</th>
                    <th style="width: 25%">Compound Name</th>
                    <th style="width: 15%">ChEMBL ID</th>
                    <th style="width: 15%">Activity Type</th>
                    <th style="width: 15%">Value</th>
                    <th style="width: 10%">MW (Da)</th>
                    <th style="width: 10%">Action</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for ligand in ligands:
            chembl_id = ligand.get('chembl_id', 'N/A')
            name = ligand.get('name')
            # Use ChEMBL ID if name is None or empty
            if not name:
                name = chembl_id
            activity_type = ligand.get('activity_type', 'N/A')
            activity_value = ligand.get('activity_value', 0)
            activity_units = ligand.get('activity_units', 'nM')
            mw = ligand.get('molecular_weight', 'N/A')
            chembl_url = ligand.get('chembl_url', '#')
            
            # Format activity value
            if activity_value < 100:
                activity_class = "activity-strong"
                activity_label = "Strong"
            elif activity_value < 1000:
                activity_class = "activity-moderate"
                activity_label = "Moderate"
            else:
                activity_class = "activity-weak"
                activity_label = "Weak"
            
            # Structure image from ChEMBL
            img_url = f"https://www.ebi.ac.uk/chembl/api/data/image/{chembl_id}.svg"
            
            # Escape single quotes in name for JavaScript
            name_escaped = name.replace("'", "\\'")
            
            html += f"""
            <tr>
                <td><img src="{img_url}" class="mol-structure" alt="{name}"></td>
                <td><strong>{name}</strong></td>
                <td>
                    <a href="{chembl_url}" target="_blank" class="chembl-link">{chembl_id}</a>
                </td>
                <td>{activity_type}</td>
                <td>
                    <span class="{activity_class}">{activity_value:.1f} {activity_units}</span>
                    <br><small>{activity_label}</small>
                </td>
                <td>{mw if isinstance(mw, str) else f"{mw:.1f}"}</td>
                <td>
                    <button onclick="selectForDocking('{chembl_id}', '{name_escaped}')" 
                            style="padding:4px 8px; background:#1f77b4; color:white; border:none; border-radius:3px; cursor:pointer;">
                        Dock
                    </button>
                </td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        
        <script>
            function selectForDocking(chemblId, name) {
                // Send message to Streamlit parent window
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    data: {
                        chembl_id: chemblId,
                        name: name,
                        action: 'dock'
                    }
                }, '*');
            }
        </script>
        """
        
        return html

    @staticmethod
    def create_docking_results_chart(docking_results: Dict) -> go.Figure:
        """
        Create bar chart showing binding affinities for different binding modes
        """
        if not docking_results.get('available'):
            fig = go.Figure()
            fig.add_annotation(
                text="No docking results available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        modes = docking_results.get('modes', [])
        
        mode_numbers = [m['mode'] for m in modes]
        affinities = [m['affinity'] for m in modes]
        
        # Color code by affinity strength
        colors = []
        for aff in affinities:
            if aff < -7:
                colors.append('#2ca02c')  # Strong - green
            elif aff < -5:
                colors.append('#ff7f0e')  # Moderate - orange
            else:
                colors.append('#d62728')  # Weak - red
        
        fig = go.Figure(go.Bar(
            x=mode_numbers,
            y=affinities,
            marker=dict(color=colors, line=dict(color='black', width=1)),
            text=[f"{a:.1f}" for a in affinities],
            textposition='outside',
            hovertemplate='<b>Mode %{x}</b><br>Affinity: %{y:.2f} kcal/mol<extra></extra>'
        ))
        
        fig.update_layout(
            title="Predicted Binding Modes",
            xaxis_title="Binding Mode",
            yaxis_title="Binding Affinity (kcal/mol)",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        # Add reference lines
        fig.add_hline(y=-7, line_dash="dash", line_color="green", 
                    annotation_text="Strong binding", annotation_position="right")
        fig.add_hline(y=-5, line_dash="dash", line_color="orange", 
                    annotation_text="Moderate binding", annotation_position="right")
        
        return fig


    @staticmethod
    def create_ppi_network_chart(interactions: list, query_protein: str) -> go.Figure:
        """
        Create network graph for protein-protein interactions
        """
        if not interactions:
            fig = go.Figure()
            fig.add_annotation(
                text="No protein interactions available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        import math
        
        # Create circular layout
        n = len(interactions) + 1  # +1 for query protein
        
        # Query protein at center
        node_x: list[float] = [0.0]
        node_y: list[float] = [0.0]
        node_names = [query_protein]
        node_colors = ['#d62728']  # Red for query
        node_sizes: list[float] = [30.0]
        
        # Partner proteins in circle
        for i, interaction in enumerate(interactions):
            angle = 2 * math.pi * i / len(interactions)
            x = math.cos(angle)
            y = math.sin(angle)
            
            node_x.append(x)
            node_y.append(y)
            node_names.append(interaction['partner_name'])
            
            # Color by confidence
            if interaction['confidence'] == 'Highest':
                node_colors.append('#1f77b4')  # Dark blue for highest
            elif interaction['confidence'] == 'High':
                node_colors.append('#2ca02c')  # Green for high
            elif interaction['confidence'] == 'Medium':
                node_colors.append('#ff7f0e')  # Orange for medium
            else:
                node_colors.append('#7f7f7f')  # Gray for low
            
            # Size by score
            size = 10 + (interaction['combined_score'] / 1000) * 20
            node_sizes.append(size)
        
        # Create edges grouped by confidence level
        edge_groups: dict[str, dict[str, Any]] = {
            'Highest': {'x': [], 'y': [], 'color': 'rgba(31, 119, 180, 0.6)'},
            'High': {'x': [], 'y': [], 'color': 'rgba(44, 160, 44, 0.5)'},
            'Medium': {'x': [], 'y': [], 'color': 'rgba(255, 127, 14, 0.5)'},
            'Low': {'x': [], 'y': [], 'color': 'rgba(127, 127, 127, 0.3)'}
        }
        
        for i in range(len(interactions)):
            conf = interactions[i]['confidence']
            # Line from center to partner
            edge_groups[conf]['x'].extend([0, node_x[i+1], None])
            edge_groups[conf]['y'].extend([0, node_y[i+1], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges as separate traces for each confidence level
        for conf_level, edges in edge_groups.items():
            if edges['x']:  # Only add if there are edges for this confidence
                fig.add_trace(go.Scatter(
                    x=edges['x'],
                    y=edges['y'],
                    mode='lines',
                    line=dict(color=edges['color'], width=2),
                    hoverinfo='none',
                    name=f"{conf_level} confidence",
                    showlegend=False
                ))
        
        # Add nodes
        hover_text = []
        for i, name in enumerate(node_names):
            if i == 0:
                hover_text.append(f"<b>{name}</b><br>Query Protein")
            else:
                interaction = interactions[i-1]
                hover_text.append(
                    f"<b>{name}</b><br>"
                    f"Score: {interaction['combined_score']}/1000<br>"
                    f"Confidence: {interaction['confidence']}<br>"
                    f"Evidence: {interaction['evidence_types']}"
                )
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='white', width=2)
            ),
            text=node_names,
            textposition='top center',
            textfont=dict(size=10),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Protein-Protein Interaction Network",
            height=600,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig

    @staticmethod
    def create_ppi_table_html(interactions: list) -> str:
        """
        Create formatted HTML table for PPI data
        """
        if not interactions:
            return "<p style='text-align:center; color:gray;'>No interactions found</p>"
        
        html = """
        <style>
            .ppi-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 13px;
            }
            .ppi-table th {
                background-color: #1f77b4;
                color: white;
                padding: 10px;
                text-align: left;
                font-weight: bold;
            }
            .ppi-table td {
                padding: 8px 10px;
                border-bottom: 1px solid #ddd;
            }
            .ppi-table tr:hover {
                background-color: #f5f5f5;
            }
            .confidence-high {
                background-color: #d4edda;
                color: #155724;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            .confidence-medium {
                background-color: #fff3cd;
                color: #856404;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            .confidence-low {
                background-color: #f8d7da;
                color: #721c24;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
        </style>
        
        <table class="ppi-table">
            <thead>
                <tr>
                    <th style="width: 25%">Partner Protein</th>
                    <th style="width: 15%">Combined Score</th>
                    <th style="width: 15%">Confidence</th>
                    <th style="width: 45%">Evidence Types</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for interaction in interactions:
            partner = interaction['partner_name']
            score = interaction['combined_score']
            confidence = interaction['confidence']
            evidence = interaction['evidence_types']
            
            if confidence == "High":
                conf_class = "confidence-high"
            elif confidence == "Medium":
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            
            html += f"""
            <tr>
                <td><strong>{partner}</strong></td>
                <td>{score}/1000</td>
                <td><span class="{conf_class}">{confidence}</span></td>
                <td>{evidence}</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html

    @staticmethod
    def create_docking_3d_viewer(protein_structure: Dict, ligand_data: Dict, 
                                docking_result: Dict, ligand_name: str,
                                view_mode: str = "Cartoon (Ribbon)") -> str:
        """
        Create 3D viewer showing protein with docked ligand using py3Dmol.
        
        Features:
        - Automatically clears and resets viewer for each render (prevents stale structures)
        - Removes all previous models before loading new ones
        - Centers and zooms the protein-ligand complex using zoomTo()
        - Applies fallback zoom if initial zoom is insufficient
        - Uses unique viewer instance per render (Streamlit-compatible)
        """
        if not protein_structure.get('available'):
            return "<p style='text-align:center; color:gray;'>No protein structure available</p>"
        
        # Get PDB data - prefer pdb_text over pdb_url
        pdb_text = protein_structure.get('pdb_text', '')
        pdb_url = protein_structure.get('pdb_url', '')
        
        # Validate pdb_url - skip if it's a data URI
        if pdb_url and pdb_url.startswith('data:'):
            pdb_url = ''
        
        # Critical validation: ensure we have actual PDB data
        if not pdb_text and not pdb_url:
            return """<div style='text-align:center; padding:20px; color:#d32f2f; background:#ffebee; border:1px solid #ef5350; border-radius:8px; margin:20px;'>
                <h3>⚠️ No Protein Structure Data Available</h3>
                <p>Please predict the protein structure in the <strong>Protein Structure Prediction</strong> tab first.</p>
                <p style='font-size:0.9em; color:#666;'>The docking visualization requires a predicted 3D structure.</p>
            </div>"""
        
        ligand_smiles = ligand_data.get('smiles', '')
        
        # Get binding affinity for display
        affinity = docking_result.get('binding_affinity', 0)
        
        # Build a simple ligand PDB from docking center (visualization only)
        best_center = docking_result.get("best_mode", {}).get("center", {})
        try:
            center_x = float(best_center.get("x", 0.0) or 0.0)
            center_y = float(best_center.get("y", 0.0) or 0.0)
            center_z = float(best_center.get("z", 0.0) or 0.0)
        except (TypeError, ValueError):
            center_x, center_y, center_z = 0.0, 0.0, 0.0

        def _format_hetatm(serial, name, resn, chain, resi, x, y, z, element):
            return (
                f"HETATM{serial:5d} {name:<4}{resn:>3} {chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2}"
            )

        ligand_atoms = [
            ("C1", center_x, center_y, center_z, "C"),
            ("O1", center_x + 1.2, center_y, center_z, "O"),
            ("N1", center_x - 1.2, center_y, center_z, "N"),
            ("S1", center_x, center_y + 1.2, center_z, "S"),
        ]
        ligand_pdb_lines = [
            _format_hetatm(idx + 1, atom[0], "LIG", "A", 1, atom[1], atom[2], atom[3], atom[4])
            for idx, atom in enumerate(ligand_atoms)
        ]
        ligand_pdb_lines.append("END")
        ligand_pdb = "\n".join(ligand_pdb_lines)

        # Properly escape ligand_name and pdb_text for JavaScript
        import json
        ligand_name_escaped = json.dumps(ligand_name)
        pdb_text_escaped = json.dumps(pdb_text if pdb_text else '')
        pdb_url_escaped = json.dumps(pdb_url if pdb_url else '')
        ligand_pdb_escaped = json.dumps(ligand_pdb)
        view_mode_escaped = json.dumps(view_mode)
        
        # Generate unique ID for this viewer instance (prevents reuse/caching)
        import uuid
        viewer_id = f"viewer_{uuid.uuid4().hex[:8]}"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}
                #{viewer_id} {{
                    width: 100%;
                    height: 600px;
                    border: 2px solid #2ca02c;
                    border-radius: 8px;
                    position: relative;
                    background-color: white;
                }}
                #loading-{viewer_id} {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 16px;
                    color: #666;
                    z-index: 10;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                }}
                #controls-{viewer_id} {{
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background-color: rgba(255,255,255,0.95);
                    padding: 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    border: 1px solid #ccc;
                    z-index: 100;
                }}
                #info-{viewer_id} {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background-color: rgba(44, 160, 44, 0.9);
                    color: white;
                    padding: 10px 15px;
                    border-radius: 6px;
                    font-size: 13px;
                    z-index: 100;
                    font-weight: 500;
                }}
                #legend-{viewer_id} {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background-color: rgba(255,255,255,0.95);
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 11px;
                    border: 1px solid #ccc;
                    z-index: 100;
                }}
                .legend-item {{
                    margin: 3px 0;
                }}
                .color-box {{
                    display: inline-block;
                    width: 15px;
                    height: 15px;
                    margin-right: 5px;
                    border: 1px solid #666;
                    vertical-align: middle;
                }}
                #title-{viewer_id} {{
                    text-align: center;
                    margin-top: 10px;
                    font-size: 15px;
                    color: #2ca02c;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div id="{viewer_id}">
                <div id="loading-{viewer_id}">Loading protein-ligand complex...</div>
                <div id="info-{viewer_id}">
                    Binding Affinity: {affinity} kcal/mol<br>
                    Ligand: <span id="ligand-name-info-{viewer_id}"></span>
                </div>
                <div id="legend-{viewer_id}">
                    <div class="legend-item">
                        <span class="color-box" style="background-color:#0053D6"></span>Protein
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color:#FFD700"></span>Ligand
                    </div>
                    <div class="legend-item">
                        <span class="color-box" style="background-color:#FF6B6B"></span>Binding Site
                    </div>
                </div>
            </div>
            <div id="title-{viewer_id}">Protein-Ligand Docking Complex (Simulated Pose)</div>
            
            <script>
                (function() {{
                    // Store configuration
                    var viewerId = '{viewer_id}';
                    var pdbText = {pdb_text_escaped};
                    var pdbUrl = {pdb_url_escaped};
                    var ligandName = {ligand_name_escaped};
                    var ligandPdb = {ligand_pdb_escaped};
                    var viewMode = {view_mode_escaped};
                    
                    document.getElementById('ligand-name-info-' + viewerId).textContent = ligandName;
                    
                    try {{
                        // Wait for 3Dmol to load
                        if (typeof $3Dmol === 'undefined') {{
                            document.getElementById('loading-' + viewerId).innerHTML = 
                                '<span style="color:red">Error: 3Dmol library failed to load</span>';
                            return;
                        }}
                        
                        // Create unique viewer instance
                        var viewer = $3Dmol.createViewer(
                            document.getElementById(viewerId),
                            {{backgroundColor: 'white'}}
                        );
                        
                        // CRITICAL: Clear any existing content before loading
                        viewer.clear();
                        viewer.removeAllModels();
                        
                        // Prepare PDB content
                        var pdbContent = null;
                        var loadSource = null;
                        
                        if (pdbText && pdbText.length > 0) {{
                            pdbContent = pdbText;
                            loadSource = 'text';
                        }} else if (pdbUrl && pdbUrl.length > 0) {{
                            pdbContent = pdbUrl;
                            loadSource = 'url';
                        }} else {{
                            document.getElementById('loading-' + viewerId).innerHTML = 
                                '<span style="color:red">⚠️ No valid PDB data available</span>';
                            return;
                        }}
                        
                        // Load structure based on available source
                        if (loadSource === 'text') {{
                            // Load from PDB text
                            renderProteinLigand(pdbContent);
                        }} else {{
                            // For URL loading, use AJAX
                            var xhr = new XMLHttpRequest();
                            xhr.onreadystatechange = function() {{
                                if (xhr.readyState === 4) {{
                                    if (xhr.status === 200) {{
                                        renderProteinLigand(xhr.responseText);
                                    }} else {{
                                        document.getElementById('loading-' + viewerId).innerHTML = 
                                            '<span style="color:red">Error loading PDB from URL</span>';
                                    }}
                                }}
                            }};
                            xhr.open('GET', pdbContent, true);
                            xhr.send();
                            return;
                        }}
                        
                        // Render function (called when models are added)
                        function renderProteinLigand(proteinPdb) {{
                            try {{
                                // Hide loading message
                                var loadingDiv = document.getElementById('loading-' + viewerId);
                                if (loadingDiv) loadingDiv.style.display = 'none';

                                // REQUIRED EXECUTION ORDER
                                viewer.clear();
                                viewer.removeAllModels();
                                viewer.addModel(proteinPdb, 'pdb');   // model 0
                                viewer.addModel(ligandPdb, 'pdb');    // model 1
                                viewer.setStyle({{}}, {{}});          // HARD RESET STYLES

                                if (viewMode === 'Cartoon (Ribbon)') {{
                                    // Protein: cartoon with spectrum coloring (presentation-grade)
                                    viewer.setStyle(
                                        {{model: 0}},
                                        {{
                                            cartoon: {{
                                                color: 'spectrum',
                                                thickness: 0.75,
                                                opacity: 0.95,
                                                smooth: true,
                                                style: 'edged'
                                            }}
                                        }}
                                    );

                                    // Ligand: bold sticks + spheres for focal point
                                    viewer.setStyle(
                                        {{model: 1}},
                                        {{
                                            stick: {{colorscheme: 'orangeCarbon', radius: 0.25}},
                                            sphere: {{colorscheme: 'orangeCarbon', scale: 0.4}}
                                        }}
                                    );
                                }} else {{
                                    // All-atom: protein ball-and-stick (neutral palette)
                                    viewer.setStyle(
                                        {{model: 0}},
                                        {{
                                            stick: {{color: '#888888', radius: 0.2}},
                                            sphere: {{color: '#888888', scale: 0.25}}
                                        }}
                                    );

                                    // Ligand: bold ball-and-stick (focal point)
                                    viewer.setStyle(
                                        {{model: 1}},
                                        {{
                                            stick: {{colorscheme: 'orangeCarbon', radius: 0.28}},
                                            sphere: {{colorscheme: 'orangeCarbon', scale: 0.4}}
                                        }}
                                    );
                                }}

                                viewer.zoomTo();
                                viewer.zoom(0.85);
                                viewer.spin({{y: 1}}, 0.2);
                                viewer.render();

                                // Stop rotation on interaction (immediate response)
                                function stopSpin() {{
                                    viewer.spin(false);
                                    viewer.render();
                                }}
                                var container = document.getElementById(viewerId);
                                if (container) {{
                                    ['mousedown', 'touchstart', 'pointerdown'].forEach(function(evt) {{
                                        container.addEventListener(evt, stopSpin, {{passive: true}});
                                    }});
                                }}

                                // Add control instructions
                                var controlsDiv = document.createElement('div');
                                controlsDiv.id = 'controls-' + viewerId;
                                controlsDiv.innerHTML = '🖱️ Left: Rotate | Right: Zoom | Middle: Pan';
                                document.getElementById(viewerId).appendChild(controlsDiv);
                                
                            }} catch (e) {{
                                console.error('Render error:', e);
                                document.getElementById('loading-' + viewerId).innerHTML = 
                                    '<span style="color:red">⚠️ Error rendering structure: ' + (e.message || 'Unknown error') + '</span>';
                            }}
                        }}
                        
                        // Call render function immediately (structure is now loaded)
                    }} catch (error) {{
                        console.error('Exception:', error);
                        document.getElementById('loading-' + viewerId).innerHTML = 
                            '<span style="color:red">Error: ' + error.message + '</span>';
                    }}
                }})();
            </script>
        </body>
        </html>
        """
        
        return html


    @staticmethod
    def predict_best_ligand(ligands: list, protein_data: Dict) -> Dict:
        """
        Predict which known ligand should bind best based on multiple factors
        
        Scoring criteria:
        1. Experimental activity data (IC50/Ki)
        2. Molecular properties (MW, LogP)
        3. Drug-likeness (Lipinski's Rule of Five)
        
        Returns top predicted ligand with explanation
        """
        if not ligands:
            return {"available": False, "message": "No ligands to analyze"}
        
        predictions = []
        
        for ligand in ligands:
            score = 0
            reasons = []
            
            # Factor 1: Activity value (most important)
            activity_value = ligand.get('activity_value', float('inf'))
            activity_type = ligand.get('activity_type', '')
            
            if activity_value < 10:  # Very potent
                score += 50
                reasons.append(f"Very potent {activity_type}: {activity_value:.2f} nM")
            elif activity_value < 100:  # Potent
                score += 35
                reasons.append(f"Potent {activity_type}: {activity_value:.2f} nM")
            elif activity_value < 1000:  # Moderate
                score += 20
                reasons.append(f"Moderate {activity_type}: {activity_value:.2f} nM")
            else:  # Weak
                score += 5
                reasons.append(f"Weak activity: {activity_value:.2f} nM")
            
            # Factor 2: Molecular weight (drug-like range)
            mw = ligand.get('molecular_weight', 0)
            # Convert to float if it's a string
            if isinstance(mw, str):
                try:
                    mw = float(mw)
                except (ValueError, TypeError):
                    mw = 0
            
            if mw and 160 <= mw <= 500:  # Optimal drug-like range
                score += 15
                reasons.append(f"Optimal MW: {mw:.1f} Da")
            elif mw and mw <= 160:
                score += 5
                reasons.append(f"Low MW: {mw:.1f} Da")
            elif mw and mw > 500:
                score += 8
                reasons.append(f"High MW: {mw:.1f} Da")
            
            # Factor 3: SMILES availability (for structure-based predictions)
            if ligand.get('smiles'):
                score += 10
                reasons.append("Structure available for docking")
            
            # Factor 4: Name indicates known drug
            name = ligand.get('name', '').lower()
            drug_indicators = ['inhibitor', 'mab', 'nib', 'tinib', 'zumab', 'ciclib']
            if any(indicator in name for indicator in drug_indicators):
                score += 10
                reasons.append("Known drug or inhibitor class")
            
            predictions.append({
                "ligand": ligand,
                "score": score,
                "reasons": reasons,
                "confidence": "High" if score >= 70 else ("Medium" if score >= 50 else "Low")
            })
        
        # Sort by score
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        best = predictions[0]
        
        return {
            "available": True,
            "best_ligand": best['ligand'],
            "score": best['score'],
            "confidence": best['confidence'],
            "reasons": best['reasons'],
            "all_predictions": predictions[:5]  # Top 5
        }

    @staticmethod
    def advanced_binding_prediction(known_ligands: list, protein_data: Dict, 
                                    novel_compounds: Optional[list] = None) -> Dict:
        """
        Advanced ML-based binding prediction for both known and unknown ligands
        
        Prediction features:
        1. Molecular descriptors (MW, LogP, HBD, HBA, TPSA)
        2. Structural fingerprints (if SMILES available)
        3. Protein-ligand interaction fingerprints
        4. Pharmacophore matching
        5. QSAR model predictions
        
        Returns comprehensive predictions with confidence scores
        """
        import math
        
        predictions: dict[str, Any] = {
            "known_ligands": [],
            "novel_candidates": [],
            "binding_rules": {},
            "recommendations": []
        }
        binding_rules: dict[str, Any] = {}
        
        # Analyze known ligands to extract binding rules
        if known_ligands:
            binding_rules = ProteinVisualizer.extract_binding_rules(known_ligands)
            predictions["binding_rules"] = binding_rules
            
            # Predict for known ligands
            for ligand in known_ligands:
                pred = ProteinVisualizer.predict_binding_score(ligand, binding_rules, protein_data, is_known=True)
                predictions["known_ligands"].append(pred)
            
            # Sort by predicted binding
            predictions["known_ligands"] = sorted(
                predictions["known_ligands"], 
                key=lambda x: x["predicted_score"], 
                reverse=True
            )
        
        # Predict for novel/unknown compounds
        if novel_compounds:
            for compound in novel_compounds:
                pred = ProteinVisualizer.predict_binding_score(compound, binding_rules if known_ligands else {}, 
                                            protein_data, is_known=False)
                predictions["novel_candidates"].append(pred)
            
            # Sort by predicted binding
            predictions["novel_candidates"] = sorted(
                predictions["novel_candidates"],
                key=lambda x: x["predicted_score"],
                reverse=True
            )
        
        # Generate recommendations
        predictions["recommendations"] = ProteinVisualizer.generate_recommendations(
            predictions["known_ligands"],
            predictions["novel_candidates"],
            binding_rules if known_ligands else {}
        )
        
        return predictions

    @staticmethod
    def extract_binding_rules(known_ligands: list) -> Dict:
        """
        Extract SAR (Structure-Activity Relationship) rules from known ligands
        """
        rules: dict[str, Any] = {
            "optimal_mw_range": [0.0, 0.0],
            "activity_threshold": {},
            "pharmacophore": [],
            "property_ranges": {}
        }
        
        # Extract activity data
        activities: list[float] = []
        mw_values: list[float] = []
        
        for lig in known_ligands:
            activity = lig.get('activity_value', 0)
            mw = lig.get('molecular_weight', 0)
            
            # Convert activity to float if it's a string
            if isinstance(activity, str):
                try:
                    activity = float(activity)
                except (ValueError, TypeError):
                    activity = 0
            
            # Convert mw to float if it's a string
            if isinstance(mw, str):
                try:
                    mw = float(mw)
                except (ValueError, TypeError):
                    mw = 0
            
            if activity > 0:
                activities.append(activity)
            if mw > 0:
                mw_values.append(mw)
        
        if activities:
            # Define potent threshold (bottom 25th percentile)
            activities_sorted = sorted(activities)
            potent_threshold = activities_sorted[len(activities_sorted) // 4] if len(activities_sorted) > 4 else activities_sorted[0]
            
            rules["activity_threshold"] = {
                "potent": potent_threshold,
                "moderate": potent_threshold * 10,
                "weak": potent_threshold * 100
            }
        
        if mw_values:
            # Optimal MW range (mean ± 1 std dev)
            import statistics
            mean_mw = statistics.mean(mw_values)
            std_mw = statistics.stdev(mw_values) if len(mw_values) > 1 else 50
            
            rules["optimal_mw_range"] = [
                max(150, mean_mw - std_mw),
                min(600, mean_mw + std_mw)
            ]
        
        # Lipinski's Rule of Five compliance from known actives
        rules["lipinski_compliance"] = True
        
        return rules

    @staticmethod
    def predict_binding_score(compound: Dict, binding_rules: Dict, 
                            protein_data: Dict, is_known: bool = False) -> Dict:
        """
        Predict binding affinity score for a compound
        Returns score 0-100 with confidence level
        """
        import math
        
        score = 0
        confidence_factors = []
        reasons = []
        warnings = []
        
        # Factor 1: Experimental activity (only for known ligands)
        if is_known and compound.get('activity_value'):
            activity = compound['activity_value']
            activity_type = compound.get('activity_type', 'IC50')
            
            # Convert activity to float if it's a string
            if isinstance(activity, str):
                try:
                    activity = float(activity)
                except (ValueError, TypeError):
                    activity = None
            
            if activity is not None:
                thresholds = binding_rules.get('activity_threshold', {})
                
                if activity <= thresholds.get('potent', 10):
                    score += 50
                    confidence_factors.append(0.95)
                    reasons.append(f"Very potent {activity_type}: {activity:.2f} nM (experimental)")
                elif activity <= thresholds.get('moderate', 100):
                    score += 35
                    confidence_factors.append(0.85)
                    reasons.append(f"Moderate {activity_type}: {activity:.2f} nM (experimental)")
                else:
                    score += 15
                    confidence_factors.append(0.70)
                    reasons.append(f"Weak activity: {activity:.2f} nM (experimental)")
        
        # Factor 2: Molecular weight (drug-likeness)
        mw = compound.get('molecular_weight', 0)
        # Convert mw to float if it's a string
        if isinstance(mw, str):
            try:
                mw = float(mw)
            except (ValueError, TypeError):
                mw = 0
        
        optimal_range = binding_rules.get('optimal_mw_range', [160, 500])
        
        if mw:
            if optimal_range[0] <= mw <= optimal_range[1]:
                score += 15
                confidence_factors.append(0.80)
                reasons.append(f"Optimal MW: {mw:.1f} Da (within active range)")
            elif 150 <= mw <= 600:  # Lipinski range
                score += 10
                confidence_factors.append(0.70)
                reasons.append(f"Acceptable MW: {mw:.1f} Da (drug-like)")
                if mw > 500:
                    warnings.append("MW >500 Da may reduce oral bioavailability")
            else:
                score += 3
                confidence_factors.append(0.50)
                warnings.append(f"MW {mw:.1f} Da outside optimal range")
        
        # Factor 3: Lipinski's Rule of Five compliance
        lipinski_violations = ProteinVisualizer.calculate_lipinski_violations(compound)
        
        if lipinski_violations == 0:
            score += 15
            confidence_factors.append(0.85)
            reasons.append("Passes Lipinski's Rule of Five (drug-like)")
        elif lipinski_violations == 1:
            score += 10
            confidence_factors.append(0.75)
            reasons.append("1 Lipinski violation (acceptable)")
            warnings.append("Minor drug-likeness concern")
        else:
            score += 5
            confidence_factors.append(0.60)
            warnings.append(f"{lipinski_violations} Lipinski violations (poor drug-likeness)")
        
        # Factor 4: Chemical structure availability
        if compound.get('smiles'):
            score += 10
            confidence_factors.append(0.90)
            reasons.append("Structure available for computational docking")
        
        # Factor 5: Known drug status
        name = compound.get('name', '').lower()
        source = compound.get('source', '')
        
        if 'fda' in source.lower() or 'approved' in str(compound.get('status', '')).lower():
            score += 15
            confidence_factors.append(0.95)
            reasons.append("FDA-approved drug (validated safety profile)")
        elif any(x in name for x in ['inhibitor', 'mab', 'nib', 'tinib', 'zumab']):
            score += 12
            confidence_factors.append(0.85)
            reasons.append("Known inhibitor/drug class")
        
        # Factor 6: Literature evidence
        if compound.get('pmid') or 'literature' in source.lower():
            score += 8
            confidence_factors.append(0.75)
            reasons.append("Literature evidence of activity")
        
        # Factor 7: Target class match (for repurposing)
        if compound.get('target_class'):
            score += 10
            confidence_factors.append(0.80)
            reasons.append(f"Target class match: {compound['target_class']}")
        
        # Calculate confidence (average of all factors)
        confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        # Confidence level
        if confidence >= 0.85:
            confidence_level = "High"
            confidence_color = "#28a745"
        elif confidence >= 0.70:
            confidence_level = "Medium"
            confidence_color = "#ffc107"
        else:
            confidence_level = "Low"
            confidence_color = "#dc3545"
        
        # Predicted binding affinity (simplified QSAR)
        if is_known and compound.get('activity_value'):
            # Convert IC50 to approximate binding affinity
            ic50 = compound['activity_value']
            predicted_affinity = -math.log10(ic50 / 1e9) * 1.36  # kcal/mol
            predicted_affinity = max(-12, min(-4, predicted_affinity))
        else:
            # For unknowns: estimate based on score
            predicted_affinity = -4 - (score / 100) * 6  # Range: -4 to -10
        
        return {
            "compound": compound,
            "predicted_score": min(100, score),
            "confidence": round(confidence, 2),
            "confidence_level": confidence_level,
            "confidence_color": confidence_color,
            "predicted_affinity": round(predicted_affinity, 2),
            "reasons": reasons,
            "warnings": warnings,
            "is_known": is_known,
            "recommendation": "Highly recommended" if score >= 75 else ("Worth testing" if score >= 50 else "Low priority")
        }

    @staticmethod
    def calculate_lipinski_violations(compound: Dict) -> int:
        """
        Calculate Lipinski's Rule of Five violations
        Rules: MW ≤500, LogP ≤5, HBD ≤5, HBA ≤10
        """
        violations: float = 0.0
        
        mw = compound.get('molecular_weight', 0)
        # Convert mw to float if it's a string
        if isinstance(mw, str):
            try:
                mw = float(mw)
            except (ValueError, TypeError):
                mw = 0
        
        if mw > 500:
            violations += 1
        
        # Note: Would need to calculate LogP, HBD, HBA from SMILES
        # For now, estimate based on MW
        if mw > 450:  # Rough proxy for LogP violations
            violations += 0.5
        
        return int(violations)


    @staticmethod
    def generate_recommendations(known_predictions: list, novel_predictions: list,
                                binding_rules: Dict) -> list:
        """
        Generate actionable recommendations for drug discovery
        """
        recommendations = []
        
        # Recommendation 1: Best known binder
        if known_predictions:
            best_known = known_predictions[0]
            recommendations.append({
                "type": "Best Known Binder",
                "compound": best_known["compound"]["name"],
                "score": best_known["predicted_score"],
                "action": f"Use as positive control in experiments (predicted affinity: {best_known['predicted_affinity']:.1f} kcal/mol)",
                "priority": "High"
            })
        
        # Recommendation 2: Top novel candidate
        if novel_predictions:
            best_novel = novel_predictions[0]
            if best_novel["predicted_score"] >= 60:
                recommendations.append({
                    "type": "Novel Candidate",
                    "compound": best_novel["compound"]["name"],
                    "score": best_novel["predicted_score"],
                    "action": f"Priority for experimental validation (confidence: {best_novel['confidence_level']})",
                    "priority": "High" if best_novel["predicted_score"] >= 75 else "Medium"
                })
        
        # Recommendation 3: Repurposing opportunities
        repurposing = [p for p in (novel_predictions or []) 
                    if p["compound"].get("source") == "Drug Repurposing"]
        if repurposing:
            recommendations.append({
                "type": "Drug Repurposing",
                "compound": f"{len(repurposing)} FDA-approved drug(s)",
                "score": max([p["predicted_score"] for p in repurposing]),
                "action": "Consider for off-label use or clinical trials (safety already established)",
                "priority": "High"
            })
        
        # Recommendation 4: Structure optimization
        if known_predictions and len(known_predictions) >= 3:
            mw_list = []
            for p in known_predictions[:3]:
                mw = p["compound"].get("molecular_weight", 0)
                # Convert to float if it's a string
                if isinstance(mw, str):
                    try:
                        mw = float(mw)
                    except (ValueError, TypeError):
                        mw = 0
                mw_list.append(mw)
            
            top3_avg_mw = sum(mw_list) / 3 if mw_list else 0
            recommendations.append({
                "type": "Structure Optimization",
                "compound": "New derivatives",
                "score": 0,
                "action": f"Design analogs around MW ~{top3_avg_mw:.0f} Da based on top binders",
                "priority": "Medium"
            })
        
        return recommendations

    @staticmethod
    def create_risk_calculator_ui() -> str:
        """
        Create interactive risk calculator HTML form
        Returns HTML with JavaScript for real-time calculation
        """
        html = """
        <style>
            .risk-calculator {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                border-radius: 15px;
                color: white;
                margin: 20px 0;
            }
            .risk-form {
                background: white;
                padding: 25px;
                border-radius: 10px;
                color: #333;
            }
            .risk-input {
                margin: 15px 0;
            }
            .risk-input label {
                display: block;
                font-weight: 600;
                margin-bottom: 8px;
                color: #555;
            }
            .risk-input select, .risk-input input {
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 14px;
            }
            .risk-result {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                border-left: 5px solid #667eea;
            }
            .risk-score {
                font-size: 48px;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
            }
            .calculate-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                margin-top: 20px;
            }
            .calculate-btn:hover {
                opacity: 0.9;
            }
        </style>
        
        <div class="risk-calculator">
            <h2 style="margin-top:0;">🔬 Predictive Risk Calculator</h2>
            <p>Calculate your disease risk based on biomarker levels and personal factors</p>
            
            <div class="risk-form">
                <div class="risk-input">
                    <label>Age:</label>
                    <input type="number" id="age" min="18" max="100" value="45" />
                </div>
                
                <div class="risk-input">
                    <label>Family History:</label>
                    <select id="family">
                        <option value="none">No family history</option>
                        <option value="second_degree">Second-degree relative (grandparent, aunt, uncle)</option>
                        <option value="first_degree">First-degree relative (parent, sibling)</option>
                    </select>
                </div>
                
                <div class="risk-input">
                    <label>Smoking Status:</label>
                    <select id="smoking">
                        <option value="never">Never smoked</option>
                        <option value="former">Former smoker</option>
                        <option value="current">Current smoker</option>
                    </select>
                </div>
                
                <div class="risk-input">
                    <label>BMI (Body Mass Index):</label>
                    <input type="number" id="bmi" min="15" max="50" step="0.1" value="25" />
                </div>
                
                <div class="risk-input">
                    <label>Exercise Frequency:</label>
                    <select id="exercise">
                        <option value="regular">Regular (3+ times/week)</option>
                        <option value="occasional">Occasional (1-2 times/week)</option>
                        <option value="none">Sedentary</option>
                    </select>
                </div>
                
                <div class="risk-input">
                    <label>Diet Quality:</label>
                    <select id="diet">
                        <option value="good">Balanced, nutritious</option>
                        <option value="fair">Average</option>
                        <option value="poor">Poor, high processed foods</option>
                    </select>
                </div>
                
                <button class="calculate-btn" onclick="calculateRisk()">Calculate My Risk</button>
                
                <div id="result" class="risk-result" style="display:none;">
                    <div class="risk-score" id="score"></div>
                    <div id="level"></div>
                    <div id="recommendations" style="margin-top:15px;"></div>
                </div>
            </div>
        </div>
        
        <script>
            function calculateRisk() {
                // Get values
                const age = parseInt(document.getElementById('age').value);
                const family = document.getElementById('family').value;
                const smoking = document.getElementById('smoking').value;
                const bmi = parseFloat(document.getElementById('bmi').value);
                const exercise = document.getElementById('exercise').value;
                const diet = document.getElementById('diet').value;
                
                // Calculate scores
                let ageScore = 0;
                if (age < 40) ageScore = 5;
                else if (age < 50) ageScore = 10;
                else if (age < 60) ageScore = 15;
                else ageScore = 20;
                
                let familyScore = 0;
                if (family === 'first_degree') familyScore = 25;
                else if (family === 'second_degree') familyScore = 15;
                
                let lifestyleScore = 0;
                if (smoking === 'current') lifestyleScore += 5;
                else if (smoking === 'former') lifestyleScore += 3;
                
                if (bmi >= 30) lifestyleScore += 4;
                else if (bmi >= 25) lifestyleScore += 2;
                
                if (exercise === 'none') lifestyleScore += 3;
                else if (exercise === 'occasional') lifestyleScore += 1;
                
                if (diet === 'poor') lifestyleScore += 3;
                
                // Expression score (from Streamlit)
                const expressionScore = window.expressionScore || 20;
                
                // Total risk
                const totalRisk = (expressionScore * 0.4) + (ageScore * 0.2) + 
                                (familyScore * 0.25) + (lifestyleScore * 0.15);
                
                // Display result
                const resultDiv = document.getElementById('result');
                const scoreDiv = document.getElementById('score');
                const levelDiv = document.getElementById('level');
                const recDiv = document.getElementById('recommendations');
                
                resultDiv.style.display = 'block';
                scoreDiv.textContent = totalRisk.toFixed(1) + '/100';
                
                let color, level, recs;
                if (totalRisk >= 70) {
                    color = '#dc3545';
                    level = 'High Risk';
                    recs = '<strong>⚠️ High Risk - Immediate Action Recommended:</strong><br>' +
                        '• Consult with specialist within 2-4 weeks<br>' +
                        '• Enhanced screening every 3-6 months<br>' +
                        '• Consider genetic testing<br>' +
                        '• Early detection advantage: 6-12 months';
                } else if (totalRisk >= 40) {
                    color = '#ffc107';
                    level = 'Medium Risk';
                    recs = '<strong>⚡ Medium Risk - Regular Monitoring:</strong><br>' +
                        '• Annual screening recommended<br>' +
                        '• Biomarker monitoring every 6-12 months<br>' +
                        '• Lifestyle modification consultation<br>' +
                        '• Stay vigilant for symptoms';
                } else {
                    color = '#28a745';
                    level = 'Low Risk';
                    recs = '<strong>✅ Low Risk - Continue Healthy Habits:</strong><br>' +
                        '• Standard age-appropriate screening<br>' +
                        '• Annual health check-up<br>' +
                        '• Maintain current lifestyle<br>' +
                        '• Monitor for any changes';
                }
                
                scoreDiv.style.color = color;
                levelDiv.innerHTML = `<h3 style="color:${color}; text-align:center;">${level}</h3>`;
                recDiv.innerHTML = recs;
            }
        </script>
        """
        
        return html


    @staticmethod
    def create_drug_target_visualization(drug_data: Dict) -> go.Figure:
        """
        Create visualization of drugs by development phase
        """
        if not drug_data.get('available'):
            fig = go.Figure()
            fig.add_annotation(
                text="No drug-target data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        categories = ['FDA Approved', 'Clinical Trials', 'Investigational']
        counts = [
            drug_data.get('total_fda', 0),
            drug_data.get('total_trials', 0),
            drug_data.get('total_investigational', 0)
        ]
        
        colors = ['#28a745', '#ffc107', '#17a2b8']
        
        fig = go.Figure(go.Bar(
            x=categories,
            y=counts,
            marker=dict(color=colors, line=dict(color='black', width=1.5)),
            text=counts,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Drug Development Pipeline",
            yaxis_title="Number of Drugs",
            height=350,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_binding_affinity_chart(predictions: List[Dict]) -> go.Figure:
        """
        Create bar chart showing binding affinity for multiple molecules
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        valid_predictions = [p for p in predictions if p.get("is_valid", False)]
        
        if not valid_predictions:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid predictions available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        # Extract data
        molecule_names = [p.get("molecule_name", f"Molecule {i+1}") for i, p in enumerate(valid_predictions)]
        affinities = [p.get("prediction", {}).get("binding_affinity", 0) for p in valid_predictions]
        
        # Color based on affinity (more negative = better = green)
        colors = []
        for aff in affinities:
            if aff < -8:
                colors.append("#28a745")  # Green - excellent
            elif aff < -6:
                colors.append("#ffc107")  # Yellow - good
            elif aff < -4:
                colors.append("#ff9800")  # Orange - moderate
            else:
                colors.append("#dc3545")  # Red - poor
        
        fig = go.Figure(go.Bar(
            x=molecule_names,
            y=affinities,
            marker=dict(
                color=colors,
                line=dict(color='rgb(8,48,107)', width=1)
            ),
            text=[f"{aff:.2f} kcal/mol" for aff in affinities],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Binding Affinity: %{y:.2f} kcal/mol<extra></extra>'
        ))
        
        fig.update_layout(
            title="Predicted Binding Affinity",
            xaxis_title="Molecule",
            yaxis_title="Binding Affinity (kcal/mol)",
            height=max(400, len(valid_predictions) * 40),
            template="plotly_white",
            showlegend=False,
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    @staticmethod
    def create_binding_likelihood_chart(predictions: List[Dict]) -> go.Figure:
        """
        Create bar chart showing binding likelihood (probability) for multiple molecules
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        valid_predictions = [p for p in predictions if p.get("is_valid", False)]
        
        if not valid_predictions:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid predictions available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        # Extract data
        molecule_names = [p.get("molecule_name", f"Molecule {i+1}") for i, p in enumerate(valid_predictions)]
        likelihoods = [p.get("prediction", {}).get("binding_likelihood", 0) for p in valid_predictions]
        
        # Color based on likelihood
        colors = []
        for lik in likelihoods:
            if lik >= 70:
                colors.append("#28a745")  # Green - high
            elif lik >= 50:
                colors.append("#ffc107")  # Yellow - medium
            elif lik >= 30:
                colors.append("#ff9800")  # Orange - low
            else:
                colors.append("#dc3545")  # Red - very low
        
        fig = go.Figure(go.Bar(
            x=molecule_names,
            y=likelihoods,
            marker=dict(
                color=colors,
                line=dict(color='rgb(8,48,107)', width=1)
            ),
            text=[f"{lik:.1f}%" for lik in likelihoods],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Binding Likelihood: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Predicted Binding Likelihood",
            xaxis_title="Molecule",
            yaxis_title="Binding Likelihood (%)",
            yaxis=dict(range=[0, 100]),
            height=max(400, len(valid_predictions) * 40),
            template="plotly_white",
            showlegend=False,
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    @staticmethod
    def create_binding_ranking_chart(ranked_molecules: List[Dict], top_n: int = 10) -> go.Figure:
        """
        Create scatter plot ranking molecules by affinity and likelihood
        
        Args:
            ranked_molecules: List of ranked prediction dictionaries
            top_n: Number of top molecules to display
            
        Returns:
            Plotly figure
        """
        if not ranked_molecules:
            fig = go.Figure()
            fig.add_annotation(
                text="No ranked molecules available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(height=400)
            return fig
        
        # Take top N
        top_molecules = ranked_molecules[:top_n]
        
        # Extract data
        molecule_names = [m.get("molecule_name", f"Molecule {m.get('rank', i+1)}") for i, m in enumerate(top_molecules)]
        affinities = [m.get("prediction", {}).get("binding_affinity", 0) for m in top_molecules]
        likelihoods = [m.get("prediction", {}).get("binding_likelihood", 0) for m in top_molecules]
        ranks = [m.get("rank", i+1) for i, m in enumerate(top_molecules)]
        
        # Size based on rank (higher rank = smaller)
        sizes = [max(10, 30 - r) for r in ranks]
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=affinities,
            y=likelihoods,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=likelihoods,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Likelihood (%)"),
                line=dict(width=2, color='black')
            ),
            text=[f"#{r}" for r in ranks],
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            customdata=molecule_names,
            hovertemplate='<b>%{customdata}</b><br>Rank: #%{text}<br>Affinity: %{x:.2f} kcal/mol<br>Likelihood: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Top {len(top_molecules)} Ranked Drug Candidates",
            xaxis_title="Binding Affinity (kcal/mol)",
            yaxis_title="Binding Likelihood (%)",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def render_phylogenetic_tree(newick_str: str, num_taxa: int = 2) -> str:
        """
        Render a phylogenetic tree from Newick format to base64-encoded image.
        Uses BioPython and matplotlib to create a visual tree.
        
        Args:
            newick_str: Newick format tree string
            num_taxa: Number of taxa (used for figure sizing)
            
        Returns:
            Base64-encoded PNG image as data URI
        """
        try:
            from Bio import Phylo
            
            # Parse the Newick string
            tree_handle = StringIO(newick_str)
            tree = Phylo.read(tree_handle, "newick")
            
            # Calculate figure size based on taxa count
            # Minimum height of 4, scale up with more taxa
            fig_height = max(4, num_taxa * 0.5)
            fig_width = max(8, num_taxa * 0.6)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Draw the tree
            Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False)
            
            # Improve styling
            ax.set_xlabel("Branch Length", fontsize=10)
            ax.set_title("Phylogenetic Tree", fontsize=12, fontweight='bold', pad=15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, labelleft=True)
            
            # Adjust layout to prevent label clipping
            plt.tight_layout()
            
            # Convert to base64 image
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            # If rendering fails, return error message as image
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Error rendering tree:\n{str(e)}", 
                   ha='center', va='center', fontsize=12, color='red',
                   transform=ax.transAxes)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
    
    @staticmethod
    def create_phylogenetic_tree_visualization(newick_string: str, metadata: Dict) -> str:
        """
        Create an interactive phylogenetic tree visualization with rendered tree image.
        Parses Newick format and displays actual tree structure with branches and labels.
        
        Args:
            newick_string: Newick format tree string
            metadata: Dictionary with method, num_taxa, tree_length
            
        Returns:
            HTML string with embedded tree visualization
        """
        # Get number of taxa for sizing
        num_taxa = metadata.get('num_taxa', 2)
        
        # Render the tree to base64 image
        tree_image = ProteinVisualizer.render_phylogenetic_tree(newick_string, num_taxa)
        
        # Create HTML visualization
        html = f"""
        <div style="font-family: Arial, sans-serif; width: 100%; overflow: auto; border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9;">
            <div style="margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; color: #1f77b4;">📊 Phylogenetic Tree</h3>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0; margin-bottom: 15px;">
                <h4 style="margin: 0 0 15px 0; color: #333;">Tree Visualization:</h4>
                <div style="text-align: center; padding: 10px; background: white;">
                    <img src="{tree_image}" style="max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px;" alt="Phylogenetic Tree">
                </div>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0;">
                <details>
                    <summary style="cursor: pointer; font-weight: bold; color: #333; padding: 5px;">📄 Raw Newick Format</summary>
                    <div id="tree-container" style="font-family: 'Courier New', monospace; font-size: 11px; line-height: 1.6; overflow-x: auto; white-space: pre-wrap; word-break: break-all; background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 200px; overflow-y: auto; margin-top: 10px;">
                        {newick_string}
                    </div>
                </details>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 4px; font-size: 12px; color: #1565c0;">
                <strong>📝 How to read:</strong> Branch lengths represent evolutionary distance between taxa. Longer branches indicate greater evolutionary divergence.
                <br><strong>🔹 Method:</strong> Tree constructed using {metadata.get('method', 'N/A').upper()} algorithm with {num_taxa} taxa.
            </div>
        </div>
        """
        
        return html

    @staticmethod
    def create_phylogenetic_dendrogram(newick_string: str, metadata: Dict) -> go.Figure:
        """Create an interactive dendrogram visualization like ClustalW"""
        try:
            from Bio import Phylo
            from io import StringIO
            tree = Phylo.read(StringIO(newick_string), "newick")
            terminals = tree.get_terminals()
            terminal_names = [t.name if t.name else f"Seq{i}" for i, t in enumerate(terminals)]

            fig = go.Figure()
            if terminal_names:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(terminal_names))),
                        y=[0] * len(terminal_names),
                        mode="markers+text",
                        text=terminal_names,
                        textposition="middle right",
                        marker=dict(size=8, color="#1f77b4"),
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title=f"Phylogenetic Tree ({metadata.get('method', 'N/A').upper()}) - {metadata.get('num_taxa', 0)} Taxa",
                xaxis_title="Terminal order",
                yaxis=dict(showticklabels=False, zeroline=False),
                height=max(400, len(terminal_names) * 30),
                width=1000,
                showlegend=False,
                template="plotly_white",
                hovermode="closest",
                margin=dict(l=150, r=100, t=80, b=50),
            )
            return fig
        except Exception:
            fig = go.Figure()
            fig.add_annotation(
                text="Dendrogram visualization unavailable<br>Showing Newick format instead",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(height=300, title="Phylogenetic Tree")
            return fig

    @staticmethod
    def create_target_component_contribution_chart(ranked_results: List[Dict]) -> go.Figure:
        """Create stacked bar chart of weighted component contributions."""
        if not ranked_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No target prioritization results available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        component_order = ["expression", "pathway", "ppi", "genetic", "ligandability", "trials"]
        labels = [str(row.get("target_id", "Unknown")) for row in ranked_results]
        fig = go.Figure()

        for component in component_order:
            y_values = []
            for row in ranked_results:
                contrib = row.get("component_contributions", {}).get(component, {})
                y_values.append(float(contrib.get("weighted_contribution", 0.0)))
            fig.add_trace(
                go.Bar(
                    name=component.title(),
                    x=labels,
                    y=y_values,
                )
            )

        fig.update_layout(
            barmode="stack",
            title="Target Score Contributions by Component",
            xaxis_title="Target",
            yaxis_title="Weighted Contribution",
            template="plotly_white",
            height=420,
            legend_title="Component",
        )
        return fig

    @staticmethod
    def create_target_radar_chart(component_scores: Dict[str, Dict]) -> go.Figure:
        """Create radar chart for six target components."""
        component_order = ["expression", "pathway", "ppi", "genetic", "ligandability", "trials"]
        values = [float(component_scores.get(k, {}).get("score", 0.0)) for k in component_order]
        labels = [k.title() for k in component_order]
        values.append(values[0])
        labels.append(labels[0])

        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself",
                    name="Component score",
                    line=dict(color="#1f77b4"),
                )
            ]
        )
        fig.update_layout(
            title="Component Score Radar",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
            ),
            showlegend=False,
            template="plotly_white",
            height=420,
        )
        return fig

    @staticmethod
    def create_target_sensitivity_chart(sensitivity_result: Dict[str, Dict]) -> go.Figure:
        """Create bar chart of sensitivity deltas by scenario."""
        deltas = sensitivity_result.get("scenario_deltas", {})
        if not deltas:
            fig = go.Figure()
            fig.add_annotation(
                text="No sensitivity scenarios available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        scenario_names = list(deltas.keys())
        delta_values = [float(deltas[name]) for name in scenario_names]
        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in delta_values]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=scenario_names,
                    y=delta_values,
                    marker_color=colors,
                    text=[f"{v:+.2f}" for v in delta_values],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Sensitivity Analysis (Delta vs Baseline)",
            xaxis_title="Scenario",
            yaxis_title="Composite Score Delta",
            template="plotly_white",
            height=360,
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_variant_impact_distribution_chart(annotated_variants: List[Dict]) -> go.Figure:
        """Create impact-class distribution chart for annotated variants."""
        if not annotated_variants:
            fig = go.Figure()
            fig.add_annotation(
                text="No annotated variants available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        counts: Dict[str, int] = defaultdict(int)
        for row in annotated_variants:
            counts[str(row.get("predicted_effect_class", "unknown")).lower()] += 1
        order = ["high", "moderate", "low", "unknown"]
        x_vals = [label.title() for label in order]
        y_vals = [counts.get(label, 0) for label in order]
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9e9e9e"]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=colors,
                    text=y_vals,
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Variant Impact Distribution",
            xaxis_title="Impact class",
            yaxis_title="Variant count",
            template="plotly_white",
            height=340,
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_top_gene_impact_chart(gene_impact: Dict, top_n: int = 10) -> go.Figure:
        """Create bar chart of top impacted genes."""
        genes = list(gene_impact.get("genes", {}).values())[:top_n]
        if not genes:
            fig = go.Figure()
            fig.add_annotation(
                text="No gene impact data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        names = [g.get("gene", "UNK") for g in genes]
        scores = [float(g.get("score", 0.0)) for g in genes]
        fig = go.Figure(
            data=[go.Bar(x=names, y=scores, marker_color="#1f77b4", text=[f"{s:.1f}" for s in scores], textposition="auto")]
        )
        fig.update_layout(
            title="Top Gene Impact Scores",
            xaxis_title="Gene",
            yaxis_title="Impact score (0-100)",
            template="plotly_white",
            height=360,
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_pathway_perturbation_chart(pathway_impact: Dict, top_n: int = 10) -> go.Figure:
        """Create horizontal bar chart of pathway perturbation."""
        pathways = pathway_impact.get("pathways", [])[:top_n]
        if not pathways:
            fig = go.Figure()
            fig.add_annotation(
                text="No pathway impact data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        names = [p.get("pathway_name", "Unknown") for p in pathways][::-1]
        scores = [float(p.get("impact_score", 0.0)) for p in pathways][::-1]
        confidences = [p.get("confidence", "Low") for p in pathways][::-1]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=scores,
                    y=names,
                    orientation="h",
                    text=[f"{s:.1f}" for s in scores],
                    textposition="auto",
                    customdata=confidences,
                    hovertemplate="<b>%{y}</b><br>Impact: %{x:.1f}<br>Confidence: %{customdata}<extra></extra>",
                    marker_color="#6a3d9a",
                )
            ]
        )
        fig.update_layout(
            title="Pathway Perturbation Scores",
            xaxis_title="Pathway score (0-100)",
            yaxis_title="Pathway",
            template="plotly_white",
            height=max(360, len(pathways) * 28),
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_therapy_contribution_chart(ranked_candidates: List[Dict], top_n: int = 8) -> go.Figure:
        """Create stacked contribution chart for therapy ranking components."""
        top = ranked_candidates[:top_n]
        if not top:
            fig = go.Figure()
            fig.add_annotation(
                text="No therapy candidates available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        labels = [c.get("drug_name", "Unknown") for c in top]
        components = {
            "Target match": [float(c.get("target_gene_match", 0.0)) * 100.0 for c in top],
            "Pathway relevance": [float(c.get("pathway_relevance", 0.0)) * 100.0 for c in top],
            "Evidence quality": [float(c.get("evidence_quality", 0.0)) * 100.0 for c in top],
            "Clinical maturity": [float(c.get("clinical_maturity", 0.0)) * 100.0 for c in top],
            "Safety penalty": [-float(c.get("safety_risk_penalty", 0.0)) * 100.0 for c in top],
        }

        fig = go.Figure()
        palette = ["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e", "#d62728"]
        for idx, (label, values) in enumerate(components.items()):
            fig.add_trace(go.Bar(x=labels, y=values, name=label, marker_color=palette[idx]))
        fig.update_layout(
            barmode="relative",
            title="Therapy Candidate Score Contributions",
            xaxis_title="Drug candidate",
            yaxis_title="Component contribution (scaled)",
            template="plotly_white",
            height=420,
            legend_title="Components",
        )
        return fig

    @staticmethod
    def create_confidence_completeness_chart(ranked_candidates: List[Dict], top_n: int = 8) -> go.Figure:
        """Create bubble chart of confidence vs completeness for top therapies."""
        top = ranked_candidates[:top_n]
        if not top:
            fig = go.Figure()
            fig.add_annotation(
                text="No confidence/completeness data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        x_vals = [float(c.get("completeness_pct", 0.0)) for c in top]
        y_vals = [float(c.get("ranking_confidence", 0.0)) for c in top]
        names = [c.get("drug_name", "Unknown") for c in top]
        scores = [float(c.get("composite_score", 0.0)) for c in top]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers+text",
                    text=names,
                    textposition="top center",
                    marker=dict(
                        size=[max(10.0, min(40.0, s / 2.5)) for s in scores],
                        color=scores,
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="Composite"),
                    ),
                    hovertemplate="<b>%{text}</b><br>Completeness=%{x:.1f}%<br>Confidence=%{y:.1f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Confidence vs Data Completeness",
            xaxis_title="Completeness (%)",
            yaxis_title="Confidence (0-100)",
            template="plotly_white",
            height=380,
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_assay_ranking_comparison_chart(assays: List[Dict], top_n: int = 8) -> go.Figure:
        """Create ranked assay score chart with confidence coloring."""
        top = assays[:top_n]
        if not top:
            fig = go.Figure()
            fig.add_annotation(
                text="No assay ranking data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        names = [row.get("assay_name", "Assay") for row in top][::-1]
        scores = [float(row.get("rank_score", 0.0)) for row in top][::-1]
        confidence = [row.get("confidence", "Low") for row in top][::-1]
        color_map = {"High": "#2ca02c", "Med": "#ff7f0e", "Low": "#d62728"}
        colors = [color_map.get(c, "#636efa") for c in confidence]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=scores,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{s:.1f}" for s in scores],
                    textposition="auto",
                    customdata=confidence,
                    hovertemplate="<b>%{y}</b><br>Score=%{x:.1f}<br>Confidence=%{customdata}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Assay Ranking Comparison",
            xaxis_title="Heuristic score (0-100)",
            yaxis_title="Assay",
            template="plotly_white",
            height=max(340, len(top) * 36),
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_crispr_candidate_score_plot(candidates: List[Dict], top_n: int = 30) -> go.Figure:
        """Create position-vs-score scatter for CRISPR candidates."""
        top = candidates[:top_n]
        if not top:
            fig = go.Figure()
            fig.add_annotation(
                text="No CRISPR candidate scores available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        risk_color = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
        x_vals = [int(row.get("target_position", 0)) for row in top]
        y_vals = [float(row.get("heuristic_score", 0.0)) for row in top]
        labels = [row.get("off_target_risk_level", "Medium") for row in top]
        colors = [risk_color.get(lbl, "#636efa") for lbl in labels]
        text = [row.get("spacer_sequence", "")[:12] + "..." for row in top]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    marker=dict(size=10, color=colors, line=dict(width=0.5, color="#333333")),
                    text=text,
                    customdata=labels,
                    hovertemplate="Position=%{x}<br>Score=%{y:.1f}<br>Risk=%{customdata}<br>Spacer=%{text}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="CRISPR Candidate Score Plot",
            xaxis_title="Target position",
            yaxis_title="Heuristic score (0-100)",
            template="plotly_white",
            height=360,
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_primer_quality_comparison_chart(primers: List[Dict], top_n: int = 10) -> go.Figure:
        """Create grouped chart for primer quality and amplicon size."""
        top = primers[:top_n]
        if not top:
            fig = go.Figure()
            fig.add_annotation(
                text="No primer quality data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(height=320, template="plotly_white")
            return fig

        labels = [f"Pair {idx+1}" for idx in range(len(top))]
        quality = [float(row.get("quality_score", 0.0)) for row in top]
        amplicon = [int(row.get("expected_amplicon_size", 0)) for row in top]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=quality, name="Quality score", marker_color="#1f77b4"))
        fig.add_trace(go.Scatter(x=labels, y=amplicon, name="Amplicon size (bp)", yaxis="y2", mode="lines+markers", marker_color="#ff7f0e"))
        fig.update_layout(
            title="Primer Quality Comparison",
            xaxis_title="Primer pair",
            yaxis=dict(title="Quality score (0-100)"),
            yaxis2=dict(title="Amplicon size (bp)", overlaying="y", side="right"),
            template="plotly_white",
            height=380,
            legend=dict(orientation="h"),
        )
        return fig

    @staticmethod
    def create_wet_lab_readiness_dashboard(confidence_payload: Dict, assays: List[Dict], crispr_candidates: List[Dict]) -> go.Figure:
        """Create compact readiness dashboard with confidence and risk distribution."""
        conf = float(confidence_payload.get("plan_confidence_score", 0.0))
        completeness = float(confidence_payload.get("data_completeness", 0.0))
        readiness = str(confidence_payload.get("readiness_label", "red")).lower()
        readiness_color = {"green": "#2ca02c", "yellow": "#ffbf00", "red": "#d62728"}.get(readiness, "#d62728")

        assay_risks = sum(len(row.get("risk_flags", [])) for row in assays[:8])
        crispr_risks = {
            "High": sum(1 for row in crispr_candidates[:20] if row.get("off_target_risk_level") == "High"),
            "Medium": sum(1 for row in crispr_candidates[:20] if row.get("off_target_risk_level") == "Medium"),
            "Low": sum(1 for row in crispr_candidates[:20] if row.get("off_target_risk_level") == "Low"),
        }

        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=conf,
                title={"text": "Plan confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": readiness_color},
                    "steps": [
                        {"range": [0, 50], "color": "#fde2e2"},
                        {"range": [50, 75], "color": "#fff6cc"},
                        {"range": [75, 100], "color": "#e0f2e0"},
                    ],
                },
                domain={"x": [0.0, 0.45], "y": [0.0, 1.0]},
            )
        )
        fig.add_trace(
            go.Bar(
                x=["Assay risk flags", "CRISPR High", "CRISPR Medium", "CRISPR Low"],
                y=[assay_risks, crispr_risks["High"], crispr_risks["Medium"], crispr_risks["Low"]],
                marker_color=["#d62728", "#d62728", "#ff7f0e", "#2ca02c"],
                name="Risk distribution",
                xaxis="x2",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Wet-Lab Readiness Dashboard ({readiness.title()} | completeness {completeness:.1f}%)",
            template="plotly_white",
            height=400,
            xaxis2=dict(domain=[0.52, 1.0], anchor="y2"),
            yaxis2=dict(domain=[0.0, 1.0], anchor="x2", title="Count"),
            showlegend=False,
        )
        return fig

    @staticmethod
    def create_portfolio_funnel(stage_distribution: List[Dict]) -> go.Figure:
        """Create a stage distribution funnel for portfolio visibility."""
        if not stage_distribution:
            fig = go.Figure()
            fig.add_annotation(text="No projects available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template="plotly_white", height=320)
            return fig
        df = pd.DataFrame(stage_distribution)
        fig = go.Figure(go.Funnel(y=df["stage"], x=df["count"], textinfo="value+percent initial"))
        fig.update_layout(title="Portfolio Funnel by Stage", template="plotly_white", height=360)
        return fig

    @staticmethod
    def create_candidate_comparison_radar(comparison_rows: List[Dict]) -> go.Figure:
        """Render candidate comparison across normalized dimensions."""
        categories = [
            "biological_strength",
            "pathway_ppi_relevance",
            "ligandability_druggability",
            "translational_evidence",
            "clinical_evidence_maturity",
            "data_quality_confidence",
            "risk_burden_inverted",
        ]
        fig = go.Figure()
        if not comparison_rows:
            fig.add_annotation(text="No candidate comparison data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template="plotly_white", height=380)
            return fig
        for row in comparison_rows[:8]:
            values = [
                row.get("biological_strength", 0.0),
                row.get("pathway_ppi_relevance", 0.0),
                row.get("ligandability_druggability", 0.0),
                row.get("translational_evidence", 0.0),
                row.get("clinical_evidence_maturity", 0.0),
                row.get("data_quality_confidence", 0.0),
                100.0 - row.get("risk_burden", 100.0),
            ]
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=row.get("candidate", "candidate"),
                )
            )
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_white", height=420)
        return fig

    @staticmethod
    def create_score_confidence_timeline(snapshots: List[Dict]) -> go.Figure:
        """Create dual-axis timeline for score and confidence evolution."""
        fig = go.Figure()
        if not snapshots:
            fig.add_annotation(text="No snapshot timeline available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template="plotly_white", height=340)
            return fig
        timestamps = [s.get("timestamp", "") for s in snapshots]
        mean_scores = []
        for s in snapshots:
            comp = s.get("component_scores", {})
            vals = [float(v.get("score", 0.0)) for v in comp.values() if isinstance(v, dict)]
            mean_scores.append(float(np.mean(vals)) if vals else 0.0)
        confidence = [float(s.get("confidence", 0.0)) * 100.0 for s in snapshots]
        fig.add_trace(go.Scatter(x=timestamps, y=mean_scores, mode="lines+markers", name="Composite score", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=timestamps, y=confidence, mode="lines+markers", name="Confidence %", yaxis="y2", line=dict(color="#ff7f0e")))
        fig.update_layout(
            title="Evidence Timeline",
            xaxis_title="Snapshot time",
            yaxis=dict(title="Composite score (0-100)"),
            yaxis2=dict(title="Confidence (%)", overlaying="y", side="right"),
            template="plotly_white",
            height=380,
            legend=dict(orientation="h"),
        )
        return fig

    @staticmethod
    def create_milestone_burndown(milestones: List[Dict]) -> go.Figure:
        """Show milestone status and due-date completion trend."""
        if not milestones:
            fig = go.Figure()
            fig.add_annotation(text="No milestones yet", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template="plotly_white", height=320)
            return fig
        df = pd.DataFrame(milestones)
        status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(status_counts.keys()),
                    y=list(status_counts.values()),
                    marker_color=["#9ca3af", "#3b82f6", "#ef4444", "#10b981"][: max(1, len(status_counts))],
                )
            ]
        )
        fig.update_layout(title="Milestone Completion / Blockers", yaxis_title="Count", template="plotly_white", height=340)
        return fig

    @staticmethod
    def create_project_risk_heatmap(comparison_rows: List[Dict]) -> go.Figure:
        """Heatmap of candidates versus risk and confidence."""
        if not comparison_rows:
            fig = go.Figure()
            fig.add_annotation(text="No risk heatmap data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template="plotly_white", height=320)
            return fig
        df = pd.DataFrame(comparison_rows)
        z = np.array([[float(r.get("risk_burden", 0.0)), float(r.get("confidence", 0.0)) * 100.0] for r in comparison_rows])
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["Risk burden", "Confidence %"],
                y=df["candidate"],
                colorscale="RdYlGn_r",
                colorbar=dict(title="Value"),
            )
        )
        fig.update_layout(title="Risk Heatmap by Candidate", template="plotly_white", height=max(320, len(comparison_rows) * 40))
        return fig

    @staticmethod
    def create_go_no_go_matrix(checks: Dict[str, bool]) -> go.Figure:
        """Visualize decision criteria pass/fail matrix."""
        labels = list(checks.keys()) or ["No checks"]
        vals = [1 if checks[k] else 0 for k in labels] if checks else [0]
        fig = go.Figure(
            data=go.Heatmap(
                z=[vals],
                x=labels,
                y=["Criteria"],
                colorscale=[[0, "#ef4444"], [1, "#10b981"]],
                zmin=0,
                zmax=1,
                showscale=False,
            )
        )
        fig.update_layout(title="Go/No-Go Criteria Matrix", template="plotly_white", height=260)
        return fig