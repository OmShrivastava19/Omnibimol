import pandas as pd
import plotly.graph_objects as go
from typing import Dict

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
            return "<p style='text-align:center; color:gray;'>No structure available</p>"

        if not pdb_url:
            debug_info = f"Debug: structure_type={structure_type}, available={structure_data.get('available')}, data={str(structure_data)[:200]}..."
            return f"<p style='text-align:center; color:red;'>No structure URL available<br><small style='color:gray;'>{debug_info}</small></p>"
        
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
                <div id="loading">Loading 3D structure...</div>
            </div>
            <div id="title">{title}</div>
            
            <script>
                (function() {{
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
                        stage.loadFile("{pdb_url}", {{defaultRepresentation: false}})
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
                                var message = (error && error.message) ? error.message : (error ? error.toString() : 'Unknown error');
                                document.getElementById('loading').innerHTML = 
                                    '<span class="error">Error loading structure: ' + message + '</span>';
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
    def create_confidence_plot(uniprot_id: str, entry_id: str = None) -> go.Figure:
        """
        Create plot showing AlphaFold confidence scores along sequence
        pLDDT scores: >90=very high, 70-90=confident, 50-70=low, <50=very low
        """
        import httpx
        try:
            if not entry_id:
                entry_id = f"AF-{uniprot_id}-F1"
            entry_id = entry_id.replace(".pdb", "").replace(".cif", "")
            if "-model_v" in entry_id:
                entry_id = entry_id.split("-model_v")[0]
            if not entry_id.startswith("AF-"):
                entry_id = f"AF-{entry_id}"
            pdb_url = f"https://alphafold.ebi.ac.uk/files/{entry_id}-model_v4.pdb"
            
            response = httpx.get(pdb_url, timeout=30.0)
            response.raise_for_status()
            pdb_content = response.text
            residues = []
            plddt_scores = []
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM') and line[13:15].strip() == 'CA':
                    try:
                        residue_num = int(line[22:26].strip())
                        bfactor = float(line[60:66].strip())
                        residues.append(residue_num)
                        plddt_scores.append(bfactor)
                    except:
                        continue
            if not residues:
                raise Exception("No confidence data found in PDB file")
            colors = []
            for score in plddt_scores:
                if score > 90:
                    colors.append('#0053D6')
                elif score > 70:
                    colors.append('#65CBF3')
                elif score > 50:
                    colors.append('#FFDB13')
                else:
                    colors.append('#FF7D45')
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
            fig.add_hrect(y0=90, y1=100, fillcolor="rgba(0, 83, 214, 0.1)", line_width=0, annotation_text="Very High", annotation_position="right")
            fig.add_hrect(y0=70, y1=90, fillcolor="rgba(101, 203, 243, 0.1)", line_width=0, annotation_text="Confident", annotation_position="right")
            fig.add_hrect(y0=50, y1=70, fillcolor="rgba(255, 219, 19, 0.1)", line_width=0, annotation_text="Low", annotation_position="right")
            fig.add_hrect(y0=0, y1=50, fillcolor="rgba(255, 125, 69, 0.1)", line_width=0, annotation_text="Very Low", annotation_position="right")
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
            fig = go.Figure()
            error_msg = f"Could not load confidence data from AlphaFold: {str(e)}"
            if "404" in str(e):
                error_msg += f"\nURL attempted: https://alphafold.ebi.ac.uk/files/{entry_id if 'entry_id' in locals() else f'AF-{uniprot_id}-F1'}-model_v4.pdb"
            fig.add_annotation(
                text=error_msg,
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(height=350)
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
        pathway_classes = {}
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
        aa_counts = {}
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
        """
        if not blast_hits:
            return "<p style='text-align:center; color:gray;'>No BLAST results available</p>"
        
        html = """
        <style>
            .blast-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 13px;
            }
            .blast-table th {
                background-color: #2ca02c;
                color: white;
                padding: 10px;
                text-align: left;
                font-weight: bold;
                font-size: 12px;
            }
            .blast-table td {
                padding: 8px 10px;
                border-bottom: 1px solid #ddd;
            }
            .blast-table tr:hover {
                background-color: #f5f5f5;
            }
            .accession-link {
                color: #1f77b4;
                text-decoration: none;
                font-family: monospace;
                font-weight: 500;
            }
            .accession-link:hover {
                text-decoration: underline;
            }
            .organism {
                font-style: italic;
                color: #666;
            }
            .identity-high {
                background-color: #d4edda;
                color: #155724;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            .identity-medium {
                background-color: #fff3cd;
                color: #856404;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            .identity-low {
                background-color: #f8d7da;
                color: #721c24;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            .e-value {
                font-family: monospace;
                font-size: 11px;
            }
        </style>
        
        <table class="blast-table">
            <thead>
                <tr>
                    <th style="width: 12%">Accession</th>
                    <th style="width: 35%">Description</th>
                    <th style="width: 18%">Organism</th>
                    <th style="width: 10%">Identity</th>
                    <th style="width: 10%">Coverage</th>
                    <th style="width: 15%">E-value</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for hit in blast_hits:
            accession = hit.get('accession', 'N/A')
            title = hit.get('title', 'Unknown')[:60] + ('...' if len(hit.get('title', '')) > 60 else '')
            organism = hit.get('organism', 'Unknown')
            identity = hit.get('identity_percent', 0)
            coverage = hit.get('coverage_percent', 0)
            e_value = hit.get('e_value', 1.0)
            
            # Color code identity
            if identity >= 80:
                identity_class = "identity-high"
            elif identity >= 50:
                identity_class = "identity-medium"
            else:
                identity_class = "identity-low"
            
            # Format e-value
            if e_value < 0.0001:
                e_value_str = f"{e_value:.2e}"
            else:
                e_value_str = f"{e_value:.4f}"
            
            html += f"""
            <tr>
                <td>
                    <a href="https://www.ncbi.nlm.nih.gov/protein/{accession}" 
                    target="_blank" class="accession-link">{accession}</a>
                </td>
                <td>{title}</td>
                <td class="organism">{organism}</td>
                <td><span class="{identity_class}">{identity:.1f}%</span></td>
                <td>{coverage:.1f}%</td>
                <td class="e-value">{e_value_str}</td>
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
        feature_types = {}
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
                alert('Docking simulation for ' + name + ' (' + chemblId + ')');
                // In production, this would trigger docking calculation
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