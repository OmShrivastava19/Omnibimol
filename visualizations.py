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
                                document.getElementById('loading').innerHTML = 
                                    '<span class="error">Error loading structure: ' + error.message + '</span>';
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
            fig.add_annotation(
                text=f"Could not load confidence data: {str(e)}",
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