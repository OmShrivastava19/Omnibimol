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