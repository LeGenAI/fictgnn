# -*- coding: utf-8 -*-
"""
Enhanced Temporal Graph Visualizer

Interactive visualization module for temporal graph analysis with multi-year data (2020-2024).
Provides comprehensive visualization capabilities for semiconductor industry network evolution.
"""

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class EnhancedTemporalGraphVisualizer:
    """Advanced temporal graph visualization for multi-year analysis (2020-2024)."""
    
    def __init__(self, graph_data, node_mapping, reverse_mapping, metadata):
        self.graph_data = graph_data
        self.node_mapping = node_mapping
        self.reverse_mapping = reverse_mapping
        self.metadata = metadata
        
        # Organize data by temporal structure
        self.yearly_data = self._organize_data_by_year()
        
    def _organize_data_by_year(self):
        """Organize data into yearly and quarterly structure."""
        yearly_data = {}
        
        for idx, (company, quarter) in self.reverse_mapping.items():
            # Extract year from quarter string (e.g., "2024Q1" -> "2024")
            if quarter and len(quarter) >= 4:
                year = quarter[:4]
                if year not in yearly_data:
                    yearly_data[year] = {
                        'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []
                    }
                
                # Extract quarter information
                if quarter.endswith('Q1'):
                    yearly_data[year]['Q1'].append({'idx': idx, 'company': company, 'quarter': quarter})
                elif quarter.endswith('Q2'):
                    yearly_data[year]['Q2'].append({'idx': idx, 'company': company, 'quarter': quarter})
                elif quarter.endswith('Q3'):
                    yearly_data[year]['Q3'].append({'idx': idx, 'company': company, 'quarter': quarter})
                elif quarter.endswith('Q4'):
                    yearly_data[year]['Q4'].append({'idx': idx, 'company': company, 'quarter': quarter})
                    
        return yearly_data
    
    def create_yearly_evolution_visualization(self, save_path: str = "semiconductor_evolution_2020_2024.html"):
        """Create interactive visualization showing yearly evolution of the network."""
        logger.info("Generating yearly evolution visualization...")
        
        # Reduce embeddings to 3D using PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(self.graph_data.x.numpy())
        
        # Define color palette for years
        year_colors = {
            '2020': '#FF6B6B',  # Red
            '2021': '#4ECDC4',  # Teal
            '2022': '#45B7D1',  # Blue
            '2023': '#96CEB4',  # Light Green
            '2024': '#FFEAA7'   # Yellow
        }
        
        # Company marker styles
        company_markers = self._assign_company_markers()
        
        # Create figure
        fig = go.Figure()
        
        # Generate traces by year
        for year in sorted(self.yearly_data.keys()):
            year_data = self.yearly_data[year]
            
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                if not year_data[quarter]:
                    continue
                    
                # Extract quarterly data
                x_coords = []
                y_coords = []
                z_coords = []
                texts = []
                markers = []
                
                for item in year_data[quarter]:
                    idx = item['idx']
                    company = item['company']
                    quarter_name = item['quarter']
                    
                    x_coords.append(embeddings_3d[idx, 0])
                    y_coords.append(embeddings_3d[idx, 1])
                    z_coords.append(embeddings_3d[idx, 2])
                    texts.append(f"<b>{company}</b><br>{quarter_name}<br>Year: {year}")
                    markers.append(company_markers.get(company, 'circle'))
                
                # Quarterly opacity adjustment
                quarter_opacity = {'Q1': 0.6, 'Q2': 0.7, 'Q3': 0.8, 'Q4': 0.9}
                
                # Add trace
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers+text',
                    marker=dict(
                        size=8,
                        color=year_colors[year],
                        opacity=quarter_opacity[quarter],
                        line=dict(width=1, color='white'),
                        symbol='circle'  # Safe default marker
                    ),
                    text=[f"{item['company']}" for item in year_data[quarter]],
                    textposition="middle center",
                    textfont=dict(size=10),
                    hovertext=texts,
                    hoverinfo='text',
                    name=f'{year} {quarter}',
                    visible=True if year == '2024' else 'legendonly'  # Show 2024 by default
                ))
        
        # Add edge traces (temporal connections)
        edge_traces = self._create_temporal_edge_traces(embeddings_3d)
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Layout configuration
        fig.update_layout(
            title={
                'text': 'Semiconductor Industry Network Evolution (2020-2024)<br><sub>Yearly/Quarterly Temporal Analysis</sub>',
                'x': 0.5,
                'font': {'size': 24}
            },
            scene=dict(
                xaxis_title='Embedding Dimension 1 (Text Features)',
                yaxis_title='Embedding Dimension 2 (Financial Features)',
                zaxis_title='Embedding Dimension 3 (Industry Features)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='rgba(0,0,0,0.9)',
                xaxis=dict(backgroundcolor="rgb(200, 200, 230)", showgrid=True),
                yaxis=dict(backgroundcolor="rgb(230, 200, 230)", showgrid=True),
                zaxis=dict(backgroundcolor="rgb(230, 230, 200)", showgrid=True)
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='Black',
                borderwidth=1
            ),
            width=1400,
            height=900,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(args=[{"visible": [True if year == '2020' else False for year in sorted(self.yearly_data.keys()) for _ in range(4)] + [True] * len(edge_traces)}],
                             label="2020", method="restyle"),
                        dict(args=[{"visible": [True if year == '2021' else False for year in sorted(self.yearly_data.keys()) for _ in range(4)] + [True] * len(edge_traces)}],
                             label="2021", method="restyle"),
                        dict(args=[{"visible": [True if year == '2022' else False for year in sorted(self.yearly_data.keys()) for _ in range(4)] + [True] * len(edge_traces)}],
                             label="2022", method="restyle"),
                        dict(args=[{"visible": [True if year == '2023' else False for year in sorted(self.yearly_data.keys()) for _ in range(4)] + [True] * len(edge_traces)}],
                             label="2023", method="restyle"),
                        dict(args=[{"visible": [True if year == '2024' else False for year in sorted(self.yearly_data.keys()) for _ in range(4)] + [True] * len(edge_traces)}],
                             label="2024", method="restyle"),
                        dict(args=[{"visible": [True] * (len(sorted(self.yearly_data.keys())) * 4 + len(edge_traces))}],
                             label="All", method="restyle")
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Save visualization
        fig.write_html(save_path)
        logger.info(f"Yearly evolution visualization saved: {save_path}")
        
        return fig
    
    def create_company_focused_networks(self, save_path: str = "company_focused_networks_2020_2024.html"):
        """Create company-focused network visualization (Samsung Electronics, SK Hynix, and other companies)"""
        logger.info("Creating company-focused network visualization...")
        
        # Reduce embeddings to 2D for clearer visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(self.graph_data.x.numpy())
        
        # Identify important companies (ensuring diversity)
        important_companies = self._identify_diverse_companies()
        
        fig = go.Figure()
        
        # Create company-specific subplots
        for i, company in enumerate(important_companies):
            company_data = self._get_company_temporal_data(company)
            
            if not company_data:
                continue
            
            # Track temporal changes for this company
            x_coords = []
            y_coords = []
            texts = []
            colors = []
            sizes = []
            
            for item in company_data:
                idx = item['idx']
                quarter = item['quarter']
                
                x_coords.append(embeddings_2d[idx, 0])
                y_coords.append(embeddings_2d[idx, 1])
                texts.append(f"<b>{company}</b><br>{quarter}")
                
                # Color by year
                year = quarter[:4] if quarter else "2024"
                colors.append(self._get_year_color_numeric(year))
                
                # Size by quarter (latest quarters are larger)
                quarter_num = self._get_quarter_numeric(quarter)
                sizes.append(8 + quarter_num * 2)
            
            # Add company-specific trace
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+lines+text',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                line=dict(width=2, color=f'rgb({i*50%255}, {i*80%255}, {i*120%255})'),
                text=[company[:6] for _ in x_coords],  # Abbreviated company names
                textposition="middle center",
                textfont=dict(size=8),
                hovertext=texts,
                hoverinfo='text',
                name=company,
                connectgaps=True
            ))
        
        # Layout configuration
        fig.update_layout(
            title={
                'text': 'Major Semiconductor Company Network Analysis (2020-2024)<br><sub>Company-wise Temporal Trajectories and Interactions</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='t-SNE Dimension 1 (Company Characteristics)',
            yaxis_title='t-SNE Dimension 2 (Market Position)',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='Black',
                borderwidth=1
            ),
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        # Save visualization
        fig.write_html(save_path)
        logger.info(f"Company-focused network visualization saved: {save_path}")
        
        return fig
    
    def _assign_company_markers(self):
        """Assign unique markers for companies (Plotly 3D compatible)"""
        # Use only markers supported by Plotly 3D scatter
        marker_styles = ['circle', 'circle-open', 'cross', 'diamond', 
                        'diamond-open', 'square', 'square-open', 'x']
        
        companies = set(company for company, _ in self.reverse_mapping.values())
        company_markers = {}
        
        for i, company in enumerate(sorted(companies)):
            company_markers[company] = marker_styles[i % len(marker_styles)]
            
        return company_markers
    
    def _create_temporal_edge_traces(self, embeddings_3d):
        """Create temporal connection edge traces"""
        edge_traces = []
        
        if not hasattr(self.graph_data, 'edge_index') or not hasattr(self.graph_data, 'edge_attr'):
            return edge_traces
        
        temporal_edges_x = []
        temporal_edges_y = []
        temporal_edges_z = []
        
        # Handle edge_index if it's a tensor
        if isinstance(self.graph_data.edge_index, torch.Tensor):
            edge_index = self.graph_data.edge_index.numpy()
        else:
            edge_index = self.graph_data.edge_index
        
        for i, attr in enumerate(self.graph_data.edge_attr):
            if i >= len(edge_index[0]) or attr.get('type') != 'temporal':
                continue
                
            source_idx = edge_index[0][i]
            target_idx = edge_index[1][i]
            
            if source_idx < len(embeddings_3d) and target_idx < len(embeddings_3d):
                source_pos = embeddings_3d[source_idx]
                target_pos = embeddings_3d[target_idx]
                
                temporal_edges_x.extend([source_pos[0], target_pos[0], None])
                temporal_edges_y.extend([source_pos[1], target_pos[1], None])
                temporal_edges_z.extend([source_pos[2], target_pos[2], None])
        
        # Temporal edge trace
        if len(temporal_edges_x) > 0:
            edge_traces.append(go.Scatter3d(
                x=temporal_edges_x,
                y=temporal_edges_y,
                z=temporal_edges_z,
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                name='Temporal Connections',
                hoverinfo='none',
                showlegend=False
            ))
        
        return edge_traces
    
    def _identify_diverse_companies(self):
        """Identify diverse companies (Samsung Electronics, SK Hynix, etc.)"""
        # Extract all companies
        all_companies = set(company for company, _ in self.reverse_mapping.values())
        
        # Priority: balance of large corporations + mid-tier + specialized companies
        priority_companies = [
            'Samsung Electronics', 'SK Hynix', 'LG Innotek',  # Large corporations
            'NEPES', 'SimTech', 'Silicon Works',  # Mid-tier companies  
            'Wonik IPS', 'TechWing', 'Duksan Neolux',  # Specialized companies
            'KOYOUNG', 'Eugene Technology', 'RENO Industrial'  # Equipment/Materials companies
        ]
        
        # Select only companies present in actual data
        selected_companies = []
        for company in priority_companies:
            if company in all_companies:
                selected_companies.append(company)
            if len(selected_companies) >= 10:  # Maximum 10 companies
                break
        
        # Fill with other companies if insufficient
        if len(selected_companies) < 8:
            remaining_companies = list(all_companies - set(selected_companies))
            selected_companies.extend(remaining_companies[:8-len(selected_companies)])
        
        return selected_companies[:10]
    
    def _get_company_temporal_data(self, company):
        """Extract temporal data for a specific company"""
        company_data = []
        
        for idx, (comp, quarter) in self.reverse_mapping.items():
            if comp == company:
                company_data.append({
                    'idx': idx,
                    'company': comp,
                    'quarter': quarter
                })
        
        # Sort by time
        company_data.sort(key=lambda x: x['quarter'] if x['quarter'] else '2024Q1')
        return company_data
    
    def _get_year_color_numeric(self, year):
        """Convert year to numeric color"""
        year_mapping = {'2020': 0, '2021': 1, '2022': 2, '2023': 3, '2024': 4}
        return year_mapping.get(year, 4)
    
    def _get_quarter_numeric(self, quarter):
        """Convert quarter to numeric value"""
        if not quarter or len(quarter) < 6:
            return 1
        quarter_char = quarter[-1]
        return int(quarter_char) if quarter_char.isdigit() else 1