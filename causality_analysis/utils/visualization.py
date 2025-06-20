#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Utilities for Semiconductor Industry Analysis

Advanced visualization utilities for academic research in financial network analysis,
including temporal graph evolution, company-focused networks, and causal relationship
visualization for the semiconductor industry.

Author: Research Team
Date: 2024
Version: 2.0 (Modularized)
"""

import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class EnhancedTemporalGraphVisualizer:
    """Enhanced temporal graph visualization class for academic research."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving visualization outputs
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_temporal_network_evolution(self, 
                                        temporal_graphs: Dict[str, Any], 
                                        causality_results: Dict[str, Any],
                                        title: str = "Temporal Network Evolution") -> str:
        """
        Create temporal network evolution visualization.
        
        Args:
            temporal_graphs: Dictionary of temporal graph data
            causality_results: Causal relationship analysis results
            title: Visualization title
            
        Returns:
            Path to generated HTML file
        """
        self.logger.info("Creating temporal network evolution visualization...")
        
        # Create subplot structure
        periods = list(temporal_graphs.keys())
        n_periods = len(periods)
        
        fig = make_subplots(
            rows=2, cols=(n_periods + 1) // 2,
            subplot_titles=[f"Period {period}" for period in periods],
            specs=[[{"type": "scatter"} for _ in range((n_periods + 1) // 2)] for _ in range(2)]
        )
        
        # Generate network layout for each period
        for i, period in enumerate(periods):
            row = (i // ((n_periods + 1) // 2)) + 1
            col = (i % ((n_periods + 1) // 2)) + 1
            
            # Extract graph data for this period
            graph_data = temporal_graphs[period]
            nodes, edges = self._extract_network_elements(graph_data, causality_results)
            
            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=[node['x'] for node in nodes],
                    y=[node['y'] for node in nodes],
                    mode='markers+text',
                    text=[node['name'] for node in nodes],
                    textposition="middle center",
                    marker=dict(
                        size=[node['size'] for node in nodes],
                        color=[node['color'] for node in nodes],
                        line=dict(width=2, color='white')
                    ),
                    name=f"Companies {period}",
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # Add edges
            for edge in edges:
                fig.add_trace(
                    go.Scatter(
                        x=[edge['x0'], edge['x1'], None],
                        y=[edge['y0'], edge['y1'], None],
                        mode='lines',
                        line=dict(
                            width=edge['width'],
                            color=edge['color']
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save to file
        output_file = self.output_dir / "temporal_network_evolution.html"
        fig.write_html(str(output_file))
        
        self.logger.info(f"Temporal network evolution saved to {output_file}")
        return str(output_file)
    
    def create_company_focused_analysis(self, 
                                      temporal_graphs: Dict[str, Any],
                                      causality_results: Dict[str, Any],
                                      focus_companies: List[str] = None) -> str:
        """
        Create company-focused network analysis visualization.
        
        Args:
            temporal_graphs: Dictionary of temporal graph data
            causality_results: Causal relationship analysis results
            focus_companies: List of companies to focus on
            
        Returns:
            Path to generated HTML file
        """
        self.logger.info("Creating company-focused analysis visualization...")
        
        if focus_companies is None:
            focus_companies = self._identify_key_companies(temporal_graphs, causality_results)
        
        # Create individual visualizations for each focus company
        fig = make_subplots(
            rows=len(focus_companies), cols=1,
            subplot_titles=[f"{company} Influence Network" for company in focus_companies],
            vertical_spacing=0.1
        )
        
        for i, company in enumerate(focus_companies):
            # Extract company-specific network data
            company_network = self._extract_company_network(
                company, temporal_graphs, causality_results
            )
            
            # Create network visualization for this company
            nodes, edges = self._create_company_network_elements(company_network)
            
            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=[node['x'] for node in nodes],
                    y=[node['y'] for node in nodes],
                    mode='markers+text',
                    text=[node['name'] for node in nodes],
                    textposition="middle center",
                    marker=dict(
                        size=[node['size'] for node in nodes],
                        color=[node['color'] for node in nodes],
                        line=dict(width=2, color='white')
                    ),
                    name=f"{company} Network",
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )
            
            # Add edges
            for edge in edges:
                fig.add_trace(
                    go.Scatter(
                        x=[edge['x0'], edge['x1'], None],
                        y=[edge['y0'], edge['y1'], None],
                        mode='lines',
                        line=dict(
                            width=edge['width'],
                            color=edge['color']
                        ),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Company-Focused Network Analysis",
            height=400 * len(focus_companies),
            showlegend=True,
            template="plotly_white"
        )
        
        # Save to file
        output_file = self.output_dir / "company_focused_analysis.html"
        fig.write_html(str(output_file))
        
        self.logger.info(f"Company-focused analysis saved to {output_file}")
        return str(output_file)
    
    def _extract_network_elements(self, graph_data: Dict[str, Any], 
                                causality_results: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract nodes and edges from graph data.
        
        Args:
            graph_data: Graph data for a specific period
            causality_results: Causal relationship results
            
        Returns:
            Tuple of (nodes, edges) lists
        """
        # Generate random layout for demonstration
        # In actual implementation, use proper graph layout algorithms
        n_nodes = len(graph_data.get('companies', []))
        
        nodes = []
        for i, company in enumerate(graph_data.get('companies', [])):
            # Generate circular layout
            angle = 2 * np.pi * i / n_nodes
            x = np.cos(angle)
            y = np.sin(angle)
            
            nodes.append({
                'name': company,
                'x': x,
                'y': y,
                'size': 20,  # Base size, can be adjusted based on company metrics
                'color': 'lightblue'
            })
        
        edges = []
        # Add edges based on causality results
        # This is a simplified implementation
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if np.random.random() > 0.7:  # Random edge for demonstration
                    edges.append({
                        'x0': nodes[i]['x'],
                        'y0': nodes[i]['y'],
                        'x1': nodes[j]['x'],
                        'y1': nodes[j]['y'],
                        'width': 2,
                        'color': 'gray'
                    })
        
        return nodes, edges
    
    def _identify_key_companies(self, temporal_graphs: Dict[str, Any], 
                              causality_results: Dict[str, Any]) -> List[str]:
        """
        Identify key companies for focused analysis.
        
        Args:
            temporal_graphs: Temporal graph data
            causality_results: Causal relationship results
            
        Returns:
            List of key company names
        """
        # Simple implementation - return first few companies
        # In actual implementation, use centrality measures or other metrics
        all_companies = set()
        for period_data in temporal_graphs.values():
            all_companies.update(period_data.get('companies', []))
        
        return list(all_companies)[:3]  # Return top 3 for demonstration
    
    def _extract_company_network(self, company: str, 
                               temporal_graphs: Dict[str, Any],
                               causality_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract network data specific to a company.
        
        Args:
            company: Company name
            temporal_graphs: Temporal graph data
            causality_results: Causal relationship results
            
        Returns:
            Company-specific network data
        """
        # Simplified implementation
        return {
            'focal_company': company,
            'connected_companies': [],
            'relationships': []
        }
    
    def _create_company_network_elements(self, company_network: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """
        Create network elements for company-specific visualization.
        
        Args:
            company_network: Company-specific network data
            
        Returns:
            Tuple of (nodes, edges) lists
        """
        # Simplified implementation
        focal_company = company_network['focal_company']
        
        nodes = [{
            'name': focal_company,
            'x': 0,
            'y': 0,
            'size': 30,
            'color': 'red'
        }]
        
        edges = []
        
        return nodes, edges