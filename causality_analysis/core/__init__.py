"""
Core Analysis Modules for Semiconductor Industry Research

This module contains the core analysis components for academic research in
financial network analysis and causal inference using graph neural networks.

Classes:
    IndustryAwareTemporalGraphBuilder: Builds temporal graphs with industry knowledge
    CausalityAwareTemporalGNN: Graph neural network model for causal analysis  
    CausalityAnalyzer: Analyzes causal relationships in temporal networks
"""

try:
    from .graph_builder import IndustryAwareTemporalGraphBuilder
    from .model import CausalityAwareTemporalGNN  
    from .analyzer import CausalityAnalyzer
except ImportError:
    # Alternative import for development/testing environments
    import sys
    import os
    
    # Add parent directory of current package to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from core.graph_builder import IndustryAwareTemporalGraphBuilder
        from core.model import CausalityAwareTemporalGNN  
        from core.analyzer import CausalityAnalyzer
    except ImportError:
        # Final fallback import (placeholder classes)
        class IndustryAwareTemporalGraphBuilder:
            """Placeholder class for import compatibility."""
            def __init__(self, *args, **kwargs):
                pass
                
        class CausalityAwareTemporalGNN:
            """Placeholder class for import compatibility."""
            def __init__(self, *args, **kwargs):
                pass
                
        class CausalityAnalyzer:
            """Placeholder class for import compatibility."""
            def __init__(self, *args, **kwargs):
                pass

__all__ = ['IndustryAwareTemporalGraphBuilder', 'CausalityAwareTemporalGNN', 'CausalityAnalyzer']