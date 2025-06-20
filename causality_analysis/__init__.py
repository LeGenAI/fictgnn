"""
Semiconductor Industry Temporal Graph Neural Network Analysis System

Industry-Aware Causality Network for Semiconductor Companies
A comprehensive framework for analyzing causal relationships in financial networks
using temporal graph neural networks with integrated industry knowledge.

Author: Research Team
Date: 2024
Version: 2.0 (Modularized)
"""

__version__ = "2.0.0"
__author__ = "FICTGNN Research Team"
__description__ = "Industry-Aware Causality Network for Semiconductor Companies"

# Import main classes
from .core.graph_builder import IndustryAwareTemporalGraphBuilder
from .core.model import CausalityAwareTemporalGNN
from .core.analyzer import CausalityAnalyzer
from .utils.visualization import EnhancedTemporalGraphVisualizer
from .config.config import Config

__all__ = [
    'IndustryAwareTemporalGraphBuilder',
    'CausalityAwareTemporalGNN', 
    'CausalityAnalyzer',
    'EnhancedTemporalGraphVisualizer',
    'Config'
]