"""
FicTGNN: Financial Technology Graph Neural Networks for Causal Analysis

A comprehensive framework for analyzing causal relationships in financial networks
using multimodal graph neural networks and temporal analysis techniques.

Modules:
- data_collection: Financial data collection and preprocessing
- embedding_generation: Multimodal embedding generation from financial text and numerical data
- causality_analysis: Graph neural network-based causal relationship analysis
- experiments: Experimental configurations and benchmark implementations
- utils: Utility functions and shared components

Author: Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "FICTGNN Research Team"

from .data_collection import *
from .embedding_generation import *
from .causality_analysis import *
from .experiments import *
from .utils import *