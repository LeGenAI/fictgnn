"""
Configuration Module for Semiconductor Industry Analysis

This module provides centralized configuration management for academic research
in financial network analysis and causal inference.

Classes:
    Config: Centralized configuration management with YAML support
    
Constants:
    SEMICONDUCTOR_INDUSTRY_KNOWLEDGE: Comprehensive industry domain knowledge database
"""

from .config import Config, config
from .industry_knowledge import SEMICONDUCTOR_INDUSTRY_KNOWLEDGE

__all__ = ['Config', 'config', 'SEMICONDUCTOR_INDUSTRY_KNOWLEDGE']