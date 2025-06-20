"""
Data Collection Module

This module handles financial data collection and preprocessing for the FicTGNN framework.
It includes automated collection from multiple financial data sources and comprehensive
data validation and preprocessing capabilities.

Classes:
- CompleteSemiconductorDataCollector: Main data collection class for semiconductor industry data
"""

from .complete_semiconductor_dataset_collector import CompleteSemiconductorDataCollector

__all__ = ['CompleteSemiconductorDataCollector']