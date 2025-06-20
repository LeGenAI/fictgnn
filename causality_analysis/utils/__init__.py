"""
Utilities Module for Semiconductor Industry Analysis

This module provides comprehensive utilities for academic research including
data processing, performance monitoring, visualization, and logging capabilities.

Classes:
    MemoryOptimizer: Memory usage optimization utilities
    DataValidator: Data validation and quality assurance
    FileManager: File management and I/O operations
    DataProcessor: Advanced data processing for financial analysis
    PerformanceMonitor: Performance timing and benchmarking
    ResourceMonitor: System resource monitoring
    EnhancedTemporalGraphVisualizer: Advanced graph visualization
    LoggerSetup: Logging configuration management
    StructuredLogger: Structured logging for research operations
    ResearchLogger: High-level research workflow logging
    
Functions:
    convert_numpy_types: Convert NumPy types to Python native types
    validate_financial_quarters: Validate financial quarter formats
    normalize_company_names: Standardize company name formats
    calculate_financial_ratios: Calculate financial ratios from raw data
    create_research_metadata: Generate standardized research metadata
    performance_timer: Decorator for automatic function timing
"""

from .data_utils import MemoryOptimizer, DataValidator, FileManager, DataProcessor
from .performance import PerformanceMonitor, ResourceMonitor, performance_timer, Benchmark
from .visualization import EnhancedTemporalGraphVisualizer
from .logging_utils import LoggerSetup, StructuredLogger, ResearchLogger
from .helper_functions import (
    convert_numpy_types, 
    validate_financial_quarters,
    normalize_company_names,
    calculate_financial_ratios,
    create_research_metadata
)

__all__ = [
    # Data utilities
    'MemoryOptimizer', 'DataValidator', 'FileManager', 'DataProcessor',
    # Performance utilities
    'PerformanceMonitor', 'ResourceMonitor', 'performance_timer', 'Benchmark',
    # Visualization utilities
    'EnhancedTemporalGraphVisualizer',
    # Logging utilities
    'LoggerSetup', 'StructuredLogger', 'ResearchLogger',
    # Helper functions
    'convert_numpy_types', 'validate_financial_quarters', 'normalize_company_names',
    'calculate_financial_ratios', 'create_research_metadata'
]