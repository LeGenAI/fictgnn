# -*- coding: utf-8 -*-
"""
Helper Functions Module

Utility functions for data type conversion, validation, and common operations
used throughout the semiconductor industry analysis framework.

Author: Research Team
Date: 2024
Version: 2.0 (Academic Research)
"""

import numpy as np
from typing import Any, Union, Dict, List


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert NumPy types to Python native types for JSON serialization.
    
    This function recursively converts NumPy data types to Python native types,
    enabling proper JSON serialization of research results.
    
    Args:
        obj: Object to convert (can be nested dictionaries, lists, etc.)
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def validate_financial_quarters(quarters: List[str]) -> List[str]:
    """
    Validate and standardize financial quarter strings.
    
    Args:
        quarters: List of quarter strings (e.g., ['2020Q1', '2020Q2'])
        
    Returns:
        List of validated and standardized quarter strings
        
    Raises:
        ValueError: If quarter format is invalid
    """
    validated_quarters = []
    
    for quarter in quarters:
        if not isinstance(quarter, str):
            raise ValueError(f"Quarter must be string, got {type(quarter)}")
        
        # Check format: YYYYQX
        if len(quarter) != 6 or quarter[4] != 'Q':
            raise ValueError(f"Invalid quarter format: {quarter}. Expected YYYYQX")
        
        try:
            year = int(quarter[:4])
            quarter_num = int(quarter[5])
        except ValueError:
            raise ValueError(f"Invalid quarter format: {quarter}")
        
        if not (2000 <= year <= 2030):
            raise ValueError(f"Year out of reasonable range: {year}")
        
        if not (1 <= quarter_num <= 4):
            raise ValueError(f"Quarter number must be 1-4, got {quarter_num}")
        
        validated_quarters.append(quarter)
    
    return validated_quarters


def normalize_company_names(companies: List[str]) -> List[str]:
    """
    Normalize company names for consistent analysis.
    
    Args:
        companies: List of company names
        
    Returns:
        List of normalized company names
    """
    normalized = []
    
    for company in companies:
        # Remove common suffixes and normalize
        normalized_name = company.strip()
        
        # Remove common Korean corporate suffixes
        suffixes_to_remove = ['(주)', '㈜', 'Co.,Ltd.', 'Ltd.', 'Inc.', 'Corp.']
        for suffix in suffixes_to_remove:
            if normalized_name.endswith(suffix):
                normalized_name = normalized_name[:-len(suffix)].strip()
        
        normalized.append(normalized_name)
    
    return normalized


def calculate_financial_ratios(financial_data: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate common financial ratios from financial data.
    
    Args:
        financial_data: Dictionary containing financial metrics
        
    Returns:
        Dictionary of calculated financial ratios
    """
    ratios = {}
    
    # Safe division helper
    def safe_divide(numerator, denominator, default=np.nan):
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    
    # Common financial ratios
    if 'operating_income' in financial_data and 'revenue' in financial_data:
        ratios['operating_margin'] = safe_divide(
            financial_data['operating_income'], 
            financial_data['revenue']
        )
    
    if 'net_income' in financial_data and 'revenue' in financial_data:
        ratios['net_margin'] = safe_divide(
            financial_data['net_income'], 
            financial_data['revenue']
        )
    
    if 'total_debt' in financial_data and 'total_equity' in financial_data:
        ratios['debt_to_equity'] = safe_divide(
            financial_data['total_debt'], 
            financial_data['total_equity']
        )
    
    if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
        ratios['current_ratio'] = safe_divide(
            financial_data['current_assets'], 
            financial_data['current_liabilities']
        )
    
    return ratios


def create_research_metadata(analysis_type: str, 
                           parameters: Dict[str, Any],
                           data_sources: List[str]) -> Dict[str, Any]:
    """
    Create standardized metadata for research analysis.
    
    Args:
        analysis_type: Type of analysis performed
        parameters: Analysis parameters used
        data_sources: List of data sources
        
    Returns:
        Standardized metadata dictionary
    """
    from datetime import datetime
    
    metadata = {
        'analysis_type': analysis_type,
        'timestamp': datetime.now().isoformat(),
        'parameters': convert_numpy_types(parameters),
        'data_sources': data_sources,
        'framework_version': '2.0',
        'reproducibility': {
            'random_seed': parameters.get('random_seed'),
            'software_versions': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'pandas': pd.__version__ if 'pd' in globals() else 'not_imported'
            }
        }
    }
    
    return metadata