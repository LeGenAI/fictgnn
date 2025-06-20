#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Utilities for Semiconductor Industry Analysis

Comprehensive data processing utilities for academic research in financial network
analysis and causal inference in the semiconductor industry.

Author: Research Team
Date: 2024
Version: 2.0 (Modularized)
"""

import logging
import psutil
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import json


class MemoryOptimizer:
    """Memory usage optimization utilities for large-scale data processing."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Memory usage information in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Physical memory
            'vms': memory_info.vms / 1024 / 1024,  # Virtual memory
            'percent': process.memory_percent(),     # Memory usage percentage
            'available': psutil.virtual_memory().available / 1024 / 1024  # Available memory
        }
    
    @staticmethod
    def check_memory_limit(max_memory_mb: int = 8192) -> bool:
        """
        Check if memory usage is within specified limit.
        
        Args:
            max_memory_mb: Maximum memory usage limit in MB
            
        Returns:
            Whether memory usage is within limit
        """
        current_memory = MemoryOptimizer.get_memory_usage()
        return current_memory['rss'] < max_memory_mb
    
    @staticmethod
    def optimize_memory():
        """
        Perform memory optimization by forcing garbage collection.
        """
        gc.collect()
        logging.info("Memory optimization completed")


class DataValidator:
    """Data validation utilities for ensuring data quality in research."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame is empty")
        
        # Basic statistics
        validation_results['statistics'] = {
            'shape': df.shape,
            'null_count': df.isnull().sum().sum(),
            'duplicate_count': df.duplicated().sum()
        }
        
        return validation_results
    
    @staticmethod
    def validate_financial_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate financial data specific requirements.
        
        Args:
            df: Financial data DataFrame
            
        Returns:
            Financial data validation results
        """
        required_financial_columns = ['company', 'quarter', 'revenue', 'operating_income']
        validation = DataValidator.validate_dataframe(df, required_financial_columns)
        
        if validation['is_valid']:
            # Additional financial data checks
            if 'revenue' in df.columns:
                negative_revenue = (df['revenue'] < 0).sum()
                if negative_revenue > 0:
                    validation['issues'].append(f"Found {negative_revenue} records with negative revenue")
        
        return validation


class FileManager:
    """File management utilities for research data handling."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def save_json(data: Any, filepath: Union[str, Path]) -> None:
        """
        Save data as JSON file with proper encoding.
        
        Args:
            data: Data to save
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Any:
        """
        Load data from JSON file.
        
        Args:
            filepath: JSON file path
            
        Returns:
            Loaded data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Data loaded from {filepath}")
        return data


class DataProcessor:
    """Advanced data processing utilities for financial network analysis."""
    
    @staticmethod
    def normalize_financial_data(df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize financial data using specified method.
        
        Args:
            df: Financial data DataFrame
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized DataFrame
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        normalized_df = df.copy()
        
        if method == 'zscore':
            normalized_df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        elif method == 'minmax':
            normalized_df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
        elif method == 'robust':
            median = df[numeric_columns].median()
            mad = np.abs(df[numeric_columns] - median).median()
            normalized_df[numeric_columns] = (df[numeric_columns] - median) / mad
        
        logging.info(f"Data normalized using {method} method")
        return normalized_df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in financial time series data.
        
        Args:
            df: DataFrame with missing values
            strategy: Strategy for handling missing values
            
        Returns:
            DataFrame with missing values handled
        """
        processed_df = df.copy()
        
        if strategy == 'forward_fill':
            processed_df = processed_df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            processed_df = processed_df.fillna(method='bfill')
        elif strategy == 'interpolate':
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_columns] = processed_df[numeric_columns].interpolate()
        elif strategy == 'mean':
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_columns] = processed_df[numeric_columns].fillna(processed_df[numeric_columns].mean())
        
        logging.info(f"Missing values handled using {strategy} strategy")
        return processed_df