#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Configuration Management for Semiconductor Industry Analysis

Centralized configuration management system with YAML support for academic research
in financial network analysis and causal inference.

Author: FICTGNN Research Team
Date: 2024
Version: 2.0 (Modularized with YAML support)
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Centralized configuration management class with YAML support."""
    
    _instance = None
    _config_data = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config_data is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """
        Load YAML configuration file.
        
        Args:
            config_path: Configuration file path (default: config/settings.yaml)
        """
        if config_path is None:
            # Find settings.yaml relative to current file location
            current_dir = Path(__file__).parent
            config_path = current_dir / "settings.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            print(f"Configuration file loaded successfully: {config_path}")
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            self._load_default_config()
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default configuration when YAML file is not available."""
        self._config_data = {
            'model': {
                'input_dim': 28,
                'hidden_dim': 256,
                'output_dim': 128,
                'num_heads': 8,
                'dropout': 0.1,
                'learning_rate': 0.001
            },
            'embedding': {
                'target_variance': 0.95,
                'min_dims': 12,
                'max_dims': 25,
                'text_weight': 0.5,
                'financial_weight': 0.3,
                'industry_weight': 0.2
            },
            'graph': {
                'base_threshold': 0.6,
                'max_lag': 4,
                'temporal_weight_decay': True,
                'causal_threshold_decay': 0.05
            },
            'paths': {
                'output_dir': 'outputs'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'filename': 'causality_analysis.log'
            }
        }
        print("Default configuration loaded")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Access configuration values using dot notation.
        
        Args:
            key_path: Configuration path (e.g., 'model.input_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default value
        """
        keys = key_path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default    
    def set(self, key_path: str, new_value: Any) -> None:
        """
        Dynamically update configuration values.
        
        Args:
            key_path: Configuration path (e.g., 'graph.base_threshold')
            new_value: New value to set
        """
        keys = key_path.split('.')
        value = self._config_data
        
        # Navigate to intermediate path, excluding last key
        for key in keys[:-1]:
            if key not in value:
                value[key] = {}
            value = value[key]
        
        # Set value for the last key
        value[keys[-1]] = new_value
        print(f"Configuration updated: {key_path} = {new_value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Return all settings for a specific section.
        
        Args:
            section: Section name (e.g., 'model', 'graph')
            
        Returns:
            Section configuration dictionary
        """
        return self._config_data.get(section, {})
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            Configuration validation result
        """
        validation_errors = []
        
        # 1. Validate embedding weight sum
        embedding = self.get_section('embedding')
        if embedding:
            weight_sum = (
                embedding.get('text_weight', 0) + 
                embedding.get('financial_weight', 0) + 
                embedding.get('industry_weight', 0)
            )
            if abs(weight_sum - 1.0) > 0.01:
                validation_errors.append(f"Embedding weight sum error: {weight_sum} (expected: 1.0)")
        
        # 2. Validate threshold ranges
        base_threshold = self.get('graph.base_threshold', 0.6)
        if not (0.0 <= base_threshold <= 1.0):
            validation_errors.append(f"base_threshold range error: {base_threshold} (expected: 0.0-1.0)")
        
        # 3. Validate dimension settings
        min_dims = self.get('embedding.min_dims', 12)
        max_dims = self.get('embedding.max_dims', 25)
        if min_dims >= max_dims:
            validation_errors.append(f"Dimension setting error: min_dims({min_dims}) >= max_dims({max_dims})")
        
        # 4. Validate required file paths
        required_paths = ['financial_data', 'embeddings', 'metadata']
        for path_key in required_paths:
            path_value = self.get(f'paths.{path_key}')
            if not path_value:
                validation_errors.append(f"Required path missing: paths.{path_key}")
        
        if validation_errors:
            print("Configuration validation failed:")
            for error in validation_errors:
                print(f"   - {error}")
            return False
        else:
            print("Configuration validation completed successfully")
            return True
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Output file path
        """
        if output_path is None:
            output_path = "config_backup.yaml"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            print(f"Configuration saved successfully: {output_path}")
        except Exception as e:
            print(f"Configuration save failed: {e}")
    
    def reload_config(self, config_path: Optional[str] = None) -> None:
        """
        Reload configuration file.
        
        Args:
            config_path: Configuration file path
        """
        self.load_config(config_path)
        print("Configuration file reloaded successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config_data.copy()
    
    def merge_config(self, additional_config: Dict[str, Any]) -> None:
        """
        Merge with additional configuration.
        
        Args:
            additional_config: Configuration dictionary to merge
        """
        def deep_merge(base_dict, merge_dict):
            for key, value in merge_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(self._config_data, additional_config)
        print("Configuration merge completed")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return model configuration (convenience method)."""
        return self.get_section('model')
    
    def get_graph_config(self) -> Dict[str, Any]:
        """Return graph configuration (convenience method)."""
        return self.get_section('graph')
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Return embedding configuration (convenience method)."""
        return self.get_section('embedding')


# Create global configuration instance
config = Config()