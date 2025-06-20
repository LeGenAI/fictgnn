#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Semiconductor Report Data Processor
Phase 2: Advanced Data Refinement and Economic Indicator Extraction System

Core Features:
1. Sophisticated economic indicator extraction
2. Duplicate report handling strategies within quarters
3. Cross-modal embedding metadata construction
4. Text preprocessing and refinement
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EconomicIndicators:
    """Economic indicator data structure"""
    
    # Financial metrics
    revenue: Optional[float] = None  # Revenue (trillion KRW)
    operating_profit: Optional[float] = None  # Operating profit (trillion KRW)
    net_profit: Optional[float] = None  # Net profit (trillion KRW)
    
    # Growth rates
    revenue_growth: Optional[float] = None  # Revenue growth rate (%)
    profit_growth: Optional[float] = None  # Profit growth rate (%)
    
    # Valuation
    target_price: Optional[float] = None  # Target price (KRW)
    current_price: Optional[float] = None  # Current price (KRW)
    per: Optional[float] = None  # PER
    pbr: Optional[float] = None  # PBR
    
    # Investment indicators
    capex: Optional[float] = None  # Capital expenditure (trillion KRW)
    r_and_d: Optional[float] = None  # R&D investment (trillion KRW)
    
    # Market indicators
    market_share: Optional[float] = None  # Market share (%)
    capacity_utilization: Optional[float] = None  # Utilization rate (%)
    
    # AI/Semiconductor specialized indicators
    hbm_revenue_ratio: Optional[float] = None  # HBM revenue ratio (%)
    ai_server_exposure: Optional[float] = None  # AI server exposure (%)
    memory_price_change: Optional[float] = None  # Memory price change (%)

    # New addition: Quarterly revenue (KRW)
    quarter_revenue: Optional[float] = None  # Quarterly revenue (KRW)

@dataclass 
class ReportMetadata:
    """Report metadata structure"""
    
    # Basic information
    filename: str
    date: datetime
    company_name: str
    stock_code: str
    securities_firm: str
    report_title: str
    
    # Classification information
    year: int
    quarter: str
    report_type: str  # "earnings", "initiation", "update", "sector"
    
    # Text metadata
    content_length: int
    summary_length: int
    has_financial_tables: bool
    confidence_level: str  # "high", "medium", "low"
    
    # Economic indicators
    economic_indicators: EconomicIndicators
    
    # Cross-modal features
    text_features: Dict[str, Any]
    numerical_features: Dict[str, float]
    
class AdvancedReportProcessor:
    """Advanced report data processor"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.financial_patterns = self._build_financial_patterns()
        self.text_processors = self._build_text_processors()
        
    def _build_financial_patterns(self) -> Dict[str, List[str]]:
        """Build sophisticated financial patterns"""
        return {
            'revenue': [
                r'revenue\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'sales\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'revenue\s*(\d+(?:[,\.]\d+)*)',
                r'(\d+(?:[,\.]\d+)*)\s*trillion.*revenue'
            ],
            'operating_profit': [
                r'operating.*profit\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'operating.*income\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'OP\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'(\d+(?:[,\.]\d+)*)\s*trillion.*operating.*profit'
            ],
            'target_price': [
                r'target.*price\s*(\d+(?:[,\.]\d+)*)',
                r'price.*target\s*(\d+(?:[,\.]\d+)*)',
                r'TP\s*(\d+(?:[,\.]\d+)*)',
                r'target.*?(\d+(?:[,\.]\d+)*)'
            ],
            'growth_rates': [
                r'growth.*rate\s*(\d+(?:\.\d+)?)\s*%',
                r'increase.*rate\s*(\d+(?:\.\d+)?)\s*%',
                r'YoY\s*(\d+(?:\.\d+)?)\s*%'
            ],
            # Addition: Quarterly revenue
            'quarter_revenue': [
                # Example: "Q3 revenue 2.3 trillion", "Q322 revenue 850 billion"
                r'([1-4]Q\d{0,2})[^\d]{0,10}?revenue\s*([\d,\.]+)\s*(trillion|billion)?',
                # Example: "2023 Q2 revenue 1.5 trillion"
                r'(\d{4})\s*Q([1-4])[^\d]{0,10}?revenue\s*([\d,\.]+)\s*(trillion|billion)?'
            ]
        }
    
    def _build_text_processors(self) -> Dict[str, Any]:
        """Build text processing tools"""
        return {
            'positive_keywords': [
                'growth', 'increase', 'improvement', 'favorable', 'expansion', 'strengthen', 'rise', 'expectation',
                'good', 'strong', 'excellent', 'success', 'leap', 'recovery'
            ],
            'negative_keywords': [
                'decrease', 'decline', 'sluggish', 'deterioration', 'concern', 'risk', 'downward', 'adjustment',
                'difficulty', 'weakness', 'burden', 'slowdown'
            ]
        }
    
    def extract_economic_indicators(self, content: str) -> EconomicIndicators:
        """Sophisticated economic indicator extraction"""
        indicators = EconomicIndicators()
        
        try:
            # Revenue extraction
            for pattern in self.financial_patterns['revenue']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.revenue = self._parse_financial_number(matches[0])
                    break
            
            # Target price extraction
            for pattern in self.financial_patterns['target_price']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.target_price = self._parse_financial_number(matches[0])
                    break
            
            # Growth rate extraction
            for pattern in self.financial_patterns['growth_rates']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.revenue_growth = float(matches[0])
                    break

            # Quarterly revenue extraction (use first match found in document)
            for pattern in self.financial_patterns['quarter_revenue']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    grp = matches[0]
                    # Pattern1 result: (qstr, num, unit) / Pattern2: (year, quarter, num, unit)
                    if len(grp) == 3:
                        _, num, unit = grp
                    elif len(grp) == 4:
                        _, _, num, unit = grp
                    else:
                        continue
                    indicators.quarter_revenue = self._parse_number_with_unit(num, unit)
                    break
                    
        except Exception as e:
            logger.warning(f"Error in economic indicator extraction: {e}")
        
        return indicators
    
    def _parse_financial_number(self, number_str: str) -> float:
        """Parse financial numbers"""
        try:
            clean_number = number_str.replace(',', '').replace('.', '')
            return float(clean_number)
        except:
            return 0.0

    # Parse numbers with units (trillion/billion)
    def _parse_number_with_unit(self, num_str: str, unit: str) -> float:
        UNIT_MAP = {
            'trillion': 1e12,
            'billion': 1e9
        }
        try:
            base = float(num_str.replace(',', ''))
            multiplier = UNIT_MAP.get(unit.strip().lower(), 1.0)
            return base * multiplier
        except:
            return 0.0
    
    def extract_text_features(self, content: str) -> Dict[str, Any]:
        """Extract text features"""
        features = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()]),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'has_tables': '|' in content and content.count('|') > 10,
            'sentiment_score': self._calculate_sentiment(content),
            'technical_density': self._calculate_technical_density(content)
        }
        
        return features
    
    def _calculate_sentiment(self, content: str) -> float:
        """Calculate simple sentiment score"""
        positive_count = sum(content.lower().count(word) for word in self.text_processors['positive_keywords'])
        negative_count = sum(content.lower().count(word) for word in self.text_processors['negative_keywords'])
        
        total_words = len(content.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words * 100
        return max(-1.0, min(1.0, sentiment))  # Normalize to -1 ~ 1
    
    def _calculate_technical_density(self, content: str) -> float:
        """Calculate technical term density"""
        technical_terms = [
            'AI', 'artificial intelligence', 'HBM', 'DRAM', 'NAND', 'semiconductor', 'memory',
            'wafer', 'foundry', 'process', 'yield', 'capacity'
        ]
        
        technical_count = sum(content.upper().count(term.upper()) for term in technical_terms)
        total_words = len(content.split())
        
        return technical_count / max(total_words, 1) * 100

# Update existing MultimodalDataProcessor with advanced features
class MultimodalDataProcessor:
    """Multimodal data preprocessor (advanced version)"""
    
    @staticmethod
    def validate_data_consistency(data: List[Dict]) -> bool:
        """Validate data consistency"""
        
        if not data:
            logger.warning("Empty dataset")
            return False
        
        # Check required keys
        required_keys = ['text', 'numerical', 'temporal', 'metadata']
        for i, sample in enumerate(data):
            if not all(key in sample for key in required_keys):
                logger.error(f"Missing required keys in sample {i}: {required_keys}")
                return False
        
        logger.info(f"Data consistency validation passed: {len(data)} samples")
        return True
    
    @staticmethod
    def normalize_numerical_features(data: List[Dict], 
                                    method: str = "standard") -> Tuple[List[Dict], Dict[str, Any]]:
        """Normalize numerical features (supports 8 features)"""
        
        logger.info(f"Normalizing numerical features (method: {method})...")
        
        # 8 standard feature names
        standard_features = [
            'econ_revenue', 'econ_target_price', 'econ_revenue_growth',
            'text_length', 'text_word_count', 'text_sentiment', 
            'text_technical_density', 'bool_has_tables'
        ]
        
        # Extract numerical data
        all_numerical = []
        for sample in data:
            if isinstance(sample['numerical'], dict):
                values = [sample['numerical'].get(feature, 0.0) for feature in standard_features]
            else:
                values = sample['numerical'][:8]  # Limit to 8
            all_numerical.append(values)
        
        all_numerical = np.array(all_numerical)
        
        # Perform normalization
        if method == "standard":
            means = np.mean(all_numerical, axis=0)
            stds = np.std(all_numerical, axis=0)
            stds = np.where(stds == 0, 1, stds)
            
            normalized = (all_numerical - means) / stds
            
            norm_stats = {
                'method': 'standard',
                'means': means.tolist(),
                'stds': stds.tolist(),
                'feature_names': standard_features
            }
            
        elif method == "minmax":
            mins = np.min(all_numerical, axis=0)
            maxs = np.max(all_numerical, axis=0)
            ranges = maxs - mins
            ranges = np.where(ranges == 0, 1, ranges)
            
            normalized = (all_numerical - mins) / ranges
            
            norm_stats = {
                'method': 'minmax',
                'mins': mins.tolist(),
                'maxs': maxs.tolist(),
                'feature_names': standard_features
            }
            
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Update with normalized data
        normalized_data = []
        for i, sample in enumerate(data):
            new_sample = sample.copy()
            
            if isinstance(sample['numerical'], dict):
                new_sample['numerical'] = {
                    feature: float(normalized[i][j]) 
                    for j, feature in enumerate(standard_features)
                }
            else:
                new_sample['numerical'] = normalized[i].tolist()
            
            normalized_data.append(new_sample)
        
        logger.info(f"Numerical feature normalization complete: {normalized.shape}")
        
        return normalized_data, norm_stats
    
    @staticmethod
    def prepare_batch_data(samples: List[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
        """Prepare batch data"""
        
        # Text data
        texts = [sample['text'] for sample in samples]
        
        # Numerical data (ensure 8 features)
        numerical_data = []
        for sample in samples:
            if isinstance(sample['numerical'], dict):
                standard_features = [
                    'econ_revenue', 'econ_target_price', 'econ_revenue_growth',
                    'text_length', 'text_word_count', 'text_sentiment', 
                    'text_technical_density', 'bool_has_tables'
                ]
                values = [sample['numerical'].get(feature, 0.0) for feature in standard_features]
            else:
                values = sample['numerical'][:8]  # Limit to 8
            numerical_data.append(values)
        
        numerical_tensor = torch.FloatTensor(numerical_data).to(device)
        
        # Temporal data
        temporal_data = []
        for sample in samples:
            temporal_data.append(sample['temporal']['values'])
        
        temporal_tensor = torch.FloatTensor(temporal_data).to(device)
        
        return {
            'texts': texts,
            'numerical': numerical_tensor,
            'temporal': temporal_tensor,
            'metadata': [sample['metadata'] for sample in samples]
        }