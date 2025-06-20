#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor for Semiconductor Report Dataset - Module Version
Modularized batch processor for handling quarterly duplicate reports and building final dataset
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from advanced_data_processor
try:
    from .advanced_data_processor import EconomicIndicators, ReportMetadata
except ImportError:
    # Fallback for standalone usage
    from advanced_data_processor import EconomicIndicators, ReportMetadata

class CompanyQuarterlyData:
    """Company quarterly data structure"""
    
    def __init__(self, company_name: str, stock_code: str, quarter: str):
        self.company_name = company_name
        self.stock_code = stock_code
        self.quarter = quarter
        self.reports = []
        self.final_report = None
        self.report_count = 0
        
    def add_report(self, report: ReportMetadata):
        self.reports.append(report)
        self.report_count += 1

class BatchProcessor:
    """Modularized batch processor"""
    
    def __init__(self, data_dir: str, output_dir: str = "processed_dataset"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data structures
        self.quarterly_data = defaultdict(lambda: defaultdict(CompanyQuarterlyData))
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'quarters_with_duplicates': 0
        }
        
        # Pattern definitions
        self.financial_patterns = {
            'revenue': [r'revenue\s*(\d+(?:[,\.]\d+)*)\s*trillion', r'sales\s*(\d+(?:[,\.]\d+)*)\s*trillion'],
            'target_price': [r'target.*price\s*(\d+(?:[,\.]\d+)*)', r'price.*target\s*(\d+(?:[,\.]\d+)*)'],
            'growth_rates': [r'growth.*rate\s*(\d+(?:\.\d+)?)\s*%', r'increase.*rate\s*(\d+(?:\.\d+)?)\s*%']
        }
        
        self.positive_keywords = ['growth', 'increase', 'improvement', 'favorable', 'expansion', 'expectation', 'good']
        self.negative_keywords = ['decrease', 'decline', 'sluggish', 'deterioration', 'concern', 'difficulty']
    
    def handle_quarterly_duplicates(self, strategy: str = "weighted_average"):
        """Handle quarterly duplicate reports"""
        
        logger.info(f"Processing quarterly duplicate reports (strategy: {strategy})...")
        
        duplicate_quarters = 0
        processed_quarters = 0
        
        for company_name, quarters in self.quarterly_data.items():
            for quarter, quarter_data in quarters.items():
                if quarter_data.report_count > 1:
                    duplicate_quarters += 1
                    
                    # Apply duplicate handling strategy
                    if strategy == "weighted_average":
                        final_report = self._average_reports(quarter_data.reports)
                    elif strategy == "latest_only":
                        final_report = max(quarter_data.reports, key=lambda x: x.date)
                    else:
                        final_report = quarter_data.reports[0]
                    
                    quarter_data.final_report = final_report
                else:
                    quarter_data.final_report = quarter_data.reports[0]
                
                processed_quarters += 1
        
        self.processing_stats['quarters_with_duplicates'] = duplicate_quarters
        
        logger.info(f"   Duplicate processing complete:")
        logger.info(f"      - Duplicate quarters: {duplicate_quarters}")
        logger.info(f"      - Total processed quarters: {processed_quarters}")
    
    def _average_reports(self, reports: List[ReportMetadata]) -> ReportMetadata:
        """Average reports"""
        if len(reports) == 1:
            return reports[0]
        
        # Base report (latest)
        base_report = max(reports, key=lambda x: x.date)
        
        # Average economic indicators
        avg_indicators = EconomicIndicators()
        
        revenues = [r.economic_indicators.revenue for r in reports if r.economic_indicators.revenue]
        if revenues:
            avg_indicators.revenue = np.mean(revenues)
        
        target_prices = [r.economic_indicators.target_price for r in reports if r.economic_indicators.target_price]
        if target_prices:
            avg_indicators.target_price = np.mean(target_prices)
        
        growth_rates = [r.economic_indicators.revenue_growth for r in reports if r.economic_indicators.revenue_growth]
        if growth_rates:
            avg_indicators.revenue_growth = np.mean(growth_rates)
        
        # Average text features
        avg_text_features = {}
        numeric_keys = ['length', 'word_count', 'sentence_count', 'sentiment_score', 'technical_density']
        
        for key in numeric_keys:
            values = [r.text_features.get(key, 0) for r in reports]
            avg_text_features[key] = np.mean(values) if values else 0.0
        
        avg_text_features['has_tables'] = any(r.text_features.get('has_tables', False) for r in reports)
        
        # Average numerical features
        avg_numerical_features = {}
        all_keys = set()
        for report in reports:
            all_keys.update(report.numerical_features.keys())
        
        for key in all_keys:
            values = [r.numerical_features.get(key, 0) for r in reports]
            avg_numerical_features[key] = np.mean(values)
        
        # Create new metadata
        averaged_report = ReportMetadata(
            filename=f"consensus_{base_report.quarter}_{base_report.company_name}",
            date=base_report.date,
            company_name=base_report.company_name,
            stock_code=base_report.stock_code,
            securities_firm="Consensus",
            report_title=f"Consensus - {base_report.company_name}",
            year=base_report.year,
            quarter=base_report.quarter,
            report_type="consensus",
            content_length=int(np.mean([r.content_length for r in reports])),
            summary_length=int(np.mean([r.summary_length for r in reports])),
            has_financial_tables=any(r.has_financial_tables for r in reports),
            confidence_level="high",
            economic_indicators=avg_indicators,
            text_features=avg_text_features,
            numerical_features=avg_numerical_features
        )
        
        return averaged_report
    
    def build_final_dataset(self) -> Dict[str, Any]:
        """Build final dataset"""
        
        logger.info(f"Building final dataset...")
        
        dataset = {
            'companies': {},
            'metadata': {
                'company_names': [],
                'quarters': [],
                'feature_names': [
                    'econ_revenue', 'econ_target_price', 'econ_revenue_growth',
                    'text_length', 'text_word_count', 'text_sentiment', 
                    'text_technical_density', 'bool_has_tables'
                ],
                'processing_stats': self.processing_stats,
                'creation_date': datetime.now().isoformat()
            }
        }
        
        # Organize company data
        company_names = sorted(self.quarterly_data.keys())
        all_quarters = set()
        
        for company_name in company_names:
            company_data = {
                'stock_code': '',
                'quarters': {},
                'total_reports': 0
            }
            
            for quarter, quarter_data in self.quarterly_data[company_name].items():
                if quarter_data.final_report:
                    all_quarters.add(quarter)
                    
                    # Convert to serializable format
                    quarter_info = {
                        'date': quarter_data.final_report.date.isoformat(),
                        'securities_firm': quarter_data.final_report.securities_firm,
                        'report_type': quarter_data.final_report.report_type,
                        'original_report_count': quarter_data.report_count,
                        'economic_indicators': {
                            'revenue': quarter_data.final_report.economic_indicators.revenue,
                            'target_price': quarter_data.final_report.economic_indicators.target_price,
                            'revenue_growth': quarter_data.final_report.economic_indicators.revenue_growth
                        },
                        'text_features': quarter_data.final_report.text_features,
                        'numerical_features': quarter_data.final_report.numerical_features
                    }
                    
                    company_data['quarters'][quarter] = quarter_info
                    company_data['total_reports'] += quarter_data.report_count
                    company_data['stock_code'] = quarter_data.final_report.stock_code
            
            dataset['companies'][company_name] = company_data
        
        # Update metadata
        dataset['metadata']['company_names'] = company_names
        dataset['metadata']['quarters'] = sorted(list(all_quarters))
        
        logger.info(f"Final dataset construction complete:")
        logger.info(f"   - Number of companies: {len(company_names)}")
        logger.info(f"   - Number of quarters: {len(all_quarters)}")
        logger.info(f"   - Feature dimensions: 8")
        
        return dataset