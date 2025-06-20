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
            'net_profit': [
                r'net.*profit\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'net.*income\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'net.*profit\s*(\d+(?:[,\.]\d+)*)\s*trillion',
                r'(\d+(?:[,\.]\d+)*)\s*trillion.*net.*profit'
            ],
            'growth_rates': [
                r'growth.*rate\s*(\d+(?:\.\d+)?)\s*%',
                r'increase.*rate\s*(\d+(?:\.\d+)?)\s*%',
                r'YoY\s*(\d+(?:\.\d+)?)\s*%',
                r'year.*over.*year.*(\d+(?:\.\d+)?)\s*%.*increase'
            ],
            'target_price': [
                r'target.*price\s*(\d+(?:[,\.]\d+)*)',
                r'price.*target\s*(\d+(?:[,\.]\d+)*)',
                r'TP\s*(\d+(?:[,\.]\d+)*)',
                r'target.*?(\d+(?:[,\.]\d+)*)'
            ],
            'per_pbr': [
                r'PER\s*(\d+(?:\.\d+)?)',
                r'PBR\s*(\d+(?:\.\d+)?)',
                r'P/E\s*(\d+(?:\.\d+)?)',
                r'P/B\s*(\d+(?:\.\d+)?)'
            ],
            'market_share': [
                r'market.*share\s*(\d+(?:\.\d+)?)\s*%',
                r'share.*(\d+(?:\.\d+)?)\s*%',
                r'market.*share\s*(\d+(?:\.\d+)?)\s*%'
            ],
            'hbm_indicators': [
                r'HBM.*?(\d+(?:\.\d+)?)\s*%',
                r'High.*Bandwidth.*(\d+(?:\.\d+)?)',
                r'AI.*server.*?(\d+(?:\.\d+)?)\s*%'
            ]
        }
    
    def _build_text_processors(self) -> Dict[str, Any]:
        """Build text processing tools"""
        return {
            'summary_keywords': [
                'summary', 'key', 'conclusion', 'comprehensive', 'summary', 'key points'
            ],
            'positive_keywords': [
                'growth', 'increase', 'improvement', 'favorable', 'expansion', 'strengthen', 'rise', 'expectation',
                'good', 'strong', 'excellent', 'success', 'leap', 'recovery'
            ],
            'negative_keywords': [
                'decrease', 'decline', 'sluggish', 'deterioration', 'concern', 'risk', 'downward', 'adjustment',
                'difficulty', 'weakness', 'burden', 'slowdown'
            ],
            'uncertainty_keywords': [
                'uncertain', 'volatile', 'mixed', 'watch', 'monitor', 'wait-and-see', 'standby'
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
            
            # Operating profit extraction
            for pattern in self.financial_patterns['operating_profit']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.operating_profit = self._parse_financial_number(matches[0])
                    break
            
            # Net profit extraction
            for pattern in self.financial_patterns['net_profit']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.net_profit = self._parse_financial_number(matches[0])
                    break
            
            # Growth rate extraction
            growth_matches = []
            for pattern in self.financial_patterns['growth_rates']:
                growth_matches.extend(re.findall(pattern, content, re.IGNORECASE))
            
            if growth_matches:
                # First growth rate as revenue growth rate
                indicators.revenue_growth = float(growth_matches[0])
                if len(growth_matches) > 1:
                    indicators.profit_growth = float(growth_matches[1])
            
            # Target price extraction
            for pattern in self.financial_patterns['target_price']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.target_price = self._parse_financial_number(matches[0])
                    break
            
            # PER/PBR extraction
            per_pbr_matches = []
            for pattern in self.financial_patterns['per_pbr']:
                per_pbr_matches.extend(re.findall(pattern, content, re.IGNORECASE))
            
            if per_pbr_matches:
                if 'PER' in content.upper() or 'P/E' in content.upper():
                    indicators.per = float(per_pbr_matches[0])
                if 'PBR' in content.upper() or 'P/B' in content.upper():
                    pbr_idx = 1 if len(per_pbr_matches) > 1 else 0
                    indicators.pbr = float(per_pbr_matches[pbr_idx])
            
            # Market share extraction
            for pattern in self.financial_patterns['market_share']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    indicators.market_share = float(matches[0])
                    break
            
            # HBM/AI related indicator extraction
            for pattern in self.financial_patterns['hbm_indicators']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if 'HBM' in content.upper():
                        indicators.hbm_revenue_ratio = float(matches[0])
                    elif 'AI' in content.upper() and 'server' in content:
                        indicators.ai_server_exposure = float(matches[0])
                    break
                    
        except Exception as e:
            logger.warning(f"Error in economic indicator extraction: {e}")
        
        return indicators
    
    def _parse_financial_number(self, number_str: str) -> float:
        """Parse financial numbers"""
        try:
            # Remove commas and convert to float
            clean_number = number_str.replace(',', '').replace('.', '')
            return float(clean_number)
        except:
            return 0.0
    
    def classify_report_type(self, title: str, content: str) -> str:
        """Classify report type"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Earnings announcement related
        if any(keyword in title_lower for keyword in ['earnings', 'review', 'quarterly', 'earnings']):
            return 'earnings'
        
        # New coverage
        if any(keyword in title_lower for keyword in ['initiation', 'new', 'coverage']):
            return 'initiation'
        
        # Update/outlook
        if any(keyword in title_lower for keyword in ['update', 'outlook', 'preview', 'target']):
            return 'update'
        
        # Sector report
        if any(keyword in title_lower for keyword in ['sector', 'industry', 'sector']):
            return 'sector'
        
        return 'general'
    
    def extract_text_features(self, content: str) -> Dict[str, Any]:
        """Extract text features"""
        features = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()]),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'has_tables': '|' in content and content.count('|') > 10,
            'has_charts': any(word in content.lower() for word in ['chart', 'chart', 'graph', 'graph']),
            'sentiment_score': self._calculate_sentiment(content),
            'technical_density': self._calculate_technical_density(content),
            'financial_density': self._calculate_financial_density(content)
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
            'wafer', 'foundry', 'process', 'yield', 'capacity', 'migration'
        ]
        
        technical_count = sum(content.upper().count(term.upper()) for term in technical_terms)
        total_words = len(content.split())
        
        return technical_count / max(total_words, 1) * 100
    
    def _calculate_financial_density(self, content: str) -> float:
        """Calculate financial term density"""
        financial_terms = [
            'revenue', 'profit', 'income', 'cost', 'margin', 'growth rate', 'valuation',
            'PER', 'PBR', 'ROE', 'ROA', 'target price', 'investment', 'return'
        ]
        
        financial_count = sum(content.count(term) for term in financial_terms)
        total_words = len(content.split())
        
        return financial_count / max(total_words, 1) * 100
    
    def handle_duplicate_reports(self, reports_by_quarter: Dict[str, List[ReportMetadata]], 
                                strategy: str = "weighted_average") -> Dict[str, ReportMetadata]:
        """Handle duplicate reports within the same quarter"""
        
        logger.info(f"Processing duplicate reports (strategy: {strategy})...")
        processed_reports = {}
        
        for quarter, reports in reports_by_quarter.items():
            if len(reports) == 1:
                # Use single report as is
                processed_reports[quarter] = reports[0]
            else:
                # Handle duplicate reports
                if strategy == "weighted_average":
                    processed_reports[quarter] = self._weighted_average_reports(reports)
                elif strategy == "latest_only":
                    processed_reports[quarter] = max(reports, key=lambda x: x.date)
                elif strategy == "best_quality":
                    processed_reports[quarter] = self._select_best_quality_report(reports)
                elif strategy == "consensus":
                    processed_reports[quarter] = self._create_consensus_report(reports)
                else:
                    processed_reports[quarter] = reports[0]  # Use first only
        
        logger.info(f"   Processing complete for {len(processed_reports)} quarters")
        return processed_reports
    
    def _weighted_average_reports(self, reports: List[ReportMetadata]) -> ReportMetadata:
        """Combine reports using weighted average"""
        
        # Calculate weights (recency + quality)
        weights = []
        for report in reports:
            recency_weight = 1.0  # Higher weight for more recent (implementable)
            quality_weight = self._calculate_report_quality(report)
            weights.append(recency_weight * quality_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Base report (highest weight)
        base_report = reports[weights.index(max(weights))]
        
        # Weighted average of economic indicators
        averaged_indicators = self._average_economic_indicators(reports, weights)
        
        # Create new metadata
        processed_report = ReportMetadata(
            filename=f"consensus_{base_report.quarter}_{base_report.company_name}",
            date=base_report.date,
            company_name=base_report.company_name,
            stock_code=base_report.stock_code,
            securities_firm="Consensus",
            report_title=f"Consensus Report - {base_report.company_name}",
            year=base_report.year,
            quarter=base_report.quarter,
            report_type="consensus",
            content_length=int(np.mean([r.content_length for r in reports])),
            summary_length=int(np.mean([r.summary_length for r in reports])),
            has_financial_tables=any(r.has_financial_tables for r in reports),
            confidence_level="high",
            economic_indicators=averaged_indicators,
            text_features=self._average_text_features(reports, weights),
            numerical_features=self._average_numerical_features(reports, weights)
        )
        
        return processed_report
    
    def _calculate_report_quality(self, report: ReportMetadata) -> float:
        """Calculate report quality score"""
        quality_score = 0.0
        
        # Content richness
        if report.content_length > 10000:
            quality_score += 0.3
        elif report.content_length > 5000:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Financial table presence
        if report.has_financial_tables:
            quality_score += 0.2
        
        # Economic indicator completeness
        indicator_count = sum(1 for field in report.economic_indicators.__dict__.values() if field is not None)
        quality_score += (indicator_count / 15) * 0.3  # Maximum 15 indicators
        
        # Securities firm credibility (simple heuristic)
        reputable_firms = ['Shinhan Investment', 'SK Securities', 'Daishin Securities', 'Kiwoom Securities', 'Hana Securities']
        if report.securities_firm in reputable_firms:
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _average_economic_indicators(self, reports: List[ReportMetadata], 
                                   weights: List[float]) -> EconomicIndicators:
        """Calculate weighted average of economic indicators"""
        averaged = EconomicIndicators()
        
        # Calculate weighted average for all indicators
        for field_name in averaged.__dataclass_fields__.keys():
            values = []
            valid_weights = []
            
            for i, report in enumerate(reports):
                value = getattr(report.economic_indicators, field_name)
                if value is not None:
                    values.append(value)
                    valid_weights.append(weights[i])
            
            if values:
                # Re-normalize weights
                total_weight = sum(valid_weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in valid_weights]
                    averaged_value = sum(v * w for v, w in zip(values, normalized_weights))
                    setattr(averaged, field_name, averaged_value)
        
        return averaged
    
    def _average_text_features(self, reports: List[ReportMetadata], 
                             weights: List[float]) -> Dict[str, Any]:
        """Weighted average of text features"""
        if not reports:
            return {}
        
        # Get base features from first report
        base_features = reports[0].text_features.copy()
        
        # Calculate average for numerical features only
        numeric_keys = ['length', 'word_count', 'sentence_count', 'paragraph_count', 
                       'sentiment_score', 'technical_density', 'financial_density']
        
        for key in numeric_keys:
            if key in base_features:
                values = [r.text_features.get(key, 0) for r in reports]
                base_features[key] = sum(v * w for v, w in zip(values, weights))
        
        # Boolean features use OR operation
        bool_keys = ['has_tables', 'has_charts']
        for key in bool_keys:
            if key in base_features:
                base_features[key] = any(r.text_features.get(key, False) for r in reports)
        
        return base_features
    
    def _average_numerical_features(self, reports: List[ReportMetadata], 
                                  weights: List[float]) -> Dict[str, float]:
        """Weighted average of numerical features"""
        if not reports:
            return {}
        
        averaged_features = {}
        all_keys = set()
        for report in reports:
            all_keys.update(report.numerical_features.keys())
        
        for key in all_keys:
            values = []
            valid_weights = []
            
            for i, report in enumerate(reports):
                if key in report.numerical_features:
                    values.append(report.numerical_features[key])
                    valid_weights.append(weights[i])
            
            if values:
                total_weight = sum(valid_weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in valid_weights]
                    averaged_features[key] = sum(v * w for v, w in zip(values, normalized_weights))
        
        return averaged_features
    
    def _select_best_quality_report(self, reports: List[ReportMetadata]) -> ReportMetadata:
        """Select the highest quality report"""
        best_report = reports[0]
        best_quality = self._calculate_report_quality(best_report)
        
        for report in reports[1:]:
            quality = self._calculate_report_quality(report)
            if quality > best_quality:
                best_quality = quality
                best_report = report
        
        return best_report
    
    def _create_consensus_report(self, reports: List[ReportMetadata]) -> ReportMetadata:
        """Create consensus report"""
        # Similar to weighted average but with equal weights for all reports
        equal_weights = [1.0 / len(reports)] * len(reports)
        return self._weighted_average_reports(reports)

def main():
    """Main execution function"""
    
    logger.info("Phase 2: Advanced Data Processing Started!")
    
    # Data directory configuration
    data_dir = r"C:\Users\Edward\Desktop\economist\final\catgnn_experiment\data\Semiconductor_Report_MD_20241025"
    
    # Initialize advanced processor
    processor = AdvancedReportProcessor(data_dir)
    
    # Test with sample files
    sample_files = list(Path(data_dir).glob("2024*.md"))[:5]  # 5 files from 2024
    
    logger.info(f"Testing advanced processing with {len(sample_files)} sample files...")
    
    processed_reports = []
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse filename
            filename = file_path.name
            match = re.match(r'(\d{8})_(.+?)\((\d{6})\)_(.+?)_(.+?)\.md$', filename)
            if not match:
                continue
            
            date_str, company_name, stock_code, title, securities_firm = match.groups()
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            
            # Advanced processing
            economic_indicators = processor.extract_economic_indicators(content)
            text_features = processor.extract_text_features(content)
            report_type = processor.classify_report_type(title, content)
            
            # Create metadata
            metadata = ReportMetadata(
                filename=filename,
                date=date_obj,
                company_name=company_name,
                stock_code=stock_code,
                securities_firm=securities_firm,
                report_title=title,
                year=date_obj.year,
                quarter=f"{date_obj.year}Q{((date_obj.month - 1) // 3) + 1}",
                report_type=report_type,
                content_length=len(content),
                summary_length=len(content.split('\n')[0]) if content else 0,
                has_financial_tables='|' in content and content.count('|') > 10,
                confidence_level="high",
                economic_indicators=economic_indicators,
                text_features=text_features,
                numerical_features={}
            )
            
            processed_reports.append(metadata)
            
            logger.info(f"   {filename[:50]}... processing complete")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    
    # Results summary
    logger.info(f"\nAdvanced processing results summary:")
    logger.info(f"   - Processed reports: {len(processed_reports)}")
    
    if processed_reports:
        # Economic indicator statistics
        revenue_values = [r.economic_indicators.revenue for r in processed_reports if r.economic_indicators.revenue]
        target_prices = [r.economic_indicators.target_price for r in processed_reports if r.economic_indicators.target_price]
        
        logger.info(f"   - Revenue extracted: {len(revenue_values)}")
        logger.info(f"   - Target price extracted: {len(target_prices)}")
        
        # Text feature statistics
        avg_sentiment = np.mean([r.text_features['sentiment_score'] for r in processed_reports])
        avg_technical_density = np.mean([r.text_features['technical_density'] for r in processed_reports])
        
        logger.info(f"   - Average sentiment score: {avg_sentiment:.3f}")
        logger.info(f"   - Average technical density: {avg_technical_density:.2f}%")
        
        # Report type distribution
        type_counts = Counter([r.report_type for r in processed_reports])
        logger.info(f"   - Report type distribution: {dict(type_counts)}")
    
    logger.info(f"\nPhase 2 advanced processing complete!")
    logger.info(f"Next step will proceed with batch processing for the entire dataset.")
    
    return processed_reports

if __name__ == "__main__":
    main()