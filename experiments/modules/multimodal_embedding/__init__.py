"""
CaT-GNN Multimodal Embedding Module

This module implements multimodal fusion for financial data, combining:
- Textual data (reports, news) 
- Numerical data (financial metrics)
- Temporal data (time series information)

Key Features:
- Cross-Modal Temporal Fusion (CMTF)
- Financial text analysis
- Causality-aware embeddings
"""

from .enhanced_text_encoder import GraphSEATInspiredEncoder
from .numerical_encoder import NumericalEncoder  
from .temporal_encoder import TemporalEncoder
from .fusion_layer import CrossModalFusionLayer
from .multimodal_embedder import MultimodalEmbedder

# 고급 데이터 처리 모듈들 (업데이트됨)
from .advanced_data_processor import (
    MultimodalDataProcessor,
    AdvancedReportProcessor, 
    EconomicIndicators,
    ReportMetadata
)
from .batch_processor import BatchProcessor, CompanyQuarterlyData

__all__ = [
    'GraphSEATInspiredEncoder',
    'NumericalEncoder', 
    'TemporalEncoder',
    'CrossModalFusionLayer',
    'MultimodalEmbedder',
    'MultimodalDataProcessor',
    'AdvancedReportProcessor',
    'BatchProcessor',
    'EconomicIndicators',
    'ReportMetadata',
    'CompanyQuarterlyData'
] 