# -*- coding: utf-8 -*-
"""
Data Loader Module

Provides functionality for loading embeddings, metadata, and financial statement data
for semiconductor industry analysis.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_data():
    """
    Load multimodal data for causality analysis.
    
    Returns:
        tuple: (embeddings, metadata, financial_data)
            - embeddings: numpy array of preprocessed embeddings
            - metadata: dictionary containing embedding metadata
            - financial_data: pandas DataFrame with financial metrics
    """
    logger.info("Loading multimodal dataset...")
    
    # Set base path relative to current working directory
    base_path = Path.cwd().parent
    
    # Load embeddings
    embedding_path = base_path / "catgnn_experiment/semiconductor_multimodal_embeddings_with_financial/semiconductor_embeddings/embeddings.npy"
    embeddings = np.load(embedding_path)
    logger.info(f"Embeddings loaded: {embeddings.shape}")
    
    # Load metadata
    metadata_path = base_path / "catgnn_experiment/semiconductor_multimodal_embeddings_with_financial/semiconductor_embeddings/metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logger.info(f"Metadata loaded: {len(metadata)} entries")
    
    # Load financial data
    financial_path = base_path / "complete_semiconductor_dataset_20250619_221652.csv"
    financial_data = pd.read_csv(financial_path)
    logger.info(f"Financial data loaded: {financial_data.shape}")
    
    return embeddings, metadata, financial_data