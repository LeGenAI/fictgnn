"""
Embedding Generation Module

This module generates multimodal embeddings from financial text and numerical data.
It combines textual analysis, financial indicators, and temporal features to create
comprehensive representations for graph neural network analysis.

Classes:
- SemiconductorMultimodalEmbeddingPipeline: Main embedding generation pipeline
"""

from .semiconductor_multimodal_embedder import SemiconductorMultimodalEmbeddingPipeline

__all__ = ['SemiconductorMultimodalEmbeddingPipeline']