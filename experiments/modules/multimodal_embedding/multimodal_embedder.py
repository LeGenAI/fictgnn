"""
Multimodal Embedder for CaT-GNN

Integrates all multimodal components:
- Text Encoder
- Numerical Encoder
- Temporal Encoder
- Cross-Modal Fusion Layer

Provides a unified interface for generating multimodal embeddings
for financial data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json

from .enhanced_text_encoder import GraphSEATInspiredEncoder
from .numerical_encoder import NumericalEncoder
from .temporal_encoder import TemporalEncoder
from .fusion_layer import CrossModalFusionLayer

logger = logging.getLogger(__name__)


class MultimodalEmbedder(nn.Module):
    """
    Complete multimodal embedding system for financial data
    
    Features:
    - Unified interface for all modalities
    - Flexible configuration
    - Batch processing
    - Interpretability tools
    - Model saving/loading
    """
    
    def __init__(
        self,
        # Text encoder config
        text_model_name: str = "all-mpnet-base-v2",
        text_hidden_dim: int = 768,
        enable_financial_features: bool = True,
        use_openai: bool = False,  # Enable OpenAI text embedding
        
        # Numerical encoder config
        numerical_input_dim: int = 42,
        numerical_hidden_dim: int = 128,
        numerical_output_dim: int = 256,
        use_numerical_attention: bool = True,
        
        # Temporal encoder config
        temporal_dim: int = 15,
        temporal_hidden_dim: int = 128,
        temporal_output_dim: int = 256,
        use_lstm: bool = True,
        # Temporal scaling factor
        temporal_scale: float = 1.0,
        
        # Fusion config
        fusion_output_dim: int = 512,
        fusion_strategy: str = "attention",
        use_causality: bool = True,
        use_gating: bool = True,
        
        # General config
        device: str = "auto",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Store configuration
        self.config = {
            'text_model_name': text_model_name,
            'text_hidden_dim': text_hidden_dim,
            'enable_financial_features': enable_financial_features,
            'use_openai': use_openai,  # OpenAI configuration
            'numerical_input_dim': numerical_input_dim,
            'numerical_hidden_dim': numerical_hidden_dim,
            'numerical_output_dim': numerical_output_dim,
            'use_numerical_attention': use_numerical_attention,
            'temporal_dim': temporal_dim,
            'temporal_hidden_dim': temporal_hidden_dim,
            'temporal_output_dim': temporal_output_dim,
            'use_lstm': use_lstm,
            'fusion_output_dim': fusion_output_dim,
            'fusion_strategy': fusion_strategy,
            'use_causality': use_causality,
            'use_gating': use_gating,
            'dropout_rate': dropout_rate,
            'temporal_scale': temporal_scale
        }
        
        # Initialize encoders
        self.text_encoder = GraphSEATInspiredEncoder(
            use_openai=use_openai,  # Enable OpenAI text embedding
            fallback_model=text_model_name,
            hidden_dim=text_hidden_dim,
            device=self.device
        )
        
        self.numerical_encoder = NumericalEncoder(
            input_dim=numerical_input_dim,
            hidden_dim=numerical_hidden_dim,
            output_dim=numerical_output_dim,
            use_attention=use_numerical_attention,
            dropout_rate=dropout_rate,
            device=self.device
        ).to(self.device)
        
        self.temporal_encoder = TemporalEncoder(
            temporal_dim=temporal_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_output_dim,
            use_lstm=use_lstm,
            dropout_rate=dropout_rate,
            device=self.device
        ).to(self.device)
        
        # Initialize fusion layer (will be created after text encoder is loaded)
        self.fusion_layer = None
        self.fusion_config = {
            'numerical_dim': numerical_output_dim,
            'temporal_dim': temporal_output_dim,
            'output_dim': fusion_output_dim,
            'fusion_strategy': fusion_strategy,
            'use_causality': use_causality,
            'use_gating': use_gating,
            'dropout_rate': dropout_rate
        }
        
        # State tracking
        self.models_loaded = False
        self.output_dim = fusion_output_dim
        self.temporal_scale = temporal_scale
        
        logger.info("MultimodalEmbedder initialization complete")
        logger.info(f"Device: {self.device}")
        logger.info(f"Fusion strategy: {fusion_strategy}")
        logger.info(f"Output dimension: {fusion_output_dim}")
    
    def _setup_device(self, device: str) -> str:
        """Configure computational device for model execution."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_models(self):
        """Load all encoder models and initialize fusion layer."""
        try:
            logger.info("Loading text encoder model...")
            if not self.text_encoder.load_model():
                raise RuntimeError("Text model loading failed")
            
            # Get text encoder dimension
            text_dim = self.text_encoder.get_embedding_dimension()
            
            # Initialize fusion layer with correct dimensions
            logger.info("Initializing fusion layer...")
            self.fusion_layer = CrossModalFusionLayer(
                text_dim=text_dim,
                **self.fusion_config
            ).to(self.device)
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def forward(
        self,
        text_data: List[str],
        numerical_data: List[Dict[str, Any]],
        temporal_data: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all encoders and fusion
        
        Args:
            text_data: List of text strings
            numerical_data: List of numerical feature dictionaries
            temporal_data: List of temporal data dictionaries
            
        Returns:
            Dict containing:
                - embeddings: Final multimodal embeddings
                - text_embeddings: Text-only embeddings
                - numerical_embeddings: Numerical-only embeddings
                - temporal_embeddings: Temporal-only embeddings
                - fusion_outputs: Detailed fusion outputs
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Please call load_models() first.")
        
        batch_size = len(text_data)
        
        # Ensure all inputs have same batch size
        assert len(numerical_data) == batch_size, "Numerical data batch size mismatch"
        assert len(temporal_data) == batch_size, "Temporal data batch size mismatch"
        
        # Encode each modality
        logger.debug("Encoding text data...")
        text_embeddings = self.text_encoder.encode_texts(text_data)
        
        logger.debug("Encoding numerical data...")
        numerical_embeddings, anomaly_scores = self.numerical_encoder.encode_batch(numerical_data)
        
        logger.debug("Encoding temporal data...")
        temporal_embeddings = self.temporal_encoder(temporal_data)
        
        # -------- temporal scaling boost --------
        if self.temporal_scale != 1.0:
            temporal_embeddings = temporal_embeddings * self.temporal_scale
        
        # Ensure all embeddings are on the same device
        text_embeddings = text_embeddings.to(self.device)
        numerical_embeddings = numerical_embeddings.to(self.device)
        temporal_embeddings = temporal_embeddings.to(self.device)
        
        # Fusion
        logger.debug("Performing multimodal fusion...")
        fusion_outputs = self.fusion_layer(
            text_embeddings=text_embeddings,
            numerical_embeddings=numerical_embeddings,
            temporal_embeddings=temporal_embeddings
        )
        
        # Compile results
        results = {
            'embeddings': fusion_outputs['fused_embeddings'],
            'text_embeddings': text_embeddings,
            'numerical_embeddings': numerical_embeddings,
            'temporal_embeddings': temporal_embeddings,
            'fusion_outputs': fusion_outputs,
            'anomaly_scores': anomaly_scores
        }
        
        return results
    
    def encode_single(
        self,
        text: str,
        numerical_data: Dict[str, Any],
        temporal_data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Encode a single data point"""
        return self.forward([text], [numerical_data], [temporal_data])
    
    def encode_batch(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch data
        
        Args:
            batch_data: List of dicts with keys 'text', 'numerical', 'temporal'
        """
        text_data = [item['text'] for item in batch_data]
        numerical_data = [item['numerical'] for item in batch_data]
        temporal_data = [item['temporal'] for item in batch_data]
        
        return self.forward(text_data, numerical_data, temporal_data)
    
    def get_attention_patterns(
        self,
        text_data: List[str],
        numerical_data: List[Dict[str, Any]],
        temporal_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze attention patterns (interpretability)"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        with torch.no_grad():
            # Encode modalities
            text_embeddings = self.text_encoder.encode_texts(text_data)
            numerical_embeddings, _ = self.numerical_encoder.encode_batch(numerical_data)
            temporal_embeddings = self.temporal_encoder(temporal_data)
            
            # Get attention patterns
            patterns = self.fusion_layer.get_attention_patterns(
                text_embeddings, numerical_embeddings, temporal_embeddings
            )
            
            return patterns
    
    def get_feature_importance(
        self,
        text_data: List[str],
        numerical_data: List[Dict[str, Any]],
        temporal_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feature importance"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        importance_scores = {}
        
        with torch.no_grad():
            # Numerical feature importance
            numerical_tensor = self.numerical_encoder.process_raw_data(numerical_data)
            if self.numerical_encoder.use_attention:
                importance_scores['numerical'] = self.numerical_encoder.get_feature_importance(numerical_tensor)
            
            # Attention patterns from fusion
            attention_patterns = self.get_attention_patterns(text_data, numerical_data, temporal_data)
            importance_scores['fusion_attention'] = attention_patterns
        
        return importance_scores
    
    def detect_anomalies(
        self,
        numerical_data: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> List[bool]:
        """Detect anomalies in numerical data"""
        return self.numerical_encoder.detect_anomalies(numerical_data, threshold)
    
    def save_model(self, save_path: str):
        """Save model"""
        if not self.models_loaded:
            raise RuntimeError("No loaded models to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model states
        torch.save({
            'numerical_encoder': self.numerical_encoder.state_dict(),
            'temporal_encoder': self.temporal_encoder.state_dict(),
            'fusion_layer': self.fusion_layer.state_dict() if self.fusion_layer else None,
            'config': self.config
        }, save_path / 'multimodal_embedder.pt')
        
        # Save configuration
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved successfully: {save_path}")
    
    def load_model(self, load_path: str):
        """Load model"""
        load_path = Path(load_path)
        
        # Load configuration
        with open(load_path / 'config.json', 'r') as f:
            saved_config = json.load(f)
        
        # Load model states
        checkpoint = torch.load(load_path / 'multimodal_embedder.pt', map_location=self.device)
        
        # Load text encoder separately
        if not self.text_encoder.load_model():
            raise RuntimeError("Text model loading failed")
        
        # Load other components
        self.numerical_encoder.load_state_dict(checkpoint['numerical_encoder'])
        self.temporal_encoder.load_state_dict(checkpoint['temporal_encoder'])
        
        if checkpoint['fusion_layer'] is not None:
            # Initialize fusion layer if not already done
            if self.fusion_layer is None:
                text_dim = self.text_encoder.get_embedding_dimension()
                self.fusion_layer = CrossModalFusionLayer(
                    text_dim=text_dim,
                    **self.fusion_config
                ).to(self.device)
            
            self.fusion_layer.load_state_dict(checkpoint['fusion_layer'])
        
        self.models_loaded = True
        logger.info(f"Model loaded successfully: {load_path}")
    
    def get_embedding_dimension(self) -> int:
        """Return output embedding dimension"""
        return self.output_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        info = {
            'config': self.config,
            'models_loaded': self.models_loaded,
            'device': self.device,
            'output_dimension': self.output_dim
        }
        
        if self.models_loaded:
            info['text_embedding_dim'] = self.text_encoder.get_embedding_dimension()
            info['numerical_embedding_dim'] = self.numerical_encoder.get_embedding_dimension()
            info['temporal_embedding_dim'] = self.temporal_encoder.get_embedding_dimension()
        
        return info
    
    def train_mode(self):
        """Set training mode"""
        super().train()
        if self.models_loaded:
            self.text_encoder.train()
            self.numerical_encoder.train()
            self.temporal_encoder.train()
            if self.fusion_layer:
                self.fusion_layer.train()
    
    def eval_mode(self):
        """Set evaluation mode"""
        super().eval()
        if self.models_loaded:
            self.text_encoder.eval()
            self.numerical_encoder.eval()
            self.temporal_encoder.eval()
            if self.fusion_layer:
                self.fusion_layer.eval()