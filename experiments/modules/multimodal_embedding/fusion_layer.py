"""
Cross-Modal Fusion Layer for Multimodal Financial Data

Advanced fusion strategies for combining text, numerical, and temporal embeddings.
Supports multiple fusion approaches including attention-based, gating-based, and causal fusion.

Features:
- Cross-modal attention mechanisms
- Causal relationship modeling
- Gating networks for adaptive fusion
- Interpretability tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import math

logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different modalities"""
    
    def __init__(
        self,
        text_dim: int,
        numerical_dim: int,
        temporal_dim: int,
        attention_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0
        
        # Projection layers for each modality
        self.text_projection = nn.Linear(text_dim, attention_dim)
        self.numerical_projection = nn.Linear(numerical_dim, attention_dim)
        self.temporal_projection = nn.Linear(temporal_dim, attention_dim)
        
        # Multi-head attention components
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(attention_dim, attention_dim)
        self.layer_norm = nn.LayerNorm(attention_dim)
        
    def forward(
        self,
        text_emb: torch.Tensor,
        numerical_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for cross-modal attention
        
        Args:
            text_emb: Text embeddings [batch_size, text_dim]
            numerical_emb: Numerical embeddings [batch_size, numerical_dim]
            temporal_emb: Temporal embeddings [batch_size, temporal_dim]
            
        Returns:
            fused_embeddings: Fused embeddings [batch_size, attention_dim]
            attention_weights: Attention weights for interpretability
        """
        batch_size = text_emb.size(0)
        
        # Project all modalities to same dimension
        text_proj = self.text_projection(text_emb)
        numerical_proj = self.numerical_projection(numerical_emb)
        temporal_proj = self.temporal_projection(temporal_emb)
        
        # Stack modalities as sequence [batch_size, 3, attention_dim]
        modality_sequence = torch.stack([text_proj, numerical_proj, temporal_proj], dim=1)
        
        # Multi-head attention
        Q = self.query(modality_sequence)
        K = self.key(modality_sequence)
        V = self.value(modality_sequence)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, 3, self.attention_dim
        )
        
        # Output linear transformation
        output = self.output_linear(attended)
        
        # Global pooling to get final fused embedding
        fused_embeddings = output.mean(dim=1)  # [batch_size, attention_dim]
        
        # Residual connection and layer norm
        fused_embeddings = self.layer_norm(fused_embeddings + modality_sequence.mean(dim=1))
        
        # Extract attention patterns for interpretability
        attention_patterns = {
            'cross_modal_attention': attention_weights.mean(dim=1).detach(),  # Average across heads
            'text_attention': attention_weights[:, :, 0, :].mean(dim=1).detach(),
            'numerical_attention': attention_weights[:, :, 1, :].mean(dim=1).detach(),
            'temporal_attention': attention_weights[:, :, 2, :].mean(dim=1).detach()
        }
        
        return fused_embeddings, attention_patterns


class CausalFusionModule(nn.Module):
    """Causal relationship modeling for financial data fusion"""
    
    def __init__(
        self,
        text_dim: int,
        numerical_dim: int,
        temporal_dim: int,
        hidden_dim: int = 256,
        num_causal_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Causal relationship networks
        # Text -> Numerical causality
        self.text_to_numerical = nn.Sequential(
            nn.Linear(text_dim + numerical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, numerical_dim)
        )
        
        # Temporal -> Numerical causality
        self.temporal_to_numerical = nn.Sequential(
            nn.Linear(temporal_dim + numerical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, numerical_dim)
        )
        
        # Text -> Temporal causality
        self.text_to_temporal = nn.Sequential(
            nn.Linear(text_dim + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, temporal_dim)
        )
        
        # Causal strength prediction
        self.causal_strength = nn.Sequential(
            nn.Linear(text_dim + numerical_dim + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3 causal relationships
            nn.Sigmoid()
        )
        
    def forward(
        self,
        text_emb: torch.Tensor,
        numerical_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for causal fusion
        
        Returns:
            enhanced_text: Causally enhanced text embeddings
            enhanced_numerical: Causally enhanced numerical embeddings
            enhanced_temporal: Causally enhanced temporal embeddings
            causal_weights: Causal relationship strengths
        """
        # Predict causal relationship strengths
        combined_input = torch.cat([text_emb, numerical_emb, temporal_emb], dim=-1)
        causal_weights = self.causal_strength(combined_input)
        
        # Apply causal relationships
        # Text affects numerical indicators
        text_numerical_input = torch.cat([text_emb, numerical_emb], dim=-1)
        text_numerical_effect = self.text_to_numerical(text_numerical_input)
        enhanced_numerical = numerical_emb + causal_weights[:, 0:1] * text_numerical_effect
        
        # Temporal patterns affect numerical values
        temporal_numerical_input = torch.cat([temporal_emb, numerical_emb], dim=-1)
        temporal_numerical_effect = self.temporal_to_numerical(temporal_numerical_input)
        enhanced_numerical = enhanced_numerical + causal_weights[:, 1:2] * temporal_numerical_effect
        
        # Text sentiment affects temporal dynamics
        text_temporal_input = torch.cat([text_emb, temporal_emb], dim=-1)
        text_temporal_effect = self.text_to_temporal(text_temporal_input)
        enhanced_temporal = temporal_emb + causal_weights[:, 2:3] * text_temporal_effect
        
        # Text embeddings remain unchanged (they are the causal source)
        enhanced_text = text_emb
        
        return enhanced_text, enhanced_numerical, enhanced_temporal, causal_weights


class GatingNetwork(nn.Module):
    """Adaptive gating network for modality fusion"""
    
    def __init__(
        self,
        text_dim: int,
        numerical_dim: int,
        temporal_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        total_input_dim = text_dim + numerical_dim + temporal_dim
        
        # Modality importance gating
        self.importance_gates = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3 modalities
            nn.Softmax(dim=-1)
        )
        
        # Feature gating for each modality
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Sigmoid()
        )
        
        self.numerical_gate = nn.Sequential(
            nn.Linear(numerical_dim, numerical_dim),
            nn.Sigmoid()
        )
        
        self.temporal_gate = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(
        self,
        text_emb: torch.Tensor,
        numerical_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for gating network
        
        Returns:
            fused_embeddings: Gated and fused embeddings
            gate_weights: Gating weights for interpretability
        """
        # Calculate modality importance weights
        combined_input = torch.cat([text_emb, numerical_emb, temporal_emb], dim=-1)
        importance_weights = self.importance_gates(combined_input)
        
        # Apply feature-level gating
        gated_text = text_emb * self.text_gate(text_emb)
        gated_numerical = numerical_emb * self.numerical_gate(numerical_emb)
        gated_temporal = temporal_emb * self.temporal_gate(temporal_emb)
        
        # Apply modality-level importance weighting
        weighted_text = gated_text * importance_weights[:, 0:1]
        weighted_numerical = gated_numerical * importance_weights[:, 1:2]
        weighted_temporal = gated_temporal * importance_weights[:, 2:3]
        
        # Final fusion
        gated_combined = torch.cat([weighted_text, weighted_numerical, weighted_temporal], dim=-1)
        fused_embeddings = self.fusion_layer(gated_combined)
        
        # Collect gating information
        gate_weights = {
            'modality_importance': importance_weights.detach(),
            'text_gates': self.text_gate(text_emb).detach(),
            'numerical_gates': self.numerical_gate(numerical_emb).detach(),
            'temporal_gates': self.temporal_gate(temporal_emb).detach()
        }
        
        return fused_embeddings, gate_weights


class CrossModalFusionLayer(nn.Module):
    """
    Complete cross-modal fusion layer combining multiple fusion strategies
    
    Features:
    - Multiple fusion strategies (attention, concatenation, gating)
    - Causal relationship modeling
    - Interpretability tools
    - Adaptive fusion weights
    """
    
    def __init__(
        self,
        text_dim: int,
        numerical_dim: int,
        temporal_dim: int,
        output_dim: int = 512,
        fusion_strategy: str = "attention",
        use_causality: bool = True,
        use_gating: bool = True,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.numerical_dim = numerical_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.fusion_strategy = fusion_strategy
        self.use_causality = use_causality
        self.use_gating = use_gating
        
        # Causal fusion module
        if use_causality:
            self.causal_module = CausalFusionModule(
                text_dim=text_dim,
                numerical_dim=numerical_dim,
                temporal_dim=temporal_dim,
                hidden_dim=hidden_dim,
                dropout=dropout_rate
            )
        
        # Main fusion strategy
        if fusion_strategy == "attention":
            self.fusion_module = CrossModalAttention(
                text_dim=text_dim,
                numerical_dim=numerical_dim,
                temporal_dim=temporal_dim,
                attention_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate
            )
            fusion_output_dim = hidden_dim
            
        elif fusion_strategy == "gating":
            self.fusion_module = GatingNetwork(
                text_dim=text_dim,
                numerical_dim=numerical_dim,
                temporal_dim=temporal_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout_rate
            )
            fusion_output_dim = output_dim
            
        elif fusion_strategy == "concatenation":
            total_dim = text_dim + numerical_dim + temporal_dim
            self.fusion_module = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            fusion_output_dim = output_dim
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Output projection (if needed)
        if fusion_output_dim != output_dim:
            self.output_projection = nn.Linear(fusion_output_dim, output_dim)
        else:
            self.output_projection = None
        
        # Store attention patterns for interpretability
        self.last_attention_patterns = None
        self.last_causal_weights = None
        self.last_gate_weights = None
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        numerical_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion layer
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            numerical_embeddings: Numerical embeddings [batch_size, numerical_dim]
            temporal_embeddings: Temporal embeddings [batch_size, temporal_dim]
            
        Returns:
            Dict containing:
                - fused_embeddings: Final fused embeddings
                - text_embeddings: (Possibly enhanced) text embeddings
                - numerical_embeddings: (Possibly enhanced) numerical embeddings
                - temporal_embeddings: (Possibly enhanced) temporal embeddings
                - fusion_weights: Fusion weights for interpretability
        """
        enhanced_text = text_embeddings
        enhanced_numerical = numerical_embeddings
        enhanced_temporal = temporal_embeddings
        
        # Apply causal fusion if enabled
        if self.use_causality:
            enhanced_text, enhanced_numerical, enhanced_temporal, causal_weights = self.causal_module(
                text_embeddings, numerical_embeddings, temporal_embeddings
            )
            self.last_causal_weights = causal_weights.detach()
        
        # Apply main fusion strategy
        if self.fusion_strategy == "attention":
            fused_embeddings, attention_patterns = self.fusion_module(
                enhanced_text, enhanced_numerical, enhanced_temporal
            )
            self.last_attention_patterns = attention_patterns
            fusion_weights = attention_patterns['cross_modal_attention']
            
        elif self.fusion_strategy == "gating":
            fused_embeddings, gate_weights = self.fusion_module(
                enhanced_text, enhanced_numerical, enhanced_temporal
            )
            self.last_gate_weights = gate_weights
            fusion_weights = gate_weights['modality_importance']
            
        elif self.fusion_strategy == "concatenation":
            concatenated = torch.cat([enhanced_text, enhanced_numerical, enhanced_temporal], dim=-1)
            fused_embeddings = self.fusion_module(concatenated)
            # Create dummy fusion weights for consistency
            batch_size = concatenated.size(0)
            fusion_weights = torch.ones(batch_size, 3, device=concatenated.device) / 3.0
            
        # Apply output projection if needed
        if self.output_projection is not None:
            fused_embeddings = self.output_projection(fused_embeddings)
        
        return {
            'fused_embeddings': fused_embeddings,
            'text_embeddings': enhanced_text,
            'numerical_embeddings': enhanced_numerical,
            'temporal_embeddings': enhanced_temporal,
            'fusion_weights': fusion_weights
        }
    
    def get_attention_patterns(
        self,
        text_embeddings: torch.Tensor,
        numerical_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """Get attention patterns for interpretability"""
        with torch.no_grad():
            _ = self.forward(text_embeddings, numerical_embeddings, temporal_embeddings)
        
        patterns = {}
        
        if self.last_attention_patterns is not None:
            patterns['attention'] = self.last_attention_patterns
        
        if self.last_causal_weights is not None:
            patterns['causal_weights'] = self.last_causal_weights
        
        if self.last_gate_weights is not None:
            patterns['gate_weights'] = self.last_gate_weights
        
        return patterns
    
    def get_modality_importance(self) -> Dict[str, float]:
        """Get average modality importance across last batch"""
        if self.fusion_strategy == "attention" and self.last_attention_patterns is not None:
            # Average attention weights across batch and heads
            attention = self.last_attention_patterns['cross_modal_attention']
            importance = attention.mean(dim=0).cpu().numpy()
            
            return {
                'text_importance': float(importance[0]),
                'numerical_importance': float(importance[1]),
                'temporal_importance': float(importance[2])
            }
            
        elif self.fusion_strategy == "gating" and self.last_gate_weights is not None:
            # Average gating weights across batch
            importance = self.last_gate_weights['modality_importance'].mean(dim=0).cpu().numpy()
            
            return {
                'text_importance': float(importance[0]),
                'numerical_importance': float(importance[1]),
                'temporal_importance': float(importance[2])
            }
        
        else:
            # Equal importance for concatenation strategy
            return {
                'text_importance': 1.0/3.0,
                'numerical_importance': 1.0/3.0,
                'temporal_importance': 1.0/3.0
            }
    
    def explain_fusion(self, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain fusion process for a specific sample"""
        explanation = {
            'fusion_strategy': self.fusion_strategy,
            'uses_causality': self.use_causality,
            'uses_gating': self.use_gating
        }
        
        if self.last_attention_patterns is not None:
            explanation['attention_patterns'] = {
                k: v[sample_idx].cpu().numpy() if v.dim() > 1 else v.cpu().numpy()
                for k, v in self.last_attention_patterns.items()
            }
        
        if self.last_causal_weights is not None:
            explanation['causal_relationships'] = {
                'text_to_numerical': float(self.last_causal_weights[sample_idx, 0]),
                'temporal_to_numerical': float(self.last_causal_weights[sample_idx, 1]),
                'text_to_temporal': float(self.last_causal_weights[sample_idx, 2])
            }
        
        if self.last_gate_weights is not None:
            explanation['gating_weights'] = {
                k: v[sample_idx].cpu().numpy() if v.dim() > 1 else v.cpu().numpy()
                for k, v in self.last_gate_weights.items()
            }
        
        return explanation