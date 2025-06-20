#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Neural Network Model for Semiconductor Industry Analysis

Advanced temporal graph neural network architecture for causal relationship
modeling in semiconductor supply chain and financial analysis.

Author: FicTGNN Research Team
Date: 2024
Version: 2.0 (Research Framework)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, List, Any, Optional, Tuple


class CausalityAwareTemporalGNN(nn.Module):
    """
    Causality-Aware Temporal Graph Neural Network Model
    
    This model implements a sophisticated architecture for modeling temporal
    causal relationships in semiconductor industry networks, featuring
    multi-head attention mechanisms and temporal encoding.
    
    Args:
        config (Dict[str, Any]): Model configuration parameters
        
    Attributes:
        config: Model configuration dictionary
        logger: Logging interface
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model hyperparameters
        self.input_dim = config.get('model', {}).get('input_dim', 28)
        self.hidden_dim = config.get('model', {}).get('hidden_dim', 256)
        self.output_dim = config.get('model', {}).get('output_dim', 128)
        self.num_heads = config.get('model', {}).get('num_heads', 8)
        self.dropout_rate = config.get('model', {}).get('dropout_rate', 0.1)
        
        # Edge type-specific processing layers
        self.temporal_gat = GATConv(
            self.input_dim, 
            self.hidden_dim, 
            heads=self.num_heads, 
            edge_dim=1
        )
        self.causal_gat = GATConv(
            self.input_dim, 
            self.hidden_dim, 
            heads=self.num_heads, 
            edge_dim=1
        )
        
        # Temporal encoding module
        self.temporal_encoder = nn.LSTM(
            self.hidden_dim * self.num_heads, 
            self.hidden_dim, 
            batch_first=True
        )
        
        # Causal relationship attention mechanism
        self.causal_attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=4
        )
        
        # Feature fusion layers
        self.fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.logger.info(f"GNN model initialized successfully")
        self.logger.info(f"  Input dimension: {self.input_dim}")
        self.logger.info(f"  Hidden dimension: {self.hidden_dim}")
        self.logger.info(f"  Output dimension: {self.output_dim}")
        self.logger.info(f"  Attention heads: {self.num_heads}")
    
    def forward(self, x: torch.Tensor, edge_index_temporal: torch.Tensor, 
                edge_index_causal: torch.Tensor, edge_attr_temporal: torch.Tensor, 
                edge_attr_causal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward propagation through the network
        
        Args:
            x: Node features (N, input_dim)
            edge_index_temporal: Temporal edge indices (2, E_temporal)
            edge_index_causal: Causal edge indices (2, E_causal)
            edge_attr_temporal: Temporal edge attributes (E_temporal, 1)
            edge_attr_causal: Causal edge attributes (E_causal, 1)
            
        Returns:
            Tuple containing:
                - output: Final node embeddings (N, output_dim)
                - h_temporal_final: Temporal features (N, hidden_dim)
                - h_causal_final: Causal features (N, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Temporal relationship processing
        if edge_index_temporal.size(1) > 0:
            h_temporal = self.temporal_gat(x, edge_index_temporal, edge_attr_temporal)
            h_temporal = self.dropout(h_temporal)
            
            # LSTM temporal encoding
            h_temporal_seq = h_temporal.view(batch_size, 1, -1)
            h_temporal_lstm, _ = self.temporal_encoder(h_temporal_seq)
            h_temporal_final = h_temporal_lstm.squeeze(1)
        else:
            h_temporal_final = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        # Causal relationship processing
        if edge_index_causal.size(1) > 0:
            h_causal = self.causal_gat(x, edge_index_causal, edge_attr_causal)
            h_causal = self.dropout(h_causal)
            
            # Multi-head attention for causal relationships
            h_causal_att, _ = self.causal_attention(
                h_causal.unsqueeze(0), 
                h_causal.unsqueeze(0), 
                h_causal.unsqueeze(0)
            )
            h_causal_final = h_causal_att.squeeze(0)
        else:
            h_causal_final = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        # Feature fusion
        h_combined = torch.cat([h_temporal_final, h_causal_final], dim=-1)
        h_fused = torch.relu(self.fusion(h_combined))
        h_fused = self.dropout(h_fused)
        
        # Output projection
        output = self.output_layer(h_fused)
        
        return output, h_temporal_final, h_causal_final
    
    def train_model(self, data_loader: Any, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module, epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the GNN model using provided data loader
        
        Args:
            data_loader: Training data loader
            optimizer: Optimization algorithm
            criterion: Loss function
            epochs: Number of training epochs
            
        Returns:
            Dict[str, List[float]]: Training history metrics
        """
        self.train()
        train_losses = []
        
        self.logger.info(f"Starting GNN model training (epochs: {epochs})")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Forward pass
                output, _, _ = self.forward(
                    batch.x,
                    batch.edge_index_temporal,
                    batch.edge_index_causal,
                    batch.edge_attr_temporal,
                    batch.edge_attr_causal
                )
                
                # Loss computation (self-supervised learning approach)
                loss = criterion(output, batch.y if hasattr(batch, 'y') else output)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
        
        self.logger.info("GNN model training completed successfully")
        
        return {"train_losses": train_losses}
    
    def predict(self, x: torch.Tensor, edge_index_temporal: torch.Tensor, 
                edge_index_causal: torch.Tensor, edge_attr_temporal: torch.Tensor, 
                edge_attr_causal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform inference with the trained model
        
        Args:
            x: Node features
            edge_index_temporal: Temporal edge indices
            edge_index_causal: Causal edge indices
            edge_attr_temporal: Temporal edge attributes
            edge_attr_causal: Causal edge attributes
            
        Returns:
            Dict[str, torch.Tensor]: Prediction results containing embeddings and features
        """
        self.eval()
        
        with torch.no_grad():
            output, h_temporal, h_causal = self.forward(
                x, edge_index_temporal, edge_index_causal,
                edge_attr_temporal, edge_attr_causal
            )
        
        return {
            "embeddings": output,
            "temporal_features": h_temporal,
            "causal_features": h_causal
        }
    
    def get_attention_weights(self, x: torch.Tensor, edge_index_temporal: torch.Tensor, 
                             edge_index_causal: torch.Tensor, edge_attr_temporal: torch.Tensor, 
                             edge_attr_causal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for interpretability analysis
        
        Args:
            x: Node features
            edge_index_temporal: Temporal edge indices
            edge_index_causal: Causal edge indices
            edge_attr_temporal: Temporal edge attributes
            edge_attr_causal: Causal edge attributes
            
        Returns:
            Dict[str, torch.Tensor]: Attention weight matrices
        """
        self.eval()
        
        # Enable attention weight extraction in GAT layers
        self.temporal_gat.explain = True
        self.causal_gat.explain = True
        
        with torch.no_grad():
            # Temporal attention weights
            if edge_index_temporal.size(1) > 0:
                _, temporal_attn = self.temporal_gat(
                    x, edge_index_temporal, edge_attr_temporal, return_attention_weights=True
                )
            else:
                temporal_attn = None
            
            # Causal attention weights
            if edge_index_causal.size(1) > 0:
                _, causal_attn = self.causal_gat(
                    x, edge_index_causal, edge_attr_causal, return_attention_weights=True
                )
            else:
                causal_attn = None
        
        return {
            "temporal_attention": temporal_attn,
            "causal_attention": causal_attn
        }
    
    def explain_prediction(self, x: torch.Tensor, edge_index_temporal: torch.Tensor, 
                          edge_index_causal: torch.Tensor, edge_attr_temporal: torch.Tensor, 
                          edge_attr_causal: torch.Tensor, node_idx: int = 0) -> Dict[str, Any]:
        """
        Generate prediction explanations for interpretability
        
        Args:
            x: Node features
            edge_index_temporal: Temporal edge indices
            edge_index_causal: Causal edge indices
            edge_attr_temporal: Temporal edge attributes
            edge_attr_causal: Causal edge attributes
            node_idx: Node index for explanation focus
            
        Returns:
            Dict[str, Any]: Comprehensive explanation results
        """
        # Extract attention weights
        attention_weights = self.get_attention_weights(
            x, edge_index_temporal, edge_index_causal,
            edge_attr_temporal, edge_attr_causal
        )
        
        # Generate predictions
        predictions = self.predict(
            x, edge_index_temporal, edge_index_causal,
            edge_attr_temporal, edge_attr_causal
        )
        
        # Calculate feature importance using gradient-based method
        x_copy = x.clone().requires_grad_(True)
        output, _, _ = self.forward(
            x_copy, edge_index_temporal, edge_index_causal,
            edge_attr_temporal, edge_attr_causal
        )
        
        # Compute gradients for specific node
        if node_idx < output.size(0):
            node_output = output[node_idx].sum()
            node_output.backward()
            feature_importance = x_copy.grad[node_idx].abs()
        else:
            feature_importance = torch.zeros_like(x[0])
        
        return {
            "predictions": predictions,
            "attention_weights": attention_weights,
            "feature_importance": feature_importance,
            "explained_node": node_idx
        }
    
    def save_model(self, path: str):
        """
        Save model state and configuration
        
        Args:
            path: File path for model saving
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_heads': self.num_heads
        }, path)
        self.logger.info(f"Model saved successfully: {path}")
    
    @classmethod
    def load_model(cls, path: str, config: Dict[str, Any] = None):
        """
        Load pre-trained model from file
        
        Args:
            path: File path to load model from
            config: Optional configuration override
            
        Returns:
            CausalityAwareTemporalGNN: Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        if config is None:
            config = checkpoint.get('config', {})
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive model summary statistics
        
        Returns:
            Dict[str, Any]: Model architecture and parameter information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # float32 assumption
        }