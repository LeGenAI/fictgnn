"""
Temporal Encoder for Financial Time Series

Advanced temporal encoder designed for financial time series data.
Handles quarterly reports, seasonal patterns, and trend analysis.

Features:
- LSTM/GRU based sequence modeling
- Positional encoding for quarterly data
- Attention mechanisms for temporal patterns
- Trend and seasonality decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for quarterly financial data"""
    
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Standard sinusoidal positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add seasonal (quarterly) encoding
        quarter_pattern = torch.zeros(max_len, d_model)
        for i in range(max_len):
            quarter = i % 4  # Quarterly pattern
            quarter_pattern[i, :] = torch.sin(2 * math.pi * quarter / 4.0)
        
        # Combine standard and seasonal encoding
        pe = pe + 0.3 * quarter_pattern
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TemporalAttention(nn.Module):
    """Multi-head attention for temporal sequences"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            attended_output: Attended features [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output linear layer
        output = self.output_linear(attended)
        
        return output, attention_weights


class FinancialTemporalPreprocessor:
    """Preprocessor for financial temporal data"""
    
    def __init__(self, sequence_length: int = 8, expected_features: int = 15):
        self.sequence_length = sequence_length
        self.expected_features = expected_features
        
        # Financial temporal feature names
        self.feature_names = [
            'quarter_revenue', 'quarter_profit', 'quarter_margin',
            'revenue_growth_qoq', 'profit_growth_qoq', 'margin_change_qoq',
            'revenue_growth_yoy', 'profit_growth_yoy', 'margin_change_yoy',
            'quarter_position', 'year_progress', 'seasonal_factor',
            'market_volatility', 'industry_trend', 'macroeconomic_indicator'
        ]
    
    def extract_quarterly_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract quarterly features from data point"""
        features = []
        
        # Get temporal values
        temporal_values = data_point.get('values', [])
        if isinstance(temporal_values, list) and len(temporal_values) >= self.expected_features:
            features = temporal_values[:self.expected_features]
        elif isinstance(temporal_values, dict):
            # Extract by feature name
            for feature_name in self.feature_names[:self.expected_features]:
                value = temporal_values.get(feature_name, 0.0)
                features.append(float(value) if value is not None else 0.0)
        else:
            # Use default values
            features = [0.0] * self.expected_features
        
        # Ensure correct length
        while len(features) < self.expected_features:
            features.append(0.0)
        
        return features[:self.expected_features]
    
    def process_batch(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Process batch of temporal data"""
        batch_data = []
        
        for data_point in data:
            # Extract features for this data point
            features = self.extract_quarterly_features(data_point)
            
            # Create sequence (for now, repeat the same values)
            # In a real scenario, this would be historical data
            sequence = []
            for i in range(self.sequence_length):
                # Add some temporal variation
                temporal_features = features.copy()
                if i > 0:
                    # Add small random variations for older quarters
                    noise_factor = 0.05 * i
                    temporal_features = [
                        f * (1 + np.random.normal(0, noise_factor))
                        for f in temporal_features
                    ]
                sequence.append(temporal_features)
            
            batch_data.append(sequence)
        
        return np.array(batch_data, dtype=np.float32)


class TemporalEncoder(nn.Module):
    """
    Advanced temporal encoder for financial time series
    
    Features:
    - LSTM/GRU for sequence modeling
    - Positional encoding for quarterly patterns
    - Attention mechanisms
    - Trend analysis
    """
    
    def __init__(
        self,
        temporal_dim: int = 15,
        hidden_dim: int = 128,
        output_dim: int = 256,
        sequence_length: int = 8,
        use_lstm: bool = True,
        num_layers: int = 2,
        use_attention: bool = True,
        use_positional_encoding: bool = True,
        dropout_rate: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        self.device = device
        
        # Preprocessor
        self.preprocessor = FinancialTemporalPreprocessor(
            sequence_length=sequence_length,
            expected_features=temporal_dim
        )
        
        # Input projection
        self.input_projection = nn.Linear(temporal_dim, hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(hidden_dim, sequence_length)
        else:
            self.positional_encoding = None
        
        # Sequence model
        if use_lstm:
            self.sequence_model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        else:
            self.sequence_model = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        
        # Attention mechanism
        if use_attention:
            self.temporal_attention = TemporalAttention(
                d_model=hidden_dim,
                num_heads=8,
                dropout=dropout_rate
            )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Feature importance tracking
        self.attention_weights = None
        
    def process_raw_data(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """Process raw temporal data into tensor"""
        processed_data = self.preprocessor.process_batch(data)
        return torch.tensor(processed_data, dtype=torch.float32, device=self.device)
    
    def forward(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: List of temporal data dictionaries
            
        Returns:
            embeddings: Temporal embeddings [batch_size, output_dim]
        """
        # Process input data
        x = self.process_raw_data(data)  # [batch_size, seq_len, temporal_dim]
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        
        # Sequence modeling
        lstm_output, (hidden, cell) = self.sequence_model(x)
        # lstm_output: [batch_size, seq_len, hidden_dim]
        
        # Apply attention if enabled
        if self.use_attention:
            attended_output, attention_weights = self.temporal_attention(lstm_output)
            self.attention_weights = attention_weights.detach().cpu().numpy()
            
            # Global average pooling with attention
            sequence_embedding = attended_output.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            # Use last hidden state
            sequence_embedding = hidden[-1]  # [batch_size, hidden_dim]
        
        # Output transformation
        embeddings = self.output_layers(sequence_embedding)
        
        return embeddings
    
    def encode_batch(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """Encode batch of temporal data"""
        with torch.no_grad():
            return self.forward(data)
    
    def encode_single(self, data: Dict[str, Any]) -> torch.Tensor:
        """Encode single temporal data point"""
        return self.encode_batch([data])
    
    def get_temporal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in data"""
        x = self.process_raw_data(data)
        
        with torch.no_grad():
            embeddings = self.forward(data)
            
            patterns = {
                'embeddings': embeddings.cpu().numpy(),
                'input_sequences': x.cpu().numpy(),
                'sequence_length': self.sequence_length,
                'feature_names': self.preprocessor.feature_names
            }
            
            if self.attention_weights is not None:
                patterns['attention_weights'] = self.attention_weights
                patterns['temporal_importance'] = np.mean(self.attention_weights, axis=(0, 1))
        
        return patterns
    
    def get_trend_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in temporal data"""
        x = self.process_raw_data(data)
        x_np = x.cpu().numpy()
        
        analysis = {}
        
        for i, feature_name in enumerate(self.preprocessor.feature_names):
            if i >= x_np.shape[2]:
                break
                
            feature_data = x_np[:, :, i]  # [batch_size, seq_len]
            
            # Calculate trends for each sample
            trends = []
            for sample in feature_data:
                # Simple linear trend
                time_points = np.arange(len(sample))
                slope = np.polyfit(time_points, sample, 1)[0]
                trends.append(slope)
            
            analysis[feature_name] = {
                'mean_trend': np.mean(trends),
                'trend_std': np.std(trends),
                'positive_trends': np.sum(np.array(trends) > 0),
                'negative_trends': np.sum(np.array(trends) < 0)
            }
        
        return analysis
    
    def get_seasonality_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonality patterns"""
        x = self.process_raw_data(data)
        x_np = x.cpu().numpy()
        
        seasonality = {}
        
        for i, feature_name in enumerate(self.preprocessor.feature_names):
            if i >= x_np.shape[2]:
                break
                
            feature_data = x_np[:, :, i]  # [batch_size, seq_len]
            
            # Calculate quarterly patterns
            quarterly_means = []
            for quarter in range(4):
                quarter_indices = [q for q in range(quarter, self.sequence_length, 4)]
                if quarter_indices:
                    quarter_values = feature_data[:, quarter_indices]
                    quarterly_means.append(np.mean(quarter_values))
                else:
                    quarterly_means.append(0.0)
            
            seasonality[feature_name] = {
                'quarterly_pattern': quarterly_means,
                'seasonal_strength': np.std(quarterly_means),
                'peak_quarter': np.argmax(quarterly_means) + 1,
                'trough_quarter': np.argmin(quarterly_means) + 1
            }
        
        return seasonality
    
    def get_embedding_dimension(self) -> int:
        """Return output embedding dimension"""
        return self.output_dim
    
    def explain_temporal_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain temporal features for a single data point"""
        patterns = self.get_temporal_patterns([data])
        
        explanation = {
            'embedding': patterns['embeddings'][0],
            'input_sequence': patterns['input_sequences'][0],
            'feature_names': patterns['feature_names']
        }
        
        if 'attention_weights' in patterns:
            explanation['temporal_attention'] = patterns['attention_weights'][0]
            explanation['most_important_timesteps'] = np.argsort(
                patterns['temporal_importance']
            )[-3:]
        
        # Add trend information
        sequence = patterns['input_sequences'][0]
        trends = {}
        for i, feature_name in enumerate(patterns['feature_names']):
            if i < sequence.shape[1]:
                time_points = np.arange(len(sequence))
                slope = np.polyfit(time_points, sequence[:, i], 1)[0]
                trends[feature_name] = slope
        
        explanation['feature_trends'] = trends
        
        return explanation