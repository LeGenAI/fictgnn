"""
Numerical Encoder for Financial Data

Advanced numerical encoder specifically designed for financial time series data.
Includes economic indicators, financial ratios, and market metrics.

Features:
- Feature normalization and scaling
- Anomaly detection
- Attention mechanisms
- Interpretability tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FinancialFeatureProcessor:
    """Financial feature preprocessing and validation"""
    
    def __init__(
        self,
        expected_features: int = 42,
        use_robust_scaling: bool = True,
        handle_missing: str = "mean"
    ):
        self.expected_features = expected_features
        self.use_robust_scaling = use_robust_scaling
        self.handle_missing = handle_missing
        
        # Initialize scalers
        if use_robust_scaling:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy=handle_missing)
        
        # Feature names (standard financial features)
        self.feature_names = [
            # Basic financial metrics
            'revenue', 'operating_profit', 'net_profit', 'total_assets', 'equity',
            'debt', 'cash', 'capex', 'fcf', 'dividend',
            
            # Financial ratios
            'roe', 'roa', 'debt_ratio', 'current_ratio', 'quick_ratio',
            'asset_turnover', 'inventory_turnover', 'receivables_turnover',
            
            # Market metrics
            'market_cap', 'enterprise_value', 'pe_ratio', 'pb_ratio', 'ps_ratio',
            'ev_ebitda', 'dividend_yield', 'beta',
            
            # Growth metrics
            'revenue_growth', 'profit_growth', 'asset_growth', 'book_value_growth',
            
            # Industry-specific metrics (semiconductor)
            'capacity_utilization', 'r_and_d_intensity', 'gross_margin', 'operating_margin',
            'net_margin', 'inventory_days', 'dso', 'dpo',
            
            # Additional features
            'working_capital', 'ebitda', 'ebit', 'interest_coverage',
            'debt_service_coverage', 'price_to_sales'
        ]
        
        self.fitted = False
    
    def process_raw_data(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Convert raw data to numerical matrix"""
        processed_data = []
        
        for sample in data:
            if isinstance(sample, dict):
                # Extract features by name
                feature_vector = []
                for feature_name in self.feature_names[:self.expected_features]:
                    value = sample.get(feature_name, 0.0)
                    if value is None or np.isnan(value):
                        value = 0.0
                    feature_vector.append(float(value))
                
                # Pad or truncate to expected size
                while len(feature_vector) < self.expected_features:
                    feature_vector.append(0.0)
                feature_vector = feature_vector[:self.expected_features]
                
            elif isinstance(sample, (list, np.ndarray)):
                # Already numerical
                feature_vector = list(sample)[:self.expected_features]
                while len(feature_vector) < self.expected_features:
                    feature_vector.append(0.0)
            else:
                # Invalid format
                feature_vector = [0.0] * self.expected_features
            
            processed_data.append(feature_vector)
        
        return np.array(processed_data, dtype=np.float32)
    
    def fit_transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Fit preprocessors and transform data"""
        raw_data = self.process_raw_data(data)
        
        # Handle missing values
        imputed_data = self.imputer.fit_transform(raw_data)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(imputed_data)
        
        self.fitted = True
        return scaled_data
    
    def transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Transform data using fitted preprocessors"""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")
        
        raw_data = self.process_raw_data(data)
        imputed_data = self.imputer.transform(raw_data)
        scaled_data = self.scaler.transform(imputed_data)
        
        return scaled_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
        
        return self.scaler.inverse_transform(data)


class AttentionMechanism(nn.Module):
    """Multi-head attention for numerical features"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, feature_dim]
            
        Returns:
            attended_features: Attended features [batch_size, feature_dim]
            attention_weights: Attention weights [batch_size, num_heads, 1, 1]
        """
        batch_size = x.size(0)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Linear transformations
        Q = self.query(x)  # [batch_size, 1, feature_dim]
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, 1, self.feature_dim
        )
        
        # Output linear layer
        output = self.output_linear(attended).squeeze(1)
        
        return output, attention_weights.squeeze(-1).squeeze(-1)


class AnomalyDetector(nn.Module):
    """Autoencoder-based anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Reconstruction error
        reconstruction_error = F.mse_loss(decoded, x, reduction='none').mean(dim=1)
        
        return decoded, reconstruction_error


class NumericalEncoder(nn.Module):
    """
    Advanced numerical encoder for financial data
    
    Features:
    - Feature preprocessing and normalization
    - Multi-layer neural network
    - Optional attention mechanism
    - Anomaly detection
    - Interpretability tools
    """
    
    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        use_attention: bool = True,
        use_anomaly_detection: bool = True,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        device: str = "cuda"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.use_anomaly_detection = use_anomaly_detection
        self.device = device
        
        # Feature processor
        self.feature_processor = FinancialFeatureProcessor(
            expected_features=input_dim,
            use_robust_scaling=True
        )
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        
        # Main neural network
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer
                layers.extend([
                    nn.Linear(current_dim, output_dim),
                    nn.LayerNorm(output_dim)
                ])
            else:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    self.activation,
                    nn.Dropout(dropout_rate),
                    nn.LayerNorm(hidden_dim)
                ])
                current_dim = hidden_dim
        
        self.main_network = nn.Sequential(*layers)
        
        # Optional attention mechanism
        if use_attention:
            self.attention = AttentionMechanism(
                feature_dim=input_dim,
                num_heads=8,
                dropout=dropout_rate
            )
        
        # Optional anomaly detection
        if use_anomaly_detection:
            self.anomaly_detector = AnomalyDetector(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2,
                latent_dim=hidden_dim // 4
            )
        
        # Feature importance tracking
        self.feature_importance = None
        
    def process_raw_data(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """Process raw data into tensor"""
        if not self.feature_processor.fitted:
            # First time - fit and transform
            processed_data = self.feature_processor.fit_transform(data)
        else:
            # Already fitted - just transform
            processed_data = self.feature_processor.transform(data)
        
        return torch.tensor(processed_data, dtype=torch.float32, device=self.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            embeddings: Encoded features [batch_size, output_dim]
            anomaly_scores: Anomaly scores [batch_size] (if enabled)
        """
        batch_size = x.size(0)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_features, attention_weights = self.attention(x)
            # Store attention weights for interpretability
            self.feature_importance = attention_weights.detach().cpu().numpy()
            x_processed = attended_features
        else:
            x_processed = x
        
        # Main encoding
        embeddings = self.main_network(x_processed)
        
        # Anomaly detection
        anomaly_scores = torch.zeros(batch_size, device=self.device)
        if self.use_anomaly_detection:
            _, reconstruction_error = self.anomaly_detector(x)
            anomaly_scores = reconstruction_error
        
        return embeddings, anomaly_scores
    
    def encode_batch(self, data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of data"""
        with torch.no_grad():
            x = self.process_raw_data(data)
            return self.forward(x)
    
    def encode_single(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single data point"""
        return self.encode_batch([data])
    
    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """Get feature importance scores"""
        if not self.use_attention:
            return np.ones(self.input_dim) / self.input_dim
        
        # Run forward pass to compute attention
        with torch.no_grad():
            self.forward(x)
        
        if self.feature_importance is not None:
            # Average across heads and samples
            importance = np.mean(self.feature_importance, axis=(0, 1))
            return importance / np.sum(importance)  # Normalize
        else:
            return np.ones(self.input_dim) / self.input_dim
    
    def detect_anomalies(self, data: List[Dict[str, Any]], threshold: float = 0.5) -> List[bool]:
        """Detect anomalies in data"""
        if not self.use_anomaly_detection:
            return [False] * len(data)
        
        with torch.no_grad():
            _, anomaly_scores = self.encode_batch(data)
            
            # Normalize scores using percentile-based threshold
            scores_np = anomaly_scores.cpu().numpy()
            threshold_value = np.percentile(scores_np, threshold * 100)
            
            return (scores_np > threshold_value).tolist()
    
    def get_embedding_dimension(self) -> int:
        """Return output embedding dimension"""
        return self.output_dim
    
    def get_feature_names(self) -> List[str]:
        """Return feature names"""
        return self.feature_processor.feature_names[:self.input_dim]
    
    def explain_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain prediction for a single data point"""
        x = self.process_raw_data([data])
        
        with torch.no_grad():
            embeddings, anomaly_score = self.forward(x)
            
            explanation = {
                'embedding': embeddings[0].cpu().numpy(),
                'anomaly_score': anomaly_score[0].item(),
                'input_features': x[0].cpu().numpy(),
                'feature_names': self.get_feature_names()
            }
            
            if self.use_attention and self.feature_importance is not None:
                explanation['feature_importance'] = self.feature_importance[0]
                
                # Top important features
                importance_scores = self.feature_importance[0].mean(axis=0)
                top_indices = np.argsort(importance_scores)[-5:]
                explanation['top_features'] = [
                    (self.get_feature_names()[i], importance_scores[i])
                    for i in reversed(top_indices)
                ]
        
        return explanation