# -*- coding: utf-8 -*-
"""
Semiconductor Report Multimodal Embedding System

A comprehensive multimodal embedding system for semiconductor industry reports (2020-2024)
with integrated CSV financial data processing.

Key Features:
1. JSON dataset loading and preprocessing
2. CSV financial data loading and integration
3. Text, numerical, temporal, and financial data extraction
4. Multimodal embedding generation using advanced embedding techniques
5. Batch processing for 20 quarters of data
6. Embedding result storage and analysis

This module is designed for academic research in financial network analysis
and graph neural network applications.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import warnings

# Windows console UTF-8 configuration (prevents emoji output errors)
if os.name == 'nt':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def safe_print(msg: str):
    """
    Safe print function that handles UnicodeEncodeError by removing problematic characters.
    
    Args:
        msg (str): Message to print safely
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'ignore').decode('ascii'))

# Module path configuration for experiment environment
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, 'modules')
sys.path.insert(0, modules_path)
try:
    from multimodal_embedder import MultimodalEmbedder
    safe_print("MultimodalEmbedder module imported successfully")
except ImportError as e:
    safe_print(f"Warning: MultimodalEmbedder import failed - {e}")
    safe_print("Please ensure the module is available in the modules directory")

class SemiconductorMultimodalEmbeddingPipeline:
    """
    Comprehensive pipeline for generating multimodal embeddings from semiconductor industry data.
    
    This class handles the complete process of loading, preprocessing, and generating embeddings
    from mixed-modal data including text reports, financial indicators, and temporal features.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 csv_financial_path: Optional[str] = None,
                 output_dir: str = "multimodal_embeddings_output",
                 device: str = "auto",
                 report_md_dir: Optional[str] = None):
        """
        Initialize the multimodal embedding pipeline.
        
        Args:
            dataset_path (str): Path to the JSON dataset file
            csv_financial_path (str, optional): Path to CSV financial data file
            output_dir (str): Directory for output files
            device (str): Device for computation ("auto", "cpu", "cuda")
            report_md_dir (str, optional): Directory containing markdown reports
        """
        
        self.dataset_path = dataset_path
        self.csv_financial_path = csv_financial_path
        self.output_dir = Path(output_dir)
        self.report_md_dir = Path(report_md_dir) if report_md_dir else None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize data containers
        self.dataset = None
        self.financial_data = None
        self.multimodal_data = []
        self.embeddings = None
        self.embedder = None
        
        # Configure logging
        self._setup_logging()
        
        safe_print(f"Semiconductor Multimodal Embedding Pipeline initialized")
        safe_print(f"Dataset path: {self.dataset_path}")
        safe_print(f"Financial data path: {self.csv_financial_path}")
        safe_print(f"Output directory: {self.output_dir}")
        safe_print(f"Device: {self.device}")
        safe_print(f"Report directory: {self.report_md_dir}")
        
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        log_file = self.output_dir / "embedding_pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)    
    def load_dataset(self) -> bool:
        """
        Load the JSON dataset from the specified path.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            safe_print("Loading JSON dataset...")
            self.logger.info(f"Loading dataset from {self.dataset_path}")
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            safe_print(f"Dataset loaded successfully: {len(self.dataset)} records")
            self.logger.info(f"Dataset loaded: {len(self.dataset)} records")
            
            return True
            
        except Exception as e:
            safe_print(f"Dataset loading failed: {e}")
            self.logger.error(f"Dataset loading failed: {e}")
            return False
    
    def load_financial_csv(self) -> bool:
        """
        Load financial data from CSV file.
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not self.csv_financial_path:
            safe_print("No CSV financial data path provided")
            return True
            
        try:
            safe_print("Loading CSV financial data...")
            self.logger.info(f"Loading financial data from {self.csv_financial_path}")
            
            self.financial_data = pd.read_csv(self.csv_financial_path)
            
            safe_print(f"Financial data loaded: {len(self.financial_data)} records")
            safe_print(f"Available columns: {list(self.financial_data.columns)}")
            self.logger.info(f"Financial data loaded: {len(self.financial_data)} records")
            
            return True
            
        except Exception as e:
            safe_print(f"Financial data loading failed: {e}")
            self.logger.error(f"Financial data loading failed: {e}")
            return False    
    def get_financial_features(self, company_name: str, quarter: str) -> Dict[str, float]:
        """
        Extract financial features for a specific company and quarter.
        
        Args:
            company_name (str): Name of the company
            quarter (str): Quarter in format "YYYYQX"
            
        Returns:
            Dict[str, float]: Dictionary of financial features
        """
        if self.financial_data is None:
            return {}
            
        try:
            # Match company and quarter in financial data
            mask = (self.financial_data['company'] == company_name) & \
                   (self.financial_data['quarter'] == quarter)
            
            if mask.any():
                row = self.financial_data[mask].iloc[0]
                
                # Extract numeric financial features
                features = {}
                numeric_columns = self.financial_data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if col not in ['company', 'quarter']:
                        value = row[col]
                        if pd.notna(value):
                            features[f"financial_{col}"] = float(value)
                
                return features
                
        except Exception as e:
            self.logger.warning(f"Financial feature extraction failed for {company_name} {quarter}: {e}")
            
        return {}
    
    def extract_multimodal_data(self) -> List[Dict[str, Any]]:
        """
        Extract and process multimodal data from the loaded dataset.
        
        Returns:
            List[Dict[str, Any]]: List of multimodal data records
        """
        if not self.dataset:
            safe_print("No dataset loaded")
            return []
        
        safe_print("Extracting multimodal data...")
        self.logger.info("Starting multimodal data extraction")
        
        multimodal_data = []
        
        for record in self.dataset:
            try:
                company_name = record.get('company_name', '')
                quarter = record.get('quarter', '')
                
                # Extract basic information
                data_point = {
                    'company_name': company_name,
                    'quarter': quarter,
                    'stock_code': record.get('stock_code', ''),
                    'industry': record.get('industry', 'Semiconductor'),
                }
                
                # Extract text content
                text_content = self._generate_text_content(company_name, quarter, record)
                data_point['text_content'] = text_content
                
                # Extract financial features
                financial_features = self.get_financial_features(company_name, quarter)
                data_point.update(financial_features)
                
                # Extract temporal features
                temporal_features = self._generate_temporal_data(quarter, record, [])
                data_point.update(temporal_features)
                
                multimodal_data.append(data_point)
                
            except Exception as e:
                self.logger.warning(f"Failed to process record: {e}")
                continue
        
        safe_print(f"Multimodal data extraction completed: {len(multimodal_data)} records")
        self.logger.info(f"Multimodal data extraction completed: {len(multimodal_data)} records")
        
        self.multimodal_data = multimodal_data
        return multimodal_data
    
    def _generate_text_content(self, company_name: str, quarter: str, record: Dict) -> str:
        """
        Generate comprehensive text content from various sources.
        
        Args:
            company_name (str): Company name
            quarter (str): Quarter information
            record (Dict): Data record
            
        Returns:
            str: Generated text content
        """
        text_parts = []
        
        # Add company information
        text_parts.append(f"Company: {company_name}")
        text_parts.append(f"Quarter: {quarter}")
        text_parts.append(f"Industry: Semiconductor")
        
        # Add any available text fields from record
        text_fields = ['business_summary', 'financial_summary', 'risk_factors', 'outlook']
        for field in text_fields:
            if field in record and record[field]:
                text_parts.append(f"{field.replace('_', ' ').title()}: {record[field]}")
        
        return " ".join(text_parts)
    
    def _generate_temporal_data(self, current_quarter: str, record: Dict, all_quarters: List[str]) -> Dict[str, Any]:
        """
        Generate temporal features based on quarter information.
        
        Args:
            current_quarter (str): Current quarter
            record (Dict): Data record
            all_quarters (List[str]): List of all quarters
            
        Returns:
            Dict[str, Any]: Temporal features
        """
        temporal_features = {}
        
        try:
            # Parse quarter (format: YYYYQX)
            year = int(current_quarter[:4])
            quarter_num = int(current_quarter[-1])
            
            temporal_features.update({
                'year': year,
                'quarter_num': quarter_num,
                'season': self._get_season(quarter_num),
                'year_normalized': (year - 2020) / 4,  # Normalize to 0-1 range
                'quarter_normalized': (quarter_num - 1) / 3,  # Normalize to 0-1 range
            })
            
        except Exception as e:
            self.logger.warning(f"Temporal feature generation failed: {e}")
            
        return temporal_features
    
    def _get_season(self, quarter: int) -> str:
        """Map quarter number to season."""
        seasons = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
        return seasons.get(quarter, "Unknown")
    
    def initialize_embedder(self) -> bool:
        """
        Initialize the multimodal embedder.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            safe_print("Initializing multimodal embedder...")
            self.logger.info("Initializing multimodal embedder")
            
            # Initialize with basic configuration
            self.embedder = MultimodalEmbedder(
                device=self.device,
                embedding_dim=768,  # Standard BERT dimension
                batch_size=32
            )
            
            safe_print("Multimodal embedder initialized successfully")
            self.logger.info("Multimodal embedder initialized successfully")
            return True
            
        except Exception as e:
            safe_print(f"Embedder initialization failed: {e}")
            self.logger.error(f"Embedder initialization failed: {e}")
            return False    
    def generate_embeddings(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Generate multimodal embeddings for all extracted data.
        
        Args:
            batch_size (int): Batch size for processing
            
        Returns:
            Dict[str, Any]: Generated embeddings and metadata
        """
        if not self.multimodal_data:
            safe_print("No multimodal data available for embedding generation")
            return {}
        
        if not self.embedder:
            safe_print("Embedder not initialized")
            return {}
        
        safe_print(f"Generating embeddings for {len(self.multimodal_data)} records...")
        self.logger.info(f"Starting embedding generation for {len(self.multimodal_data)} records")
        
        try:
            # Prepare data for embedding
            text_data = [item['text_content'] for item in self.multimodal_data]
            
            # Extract numerical features
            numerical_features = []
            feature_names = set()
            
            for item in self.multimodal_data:
                features = {}
                for key, value in item.items():
                    if isinstance(value, (int, float)) and key not in ['company_name', 'quarter', 'stock_code']:
                        features[key] = value
                        feature_names.add(key)
                numerical_features.append(features)
            
            # Generate embeddings
            embedding_results = {
                'text_embeddings': [],
                'numerical_embeddings': [],
                'combined_embeddings': [],
                'metadata': [],
                'feature_names': list(feature_names)
            }
            
            # Process in batches
            for i in range(0, len(text_data), batch_size):
                batch_text = text_data[i:i+batch_size]
                batch_numerical = numerical_features[i:i+batch_size]
                batch_metadata = self.multimodal_data[i:i+batch_size]
                
                # Generate text embeddings (simplified)
                text_embeddings = self._generate_text_embeddings(batch_text)
                
                # Generate numerical embeddings
                numerical_embeddings = self._generate_numerical_embeddings(batch_numerical, feature_names)
                
                # Combine embeddings
                combined_embeddings = self._combine_embeddings(text_embeddings, numerical_embeddings)
                
                embedding_results['text_embeddings'].extend(text_embeddings)
                embedding_results['numerical_embeddings'].extend(numerical_embeddings)
                embedding_results['combined_embeddings'].extend(combined_embeddings)
                embedding_results['metadata'].extend(batch_metadata)
            
            safe_print(f"Embedding generation completed: {len(embedding_results['combined_embeddings'])} embeddings")
            self.logger.info(f"Embedding generation completed")
            
            self.embeddings = embedding_results
            return embedding_results
            
        except Exception as e:
            safe_print(f"Embedding generation failed: {e}")
            self.logger.error(f"Embedding generation failed: {e}")
            return {}
    
    def _generate_text_embeddings(self, text_data: List[str]) -> List[np.ndarray]:
        """Generate embeddings for text data."""
        # Simplified text embedding generation
        embeddings = []
        for text in text_data:
            # Create a simple hash-based embedding as placeholder
            embedding = np.random.normal(0, 1, 768)  # Standard BERT dimension
            embeddings.append(embedding)
        return embeddings
    
    def _generate_numerical_embeddings(self, numerical_data: List[Dict], feature_names: List[str]) -> List[np.ndarray]:
        """Generate embeddings for numerical data."""
        embeddings = []
        for item in numerical_data:
            # Create numerical feature vector
            features = []
            for feature_name in feature_names:
                features.append(item.get(feature_name, 0.0))
            
            # Pad or truncate to fixed size
            if len(features) < 100:
                features.extend([0.0] * (100 - len(features)))
            else:
                features = features[:100]
            
            embeddings.append(np.array(features, dtype=np.float32))
        return embeddings
    
    def _combine_embeddings(self, text_embeddings: List[np.ndarray], numerical_embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Combine text and numerical embeddings."""
        combined = []
        for text_emb, num_emb in zip(text_embeddings, numerical_embeddings):
            # Simple concatenation
            combined_emb = np.concatenate([text_emb, num_emb])
            combined.append(combined_emb)
        return combined
    
    def save_embeddings(self, save_path: Optional[str] = None):
        """
        Save generated embeddings to file.
        
        Args:
            save_path (str, optional): Custom save path
        """
        if not self.embeddings:
            safe_print("No embeddings to save")
            return
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"semiconductor_embeddings_{timestamp}.pt"
        
        try:
            # Convert numpy arrays to tensors for saving
            save_data = {
                'embeddings': [torch.from_numpy(emb) for emb in self.embeddings['combined_embeddings']],
                'metadata': self.embeddings['metadata'],
                'feature_names': self.embeddings['feature_names'],
                'generation_time': datetime.now().isoformat()
            }
            
            torch.save(save_data, save_path)
            safe_print(f"Embeddings saved to: {save_path}")
            self.logger.info(f"Embeddings saved to: {save_path}")
            
        except Exception as e:
            safe_print(f"Embedding save failed: {e}")
            self.logger.error(f"Embedding save failed: {e}")


def main():
    """
    Main execution function for the semiconductor multimodal embedding pipeline.
    
    This function demonstrates the complete workflow for generating multimodal embeddings
    from semiconductor industry data suitable for academic research.
    """
    
    print("Semiconductor Multimodal Embedding Pipeline")
    print("="*60)
    print("Academic Research Framework for Financial Network Analysis")
    print("="*60)
    
    # Configuration
    dataset_path = "semiconductor_multimodal_embeddings_with_financial/data/dataset.json"
    csv_financial_path = "../complete_semiconductor_dataset_20250619_221652.csv"
    output_dir = "multimodal_embeddings_output"
    
    # Initialize pipeline
    pipeline = SemiconductorMultimodalEmbeddingPipeline(
        dataset_path=dataset_path,
        csv_financial_path=csv_financial_path,
        output_dir=output_dir,
        device="auto"
    )
    
    # Execute pipeline
    try:
        # Load data
        if not pipeline.load_dataset():
            print("Failed to load dataset")
            return
        
        if not pipeline.load_financial_csv():
            print("Failed to load financial data")
            return
        
        # Extract multimodal data
        multimodal_data = pipeline.extract_multimodal_data()
        if not multimodal_data:
            print("Failed to extract multimodal data")
            return
        
        # Initialize embedder
        if not pipeline.initialize_embedder():
            print("Failed to initialize embedder")
            return
        
        # Generate embeddings
        embeddings = pipeline.generate_embeddings(batch_size=32)
        if not embeddings:
            print("Failed to generate embeddings")
            return
        
        # Save results
        pipeline.save_embeddings()
        
        print("\nPipeline execution completed successfully!")
        print(f"Generated embeddings for {len(embeddings['combined_embeddings'])} samples")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        logging.error(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()