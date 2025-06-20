# -*- coding: utf-8 -*-
"""
Industry-Aware Causality Network for Semiconductor Companies

A comprehensive framework for analyzing causal relationships in the semiconductor industry
using temporal graph neural networks with integrated industry knowledge.

This main execution file orchestrates the complete analysis pipeline for academic research
in financial network analysis and causal inference.

Author: Research Team
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Conditional imports (try absolute imports first)
try:
    from config.industry_knowledge import SEMICONDUCTOR_INDUSTRY_KNOWLEDGE
    from core.graph_builder import IndustryAwareTemporalGraphBuilder
    from core.model import CausalityAwareTemporalGNN
    from core.analyzer import CausalityAnalyzer
    from utils.visualizer import EnhancedTemporalGraphVisualizer
    from utils.data_loader import load_data
    from utils.helper_functions import convert_numpy_types
except ImportError:
    # Try relative imports as fallback
    try:
        from .config.industry_knowledge import SEMICONDUCTOR_INDUSTRY_KNOWLEDGE
        from .core.graph_builder import IndustryAwareTemporalGraphBuilder
        from .core.model import CausalityAwareTemporalGNN
        from .core.analyzer import CausalityAnalyzer
        from .utils.visualizer import EnhancedTemporalGraphVisualizer
        from .utils.data_loader import load_data
        from .utils.helper_functions import convert_numpy_types
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are properly installed and accessible.")
        raise


def main():
    """
    Main execution function for the semiconductor industry causal analysis pipeline.
    
    This function orchestrates the complete workflow including:
    1. Data loading and preprocessing
    2. Industry-aware graph construction
    3. Temporal GNN model training
    4. Causal relationship analysis
    5. Results visualization and export
    """
    
    print("Industry-Aware Causality Network Analysis")
    print("="*60)
    print("Academic Research Framework for Semiconductor Industry")
    print("Temporal Graph Neural Networks with Industry Knowledge")
    print("="*60)
    
    try:
        # Configuration
        config = {
            'data_path': 'data/semiconductor_dataset.csv',
            'output_dir': 'outputs',
            'model_params': {
                'input_dim': 28,
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 3
            },
            'analysis_params': {
                'temporal_window': 4,
                'causality_threshold': 0.6,
                'significance_level': 0.05
            }
        }
        
        # Ensure output directory exists
        Path(config['output_dir']).mkdir(exist_ok=True)
        
        # Step 1: Data Loading
        print("\n1. Loading and preprocessing data...")
        data = load_data(config['data_path'])
        if data is None:
            print("Failed to load data. Please check the data path.")
            return
        
        print(f"   Loaded data: {len(data)} records")
        print(f"   Time range: {data['quarter'].min()} to {data['quarter'].max()}")
        
        # Step 2: Graph Construction
        print("\n2. Constructing industry-aware temporal graph...")
        graph_builder = IndustryAwareTemporalGraphBuilder(
            industry_knowledge=SEMICONDUCTOR_INDUSTRY_KNOWLEDGE
        )
        
        temporal_graphs = graph_builder.build_temporal_graphs(data)
        print(f"   Generated {len(temporal_graphs)} temporal graphs")
        
        # Step 3: Model Training
        print("\n3. Training causality-aware temporal GNN...")
        model = CausalityAwareTemporalGNN(
            input_dim=config['model_params']['input_dim'],
            hidden_dim=config['model_params']['hidden_dim'],
            num_heads=config['model_params']['num_heads'],
            num_layers=config['model_params']['num_layers']
        )
        
        # Train model on temporal graphs
        training_results = model.train_on_temporal_data(temporal_graphs)
        print(f"   Training completed. Final loss: {training_results['final_loss']:.4f}")
        
        # Step 4: Causal Analysis
        print("\n4. Performing causal relationship analysis...")
        analyzer = CausalityAnalyzer(
            model=model,
            temporal_window=config['analysis_params']['temporal_window'],
            threshold=config['analysis_params']['causality_threshold']
        )
        
        causality_results = analyzer.analyze_causality(temporal_graphs)
        print(f"   Identified {len(causality_results['causal_relationships'])} significant causal relationships")
        
        # Step 5: Visualization
        print("\n5. Generating visualizations...")
        visualizer = EnhancedTemporalGraphVisualizer(
            output_dir=config['output_dir']
        )
        
        # Generate network evolution visualization
        visualizer.create_temporal_network_evolution(
            temporal_graphs, 
            causality_results,
            title="Semiconductor Industry Causal Network Evolution 2020-2024"
        )
        
        # Generate company-focused analysis
        visualizer.create_company_focused_analysis(
            temporal_graphs,
            causality_results,
            focus_companies=['Samsung', 'TSMC', 'Intel']
        )
        
        # Step 6: Export Results
        print("\n6. Exporting results...")
        
        # Export causality analysis results
        results_file = Path(config['output_dir']) / 'causality_analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(causality_results), f, indent=2, ensure_ascii=False)
        
        # Export model performance metrics
        metrics_file = Path(config['output_dir']) / 'performance_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(training_results), f, indent=2, ensure_ascii=False)
        
        # Export temporal graph data
        graph_file = Path(config['output_dir']) / 'temporal_graph_data.pt'
        torch.save(temporal_graphs, graph_file)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {config['output_dir']}")
        print("\nGenerated files:")
        print(f"  - causality_analysis_results.json")
        print(f"  - performance_metrics.json")
        print(f"  - temporal_graph_data.pt")
        print(f"  - semiconductor_evolution_2020_2024.html")
        print(f"  - company_focused_networks_2020_2024.html")
        
        # Summary statistics
        print(f"\nAnalysis Summary:")
        print(f"  Companies analyzed: {len(set(data['company']))}")
        print(f"  Time periods: {len(set(data['quarter']))}")
        print(f"  Causal relationships identified: {len(causality_results['causal_relationships'])}")
        print(f"  Model accuracy: {training_results.get('accuracy', 'N/A')}")
        print(f"  Analysis confidence: {causality_results.get('confidence_score', 'N/A')}")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with exception: {e}")
        print("Please check the configuration and ensure all required data files are available.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()