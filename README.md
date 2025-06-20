# FicTGNN: Financial Technology Graph Neural Networks for Causal Analysis

## Overview

FicTGNN is a comprehensive framework for analyzing causal relationships in financial networks using multimodal graph neural networks and temporal analysis techniques. This research project combines textual, financial, and industry knowledge to model complex interactions between companies in the semiconductor industry.

## Key Features

### Core Analysis Capabilities
- **Industry Knowledge-Based Graph Construction**: Leverages verified supply chain relationships and company characteristics
- **Temporal Causal Relationship Analysis**: Models influence propagation with temporal delays
- **Multimodal Embedding Fusion**: Balanced combination of text (50%) + financial (30%) + industry knowledge (20%)
- **Adaptive Dimensionality Reduction**: Dynamic dimension selection based on variance maximization

### Visualization and Analysis
- **Year-by-Year Network Evolution**: Tracks temporal changes from 2020-2024
- **Company-Centric Analysis**: Visualizes individual company influence propagation paths
- **Interactive Dashboard**: Real-time filtering and exploration capabilities

## Project Structure

```
fictgnn/
├── data_collection/           # Data collection and preprocessing
│   └── complete_semiconductor_dataset_collector.py
├── embedding_generation/      # Multimodal embedding generation
│   └── semiconductor_multimodal_embedder.py
├── causality_analysis/        # Graph neural network analysis
│   ├── config/               # Configuration management
│   ├── core/                 # Core analysis logic
│   ├── utils/                # Utility modules
│   ├── tests/                # Test modules
│   ├── data/                 # Analysis data storage
│   ├── outputs/              # Results output
│   ├── scripts/              # Auxiliary scripts
│   └── main.py               # Main execution file
├── experiments/              # Experimental configurations
│   ├── semiconductor_multimodal_embeddings_with_financial/
│   ├── data/
│   ├── semiconductor_embeddings/
│   └── modules/
├── data/                     # Raw dataset storage
│   └── complete_semiconductor_dataset_20250619_221652.csv
├── utils/                    # Shared utilities
└── README.md                 # Project documentation
```

## Installation

### System Requirements
- Python 3.8+
- Memory: 8GB+ recommended
- Storage: 5GB+

### Dependencies
```bash
pip install torch torch-geometric
pip install numpy pandas scikit-learn
pip install plotly dash
pip install psutil pyyaml
```

### PyTorch Geometric Installation (for GPU usage)
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

## Usage

### Basic Execution
```bash
cd fictgnn/causality_analysis
python main.py
```

### Advanced Options
```bash
# Use custom configuration file
python main.py --config my_settings.yaml

# Specify output directory
python main.py --output ./results

# Configuration validation only
python main.py --validate-only

# Verbose logging
python main.py --verbose

# Minimal logging
python main.py --quiet
```

### Configuration Customization

Modify the `config/settings.yaml` file to adjust analysis options:

```yaml
# Model configuration
model:
  input_dim: 28
  hidden_dim: 256
  num_heads: 8

# Embedding configuration
embedding:
  target_variance: 0.95
  text_weight: 0.5
  financial_weight: 0.3

# Graph configuration
graph:
  base_threshold: 0.6
  max_lag: 4
```

## Output Results

After execution, the following files will be generated:

### Analysis Results
- `causality_analysis_results.json`: Causal relationship analysis results
- `performance_metrics.json`: Performance metrics
- `temporal_graph_data.pt`: Graph data

### Visualization Files
- `semiconductor_evolution_2020_2024.html`: Year-by-year network evolution
- `company_focused_networks_2020_2024.html`: Company-centric networks

### Log Files
- `semiconductor_analysis.log`: Main log
- `performance.log`: Performance log
- `error.log`: Error log

## Module Architecture

### Data Collection
- **Collector Framework**: Automated financial data collection from multiple sources
- **Data Validation**: Comprehensive data quality checks and preprocessing
- **Batch Processing**: Efficient handling of large-scale data collection

### Embedding Generation
- **Multimodal Processing**: Integration of textual, financial, and temporal features
- **Text Analysis**: Advanced NLP techniques for financial document processing
- **Feature Engineering**: Sophisticated financial indicator calculation

### Causality Analysis
- **Graph Construction**: Industry knowledge-based graph building
- **GNN Model**: Advanced graph neural network architectures
- **Temporal Analysis**: Time-aware causal relationship detection

## Performance Optimization

### Memory Optimization
- **Chunked Processing**: Batch-wise handling of large datasets
- **Garbage Collection**: Periodic memory cleanup
- **Sparse Matrices**: Memory-efficient data structures

### Speed Optimization
- **Parallel Processing**: Multiprocessing support
- **Caching**: Intermediate result storage
- **JIT Compilation**: NumPy operation optimization

## Testing

### Basic Test Execution
```bash
python -m pytest tests/
```

### Configuration Validation
```bash
python main.py --validate-only
```

### Memory Profiling
```bash
python -m memory_profiler main.py
```

## Research Applications

This framework is designed for academic research in:
- Financial network analysis
- Causal inference in economic systems
- Graph neural network applications in finance
- Multimodal learning for financial data

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{fictgnn2025,
  title={FicTGNN: Financial Technology Graph Neural Networks for Causal Analysis},
  author={Research Team},
  year={2025},
  howpublished={\url{https://github.com/LeGenAI/fictgnn}}
}
```

## License

This project is distributed under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- PyTorch Geometric team for excellent graph neural network library
- Plotly team for interactive visualization tools
- Financial data providers

---

**Version**: 1.0.0  
**Last Updated**: 2025-06-20