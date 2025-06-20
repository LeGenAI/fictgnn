# Semiconductor Industry Temporal Graph Neural Network Analysis System v2.0

## Overview

A temporal graph neural network system for analyzing causal relationships between companies in the semiconductor industry. This system integrates text, financial, and industry knowledge to model complex interactions between enterprises.

## Key Features

### Core Analysis Capabilities
- **Industry Knowledge-Based Graph Construction**: Reflects verified supply chain relationships and company characteristics
- **Temporal Causal Relationship Analysis**: Models influence propagation considering temporal delays
- **Multidimensional Embedding Fusion**: Balanced combination of text (50%) + financial (30%) + industry knowledge (20%)
- **Adaptive Dimensionality Reduction**: Dynamic dimension selection based on variance maximization

### Visualization and Analysis
- **Year-by-Year Network Evolution**: Tracks temporal changes from 2020-2024
- **Company-Centric Analysis**: Visualizes individual company influence propagation paths
- **Interactive Dashboard**: Real-time filtering and exploration capabilities

## System Architecture

```
causality_analysis/
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── config.py          # YAML-supported configuration class
│   └── settings.yaml      # Default configuration file
├── core/                   # Core analysis logic
│   ├── __init__.py
│   ├── graph_builder.py   # Graph construction
│   ├── model.py          # GNN model
│   └── analyzer.py       # Causal relationship analysis
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── data_utils.py     # Data processing
│   ├── performance.py    # Performance monitoring
│   ├── visualization.py  # Visualization
│   └── logging_utils.py  # Logging system
├── tests/                  # Test modules
│   └── __init__.py
├── data/                   # Data storage
├── outputs/                # Result output
├── scripts/                # Auxiliary scripts
├── main.py                # Main execution file
└── README.md              # Documentation
```# Semiconductor Industry Temporal Graph Neural Network Analysis System v2.0

## Overview

A temporal graph neural network system for analyzing causal relationships between companies in the semiconductor industry. This system integrates text, financial, and industry knowledge to model complex interactions between enterprises.

## Key Features

### Core Analysis Capabilities
- **Industry Knowledge-Based Graph Construction**: Reflects verified supply chain relationships and company characteristics
- **Temporal Causal Relationship Analysis**: Models influence propagation considering temporal delays
- **Multidimensional Embedding Fusion**: Balanced combination of text (50%) + financial (30%) + industry knowledge (20%)
- **Adaptive Dimensionality Reduction**: Dynamic dimension selection based on variance maximization

### Visualization and Analysis
- **Year-by-Year Network Evolution**: Tracks temporal changes from 2020-2024
- **Company-Centric Analysis**: Visualizes individual company influence propagation paths
- **Interactive Dashboard**: Real-time filtering and exploration capabilities

## System Architecture

```
causality_analysis/
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── config.py          # YAML-supported configuration class
│   └── settings.yaml      # Default configuration file
├── core/                   # Core analysis logic
│   ├── __init__.py
│   ├── graph_builder.py   # Graph construction
│   ├── model.py          # GNN model
│   └── analyzer.py       # Causal relationship analysis
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── data_utils.py     # Data processing
│   ├── performance.py    # Performance monitoring
│   ├── visualization.py  # Visualization
│   └── logging_utils.py  # Logging system
├── tests/                  # Test modules
│   └── __init__.py
├── data/                   # Data storage
├── outputs/                # Result output
├── scripts/                # Auxiliary scripts
├── main.py                # Main execution file
└── README.md              # Documentation
```

## Installation and Requirements

### System Requirements
- Python 3.8+
- Memory: 8GB+ recommended
- Storage: 5GB+

### Dependencies Installation
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