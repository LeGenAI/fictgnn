#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FicTGNN: Financial Technology Graph Neural Networks for Causal Analysis

Setup script for academic research framework.
"""

from setuptools import setup, find_packages
import pathlib

# Read README for long description
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')

setup(
    name="fictgnn",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Financial Technology Graph Neural Networks for Causal Analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/username/fictgnn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "requests>=2.25.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "networkx>=2.6.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch-scatter>=2.0.9",
            "torch-sparse>=0.6.12",
            "torch-cluster>=1.5.9",
            "torch-spline-conv>=1.2.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "fictgnn-analyze=fictgnn.causality_analysis.main:main",
            "fictgnn-collect=fictgnn.data_collection.complete_semiconductor_dataset_collector:main",
            "fictgnn-embed=fictgnn.embedding_generation.semiconductor_multimodal_embedder:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fictgnn": ["config/*.yaml", "data/*.csv"],
    },
    keywords="graph neural networks, financial analysis, causal inference, academic research",
)