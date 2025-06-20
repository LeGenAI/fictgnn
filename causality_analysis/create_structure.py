#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory Structure Creation Script

Creates the necessary directory structure for the causality analysis module.
This script is used for academic research project initialization.

Author: Research Team
Date: 2024
"""

import os

# Directory structure to create
directories = [
    'core',
    'utils', 
    'tests',
    'data',
    'outputs',
    'scripts'
]

# Execute in current directory
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

print("\nDirectory structure creation completed.")