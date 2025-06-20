#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Utilities for Semiconductor Industry Analysis

Comprehensive logging utilities for academic research in financial network analysis,
providing structured logging, performance tracking, and research reproducibility.

Author: Research Team
Date: 2024
Version: 2.0 (Modularized)
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class LoggerSetup:
    """Logging configuration and management utilities."""
    
    @staticmethod
    def setup_logger(name: str = 'semiconductor_analysis',
                    level: str = 'INFO',
                    log_format: Optional[str] = None,
                    filename: Optional[str] = None,
                    max_bytes: int = 10*1024*1024,  # 10MB
                    backup_count: int = 5,
                    console_output: bool = True) -> logging.Logger:
        """
        Set up structured logger for research operations.
        
        Args:
            name: Logger name
            level: Logging level
            log_format: Log format string
            filename: Log file name
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            console_output: Whether to output to console
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Default format for academic research
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        formatter = logging.Formatter(log_format)
        
        # File handler with rotation
        if filename:
            file_handler = logging.handlers.RotatingFileHandler(
                filename, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger


class StructuredLogger:
    """Structured logging for research operations and metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize structured logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_operation_start(self, operation: str, **kwargs) -> None:
        """
        Log the start of a research operation.
        
        Args:
            operation: Operation name
            **kwargs: Additional operation parameters
        """
        params_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"OPERATION_START: {operation} ({params_str})")
    
    def log_operation_end(self, operation: str, duration: float, **kwargs) -> None:
        """
        Log the completion of a research operation.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            **kwargs: Additional result parameters
        """
        params_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"OPERATION_END: {operation} (duration={duration:.3f}s, {params_str})")
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", **kwargs) -> None:
        """
        Log a research metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            **kwargs: Additional metric metadata
        """
        metadata_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"METRIC: {metric_name}={value}{unit} ({metadata_str})")
    
    def log_progress(self, current: int, total: int, operation: str = "") -> None:
        """
        Log progress of a long-running operation.
        
        Args:
            current: Current progress count
            total: Total expected count
            operation: Operation description
        """
        percentage = (current / total) * 100 if total > 0 else 0
        op_str = f" ({operation})" if operation else ""
        self.logger.info(f"PROGRESS: {current}/{total} ({percentage:.1f}%){op_str}")
    
    def log_data_summary(self, data_name: str, shape: tuple, **stats) -> None:
        """
        Log summary statistics for research data.
        
        Args:
            data_name: Name of the dataset
            shape: Data shape (rows, columns)
            **stats: Additional statistics
        """
        stats_str = ', '.join([f"{k}={v}" for k, v in stats.items()])
        self.logger.info(f"DATA_SUMMARY: {data_name} shape={shape} ({stats_str})")
    
    def log_model_info(self, model_name: str, parameters: int, **config) -> None:
        """
        Log model configuration and information.
        
        Args:
            model_name: Name of the model
            parameters: Number of parameters
            **config: Model configuration
        """
        config_str = ', '.join([f"{k}={v}" for k, v in config.items()])
        self.logger.info(f"MODEL_INFO: {model_name} params={parameters:,} ({config_str})")
    
    def log_error_with_context(self, error: Exception, operation: str, **context) -> None:
        """
        Log error with detailed context for debugging.
        
        Args:
            error: Exception that occurred
            operation: Operation that failed
            **context: Additional context information
        """
        context_str = ', '.join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(f"ERROR: {operation} failed - {str(error)} ({context_str})")


class ResearchLogger:
    """High-level logger for academic research workflows."""
    
    def __init__(self, name: str = 'research', output_dir: str = 'logs'):
        """
        Initialize research logger.
        
        Args:
            name: Logger name
            output_dir: Directory for log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        log_file = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = LoggerSetup.setup_logger(
            name=name,
            filename=str(log_file),
            console_output=True
        )
        
        # Setup structured logger
        self.structured = StructuredLogger(self.logger)
        
        # Log initialization
        self.logger.info(f"Research logger initialized: {name}")
        self.logger.info(f"Log file: {log_file}")
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """
        Log the start of a research experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
        """
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT START: {experiment_name}")
        self.logger.info("="*60)
        
        for key, value in config.items():
            self.logger.info(f"CONFIG: {key} = {value}")
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """
        Log the completion of a research experiment.
        
        Args:
            experiment_name: Name of the experiment
            results: Experiment results
        """
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT END: {experiment_name}")
        self.logger.info("="*60)
        
        for key, value in results.items():
            self.logger.info(f"RESULT: {key} = {value}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger
    
    def get_structured(self) -> StructuredLogger:
        """Get the structured logger instance."""
        return self.structured