#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitoring Utilities for Semiconductor Industry Analysis

Comprehensive performance monitoring and benchmarking utilities for academic research
in financial network analysis and graph neural network training.

Author: Research Team
Date: 2024
Version: 2.0 (Modularized)
"""

import time
import logging
import functools
import platform
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from .data_utils import MemoryOptimizer


class PerformanceMonitor:
    """Performance monitoring utility for research operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation to time
        """
        self.start_times[operation] = time.time()
        memory_before = MemoryOptimizer.get_memory_usage()
        self.metrics[operation] = {'memory_before': memory_before}
        
        logging.info(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> Dict[str, Any]:
        """
        End timing and record performance metrics.
        
        Args:
            operation: Name of the operation to stop timing
            
        Returns:
            Performance metrics dictionary
        """
        if operation not in self.start_times:
            raise ValueError(f"Timer not started for operation: {operation}")
        
        end_time = time.time()
        duration = end_time - self.start_times[operation]
        memory_after = MemoryOptimizer.get_memory_usage()
        
        # Calculate memory delta
        memory_before = self.metrics[operation]['memory_before']
        memory_delta = memory_after['rss'] - memory_before['rss']
        
        metrics = {
            'duration': duration,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_delta': memory_delta,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics[operation].update(metrics)
        
        logging.info(f"Completed: {operation} (Duration: {duration:.3f}s, Memory: {memory_delta:+.1f}MB)")
        
        # Clean up start time
        del self.start_times[operation]
        
        return metrics
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recorded metrics.
        
        Args:
            operation: Specific operation name, or None for all metrics
            
        Returns:
            Metrics dictionary
        """
        if operation:
            return self.metrics.get(operation, {})
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all recorded metrics."""
        self.metrics.clear()
        self.start_times.clear()
        logging.info("Performance metrics reset")


class ResourceMonitor:
    """System resource monitoring utilities."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            System information dictionary
        """
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            'disk_usage': {
                'total': psutil.disk_usage('/').total / 1024 / 1024 / 1024,  # GB
                'free': psutil.disk_usage('/').free / 1024 / 1024 / 1024,  # GB
            }
        }
    
    @staticmethod
    def log_system_info() -> None:
        """Log system information for research reproducibility."""
        info = ResourceMonitor.get_system_info()
        logging.info("System Information:")
        for key, value in info.items():
            if isinstance(value, dict):
                logging.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logging.info(f"    {sub_key}: {sub_value}")
            else:
                logging.info(f"  {key}: {value}")


def performance_timer(func: Callable) -> Callable:
    """
    Decorator for automatic performance timing of functions.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with performance timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        memory_before = MemoryOptimizer.get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logging.error(f"Function {func.__name__} failed: {e}")
            raise
        finally:
            end_time = time.time()
            memory_after = MemoryOptimizer.get_memory_usage()
            duration = end_time - start_time
            memory_delta = memory_after['rss'] - memory_before['rss']
            
            status = "SUCCESS" if success else "FAILED"
            logging.info(f"Performance [{status}] {func.__name__}: "
                        f"{duration:.3f}s, Memory: {memory_delta:+.1f}MB")
        
        return result
    
    return wrapper


class Benchmark:
    """Benchmarking utilities for comparing algorithm performance."""
    
    def __init__(self):
        self.results = {}
    
    def run_benchmark(self, name: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Run a single benchmark test.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        monitor = PerformanceMonitor()
        monitor.start_timer(name)
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logging.error(f"Benchmark {name} failed: {e}")
        
        metrics = monitor.end_timer(name)
        metrics['success'] = success
        metrics['result_size'] = len(str(result)) if result is not None else 0
        
        self.results[name] = metrics
        return metrics
    
    def compare_benchmarks(self, benchmarks: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Compare multiple benchmark functions.
        
        Args:
            benchmarks: Dictionary of {name: function} pairs
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for name, func in benchmarks.items():
            comparison[name] = self.run_benchmark(name, func)
        
        # Generate comparison summary
        fastest = min(comparison.keys(), key=lambda x: comparison[x]['duration'])
        most_memory_efficient = min(comparison.keys(), 
                                   key=lambda x: comparison[x]['memory_delta'])
        
        summary = {
            'fastest': fastest,
            'most_memory_efficient': most_memory_efficient,
            'results': comparison
        }
        
        logging.info(f"Benchmark Summary - Fastest: {fastest}, "
                    f"Most Memory Efficient: {most_memory_efficient}")
        
        return summary