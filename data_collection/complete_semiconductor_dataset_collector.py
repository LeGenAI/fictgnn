#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Semiconductor Dataset Collector: 168 Companies × 20 Quarters × 22 Financial Indicators

A comprehensive data collection framework for gathering financial data from semiconductor companies
for academic research in financial network analysis.
"""

import requests
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile
import io
import time
from datetime import datetime
import json
import os
from fixed_dart_collector import FixedDARTCollector

class CompleteSemiconductorDataCollector(FixedDARTCollector):
    """
    A specialized data collector for comprehensive semiconductor industry financial data.
    
    This class extends the FixedDARTCollector to systematically collect financial data
    from multiple semiconductor companies across multiple quarters for research purposes.
    """
    
    def __init__(self, api_key):
        """
        Initialize the semiconductor data collector.
        
        Args:
            api_key (str): API key for data access
        """
        super().__init__(api_key)
        self.companies = []
        self.quarters = []
        
    def load_company_quarter_matrix(self, matrix_file):
        """
        Load the company-quarter matrix file.
        
        Args:
            matrix_file (str): Path to the matrix file containing company-quarter relationships
            
        Returns:
            pd.DataFrame or None: Loaded matrix data or None if failed
        """
        print("Loading company-quarter matrix...")
        
        try:
            df_matrix = pd.read_csv(matrix_file)
            
            # Extract quarter list (excluding first column)
            self.quarters = df_matrix['quarter'].tolist()
            
            # Extract company list (excluding first column)  
            self.companies = [col for col in df_matrix.columns if col != 'quarter']
            
            print(f"Matrix loading completed:")
            print(f"  Quarters: {len(self.quarters)} items")
            print(f"  Companies: {len(self.companies)} items") 
            print(f"  Total data points: {len(self.quarters) * len(self.companies):,} items")
            
            return df_matrix
            
        except Exception as e:
            print(f"Matrix loading failed: {e}")
            return None    
    def generate_all_company_quarter_combinations(self):
        """
        Generate all possible company-quarter combinations for comprehensive data collection.
        
        Returns:
            list: List of dictionaries containing company-quarter combinations
        """
        combinations = []
        
        for quarter in self.quarters:
            for company in self.companies:
                combinations.append({
                    'company': company,
                    'quarter': quarter
                })
        
        return combinations
    
    def collect_complete_dataset(self, matrix_file, batch_size=50):
        """
        Collect comprehensive dataset for all company-quarter combinations.
        
        Args:
            matrix_file (str): Path to the company-quarter matrix file
            batch_size (int): Number of requests per batch for efficient processing
            
        Returns:
            list: Collected financial data for all combinations
        """
        
        print("Starting comprehensive semiconductor dataset collection!")
        print("="*80)
        
        # Load matrix
        df_matrix = self.load_company_quarter_matrix(matrix_file)
        if df_matrix is None:
            return None
        
        # Load company codes
        self.get_corp_codes()
        
        # Generate all combinations
        all_combinations = self.generate_all_company_quarter_combinations()
        total_combinations = len(all_combinations)
        
        print(f"\nCollection targets:")
        print(f"  Total combinations: {total_combinations:,} items")
        print(f"  Companies: {len(self.companies)} items")
        print(f"  Quarters: {len(self.quarters)} items")
        print(f"  Batch size: {batch_size} items")
        
        # Data collection
        collected_data = []
        success_count = 0
        
        for i in range(0, total_combinations, batch_size):
            batch = all_combinations[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_combinations + batch_size - 1) // batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches}...")
            print(f"  Range: {i+1}~{min(i+batch_size, total_combinations)}")
            
            batch_success = 0
            
            for combination in batch:
                company = combination['company']
                quarter = combination['quarter']
                
                try:
                    # Collect financial data
                    financial_data = self.get_financial_data(company, quarter)
                    
                    if financial_data:
                        financial_data['company'] = company
                        financial_data['quarter'] = quarter
                        financial_data['collection_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        collected_data.append(financial_data)
                        success_count += 1
                        batch_success += 1
                        
                        # Progress indicator
                        if success_count % 10 == 0:
                            print(f"    Progress: {success_count}/{total_combinations} completed")
                    
                    # Prevent API rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"    Error {company} {quarter}: {str(e)[:50]}...")
                    continue
            
            print(f"  Batch success rate: {batch_success}/{len(batch)} ({batch_success/len(batch)*100:.1f}%)")
            print(f"  Overall success rate: {success_count}/{i+len(batch)} ({success_count/(i+len(batch))*100:.1f}%)")
            
            # Intermediate saving (every 2 batches)
            if batch_num % 2 == 0 and collected_data:
                self.save_intermediate_results(collected_data, batch_num)
        
        print(f"\nData collection completed!")
        print(f"  Total success: {success_count}/{total_combinations}")
        print(f"  Final success rate: {success_count/total_combinations*100:.1f}%")
        
        return collected_data    
    def save_intermediate_results(self, data, batch_num):
        """
        Save intermediate results during collection process.
        
        Args:
            data (list): Collected data to save
            batch_num (int): Current batch number for file naming
        """
        if not data:
            return
            
        try:
            df = pd.DataFrame(data)
            filename = f"intermediate_semiconductor_data_batch_{batch_num}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"    Intermediate save: {filename} ({len(df)} items)")
        except Exception as e:
            print(f"    Intermediate save failed: {e}")
    
    def save_complete_dataset(self, collected_data, filename=None):
        """
        Save the complete collected dataset to CSV file.
        
        Args:
            collected_data (list): Complete collected data
            filename (str, optional): Custom filename for output
            
        Returns:
            pd.DataFrame or None: Saved dataframe or None if failed
        """
        if not collected_data:
            print("No data to save.")
            return None
        
        try:
            df = pd.DataFrame(collected_data)
            
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"complete_semiconductor_dataset_{timestamp}.csv"
            
            # Save CSV
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"\nComplete dataset saved successfully!")
            print(f"  Filename: {filename}")
            print(f"  Size: {len(df):,} rows × {len(df.columns)} columns")
            print(f"  Companies: {df['company'].nunique()} items")
            print(f"  Quarters: {df['quarter'].nunique()} items")
            
            # Basic statistics
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_columns) > 0:
                print(f"\nBasic statistics:")
                for col in numeric_columns[:3]:  # Show first 3 numeric columns
                    mean_val = df[col].mean()
                    if not pd.isna(mean_val):
                        print(f"  Average {col}: {mean_val:,.2f}")
            
            return df
            
        except Exception as e:
            print(f"Dataset save failed: {e}")
            return None


def main():
    """
    Main execution function for comprehensive data collection.
    
    This function orchestrates the complete data collection process for
    semiconductor industry financial data suitable for academic research.
    """
    
    print("Complete Semiconductor Dataset Collector")
    print("="*60)
    print("Target: 168 Companies × 20 Quarters × 22 Financial Indicators")
    print("Period: 2020Q1 ~ 2024Q4")
    print("Metrics: Revenue, Operating Income, ROE, Debt Ratio, etc.")
    print("="*60)
    
    # API Key
    API_KEY = "OPEN_DART_API_KEY"
    
    # Initialize collector
    collector = CompleteSemiconductorDataCollector(API_KEY)
    
    # Matrix file path
    matrix_file = "catgnn_experiment/quarter_revenue_matrix.csv"
    
    if not os.path.exists(matrix_file):
        print(f"Matrix file not found: {matrix_file}")
        return
    
    # Collect complete dataset
    collected_data = collector.collect_complete_dataset(matrix_file, batch_size=30)
    
    if collected_data:
        # Final save
        df = collector.save_complete_dataset(collected_data)
        
        if df is not None:
            print(f"\nCollection completed successfully!")
            print(f"Complete dataset for Korean semiconductor industry is now ready for analysis!")
            
            # Show collected columns
            print(f"\nCollected financial indicators ({len(df.columns)} items):")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i:2d}. {col}")
        else:
            print("\nDataset save failed.")
    else:
        print("\nData collection failed.")


if __name__ == "__main__":
    main()