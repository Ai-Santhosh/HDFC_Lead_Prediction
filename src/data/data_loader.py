"""
Data Loader - Simple & Beginner Friendly
=========================================
Loads data from CSV file.
"""

import pandas as pd


def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df
