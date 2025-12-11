"""
Preprocessing - Simple & Beginner Friendly
===========================================
Simple functions for data cleaning and preprocessing.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def clean_data(df, columns_to_drop):
    """
    Clean data by removing duplicates and dropping specified columns.
    
    Args:
        df: Input DataFrame
        columns_to_drop: List of column names to drop
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Drop duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Dropped {initial_count - len(df_clean)} duplicate rows")
    
    # Drop specified columns
    cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
    
    return df_clean


def extract_date_features(df, date_columns):
    """
    Extract year, month, day of week from date columns.
    
    Args:
        df: Input DataFrame
        date_columns: List of date column names
        
    Returns:
        DataFrame with date features extracted
    """
    df = df.copy()
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df = df.drop(columns=[col])
            
    print(f"Extracted date features from: {date_columns}")
    return df


def build_preprocessor(numerical_cols, categorical_cols):
    """
    Build a preprocessing pipeline.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        ColumnTransformer preprocessor
    """
    # Numerical pipeline: impute missing with median, then scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing with most frequent, then one-hot encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    return preprocessor
