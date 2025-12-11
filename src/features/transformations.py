"""
Custom Transformations Module
=============================
Contains custom sklearn transformers and preprocessing functions.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract features from date columns.
    """
    
    def __init__(self, date_columns: List[str], features: List[str] = None):
        """
        Initialize DateFeatureExtractor.
        
        Args:
            date_columns: List of date column names
            features: List of features to extract ['month', 'year', 'dayofweek', 'quarter', 'is_weekend']
        """
        self.date_columns = date_columns
        self.features = features or ['month', 'year', 'dayofweek']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.date_columns:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='coerce')
                
                if 'month' in self.features:
                    X[f'{col}_month'] = X[col].dt.month
                if 'year' in self.features:
                    X[f'{col}_year'] = X[col].dt.year
                if 'dayofweek' in self.features:
                    X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                if 'quarter' in self.features:
                    X[f'{col}_quarter'] = X[col].dt.quarter
                if 'is_weekend' in self.features:
                    X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                if 'day' in self.features:
                    X[f'{col}_day'] = X[col].dt.day
                    
                # Drop original date column
                X = X.drop(columns=[col])
                
        return X


class IncomeBinner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to bin income into categories.
    """
    
    def __init__(
        self,
        column: str = 'annual_income',
        bins: List[float] = None,
        labels: List[str] = None
    ):
        self.column = column
        self.bins = bins or [0, 500000, 1000000, 2500000, 5000000, float('inf')]
        self.labels = labels or ['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[f'{self.column}_category'] = pd.cut(
                X[self.column],
                bins=self.bins,
                labels=self.labels,
                include_lowest=True
            )
        return X


class AgeBinner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to bin age into categories.
    """
    
    def __init__(
        self,
        column: str = 'age',
        bins: List[int] = None,
        labels: List[str] = None
    ):
        self.column = column
        self.bins = bins or [18, 25, 35, 45, 55, 65, 100]
        self.labels = labels or ['Young', 'Young-Adult', 'Adult', 'Middle-Aged', 'Senior', 'Elderly']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[f'{self.column}_category'] = pd.cut(
                X[self.column],
                bins=self.bins,
                labels=self.labels,
                include_lowest=True
            )
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns.
    """
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        cols_to_drop = [col for col in self.columns if col in X.columns]
        return X.drop(columns=cols_to_drop)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle outliers using IQR or Z-score method.
    """
    
    def __init__(
        self,
        columns: List[str] = None,
        method: str = 'iqr',
        factor: float = 1.5
    ):
        self.columns = columns
        self.method = method
        self.factor = factor
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        X = X.copy()
        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in cols:
            if col in X.columns:
                if self.method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.bounds_[col] = (
                        Q1 - self.factor * IQR,
                        Q3 + self.factor * IQR
                    )
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    self.bounds_[col] = (
                        mean - self.factor * std,
                        mean + self.factor * std
                    )
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating derived features.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Credit utilization to loan ratio
        if 'credit_utilization_ratio' in X.columns and 'existing_loans_count' in X.columns:
            X['credit_per_loan'] = X['credit_utilization_ratio'] / (X['existing_loans_count'] + 1)
        
        # Income to EMI ratio
        if 'annual_income' in X.columns and 'existing_monthly_emi' in X.columns:
            X['income_emi_ratio'] = X['annual_income'] / (X['existing_monthly_emi'] * 12 + 1)
        
        # Balance to income ratio
        if 'avg_monthly_balance' in X.columns and 'annual_income' in X.columns:
            X['balance_income_ratio'] = X['avg_monthly_balance'] * 12 / (X['annual_income'] + 1)
            
        # Credit card spend to income ratio
        if 'credit_card_spend_last_6m' in X.columns and 'annual_income' in X.columns:
            X['spend_income_ratio'] = X['credit_card_spend_last_6m'] * 2 / (X['annual_income'] + 1)
        
        return X


def build_preprocessor(config: dict, numerical_cols: List[str], categorical_cols: List[str]):
    """
    Build the preprocessing pipeline based on configuration.
    
    Args:
        config: Preprocessing configuration
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        ColumnTransformer preprocessor
    """
    preproc_config = config.get('preprocessing', {})
    num_config = preproc_config.get('numerical', {})
    cat_config = preproc_config.get('categorical', {})
    
    # Numerical pipeline
    num_imputer = SimpleImputer(strategy=num_config.get('imputer_strategy', 'median'))
    
    scaler_type = num_config.get('scaler', 'standard')
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    num_pipeline = Pipeline(steps=[
        ('imputer', num_imputer),
        ('scaler', scaler)
    ])
    
    # Categorical pipeline
    cat_imputer = SimpleImputer(strategy=cat_config.get('imputer_strategy', 'most_frequent'))
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', cat_imputer),
        ('encoder', encoder)
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def get_feature_names(preprocessor, numerical_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_cols: Original numerical column names
        categorical_cols: Original categorical column names
        
    Returns:
        List of feature names
    """
    try:
        cat_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
        return list(numerical_cols) + list(cat_features)
    except Exception as e:
        logger.warning(f"Could not get feature names: {e}")
        return None
