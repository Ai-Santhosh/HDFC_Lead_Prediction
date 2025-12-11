"""
Model Trainer - Simple & Beginner Friendly
===========================================
Trains a Random Forest model and logs to MLflow.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def train_model(X_train, y_train, model_params):
    """
    Train a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_params: Dictionary of model parameters
        
    Returns:
        Trained model
    """
    print("Training Random Forest model...")
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_model(model, X, y, dataset_name=""):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        dataset_name: Name of dataset (train/test)
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{dataset_name} Evaluation Results:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def log_to_mlflow(model, preprocessor, train_metrics, test_metrics, config, 
                  X_train_sample, X_test_sample):
    """
    Log model, parameters, training metrics, test metrics, and sample data to MLflow.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        train_metrics: Dictionary of training metrics
        test_metrics: Dictionary of test metrics
        config: Configuration dictionary
        X_train_sample: Sample of training data (raw, before preprocessing)
        X_test_sample: Sample of test data (processed, for signature)
    """
    mlflow_config = config.get('mlflow', {})
    tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
    experiment_name = mlflow_config.get('experiment_name', 'ML_Experiment')
    model_name = mlflow_config.get('model_name', 'lead_prediction_model')
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="training"):
        # Log model parameters
        model_params = config.get('model', {}).get('params', {})
        mlflow.log_params(model_params)
        
        # Log training metrics with "train_" prefix
        train_metrics_prefixed = {f"train_{k}": v for k, v in train_metrics.items()}
        mlflow.log_metrics(train_metrics_prefixed)
        
        # Log test metrics with "test_" prefix
        test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
        mlflow.log_metrics(test_metrics_prefixed)
        
        # Create full pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Create input example from sample training data (5 rows)
        input_example = X_train_sample.head(5)
        
        # Infer signature from sample data
        # Use raw input and model predictions for signature
        sample_predictions = full_pipeline.predict(input_example)
        signature = infer_signature(input_example, sample_predictions)
        
        # Log model with signature and input example
        mlflow.sklearn.log_model(
            full_pipeline,
            "model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example
        )
        
        # Log sample training data as artifact
        sample_data_path = "sample_training_data.csv"
        X_train_sample.head(100).to_csv(sample_data_path, index=False)
        mlflow.log_artifact(sample_data_path)
        os.remove(sample_data_path)  # Clean up local file
        
        print(f"\nModel logged to MLflow!")
        print(f"Experiment: {experiment_name}")
        print(f"Model registered as: {model_name}")
        print(f"Parameters logged: {model_params}")
        print(f"Training metrics logged: {train_metrics_prefixed}")
        print(f"Test metrics logged: {test_metrics_prefixed}")
        print(f"Input signature: {signature}")
        print(f"Sample input data: {len(input_example)} rows saved")


def save_model(model, preprocessor, filepath):
    """
    Save model pipeline to disk.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    joblib.dump(full_pipeline, filepath)
    print(f"Model saved to: {filepath}")
