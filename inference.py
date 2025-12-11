"""
Inference Script - Simple & Beginner Friendly
==============================================
Run this script to make predictions using the trained model.

Usage:
    python inference.py
"""

import joblib
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our modules
from src.utils import load_config
from src.data import load_data
from src.features import clean_data, extract_date_features


def load_model(model_path):
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model pipeline
    """
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def make_predictions(model, X):
    """
    Make predictions and get probabilities.
    
    Args:
        model: Trained model pipeline
        X: Features DataFrame
        
    Returns:
        predictions, probabilities
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


def log_inference_to_mlflow(predictions, probabilities, config, y_true=None):
    """
    Log inference metrics to MLflow.
    
    Args:
        predictions: Model predictions
        probabilities: Prediction probabilities
        config: Configuration dictionary
        y_true: True labels (if available for evaluation)
    """
    mlflow_config = config.get('mlflow', {})
    tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
    experiment_name = mlflow_config.get('experiment_name', 'ML_Experiment')
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="inference"):
        # Log model parameters (same as training)
        model_params = config.get('model', {}).get('params', {})
        mlflow.log_params(model_params)
        
        # Log inference statistics
        inference_stats = {
            'inference_total_samples': len(predictions),
            'inference_positive_predictions': int(predictions.sum()),
            'inference_negative_predictions': int(len(predictions) - predictions.sum()),
            'inference_mean_probability': float(probabilities.mean()),
            'inference_std_probability': float(probabilities.std())
        }
        mlflow.log_metrics(inference_stats)
        
        # If true labels available, log inference metrics
        if y_true is not None:
            inference_metrics = {
                'inference_accuracy': accuracy_score(y_true, predictions),
                'inference_precision': precision_score(y_true, predictions, zero_division=0),
                'inference_recall': recall_score(y_true, predictions, zero_division=0),
                'inference_f1_score': f1_score(y_true, predictions, zero_division=0)
            }
            mlflow.log_metrics(inference_metrics)
            print(f"Inference metrics logged: {inference_metrics}")
        
        print(f"\nInference logged to MLflow!")
        print(f"Experiment: {experiment_name}")
        print(f"Parameters logged: {model_params}")
        print(f"Inference stats logged: {inference_stats}")


def main():
    """
    Main function to run inference.
    """
    print("="*60)
    print("HDFC Lead Prediction - Inference Pipeline")
    print("="*60)
    
    # Load configuration
    print("\n[Step 1] Loading configuration...")
    config = load_config()
    data_config = config['data']
    output_config = config['output']
    
    # Load trained model
    print("\n[Step 2] Loading trained model...")
    model = load_model(output_config['model_path'])
    
    # Load data for inference
    print("\n[Step 3] Loading data...")
    df = load_data(data_config['csv_filepath'])
    
    # Clean and preprocess data
    print("\n[Step 4] Preprocessing data...")
    df_clean = clean_data(df, data_config['drop_columns'])
    df_processed = extract_date_features(df_clean, data_config['date_columns'])
    
    # Get true labels if available (for evaluation)
    target_col = data_config['target_column']
    y_true = None
    if target_col in df_processed.columns:
        y_true = df_processed[target_col]
        df_processed = df_processed.drop(columns=[target_col])
    
    # Make predictions
    print("\n[Step 5] Making predictions...")
    predictions, probabilities = make_predictions(model, df_processed)
    
    # Log to MLflow
    print("\n[Step 6] Logging inference to MLflow...")
    log_inference_to_mlflow(predictions, probabilities, config, y_true)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    # Save results
    output_file = 'predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Inference Summary")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Positive predictions: {predictions.sum()}")
    print(f"Negative predictions: {len(predictions) - predictions.sum()}")
    print(f"Mean probability: {probabilities.mean():.4f}")
    
    if y_true is not None:
        print(f"\nInference Metrics (vs actual labels):")
        print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
        print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true, predictions, zero_division=0):.4f}")
        print(f"F1 Score: {f1_score(y_true, predictions, zero_division=0):.4f}")
    
    print("="*60)
    print("Inference Completed Successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
