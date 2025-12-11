"""
Main Training Script - Simple & Beginner Friendly
==================================================
Run this script to train the ML model.

Usage:
    python main.py
"""

from sklearn.model_selection import train_test_split

# Import our modules
from src.utils import load_config
from src.data import load_data
from src.features import clean_data, extract_date_features, build_preprocessor
from src.models import train_model, evaluate_model, log_to_mlflow, save_model


def main():
    """
    Main function to run the training pipeline.
    """
    print("="*60)
    print("HDFC Lead Prediction - Training Pipeline")
    print("="*60)
    
    # Step 1: Load configuration
    print("\n[Step 1] Loading configuration...")
    config = load_config()
    data_config = config['data']
    
    # Step 2: Load data
    print("\n[Step 2] Loading data...")
    df = load_data(data_config['csv_filepath'])
    
    # Step 3: Clean data
    print("\n[Step 3] Cleaning data...")
    df = clean_data(df, data_config['drop_columns'])
    
    # Step 4: Extract date features
    print("\n[Step 4] Extracting date features...")
    df = extract_date_features(df, data_config['date_columns'])
    
    # Step 5: Separate features and target
    print("\n[Step 5] Preparing features and target...")
    target_col = data_config['target_column']
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Step 6: Split data
    print("\n[Step 6] Splitting data...")
    preproc_config = config['preprocessing']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=preproc_config['test_size'],
        random_state=preproc_config['random_state'],
        stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 7: Build and fit preprocessor
    print("\n[Step 7] Preprocessing data...")
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed features shape: {X_train_processed.shape}")
    
    # Step 8: Train model
    print("\n[Step 8] Training model...")
    model_params = config['model']['params']
    model = train_model(X_train_processed, y_train, model_params)
    
    # Step 9: Evaluate on Training data
    print("\n[Step 9] Evaluating on Training data...")
    train_metrics = evaluate_model(model, X_train_processed, y_train, "Training")
    
    # Step 10: Evaluate on Test data
    print("\n[Step 10] Evaluating on Test data...")
    test_metrics = evaluate_model(model, X_test_processed, y_test, "Test")
    
    # Step 11: Log to MLflow (with sample data and signature)
    print("\n[Step 11] Logging to MLflow...")
    log_to_mlflow(
        model, 
        preprocessor, 
        train_metrics, 
        test_metrics, 
        config,
        X_train,  # Raw training data (before preprocessing) for input example
        X_test_processed  # Processed test data for verification
    )
    
    # Step 12: Save model locally
    print("\n[Step 12] Saving model...")
    output_config = config['output']
    save_model(model, preprocessor, output_config['model_path'])
    
    print("\n" + "="*60)
    print("Training Pipeline Completed Successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
