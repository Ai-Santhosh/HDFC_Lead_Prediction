"""
FastAPI Inference API - HDFC Lead Prediction
=============================================
A REST API for making predictions using the trained ML model.

Usage:
    uvicorn api:app --reload --port 8000
    
Endpoints:
    GET  /              - API info
    GET  /health        - Health check
    POST /predict       - Single prediction
    POST /predict/batch - Batch predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import joblib
import pandas as pd
import numpy as np
import mlflow
from contextlib import asynccontextmanager

from src.utils import load_config
from src.features import extract_date_features


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class LeadInput(BaseModel):
    """Input schema for a single lead prediction."""
    
    # Demographics
    gender: Optional[str] = Field(None, example="Male")
    age: Optional[int] = Field(None, example=35, ge=18, le=100)
    date_of_birth: Optional[str] = Field(None, example="1989-05-15")
    marital_status: Optional[str] = Field(None, example="Married")
    dependents_count: Optional[int] = Field(None, example=2, ge=0)
    education_level: Optional[str] = Field(None, example="Graduate")
    occupation: Optional[str] = Field(None, example="Salaried")
    annual_income: Optional[float] = Field(None, example=750000)
    
    # Location
    city: Optional[str] = Field(None, example="Chennai")
    pincode: Optional[int] = Field(None, example=600001)
    
    # Preferences
    preferred_language: Optional[str] = Field(None, example="English")
    contact_channel_preference: Optional[str] = Field(None, example="Email")
    mobile_app_usage: Optional[str] = Field(None, example="High")
    netbanking_active: Optional[str] = Field(None, example="Yes")
    last_login_date: Optional[str] = Field(None, example="2024-01-15")
    avg_monthly_app_visits: Optional[int] = Field(None, example=10)
    
    # Financial
    credit_card_spend_last_6m: Optional[float] = Field(None, example=150000)
    cibil_score: Optional[int] = Field(None, example=750, ge=300, le=900)
    credit_utilization_ratio: Optional[float] = Field(None, example=0.35, ge=0, le=1)
    existing_loans_count: Optional[int] = Field(None, example=1, ge=0)
    existing_monthly_emi: Optional[float] = Field(None, example=15000)
    avg_monthly_balance: Optional[float] = Field(None, example=50000)
    account_tenure_years: Optional[int] = Field(None, example=5, ge=0)
    
    # Lead Info
    website_lead_source: Optional[str] = Field(None, example="Google")
    lead_creation_date: Optional[str] = Field(None, example="2024-01-10")
    product_category: Optional[str] = Field(None, example="Loans")
    sub_product: Optional[str] = Field(None, example="Personal Loan")
    lead_source: Optional[str] = Field(None, example="Website")
    campaign_name: Optional[str] = Field(None, example="New Year Campaign")
    followup_count: Optional[int] = Field(None, example=3, ge=0)
    last_followup_date: Optional[str] = Field(None, example="2024-01-20")
    data_year: Optional[int] = Field(None, example=2024)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 35,
                "date_of_birth": "1989-05-15",
                "marital_status": "Married",
                "dependents_count": 2,
                "education_level": "Graduate",
                "occupation": "Salaried",
                "annual_income": 750000,
                "city": "Chennai",
                "pincode": 600001,
                "preferred_language": "English",
                "contact_channel_preference": "Email",
                "mobile_app_usage": "High",
                "netbanking_active": "Yes",
                "last_login_date": "2024-01-15",
                "avg_monthly_app_visits": 10,
                "credit_card_spend_last_6m": 150000,
                "cibil_score": 750,
                "credit_utilization_ratio": 0.35,
                "existing_loans_count": 1,
                "existing_monthly_emi": 15000,
                "avg_monthly_balance": 50000,
                "account_tenure_years": 5,
                "website_lead_source": "Google",
                "lead_creation_date": "2024-01-10",
                "product_category": "Loans",
                "sub_product": "Personal Loan",
                "lead_source": "Website",
                "campaign_name": "New Year Campaign",
                "followup_count": 3,
                "last_followup_date": "2024-01-20",
                "data_year": 2024
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    prediction: int = Field(..., description="0 = Will not convert, 1 = Will convert")
    probability: float = Field(..., description="Probability of conversion (0-1)")
    confidence: str = Field(..., description="Confidence level: Low/Medium/High")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    leads: List[LeadInput]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_leads: int
    predicted_conversions: int
    conversion_rate: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str


class APIInfoResponse(BaseModel):
    """API information response."""
    name: str
    version: str
    description: str
    endpoints: dict


# =============================================================================
# Global Variables
# =============================================================================

model = None
config = None
date_columns = None

# Expected columns for the model (after preprocessing, before date feature extraction)
# These are the columns the model was trained on
EXPECTED_COLUMNS = [
    # Demographics
    'gender', 'age', 'marital_status', 'dependents_count', 
    'education_level', 'occupation', 'annual_income',
    # Location
    'city', 'pincode',
    # Preferences
    'preferred_language', 'contact_channel_preference', 
    'mobile_app_usage', 'netbanking_active', 'avg_monthly_app_visits',
    # Financial
    'credit_card_spend_last_6m', 'cibil_score', 'credit_utilization_ratio',
    'existing_loans_count', 'existing_monthly_emi', 'avg_monthly_balance',
    'account_tenure_years',
    # Lead Info
    'website_lead_source', 'product_category', 'sub_product', 
    'lead_source', 'campaign_name', 'followup_count', 'data_year',
    # Date columns (will be converted to features)
    'date_of_birth', 'last_login_date', 'lead_creation_date', 'last_followup_date'
]


# =============================================================================
# Lifespan Context Manager (Load Model on Startup)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, config, date_columns
    
    print("🚀 Starting HDFC Lead Prediction API...")
    
    # Load configuration
    config = load_config()
    date_columns = config['data']['date_columns']
    
    # Load trained model
    model_path = config['output']['model_path']
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found at: {model_path}")
        print("   Please run 'python main.py' to train the model first.")
        raise RuntimeError(f"Model not found at {model_path}")
    
    yield
    
    # Cleanup
    print("👋 Shutting down API...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="HDFC Lead Prediction API",
    description="""
    🏦 **HDFC Lead Prediction API**
    
    A Machine Learning API that predicts which leads are likely to convert.
    
    ## Features
    - **Single Prediction**: Predict conversion for a single lead
    - **Batch Prediction**: Predict conversions for multiple leads at once
    - **MLflow Integration**: All predictions are logged for tracking
    
    ## Model Details
    - **Algorithm**: Random Forest Classifier
    - **Training Data**: HDFC Tamil Nadu Leads (120k records)
    
    ## Note
    All fields are optional. Missing values will be imputed by the model.
    Provide as many fields as possible for better predictions.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def preprocess_input(lead_data: dict) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    Ensures all expected columns exist (missing ones are set to NaN).
    
    Args:
        lead_data: Dictionary of lead features
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Create a complete data dict with all expected columns (NaN for missing)
    complete_data = {col: np.nan for col in EXPECTED_COLUMNS}
    
    # Update with provided values
    for key, value in lead_data.items():
        if key in complete_data:
            complete_data[key] = value
    
    # Convert to DataFrame
    df = pd.DataFrame([complete_data])
    
    # Extract date features (handles NaN dates gracefully)
    df = extract_date_features(df, date_columns)
    
    return df


def get_confidence_level(probability: float) -> str:
    """Get confidence level based on probability."""
    if probability >= 0.8 or probability <= 0.2:
        return "High"
    elif probability >= 0.6 or probability <= 0.4:
        return "Medium"
    else:
        return "Low"


def log_prediction_to_mlflow(predictions: List[int], probabilities: List[float], 
                              num_leads: int):
    """Log predictions to MLflow."""
    try:
        mlflow_config = config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
        experiment_name = mlflow_config.get('experiment_name', 'ML_Experiment')
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="api_inference"):
            mlflow.log_params(config.get('model', {}).get('params', {}))
            
            mlflow.log_metrics({
                'api_total_predictions': num_leads,
                'api_positive_predictions': int(np.sum(predictions)),
                'api_mean_probability': float(np.mean(probabilities)),
            })
    except Exception as e:
        print(f"Warning: Failed to log to MLflow: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=APIInfoResponse, tags=["Info"])
async def root():
    """Get API information."""
    return {
        "name": "HDFC Lead Prediction API",
        "version": "1.0.0",
        "description": "ML-powered API for predicting lead conversions",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Single lead prediction",
            "POST /predict/batch": "Batch predictions"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": config['output']['model_path'] if config else "N/A"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(lead: LeadInput):
    """
    Predict conversion for a single lead.
    
    - **lead**: Lead details for prediction
    
    Returns prediction (0/1), probability, and confidence level.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict and preprocess
        lead_data = lead.model_dump(exclude_none=True)
        df = preprocess_input(lead_data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        confidence = get_confidence_level(probability)
        
        # Log to MLflow
        log_prediction_to_mlflow([prediction], [probability], 1)
        
        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict conversions for multiple leads.
    
    - **leads**: List of lead details for batch prediction
    
    Returns predictions for all leads with summary statistics.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.leads:
        raise HTTPException(status_code=400, detail="No leads provided")
    
    try:
        predictions_list = []
        all_predictions = []
        all_probabilities = []
        
        for lead in request.leads:
            # Convert to dict and preprocess
            lead_data = lead.model_dump(exclude_none=True)
            df = preprocess_input(lead_data)
            
            # Make prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            confidence = get_confidence_level(probability)
            
            predictions_list.append({
                "prediction": int(prediction),
                "probability": round(float(probability), 4),
                "confidence": confidence
            })
            
            all_predictions.append(prediction)
            all_probabilities.append(probability)
        
        # Log to MLflow
        log_prediction_to_mlflow(all_predictions, all_probabilities, len(request.leads))
        
        # Calculate summary
        total_leads = len(request.leads)
        predicted_conversions = int(np.sum(all_predictions))
        conversion_rate = round(predicted_conversions / total_leads * 100, 2)
        
        return {
            "predictions": predictions_list,
            "total_leads": total_leads,
            "predicted_conversions": predicted_conversions,
            "conversion_rate": conversion_rate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# =============================================================================
# Run with Uvicorn (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
