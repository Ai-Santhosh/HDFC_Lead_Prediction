# HDFC Bank Lead Prediction System

<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/2/28/HDFC_Bank_Logo.svg" alt="HDFC Bank Logo" width="280"/>

**AI-Powered Lead Conversion Prediction for HDFC Bank**

---

![Python](https://img.shields.io/badge/Python-3.8+-004C8F.svg?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-004C8F.svg?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ED1C24.svg?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-004C8F.svg?style=flat-square&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-ED1C24.svg?style=flat-square&logo=scikitlearn&logoColor=white)

</div>

---

## Overview

This **end-to-end Machine Learning solution** helps **HDFC Bank's sales and marketing teams** identify high-potential leads for various banking products.

### Supported Products

| Product | Description |
|---------|-------------|
| **Home Loans** | Housing finance lead conversion |
| **Credit Cards** | Premium card offerings targeting |
| **Vehicle Loans** | Auto loan predictions |
| **Personal Loans** | Personal loan prioritization |
| **Insurance** | Insurance product leads |

### Business Impact

- **Increase conversion rates** by focusing on high-probability leads
- **Save time** for relationship managers with AI-powered prioritization
- **Data-driven decisions** using ML predictions with reasoning
- **Track performance** with comprehensive MLflow dashboards

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Inference Options](#inference-options)
- [FastAPI REST API](#fastapi-rest-api)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Configuration](#configuration)
- [MLflow Tracking](#mlflow-tracking)
- [Model Details](#model-details)
- [Lead Scoring Factors](#lead-scoring-factors)
- [Troubleshooting](#troubleshooting)

---

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| **One-Click Training** | Train model with single command |
| **REST API** | Production-ready FastAPI endpoints |
| **Interactive Dashboard** | Streamlit UI for batch predictions |
| **MLflow Integration** | Experiment tracking and model registry |

### HDFC-Specific Features

| Feature | Description |
|---------|-------------|
| **Product Filtering** | Filter by loan/card type |
| **AI Reasoning** | Explains why a lead may convert |
| **Export to CSV** | Download for CRM integration |
| **Product Breakdown** | Statistics by banking product |

---

## Project Structure

```
HDFC_Lead_Prediction/
│
├── src/                              # Source code modules
│   ├── config/
│   │   └── config.yaml               # All configuration settings
│   ├── data/
│   │   └── data_loader.py            # Data loading utilities
│   ├── features/
│   │   └── preprocessing.py          # Data cleaning and feature engineering
│   ├── models/
│   │   └── trainer.py                # Model training and evaluation
│   └── utils/
│       └── config_loader.py          # Configuration utilities
│
├── models/                           # Saved trained models
│   └── best_model.pkl                # Trained model pipeline
│
├── mlruns/                           # MLflow experiment logs
│
├── main.py                           # Training pipeline script
├── inference.py                      # CLI inference script
├── api.py                            # FastAPI REST API
├── app.py                            # Streamlit Dashboard
│
├── requirements.txt                  # Python dependencies
├── Steps.txt                         # Detailed usage guide
└── README.md                         # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- HDFC Lead Data (CSV format)

### Step 1: Setup Environment

```bash
# Navigate to project directory
cd ML_End_To_End

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python main.py
```

### Step 3: Start the Services

```bash
# Terminal 1: Start FastAPI (for predictions)
uvicorn api:app --reload --port 8000

# Terminal 2: Start Streamlit Dashboard
streamlit run app.py

# Terminal 3 (Optional): Start MLflow UI
mlflow ui
```

### Step 4: Access Applications

| Application | URL | Description |
|-------------|-----|-------------|
| **Dashboard** | http://localhost:8501 | Streamlit UI for predictions |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **MLflow** | http://localhost:5000 | Experiment tracking UI |

---

## Training Pipeline

The training pipeline transforms raw HDFC lead data into actionable predictions.

### Pipeline Flow

```
Load Data → Clean Data → Extract Features → Split Data → Preprocess → Train Model → Evaluate → Save
```

### Run Training

```bash
python main.py
```

### Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1. Load Data | Read HDFC lead CSV | DataFrame with 120k+ records |
| 2. Clean Data | Remove duplicates, drop PII columns | Clean DataFrame |
| 3. Extract Features | Convert dates to year/month/day | Enhanced features |
| 4. Split Data | 80% train, 20% test (stratified) | Train and test sets |
| 5. Preprocess | Impute, scale, encode | Transformed features |
| 6. Train Model | Random Forest fitting | Trained classifier |
| 7. Evaluate | Calculate metrics | Accuracy, F1, etc. |
| 8. Log to MLflow | Store experiment | Tracked experiment |
| 9. Save Model | Pickle pipeline | `models/best_model.pkl` |

---

## Inference Options

### Option 1: Streamlit Dashboard (Recommended)

Best for **Relationship Managers** and **Sales Teams**:

```bash
streamlit run app.py
```

Features:
- Upload CSV with leads
- View predictions with reasoning
- Filter by product (Home Loan, Credit Card, etc.)
- Download results for CRM

### Option 2: REST API

Best for **System Integration** and **Automation**:

```bash
uvicorn api:app --reload --port 8000
```

Integrate with existing HDFC systems via API calls.

### Option 3: Command Line

Best for **Batch Processing**:

```bash
python inference.py
```

Outputs to `predictions.csv`

---

## FastAPI REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check and model status |
| `POST` | `/predict` | Single lead prediction |
| `POST` | `/predict/batch` | Batch predictions |

### Single Lead Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "gender": "Male",
           "age": 35,
           "marital_status": "Married",
           "annual_income": 750000,
           "city": "Chennai",
           "cibil_score": 750,
           "product_category": "Home Loan",
           "lead_source": "Website",
           "account_tenure_years": 5
         }'
```

**Response:**
```json
{
    "prediction": 1,
    "probability": 0.8542,
    "confidence": "High"
}
```

### Batch Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
           "leads": [
             {"gender": "Male", "age": 35, "cibil_score": 780, "product_category": "Home Loan"},
             {"gender": "Female", "age": 28, "cibil_score": 720, "product_category": "Credit Card"}
           ]
         }'
```

**Response:**
```json
{
    "predictions": [
        {"prediction": 1, "probability": 0.8921, "confidence": "High"},
        {"prediction": 1, "probability": 0.8915, "confidence": "High"}
    ],
    "total_leads": 2,
    "predicted_conversions": 2,
    "conversion_rate": 100.0
}
```

---

## Streamlit Dashboard

### Features

- **Upload CSV**: Drag and drop or browse for lead data
- **Run Predictions**: One-click batch predictions
- **View Results**: Table with predictions and reasoning
- **Filter by Product**: Focus on specific product categories
- **Download Results**: Export to CSV for CRM

### How to Use

1. Start the dashboard: `streamlit run app.py`
2. Upload your CSV file with lead data
3. Click "Run Predictions"
4. View results with conversion probability and confidence
5. Filter by product, status, or confidence level
6. Download results as CSV

---

## Configuration

All settings are in `src/config/config.yaml`:

```yaml
# Data Settings
data:
  file_path: "HDFC_TN_Leads_120k.csv"
  target_column: "conversion_flag"

# Model Settings
model:
  algorithm: "RandomForest"
  parameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42

# Training Settings
training:
  test_size: 0.2
  stratify: true
```

---

## MLflow Tracking

MLflow tracks all experiments with metrics prefixed by stage:

| Stage | Prefix | Metrics Tracked |
|-------|--------|-----------------|
| Training | `train_` | accuracy, f1, precision, recall |
| Testing | `test_` | accuracy, f1, precision, recall |
| Inference | `inference_` | predictions count, conversion rate |

### View Experiments

```bash
mlflow ui
```

Access at http://localhost:5000

---

## Model Details

### Algorithm

**Random Forest Classifier** with optimized parameters:

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 10 |
| min_samples_split | 2 |
| min_samples_leaf | 1 |

### Preprocessing Pipeline

1. **Numeric Features**: Mean imputation + Standard scaling
2. **Categorical Features**: Mode imputation + One-hot encoding
3. **Date Features**: Extract year, month, day components

---

## Lead Scoring Factors

### Positive Indicators

| Factor | Impact |
|--------|--------|
| CIBIL Score >= 750 | High conversion likelihood |
| Annual Income >= 10L | Premium segment, higher conversion |
| Account Tenure >= 5 years | Loyal customer, trust established |
| Low Credit Utilization | Healthy finances |
| High App Engagement | Digital-savvy, responsive |

### Risk Indicators

| Factor | Impact |
|--------|--------|
| CIBIL Score < 650 | Credit risk, lower approval |
| High Credit Utilization | Financial stress indicator |
| Low Account Tenure | New customer, less data |
| No Digital Engagement | Harder to reach |

### Sample AI Reasoning

```
High conversion likelihood | Excellent credit score (780) | High income segment | Long-term customer (6 years)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Model not found | Run `python main.py` to train |
| API not responding | Check if uvicorn is running on port 8000 |
| Streamlit error | Ensure API is running before dashboard |
| Import errors | Install dependencies: `pip install -r requirements.txt` |

### Health Check Commands

```bash
# Check API health
curl http://localhost:8000/health

# Check if model is loaded
curl http://localhost:8000/
```

---

## Quick Reference

### Commands

| Action | Command |
|--------|---------|
| Train Model | `python main.py` |
| Start API | `uvicorn api:app --reload --port 8000` |
| Start Dashboard | `streamlit run app.py` |
| Start MLflow | `mlflow ui` |
| CLI Inference | `python inference.py` |

### URLs

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Health Check | http://localhost:8000/health |

---

<div align="center">

**HDFC Bank Lead Prediction System**

Built with Python, FastAPI, Streamlit, and MLflow

---

*For internal use by HDFC Bank Sales and Marketing Teams*

</div>
