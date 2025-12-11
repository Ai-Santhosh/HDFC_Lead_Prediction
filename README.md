<div align="center">

# 🏦 HDFC Bank Lead Prediction System

<img src="https://upload.wikimedia.org/wikipedia/commons/2/28/HDFC_Bank_Logo.svg" alt="HDFC Bank Logo" width="300"/>

### *AI-Powered Lead Conversion Prediction for HDFC Bank*

---

![Python](https://img.shields.io/badge/Python-3.8+-004C8F.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-004C8F.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ED1C24.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-004C8F.svg?style=for-the-badge&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-ED1C24.svg?style=for-the-badge&logo=scikitlearn&logoColor=white)

**🎯 Predict which leads will convert | 📊 Track experiments with MLflow | 🚀 Deploy with FastAPI & Streamlit**

[Features](#-features) •
[Quick Start](#-quick-start) •
[API Reference](#-fastapi-rest-api) •
[Dashboard](#-streamlit-dashboard) •
[Configuration](#%EF%B8%8F-configuration)

</div>

---

## 🏛️ About HDFC Bank Lead Prediction

This **end-to-end Machine Learning solution** is designed for **HDFC Bank's sales and marketing teams** to identify high-potential leads for various banking products including:

| 🏠 **Home Loans** | � **Credit Cards** | 🚗 **Vehicle Loans** | 💰 **Personal Loans** |
|:-----------------:|:-------------------:|:--------------------:|:--------------------:|
| Identify customers likely to convert for housing finance | Target customers for premium card offerings | Predict auto loan conversions | Personal loan lead prioritization |

### 🎯 Business Impact

- **📈 Increase conversion rates** by focusing on high-probability leads
- **⏱️ Save time** for relationship managers with AI-powered prioritization  
- **💡 Data-driven decisions** using ML predictions with reasoning
- **📊 Track performance** with comprehensive MLflow dashboards

---

## �📋 Table of Contents

- [About HDFC Bank Lead Prediction](#%EF%B8%8F-about-hdfc-bank-lead-prediction)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training Pipeline](#-training-pipeline)
- [Inference Options](#-inference-options)
- [FastAPI REST API](#-fastapi-rest-api)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Configuration](#%EF%B8%8F-configuration)
- [MLflow Tracking](#-mlflow-tracking)
- [Model Details](#-model-details)
- [Lead Scoring Factors](#-lead-scoring-factors)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔷 Core Features

| Feature | Description |
|---------|-------------|
| 🎯 **One-Click Training** | Train model with single command |
| 🚀 **REST API** | Production-ready FastAPI endpoints |
| 📊 **Interactive Dashboard** | Streamlit UI for batch predictions |
| 📈 **MLflow Integration** | Experiment tracking & model registry |

</td>
<td width="50%">

### � HDFC-Specific Features

| Feature | Description |
|---------|-------------|
| 🏦 **Product Filtering** | Filter by loan/card type |
| 💡 **AI Reasoning** | Explains why a lead may convert |
| 📥 **Export to CSV** | Download for CRM integration |
| 📦 **Product Breakdown** | Stats by banking product |

</td>
</tr>
</table>

---

## 📁 Project Structure

```
🏦 HDFC_Lead_Prediction/
│
├── 📂 src/                          # Source code modules
│   ├── 📂 config/
│   │   └── ⚙️ config.yaml           # All configuration settings
│   ├── 📂 data/
│   │   └── 📄 data_loader.py        # Data loading utilities
│   ├── 📂 features/
│   │   └── 🔧 preprocessing.py      # Data cleaning & feature engineering
│   ├── 📂 models/
│   │   └── 🤖 trainer.py            # Model training & evaluation
│   └── 📂 utils/
│       └── 🛠️ config_loader.py      # Configuration utilities
│
├── 📂 models/                        # Saved trained models
│   └── 💾 best_model.pkl            # Trained model pipeline
│
├── 📂 mlruns/                        # MLflow experiment logs
│
├── 🐍 main.py                        # Training pipeline script
├── 🐍 inference.py                   # CLI inference script
├── � api.py                         # FastAPI REST API
├── � app.py                         # Streamlit Dashboard
│
├── 📄 requirements.txt               # Python dependencies
├── � Steps.txt                      # Detailed usage guide
└── � README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- HDFC Lead Data (CSV format)

### 1️⃣ Setup Environment

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

### 2️⃣ Train the Model

```bash
python main.py
```

### 3️⃣ Start the Services

```bash
# Terminal 1: Start FastAPI (for predictions)
uvicorn api:app --reload --port 8000

# Terminal 2: Start Streamlit Dashboard
streamlit run app.py

# Terminal 3 (Optional): Start MLflow UI
mlflow ui
```

### 4️⃣ Access Applications

<table align="center">
<tr>
<td align="center">
<h3>📊 Dashboard</h3>
<a href="http://localhost:8501">localhost:8501</a>
</td>
<td align="center">
<h3>📖 API Docs</h3>
<a href="http://localhost:8000/docs">localhost:8000/docs</a>
</td>
<td align="center">
<h3>📈 MLflow</h3>
<a href="http://localhost:5000">localhost:5000</a>
</td>
</tr>
</table>

---

## 🎓 Training Pipeline

The training pipeline transforms raw HDFC lead data into actionable predictions:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  📂 Load    │ ─► │  🧹 Clean   │ ─► │  📅 Extract │ ─► │  ✂️ Split   │
│    Data     │    │    Data     │    │   Features  │    │    Data     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  � Save    │ ◄─ │  📊 Evaluate│ ◄─ │  🤖 Train   │ ◄─ │  🔧 Preproc │
│   Model     │    │   Model     │    │   Model     │    │    Data     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
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
| 4. Split Data | 80% train, 20% test (stratified) | Train & test sets |
| 5. Preprocess | Impute, scale, encode | Transformed features |
| 6. Train Model | Random Forest fitting | Trained classifier |
| 7. Evaluate | Calculate metrics | Accuracy, F1, etc. |
| 8. Log to MLflow | Store experiment | Tracked experiment |
| 9. Save Model | Pickle pipeline | `models/best_model.pkl` |

---

## 🔮 Inference Options

### Option 1: 📊 Streamlit Dashboard (Recommended)

Best for **Relationship Managers** and **Sales Teams**:

```bash
streamlit run app.py
```

- Upload CSV with leads
- View predictions with reasoning
- Filter by product (Home Loan, Credit Card, etc.)
- Download results for CRM

### Option 2: 🚀 REST API

Best for **System Integration** and **Automation**:

```bash
uvicorn api:app --reload --port 8000
```

Integrate with existing HDFC systems via API calls.

### Option 3: 💻 Command Line

Best for **Batch Processing**:

```bash
python inference.py
```

Outputs to `predictions.csv`

---

## 🌐 FastAPI REST API

### 🔷 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check & model status |
| `POST` | `/predict` | Single lead prediction |
| `POST` | `/predict/batch` | Batch predictions |

### 🏦 Single Lead Prediction

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
    "probability": 0.7856,
    "confidence": "High"
}
```

### 📦 Batch Prediction

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
        {"prediction": 1, "probability": 0.82, "confidence": "High"},
        {"prediction": 0, "probability": 0.35, "confidence": "Medium"}
    ],
    "total_leads": 2,
    "predicted_conversions": 1,
    "conversion_rate": 50.0
}
```

### 📖 Interactive Documentation

Visit **http://localhost:8000/docs** for Swagger UI

---

## 📊 Streamlit Dashboard

<div align="center">

### 🏦 HDFC Lead Prediction Dashboard

*An intuitive interface for sales and marketing teams*

</div>

### Dashboard Features

| Section | Description |
|---------|-------------|
| 📊 **Summary Metrics** | Total leads, predicted conversions, conversion rate |
| 🔍 **Smart Filters** | Filter by product, status, confidence |
| � **AI Reasoning** | Understand why each lead is scored |
| �📥 **Download Options** | Full CSV, summary, or by product |
| 📦 **Product Breakdown** | Stats per banking product |

### Start Dashboard

```bash
streamlit run app.py
```

### Usage Flow

```
1️⃣ Upload CSV → 2️⃣ Run Predictions → 3️⃣ View Results → 4️⃣ Filter Data → 5️⃣ Download CSV
```

---

## ⚙️ Configuration

All settings in `src/config/config.yaml`:

```yaml
# HDFC Project Configuration
project:
  name: "HDFC_Lead_Prediction"
  version: "1.0.0"

# Data Settings
data:
  csv_filepath: "HDFC_TN_Leads_120k.csv"
  target_column: "conversion_flag"
  
  # Columns to remove (PII, IDs, target leakage)
  drop_columns:
    - "customer_id"
    - "first_name"
    - "last_name"
    - "mobile_number"
    - "email"
    # ... more

# Model Configuration (Optimized for HDFC data)
model:
  name: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42

# MLflow Tracking
mlflow:
  experiment_name: "HDFC_Lead_Prediction"
```

---

## 📈 MLflow Tracking

### Start MLflow UI

```bash
mlflow ui
```

Visit **http://localhost:5000**

### Tracked Metrics

| Metric | Description |
|--------|-------------|
| `train_accuracy` | Training set accuracy |
| `train_precision` | Training precision |
| `train_recall` | Training recall |
| `train_f1_score` | Training F1 score |
| `test_accuracy` | Test set accuracy |
| `test_precision` | Test precision |
| `test_recall` | Test recall |
| `test_f1_score` | Test F1 score |

### Metrics Prefixes

| Prefix | Source |
|--------|--------|
| `train_*` | Training evaluation |
| `test_*` | Test evaluation |
| `inference_*` | CLI inference |
| `api_*` | API predictions |

---

## 🤖 Model Details

### Algorithm: Random Forest Classifier

An ensemble of **100 decision trees** optimized for HDFC lead data.

### Optimized Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 100 | Number of trees in forest |
| `max_depth` | 10 | Maximum depth of trees |
| `min_samples_split` | 2 | Min samples to split node |
| `min_samples_leaf` | 1 | Min samples in leaf node |

### Preprocessing Pipeline

| Step | Numerical | Categorical |
|------|-----------|-------------|
| **Missing Values** | Median imputation | Mode imputation |
| **Transformation** | StandardScaler | OneHotEncoder |

---

## 💡 Lead Scoring Factors

The model considers these key factors when scoring HDFC leads:

<table>
<tr>
<td width="50%">

### 🔷 Positive Indicators

| Factor | Impact |
|--------|--------|
| ✅ High CIBIL score (750+) | Strong positive |
| ✅ High annual income | Strong positive |
| ✅ Long account tenure (5+ years) | Positive |
| ✅ Low credit utilization (<30%) | Positive |
| ✅ Multiple followups | Indicates interest |
| ✅ High app engagement | Digital savvy |

</td>
<td width="50%">

### 🔶 Risk Indicators

| Factor | Impact |
|--------|--------|
| ⚠️ Low CIBIL score (<650) | Negative |
| ⚠️ High credit utilization (>70%) | Concerning |
| ⚠️ Multiple existing loans | Risk factor |
| ⚠️ Low income for product type | May not qualify |
| ⚠️ No followup engagement | Low interest |

</td>
</tr>
</table>

### AI Reasoning Examples

The dashboard provides reasoning like:

- *"🎯 High conversion likelihood | ✅ Excellent credit score (782) | 💰 High income (₹12,00,000) | 🏦 Long-term customer (7 years)"*
- *"📉 Lower conversion probability | ⚠️ High credit utilization (78%) | 📞 Multiple followups (4)"*

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| `FileNotFoundError` (CSV) | Check `csv_filepath` in config.yaml |
| `Model not found` | Run training: `python main.py` |
| `API Not Connected` | Start API: `uvicorn api:app --port 8000` |
| Port in use | Use different port: `--port 8001` |

### Health Check Commands

```bash
# Check API status
curl http://localhost:8000/health

# Verify model exists
ls -la models/best_model.pkl

# Check Streamlit
curl http://localhost:8501/_stcore/health
```

---

## 📋 Quick Reference

<table align="center">
<tr>
<th>Task</th>
<th>Command</th>
</tr>
<tr>
<td>🎓 Train Model</td>
<td><code>python main.py</code></td>
</tr>
<tr>
<td>💻 CLI Inference</td>
<td><code>python inference.py</code></td>
</tr>
<tr>
<td>🚀 Start API</td>
<td><code>uvicorn api:app --reload --port 8000</code></td>
</tr>
<tr>
<td>📊 Start Dashboard</td>
<td><code>streamlit run app.py</code></td>
</tr>
<tr>
<td>📈 Start MLflow</td>
<td><code>mlflow ui</code></td>
</tr>
</table>

### 🔗 Quick Links

| Service | Local URL |
|---------|-----------|
| 📊 Dashboard | http://localhost:8501 |
| 📖 API Docs | http://localhost:8000/docs |
| 📈 MLflow | http://localhost:5000 |

---

<div align="center">

### 🏦 HDFC Bank Lead Prediction System

**Empowering sales teams with AI-driven insights**

---

*Built for HDFC Bank's Lead Management & Conversion Optimization*

<sub>Made with ❤️ using Python, FastAPI, Streamlit & MLflow</sub>

</div>
