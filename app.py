"""
Streamlit App - HDFC Lead Prediction Dashboard
===============================================
Upload CSV data, get predictions via FastAPI, and download results.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="HDFC Lead Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS for Premium UI
# =============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #b8d4e8;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2d5a87;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        color: #2d5a87;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Status badges */
    .status-convert {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-not-convert {
        background-color: #dc3545;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.75rem;
    }
    
    .confidence-medium {
        background-color: #ffc107;
        color: #333;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.75rem;
    }
    
    .confidence-low {
        background-color: #dc3545;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.75rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #2d5a87;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a87 100%);
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Info box */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2d5a87;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Constants
# =============================================================================

API_BASE_URL = "http://localhost:8000"

# Columns to drop before sending to API (same as config)
COLUMNS_TO_DROP = [
    "customer_id", "first_name", "last_name", "mobile_number", "email",
    "lead_id", "conversion_date", "revenue_generated", "lead_status_reason",
    "lead_stage", "expected_conversion_probability", "created_timestamp",
    "updated_timestamp", "assigned_rm_name", "conversion_flag"
]

# Expected columns for the model
EXPECTED_COLUMNS = [
    'gender', 'age', 'marital_status', 'dependents_count', 
    'education_level', 'occupation', 'annual_income',
    'city', 'pincode', 'preferred_language', 'contact_channel_preference', 
    'mobile_app_usage', 'netbanking_active', 'avg_monthly_app_visits',
    'credit_card_spend_last_6m', 'cibil_score', 'credit_utilization_ratio',
    'existing_loans_count', 'existing_monthly_emi', 'avg_monthly_balance',
    'account_tenure_years', 'website_lead_source', 'product_category', 
    'sub_product', 'lead_source', 'campaign_name', 'followup_count', 
    'data_year', 'date_of_birth', 'last_login_date', 'lead_creation_date', 
    'last_followup_date'
]

# =============================================================================
# Helper Functions
# =============================================================================

def check_api_health():
    """Check if the FastAPI is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_reasoning(row):
    """Generate reasoning for the prediction based on lead attributes."""
    reasons = []
    probability = row.get('Probability', 0)
    
    # CIBIL Score reasoning
    cibil = row.get('cibil_score', None)
    if pd.notna(cibil):
        if cibil >= 750:
            reasons.append(f"✅ Excellent credit score ({int(cibil)})")
        elif cibil >= 650:
            reasons.append(f"📊 Good credit score ({int(cibil)})")
        else:
            reasons.append(f"⚠️ Low credit score ({int(cibil)})")
    
    # Annual Income reasoning
    income = row.get('annual_income', None)
    if pd.notna(income):
        if income >= 1000000:
            reasons.append(f"💰 High income (₹{income:,.0f})")
        elif income >= 500000:
            reasons.append(f"💵 Moderate income (₹{income:,.0f})")
        else:
            reasons.append(f"📉 Lower income bracket")
    
    # Account Tenure
    tenure = row.get('account_tenure_years', None)
    if pd.notna(tenure):
        if tenure >= 5:
            reasons.append(f"🏦 Long-term customer ({int(tenure)} years)")
        elif tenure >= 2:
            reasons.append(f"👤 Established customer ({int(tenure)} years)")
    
    # Credit Utilization
    util = row.get('credit_utilization_ratio', None)
    if pd.notna(util):
        if util <= 0.3:
            reasons.append(f"✅ Low credit utilization ({util:.0%})")
        elif util >= 0.7:
            reasons.append(f"⚠️ High credit utilization ({util:.0%})")
    
    # Followup Count
    followups = row.get('followup_count', None)
    if pd.notna(followups):
        if followups >= 3:
            reasons.append(f"📞 Multiple followups ({int(followups)})")
    
    # App Usage
    app_usage = row.get('mobile_app_usage', None)
    if pd.notna(app_usage) and app_usage == 'High':
        reasons.append("📱 High app engagement")
    
    # Final reasoning based on probability
    if probability >= 0.7:
        reasons.insert(0, "🎯 High conversion likelihood")
    elif probability >= 0.5:
        reasons.insert(0, "📊 Moderate conversion chance")
    else:
        reasons.insert(0, "📉 Lower conversion probability")
    
    return " | ".join(reasons[:4]) if reasons else "Insufficient data for reasoning"


def prepare_data_for_api(df):
    """Prepare dataframe for API prediction."""
    # Make a copy
    df_clean = df.copy()
    
    # Drop columns that should not be sent to API
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df_clean.columns:
            df_clean[col] = np.nan
    
    return df_clean


def predict_batch(df):
    """Send batch prediction request to FastAPI."""
    # Prepare data
    df_clean = prepare_data_for_api(df)
    
    # Convert DataFrame to list of dicts (handling NaN values)
    leads = df_clean.replace({np.nan: None}).to_dict(orient='records')
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"leads": leads},
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try with fewer records.")
        return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


def create_results_dataframe(original_df, predictions):
    """Combine original data with predictions."""
    # Create results dataframe
    results_df = original_df.copy()
    
    # Add prediction columns
    pred_data = predictions['predictions']
    results_df['Conversion_Status'] = [
        'Will Convert' if p['prediction'] == 1 else 'Will Not Convert' 
        for p in pred_data
    ]
    results_df['Probability'] = [p['probability'] for p in pred_data]
    results_df['Confidence'] = [p['confidence'] for p in pred_data]
    
    # Add reasoning for each row
    results_df['Reasoning'] = results_df.apply(get_reasoning, axis=1)
    
    return results_df


@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 HDFC Lead Prediction Dashboard</h1>
        <p>Upload your leads data, get AI-powered predictions, and download results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/28/HDFC_Bank_Logo.svg", width=200)
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        # Check API health
        health = check_api_health()
        if health and health.get('model_loaded'):
            st.success("✅ API Connected")
            st.info(f"📁 Model: {health.get('model_path', 'N/A')}")
        else:
            st.error("❌ API Not Connected")
            st.warning("Start the API with:\n```\nuvicorn api:app --port 8000\n```")
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This dashboard uses a **Random Forest** model trained on 120k+ HDFC leads 
        to predict which leads are likely to convert.
        
        **Features:**
        - Upload CSV data
        - Get instant predictions
        - View reasoning for each prediction
        - Filter by product
        - Download results
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("[📖 API Docs](http://localhost:8000/docs)")
        st.markdown("[📈 MLflow Dashboard](http://localhost:5000)")
    
    # Main content
    st.markdown("### 📁 Upload Lead Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file with lead data",
        type=['csv'],
        help="Upload a CSV file containing lead information. The file should have columns like gender, age, cibil_score, etc."
    )
    
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
            
            # Show data preview
            with st.expander("📋 Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Prediction button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_button = st.button(
                    "🚀 Run Predictions",
                    type="primary",
                    use_container_width=True
                )
            
            if predict_button:
                # Check API
                if not check_api_health():
                    st.error("❌ Cannot connect to the prediction API. Please ensure it's running.")
                    return
                
                # Run predictions
                with st.spinner("🔄 Running predictions... This may take a moment for large datasets."):
                    predictions = predict_batch(df)
                
                if predictions:
                    # Store in session state
                    results_df = create_results_dataframe(df, predictions)
                    st.session_state['results'] = results_df
                    st.session_state['predictions'] = predictions
                    st.success("✅ Predictions completed successfully!")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return
    
    # Display results if available
    if 'results' in st.session_state:
        results_df = st.session_state['results']
        predictions = st.session_state['predictions']
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Leads</h3>
                <div class="value">{predictions['total_leads']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Conversions</h3>
                <div class="value" style="color: #28a745;">{predictions['predicted_conversions']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            non_conversions = predictions['total_leads'] - predictions['predicted_conversions']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Non-Conversions</h3>
                <div class="value" style="color: #dc3545;">{non_conversions:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Conversion Rate</h3>
                <div class="value">{predictions['conversion_rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filter options
        st.markdown("### 🔍 Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Product filter
            products = ['All Products']
            if 'product_category' in results_df.columns:
                products += results_df['product_category'].dropna().unique().tolist()
            selected_product = st.selectbox("📦 Filter by Product", products)
        
        with col2:
            # Conversion status filter
            status_filter = st.selectbox(
                "📊 Conversion Status",
                ['All', 'Will Convert', 'Will Not Convert']
            )
        
        with col3:
            # Confidence filter
            confidence_filter = st.selectbox(
                "🎯 Confidence Level",
                ['All', 'High', 'Medium', 'Low']
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if selected_product != 'All Products':
            filtered_df = filtered_df[filtered_df['product_category'] == selected_product]
        
        if status_filter != 'All':
            filtered_df = filtered_df[filtered_df['Conversion_Status'] == status_filter]
        
        if confidence_filter != 'All':
            filtered_df = filtered_df[filtered_df['Confidence'] == confidence_filter]
        
        # Show filtered count
        st.info(f"📋 Showing {len(filtered_df):,} of {len(results_df):,} records")
        
        # Display columns to show
        display_cols = ['Conversion_Status', 'Probability', 'Confidence', 'Reasoning']
        
        # Add some important feature columns if they exist
        important_cols = ['product_category', 'sub_product', 'cibil_score', 
                         'annual_income', 'city', 'lead_source']
        for col in important_cols:
            if col in filtered_df.columns:
                display_cols.insert(0, col)
        
        # Add customer identifiers if they exist
        id_cols = ['customer_id', 'lead_id', 'first_name', 'last_name']
        for col in id_cols:
            if col in filtered_df.columns:
                display_cols.insert(0, col)
        
        # Display the dataframe
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            height=400
        )
        
        if len(filtered_df) > 100:
            st.caption(f"⚠️ Showing first 100 records. Download to see all {len(filtered_df):,} records.")
        
        st.markdown("---")
        
        # Download section
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Full Results (All Columns)**")
            csv_full = convert_df_to_csv(filtered_df)
            filename = f"lead_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            if selected_product != 'All Products':
                filename = f"lead_predictions_{selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="⬇️ Download Full CSV",
                data=csv_full,
                file_name=filename,
                mime="text/csv",
            )
        
        with col2:
            st.markdown("**Summary Only (Key Columns)**")
            summary_cols = ['Conversion_Status', 'Probability', 'Confidence', 'Reasoning']
            if 'customer_id' in filtered_df.columns:
                summary_cols.insert(0, 'customer_id')
            if 'lead_id' in filtered_df.columns:
                summary_cols.insert(0, 'lead_id')
            if 'product_category' in filtered_df.columns:
                summary_cols.append('product_category')
            
            summary_df = filtered_df[summary_cols]
            csv_summary = convert_df_to_csv(summary_df)
            
            st.download_button(
                label="⬇️ Download Summary CSV",
                data=csv_summary,
                file_name=f"lead_predictions_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        # Product breakdown (if applicable)
        if 'product_category' in results_df.columns:
            st.markdown("---")
            st.markdown("### 📦 Product-wise Breakdown")
            
            product_summary = results_df.groupby('product_category').agg({
                'Conversion_Status': 'count',
                'Probability': 'mean'
            }).reset_index()
            product_summary.columns = ['Product', 'Total Leads', 'Avg Probability']
            
            # Add conversion count
            conversions = results_df[results_df['Conversion_Status'] == 'Will Convert'].groupby('product_category').size()
            product_summary['Predicted Conversions'] = product_summary['Product'].map(conversions).fillna(0).astype(int)
            product_summary['Conversion Rate %'] = (product_summary['Predicted Conversions'] / product_summary['Total Leads'] * 100).round(1)
            product_summary['Avg Probability'] = (product_summary['Avg Probability'] * 100).round(1)
            
            st.dataframe(product_summary, use_container_width=True)
            
            # Download product-wise files
            st.markdown("#### 📁 Download by Product")
            cols = st.columns(min(4, len(product_summary)))
            
            for idx, (_, row) in enumerate(product_summary.iterrows()):
                product = row['Product']
                product_df = results_df[results_df['product_category'] == product]
                product_csv = convert_df_to_csv(product_df)
                
                with cols[idx % len(cols)]:
                    st.download_button(
                        label=f"📥 {product}",
                        data=product_csv,
                        file_name=f"leads_{product.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"download_{product}"
                    )
    
    else:
        # Show instructions when no data is loaded
        st.markdown("""
        <div class="info-box">
            <h4>📝 How to use this dashboard:</h4>
            <ol>
                <li><strong>Ensure the API is running</strong> - Check the sidebar for connection status</li>
                <li><strong>Upload your CSV file</strong> - Click the upload button above</li>
                <li><strong>Click "Run Predictions"</strong> - The model will analyze each lead</li>
                <li><strong>View results</strong> - See predictions with reasoning</li>
                <li><strong>Download</strong> - Export results as CSV</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
