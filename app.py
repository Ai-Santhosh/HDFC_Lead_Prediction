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
    page_icon="",
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
    
    /* HDFC Theme Colors: Blue #004C8F, Red #ED1C24 */
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #004C8F 0%, #003366 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        border-bottom: 4px solid #ED1C24;
    }
    
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: #cce0f5;
        font-size: 1.1rem;
    }
    
    /* Metric cards - HDFC Style */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 76, 143, 0.1);
        border-left: 4px solid #004C8F;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        color: #004C8F;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .metric-card .value {
        color: #003366;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Status badges */
    .status-convert {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-not-convert {
        background-color: #ED1C24;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    .confidence-medium {
        background-color: #ffc107;
        color: #333;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    .confidence-low {
        background-color: #ED1C24;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #004C8F;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Download button styling - HDFC Red */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #004C8F 0%, #003366 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #ED1C24 0%, #c41920 100%);
        box-shadow: 0 4px 15px rgba(237, 28, 36, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #004C8F 0%, #003366 100%);
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Info box - HDFC Blue */
    .info-box {
        background-color: #e6f0fa;
        border-left: 4px solid #004C8F;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
    }
    
    /* HDFC Accent line */
    .hdfc-accent {
        height: 4px;
        background: linear-gradient(90deg, #004C8F 0%, #ED1C24 100%);
        border-radius: 2px;
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
    """Generate professional reasoning for the prediction based on lead attributes."""
    reasons = []
    probability = row.get('Probability', 0)
    
    # CIBIL Score reasoning
    cibil = row.get('cibil_score', None)
    if pd.notna(cibil):
        if cibil >= 750:
            reasons.append(f"Excellent credit score ({int(cibil)})")
        elif cibil >= 650:
            reasons.append(f"Good credit score ({int(cibil)})")
        else:
            reasons.append(f"Low credit score ({int(cibil)})")
    
    # Annual Income reasoning
    income = row.get('annual_income', None)
    if pd.notna(income):
        if income >= 1000000:
            reasons.append(f"High income segment")
        elif income >= 500000:
            reasons.append(f"Moderate income segment")
        else:
            reasons.append(f"Lower income bracket")
    
    # Account Tenure
    tenure = row.get('account_tenure_years', None)
    if pd.notna(tenure):
        if tenure >= 5:
            reasons.append(f"Long-term customer ({int(tenure)} years)")
        elif tenure >= 2:
            reasons.append(f"Established customer ({int(tenure)} years)")
    
    # Credit Utilization
    util = row.get('credit_utilization_ratio', None)
    if pd.notna(util):
        if util <= 0.3:
            reasons.append(f"Low credit utilization ({util:.0%})")
        elif util >= 0.7:
            reasons.append(f"High credit utilization ({util:.0%})")
    
    # Followup Count
    followups = row.get('followup_count', None)
    if pd.notna(followups):
        if followups >= 3:
            reasons.append(f"Multiple followups ({int(followups)})")
    
    # App Usage
    app_usage = row.get('mobile_app_usage', None)
    if pd.notna(app_usage) and app_usage == 'High':
        reasons.append("High digital engagement")
    
    # Final reasoning based on probability
    if probability >= 0.7:
        reasons.insert(0, "High conversion likelihood")
    elif probability >= 0.5:
        reasons.insert(0, "Moderate conversion chance")
    else:
        reasons.insert(0, "Lower conversion probability")
    
    return " | ".join(reasons[:4]) if reasons else "Insufficient data for analysis"


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
    
    # Normalize credit_utilization_ratio (convert from percentage to decimal if > 1)
    if 'credit_utilization_ratio' in df_clean.columns:
        df_clean['credit_utilization_ratio'] = df_clean['credit_utilization_ratio'].apply(
            lambda x: x / 100 if pd.notna(x) and x > 1 else x
        )
    
    # Convert numeric columns to proper types
    numeric_cols = ['age', 'dependents_count', 'annual_income', 'pincode', 
                   'avg_monthly_app_visits', 'credit_card_spend_last_6m', 
                   'cibil_score', 'existing_loans_count', 'existing_monthly_emi',
                   'avg_monthly_balance', 'account_tenure_years', 'followup_count', 'data_year']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
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
        <h1>HDFC Bank Lead Prediction Dashboard</h1>
        <p>Upload your leads data, get AI-powered predictions, and download results</p>
    </div>
    <div class="hdfc-accent"></div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/28/HDFC_Bank_Logo.svg", width=200)
        st.markdown("---")
        st.markdown("### System Status")
        
        # Check API health
        health = check_api_health()
        if health and health.get('model_loaded'):
            st.success("API Connected")
            st.info(f"Model: {health.get('model_path', 'N/A')}")
        else:
            st.error("API Not Connected")
            st.warning("Start the API with:\n```\nuvicorn api:app --port 8000\n```")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard uses a **Random Forest** model trained on 120k+ HDFC leads 
        to predict which leads are likely to convert.
        
        **Features:**
        - Upload CSV data for batch predictions
        - Manual entry for individual leads
        - View reasoning for each prediction
        - Filter by product
        - Download results
        """)
        
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("[API Documentation](http://localhost:8000/docs)")
        st.markdown("[MLflow Dashboard](http://localhost:5000)")
    
    # Page Selection with Tabs
    tab1, tab2 = st.tabs(["Batch Upload", "Manual Entry"])
    
    # =================================================================
    # TAB 1: Batch Upload
    # =================================================================
    with tab1:
        st.markdown("### Upload Lead Data")
        
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
            st.success(f"Loaded {len(df):,} records with {len(df.columns)} columns")
            
            # Show data preview
            with st.expander("Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Prediction button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_button = st.button(
                    "Run Predictions",
                    type="primary",
                    use_container_width=True
                )
            
            if predict_button:
                # Check API
                if not check_api_health():
                    st.error("Cannot connect to the prediction API. Please ensure it's running.")
                    return
                
                # Run predictions
                with st.spinner("Running predictions... This may take a moment for large datasets."):
                    predictions = predict_batch(df)
                
                if predictions:
                    # Store in session state
                    results_df = create_results_dataframe(df, predictions)
                    st.session_state['results'] = results_df
                    st.session_state['predictions'] = predictions
                    st.success("Predictions completed successfully!")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return
    
    # Display results if available
    if 'results' in st.session_state:
        results_df = st.session_state['results']
        predictions = st.session_state['predictions']
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
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
                <div class="value" style="color: #ED1C24;">{non_conversions:,}</div>
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
        st.markdown("### Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Product filter
            products = ['All Products']
            if 'product_category' in results_df.columns:
                products += results_df['product_category'].dropna().unique().tolist()
            selected_product = st.selectbox("Product Category", products)
        
        with col2:
            # Conversion status filter
            status_filter = st.selectbox(
                "Conversion Status",
                ['All', 'Will Convert', 'Will Not Convert']
            )
        
        with col3:
            # Confidence filter
            confidence_filter = st.selectbox(
                "Confidence Level",
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
        st.info(f"Showing {len(filtered_df):,} of {len(results_df):,} records")
        
        # =================================================================
        # DYNAMIC INSIGHTS - 8 Key Metric Cards (same style as Prediction Results)
        # =================================================================
        if len(filtered_df) > 0:
            st.markdown("---")
            st.markdown("### Segment Insights")
            st.caption(f"*{selected_product} | {status_filter} | {confidence_filter}*")
            
            # Calculate all metrics
            filtered_total = len(filtered_df)
            filtered_conversions = len(filtered_df[filtered_df['Conversion_Status'] == 'Will Convert'])
            conversion_rate = (filtered_conversions / filtered_total * 100) if filtered_total > 0 else 0
            avg_probability = filtered_df['Probability'].mean() * 100 if 'Probability' in filtered_df.columns else 0
            high_prob_leads = len(filtered_df[filtered_df['Probability'] >= 0.7])
            priority_leads = len(filtered_df[(filtered_df['Probability'] >= 0.7) & (filtered_df['Confidence'] == 'High')])
            
            avg_cibil = filtered_df['cibil_score'].mean() if 'cibil_score' in filtered_df.columns else None
            avg_income = filtered_df['annual_income'].mean() if 'annual_income' in filtered_df.columns else None
            high_income = len(filtered_df[filtered_df['annual_income'] >= 1000000]) if 'annual_income' in filtered_df.columns else 0
            
            # Row 1: 4 Primary Metrics (same style as Prediction Results)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Filtered Leads</h3>
                    <div class="value">{filtered_total:,}</div>
                    <small style="color: #666;">{(filtered_total/len(results_df)*100):.1f}% of total</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Will Convert</h3>
                    <div class="value" style="color: #28a745;">{filtered_conversions:,}</div>
                    <small style="color: #28a745;">{conversion_rate:.1f}% rate</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Probability</h3>
                    <div class="value" style="color: #004C8F;">{avg_probability:.1f}%</div>
                    <small style="color: #666;">conversion chance</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Priority Leads</h3>
                    <div class="value" style="color: #ED1C24;">{priority_leads:,}</div>
                    <small style="color: #666;">high prob + conf</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Row 2: 4 Financial/Profile Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cibil_display = f"{avg_cibil:.0f}" if pd.notna(avg_cibil) else "N/A"
                cibil_color = "#28a745" if pd.notna(avg_cibil) and avg_cibil >= 700 else "#ED1C24"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg CIBIL Score</h3>
                    <div class="value" style="color: {cibil_color};">{cibil_display}</div>
                    <small style="color: #666;">{"Good" if pd.notna(avg_cibil) and avg_cibil >= 700 else "Below threshold"}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                income_display = f"Rs.{avg_income/100000:.1f}L" if pd.notna(avg_income) else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Income</h3>
                    <div class="value" style="color: #004C8F;">{income_display}</div>
                    <small style="color: #666;">per annum</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>High Probability</h3>
                    <div class="value" style="color: #28a745;">{high_prob_leads:,}</div>
                    <small style="color: #666;">70%+ chance</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>High Value Leads</h3>
                    <div class="value" style="color: #004C8F;">{high_income:,}</div>
                    <small style="color: #666;">income 10L+</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick Insight Summary
            if conversion_rate >= 70:
                insight_text = f"<strong>Excellent segment!</strong> {conversion_rate:.0f}% predicted to convert"
                insight_color = "#28a745"
            elif conversion_rate >= 50:
                insight_text = f"<strong>Good potential</strong> - {conversion_rate:.0f}% conversion rate"
                insight_color = "#004C8F"
            else:
                insight_text = f"<strong>Needs attention</strong> - Only {conversion_rate:.0f}% predicted to convert"
                insight_color = "#ED1C24"
            
            if priority_leads > 0:
                insight_text += f" | Focus on {priority_leads} priority leads"
            
            st.markdown(f"""
            <div style="background-color: {insight_color}20; border-left: 4px solid {insight_color}; 
                        padding: 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0;">
                {insight_text}
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.warning("No data matches the current filters. Try adjusting your selection.")
        
        st.markdown("---")
        
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
            st.caption(f"Note: Showing first 100 records. Download to see all {len(filtered_df):,} records.")
        
        st.markdown("---")
        
        # Download section
        st.markdown("### Download Results")
        
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
                        label=f"Download {product}",
                        data=product_csv,
                        file_name=f"leads_{product.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"download_{product}"
                    )
    
        else:
            # Show instructions when no data is loaded
            st.markdown("""
            <div class="info-box">
                <h4>How to use Batch Upload:</h4>
                <ol>
                    <li><strong>Ensure the API is running</strong> - Check the sidebar for connection status</li>
                    <li><strong>Upload your CSV file</strong> - Click the upload button above</li>
                    <li><strong>Click "Run Predictions"</strong> - The model will analyze each lead</li>
                    <li><strong>View results</strong> - See predictions with reasoning</li>
                    <li><strong>Download</strong> - Export results as CSV</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # =================================================================
    # TAB 2: Manual Entry
    # =================================================================
    with tab2:
        st.markdown("### Manual Lead Entry")
        st.markdown("Enter lead details below to get an individual prediction.")
        
        # Check API status first
        if not check_api_health():
            st.error("API is not connected. Please start the API first.")
        else:
            # Create form for manual entry
            with st.form("manual_entry_form"):
                st.markdown("#### Customer Demographics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                with col2:
                    age = st.number_input("Age", min_value=18, max_value=100, value=35)
                with col3:
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
                with col4:
                    dependents_count = st.number_input("Dependents", min_value=0, max_value=10, value=0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    education_level = st.selectbox("Education", ["10th", "12th", "Graduate", "Post Graduate", "Professional"])
                with col2:
                    occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Professional", "Student", "Retired"])
                with col3:
                    annual_income = st.number_input("Annual Income (Rs.)", min_value=0, max_value=50000000, value=500000, step=50000)
                
                st.markdown("---")
                st.markdown("#### Location Details")
                col1, col2 = st.columns(2)
                with col1:
                    city = st.text_input("City", value="Chennai")
                with col2:
                    pincode = st.number_input("Pincode", min_value=100000, max_value=999999, value=600001)
                
                st.markdown("---")
                st.markdown("#### Banking Details")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
                with col2:
                    credit_utilization_ratio = st.slider("Credit Utilization %", min_value=0, max_value=100, value=30) / 100
                with col3:
                    existing_loans_count = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
                with col4:
                    existing_monthly_emi = st.number_input("Monthly EMI (Rs.)", min_value=0, max_value=500000, value=0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_monthly_balance = st.number_input("Avg Monthly Balance (Rs.)", min_value=0, max_value=10000000, value=50000)
                with col2:
                    account_tenure_years = st.number_input("Account Tenure (years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
                with col3:
                    credit_card_spend_last_6m = st.number_input("Credit Card Spend (6 months)", min_value=0, max_value=5000000, value=50000)
                
                st.markdown("---")
                st.markdown("#### Digital Engagement")
                col1, col2, col3 = st.columns(3)
                with col1:
                    mobile_app_usage = st.selectbox("Mobile App Usage", ["None", "Low", "Medium", "High"])
                with col2:
                    netbanking_active = st.selectbox("Netbanking Active", ["Yes", "No"])
                with col3:
                    avg_monthly_app_visits = st.number_input("Avg Monthly App Visits", min_value=0, max_value=100, value=5)
                
                col1, col2 = st.columns(2)
                with col1:
                    preferred_language = st.selectbox("Preferred Language", ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Marathi", "Bengali", "Gujarati"])
                with col2:
                    contact_channel_preference = st.selectbox("Contact Preference", ["Email", "SMS", "Phone", "WhatsApp"])
                
                st.markdown("---")
                st.markdown("#### Lead Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    product_category = st.selectbox("Product Category", ["Home Loan", "Personal Loan", "Credit Card", "Vehicle Loan", "Insurance", "Deposits"])
                with col2:
                    sub_product = st.text_input("Sub Product", value="Standard")
                with col3:
                    lead_source = st.selectbox("Lead Source", ["Website", "Branch Walk-in", "Referral", "Digital", "Telemarketing", "Partner"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    campaign_name = st.text_input("Campaign Name", value="General")
                with col2:
                    website_lead_source = st.selectbox("Website Source", ["Organic", "Paid", "Referral", "Direct", "Social Media"])
                with col3:
                    followup_count = st.number_input("Followup Count", min_value=0, max_value=20, value=0)
                
                st.markdown("---")
                
                # Submit button
                submitted = st.form_submit_button("Get Prediction", type="primary", use_container_width=True)
                
                if submitted:
                    # Prepare data for API
                    lead_data = {
                        "gender": gender[0] if gender != "Other" else "O",  # M, F, O
                        "age": age,
                        "marital_status": marital_status,
                        "dependents_count": dependents_count,
                        "education_level": education_level,
                        "occupation": occupation,
                        "annual_income": annual_income,
                        "city": city,
                        "pincode": pincode,
                        "preferred_language": preferred_language,
                        "contact_channel_preference": contact_channel_preference,
                        "mobile_app_usage": mobile_app_usage,
                        "netbanking_active": netbanking_active,
                        "avg_monthly_app_visits": avg_monthly_app_visits,
                        "credit_card_spend_last_6m": credit_card_spend_last_6m,
                        "cibil_score": cibil_score,
                        "credit_utilization_ratio": credit_utilization_ratio,
                        "existing_loans_count": existing_loans_count,
                        "existing_monthly_emi": existing_monthly_emi,
                        "avg_monthly_balance": avg_monthly_balance,
                        "account_tenure_years": account_tenure_years,
                        "website_lead_source": website_lead_source,
                        "product_category": product_category,
                        "sub_product": sub_product,
                        "lead_source": lead_source,
                        "campaign_name": campaign_name,
                        "followup_count": followup_count,
                        "data_year": 2024
                    }
                    
                    # Make API call
                    with st.spinner("Getting prediction..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/predict",
                                json=lead_data,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                st.markdown("---")
                                st.markdown("### Prediction Result")
                                
                                # Display result in cards
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    prediction_text = "Will Convert" if result['prediction'] == 1 else "Will Not Convert"
                                    prediction_color = "#28a745" if result['prediction'] == 1 else "#ED1C24"
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Prediction</h3>
                                        <div class="value" style="color: {prediction_color};">{prediction_text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Probability</h3>
                                        <div class="value" style="color: #004C8F;">{result['probability']*100:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    confidence_color = "#28a745" if result['confidence'] == "High" else ("#004C8F" if result['confidence'] == "Medium" else "#ED1C24")
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Confidence</h3>
                                        <div class="value" style="color: {confidence_color};">{result['confidence']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Generate reasoning
                                lead_data['Probability'] = result['probability']
                                reasoning = get_reasoning(lead_data)
                                
                                st.markdown("#### Analysis")
                                st.info(reasoning)
                                
                            else:
                                st.error(f"API Error: {response.status_code} - {response.text}")
                        
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")


if __name__ == "__main__":
    main()

