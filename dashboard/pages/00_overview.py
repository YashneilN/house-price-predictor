"""Overview page — system health, architecture summary, and workflow guide."""

from __future__ import annotations

import os

import requests
import streamlit as st

from dashboard.theme import apply_theme

apply_theme()

API_BASE_URL = st.session_state.get("api_base_url", os.environ.get("API_BASE_URL", "http://localhost:8000"))

st.title("🏠 Ames Housing AI")
st.caption("Real Estate Valuation & Analytics · Stacked ML Ensemble · Ames Housing Dataset")

col1, col2, col3 = st.columns(3)

try:
    resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    healthy = resp.ok
    model_loaded = resp.json().get("model_loaded", False) if healthy else False
except requests.exceptions.RequestException:
    healthy = False
    model_loaded = False

with col1:
    st.metric("API Status", "Online" if healthy else "Offline")
with col2:
    st.metric("Model Status", "Stacked / Trained" if model_loaded else "Not Trained Yet")
with col3:
    st.metric("Backend Endpoint", API_BASE_URL)

st.divider()

if not healthy:
    st.error(
        f"Unable to reach the FastAPI backend at `{API_BASE_URL}`. "
        "Start the server with:\n\n```bash\nuvicorn api.main:app --reload\n```"
    )
elif not model_loaded:
    st.warning(
        "No trained model found. Navigate to **Model Training** in the sidebar, ensure `train.csv` is present in "
        "`data/raw/`, and select **Start Training**."
    )
else:
    st.success("Backend is online and a model is loaded. Navigate to **Price Estimator** to test predictions.")

st.markdown(
    """
    ### System Architecture & Workflow

    1. **Exploratory Data Analysis (EDA)**
       - Comprehensive statistical breakdown of the 79 Ames residential features.
       - Correlation heatmaps, feature distributions, and missing data diagnostics.

    2. **Model Training & Validation**
       - Automated cross-validation with hyperparameter search across multiple regressor families.
       - Real-time convergence tracking and performance benchmarking (RMSE, MAE, R²).
       - Serving an optimized **50% XGBoost + 30% LightGBM + 20% Ridge** stacked ensemble.

    3. **Price Estimator**
       - Interactive property valuation with predefined archetype presets (*Starter Home*, *Suburban Average*, *Luxury Estate*).
       - 95% confidence interval bounds, price-per-square-foot metrics, and model feature importance attribution.
    """
)
