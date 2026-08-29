"""
House Price Predictor — Streamlit Dashboard

Main entry point. Streamlit's native multi-page app support picks up
everything in pages/ automatically and lists them in the sidebar; this
file just sets page config and shows a landing/overview screen.
"""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# API base URL, overridable via env var so the dashboard can point at a
# deployed backend without code changes.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = API_BASE_URL

st.title("🏠 House Price Predictor")
st.caption("Full-stack ML pipeline — FastAPI backend, scikit-learn/XGBoost models, live training dashboard")

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state["api_base_url"] = st.text_input("API Base URL", value=st.session_state["api_base_url"])
    st.divider()
    st.markdown(
        "**Pages**\n\n"
        "- 📝 Predict — get a price estimate\n"
        "- 📈 Training Monitor — train models, watch live metrics\n"
        "- 📊 EDA — explore the dataset\n\n"
        "Use the sidebar navigation above to switch pages."
    )

col1, col2, col3 = st.columns(3)

try:
    resp = requests.get(f"{st.session_state['api_base_url']}/health", timeout=3)
    healthy = resp.ok
    model_loaded = resp.json().get("model_loaded", False) if healthy else False
except requests.exceptions.RequestException:
    healthy = False
    model_loaded = False

with col1:
    st.metric("API Status", "🟢 Online" if healthy else "🔴 Offline")
with col2:
    st.metric("Model Status", "✅ Trained" if model_loaded else "⚠️ Not trained yet")
with col3:
    st.metric("Backend", st.session_state["api_base_url"])

st.divider()

if not healthy:
    st.error(
        f"Can't reach the FastAPI backend at `{st.session_state['api_base_url']}`. "
        "Start it with:\n\n```bash\nuvicorn api.main:app --reload\n```"
    )
elif not model_loaded:
    st.warning(
        "No trained model yet. Go to **Training Monitor** in the sidebar, add `train.csv` to "
        "`data/raw/`, and click **Start Training**."
    )
else:
    st.success("Backend is online and a model is loaded. Head to **Predict** to try it out, "
               "or **EDA** to explore the dataset.")

st.markdown(
    """
    ### How this works
    1. **EDA** — explore the Ames Housing dataset: distributions, correlations, missing data.
    2. **Training Monitor** — trigger training; a background job cross-validates and tunes
       several models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost), and
       this page polls `GET /metrics` every 5 seconds to chart progress live.
    3. **Predict** — fill in house features, get a predicted price with a confidence range
       and the features that drove the prediction.
    """
)
