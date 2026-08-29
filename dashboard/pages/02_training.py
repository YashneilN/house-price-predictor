"""Training Monitor page — trigger training, poll /metrics every 5s, show live charts."""

import time

import pandas as pd
import requests
import streamlit as st

from dashboard.components.charts import (
    create_comparison_bar,
    create_fold_convergence_chart,
    create_metric_line_chart,
)

st.set_page_config(page_title="Training Monitor — House Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Training Monitor")

API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

ALL_MODELS = ["LinearRegression", "Ridge", "Lasso", "RandomForest", "GradientBoosting", "XGBoost"]

col_trigger, col_status = st.columns([1, 2])

with col_trigger:
    st.subheader("Start a Training Run")
    selected_models = st.multiselect("Models to train", ALL_MODELS, default=ALL_MODELS)
    start_clicked = st.button("🚀 Start Training", use_container_width=True, type="primary")

    if start_clicked:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/train",
                json={"models": selected_models or None},
                timeout=10,
            )
            if resp.status_code == 202:
                st.success("Training started! Metrics below will update automatically.")
                st.session_state["training_started"] = True
            elif resp.status_code == 409:
                st.warning("Training is already running.")
            else:
                st.error(f"Couldn't start training: {resp.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Couldn't reach the API: {exc}")

with col_status:
    st.subheader("Status")
    status_placeholder = st.empty()

st.divider()

# ── Fetch current metrics ───────────────────────────────────────────────
try:
    metrics_resp = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
    metrics = metrics_resp.json() if metrics_resp.ok else {}
except requests.exceptions.RequestException as exc:
    st.error(f"Couldn't reach the API: {exc}")
    metrics = {}

status = metrics.get("status", "idle")
results = metrics.get("results", [])
current_model = metrics.get("current_model")
models_total = metrics.get("models_total") or len(ALL_MODELS)
models_completed = metrics.get("models_completed", 0)

with status_placeholder.container():
    status_emoji = {"idle": "⚪", "training": "🟡", "complete": "🟢", "failed": "🔴"}.get(status, "⚪")
    st.markdown(f"**{status_emoji} Status:** `{status}`")
    if status == "training":
        st.markdown(f"**Current model:** `{current_model}`")
        st.progress(models_completed / models_total if models_total else 0)
        st.caption(f"{models_completed} / {models_total} models complete")
    elif status == "complete":
        st.markdown(f"**✅ Best model:** `{metrics.get('best_model')}`")
    elif status == "failed":
        st.error(f"Training failed: {metrics.get('error', 'unknown error')}")

if results:
    st.subheader("📊 Live Training Metrics")

    tab1, tab2, tab3 = st.tabs(["RMSE Convergence", "Model Comparison", "Raw Log"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(create_metric_line_chart(results, "cv_rmse_mean", "Mean CV RMSE by Model (as trained)"), use_container_width=True)
        with col_b:
            st.plotly_chart(create_fold_convergence_chart(results), use_container_width=True)

    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(create_comparison_bar(results, "cv_rmse_mean", "Model Comparison — CV RMSE (lower is better)"), use_container_width=True)
        with col_d:
            st.plotly_chart(create_comparison_bar(results, "cv_r2_mean", "Model Comparison — CV R² (higher is better)"), use_container_width=True)

    with tab3:
        df_results = pd.DataFrame([
            {
                "Model": r["model"],
                "CV RMSE": round(r["cv_rmse_mean"], 4),
                "CV MAE": round(r["cv_mae_mean"], 4),
                "CV R²": round(r["cv_r2_mean"], 4),
                "Training Time (s)": r["training_seconds"],
                "Timestamp": r["timestamp"],
            }
            for r in results
        ])
        st.dataframe(df_results, use_container_width=True, hide_index=True)
else:
    st.info("No training runs yet. Click **Start Training** above (make sure `data/raw/train.csv` exists).")

# ── Auto-refresh while training is in progress ──────────────────────────
if status == "training":
    time.sleep(5)
    st.rerun()
