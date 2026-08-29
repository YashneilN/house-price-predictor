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
from dashboard.theme import apply_theme

apply_theme()
st.title("⚙️ Model Training")
st.caption("Cross-Validation Convergence · Hyperparameter Tuning · Ensemble Benchmarking")

API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

ALL_MODELS = [
    "LinearRegression", "Ridge", "Lasso", "RandomForest",
    "GradientBoosting", "XGBoost", "LightGBM",
]

col_trigger, col_status = st.columns([1, 2])

with col_trigger:
    st.subheader("Start a Training Run")
    selected_models = st.multiselect("Models to Train", ALL_MODELS, default=ALL_MODELS)
    start_clicked = st.button("🚀 Start Training", use_container_width=True, type="primary")

    if start_clicked:
        try:
            resp = requests.post(
                f"{API_BASE_URL}/train",
                json={"models": selected_models or None},
                timeout=10,
            )
            if resp.status_code == 202:
                st.success("Training initiated! Live metrics below will update automatically.")
                st.session_state["training_started"] = True
            elif resp.status_code == 409:
                st.warning("A training session is already in progress.")
            else:
                st.error(f"Could not start training: {resp.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach API: {exc}")

with col_status:
    st.subheader("Execution Status")
    status_placeholder = st.empty()

st.divider()

# ── Fetch current metrics ───────────────────────────────────────────────
try:
    metrics_resp = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
    metrics = metrics_resp.json() if metrics_resp.ok else {}
except requests.exceptions.RequestException as exc:
    st.error(f"Could not reach API: {exc}")
    metrics = {}

status = metrics.get("status", "idle")
results = metrics.get("results", [])
current_model = metrics.get("current_model")
models_total = metrics.get("models_total") or len(ALL_MODELS)
models_completed = metrics.get("models_completed", 0)

with status_placeholder.container():
    status_emoji = {"idle": "⚪", "training": "🟡", "complete": "🟢", "failed": "🔴"}.get(status, "⚪")
    st.markdown(f"**{status_emoji} Status:** `{status.title()}`")
    if status == "training":
        st.markdown(f"**Current Model:** `{current_model}`")
        st.progress(models_completed / models_total if models_total else 0)
        st.caption(f"{models_completed} of {models_total} Models Completed")
    elif status == "complete":
        st.markdown(f"**✅ Best Model:** `{metrics.get('best_model')}`")
    elif status == "failed":
        st.error(f"Training Failed: {metrics.get('error', 'Unknown error')}")

if results:
    st.subheader("📊 Live Training Metrics")

    tab1, tab2, tab3 = st.tabs(["RMSE Convergence", "Model Comparison", "Raw Training Logs"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(create_metric_line_chart(results, "cv_rmse_mean", "Mean CV RMSE by Model"), use_container_width=True)
        with col_b:
            st.plotly_chart(create_fold_convergence_chart(results), use_container_width=True)

    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(create_comparison_bar(results, "cv_rmse_mean", "Model Comparison — CV RMSE (Lower Is Better)"), use_container_width=True)
        with col_d:
            st.plotly_chart(create_comparison_bar(results, "cv_r2_mean", "Model Comparison — CV R² Score (Higher Is Better)"), use_container_width=True)

    with tab3:
        df_results = pd.DataFrame([
            {
                "Model": r["model"],
                "CV RMSE (Log Scale)": round(r["cv_rmse_mean"], 4),
                "CV MAE (Log Scale)": round(r["cv_mae_mean"], 4),
                "CV R² Score": round(r["cv_r2_mean"], 4),
                "Training Duration (s)": r["training_seconds"],
                "Timestamp": r["timestamp"],
            }
            for r in results
        ])
        st.dataframe(df_results, use_container_width=True, hide_index=True)
else:
    st.info("No training runs recorded yet. Click **Start Training** above to commence model evaluation.")

# ── Auto-refresh while training is in progress ──────────────────────────
if status == "training":
    time.sleep(5)
    st.rerun()
