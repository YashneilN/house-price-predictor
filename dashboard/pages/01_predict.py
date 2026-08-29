"""Prediction page — form inputs, predicted price gauge, feature importances, history."""

import pandas as pd
import requests
import streamlit as st

from dashboard.components.charts import (
    create_feature_importance_bar,
    create_gauge,
    create_prediction_history_line,
)

st.set_page_config(page_title="Predict — House Price Predictor", page_icon="📝", layout="wide")
st.title("📝 Predict House Price")

API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

NEIGHBORHOODS = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt", "Sawyer",
    "NWAmes", "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR",
    "ClearCr", "StoneBr", "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste",
]
QUALITY_OPTIONS = ["Ex", "Gd", "TA", "Fa", "Po"]

with st.form("prediction_form"):
    st.subheader("House Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        gr_liv_area = st.number_input("Above-Grade Living Area (sq ft)", min_value=200, max_value=6000, value=1500)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=800)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=50000, value=9000)
        year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=2000)
        year_remod = st.number_input("Year Remodeled", min_value=1870, max_value=2026, value=2005)

    with c2:
        full_bath = st.slider("Full Bathrooms", 0, 5, 2)
        half_bath = st.slider("Half Bathrooms", 0, 3, 1)
        bedrooms = st.slider("Bedrooms Above Grade", 0, 10, 3)
        tot_rooms = st.slider("Total Rooms Above Grade", 1, 15, 6)
        fireplaces = st.slider("Fireplaces", 0, 4, 1)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)

    with c3:
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=480)
        neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=0)
        exter_qual = st.selectbox("Exterior Quality", QUALITY_OPTIONS, index=2)
        kitchen_qual = st.selectbox("Kitchen Quality", QUALITY_OPTIONS, index=2)
        bsmt_qual = st.selectbox("Basement Quality", QUALITY_OPTIONS, index=2)
        central_air = st.selectbox("Central Air", ["Y", "N"], index=0)

    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

if submitted:
    payload = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "TotRmsAbvGrd": tot_rooms,
        "Fireplaces": fireplaces,
        "LotArea": lot_area,
        "BedroomAbvGr": bedrooms,
        "Neighborhood": neighborhood,
        "ExterQual": exter_qual,
        "KitchenQual": kitchen_qual,
        "BsmtQual": bsmt_qual,
        "CentralAir": central_air,
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't reach the API: {exc}")
        st.stop()

    if resp.status_code == 503:
        st.warning("No trained model yet. Go to **Training Monitor** and train one first.")
        st.stop()
    elif not resp.ok:
        st.error(f"Prediction failed: {resp.text}")
        st.stop()

    result = resp.json()
    st.session_state["last_prediction"] = result

    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        fig = create_gauge(
            result["predicted_price"],
            min_val=result["confidence_interval_low"] * 0.8,
            max_val=result["confidence_interval_high"] * 1.2,
            title=f"Predicted Price ({result['model_used']})",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.metric("Predicted Price", f"${result['predicted_price']:,.0f}")
        st.caption(
            f"95% range: ${result['confidence_interval_low']:,.0f} — "
            f"${result['confidence_interval_high']:,.0f}"
        )
        if result.get("top_feature_importances"):
            fig_imp = create_feature_importance_bar(result["top_feature_importances"])
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.caption("Feature importances aren't available for this model type.")

st.divider()
st.subheader("📈 Prediction History")

auto_refresh = st.checkbox("Auto-refresh every 5s", value=False)

try:
    hist_resp = requests.get(f"{API_BASE_URL}/predictions/history", params={"limit": 50}, timeout=10)
    if hist_resp.ok:
        history = hist_resp.json()["predictions"]
        if history:
            df_hist = pd.DataFrame(history)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist = df_hist.sort_values("timestamp")

            fig_hist = create_prediction_history_line(df_hist["timestamp"], df_hist["predicted_price"])
            st.plotly_chart(fig_hist, use_container_width=True)

            st.dataframe(
                df_hist[["id", "timestamp", "predicted_price", "inputs_summary"]]
                .sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No predictions yet — submit the form above to get started.")
    else:
        st.warning("Couldn't load prediction history.")
except requests.exceptions.RequestException as exc:
    st.warning(f"Couldn't reach the API for history: {exc}")

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
