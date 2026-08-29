"""Prediction page — zinc layout, presets, quality sync, KPI + gauge, importance."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from dashboard.components.charts import (
    create_feature_importance_bar,
    create_gauge,
    create_prediction_history_line,
)
from dashboard.theme import apply_theme

apply_theme()

st.title("📊 Price Estimator")
st.caption("Stacked Ensemble · Log-Rescaled USD · Domain Features")

API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")

NEIGHBORHOODS = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt", "Sawyer",
    "NWAmes", "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR",
    "ClearCr", "StoneBr", "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste",
]
QUALITY_OPTIONS = ["Ex", "Gd", "TA", "Fa", "Po"]
QUALITY_DISPLAY = {
    "Ex": "Excellent (Ex)",
    "Gd": "Good (Gd)",
    "TA": "Typical / Average (TA)",
    "Fa": "Fair (Fa)",
    "Po": "Poor (Po)",
}
CENTRAL_AIR_DISPLAY = {
    "Y": "Yes (Y)",
    "N": "No (N)",
}

PRESETS = {
    "Custom": {},
    "Starter Home": {
        "overall_qual": 4,
        "gr_liv_area": 1100,
        "total_bsmt_sf": 500,
        "lot_area": 6200,
        "year_built": 1972,
        "year_remod": 1972,
        "full_bath": 1,
        "half_bath": 0,
        "bedrooms": 2,
        "tot_rooms": 5,
        "fireplaces": 0,
        "garage_cars": 1,
        "garage_area": 240,
        "neighborhood": "Edwards",
        "central_air": "Y",
    },
    "Suburban Average": {
        "overall_qual": 6,
        "gr_liv_area": 1550,
        "total_bsmt_sf": 850,
        "lot_area": 9500,
        "year_built": 1998,
        "year_remod": 2004,
        "full_bath": 2,
        "half_bath": 1,
        "bedrooms": 3,
        "tot_rooms": 6,
        "fireplaces": 1,
        "garage_cars": 2,
        "garage_area": 480,
        "neighborhood": "NAmes",
        "central_air": "Y",
    },
    "Luxury Estate": {
        "overall_qual": 9,
        "gr_liv_area": 3200,
        "total_bsmt_sf": 1800,
        "lot_area": 18000,
        "year_built": 2014,
        "year_remod": 2016,
        "full_bath": 3,
        "half_bath": 1,
        "bedrooms": 5,
        "tot_rooms": 10,
        "fireplaces": 2,
        "garage_cars": 3,
        "garage_area": 850,
        "neighborhood": "NridgHt",
        "central_air": "Y",
    },
}


def quality_from_overall(overall: int) -> str:
    if overall >= 9:
        return "Ex"
    if overall >= 7:
        return "Gd"
    if overall >= 5:
        return "TA"
    return "Fa"


def apply_preset() -> None:
    preset = st.session_state.get("house_preset", "Custom")
    values = PRESETS.get(preset) or {}
    for key, value in values.items():
        st.session_state[key] = value
    if "overall_qual" in values:
        mapped = quality_from_overall(int(values["overall_qual"]))
        st.session_state["exter_qual"] = mapped
        st.session_state["kitchen_qual"] = mapped
        st.session_state["bsmt_qual"] = mapped


def sync_quality_from_overall() -> None:
    mapped = quality_from_overall(int(st.session_state.get("overall_qual", 6)))
    st.session_state["exter_qual"] = mapped
    st.session_state["kitchen_qual"] = mapped
    st.session_state["bsmt_qual"] = mapped


_DEFAULTS = {
    "house_preset": "Custom",
    "overall_qual": 6,
    "gr_liv_area": 1500,
    "total_bsmt_sf": 800,
    "lot_area": 9000,
    "year_built": 2000,
    "year_remod": 2005,
    "full_bath": 2,
    "half_bath": 1,
    "bedrooms": 3,
    "tot_rooms": 6,
    "fireplaces": 1,
    "garage_cars": 2,
    "garage_area": 480,
    "neighborhood": "NAmes",
    "central_air": "Y",
    "kitchen_qual": "TA",
    "exter_qual": "TA",
    "bsmt_qual": "TA",
}
for _key, _default in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

result = st.session_state.get("last_prediction")

if result:
    price = result["predicted_price"]
    low = result["confidence_interval_low"]
    high = result["confidence_interval_high"]
    pps = result.get("price_per_sqft") or (
        price / max(float(st.session_state.get("gr_liv_area") or 1), 1)
    )
    half_width = (high - low) / 2
    deviation_pct = (half_width / price * 100) if price else 0

    kpi, gauge = st.columns([1, 1.1], gap="medium")
    with kpi:
        st.markdown(
            f"""
            <div class="kpi-hero">
              <div class="kpi-label">Stacked Price Estimate</div>
              <p class="kpi-price">${price:,.0f}</p>
              <div class="kpi-badges">
                <div class="kpi-badge"><span>95% CI:</span> ${low:,.0f} – ${high:,.0f}</div>
                <div class="kpi-badge"><span>Price / Sq Ft:</span> ${pps:,.0f}</div>
                <div class="kpi-badge"><span>Deviation:</span> ±{deviation_pct:.1f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        subs = result.get("sub_model_predictions") or {}
        if subs:
            st.caption("Sub-Model Estimates")
            cols = st.columns(max(len(subs), 1))
            for col, (name, value) in zip(cols, subs.items()):
                col.metric(name, f"${value:,.0f}")
    with gauge:
        st.plotly_chart(
            create_gauge(
                price,
                min_val=low * 0.85,
                max_val=high * 1.15,
                title=f"{result.get('model_used', 'Ensemble')} Gauge",
            ),
            use_container_width=True,
        )

controls, outputs = st.columns([1.15, 1], gap="medium")

with controls:
    with st.container(border=True):
        st.markdown("**House Profile**")
        st.selectbox(
            "Preset Profile",
            list(PRESETS.keys()),
            key="house_preset",
            on_change=apply_preset,
            help="Starter Home, Suburban Average, and Luxury Estate populate realistic defaults. Custom allows full manual adjustment.",
        )
        st.slider(
            "Overall Quality",
            1, 10,
            key="overall_qual",
            on_change=sync_quality_from_overall,
            help="Overall material and finish rating (1–10). Kitchen, Exterior, and Basement qualities synchronize automatically.",
        )

        left, right = st.columns(2, gap="small")
        with left:
            st.number_input("Living Area (Sq Ft)", min_value=200, max_value=6000, key="gr_liv_area")
            st.number_input("Basement Area (Sq Ft)", min_value=0, max_value=4000, key="total_bsmt_sf")
            st.number_input("Lot Area (Sq Ft)", min_value=500, max_value=50000, key="lot_area")
            st.number_input("Year Built", min_value=1870, max_value=2026, key="year_built")
            st.number_input("Year Remodeled", min_value=1870, max_value=2026, key="year_remod")
            st.slider("Full Bathrooms", 0, 5, key="full_bath")
            st.slider("Half Bathrooms", 0, 3, key="half_bath")
        with right:
            st.slider("Bedrooms Above Grade", 0, 10, key="bedrooms")
            st.slider("Total Rooms Above Grade", 1, 15, key="tot_rooms")
            st.slider("Fireplaces", 0, 4, key="fireplaces")
            st.slider("Garage Capacity (Cars)", 0, 5, key="garage_cars")
            st.number_input("Garage Area (Sq Ft)", min_value=0, max_value=1500, key="garage_area")
            st.selectbox("Neighborhood", NEIGHBORHOODS, key="neighborhood")
            st.selectbox(
                "Central Air Conditioning",
                ["Y", "N"],
                key="central_air",
                format_func=lambda x: CENTRAL_AIR_DISPLAY.get(x, x),
            )

        q1, q2, q3 = st.columns(3)
        with q1:
            st.selectbox(
                "Kitchen Quality",
                QUALITY_OPTIONS,
                key="kitchen_qual",
                format_func=lambda x: QUALITY_DISPLAY.get(x, x),
            )
        with q2:
            st.selectbox(
                "Exterior Quality",
                QUALITY_OPTIONS,
                key="exter_qual",
                format_func=lambda x: QUALITY_DISPLAY.get(x, x),
            )
        with q3:
            st.selectbox(
                "Basement Quality",
                QUALITY_OPTIONS,
                key="bsmt_qual",
                format_func=lambda x: QUALITY_DISPLAY.get(x, x),
            )

        predict_clicked = st.button("Predict Price", use_container_width=True, type="primary")

if predict_clicked:
    payload = {
        "OverallQual": int(st.session_state["overall_qual"]),
        "GrLivArea": float(st.session_state["gr_liv_area"]),
        "TotalBsmtSF": float(st.session_state["total_bsmt_sf"]),
        "GarageCars": int(st.session_state["garage_cars"]),
        "GarageArea": float(st.session_state["garage_area"]),
        "YearBuilt": int(st.session_state["year_built"]),
        "YearRemodAdd": int(st.session_state["year_remod"]),
        "FullBath": int(st.session_state["full_bath"]),
        "HalfBath": int(st.session_state["half_bath"]),
        "TotRmsAbvGrd": int(st.session_state["tot_rooms"]),
        "Fireplaces": int(st.session_state["fireplaces"]),
        "LotArea": float(st.session_state["lot_area"]),
        "BedroomAbvGr": int(st.session_state["bedrooms"]),
        "Neighborhood": st.session_state["neighborhood"],
        "ExterQual": st.session_state["exter_qual"],
        "KitchenQual": st.session_state["kitchen_qual"],
        "BsmtQual": st.session_state["bsmt_qual"],
        "CentralAir": st.session_state["central_air"],
        "OverallCond": 5,
        "1stFlrSF": float(st.session_state["gr_liv_area"]),
        "2ndFlrSF": 0,
        "YrSold": 2024,
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to API: {exc}")
        st.stop()

    if resp.status_code == 503:
        st.warning("No trained model found. Please go to **Training Monitor** and train a model first.")
        st.stop()
    elif not resp.ok:
        st.error(f"Prediction request failed: {resp.text}")
        st.stop()

    st.session_state["last_prediction"] = resp.json()
    st.rerun()

with outputs:
    if not result:
        with st.container(border=True):
            st.markdown("**Feature Importance**")
            st.caption("Generate a prediction above to view relative feature contributions.")
    else:
        with st.container(border=True):
            importances = result.get("top_feature_importances") or {}
            if importances:
                st.plotly_chart(
                    create_feature_importance_bar(importances),
                    use_container_width=True,
                )
            else:
                st.caption("Feature importances are not available for this model type.")

st.divider()
st.subheader("Prediction History")

auto_refresh = st.checkbox("Auto-Refresh Every 5 Seconds", value=False)

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

            df_display = df_hist[["id", "timestamp", "predicted_price", "inputs_summary"]].rename(
                columns={
                    "id": "Prediction ID",
                    "timestamp": "Timestamp",
                    "predicted_price": "Predicted Price ($)",
                    "inputs_summary": "Inputs Summary",
                }
            ).sort_values("Timestamp", ascending=False)

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No predictions recorded yet. Submit the form above to generate your first prediction.")
    else:
        st.warning("Unable to load prediction history.")
except requests.exceptions.RequestException as exc:
    st.warning(f"Could not reach API for prediction history: {exc}")

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
