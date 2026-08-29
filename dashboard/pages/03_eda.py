"""EDA page — correlation heatmap, distributions, missing values, neighborhood comparison."""

import pandas as pd
import requests
import streamlit as st

from dashboard.components.charts import create_correlation_bar, create_histogram
from dashboard.theme import apply_theme

apply_theme()
st.title("📈 Exploratory Data Analysis")
st.caption("Distribution Analysis · Correlation Matrices · Feature Diagnostics")

API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")


def fetch(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=15)
        if resp.ok:
            return resp.json()
        st.error(f"Request to {path} failed: {resp.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't reach the API: {exc}")
    return None


summary = fetch("/eda/summary")

if summary is None:
    st.warning("Could not load dataset summary. Ensure `data/raw/train.csv` is present and the API is running.")
    st.stop()

st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{summary['n_rows']:,}")
c2.metric("Total Columns", summary["n_columns"])
c3.metric("Numerical Features", summary["numeric_columns"])
c4.metric("Categorical Features", summary["categorical_columns"])

st.subheader("Sale Price Statistics")
ts = summary["target_stats"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Price", f"${ts['mean']:,.0f}")
c2.metric("Median Price", f"${ts['median']:,.0f}")
c3.metric("Standard Deviation", f"${ts['std']:,.0f}")
c4.metric("Minimum Price", f"${ts['min']:,.0f}")
c5.metric("Maximum Price", f"${ts['max']:,.0f}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔗 Top Correlations with Sale Price")
    corr_data = fetch("/eda/correlations", params={"top_n": 15})
    if corr_data:
        st.plotly_chart(create_correlation_bar(corr_data["correlations"]), use_container_width=True)

with col_right:
    st.subheader("❓ Missing Values (Top 10 Features)")
    missing = summary.get("missing_pct_top10", {})
    if missing and max(missing.values()) > 0:
        df_missing = pd.DataFrame(list(missing.items()), columns=["Feature Column", "Missing Percentage (%)"])
        df_missing = df_missing[df_missing["Missing Percentage (%)"] > 0]
        if not df_missing.empty:
            st.bar_chart(df_missing.set_index("Feature Column"))
        else:
            st.success("No missing values found in this dataset! 🎉")
    else:
        st.success("No missing values found in this dataset! 🎉")

st.divider()
st.subheader("📉 Feature Distributions")

dist_data = fetch("/eda/distributions", params={"bins": 30})
if dist_data:
    st.markdown("**Sale Price Distribution**")
    target = dist_data["target"]
    st.plotly_chart(
        create_histogram(target["bin_edges"], target["counts"], "Sale Price Distribution", "Sale Price ($)"),
        use_container_width=True,
    )

    st.markdown("**Numerical Feature Distributions**")
    features = dist_data["features"]
    feature_names = list(features.keys())

    cols = st.columns(3)
    for i, fname in enumerate(feature_names):
        fdata = features[fname]
        with cols[i % 3]:
            st.plotly_chart(
                create_histogram(fdata["bin_edges"], fdata["counts"], fname, fname),
                use_container_width=True,
            )
