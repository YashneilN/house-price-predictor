"""
Ames Housing AI — Commercial Real Estate Valuation & Analytics Dashboard

Main entry point and multi-page router with custom branded sidebar navigation,
live service health telemetry, and zinc/slate dark UI theme.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

from dashboard.theme import apply_theme

# Top-level page configuration
st.set_page_config(
    page_title="Ames Housing AI",
    page_icon="🏠",
    layout="wide",
)
apply_theme()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = API_BASE_URL

# ── Sidebar Branding Header ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-top: -8px; margin-bottom: 8px;">
            <div style="font-size: 2rem; line-height: 1;">🏠</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">Ames Housing AI</div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 400;">Real Estate Valuation & Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

# ── Navigation Pages Definition ──────────────────────────────────────────
PAGES_DIR = Path(__file__).parent / "pages"

overview_page = st.Page(
    PAGES_DIR / "00_overview.py",
    title="Overview",
    icon="🏠",
    default=True,
)
predict_page = st.Page(
    PAGES_DIR / "01_predict.py",
    title="Price Estimator",
    icon="📊",
)
train_page = st.Page(
    PAGES_DIR / "02_training.py",
    title="Model Training",
    icon="⚙️",
)
eda_page = st.Page(
    PAGES_DIR / "03_eda.py",
    title="Exploratory Data Analysis",
    icon="📈",
)

pg = st.navigation([overview_page, predict_page, train_page, eda_page])

# ── Sidebar Footer & Telemetry Metadata ──────────────────────────────────
with st.sidebar:
    st.divider()

    # Telemetry health check
    try:
        resp = requests.get(f"{st.session_state['api_base_url']}/health", timeout=2)
        api_connected = resp.ok
    except Exception:
        api_connected = False

    if api_connected:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.82rem; font-weight: 600; color: #10b981;">
                <span style="font-size: 0.75rem;">🟢</span> API Connected
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.82rem; font-weight: 600; color: #ef4444;">
                <span style="font-size: 0.75rem;">🔴</span> API Disconnected
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Model Type:</span>
                <span style="color: #f8fafc; font-weight: 500;">Stacked Ensemble</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Dataset:</span>
                <span style="color: #f8fafc; font-weight: 500;">Ames Iowa Housing</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 6px; padding-top: 6px; border-top: 1px solid #334155;">
                <span style="color: #94a3b8;">Source Code:</span>
                <a href="https://github.com/Yashneil/house-price-predictor" target="_blank" style="color: #6366f1; text-decoration: none; font-weight: 500;">GitHub ↗</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
