"""Shared zinc/slate Streamlit theme (OPTIMIZATION_PROMPT design tokens)."""

from __future__ import annotations

import streamlit as st

ACCENT = "#6366f1"
BG = "#0f172a"
CARD = "#1e293b"
BORDER = "#334155"
TEXT = "#f8fafc"
MUTED = "#94a3b8"

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], .stMarkdown, .stText {{
  font-family: Inter, system-ui, -apple-system, sans-serif !important;
}}

.stApp {{
  background-color: {BG};
  color: {TEXT};
}}

[data-testid="stHeader"] {{
  background: {BG};
}}

[data-testid="stSidebar"] {{
  background-color: {CARD};
  border-right: 1px solid {BORDER};
}}

[data-testid="stSidebar"] * {{
  color: {TEXT};
}}

h1, h2, h3, h4 {{
  color: {TEXT} !important;
  letter-spacing: -0.02em;
}}

p, label, span, .stCaption, [data-testid="stCaption"] {{
  color: {MUTED};
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {CARD};
  border: 1px solid {BORDER} !important;
  border-radius: 12px;
  padding: 16px !important;
}}

[data-testid="stMetric"] {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 16px;
}}

[data-testid="stMetricValue"] {{
  color: {TEXT} !important;
}}

[data-testid="stMetricLabel"] {{
  color: {MUTED} !important;
}}

div[data-testid="stSlider"] {{
  padding-bottom: 8px;
}}

.stButton > button {{
  background: {ACCENT};
  color: {TEXT};
  border: 1px solid {ACCENT};
  border-radius: 8px;
  padding: 8px 16px;
  font-weight: 600;
}}

.stButton > button:hover {{
  filter: brightness(1.08);
}}

hr {{
  border-color: {BORDER} !important;
}}

.kpi-hero {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 16px;
}}

.kpi-price {{
  font-size: 2.4rem;
  font-weight: 700;
  color: {TEXT};
  margin: 0;
  letter-spacing: -0.03em;
}}

.kpi-label {{
  color: {MUTED};
  font-size: 0.85rem;
  margin-bottom: 8px;
}}

.kpi-badges {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}}

.kpi-badge {{
  background: {BG};
  border: 1px solid {BORDER};
  color: {TEXT};
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 0.8rem;
}}

.kpi-badge span {{
  color: {MUTED};
}}

[data-testid="stSidebarNav"] {{
  padding-top: 4px;
  padding-bottom: 4px;
}}

[data-testid="stSidebarNav"] a {{
  border-radius: 8px;
  margin-bottom: 3px;
  transition: all 0.15s ease-in-out;
}}

[data-testid="stSidebarNav"] a:hover {{
  background-color: {BORDER} !important;
}}

[data-testid="stSidebarNav"] a[aria-current="page"] {{
  background-color: rgba(99, 102, 241, 0.18) !important;
  border-left: 3px solid {ACCENT} !important;
}}
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
