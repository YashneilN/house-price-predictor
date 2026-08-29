"""Reusable Plotly chart builders — zinc/slate dark theme, indigo accent."""

from __future__ import annotations

import plotly.graph_objects as go

ACCENT = "#6366f1"
BG = "#0f172a"
CARD = "#1e293b"
BORDER = "#334155"
TEXT = "#f8fafc"
MUTED = "#94a3b8"

COLOR_SEQUENCE = [ACCENT, "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff", "#94a3b8", "#64748b"]

DARK_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
    title_font=dict(color=TEXT, size=14),
    margin=dict(t=48, b=16, l=16, r=16),
    xaxis=dict(showgrid=False, zeroline=False, color=MUTED, linecolor=BORDER),
    yaxis=dict(showgrid=False, zeroline=False, color=MUTED, linecolor=BORDER),
    legend=dict(font=dict(color=MUTED)),
)


def create_metric_line_chart(results: list[dict], metric_key: str = "cv_rmse_mean", title: str = "Mean CV RMSE by Model") -> go.Figure:
    models = [r["model"] for r in results]
    values = [r.get(metric_key) for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=models, y=values, mode="lines+markers",
        line=dict(color=ACCENT, width=3),
        marker=dict(size=10, color=ACCENT),
    ))
    fig.update_layout(title=title, height=380, **DARK_LAYOUT)
    fig.update_xaxes(title_text="Model Architecture")
    fig.update_yaxes(title_text="Cross-Validation RMSE (Log Scale)")
    return fig


def create_fold_convergence_chart(results: list[dict]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results):
        fold_rmse = r.get("fold_rmse", [])
        fig.add_trace(go.Scatter(
            x=list(range(1, len(fold_rmse) + 1)),
            y=fold_rmse,
            mode="lines+markers",
            name=r["model"],
            line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]),
        ))
    fig.update_layout(
        title="RMSE per Cross-Validation Fold",
        height=380,
        **DARK_LAYOUT,
    )
    fig.update_xaxes(title_text="Fold Number")
    fig.update_yaxes(title_text="RMSE (Log Scale)")
    return fig


def create_comparison_bar(results: list[dict], metric_key: str = "cv_rmse_mean", title: str = "Model Comparison — Final CV Scores") -> go.Figure:
    sorted_results = sorted(results, key=lambda r: r.get(metric_key, 0))
    models = [r["model"] for r in sorted_results]
    values = [r.get(metric_key) for r in sorted_results]
    colors = [ACCENT if i == 0 else BORDER for i in range(len(models))]

    fig = go.Figure(go.Bar(
        x=models, y=values, marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition="outside",
    ))
    fig.update_layout(title=title, height=380, **DARK_LAYOUT)
    fig.update_xaxes(title_text="Model Architecture")
    fig.update_yaxes(title_text="Metric Score")
    return fig


def create_gauge(value: float, min_val: float, max_val: float, title: str = "Predicted Price") -> go.Figure:
    """High-contrast half-donut gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"prefix": "$", "valueformat": ",.0f", "font": {"color": TEXT, "size": 32}},
        title={"text": title, "font": {"color": MUTED, "size": 13}},
        gauge={
            "shape": "angular",
            "axis": {"range": [min_val, max_val], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": ACCENT, "thickness": 0.75},
            "bgcolor": BG,
            "borderwidth": 1,
            "bordercolor": BORDER,
            "steps": [
                {"range": [min_val, max_val], "color": BORDER},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        height=280,
        margin=dict(t=48, b=8, l=24, r=24),
    )
    return fig


def _format_feature_label(name: str) -> str:
    """Format raw feature names into clean, readable Title Case labels."""
    for prefix in ("num__", "cat__", "ord__", "bin__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]

    special_names = {
        "TotalSF": "Total Square Footage",
        "GrLivArea": "Living Area Above Grade",
        "TotalBsmtSF": "Total Basement Area",
        "1stFlrSF": "First Floor Area",
        "2ndFlrSF": "Second Floor Area",
        "OverallQual": "Overall Quality",
        "OverallCond": "Overall Condition",
        "QualityScore": "Overall Quality Score",
        "HouseAge": "House Age (Years)",
        "RemodAge": "Years Since Remodel",
        "TotalBaths": "Total Bathrooms",
        "FullBath": "Full Bathrooms",
        "HalfBath": "Half Bathrooms",
        "BsmtFullBath": "Basement Full Baths",
        "BsmtHalfBath": "Basement Half Baths",
        "GarageCars": "Garage Capacity (Cars)",
        "GarageArea": "Garage Area (Sq Ft)",
        "YearBuilt": "Year Built",
        "YearRemodAdd": "Year Remodeled",
        "LotArea": "Lot Area (Sq Ft)",
        "LotFrontage": "Lot Frontage (Ft)",
        "Fireplaces": "Fireplaces",
        "IsRemodeled": "Is Remodeled (Flag)",
        "HasPool": "Has Pool (Flag)",
        "HasGarage": "Has Garage (Flag)",
        "HasFireplace": "Has Fireplace (Flag)",
        "KitchenQual": "Kitchen Quality",
        "ExterQual": "Exterior Quality",
        "BsmtQual": "Basement Quality",
        "HeatingQC": "Heating Quality",
        "CentralAir": "Central Air Conditioning",
        "TotRmsAbvGrd": "Total Rooms Above Grade",
        "BedroomAbvGr": "Bedrooms Above Grade",
        "KitchenAbvGr": "Kitchens Above Grade",
    }
    if name in special_names:
        return special_names[name]

    if "_" in name:
        parts = name.split("_", 1)
        base = special_names.get(parts[0], parts[0])
        return f"{base}: {parts[1]}"

    return name.title()


def create_feature_importance_bar(importances: dict[str, float], title: str = "Top Feature Importances") -> go.Figure:
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [_format_feature_label(k) for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=ACCENT,
        marker_line_width=0,
    ))
    fig.update_layout(title=title, height=max(320, 28 * max(len(names), 1)), **DARK_LAYOUT)
    fig.update_xaxes(title_text="Relative Importance Score", showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="Feature", showgrid=False, zeroline=False)
    return fig


def create_prediction_history_line(timestamps: list, prices: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=timestamps, y=prices, mode="lines+markers",
        line=dict(color=ACCENT), marker=dict(color=ACCENT),
    ))
    fig.update_layout(title="Prediction History", height=320, **DARK_LAYOUT)
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Predicted Price ($)")
    return fig


def create_histogram(bin_edges: list[float], counts: list[int], title: str, x_label: str) -> go.Figure:
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=ACCENT))
    fig.update_layout(title=title, height=350, bargap=0.02, **DARK_LAYOUT)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="Sample Count")
    return fig


def create_correlation_bar(correlations: dict[str, float]) -> go.Figure:
    items = sorted(correlations.items(), key=lambda kv: kv[1])
    names = [_format_feature_label(k) for k, _ in items]
    values = [v for _, v in items]
    colors = [ACCENT if v >= 0 else "#f43f5e" for v in values]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(title="Top Correlations with Sale Price", height=max(350, 25 * len(names)), **DARK_LAYOUT)
    fig.update_xaxes(title_text="Pearson Correlation Coefficient", showgrid=False)
    fig.update_yaxes(title_text="Feature", showgrid=False)
    return fig
