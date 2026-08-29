"""Reusable Plotly chart builders, kept consistent in style across pages."""

from __future__ import annotations

import plotly.graph_objects as go

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]


def create_metric_line_chart(results: list[dict], metric_key: str = "cv_rmse_mean", title: str = "CV RMSE by Model") -> go.Figure:
    """Line/marker chart showing a chosen metric across models, in the order trained."""
    models = [r["model"] for r in results]
    values = [r.get(metric_key) for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=models, y=values, mode="lines+markers",
        line=dict(color=COLOR_SEQUENCE[0], width=3),
        marker=dict(size=10),
    ))
    fig.update_layout(
        title=title, template=TEMPLATE,
        xaxis_title="Model", yaxis_title=metric_key,
        height=380,
    )
    return fig


def create_fold_convergence_chart(results: list[dict]) -> go.Figure:
    """Per-fold RMSE for every model trained so far — shows convergence/variance across folds."""
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
        template=TEMPLATE,
        xaxis_title="Fold", yaxis_title="RMSE (log scale)",
        height=380,
    )
    return fig


def create_comparison_bar(results: list[dict], metric_key: str = "cv_rmse_mean", title: str = "Model Comparison — Final CV Scores") -> go.Figure:
    """Bar chart comparing final CV scores across all trained models, best highlighted."""
    sorted_results = sorted(results, key=lambda r: r.get(metric_key, 0))
    models = [r["model"] for r in sorted_results]
    values = [r.get(metric_key) for r in sorted_results]
    colors = [COLOR_SEQUENCE[0] if i == 0 else "#C7D3E0" for i in range(len(models))]

    fig = go.Figure(go.Bar(x=models, y=values, marker_color=colors, text=[f"{v:.4f}" for v in values], textposition="outside"))
    fig.update_layout(title=title, template=TEMPLATE, xaxis_title="Model", yaxis_title=metric_key, height=380)
    return fig


def create_gauge(value: float, min_val: float, max_val: float, title: str = "Predicted Price") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"prefix": "$", "valueformat": ",.0f"},
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": COLOR_SEQUENCE[0]},
            "steps": [
                {"range": [min_val, (min_val + max_val) / 2], "color": "#EAF0F7"},
                {"range": [(min_val + max_val) / 2, max_val], "color": "#D6E2F0"},
            ],
        },
    ))
    fig.update_layout(template=TEMPLATE, height=320)
    return fig


def create_feature_importance_bar(importances: dict[str, float], title: str = "Top Feature Importances") -> go.Figure:
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=COLOR_SEQUENCE[2]))
    fig.update_layout(title=title, template=TEMPLATE, xaxis_title="Importance", height=max(300, 30 * len(names)))
    return fig


def create_prediction_history_line(timestamps: list, prices: list) -> go.Figure:
    fig = go.Figure(go.Scatter(x=timestamps, y=prices, mode="lines+markers", line=dict(color=COLOR_SEQUENCE[1])))
    fig.update_layout(
        title="Prediction History", template=TEMPLATE,
        xaxis_title="Time", yaxis_title="Predicted Price ($)", height=350,
    )
    return fig


def create_histogram(bin_edges: list[float], counts: list[int], title: str, x_label: str) -> go.Figure:
    # Plotly bar chart from precomputed histogram bins (bin centers as x)
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=COLOR_SEQUENCE[0]))
    fig.update_layout(title=title, template=TEMPLATE, xaxis_title=x_label, yaxis_title="Count", height=350, bargap=0.02)
    return fig


def create_correlation_bar(correlations: dict[str, float]) -> go.Figure:
    items = sorted(correlations.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [COLOR_SEQUENCE[2] if v >= 0 else COLOR_SEQUENCE[3] for v in values]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(
        title="Top Correlations with SalePrice", template=TEMPLATE,
        xaxis_title="Correlation", height=max(350, 25 * len(names)),
    )
    return fig
