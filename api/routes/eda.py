"""GET /eda/summary, /eda/correlations, /eda/distributions routes.

These read directly from data/raw/train.csv on each call (the dataset is
small — ~1460 rows — so no caching layer is needed). Kept separate from
the ML training pipeline since EDA is read-only and doesn't touch models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ml.config import NUMERICAL_FEATURES, RAW_TRAIN_PATH, TARGET_COLUMN
from ml.data_loader import DataValidationError, load_raw_data

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eda", tags=["eda"])


def _load_df() -> pd.DataFrame:
    if not RAW_TRAIN_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No dataset found at {RAW_TRAIN_PATH}. Upload train.csv to data/raw/ first.",
        )
    try:
        df, _ = load_raw_data(RAW_TRAIN_PATH, require_target=True)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return df


@router.get("/summary")
def eda_summary():
    df = _load_df()
    numeric_df = df.select_dtypes(include=[np.number])

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target_stats": {
            "mean": float(df[TARGET_COLUMN].mean()),
            "median": float(df[TARGET_COLUMN].median()),
            "std": float(df[TARGET_COLUMN].std()),
            "min": float(df[TARGET_COLUMN].min()),
            "max": float(df[TARGET_COLUMN].max()),
        },
        "missing_pct_top10": (
            df.isna().mean().mul(100).round(2).sort_values(ascending=False).head(10).to_dict()
        ),
        "numeric_columns": len(numeric_df.columns),
        "categorical_columns": len(df.columns) - len(numeric_df.columns),
    }


@router.get("/correlations")
def eda_correlations(top_n: int = 15):
    df = _load_df()
    numeric_df = df.select_dtypes(include=[np.number])

    if TARGET_COLUMN not in numeric_df.columns:
        raise HTTPException(status_code=400, detail=f"{TARGET_COLUMN} is not numeric in this dataset.")

    correlations = (
        numeric_df.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
    )

    return {
        "target": TARGET_COLUMN,
        "correlations": {k: round(float(v), 4) for k, v in correlations.items()},
    }


@router.get("/distributions")
def eda_distributions(features: str | None = None, bins: int = 30):
    """
    Return histogram bin edges + counts for a set of features, plus the
    target distribution. `features` is a comma-separated list; defaults
    to a handful of the most commonly-useful ones if omitted.
    """
    df = _load_df()

    default_features = ["GrLivArea", "OverallQual", "TotalBsmtSF", "YearBuilt", "GarageCars", "LotArea"]
    requested = [f.strip() for f in features.split(",")] if features else default_features
    valid_features = [f for f in requested if f in df.columns and f in NUMERICAL_FEATURES + [TARGET_COLUMN]]

    if not valid_features:
        raise HTTPException(status_code=400, detail="None of the requested features are valid numeric columns.")

    distributions = {}
    for col in valid_features:
        series = df[col].dropna()
        counts, edges = np.histogram(series, bins=bins)
        distributions[col] = {
            "bin_edges": edges.round(2).tolist(),
            "counts": counts.tolist(),
        }

    target_series = df[TARGET_COLUMN].dropna()
    target_counts, target_edges = np.histogram(target_series, bins=bins)

    return {
        "features": distributions,
        "target": {
            "name": TARGET_COLUMN,
            "bin_edges": target_edges.round(2).tolist(),
            "counts": target_counts.tolist(),
        },
    }
