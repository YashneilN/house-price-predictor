"""
Training-time preprocessing: outlier sanitation, target log scaling,
and the shared feature pipeline used at train and inference.
"""

from __future__ import annotations

import pandas as pd

from ml.config import TARGET_COLUMN
from ml.feature_engineering import (  # noqa: F401 — re-exported as the preprocessing surface
    FeatureCreator,
    OrdinalMapper,
    build_preprocessing_pipeline,
    inverse_log_transform,
    log_transform_target,
)

# Dean De Cock Ames Housing benchmark: giant cheap houses are data errors.
OUTLIER_GRLIVAREA_THRESHOLD = 4000
OUTLIER_SALEPRICE_THRESHOLD = 300_000


def remove_training_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with GrLivArea > 4000 and SalePrice < 300000."""
    if "GrLivArea" not in df.columns or TARGET_COLUMN not in df.columns:
        return df
    mask = (df["GrLivArea"] > OUTLIER_GRLIVAREA_THRESHOLD) & (
        df[TARGET_COLUMN] < OUTLIER_SALEPRICE_THRESHOLD
    )
    return df.loc[~mask].copy()
