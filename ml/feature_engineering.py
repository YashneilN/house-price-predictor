"""
Feature engineering for the house price pipeline.

Everything here is built as scikit-learn transformers wired into a single
Pipeline + ColumnTransformer, so the *exact* same steps run at train time
and at inference time (no train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)


def _col(X: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric series for `name`, treating missing columns / NaNs as 0."""
    if name not in X.columns:
        return pd.Series(0, index=X.index, dtype=float)
    return pd.to_numeric(X[name], errors="coerce").fillna(0)


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates derived features from raw columns.

    - TotalSF       = GrLivArea + TotalBsmtSF + 1stFlrSF + 2ndFlrSF
    - TotalBaths    = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
    - HouseAge      = YrSold - YearBuilt
    - RemodAge      = YrSold - YearRemodAdd
    - IsRemodeled   = 1 if YearRemodAdd != YearBuilt else 0
    - QualityScore  = OverallQual * OverallCond
    - HasPool       = 1 if PoolArea > 0 else 0
    - HasGarage     = 1 if GarageArea > 0 else 0
    - HasFireplace  = 1 if Fireplaces > 0 else 0

    Stateless (no fitting needed), but implements fit() for Pipeline compatibility.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["TotalSF"] = (
            _col(X, "GrLivArea")
            + _col(X, "TotalBsmtSF")
            + _col(X, "1stFlrSF")
            + _col(X, "2ndFlrSF")
        )
        X["HouseAge"] = _col(X, "YrSold") - _col(X, "YearBuilt")
        X["RemodAge"] = _col(X, "YrSold") - _col(X, "YearRemodAdd")
        X["TotalBaths"] = (
            _col(X, "FullBath")
            + 0.5 * _col(X, "HalfBath")
            + _col(X, "BsmtFullBath")
            + 0.5 * _col(X, "BsmtHalfBath")
        )
        X["IsRemodeled"] = (_col(X, "YearRemodAdd") != _col(X, "YearBuilt")).astype(int)
        X["QualityScore"] = _col(X, "OverallQual") * _col(X, "OverallCond")
        X["HasPool"] = (_col(X, "PoolArea") > 0).astype(int)
        X["HasGarage"] = (_col(X, "GarageArea") > 0).astype(int)
        X["HasFireplace"] = (_col(X, "Fireplaces") > 0).astype(int)

        # Ages can go negative if data entry errors exist (remod before build, etc).
        # Clip rather than drop so we don't lose rows at inference time.
        X["HouseAge"] = X["HouseAge"].clip(lower=0)
        X["RemodAge"] = X["RemodAge"].clip(lower=0)

        return X


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Maps quality-style categorical columns (e.g. 'Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA')
    to integers reflecting their natural order, using the maps in config.ORDINAL_FEATURES.

    Unmapped / unseen values fall back to the median of the mapping (a neutral
    guess) rather than raising, so the pipeline doesn't break on production input.
    """

    def __init__(self, ordinal_maps: dict[str, dict] | None = None):
        self.ordinal_maps = ordinal_maps or ORDINAL_FEATURES

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.ordinal_maps.items():
            if col not in X.columns:
                X[col] = "NA"
                continue
            fallback = int(np.median(list(mapping.values())))
            X[col] = (
                X[col]
                .fillna("NA")
                .map(mapping)
                .fillna(fallback)
                .astype(float)
            )
        return X


def get_ordinal_columns() -> list[str]:
    return list(ORDINAL_FEATURES.keys())


def build_preprocessing_pipeline() -> Pipeline:
    """
    Assemble the full preprocessing pipeline:

    1. FeatureCreator   -> adds engineered columns (needs raw cols present)
    2. OrdinalMapper     -> converts quality strings to ordered ints
    3. ColumnTransformer -> impute + scale numerics, impute + one-hot categoricals,
                             impute ordinals (already numeric after step 2)
    """
    numerical_cols = NUMERICAL_FEATURES + [
        "TotalSF", "HouseAge", "RemodAge", "TotalBaths", "QualityScore",
    ]
    binary_cols = ["HasPool", "HasGarage", "HasFireplace", "IsRemodeled"]
    ordinal_cols = get_ordinal_columns()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("ord", ordinal_transformer, ordinal_cols),
            ("bin", binary_transformer, binary_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[
        ("feature_creator", FeatureCreator()),
        ("ordinal_mapper", OrdinalMapper()),
        ("column_transformer", column_transformer),
    ])

    return pipeline


def log_transform_target(y: pd.Series) -> np.ndarray:
    """SalePrice is right-skewed; log1p makes the target closer to normal,
    which helps linear models and stabilizes variance for tree models too."""
    return np.log1p(y)


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """Invert log1p to get back to dollar scale for reporting/predictions."""
    return np.expm1(y_log)
