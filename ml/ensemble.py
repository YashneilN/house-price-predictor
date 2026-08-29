"""Weighted stack of fitted pipelines (XGBoost + LightGBM + Ridge)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ml.config import ENSEMBLE_WEIGHTS
from ml.feature_engineering import inverse_log_transform


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """
    Dollar-scale blend:

        FinalPrice = 0.50 * XGBoost + 0.30 * LightGBM + 0.20 * Ridge

    Member pipelines still predict on log1p(SalePrice); this class applies
    expm1 before blending so the weights match the published formula.
    Missing members have their weights redistributed over the rest.
    """

    def __init__(
        self,
        models: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.models = models or {}
        self.weights = dict(weights or ENSEMBLE_WEIGHTS)

    def _normalized_weights(self) -> dict[str, float]:
        available = {name: w for name, w in self.weights.items() if name in self.models}
        total = sum(available.values())
        if total <= 0:
            n = len(self.models)
            if n == 0:
                return {}
            return {name: 1.0 / n for name in self.models}
        return {name: w / total for name, w in available.items()}

    def predict_components(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return each sub-model's predictions in USD."""
        out: dict[str, np.ndarray] = {}
        for name, pipeline in self.models.items():
            pred_log = np.asarray(pipeline.predict(X), dtype=float)
            out[name] = inverse_log_transform(pred_log)
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        components = self.predict_components(X)
        weights = self._normalized_weights()
        if not components:
            raise ValueError("StackedEnsemble has no member models to predict with.")

        blended = None
        for name, preds in components.items():
            w = weights.get(name, 0.0)
            blended = preds * w if blended is None else blended + preds * w
        return blended

    def named_estimator(self, preferred: str = "XGBoost"):
        """Pick a member pipeline for feature-importance extraction."""
        if preferred in self.models:
            return self.models[preferred]
        return next(iter(self.models.values()))
