"""Weighted stack of fitted pipelines (XGBoost + LightGBM + Ridge).

Default weights are defined in ``ml.config.ENSEMBLE_WEIGHTS``::

    XGBoost  0.50
    LightGBM 0.30
    Ridge    0.20

If any member model is unavailable (not installed, or not passed into
the constructor), the remaining weights are **renormalized** so they
still sum to 1.0, and a warning is logged — the ensemble never fails
simply because a library is missing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ml.config import ENSEMBLE_WEIGHTS
from ml.feature_engineering import inverse_log_transform

logger = logging.getLogger(__name__)


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """
    Dollar-scale blend:

        FinalPrice = 0.50 * XGBoost + 0.30 * LightGBM + 0.20 * Ridge

    Member pipelines still predict on log1p(SalePrice); this class applies
    expm1 before blending so the weights match the published formula.
    Missing members have their weights dynamically renormalized over the
    remaining available models rather than raising an error.
    """

    def __init__(
        self,
        models: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.models = models or {}
        self.weights = dict(weights or ENSEMBLE_WEIGHTS)

    # ------------------------------------------------------------------
    # sklearn contract
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None):
        """No-op — member pipelines are pre-fitted before being passed in.

        Provided so that sklearn helpers (``check_is_fitted``, ``clone``,
        pipeline wrappers) do not error out when they expect a ``fit``
        method.  Returns ``self`` for method-chaining.
        """
        return self

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _normalized_weights(self) -> dict[str, float]:
        """Return weights renormalized to available models only.

        If any configured member is missing from ``self.models``, its
        share is redistributed proportionally and a warning is logged.
        """
        available = {name: w for name, w in self.weights.items() if name in self.models}
        missing = set(self.weights) - set(available)
        if missing:
            logger.warning(
                "Ensemble members unavailable, renormalizing weights: %s",
                ", ".join(sorted(missing)),
            )
        total = sum(available.values())
        if total <= 0:
            n = len(self.models)
            if n == 0:
                return {}
            return {name: 1.0 / n for name in self.models}
        return {name: w / total for name, w in available.items()}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def named_estimator(self, preferred: str = "XGBoost"):
        """Pick a member pipeline for feature-importance extraction."""
        if preferred in self.models:
            return self.models[preferred]
        return next(iter(self.models.values()))

    def __repr__(self) -> str:
        members = list(self.models.keys()) if self.models else []
        weights = self._normalized_weights() if members else {}
        parts = [f"{n}={weights.get(n, 0):.0%}" for n in members]
        return f"StackedEnsemble({', '.join(parts) or 'empty'})"
