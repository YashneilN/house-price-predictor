"""POST /predict and GET /predictions/history routes."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_app_state
from api.schemas import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
)
from ml.ensemble import StackedEnsemble
from ml.feature_engineering import inverse_log_transform

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])

# A rough, static log-scale RMSE fallback used to build a confidence band
# if the trained model's own CV RMSE isn't available in metadata.
DEFAULT_LOG_RMSE = 0.12


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, state: AppState = Depends(get_app_state)):
    if not state.has_model():
        raise HTTPException(
            status_code=503,
            detail="No trained model available yet. Trigger training via POST /train first.",
        )

    input_dict = request.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    try:
        predicted_price, sub_preds, ensemble_weights = _predict_usd(state.model, input_df)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    cv_rmse_log = state.model_metadata.get("metrics", {}).get("cv_rmse_mean", DEFAULT_LOG_RMSE)
    pred_log = float(np.log1p(max(predicted_price, 0.0)))
    low = float(inverse_log_transform(np.array([pred_log - 1.96 * cv_rmse_log]))[0])
    high = float(inverse_log_transform(np.array([pred_log + 1.96 * cv_rmse_log]))[0])

    living = float(request.GrLivArea) or 1.0
    price_per_sqft = predicted_price / living

    top_importances = _extract_feature_importances(state)

    record = state.add_prediction(
        predicted_price=predicted_price,
        inputs_summary={
            "OverallQual": request.OverallQual,
            "GrLivArea": request.GrLivArea,
            "Neighborhood": request.Neighborhood,
            "YearBuilt": request.YearBuilt,
        },
    )

    model_used = state.model_metadata.get("model_name", "unknown")
    if isinstance(state.model, StackedEnsemble):
        model_used = "StackedEnsemble"

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        confidence_interval_low=round(low, 2),
        confidence_interval_high=round(high, 2),
        top_feature_importances=top_importances,
        sub_model_predictions={k: round(float(v), 2) for k, v in sub_preds.items()},
        ensemble_weights={k: round(float(v), 4) for k, v in ensemble_weights.items()},
        price_per_sqft=round(price_per_sqft, 2),
        model_used=model_used,
        timestamp=record["timestamp"],
    )


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
def prediction_history(limit: int = 50, state: AppState = Depends(get_app_state)):
    limit = max(1, min(limit, 500))
    recent = state.get_recent_predictions(limit=limit)
    items = [PredictionHistoryItem(**r) for r in recent]
    return PredictionHistoryResponse(predictions=items, count=len(items))


def _predict_usd(model, input_df: pd.DataFrame) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return (ensemble_or_single USD, sub-model USD map, weights)."""
    if isinstance(model, StackedEnsemble):
        components = model.predict_components(input_df)
        sub_preds = {name: float(vals[0]) for name, vals in components.items()}
        predicted = float(model.predict(input_df)[0])
        return predicted, sub_preds, model._normalized_weights()

    pred_log = model.predict(input_df)[0]
    predicted = float(inverse_log_transform(np.array([pred_log]))[0])
    name = type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else "model"
    return predicted, {name: predicted}, {}


def _extract_feature_importances(state: AppState, top_n: int = 10) -> dict[str, float]:
    """
    Best-effort extraction of feature importances from the fitted pipeline,
    for tree-based models. Returns an empty dict for linear models or if
    anything about the pipeline shape isn't as expected (never raises).
    """
    try:
        estimator = state.model
        if isinstance(estimator, StackedEnsemble):
            estimator = estimator.named_estimator()

        model = estimator.named_steps["model"]
        preprocessing = estimator.named_steps["preprocessing"]
        column_transformer = preprocessing.named_steps["column_transformer"]
        feature_names = column_transformer.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return {}

        order = np.argsort(importances)[::-1][:top_n]
        return {str(feature_names[i]): float(importances[i]) for i in order}
    except Exception:  # noqa: BLE001 — feature importance is a nice-to-have, never block prediction
        logger.debug("Could not extract feature importances", exc_info=True)
        return {}
