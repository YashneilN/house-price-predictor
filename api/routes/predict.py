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
        pred_log = state.model.predict(input_df)[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    predicted_price = float(inverse_log_transform(np.array([pred_log]))[0])

    # Build a simple confidence interval on the log scale using the model's
    # own CV RMSE (from metadata) where available.
    cv_rmse_log = state.model_metadata.get("metrics", {}).get("cv_rmse_mean", DEFAULT_LOG_RMSE)
    low = float(inverse_log_transform(np.array([pred_log - 1.96 * cv_rmse_log]))[0])
    high = float(inverse_log_transform(np.array([pred_log + 1.96 * cv_rmse_log]))[0])

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

    return PredictionResponse(
        predicted_price=round(predicted_price, 2),
        confidence_interval_low=round(low, 2),
        confidence_interval_high=round(high, 2),
        top_feature_importances=top_importances,
        model_used=state.model_metadata.get("model_name", "unknown"),
        timestamp=record["timestamp"],
    )


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
def prediction_history(limit: int = 50, state: AppState = Depends(get_app_state)):
    limit = max(1, min(limit, 500))
    recent = state.get_recent_predictions(limit=limit)
    items = [PredictionHistoryItem(**r) for r in recent]
    return PredictionHistoryResponse(predictions=items, count=len(items))


def _extract_feature_importances(state: AppState, top_n: int = 10) -> dict[str, float]:
    """
    Best-effort extraction of feature importances from the fitted pipeline,
    for tree-based models. Returns an empty dict for linear models or if
    anything about the pipeline shape isn't as expected (never raises).
    """
    try:
        model = state.model.named_steps["model"]
        preprocessing = state.model.named_steps["preprocessing"]
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
