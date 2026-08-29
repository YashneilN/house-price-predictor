"""POST /train, GET /metrics, and GET /model/info routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.dependencies import AppState, get_app_state
from api.schemas import (
    ModelInfoResponse,
    TrainingMetricsResponse,
    TrainingRequest,
    TrainingTriggerResponse,
)
from ml.config import RAW_TRAIN_PATH
from ml.data_loader import DataValidationError, load_raw_data
from ml.model_registry import save_model
from ml.model_trainer import MetricsLogger, train_all_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["train"])

_metrics_logger = MetricsLogger()

# Guards against overlapping training runs (BackgroundTasks run in the same process)
_training_lock_state = {"running": False}


def _run_training_job(models: list[str] | None, state: AppState) -> None:
    """Executed in the background by FastAPI's BackgroundTasks."""
    _training_lock_state["running"] = True
    try:
        df, _report = load_raw_data(RAW_TRAIN_PATH, require_target=True)
        best_result, best_pipeline = train_all_models(
            df, models_to_train=models, metrics_logger=_metrics_logger
        )
        save_model(
            pipeline=best_pipeline,
            model_name=best_result["model"],
            metrics=best_result,
            hyperparameters=best_result.get("best_params", {}),
        )
        state.try_load_model()
        logger.info("Background training job finished. Best model: %s", best_result["model"])
    except Exception:  # noqa: BLE001 — MetricsLogger.fail() already records this for the dashboard
        logger.exception("Background training job failed")
    finally:
        _training_lock_state["running"] = False


@router.post("/train", response_model=TrainingTriggerResponse, status_code=202)
def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_app_state),
):
    if _training_lock_state["running"]:
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    if not RAW_TRAIN_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No training data found at {RAW_TRAIN_PATH}. Upload train.csv to data/raw/ first.",
        )

    try:
        # Fail fast on obviously bad data before handing off to the background task
        load_raw_data(RAW_TRAIN_PATH, require_target=True)
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(_run_training_job, request.models, state)

    return TrainingTriggerResponse(
        status="accepted",
        message="Training started in the background. Poll GET /metrics for progress.",
    )


@router.get("/metrics", response_model=TrainingMetricsResponse)
def get_metrics():
    state = _metrics_logger.read()
    return TrainingMetricsResponse(**state)


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(state: AppState = Depends(get_app_state)):
    if not state.has_model():
        return ModelInfoResponse(model_exists=False)

    meta = state.model_metadata
    return ModelInfoResponse(
        model_exists=True,
        model_name=meta.get("model_name"),
        trained_at=meta.get("trained_at"),
        metrics=meta.get("metrics", {}),
        hyperparameters=meta.get("hyperparameters", {}),
        features_used=meta.get("features_used", []),
    )
