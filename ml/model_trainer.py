"""
Model training for the house price pipeline.

Trains several regressors, tunes each with cross-validated search, and
writes progress to a JSON file after every model finishes so that the
FastAPI /metrics endpoint (and therefore the Streamlit dashboard) can
poll and show live convergence charts.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

from ml.config import (
    CV_FOLDS,
    METRICS_LOG_PATH,
    MODEL_CONFIGS,
    RANDOM_SEARCH_ITER,
    RANDOM_STATE,
    SEARCH_STRATEGY,
    TARGET_COLUMN,
)
from ml.feature_engineering import (
    build_preprocessing_pipeline,
    inverse_log_transform,
    log_transform_target,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
}
if XGBOOST_AVAILABLE:
    MODEL_CLASSES["XGBoost"] = XGBRegressor

MODEL_DEFAULT_KWARGS = {
    "RandomForest": {"random_state": RANDOM_STATE, "n_jobs": -1},
    "GradientBoosting": {"random_state": RANDOM_STATE},
    "XGBoost": {"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0},
    "Ridge": {"random_state": RANDOM_STATE},
    "Lasso": {"random_state": RANDOM_STATE, "max_iter": 5000},
}


class MetricsLogger:
    """
    Appends training progress to a JSON file that acts as the shared state
    between the background training task and the /metrics API endpoint.

    Structure of the JSON file:
    {
        "status": "training" | "complete" | "failed" | "idle",
        "started_at": iso timestamp,
        "updated_at": iso timestamp,
        "current_model": "RandomForest" | null,
        "models_total": 6,
        "models_completed": 2,
        "results": [ {model, cv_rmse_mean, cv_rmse_std, cv_mae_mean, cv_r2_mean,
                       best_params, fold_rmse: [...], timestamp}, ... ],
        "best_model": "XGBoost" | null,
        "error": null | str
    }
    """

    def __init__(self, path: Path = METRICS_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, state: dict) -> None:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2, default=str))
        tmp_path.replace(self.path)  # atomic-ish swap so readers never see a half-written file

    def read(self) -> dict:
        if not self.path.exists():
            return {"status": "idle", "results": [], "current_model": None, "best_model": None}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {"status": "idle", "results": [], "current_model": None, "best_model": None}

    def start(self, models_total: int) -> None:
        state = {
            "status": "training",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "current_model": None,
            "models_total": models_total,
            "models_completed": 0,
            "results": [],
            "best_model": None,
            "error": None,
        }
        self._write(state)

    def set_current_model(self, model_name: str) -> None:
        state = self.read()
        state["current_model"] = model_name
        self._write(state)

    def append_result(self, result: dict) -> None:
        state = self.read()
        state["results"].append(result)
        state["models_completed"] = len(state["results"])
        self._write(state)

    def finish(self, best_model: str) -> None:
        state = self.read()
        state["status"] = "complete"
        state["current_model"] = None
        state["best_model"] = best_model
        self._write(state)

    def fail(self, error_message: str) -> None:
        state = self.read()
        state["status"] = "failed"
        state["error"] = error_message
        self._write(state)


def _build_search(model_name: str, pipeline: Pipeline, param_grid: dict):
    """Wrap the pipeline in GridSearchCV or RandomizedSearchCV depending on config."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = "neg_root_mean_squared_error"

    if not param_grid:
        # Nothing to tune (e.g. plain LinearRegression) — just cross-validate it directly.
        return None, cv

    if SEARCH_STRATEGY == "grid":
        search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1,
            n_iter=RANDOM_SEARCH_ITER, random_state=RANDOM_STATE,
        )
    return search, cv


def train_single_model(
    model_name: str,
    X: pd.DataFrame,
    y_log: np.ndarray,
    metrics_logger: MetricsLogger,
) -> dict:
    """
    Cross-validate (and tune, if a param grid exists) a single model.
    Returns a result dict that also gets appended to the metrics log.
    """
    metrics_logger.set_current_model(model_name)
    logger.info("Training %s ...", model_name)
    start = time.time()

    model_cls = MODEL_CLASSES[model_name]
    model_kwargs = MODEL_DEFAULT_KWARGS.get(model_name, {})
    param_grid = MODEL_CONFIGS[model_name]["param_grid"]

    preprocessing = build_preprocessing_pipeline()
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model_cls(**model_kwargs)),
    ])

    search, cv = _build_search(model_name, pipeline, param_grid)

    if search is not None:
        search.fit(X, y_log)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        # Re-run cross_validate on the winning params to get per-fold metrics for the chart
        cv_results = cross_validate(
            best_pipeline, X, y_log, cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            n_jobs=-1,
        )
    else:
        best_params = {}
        cv_results = cross_validate(
            pipeline, X, y_log, cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            n_jobs=-1,
        )
        # cross_validate only fits internal clones, not `pipeline` itself — fit it now
        # on the full dataset so we have a usable model to save/serve.
        best_pipeline = pipeline.fit(X, y_log)

    fold_rmse = (-cv_results["test_neg_root_mean_squared_error"]).tolist()
    fold_mae = (-cv_results["test_neg_mean_absolute_error"]).tolist()
    fold_r2 = cv_results["test_r2"].tolist()

    elapsed = time.time() - start

    result = {
        "model": model_name,
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse)),
        "cv_mae_mean": float(np.mean(fold_mae)),
        "cv_r2_mean": float(np.mean(fold_r2)),
        "fold_rmse": fold_rmse,
        "fold_mae": fold_mae,
        "fold_r2": fold_r2,
        "best_params": best_params,
        "training_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metrics_logger.append_result(result)
    logger.info(
        "%s done in %.1fs — CV RMSE (log-scale): %.4f ± %.4f, R²: %.4f",
        model_name, elapsed, result["cv_rmse_mean"], result["cv_rmse_std"], result["cv_r2_mean"],
    )

    # Stash the fitted pipeline on the result dict (not JSON-serialized/logged) so the
    # orchestrator can pick the winner without refitting.
    result["_fitted_pipeline"] = best_pipeline
    return result


def train_all_models(
    df: pd.DataFrame,
    models_to_train: list[str] | None = None,
    metrics_logger: MetricsLogger | None = None,
) -> tuple[dict, Pipeline]:
    """
    Orchestrates training every configured model, logging progress as it goes,
    and returns (best_result, best_fitted_pipeline).

    `df` must include the target column (TARGET_COLUMN).
    """
    metrics_logger = metrics_logger or MetricsLogger()
    models_to_train = models_to_train or list(MODEL_CLASSES.keys())
    models_to_train = [m for m in models_to_train if m in MODEL_CLASSES]

    if not models_to_train:
        raise ValueError("No valid models to train. Check MODEL_CLASSES / requested model names.")

    X = df.drop(columns=[TARGET_COLUMN])
    y_log = log_transform_target(df[TARGET_COLUMN])

    metrics_logger.start(models_total=len(models_to_train))

    results = []
    try:
        for model_name in models_to_train:
            result = train_single_model(model_name, X, y_log, metrics_logger)
            results.append(result)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the dashboard
        logger.exception("Training failed")
        metrics_logger.fail(str(exc))
        raise

    best_result = min(results, key=lambda r: r["cv_rmse_mean"])
    metrics_logger.finish(best_model=best_result["model"])

    best_pipeline = best_result.pop("_fitted_pipeline")
    for r in results:
        r.pop("_fitted_pipeline", None)

    logger.info("Best model: %s (CV RMSE log-scale: %.4f)", best_result["model"], best_result["cv_rmse_mean"])
    return best_result, best_pipeline


def evaluate_on_holdout(pipeline: Pipeline, X_test: pd.DataFrame, y_test_actual: np.ndarray) -> dict:
    """Evaluate a fitted pipeline on real dollar-scale predictions (inverse log)."""
    y_pred_log = pipeline.predict(X_test)
    y_pred = inverse_log_transform(y_pred_log)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test_actual, y_pred))),
        "mae": float(mean_absolute_error(y_test_actual, y_pred)),
        "r2": float(r2_score(y_test_actual, y_pred)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ml.data_loader import load_raw_data

    data, _ = load_raw_data()
    best, fitted = train_all_models(data)
    print(f"Best model: {best['model']}")
    print(json.dumps(best, indent=2))
