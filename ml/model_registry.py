"""
Model registry: save and load trained pipelines + their metadata.

Keeps a single "current best" model on disk (model.joblib +
model_metadata.json). Simple by design — swap for MLflow or similar
if this ever needs to track many model versions at once.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from ml.config import MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH, MODELS_DIR
from ml.ensemble import StackedEnsemble

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when no trained model exists in the registry yet."""


def save_model(
    pipeline: Pipeline | StackedEnsemble,
    model_name: str,
    metrics: dict[str, Any],
    features_used: list[str] | None = None,
    hyperparameters: dict | None = None,
) -> None:
    """Persist the fitted pipeline and a metadata sidecar JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_ARTIFACT_PATH)

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "features_used": features_used or [],
        "hyperparameters": hyperparameters or {},
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info("Saved model '%s' to %s", model_name, MODEL_ARTIFACT_PATH)


def load_model() -> Pipeline | StackedEnsemble:
    """Load the current best pipeline (or stacked ensemble) from disk. Raises if none exists."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise ModelNotFoundError(
            f"No trained model found at {MODEL_ARTIFACT_PATH}. Train a model first via POST /train."
        )
    return joblib.load(MODEL_ARTIFACT_PATH)


def load_metadata() -> dict:
    """Load metadata for the current best model. Returns empty dict if none exists."""
    if not MODEL_METADATA_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def model_exists() -> bool:
    return MODEL_ARTIFACT_PATH.exists() and MODEL_METADATA_PATH.exists()
