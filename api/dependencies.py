"""
Shared application state.

Holds the currently-loaded model pipeline and the in-memory prediction
history. A simple module-level singleton is enough for a single-process
FastAPI app; swap for a proper store (Redis/SQLite) if this ever needs
to run across multiple workers.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from ml.model_registry import ModelNotFoundError, load_metadata, load_model, model_exists

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: Optional[object] = None
        self.model_metadata: dict = {}
        self.predictions: list[dict[str, Any]] = []
        self._next_prediction_id = 1

    # ── Model ────────────────────────────────────────────────────────
    def try_load_model(self) -> bool:
        """Attempt to load the current best model from the registry. Returns success bool."""
        if not model_exists():
            logger.info("No trained model found yet. /predict will 503 until training completes.")
            return False
        try:
            with self._lock:
                self.model = load_model()
                self.model_metadata = load_metadata()
            logger.info("Loaded model '%s' from registry.", self.model_metadata.get("model_name"))
            return True
        except ModelNotFoundError:
            return False

    def has_model(self) -> bool:
        return self.model is not None

    # ── Predictions ──────────────────────────────────────────────────
    def add_prediction(self, predicted_price: float, inputs_summary: dict) -> dict:
        with self._lock:
            record = {
                "id": self._next_prediction_id,
                "predicted_price": predicted_price,
                "timestamp": datetime.now(timezone.utc),
                "inputs_summary": inputs_summary,
            }
            self._next_prediction_id += 1
            self.predictions.append(record)
            # Cap history so memory doesn't grow unbounded in a long-running demo
            if len(self.predictions) > 500:
                self.predictions = self.predictions[-500:]
            return record

    def get_recent_predictions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(reversed(self.predictions[-limit:]))


# Singleton used across the app via FastAPI dependency injection
app_state = AppState()


def get_app_state() -> AppState:
    return app_state
