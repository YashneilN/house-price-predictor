import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml.config import TARGET_COLUMN
from ml.ensemble import StackedEnsemble
from ml.feature_engineering import build_preprocessing_pipeline, log_transform_target


@pytest.fixture
def trained_ensemble():
    """Three identical linear members labeled as the production stack."""
    from tests.test_model_trainer import _make_synthetic_dataset

    df = _make_synthetic_dataset(n_rows=60)
    X = df.drop(columns=[TARGET_COLUMN])
    y_log = log_transform_target(df[TARGET_COLUMN])

    members = {}
    for name in ("XGBoost", "LightGBM", "Ridge"):
        pipeline = Pipeline(steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", LinearRegression()),
        ])
        pipeline.fit(X, y_log)
        members[name] = pipeline
    return StackedEnsemble(models=members)


@pytest.fixture
def trained_pipeline():
    """A minimal but real fitted pipeline, so /predict has something to call."""
    from tests.test_model_trainer import _make_synthetic_dataset

    df = _make_synthetic_dataset(n_rows=60)
    X = df.drop(columns=[TARGET_COLUMN])
    y_log = log_transform_target(df[TARGET_COLUMN])

    pipeline = Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X, y_log)
    return pipeline


@pytest.fixture
def client(monkeypatch, tmp_path, trained_ensemble):
    """
    TestClient wired to a fresh AppState with a stacked ensemble
    pre-loaded, so we don't depend on disk artifacts or training having run.
    """
    from api.main import app
    from api import dependencies

    fresh_state = dependencies.AppState()
    fresh_state.model = trained_ensemble
    fresh_state.model_metadata = {
        "model_name": "StackedEnsemble",
        "metrics": {"cv_rmse_mean": 0.15},
    }

    monkeypatch.setattr(dependencies, "app_state", fresh_state)
    # Routers imported `get_app_state` directly, so also patch the module-level singleton
    monkeypatch.setattr("api.routes.predict.get_app_state", lambda: fresh_state)
    monkeypatch.setattr("api.routes.train.get_app_state", lambda: fresh_state)

    with TestClient(app) as test_client:
        yield test_client, fresh_state


VALID_PREDICTION_PAYLOAD = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "TotalBsmtSF": 900,
    "GarageCars": 2,
    "GarageArea": 480,
    "YearBuilt": 2005,
    "YearRemodAdd": 2005,
    "FullBath": 2,
    "HalfBath": 1,
    "Neighborhood": "CollgCr",
}


class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_payload(self, client):
        test_client, _ = client
        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_has_expected_fields(self, client):
        test_client, _ = client
        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        body = resp.json()
        assert "predicted_price" in body
        assert "confidence_interval_low" in body
        assert "confidence_interval_high" in body
        assert "sub_model_predictions" in body
        assert set(body["sub_model_predictions"]) == {"XGBoost", "LightGBM", "Ridge"}
        assert body["model_used"] == "StackedEnsemble"
        assert body["predicted_price"] > 10_000, "Price should be rescaled to USD, not in log-space"
        assert "top_feature_importances" in body
        assert len(body["top_feature_importances"]) > 0
        assert len(body["top_feature_importances"]) <= 10
        scores = list(body["top_feature_importances"].values())
        assert scores == sorted(scores, reverse=True)
        assert all(s >= 0 for s in scores)

    def test_ensemble_is_weighted_blend_of_submodels(self, client):
        test_client, _ = client
        body = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD).json()
        subs = body["sub_model_predictions"]
        expected = 0.50 * subs["XGBoost"] + 0.30 * subs["LightGBM"] + 0.20 * subs["Ridge"]
        assert body["predicted_price"] == pytest.approx(expected, rel=1e-4)

    def test_predict_confidence_interval_brackets_point_estimate(self, client):
        test_client, _ = client
        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        body = resp.json()
        assert body["confidence_interval_low"] <= body["predicted_price"] <= body["confidence_interval_high"]

    def test_feature_importances_weighted_blend_matches_weights(self, client):
        test_client, state = client
        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        importances = body["top_feature_importances"]
        assert isinstance(importances, dict)
        assert len(importances) > 0

        # In our trained_ensemble fixture, all 3 models are identical LinearRegression models.
        # Check that individual model normalized coefficients match the blended scores.
        from api.routes.predict import _extract_pipeline_feature_importances

        single_imp = _extract_pipeline_feature_importances(state.model.models["XGBoost"])
        for feat, score in list(importances.items())[:5]:
            assert score == pytest.approx(single_imp[feat], rel=1e-3)

    def test_feature_importances_graceful_fallback_when_member_lacks_importance(self, client):
        from sklearn.dummy import DummyRegressor
        from tests.test_model_trainer import _make_synthetic_dataset

        test_client, state = client
        df = _make_synthetic_dataset(n_rows=60)
        X = df.drop(columns=[TARGET_COLUMN])
        y_log = log_transform_target(df[TARGET_COLUMN])

        dummy_pipe = Pipeline(steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", DummyRegressor()),
        ])
        dummy_pipe.fit(X, y_log)

        # Replace LightGBM with DummyRegressor (lacks coef_ and feature_importances_)
        new_models = dict(state.model.models)
        new_models["LightGBM"] = dummy_pipe
        state.model = StackedEnsemble(models=new_models)

        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert "top_feature_importances" in body
        assert len(body["top_feature_importances"]) > 0

    def test_feature_importances_empty_when_no_members_have_importances(self, client):
        from sklearn.dummy import DummyRegressor
        from tests.test_model_trainer import _make_synthetic_dataset

        test_client, state = client
        df = _make_synthetic_dataset(n_rows=60)
        X = df.drop(columns=[TARGET_COLUMN])
        y_log = log_transform_target(df[TARGET_COLUMN])

        dummy_models = {}
        for name in ("XGBoost", "LightGBM", "Ridge"):
            pipe = Pipeline(steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", DummyRegressor()),
            ])
            pipe.fit(X, y_log)
            dummy_models[name] = pipe
        state.model = StackedEnsemble(models=dummy_models)

        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["top_feature_importances"] == {}

    def test_predict_rejects_invalid_overall_qual(self, client):
        test_client, _ = client
        bad_payload = {**VALID_PREDICTION_PAYLOAD, "OverallQual": 99}  # out of 1-10 range
        resp = test_client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_rejects_missing_required_field(self, client):
        test_client, _ = client
        payload = {k: v for k, v in VALID_PREDICTION_PAYLOAD.items() if k != "OverallQual"}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_returns_503_when_no_model_loaded(self, client):
        test_client, state = client
        state.model = None
        resp = test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        assert resp.status_code == 503

    def test_prediction_is_recorded_in_history(self, client):
        test_client, state = client
        test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        history = test_client.get("/predictions/history").json()
        assert history["count"] >= 1
        assert history["predictions"][0]["predicted_price"] > 0


class TestPredictionHistoryEndpoint:
    def test_history_empty_initially(self, client):
        test_client, _ = client
        resp = test_client.get("/predictions/history")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_history_respects_limit_param(self, client):
        test_client, _ = client
        for _ in range(5):
            test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        resp = test_client.get("/predictions/history", params={"limit": 2})
        assert resp.json()["count"] == 2

    def test_history_most_recent_first(self, client):
        test_client, _ = client
        test_client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        test_client.post("/predict", json={**VALID_PREDICTION_PAYLOAD, "OverallQual": 3})
        history = test_client.get("/predictions/history").json()["predictions"]
        assert history[0]["id"] > history[1]["id"]


class TestModelInfoEndpoint:
    def test_model_info_reflects_loaded_model(self, client):
        test_client, _ = client
        resp = test_client.get("/model/info")
        body = resp.json()
        assert body["model_exists"] is True
        assert body["model_name"] == "StackedEnsemble"

    def test_model_info_when_no_model_loaded(self, client):
        test_client, state = client
        state.model = None
        resp = test_client.get("/model/info")
        assert resp.json()["model_exists"] is False


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        test_client, _ = client
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_returns_service_info(self, client):
        test_client, _ = client
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "service" in resp.json()


class TestTrainEndpoint:
    def test_train_returns_400_without_data_file(self, client, monkeypatch, tmp_path):
        test_client, _ = client
        fake_missing_path = tmp_path / "does_not_exist.csv"
        monkeypatch.setattr("api.routes.train.RAW_TRAIN_PATH", fake_missing_path)
        resp = test_client.post("/train", json={})
        assert resp.status_code == 400
