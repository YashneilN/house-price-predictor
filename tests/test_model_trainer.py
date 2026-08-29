import json

import numpy as np
import pandas as pd
import pytest

from ml.model_trainer import MetricsLogger, train_all_models, train_single_model
from ml.config import TARGET_COLUMN


def _make_synthetic_dataset(n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Small synthetic dataset with the minimum columns the preprocessing
    pipeline expects, and a target that's actually a (noisy) function of
    the features so models have something real to learn — this keeps CV
    RMSE meaningfully finite instead of testing on pure noise.
    """
    rng = np.random.default_rng(seed)

    overall_qual = rng.integers(1, 11, n_rows)
    gr_liv_area = rng.integers(500, 3500, n_rows)
    year_built = rng.integers(1950, 2020, n_rows)

    base_price = (
        10000 * overall_qual
        + 80 * gr_liv_area
        + 200 * (year_built - 1950)
        + rng.normal(0, 15000, n_rows)
    )
    base_price = np.clip(base_price, 40000, None)

    df = pd.DataFrame({
        "OverallQual": overall_qual,
        "OverallCond": rng.integers(1, 11, n_rows),
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": rng.integers(0, 2000, n_rows),
        "1stFlrSF": rng.integers(400, 2000, n_rows),
        "2ndFlrSF": rng.integers(0, 1500, n_rows),
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "YrSold": rng.integers(2006, 2011, n_rows),
        "MoSold": rng.integers(1, 13, n_rows),
        "FullBath": rng.integers(0, 4, n_rows),
        "HalfBath": rng.integers(0, 2, n_rows),
        "BsmtFullBath": rng.integers(0, 2, n_rows),
        "BsmtHalfBath": rng.integers(0, 2, n_rows),
        "BedroomAbvGr": rng.integers(1, 6, n_rows),
        "KitchenAbvGr": np.ones(n_rows, dtype=int),
        "TotRmsAbvGrd": rng.integers(3, 12, n_rows),
        "Fireplaces": rng.integers(0, 3, n_rows),
        "GarageYrBlt": year_built,
        "GarageCars": rng.integers(0, 4, n_rows),
        "GarageArea": rng.integers(0, 900, n_rows),
        "WoodDeckSF": rng.integers(0, 300, n_rows),
        "OpenPorchSF": rng.integers(0, 200, n_rows),
        "EnclosedPorch": np.zeros(n_rows, dtype=int),
        "3SsnPorch": np.zeros(n_rows, dtype=int),
        "ScreenPorch": np.zeros(n_rows, dtype=int),
        "PoolArea": np.zeros(n_rows, dtype=int),
        "MiscVal": np.zeros(n_rows, dtype=int),
        "LotFrontage": rng.integers(30, 120, n_rows),
        "LotArea": rng.integers(3000, 15000, n_rows),
        "MasVnrArea": np.zeros(n_rows, dtype=int),
        "BsmtFinSF1": rng.integers(0, 1000, n_rows),
        "BsmtFinSF2": np.zeros(n_rows, dtype=int),
        "BsmtUnfSF": rng.integers(0, 1000, n_rows),
        "LowQualFinSF": np.zeros(n_rows, dtype=int),
        "MSZoning": rng.choice(["RL", "RM", "FV"], n_rows),
        "Street": np.full(n_rows, "Pave"),
        "LotShape": rng.choice(["Reg", "IR1"], n_rows),
        "LandContour": np.full(n_rows, "Lvl"),
        "LotConfig": rng.choice(["Inside", "Corner"], n_rows),
        "LandSlope": np.full(n_rows, "Gtl"),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown"], n_rows),
        "Condition1": np.full(n_rows, "Norm"),
        "Condition2": np.full(n_rows, "Norm"),
        "BldgType": np.full(n_rows, "1Fam"),
        "HouseStyle": rng.choice(["1Story", "2Story"], n_rows),
        "RoofStyle": np.full(n_rows, "Gable"),
        "RoofMatl": np.full(n_rows, "CompShg"),
        "Exterior1st": np.full(n_rows, "VinylSd"),
        "Exterior2nd": np.full(n_rows, "VinylSd"),
        "MasVnrType": np.full(n_rows, "None"),
        "Foundation": rng.choice(["PConc", "CBlock"], n_rows),
        "Heating": np.full(n_rows, "GasA"),
        "CentralAir": rng.choice(["Y", "N"], n_rows),
        "Electrical": np.full(n_rows, "SBrkr"),
        "GarageType": rng.choice(["Attchd", "Detchd"], n_rows),
        "GarageFinish": rng.choice(["Unf", "RFn", "Fin"], n_rows),
        "PavedDrive": np.full(n_rows, "Y"),
        "SaleType": np.full(n_rows, "WD"),
        "SaleCondition": np.full(n_rows, "Normal"),
        "ExterQual": rng.choice(["Gd", "TA"], n_rows),
        "ExterCond": np.full(n_rows, "TA"),
        "BsmtQual": rng.choice(["Gd", "TA"], n_rows),
        "BsmtCond": np.full(n_rows, "TA"),
        "HeatingQC": rng.choice(["Ex", "Gd", "TA"], n_rows),
        "KitchenQual": rng.choice(["Gd", "TA"], n_rows),
        "FireplaceQu": np.full(n_rows, "TA"),
        "GarageQual": np.full(n_rows, "TA"),
        "GarageCond": np.full(n_rows, "TA"),
        "PoolQC": np.full(n_rows, "NA"),
        "BsmtExposure": np.full(n_rows, "No"),
        "BsmtFinType1": np.full(n_rows, "Unf"),
        "BsmtFinType2": np.full(n_rows, "Unf"),
        "Functional": np.full(n_rows, "Typ"),
        "Fence": np.full(n_rows, "NA"),
        "Utilities": np.full(n_rows, "AllPub"),
        TARGET_COLUMN: base_price,
    })
    return df


@pytest.fixture
def synthetic_df():
    return _make_synthetic_dataset()


@pytest.fixture
def metrics_logger(tmp_path):
    return MetricsLogger(path=tmp_path / "training_metrics.json")


class TestMetricsLogger:
    def test_read_before_any_write_returns_idle_state(self, metrics_logger):
        state = metrics_logger.read()
        assert state["status"] == "idle"
        assert state["results"] == []

    def test_start_sets_training_status(self, metrics_logger):
        metrics_logger.start(models_total=3)
        state = metrics_logger.read()
        assert state["status"] == "training"
        assert state["models_total"] == 3
        assert state["models_completed"] == 0

    def test_append_result_increments_completed_count(self, metrics_logger):
        metrics_logger.start(models_total=2)
        metrics_logger.append_result({"model": "Ridge", "cv_rmse_mean": 0.15})
        state = metrics_logger.read()
        assert state["models_completed"] == 1
        assert state["results"][0]["model"] == "Ridge"

    def test_finish_sets_complete_status_and_best_model(self, metrics_logger):
        metrics_logger.start(models_total=1)
        metrics_logger.append_result({"model": "Ridge", "cv_rmse_mean": 0.15})
        metrics_logger.finish(best_model="Ridge")
        state = metrics_logger.read()
        assert state["status"] == "complete"
        assert state["best_model"] == "Ridge"

    def test_fail_sets_failed_status_with_error_message(self, metrics_logger):
        metrics_logger.start(models_total=1)
        metrics_logger.fail("something broke")
        state = metrics_logger.read()
        assert state["status"] == "failed"
        assert state["error"] == "something broke"

    def test_written_file_is_valid_json(self, metrics_logger):
        metrics_logger.start(models_total=1)
        raw = json.loads(metrics_logger.path.read_text())
        assert raw["status"] == "training"


class TestTrainSingleModel:
    def test_linear_regression_produces_valid_result_shape(self, synthetic_df, metrics_logger):
        from ml.feature_engineering import log_transform_target

        X = synthetic_df.drop(columns=[TARGET_COLUMN])
        y_log = log_transform_target(synthetic_df[TARGET_COLUMN])
        metrics_logger.start(models_total=1)

        result = train_single_model("LinearRegression", X, y_log, metrics_logger)

        assert result["model"] == "LinearRegression"
        assert result["cv_rmse_mean"] > 0
        assert len(result["fold_rmse"]) == 5  # CV_FOLDS
        assert "_fitted_pipeline" in result

    def test_ridge_with_hyperparameter_search_picks_best_params(self, synthetic_df, metrics_logger):
        from ml.feature_engineering import log_transform_target

        X = synthetic_df.drop(columns=[TARGET_COLUMN])
        y_log = log_transform_target(synthetic_df[TARGET_COLUMN])
        metrics_logger.start(models_total=1)

        result = train_single_model("Ridge", X, y_log, metrics_logger)

        assert "model__alpha" in result["best_params"]
        assert result["cv_rmse_mean"] > 0


class TestTrainAllModels:
    def test_selects_best_model_by_lowest_rmse(self, synthetic_df, metrics_logger):
        best_result, best_pipeline = train_all_models(
            synthetic_df,
            models_to_train=["LinearRegression", "Ridge"],
            metrics_logger=metrics_logger,
        )
        assert best_result["model"] in {"LinearRegression", "Ridge"}
        assert "_fitted_pipeline" not in best_result  # popped before returning
        assert hasattr(best_pipeline, "predict")

    def test_fitted_pipeline_can_predict_on_new_data(self, synthetic_df, metrics_logger):
        best_result, best_pipeline = train_all_models(
            synthetic_df,
            models_to_train=["LinearRegression"],
            metrics_logger=metrics_logger,
        )
        X_new = synthetic_df.drop(columns=[TARGET_COLUMN]).iloc[:5]
        preds = best_pipeline.predict(X_new)
        assert len(preds) == 5
        assert np.all(np.isfinite(preds))

    def test_metrics_log_reflects_complete_status_after_run(self, synthetic_df, metrics_logger):
        train_all_models(
            synthetic_df,
            models_to_train=["LinearRegression"],
            metrics_logger=metrics_logger,
        )
        state = metrics_logger.read()
        assert state["status"] == "complete"
        assert state["best_model"] == "LinearRegression"

    def test_raises_on_empty_model_list(self, synthetic_df, metrics_logger):
        with pytest.raises(ValueError):
            train_all_models(synthetic_df, models_to_train=["NotARealModel"], metrics_logger=metrics_logger)
