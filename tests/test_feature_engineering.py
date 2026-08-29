import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    FeatureCreator,
    OrdinalMapper,
    build_preprocessing_pipeline,
    inverse_log_transform,
    log_transform_target,
)


@pytest.fixture
def sample_raw_df():
    """A tiny but schema-complete slice of Ames-like data, including some NaNs."""
    return pd.DataFrame({
        "TotalBsmtSF": [800, np.nan, 0],
        "1stFlrSF": [1000, 900, 1200],
        "2ndFlrSF": [500, 0, 0],
        "YrSold": [2010, 2008, 2009],
        "YearBuilt": [2000, 1995, 1980],
        "YearRemodAdd": [2005, 1995, 1980],
        "FullBath": [2, 1, 2],
        "HalfBath": [1, 0, 0],
        "BsmtFullBath": [0, 1, 0],
        "BsmtHalfBath": [0, 0, 0],
        "PoolArea": [0, 0, 500],
        "GarageArea": [400, 0, 300],
        "Fireplaces": [1, 0, 2],
        "OverallQual": [7, 5, 6],
        "OverallCond": [5, 6, 5],
        "LotFrontage": [70, np.nan, 60],
        "LotArea": [9000, 8000, 10000],
        "MasVnrArea": [0, 0, 100],
        "BsmtFinSF1": [400, 0, 0],
        "BsmtFinSF2": [0, 0, 0],
        "BsmtUnfSF": [400, 0, 0],
        "LowQualFinSF": [0, 0, 0],
        "GrLivArea": [1500, 900, 1200],
        "BedroomAbvGr": [3, 2, 3],
        "KitchenAbvGr": [1, 1, 1],
        "TotRmsAbvGrd": [7, 5, 6],
        "GarageYrBlt": [2000, np.nan, 1980],
        "GarageCars": [2, 0, 1],
        "WoodDeckSF": [0, 0, 0],
        "OpenPorchSF": [0, 0, 0],
        "EnclosedPorch": [0, 0, 0],
        "3SsnPorch": [0, 0, 0],
        "ScreenPorch": [0, 0, 0],
        "MiscVal": [0, 0, 0],
        "MoSold": [6, 3, 9],
        "MSZoning": ["RL", "RM", "RL"],
        "Street": ["Pave", "Pave", "Pave"],
        "LotShape": ["Reg", "IR1", "Reg"],
        "LandContour": ["Lvl", "Lvl", "Lvl"],
        "LotConfig": ["Inside", "Corner", "Inside"],
        "LandSlope": ["Gtl", "Gtl", "Gtl"],
        "Neighborhood": ["NAmes", "OldTown", "CollgCr"],
        "Condition1": ["Norm", "Norm", "Norm"],
        "Condition2": ["Norm", "Norm", "Norm"],
        "BldgType": ["1Fam", "1Fam", "1Fam"],
        "HouseStyle": ["1Story", "2Story", "1Story"],
        "RoofStyle": ["Gable", "Gable", "Gable"],
        "RoofMatl": ["CompShg", "CompShg", "CompShg"],
        "Exterior1st": ["VinylSd", "Wd Sdng", "VinylSd"],
        "Exterior2nd": ["VinylSd", "Wd Sdng", "VinylSd"],
        "MasVnrType": ["None", "None", "BrkFace"],
        "Foundation": ["PConc", "CBlock", "PConc"],
        "Heating": ["GasA", "GasA", "GasA"],
        "CentralAir": ["Y", "Y", "N"],
        "Electrical": ["SBrkr", "SBrkr", "SBrkr"],
        "GarageType": ["Attchd", "NA", "Detchd"],
        "GarageFinish": ["Unf", "NA", "RFn"],
        "PavedDrive": ["Y", "N", "Y"],
        "SaleType": ["WD", "WD", "New"],
        "SaleCondition": ["Normal", "Normal", "Partial"],
        "ExterQual": ["Gd", "TA", "TA"],
        "ExterCond": ["TA", "TA", "TA"],
        "BsmtQual": ["Gd", np.nan, "TA"],
        "BsmtCond": ["TA", np.nan, "TA"],
        "HeatingQC": ["Ex", "TA", "Gd"],
        "KitchenQual": ["Gd", "TA", "TA"],
        "FireplaceQu": ["Gd", np.nan, "TA"],
        "GarageQual": ["TA", np.nan, "TA"],
        "GarageCond": ["TA", np.nan, "TA"],
        "PoolQC": [np.nan, np.nan, "Gd"],
        "BsmtExposure": ["No", np.nan, "Gd"],
        "BsmtFinType1": ["GLQ", np.nan, "Unf"],
        "BsmtFinType2": ["Unf", np.nan, "Unf"],
        "Functional": ["Typ", "Typ", "Typ"],
        "Fence": [np.nan, "MnPrv", np.nan],
        "Utilities": ["AllPub", "AllPub", "AllPub"],
    })


class TestFeatureCreator:
    def test_total_sf_is_sum_of_floor_areas(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        assert out.loc[0, "TotalSF"] == 800 + 1000 + 500

    def test_total_sf_handles_nan_basement(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        # Row 1 has NaN TotalBsmtSF -> should be treated as 0, not propagate NaN
        assert out.loc[1, "TotalSF"] == 900 + 0

    def test_house_age_computed_correctly(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        assert out.loc[0, "HouseAge"] == 2010 - 2000

    def test_house_age_clipped_at_zero(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "YearBuilt"] = 2020  # built after "sold" -> would be negative
        out = FeatureCreator().fit_transform(df)
        assert out.loc[0, "HouseAge"] >= 0

    def test_total_bath_weights_half_baths(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        # Row 0: FullBath=2, HalfBath=1, BsmtFullBath=0, BsmtHalfBath=0 -> 2.5
        assert out.loc[0, "TotalBath"] == pytest.approx(2.5)

    def test_has_pool_binary_flag(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        assert out.loc[0, "HasPool"] == 0
        assert out.loc[2, "HasPool"] == 1

    def test_has_garage_binary_flag(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        assert out.loc[0, "HasGarage"] == 1
        assert out.loc[1, "HasGarage"] == 0

    def test_output_row_count_unchanged(self, sample_raw_df):
        out = FeatureCreator().fit_transform(sample_raw_df)
        assert len(out) == len(sample_raw_df)


class TestOrdinalMapper:
    def test_known_quality_values_mapped_correctly(self, sample_raw_df):
        out = OrdinalMapper().fit_transform(sample_raw_df)
        # ExterQual: Gd=4, TA=3, TA=3
        assert list(out["ExterQual"]) == [4.0, 3.0, 3.0]

    def test_missing_values_get_fallback_not_nan(self, sample_raw_df):
        out = OrdinalMapper().fit_transform(sample_raw_df)
        assert out["BsmtQual"].isna().sum() == 0

    def test_missing_column_defaults_to_na_mapping(self):
        df = pd.DataFrame({"OverallQual": [5]})  # no ordinal columns present at all
        out = OrdinalMapper().fit_transform(df)
        assert "ExterQual" in out.columns
        assert not out["ExterQual"].isna().any()


class TestPreprocessingPipeline:
    def test_pipeline_fits_and_transforms_without_error(self, sample_raw_df):
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(sample_raw_df)
        assert transformed.shape[0] == len(sample_raw_df)
        assert transformed.shape[1] > 0

    def test_pipeline_output_has_no_nans(self, sample_raw_df):
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(sample_raw_df)
        assert not np.isnan(transformed).any()

    def test_pipeline_handles_unseen_category_at_transform_time(self, sample_raw_df):
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(sample_raw_df)

        new_row = sample_raw_df.iloc[[0]].copy()
        new_row["Neighborhood"] = "TotallyMadeUpNeighborhood"
        # Should not raise thanks to handle_unknown="ignore" on the OneHotEncoder
        result = pipeline.transform(new_row)
        assert result.shape[0] == 1


class TestTargetTransform:
    def test_log_transform_and_inverse_are_reciprocal(self):
        prices = np.array([100000.0, 250000.0, 500000.0])
        log_prices = log_transform_target(pd.Series(prices))
        recovered = inverse_log_transform(np.array(log_prices))
        np.testing.assert_allclose(recovered, prices, rtol=1e-6)

    def test_log_transform_reduces_skew(self):
        # Right-skewed price-like distribution
        prices = pd.Series(np.random.default_rng(0).lognormal(mean=12, sigma=0.5, size=500))
        raw_skew = prices.skew()
        log_skew = pd.Series(log_transform_target(prices)).skew()
        assert abs(log_skew) < abs(raw_skew)
