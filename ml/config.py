"""
Central configuration for the ML pipeline.

Keeping all paths, feature lists, and hyperparameter grids in one place
makes it easy to tweak the pipeline without hunting through every module.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RAW_TRAIN_PATH = DATA_RAW_DIR / "train.csv"
METRICS_LOG_PATH = MODELS_DIR / "training_metrics.json"
MODEL_ARTIFACT_PATH = MODELS_DIR / "model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

TARGET_COLUMN = "SalePrice"

# ── Feature Groups (Ames Housing dataset schema) ───────────────────────
# These are grouped by how they need to be preprocessed, not just by dtype.

# Raw numerical columns that get used directly (or as inputs to engineered features)
NUMERICAL_FEATURES = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

# Nominal categorical columns (no inherent order) -> one-hot encoded
CATEGORICAL_FEATURES = [
    "MSZoning",
    "Street",
    "LotShape",
    "LandContour",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "Electrical",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
    "SaleType",
    "SaleCondition",
]

# Ordinal categorical columns -> mapped to integers reflecting quality order
QUALITY_MAP = {
    "NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5,
}

ORDINAL_FEATURES = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
}

# Engineered features created in feature_engineering.py
ENGINEERED_FEATURES = [
    "TotalSF",
    "HouseAge",
    "RemodAge",
    "TotalBaths",
    "IsRemodeled",
    "QualityScore",
    "HasPool",
    "HasGarage",
    "HasFireplace",
]

# Weighted stack served by POST /predict (dollar-scale blend after expm1).
ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.50,
    "LightGBM": 0.30,
    "Ridge": 0.20,
}

ID_COLUMN = "Id"

# ── Model Hyperparameter Grids ──────────────────────────────────────────
# Kept intentionally small so a full training run finishes in a reasonable
# time on a laptop. Widen these once you're happy with the pipeline.

MODEL_CONFIGS = {
    "LinearRegression": {
        "param_grid": {},
    },
    "Ridge": {
        "param_grid": {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
    },
    "Lasso": {
        "param_grid": {"model__alpha": [0.0005, 0.001, 0.01, 0.1]},
    },
    "RandomForest": {
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [10, 20, None],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "GradientBoosting": {
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4],
        },
    },
    "XGBoost": {
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5],
        },
    },
    "LightGBM": {
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5],
        },
    },
}

CV_FOLDS = 5
RANDOM_STATE = 42
# "grid" for exhaustive GridSearchCV, "random" for RandomizedSearchCV (faster)
SEARCH_STRATEGY = "random"
RANDOM_SEARCH_ITER = 15
