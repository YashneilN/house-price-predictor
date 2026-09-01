"""Pydantic request/response models for the FastAPI backend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    House features for a single prediction.

    Only the features that matter most for price are required; everything
    else has a sensible default so the form (and API consumers) don't need
    to supply all ~80 Ames Housing columns.
    """

    # Core size / quality features (most predictive)
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality (1-10)")
    GrLivArea: float = Field(..., gt=0, description="Above-grade living area, sq ft")
    TotalBsmtSF: float = Field(0, ge=0, description="Total basement area, sq ft")
    GarageCars: int = Field(0, ge=0, le=10, description="Garage capacity in cars")
    GarageArea: float = Field(0, ge=0)
    YearBuilt: int = Field(2000, ge=1800, le=2026)
    YearRemodAdd: int = Field(2000, ge=1800, le=2026)
    FullBath: int = Field(2, ge=0, le=10)
    HalfBath: int = Field(0, ge=0, le=5)
    BsmtFullBath: int = Field(0, ge=0, le=5)
    BsmtHalfBath: int = Field(0, ge=0, le=5)
    TotRmsAbvGrd: int = Field(6, ge=1, le=20)
    Fireplaces: int = Field(0, ge=0, le=5)
    LotArea: float = Field(9000, gt=0)
    LotFrontage: float = Field(70, ge=0)
    Neighborhood: str = Field("NAmes", description="Ames neighborhood code, e.g. 'NAmes', 'CollgCr'")
    BldgType: str = Field("1Fam")
    HouseStyle: str = Field("1Story")
    ExterQual: str = Field("TA", description="Exterior quality: Ex/Gd/TA/Fa/Po")
    KitchenQual: str = Field("TA")
    BsmtQual: str = Field("TA")
    CentralAir: str = Field("Y")
    MSZoning: str = Field("RL")
    SaleCondition: str = Field("Normal")
    SaleType: str = Field("WD")

    # Everything below has a default of 0/"NA"/typical and rarely needs setting
    OverallCond: int = Field(5, ge=1, le=10)
    MasVnrArea: float = Field(0, ge=0)
    BsmtFinSF1: float = Field(0, ge=0)
    BsmtFinSF2: float = Field(0, ge=0)
    BsmtUnfSF: float = Field(0, ge=0)
    firstFlrSF: float = Field(0, ge=0, alias="1stFlrSF")
    secondFlrSF: float = Field(0, ge=0, alias="2ndFlrSF")
    LowQualFinSF: float = Field(0, ge=0)
    BedroomAbvGr: int = Field(3, ge=0, le=15)
    KitchenAbvGr: int = Field(1, ge=0, le=5)
    GarageYrBlt: int = Field(2000, ge=1800, le=2026)
    WoodDeckSF: float = Field(0, ge=0)
    OpenPorchSF: float = Field(0, ge=0)
    EnclosedPorch: float = Field(0, ge=0)
    threeSsnPorch: float = Field(0, ge=0, alias="3SsnPorch")
    ScreenPorch: float = Field(0, ge=0)
    PoolArea: float = Field(0, ge=0)
    MiscVal: float = Field(0, ge=0)
    MoSold: int = Field(6, ge=1, le=12)
    YrSold: int = Field(2024, ge=1900, le=2030)
    Street: str = Field("Pave")
    LotShape: str = Field("Reg")
    LandContour: str = Field("Lvl")
    LotConfig: str = Field("Inside")
    LandSlope: str = Field("Gtl")
    Condition1: str = Field("Norm")
    Condition2: str = Field("Norm")
    RoofStyle: str = Field("Gable")
    RoofMatl: str = Field("CompShg")
    Exterior1st: str = Field("VinylSd")
    Exterior2nd: str = Field("VinylSd")
    MasVnrType: str = Field("None")
    Foundation: str = Field("PConc")
    Heating: str = Field("GasA")
    Electrical: str = Field("SBrkr")
    GarageType: str = Field("Attchd")
    GarageFinish: str = Field("Unf")
    PavedDrive: str = Field("Y")
    ExterCond: str = Field("TA")
    BsmtCond: str = Field("TA")
    HeatingQC: str = Field("TA")
    FireplaceQu: str = Field("NA")
    GarageQual: str = Field("TA")
    GarageCond: str = Field("TA")
    PoolQC: str = Field("NA")
    BsmtExposure: str = Field("No")
    BsmtFinType1: str = Field("Unf")
    BsmtFinType2: str = Field("Unf")
    Functional: str = Field("Typ")
    Fence: str = Field("NA")
    Utilities: str = Field("AllPub")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Final ensemble predicted house price in USD.")
    confidence_interval_low: float = Field(..., description="Lower bound of 95% confidence interval in USD.")
    confidence_interval_high: float = Field(..., description="Upper bound of 95% confidence interval in USD.")
    top_feature_importances: dict[str, float] = Field(
        default_factory=dict,
        description="Top predictive features and their relative importance scores.",
    )
    sub_model_predictions: dict[str, float] = Field(
        default_factory=dict,
        description="Individual USD estimates from ensemble members (e.g., XGBoost, LightGBM, Ridge).",
    )
    ensemble_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Normalized weights applied to member model predictions in the ensemble.",
    )
    price_per_sqft: float = Field(0.0, description="Estimated price per square foot of living area.")
    model_used: str = Field(..., description="Name or type of model used for the prediction.")
    timestamp: datetime = Field(..., description="Timestamp when the prediction was generated.")


class PredictionHistoryItem(BaseModel):
    id: int
    predicted_price: float
    timestamp: datetime
    inputs_summary: dict[str, Any]


class PredictionHistoryResponse(BaseModel):
    predictions: list[PredictionHistoryItem]
    count: int


class TrainingRequest(BaseModel):
    models: Optional[list[str]] = Field(
        None,
        description="Subset of model names to train, e.g. ['RandomForest', 'XGBoost']. Trains all configured models if omitted.",
    )


class TrainingTriggerResponse(BaseModel):
    status: str
    message: str


class FoldMetric(BaseModel):
    model: str
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_mae_mean: float
    cv_r2_mean: float
    fold_rmse: list[float]
    fold_mae: list[float]
    fold_r2: list[float]
    best_params: dict[str, Any]
    training_seconds: float
    timestamp: str


class TrainingMetricsResponse(BaseModel):
    status: str
    current_model: Optional[str] = None
    models_total: Optional[int] = None
    models_completed: int = 0
    results: list[FoldMetric] = Field(default_factory=list)
    best_model: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_exists: bool
    model_name: Optional[str] = None
    trained_at: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    features_used: list[str] = Field(default_factory=list)
