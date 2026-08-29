"""FastAPI app entry point for the House Price Predictor backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import app_state
from api.routes import eda, predict, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = app_state.try_load_model()
    if loaded:
        logger.info("Startup: existing model loaded, /predict is ready.")
    else:
        logger.info("Startup: no model found yet. Call POST /train, then reload, to enable /predict.")
    yield


app = FastAPI(
    title="House Price Predictor API",
    description="ML-backed API for training, evaluating, and serving Ames Housing price predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is wide open for local dev / the future React frontend. Tighten
# allow_origins before deploying this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(train.router)
app.include_router(eda.router)


@app.get("/")
def root():
    return {
        "service": "House Price Predictor API",
        "status": "ok",
        "model_loaded": app_state.has_model(),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app_state.has_model()}
