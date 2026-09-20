"""
FastAPI Model Serving for Wine Classification.
Loads the registered model directly from the MLflow Model Registry at startup
using either version number or alias (e.g. @champion).
"""
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.data import EXPECTED_FEATURES, TARGET_NAMES

MODEL_NAME = os.getenv("MODEL_NAME", "wine-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Support both alias-based and version-based URI
if os.getenv("USE_ALIAS", "true").lower() == "true":
    MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
else:
    MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

model = None
load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Lifecycle event to fetch and cache the model from MLflow Registry on startup.
    """
    global model, load_error
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        print(f"Connecting to MLflow Tracking Server at: {tracking_uri}")
        print(f"Loading registered model from: {MODEL_URI}...")
        model = mlflow.sklearn.load_model(MODEL_URI)
        load_error = None
        print(f"Successfully loaded model from {MODEL_URI}")
    except Exception as exc:
        model = None
        load_error = f"{type(exc).__name__}: {str(exc)}"
        print(f"Warning: Failed to load {MODEL_URI} on startup: {load_error}")
    yield


app = FastAPI(
    title="Wine Classifier API — MLOps Pipeline Serving",
    description="Serves predictions directly from MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


class WineFeatures(BaseModel):
    """Chemical attributes of a wine sample."""
    model_config = ConfigDict(populate_by_name=True)

    alcohol: float = Field(..., examples=[13.2])
    malic_acid: float = Field(..., examples=[1.78])
    ash: float = Field(..., examples=[2.14])
    alcalinity_of_ash: float = Field(..., examples=[11.2])
    magnesium: float = Field(..., examples=[100.0])
    total_phenols: float = Field(..., examples=[2.65])
    flavanoids: float = Field(..., examples=[2.76])
    nonflavanoid_phenols: float = Field(..., examples=[0.26])
    proanthocyanins: float = Field(..., examples=[1.28])
    color_intensity: float = Field(..., examples=[4.38])
    hue: float = Field(..., examples=[1.05])
    od280_od315_of_diluted_wines: float = Field(
        ..., alias="od280/od315_of_diluted_wines", examples=[3.40]
    )
    proline: float = Field(..., examples=[1050.0])


class PredictResponse(BaseModel):
    predicted_class: str
    class_id: int
    probabilities: Dict[str, float]
    served_by: str


@app.get("/")
def root():
    return {
        "service": "Wine Classification Serving API",
        "model_name": MODEL_NAME,
        "model_uri": MODEL_URI,
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health")
def health_check():
    """Check health status and verify if model is successfully loaded."""
    if model is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_uri": MODEL_URI,
            "error": load_error,
            "message": "Model not loaded yet. Register a model in MLflow and restart API.",
        }
    return {
        "status": "ok",
        "model_loaded": True,
        "model_uri": MODEL_URI,
        "classes": TARGET_NAMES,
    }


@app.get("/model-info")
def model_info():
    """Detailed model metadata and features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently not loaded.")
    return {
        "model_uri": MODEL_URI,
        "features": EXPECTED_FEATURES,
        "classes": TARGET_NAMES,
        "pipeline_steps": [name for name, _ in model.steps] if hasattr(model, "steps") else "unknown",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: WineFeatures):
    """Predict wine cultivar class based on chemical measurements."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: model {MODEL_URI} is not loaded.",
        )

    # Convert Pydantic model into DataFrame with exact feature names
    data_dict = {
        "alcohol": payload.alcohol,
        "malic_acid": payload.malic_acid,
        "ash": payload.ash,
        "alcalinity_of_ash": payload.alcalinity_of_ash,
        "magnesium": payload.magnesium,
        "total_phenols": payload.total_phenols,
        "flavanoids": payload.flavanoids,
        "nonflavanoid_phenols": payload.nonflavanoid_phenols,
        "proanthocyanins": payload.proanthocyanins,
        "color_intensity": payload.color_intensity,
        "hue": payload.hue,
        "od280/od315_of_diluted_wines": payload.od280_od315_of_diluted_wines,
        "proline": payload.proline,
    }

    input_df = pd.DataFrame([data_dict], columns=EXPECTED_FEATURES)

    # Predict class and probabilities
    pred_idx = int(model.predict(input_df)[0])
    probabilities_raw = model.predict_proba(input_df)[0]

    prob_dict = {
        TARGET_NAMES[i]: round(float(prob), 4)
        for i, prob in enumerate(probabilities_raw)
    }

    return PredictResponse(
        predicted_class=TARGET_NAMES[pred_idx],
        class_id=pred_idx,
        probabilities=prob_dict,
        served_by=MODEL_URI,
    )
