"""
FastAPI backend for the Smart Crop Recommendation System.

Serves the already-trained RandomForestClassifier (.pkl) plus its
label_encoder.pkl and feature_names.json, and returns the TOP 3 crops.

Endpoints
---------
GET  /health   -> {"status": "ok"}
POST /predict  -> {"recommendations": [{"rank", "crop", "probability"}, ...]}

The model is NEVER retrained here - it is loaded ONCE at import time.
If loading fails, endpoints respond 503 so the failure is explicit.

Deploy on Render (see render.yaml): binds 0.0.0.0:$PORT, rootDir=backend.
"""

import logging
import pickle
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from preprocessing import FEATURES, preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop_api")

# backend/ -> repo root -> model/  (model lives at repo-root model/, never on ESP32)
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

MODEL = None
ENCODER = None


def load_artifacts():
    """Load .pkl model + label encoder once. Raises on failure."""
    global MODEL, ENCODER
    with open(MODEL_DIR / "crop_random_forest_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        ENCODER = pickle.load(f)
    # Warm-up: confirm predict() works with the 7 expected features.
    _ = MODEL.predict(preprocess({feat: 0.0 for feat in FEATURES}))
    logger.info("Model loaded: %s features, %d classes",
                FEATURES, len(ENCODER.classes_))


try:
    load_artifacts()
except Exception as exc:  # noqa: BLE001 - surface a clean 503 later
    logger.exception("MODEL LOAD FAILED: %s", exc)


app = FastAPI(
    title="Smart Crop Recommendation API",
    version="1.0.0",
    description="Random Forest crop prediction served from a pre-trained model. "
                "POST 7 agronomic features, get the Top-3 crops + probabilities.",
)

# CORS: open for development / local frontend. No credentials exposed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CropInput(BaseModel):
    """Exactly the 7 model features, in the order the model expects."""
    N: float = Field(..., ge=0, description="Nitrogen (kg/ha)")
    P: float = Field(..., ge=0, description="Phosphorus (kg/ha)")
    K: float = Field(..., ge=0, description="Potassium (kg/ha)")
    temperature: float = Field(..., ge=-20, le=60, description="deg C")
    humidity: float = Field(..., ge=0, le=100, description="% relative")
    ph: float = Field(..., ge=0, le=14, description="soil pH")
    rainfall: float = Field(..., ge=0, description="mm")

    model_config = {"extra": "forbid"}  # reject unknown / mis-cased fields


@app.exception_handler(RequestValidationError)
async def handle_validation(_: Request, exc: RequestValidationError):
    """Map Pydantic errors to the spec's simple error shape."""
    types = {e["type"] for e in exc.errors()}
    if "missing" in types:
        msg = "Missing required sensor value"
    else:
        msg = "Invalid sensor value"
    return JSONResponse(status_code=400, content={"error": msg})


@app.get("/health")
def health():
    if MODEL is None or ENCODER is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "model not loaded"},
        )
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CropInput):
    if MODEL is None:
        return JSONResponse(
            status_code=503, content={"error": "model not loaded"}
        )
    try:
        # Build input in EXACTLY the saved feature order (no reordering).
        X = preprocess(payload.model_dump())

        proba = MODEL.predict_proba(X)[0]
        classes = ENCODER.inverse_transform(range(len(proba)))
        order = proba.argsort()[::-1][:3]  # top 3, highest probability first

        recommendations = [
            {
                "rank": i + 1,
                "crop": str(classes[j]),
                "probability": round(float(proba[j]), 4),
            }
            for i, j in enumerate(order)
        ]
        return {"recommendations": recommendations}
    except Exception as exc:  # noqa: BLE001 - return a useful 500
        logger.exception("prediction failed")
        return JSONResponse(
            status_code=500, content={"error": "prediction failed"}
        )
