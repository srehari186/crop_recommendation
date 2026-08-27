import logging
import pickle
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from preprocessing import FEATURES, preprocess

CROP_LABELS = {
    0: "apple",
    1: "banana",
    2: "blackgram",
    3: "chickpea",
    4: "coconut",
    5: "coffee",
    6: "cotton",
    7: "grapes",
    8: "jute",
    9: "kidneybeans",
    10: "lentil",
    11: "maize",
    12: "mango",
    13: "mothbeans",
    14: "mungbean",
    15: "muskmelon",
    16: "orange",
    17: "papaya",
    18: "pigeonpeas",
    19: "pomegranate",
    20: "rice",
    21: "watermelon"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop_api")

# The model is in the same directory as app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_random_forest_model.pkl"

MODEL = None

try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)

    # Test that the model can accept the expected 7 inputs.
    test_input = preprocess({
        "N": 0,
        "P": 0,
        "K": 0,
        "temperature": 0,
        "humidity": 0,
        "ph": 0,
        "rainfall": 0,
    })

    MODEL.predict(test_input)

    logger.info("Model loaded successfully")
    logger.info("Features: %s", FEATURES)

except Exception as exc:
    logger.exception("MODEL LOAD FAILED: %s", exc)


app = FastAPI(
    title="Smart Crop Recommendation API",
    version="1.0.0",
    description="Crop recommendation using a trained Random Forest model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CropInput(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float = Field(..., ge=-20, le=60)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)


@app.get("/")
def root():
    return {
        "message": "Crop Recommendation API is running"
    }


@app.get("/health")
def health():
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "detail": "Model not loaded"
            }
        )

    return {
        "status": "ok",
        "model": "loaded"
    }


@app.post("/predict")
def predict(payload: CropInput):

    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )

    try:

        # Convert request data into the exact model input format.
        X = preprocess(payload.model_dump())

        # Get prediction.
   prediction = int(MODEL.predict(X)[0])

   result = {
       "recommended_crop": CROP_LABELS[prediction]
   }

        # If the model supports probabilities, return top 3.
        if hasattr(MODEL, "predict_proba"):

            probabilities = MODEL.predict_proba(X)[0]
            classes = MODEL.classes_

            order = probabilities.argsort()[::-1][:3]

            recommendations = []

            for rank, index in enumerate(order, start=1):
                recommendations.append({
                    "rank": rank,
                    "crop": CROP_LABELS[int(classes[index])],
                    "probability": round(
                        float(probabilities[index]), 4
                    )
                })

            result["recommendations"] = recommendations

        return result

    except Exception as exc:

        logger.exception("Prediction failed")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "detail": str(exc)
            }
        )
