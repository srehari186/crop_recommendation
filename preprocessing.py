"""
preprocessing.py - input preprocessing for the crop prediction API.

IMPORTANT: during training NO scaling/encoding was applied. The Random Forest
was trained on the 7 RAW feature values in the exact order below. This module
therefore only:
  * loads the canonical feature order from ../model/feature_names.json
  * builds a pandas DataFrame in that exact order from the incoming values

Changing this order would silently break predictions, so it is pinned to the
saved file and never hard-coded elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# backend/ -> repo root -> model/
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def load_feature_names() -> list[str]:
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


FEATURES: list[str] = load_feature_names()

# Human-readable valid ranges (used by the API for input validation).
# These mirror the physical meaning of each feature, not the model internals.
RANGES = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 150),
    "temperature": (-20, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 400),
}


def preprocess(values: dict) -> pd.DataFrame:
    """Return a 1-row DataFrame in the model's exact feature order (raw units)."""
    row = {feat: float(values[feat]) for feat in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)
