from __future__ import annotations

import pandas as pd

# These are the exact 7 features used by the trained Random Forest model.
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

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
    """Create a DataFrame in the exact feature order expected by the model."""

    row = {feature: float(values[feature]) for feature in FEATURES}

    return pd.DataFrame([row], columns=FEATURES)
