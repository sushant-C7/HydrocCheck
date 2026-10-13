"""Local unsupervised anomaly detection for HydroCheck AI.

An anomaly is a sample that differs from patterns in the training dataset. It is
not, by itself, proof that water is unsafe or safe to consume.
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import DATA, FEATURES, MODELS, SEED

ANOMALY_MODEL_PATH = MODELS / "isolation_forest.pkl"


def build_anomaly_pipeline():
    """Create the preprocessing + Isolation Forest pipeline."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.05,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_anomaly_detector(save=True):
    """Train Isolation Forest on the available water-quality measurements."""
    if not DATA.exists():
        raise FileNotFoundError(f"Training dataset was not found: {DATA}")

    dataset = pd.read_csv(DATA)
    missing = [feature for feature in FEATURES if feature not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required features: {missing}")

    X = dataset[FEATURES].copy()
    model = build_anomaly_pipeline()
    model.fit(X)

    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ANOMALY_MODEL_PATH)

    return model


def load_anomaly_detector():
    """Load the saved detector, training it once if it does not exist yet."""
    if ANOMALY_MODEL_PATH.exists():
        return joblib.load(ANOMALY_MODEL_PATH)
    return train_anomaly_detector(save=True)


def detect_anomaly(measurements, model=None):
    """Return an anomaly flag and score for one water sample."""
    if model is None:
        model = load_anomaly_detector()

    row = {feature: measurements.get(feature) for feature in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    prediction = int(model.predict(X)[0])
    decision_score = float(model.decision_function(X)[0])
    is_anomaly = prediction == -1

    return {
        "is_anomaly": is_anomaly,
        "label": "Unusual pattern detected" if is_anomaly else "Pattern is within the learned profile",
        "decision_score": round(decision_score, 4),
        "method": "Isolation Forest",
        "note": (
            "An anomaly means the measurements differ from patterns in the training data; "
            "it does not independently establish whether water is safe or unsafe."
        ),
    }


def check_training_ranges(measurements):
    """Check whether readings fall outside the min/max values seen in training data."""
    dataset = pd.read_csv(DATA)
    out_of_range = []

    for feature in FEATURES:
        value = measurements.get(feature)
        if value is None:
            continue

        minimum = float(dataset[feature].min())
        maximum = float(dataset[feature].max())
        numeric_value = float(value)

        if numeric_value < minimum or numeric_value > maximum:
            out_of_range.append(
                {
                    "feature": feature,
                    "value": numeric_value,
                    "training_min": round(minimum, 4),
                    "training_max": round(maximum, 4),
                }
            )

    return {
        "outside_training_range": bool(out_of_range),
        "details": out_of_range,
        "note": (
            "Training-data ranges are not regulatory limits. A value outside them means "
            "the model is being asked to assess an unfamiliar input."
        ),
    }
