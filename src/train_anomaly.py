"""Train and save the HydroCheck Isolation Forest detector.

Run from the project root with: python -m src.train_anomaly
"""
from .anomaly import ANOMALY_MODEL_PATH, train_anomaly_detector

if __name__ == "__main__":
    train_anomaly_detector(save=True)
    print(f"Isolation Forest saved to: {ANOMALY_MODEL_PATH}")
