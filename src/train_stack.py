
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import *
from .pipeline import make_pipeline


print("Starting train_stack.py...")

# Step 1: Load training and testing data
print("Loading training and testing data...")

X_tr = pd.read_pickle(MODELS / "X_tr.pkl")
y_tr = pd.read_pickle(MODELS / "y_tr.pkl")
X_te = pd.read_pickle(MODELS / "X_te.pkl")
y_te = pd.read_pickle(MODELS / "y_te.pkl")

# Step 2: Load optimized XGBoost parameters
print("Loading optimized parameters...")

with open(MODELS / "best_params.json", "r") as f:
    bp = json.load(f)

# Step 3: Initialize tree-based models
print("Initializing XGBoost and LightGBM...")

xgb = XGBClassifier(
    **bp,
    scale_pos_weight=2.5,
    eval_metric="logloss",
    random_state=SEED
)

lgb = LGBMClassifier(
    scale_pos_weight=2.5,
    random_state=SEED,
    verbose=-1
)

# Step 4: Create the stacking ensemble
print("Creating stacking ensemble...")

stack = StackingClassifier(
    estimators=[
        ("xgb", make_pipeline(xgb)),
        ("lgbm", make_pipeline(lgb))
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method="predict_proba"
)

# Step 5: Calculate the threshold using training data only
print("Calculating threshold using cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

oof = cross_val_predict(
    stack,
    X_tr,
    y_tr,
    cv=cv,
    method="predict_proba"
)[:, 1]

prec, rec, thr = precision_recall_curve(y_tr, oof)

f1 = 2 * prec * rec / (prec + rec + 1e-9)

threshold = float(thr[np.argmax(f1[:-1])])

print("Selected threshold:", threshold)

# Step 6: Train the final stacking model
print("Starting final model training...")

stack.fit(X_tr, y_tr)

print("Final model training completed.")

# Step 7: Evaluate the model on test data
print("Evaluating model...")

p = stack.predict_proba(X_te)[:, 1]
pred = (p >= threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_te, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_te, pred))

print(
    "\nPR-AUC:",
    average_precision_score(y_te, p)
)

# Step 8: Save the trained model
print("\nSaving model files...")

joblib.dump(stack, MODELS / "stack.pkl")

with open(MODELS / "threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f, indent=4)

print("Model saved to:", MODELS / "stack.pkl")
print("Threshold saved to:", MODELS / "threshold.json")
print("Training pipeline completed successfully.")