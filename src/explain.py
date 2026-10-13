
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from .config import *

print("Loading trained stacking model...")

stack = joblib.load(MODELS / "stack.pkl")
X_te = pd.read_pickle(MODELS / "X_te.pkl")

# Select the XGBoost pipeline from the stacking model
xgb_pipe = stack.named_estimators_["xgb"]

print("Preparing data for SHAP explanations...")

# Apply preprocessing steps individually
preprocessor = xgb_pipe[:-1]

# IterativeImputer and feature engineering
X_imputed = preprocessor.named_steps["impute"].transform(X_te)

X_ratios = preprocessor.named_steps["ratios"].transform(X_imputed)

X_scaled = preprocessor.named_steps["scale"].transform(X_ratios)

# Create a DataFrame with feature names
Xp = pd.DataFrame(X_scaled, columns=ALL_FEATURES)

# Get the trained XGBoost classifier
xgb_model = xgb_pipe.named_steps["clf"]

print("Generating SHAP explanations...")

explainer = shap.TreeExplainer(xgb_model)
sv = explainer(Xp)

# Create reports directory if needed
REPORTS.mkdir(parents=True, exist_ok=True)

# SHAP summary plot
shap.summary_plot(sv.values, Xp, show=False)
plt.savefig(
    REPORTS / "shap_summary.png",
    bbox_inches="tight",
    dpi=150
)
plt.close()

print("Saved SHAP summary plot.")

# SHAP waterfall plot for the first test sample
shap.plots.waterfall(sv[0], show=False)
plt.savefig(
    REPORTS / "shap_sample0.png",
    bbox_inches="tight",
    dpi=150
)
plt.close()

print("Saved SHAP waterfall plot.")
print("SHAP explanations completed successfully.")