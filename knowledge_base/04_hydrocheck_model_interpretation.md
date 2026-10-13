---
title: HydroCheck AI — Model Interpretation and Limitations
publisher: HydroCheck AI project documentation
jurisdiction: Project-specific technical note
published: 2026-10-13
source_url: 
---

# Understanding HydroCheck AI results

## What the prediction means
The application uses a trained stacking machine-learning classifier to produce a model probability and a class label at the configured decision threshold. A classification is a model output for the features and data distribution it was trained on. It is not a laboratory result, official certification, or guarantee of drinking-water safety.

## Explainable AI output
The current SHAP explanation is produced for the XGBoost component of the stacking ensemble. Unless the software explicitly implements an explanation for the full stacked model, describe this as an explanation of the XGBoost component rather than a full-ensemble explanation. SHAP feature contributions describe how the selected model output is attributed across features; they do not prove causation.

## Anomaly detection
Isolation Forest flags samples that appear unusual compared with patterns in the reference training data. An anomaly can arise for many reasons, including a genuine unusual sample, a unit mismatch, a data-entry issue or a distribution shift. An anomaly flag alone does not mean the water is unsafe; absence of an anomaly does not mean it is safe.

## Training-range checks
Whether a value is inside the minimum and maximum seen in training only describes the observed data range. It is not a regulatory limit or a safety threshold.

## Recommendations for responsible interpretation
Report the model's output, its known limitations and any relevant retrieved sources separately. Do not imply causation from feature attribution. Recommend appropriate laboratory confirmation and consultation of applicable official standards for consequential decisions.
