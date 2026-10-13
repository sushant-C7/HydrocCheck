import json
import math
import joblib
import numpy as np
import pandas as pd
import shap

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from src.config import *
from src.rules import rule_gate
from src.anomaly import check_training_ranges, detect_anomaly, load_anomaly_detector
from src.agent import AgentConfigurationError, run_hydrocheck_agent, run_rag_research_assistant
from src.rag import search_knowledge, knowledge_base_status

load_dotenv(ROOT / ".env")

app = Flask(__name__)

stack = joblib.load(MODELS / "stack.pkl")

with open(MODELS / "threshold.json", "r", encoding="utf-8") as f:
    THR = json.load(f)["threshold"]

xgb_pipe = stack.named_estimators_["xgb"]
explainer = shap.TreeExplainer(xgb_pipe.named_steps["clf"])

# The anomaly detector runs locally and trains once automatically if its pickle is absent.
anomaly_model = load_anomaly_detector()


def validate_measurements(data):
    """Validate request shape and convert each supplied measurement to a finite number."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return None, f"Missing fields: {', '.join(missing)}"

    cleaned = {}
    for feature in FEATURES:
        value = data[feature]
        if value is None or value == "":
            cleaned[feature] = np.nan
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None, f"'{feature}' must be a numeric value or null."
        if not math.isfinite(numeric):
            return None, f"'{feature}' must be a finite number."
        if numeric < 0:
            return None, f"'{feature}' cannot be negative."
        cleaned[feature] = numeric

    return cleaned, None


def predict_sample(data):
    """Run the existing rule gate + stacking ensemble + SHAP explanation."""
    cleaned, error = validate_measurements(data)
    if error:
        return {"error": error}

    # Isolation Forest is independent of the classifier and its result is returned
    # even when the pre-existing rule gate flags a sample.
    anomaly_result = detect_anomaly(cleaned, anomaly_model)
    reasons = rule_gate(cleaned)
    if reasons:
        return {
            "potable": False,
            "source": "rule_gate",
            "reasons": reasons,
            "anomaly": anomaly_result,
            "message": "A rule-based check flagged the sample. This is not a certified laboratory result.",
        }

    X = pd.DataFrame([{feature: cleaned[feature] for feature in FEATURES}], columns=FEATURES)
    probability = float(stack.predict_proba(X)[0, 1])

    # The current model's SHAP explanation is computed for its XGBoost component,
    # which is part of the full stacking ensemble. The UI labels it as a component explanation.
    X_imputed = xgb_pipe.named_steps["impute"].transform(X)
    X_ratios = xgb_pipe.named_steps["ratios"].transform(X_imputed)
    X_scaled = xgb_pipe.named_steps["scale"].transform(X_ratios)
    X_shap = pd.DataFrame(X_scaled, columns=ALL_FEATURES)
    shap_values = explainer.shap_values(X_shap)[0]

    top_factors = sorted(
        zip(ALL_FEATURES, shap_values),
        key=lambda item: -abs(float(item[1])),
    )[:3]

    return {
        "potable": bool(probability >= THR),
        "probability": round(probability, 3),
        "threshold": round(float(THR), 3),
        "source": "model",
        "top_factors": [
            {"feature": feature, "shap_log_odds": round(float(value), 3)}
            for feature, value in top_factors
        ],
        "explanation_scope": "XGBoost component of the stacking ensemble",
        "anomaly": anomaly_result,
        "message": "Machine-learning assessment only; verify water quality with certified laboratory testing.",
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "anomaly_detector": "loaded", "rag": knowledge_base_status()}


@app.post("/research")
def research():
    """Answer a reference question with local RAG; does not require sample measurements."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    question = data.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify(error="Enter a research question."), 400
    if len(question) > 1200:
        return jsonify(error="Please keep the research question under 1,200 characters."), 400

    try:
        return jsonify(run_rag_research_assistant(question, search_knowledge))
    except AgentConfigurationError as exc:
        return jsonify(error=str(exc), code="agent_not_configured"), 503
    except Exception:
        app.logger.exception("HydroCheck RAG research request failed")
        return jsonify(error="The research assistant failed. Check the API key, model name, network connection, and API usage limits."), 502


@app.post("/knowledge/search")
def knowledge_search():
    """Search the local reference collection without making an LLM API call."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    query = data.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return jsonify(error="Enter a query for the local knowledge base."), 400
    if len(query) > 600:
        return jsonify(error="Please keep the knowledge search under 600 characters."), 400
    return jsonify(search_knowledge(query, top_k=4))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be valid JSON."), 400

    result = predict_sample(data)
    if "error" in result:
        return jsonify(error=result["error"]), 400
    return jsonify(result)


@app.post("/agent")
def agent():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be valid JSON."), 400

    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Enter a question or task for the HydroCheck AI Agent."), 400
    if len(message) > 1200:
        return jsonify(error="Please keep the agent request under 1,200 characters."), 400

    measurements, error = validate_measurements(data.get("measurements"))
    if error:
        return jsonify(error=f"Measurement input: {error}"), 400

    try:
        result = run_hydrocheck_agent(
            message=message,
            measurements=measurements,
            prediction_fn=predict_sample,
            anomaly_fn=lambda values: detect_anomaly(values, anomaly_model),
            range_fn=check_training_ranges,
            knowledge_fn=search_knowledge,
        )
        return jsonify(result)
    except AgentConfigurationError as exc:
        return jsonify(error=str(exc), code="agent_not_configured"), 503
    except Exception:
        app.logger.exception("HydroCheck agent call failed")
        return jsonify(
            error="The AI agent request failed. Check the API key, model name, network connection, and API usage limits."
        ), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
