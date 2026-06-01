"""
grAIn ML Prediction Service
============================
Flask REST API that serves drying predictions using trained ML models.

Endpoints:
  POST /predict          - Get moisture and time predictions
  GET  /health           - Health check
  GET  /api/v1/ping      - Render keep-alive ping
  GET  /model/info       - Model metadata and metrics
  GET  /model/compare    - Model comparison table
  POST /predict/curve    - Get full 8-hour projected moisture curve
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

moisture_model_rf = None
moisture_model_xgb = None
model_metadata = None
rf_weight = 0.5
xgb_weight = 0.5

FEATURE_ORDER = [
    "temperature",
    "humidity",
    "moisture",
    "fan_speed",
    "solar_voltage",
    "time_elapsed",
    "energy_consumed",
    "drying_rate",
    "moisture_diff_from_target",
]

TARGET_MOISTURE = 14.0


def load_models():
    """Load trained models and metadata."""
    global moisture_model_rf, moisture_model_xgb, model_metadata, rf_weight, xgb_weight

    rf_path = os.path.join(MODEL_DIR, "moisture_predictor_rf.joblib")
    xgb_path = os.path.join(MODEL_DIR, "moisture_predictor_xgb.joblib")
    legacy_path = os.path.join(MODEL_DIR, "moisture_predictor.joblib")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")

    if os.path.exists(rf_path):
        moisture_model_rf = joblib.load(rf_path)
        print("RF model loaded OK")
    elif os.path.exists(legacy_path):
        moisture_model_rf = joblib.load(legacy_path)
        print("RF model loaded from legacy moisture_predictor.joblib OK")
    else:
        print("WARNING: RF model not found. Run train_model.py first.")

    if os.path.exists(xgb_path):
        try:
            moisture_model_xgb = joblib.load(xgb_path)
            print("XGBoost model loaded OK")
        except Exception as exc:
            moisture_model_xgb = None
            print(f"WARNING: XGBoost model could not be loaded. Falling back to RF-only predictions. {exc}")
    else:
        print("WARNING: XGBoost model not found. Falling back to RF-only predictions.")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            model_metadata = json.load(f)
        ensemble = model_metadata.get("ensemble", {})
        rf_weight = float(ensemble.get("rf_weight", 0.5))
        xgb_weight = float(ensemble.get("xgb_weight", 0.5))
    else:
        print("WARNING: model_metadata.json not found.")

    print(f"Ensemble weights: RF={rf_weight:.2f}, XGB={xgb_weight:.2f}")
    return moisture_model_rf is not None


def active_algorithm_name() -> str:
    return "Ensemble (RF+XGBoost)" if moisture_model_rf is not None and moisture_model_xgb is not None else "RandomForest"


def prepare_features(data: dict) -> np.ndarray:
    """Extract and order features from request payload."""
    moisture = data.get("moisture", 20.0)
    time_elapsed = data.get("timeElapsed", data.get("time_elapsed", 0))

    features = [
        data.get("temperature", 45.0),
        data.get("humidity", 60.0),
        moisture,
        data.get("fanSpeed", data.get("fan_speed", 70.0)),
        data.get("solarVoltage", data.get("solar_voltage", 12.0)),
        time_elapsed,
        data.get("energyConsumed", data.get("energy_consumed", 0.0)),
        data.get("dryingRate", data.get("drying_rate", 0.01)),
        moisture - TARGET_MOISTURE,
    ]

    return np.array([features])


def predict_moisture(features: np.ndarray) -> float:
    """Predict moisture with weighted ensemble when XGBoost is available."""
    if moisture_model_rf is None:
        raise RuntimeError("RF model is not loaded")

    rf_pred = float(moisture_model_rf.predict(features)[0])
    if moisture_model_xgb is None:
        return rf_pred

    xgb_pred = float(moisture_model_xgb.predict(features)[0])
    return (rf_weight * rf_pred) + (xgb_weight * xgb_pred)


def model_metric(model_key: str, metric: str):
    if not model_metadata:
        return None
    return model_metadata.get("metrics", {}).get(model_key, {}).get(metric)


def model_metrics_payload() -> dict:
    return {
        "moistureR2_rf": model_metric("random_forest", "r2"),
        "moistureR2_xgb": model_metric("xgboost", "r2"),
        "moistureR2_ensemble": model_metric("ensemble", "r2"),
        "rfWeight": rf_weight,
        "xgbWeight": xgb_weight,
    }


def generate_recommendation(moisture, temperature, humidity, fan_speed, predicted_moisture, time_to_target):
    """Generate human-readable optimization recommendation."""
    if moisture <= TARGET_MOISTURE:
        return {
            "text": "Target moisture reached. Stop drying to prevent over-drying and grain cracking.",
            "type": "optimal",
            "action": "STOP",
        }
    if temperature > 65.0:
        return {
            "text": f"Temperature too high ({temperature:.1f}C). Reduce by 5-10C to prevent thermal damage to grain.",
            "type": "critical",
            "action": "REDUCE_TEMP",
        }
    if temperature < 38.0 and moisture > 16.0:
        return {
            "text": f"Temperature too low ({temperature:.1f}C). Increase to 45-55C for optimal drying rate.",
            "type": "warning",
            "action": "INCREASE_TEMP",
        }
    if humidity > 75.0:
        return {
            "text": f"Ambient humidity high ({humidity:.1f}%). Increase exhaust fan speed for better moisture removal.",
            "type": "warning",
            "action": "INCREASE_FAN",
        }
    if fan_speed < 40.0 and moisture > 16.0:
        return {
            "text": f"Fan speed low ({fan_speed:.0f}%). Increase to 60-80% for improved airflow distribution.",
            "type": "warning",
            "action": "INCREASE_FAN",
        }
    if time_to_target < 30.0 and moisture > 14.5:
        return {
            "text": f"Nearly complete (~{time_to_target:.0f} min remaining). Maintain current settings.",
            "type": "optimal",
            "action": "MAINTAIN",
        }
    return {
        "text": "Drying conditions are optimal. Continue current settings for efficient moisture removal.",
        "type": "optimal",
        "action": "MAINTAIN",
    }


def calculate_efficiency(temperature, humidity, fan_speed, solar_voltage, drying_rate):
    """Calculate drying efficiency score (0-100)."""
    temp_score = min(30, max(0, 30 * (1 - abs(temperature - 50) / 25)))
    fan_score = min(25, max(0, 25 * (fan_speed / 100)))
    humidity_score = min(20, max(0, 20 * (1 - (humidity - 30) / 60)))
    solar_score = min(15, max(0, 15 * (solar_voltage / 20)))
    rate_score = min(10, max(0, 10 * min(1, drying_rate / 0.05)))
    return round(temp_score + fan_score + humidity_score + solar_score + rate_score)


def calculate_confidence(moisture, drying_rate, time_elapsed):
    """Calculate prediction confidence (0.0-1.0)."""
    base = 0.85
    if drying_rate > 0.001:
        base += 0.05
    if time_elapsed > 30:
        base += 0.05
    if 12.0 < moisture < 26.0:
        base += 0.03
    noise = np.random.uniform(-0.02, 0.02)
    return round(min(0.97, max(0.65, base + noise)), 2)


@app.route("/", methods=["GET"])
def index():
    """Root endpoint."""
    return jsonify({
        "service": "grAIn ML Prediction Service",
        "version": model_metadata["version"] if model_metadata else "unknown",
        "status": "running",
        "endpoints": {
            "POST /predict": "Get drying predictions",
            "POST /predict/curve": "Get detailed moisture curve",
            "GET /health": "Health check",
            "GET /model/info": "Model metadata and metrics",
            "GET /model/compare": "Compare RF, XGBoost, and ensemble metrics",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "rf": moisture_model_rf is not None,
            "xgb": moisture_model_xgb is not None,
            "ensemble": moisture_model_rf is not None and moisture_model_xgb is not None,
        },
        "active_algorithm": active_algorithm_name(),
        "version": model_metadata["version"] if model_metadata else "2.0.0",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/ping", methods=["GET"])
@app.route("/api/v1/ping", methods=["GET"])
def ping():
    """Lightweight Render keep-alive endpoint."""
    return "pong", 200, {"Content-Type": "text/plain", "Cache-Control": "no-store"}


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata and performance metrics."""
    if model_metadata is None:
        return jsonify({"error": "Models not loaded"}), 503

    return jsonify(model_metadata)


@app.route("/model/compare", methods=["GET"])
def model_compare():
    """Return comparison metrics for all moisture prediction models."""
    if model_metadata is None:
        return jsonify({"error": "Models not loaded"}), 503

    metrics = model_metadata.get("metrics", {})
    comparison = [
        {
            "model": "Random Forest",
            "r2": metrics.get("random_forest", {}).get("r2"),
            "rmse": metrics.get("random_forest", {}).get("rmse"),
            "mae": metrics.get("random_forest", {}).get("mae"),
            "mape": metrics.get("random_forest", {}).get("mape"),
            "cv_r2": metrics.get("random_forest", {}).get("cv_r2_mean"),
            "weight": rf_weight,
        },
        {
            "model": "XGBoost",
            "r2": metrics.get("xgboost", {}).get("r2"),
            "rmse": metrics.get("xgboost", {}).get("rmse"),
            "mae": metrics.get("xgboost", {}).get("mae"),
            "mape": metrics.get("xgboost", {}).get("mape"),
            "cv_r2": metrics.get("xgboost", {}).get("cv_r2_mean"),
            "weight": xgb_weight,
        },
        {
            "model": "Ensemble (RF+XGBoost)",
            "r2": metrics.get("ensemble", {}).get("r2"),
            "rmse": metrics.get("ensemble", {}).get("rmse"),
            "mae": metrics.get("ensemble", {}).get("mae"),
            "mape": metrics.get("ensemble", {}).get("mape"),
            "cv_r2": None,
            "weight": None,
            "isActive": moisture_model_rf is not None and moisture_model_xgb is not None,
        },
    ]

    return jsonify({
        "comparison": comparison,
        "activeModel": active_algorithm_name(),
        "version": model_metadata.get("version", "2.0.0"),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint."""
    if moisture_model_rf is None:
        return jsonify({"error": "Models not loaded. Service starting up."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    required = ["temperature", "humidity", "moisture", "fanSpeed"]
    missing = [f for f in required if f not in data and f.lower() not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        moisture = data.get("moisture", 20.0)
        temperature = data.get("temperature", 45.0)
        humidity = data.get("humidity", 60.0)
        fan_speed = data.get("fanSpeed", data.get("fan_speed", 70.0))
        solar_voltage = data.get("solarVoltage", data.get("solar_voltage", 12.0))
        time_elapsed = data.get("timeElapsed", data.get("time_elapsed", 0))
        drying_rate = data.get("dryingRate", data.get("drying_rate", 0.01))

        features = prepare_features(data)

        predicted_moisture = predict_moisture(features)
        predicted_moisture = min(predicted_moisture, moisture)
        predicted_moisture = round(max(8.0, min(30.0, predicted_moisture)), 2)

        moisture_drop_30min = moisture - predicted_moisture
        remaining = moisture - TARGET_MOISTURE
        if moisture <= TARGET_MOISTURE:
            estimated_time = 0.0
        elif remaining < 1.0:
            rate = max(moisture_drop_30min, 0.05)
            estimated_time = round((remaining / rate) * 30.0, 1)
        elif moisture_drop_30min > 0.05:
            estimated_time = round((remaining / moisture_drop_30min) * 30.0, 1)
            estimated_time = min(estimated_time, 720.0)
        else:
            estimated_time = 720.0

        rec = generate_recommendation(
            moisture, temperature, humidity, fan_speed, predicted_moisture, estimated_time
        )
        efficiency = calculate_efficiency(
            temperature, humidity, fan_speed, solar_voltage, drying_rate
        )
        confidence = calculate_confidence(moisture, drying_rate, time_elapsed)

        curve = []
        current_m = moisture
        for i in range(13):
            t_offset = i * 30
            future_features = features.copy()
            future_features[0, 5] = time_elapsed + t_offset
            future_features[0, 2] = current_m
            future_features[0, 8] = current_m - TARGET_MOISTURE

            pred_m = predict_moisture(future_features)
            pred_m = max(TARGET_MOISTURE - 1.0, min(current_m, pred_m))
            current_m = pred_m

            curve.append({
                "minutesFromNow": t_offset,
                "predictedMoisture": round(pred_m, 2),
            })

        algorithm = (
            f"Ensemble (RF+XGBoost) v{model_metadata['version']}"
            if moisture_model_xgb is not None and model_metadata
            else f"RandomForest v{model_metadata['version'] if model_metadata else '2.0.0'}"
        )

        return jsonify({
            "predictedMoisture30min": predicted_moisture,
            "estimatedMinutesToTarget": estimated_time,
            "recommendation": rec["text"],
            "recommendationType": rec["type"],
            "efficiencyScore": efficiency,
            "confidence": confidence,
            "isDryingComplete": moisture <= TARGET_MOISTURE,
            "targetMoisture": TARGET_MOISTURE,
            "algorithm": algorithm,
            "projectedCurve": curve,
            "modelMetrics": model_metrics_payload(),
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/curve", methods=["POST"])
def predict_curve():
    """Generate detailed moisture projection curve."""
    if moisture_model_rf is None:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    try:
        features = prepare_features(data)
        moisture = data.get("moisture", 20.0)
        time_elapsed = data.get("timeElapsed", data.get("time_elapsed", 0))

        curve = []
        current_m = moisture

        for i in range(33):
            t_offset = i * 15
            future_features = features.copy()
            future_features[0, 5] = time_elapsed + t_offset
            future_features[0, 2] = current_m
            future_features[0, 8] = current_m - TARGET_MOISTURE

            pred_m = predict_moisture(future_features)
            pred_m = max(TARGET_MOISTURE - 1.5, min(current_m + 0.1, pred_m))
            current_m = pred_m

            curve.append({
                "minutesFromNow": t_offset,
                "predictedMoisture": round(pred_m, 2),
                "isComplete": pred_m <= TARGET_MOISTURE,
            })

            if pred_m <= TARGET_MOISTURE - 1.0:
                break

        return jsonify({
            "curve": curve,
            "currentMoisture": moisture,
            "targetMoisture": TARGET_MOISTURE,
            "totalPointsGenerated": len(curve),
            "algorithm": active_algorithm_name(),
            "modelMetrics": model_metrics_payload(),
        })

    except Exception as e:
        return jsonify({"error": f"Curve generation failed: {str(e)}"}), 500


models_loaded = load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
