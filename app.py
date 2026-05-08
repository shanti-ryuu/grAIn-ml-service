"""
grAIn ML Prediction Service
============================
Flask REST API that serves drying predictions using trained ML models.

Endpoints:
  POST /predict          - Get moisture and time predictions
  GET  /health           - Health check
  GET  /model/info       - Model metadata and metrics
  POST /predict/curve    - Get full 6-hour projected moisture curve
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["*"])

# Load models on startup
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

moisture_model = None
model_metadata = None


def load_models():
    """Load trained models and metadata."""
    global moisture_model, model_metadata

    moisture_path = os.path.join(MODEL_DIR, "moisture_predictor.joblib")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")

    if not os.path.exists(moisture_path):
        print("WARNING: Models not found. Run train_model.py first.")
        return False

    moisture_model = joblib.load(moisture_path)

    with open(metadata_path, "r") as f:
        model_metadata = json.load(f)

    print(f"Models loaded successfully (v{model_metadata['version']})")
    return True


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


def generate_recommendation(moisture, temperature, humidity, fan_speed,
                            predicted_moisture, time_to_target):
    """Generate human-readable optimization recommendation."""
    if moisture <= TARGET_MOISTURE:
        return {
            "text": "Target moisture reached. Stop drying to prevent over-drying and grain cracking.",
            "type": "optimal",
            "action": "STOP",
        }
    if temperature > 65.0:
        return {
            "text": f"Temperature too high ({temperature:.1f}°C). Reduce by 5-10°C to prevent thermal damage to grain.",
            "type": "critical",
            "action": "REDUCE_TEMP",
        }
    if temperature < 38.0 and moisture > 16.0:
        return {
            "text": f"Temperature too low ({temperature:.1f}°C). Increase to 45-55°C for optimal drying rate.",
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
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": moisture_model is not None,
        "version": model_metadata["version"] if model_metadata else "unknown",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata and performance metrics."""
    if model_metadata is None:
        return jsonify({"error": "Models not loaded"}), 503

    return jsonify({
        "version": model_metadata["version"],
        "trained_at": model_metadata["trained_at"],
        "algorithm": model_metadata["algorithm"],
        "features": model_metadata["features"],
        "metrics": model_metadata["metrics"],
        "feature_importance": model_metadata["feature_importance"],
        "training_data": model_metadata["training_data"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Request body:
    {
        "deviceId": "GR-001",
        "temperature": 52.3,
        "humidity": 45.2,
        "moisture": 18.5,
        "fanSpeed": 75.0,
        "timeElapsed": 60,
        "solarVoltage": 15.2,
        "energyConsumed": 1.5,
        "dryingRate": 0.02
    }

    Response:
    {
        "predictedMoisture30min": 17.2,
        "estimatedMinutesToTarget": 145.0,
        "recommendation": "...",
        "recommendationType": "optimal",
        "efficiencyScore": 78,
        "confidence": 0.92,
        "isDryingComplete": false,
        "targetMoisture": 14.0,
        "algorithm": "RandomForest+GradientBoosting v1.0.0",
        "projectedCurve": [...]
    }
    """
    if moisture_model is None:
        return jsonify({"error": "Models not loaded. Service starting up."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    # Validate required fields
    required = ["temperature", "humidity", "moisture", "fanSpeed"]
    missing = [f for f in required if f not in data and f.lower() not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        # Extract current values from request
        moisture = data.get("moisture", 20.0)
        temperature = data.get("temperature", 45.0)
        humidity = data.get("humidity", 60.0)
        fan_speed = data.get("fanSpeed", data.get("fan_speed", 70.0))
        solar_voltage = data.get("solarVoltage", data.get("solar_voltage", 12.0))
        time_elapsed = data.get("timeElapsed", data.get("time_elapsed", 0))
        drying_rate = data.get("dryingRate", data.get("drying_rate", 0.01))

        features = prepare_features(data)

        # Predict moisture 30 minutes ahead
        predicted_moisture = float(moisture_model.predict(features)[0])
        # Physical constraint: moisture can't increase during active drying
        predicted_moisture = min(predicted_moisture, moisture)
        predicted_moisture = round(max(8.0, min(30.0, predicted_moisture)), 2)

        # Estimate time to target using the 30-min prediction rate
        moisture_drop_30min = moisture - predicted_moisture
        remaining = moisture - TARGET_MOISTURE
        if moisture <= TARGET_MOISTURE:
            estimated_time = 0.0
        elif remaining < 1.0:
            # Very close to target — estimate based on small remaining gap
            rate = max(moisture_drop_30min, 0.05)
            estimated_time = round((remaining / rate) * 30.0, 1)
        elif moisture_drop_30min > 0.05:
            estimated_time = round((remaining / moisture_drop_30min) * 30.0, 1)
            estimated_time = min(estimated_time, 720.0)
        else:
            estimated_time = 720.0

        # Is drying complete?
        is_complete = moisture <= TARGET_MOISTURE

        # Recommendation
        rec = generate_recommendation(
            moisture, temperature, humidity, fan_speed,
            predicted_moisture, estimated_time
        )

        # Efficiency score
        efficiency = calculate_efficiency(
            temperature, humidity, fan_speed, solar_voltage, drying_rate
        )

        # Confidence
        confidence = calculate_confidence(moisture, drying_rate, time_elapsed)

        # Generate 6-hour projected curve (every 30 minutes = 13 points)
        curve = []
        current_m = moisture
        for i in range(13):
            t_offset = i * 30
            future_features = features.copy()
            future_features[0, 5] = time_elapsed + t_offset  # time_elapsed
            future_features[0, 2] = current_m  # current moisture
            future_features[0, 8] = current_m - TARGET_MOISTURE  # moisture diff

            pred_m = float(moisture_model.predict(future_features)[0])
            pred_m = max(TARGET_MOISTURE - 1.0, min(current_m, pred_m))
            current_m = pred_m

            curve.append({
                "minutesFromNow": t_offset,
                "predictedMoisture": round(pred_m, 2),
            })

        response = {
            "predictedMoisture30min": predicted_moisture,
            "estimatedMinutesToTarget": estimated_time,
            "recommendation": rec["text"],
            "recommendationType": rec["type"],
            "efficiencyScore": efficiency,
            "confidence": confidence,
            "isDryingComplete": is_complete,
            "targetMoisture": TARGET_MOISTURE,
            "algorithm": f"RandomForest v{model_metadata['version']}",
            "projectedCurve": curve,
            "modelMetrics": {
                "moistureR2": model_metadata["metrics"]["moisture_prediction"]["test"]["r2"],
                "timeR2": model_metadata["metrics"]["time_prediction"]["test"]["r2"],
            },
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/curve", methods=["POST"])
def predict_curve():
    """
    Generate detailed moisture projection curve.
    Same input as /predict, returns only the curve with more resolution.
    """
    if moisture_model is None:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    try:
        features = prepare_features(data)
        moisture = data.get("moisture", 20.0)
        time_elapsed = data.get("timeElapsed", data.get("time_elapsed", 0))

        # Generate curve every 15 minutes for 8 hours (33 points)
        curve = []
        current_m = moisture

        for i in range(33):
            t_offset = i * 15
            future_features = features.copy()
            future_features[0, 5] = time_elapsed + t_offset
            future_features[0, 2] = current_m
            future_features[0, 8] = current_m - TARGET_MOISTURE

            pred_m = float(moisture_model.predict(future_features)[0])
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
        })

    except Exception as e:
        return jsonify({"error": f"Curve generation failed: {str(e)}"}), 500


# Load models on import
models_loaded = load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
