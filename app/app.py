"""
app/app.py - Task 6: Professional Front-End Solar Generation Predictor

Serves an interactive dashboard using Flask and modern HTML5/CSS3/JavaScript.
Loads pre-trained model weights from results/saved_weights.json, computes
predictions via NumPy, and offers real-time simulation controls with
comparisons across Normal Equation, Batch GD, and Stochastic GD solvers.
"""

from pathlib import Path
import numpy as np
import json
import os
import sys

from flask import Flask, request, jsonify, render_template

# Project paths
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "results" / "saved_weights.json"
TEMPLATES_DIR = ROOT / "app" / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Load pre-trained weights and scaler statistics
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {WEIGHTS_PATH}. Run src/train_eval.py first to train models and export weights."
    )

with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
    MODEL_DATA = json.load(f)

SCALER_B = MODEL_DATA["scaler_B"]
MU_B = np.array(SCALER_B["mean"])
SIGMA_B = np.array(SCALER_B["std"])
THETA_NORM_B = np.array(MODEL_DATA["theta_norm_B"])
THETA_BGD_B = np.array(MODEL_DATA["theta_bgd_B"])
THETA_SGD_B = np.array(MODEL_DATA["theta_sgd_B"])
PEAK_POWER = MODEL_DATA.get("peak_hourly_power", 27325.90)


def predict_power(hour: float, sw_radiation: float, temp_2m: float, cloud_cover: float, solver: str = "normal") -> dict:
    """
    Compute AC power prediction given Set B public weather inputs.
    """
    # 1. Cyclical time features: sin(2pi*h/24) and cos(2pi*h/24)
    sin_hour = float(np.sin(2.0 * np.pi * hour / 24.0))
    cos_hour = float(np.cos(2.0 * np.pi * hour / 24.0))

    # Raw features: [sw_radiation, temp_2m, cloud_cover, sin_hour, cos_hour]
    raw_features = np.array([sw_radiation, temp_2m, cloud_cover, sin_hour, cos_hour], dtype=float)

    # 2. Standardize with training set statistics (no leakage)
    scaled_features = (raw_features - MU_B) / SIGMA_B

    # 3. Add intercept x_0 = 1
    x_design = np.insert(scaled_features, 0, 1.0)

    # 4. Compute predictions for all three solvers
    preds = {
        "normal": max(0.0, float(np.dot(x_design, THETA_NORM_B))),
        "batch_gd": max(0.0, float(np.dot(x_design, THETA_BGD_B))),
        "sgd": max(0.0, float(np.dot(x_design, THETA_SGD_B))),
    }

    selected_power = preds.get(solver, preds["normal"])
    capacity_pct = min(100.0, max(0.0, (selected_power / PEAK_POWER) * 100.0))

    return {
        "hour": hour,
        "sw_radiation": sw_radiation,
        "temp_2m": temp_2m,
        "cloud_cover": cloud_cover,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "predicted_power_kw": round(selected_power, 2),
        "capacity_utilization_pct": round(capacity_pct, 2),
        "all_solvers": {
            "Normal Equation": round(preds["normal"], 2),
            "Batch GD": round(preds["batch_gd"], 2),
            "Stochastic GD": round(preds["sgd"], 2),
        },
        "peak_plant_capacity_kw": round(PEAK_POWER, 2),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the interactive professional dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def api_predict():
    """REST API endpoint returning JSON prediction for interactive sliders."""
    data = request.get_json(force=True)
    hour = float(data.get("hour", 12))
    sw = float(data.get("sw_radiation", 850.0))
    temp = float(data.get("temp_2m", 33.5))
    cloud = float(data.get("cloud_cover", 15))
    solver = data.get("solver", "normal")

    res = predict_power(hour, sw, temp, cloud, solver)
    return jsonify(res)


if __name__ == "__main__":
    print("=" * 60)
    print("SOLARIS AI • SOLAR POWER PLANT GENERATION DASHBOARD")
    print("Running on: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
