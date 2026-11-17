"""
Part 9: Flask web app for crop recommendation.

Run from the project root:   python -m app.app      (opens at http://127.0.0.1:5000)
Production (Render etc.):    gunicorn app.app:app
"""
import io
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from src import config
from src.features import FeatureSelector  # noqa: F401  (needed to load the saved pipeline)

warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024        # 2 MB upload limit

MODEL = joblib.load(config.MODELS_DIR / "crop_model.pkl")
ENCODER = joblib.load(config.LABEL_ENCODER_PATH)
RANGES = json.load(open(config.MODELS_DIR / "feature_ranges.json"))
INFO = json.load(open(config.MODELS_DIR / "model_info.json"))

FIELDS = [  # name, label, unit, hard min, hard max, example
    ("N", "Nitrogen (N)", "dataset units", 0, 300, 90),
    ("P", "Phosphorus (P)", "dataset units", 0, 300, 42),
    ("K", "Potassium (K)", "dataset units", 0, 300, 43),
    ("temperature", "Temperature", "°C", -10, 60, 21),
    ("humidity", "Humidity", "%", 0, 100, 82),
    ("ph", "Soil pH", "0 to 14", 0, 14, 6.5),
    ("rainfall", "Rainfall", "mm", 0, 500, 203),
]
LIMITS = {f[0]: (f[3], f[4]) for f in FIELDS}


def validate(values: dict):
    """Return (clean dict of floats, list of errors, list of warnings)."""
    clean, errors, warns = {}, [], []
    for name, label, unit, lo, hi, _ in FIELDS:
        raw = values.get(name)
        try:
            v = float(raw)
            if not np.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{label}: enter a number.")
            continue
        if not lo <= v <= hi:
            errors.append(f"{label}: {v:g} is outside the allowed range {lo} to {hi}.")
            continue
        clean[name] = v
        tr_lo, tr_hi = RANGES[name]
        if not tr_lo <= v <= tr_hi:
            warns.append(f"{label} ({v:g}) is outside the range the model was trained on "
                         f"({tr_lo:g} to {tr_hi:g}), so this prediction may be less reliable.")
    return clean, errors, warns


def predict_df(df: pd.DataFrame, top=3):
    proba = MODEL.predict_proba(df[config.FEATURES])
    order = np.argsort(-proba, axis=1)[:, :top]
    out = []
    for i, idx in enumerate(order):
        out.append([{"crop": ENCODER.classes_[j], "probability": float(proba[i, j])} for j in idx])
    return out


@app.route("/")
def index():
    return render_template("index.html", fields=FIELDS, result=None, errors=[], warns=[], values={}, batch=None)


@app.route("/predict", methods=["POST"])
def predict():
    values = {f[0]: request.form.get(f[0]) for f in FIELDS}
    clean, errors, warns = validate(values)
    result = None
    if not errors:
        result = predict_df(pd.DataFrame([clean]))[0]
    return render_template("index.html", fields=FIELDS, result=result, errors=errors, warns=warns,
                           values=values, batch=None)


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        return render_template("index.html", fields=FIELDS, result=None, values={}, warns=[], batch=None,
                               errors=["Please choose a .csv file."]), 400
    try:
        df = pd.read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        missing = [c for c in config.FEATURES if c not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        df = df[config.FEATURES].apply(pd.to_numeric, errors="coerce")
        if df.isnull().any().any():
            raise ValueError("Some values are empty or not numeric.")
        if len(df) > 5000:
            raise ValueError("Please upload at most 5,000 rows.")
    except Exception as e:  # report any problem with the file to the user
        return render_template("index.html", fields=FIELDS, result=None, values={}, warns=[], batch=None,
                               errors=[f"Could not read the file: {e}"]), 400
    preds = predict_df(df)
    res = df.copy()
    res["recommended_crop"] = [p[0]["crop"] for p in preds]
    res["confidence"] = [round(p[0]["probability"], 3) for p in preds]
    buf = io.StringIO()
    res.to_csv(buf, index=False)
    app.config["LAST_BATCH"] = buf.getvalue()
    return render_template("index.html", fields=FIELDS, result=None, errors=[], warns=[], values={},
                           batch={"rows": res.head(20).to_dict("records"), "total": len(res)})


@app.route("/download")
def download():
    data = app.config.get("LAST_BATCH")
    if not data:
        return "No predictions to download yet.", 404
    return send_file(io.BytesIO(data.encode()), mimetype="text/csv", as_attachment=True,
                     download_name="crop_predictions.csv")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    clean, errors, warns = validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify({"predictions": predict_df(pd.DataFrame([clean]))[0], "warnings": warns})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": INFO["best_model"], "classes": len(ENCODER.classes_)})


if __name__ == "__main__":
    app.run(debug=False)
