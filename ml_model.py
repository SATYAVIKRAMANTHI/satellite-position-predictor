"""
ml_model.py
-----------
ML pipeline for satellite position prediction.

Models:
    - Random Forest Regressor
    - XGBoost Regressor
    - Best model selected per target variable by R² score

Targets:
    latitude, longitude, altitude_km, speed_km_s

Evaluation:
    MAE, RMSE, R² per target
"""

import os
import math
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from feature_engineering import FEATURE_COLS, TARGET_COLS, generate_training_dataset, build_prediction_features

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "cache", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODELS_PATH = os.path.join(MODEL_DIR, "best_models.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.joblib")
ACTUALS_PATH = os.path.join(MODEL_DIR, "actuals.joblib")


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(tle_data: dict, hours: int = 168, step_minutes: int = 5) -> dict:
    """
    Train Random Forest and XGBoost models on propagated TLE data.

    Parameters
    ----------
    tle_data     : dict — TLE dict from tle_fetcher
    hours        : int  — Propagation hours for training data
    step_minutes : int  — Interval between training samples

    Returns
    -------
    dict with keys:
        metrics      — {target: {model: {mae, rmse, r2}}}
        best_model   — {target: model_name}
        actuals      — {target: (y_test, y_pred)}
        total_samples — int
    """
    df = generate_training_dataset(tle_data, hours=hours, step_minutes=step_minutes)

    if len(df) < 20:
        raise ValueError(
            f"Too few training samples ({len(df)}). "
            "Try increasing hours or using a fresher TLE."
        )

    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values  # shape: (N, 4)

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Define models ─────────────────────────────────────────────────────
    candidate_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        )
    }
    if XGBOOST_AVAILABLE:
        candidate_models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            eval_metric="rmse",
        )

    # ── Train and evaluate each model on each target ──────────────────────
    all_metrics = {target: {} for target in TARGET_COLS}
    all_predictions = {target: {} for target in TARGET_COLS}

    for model_name, base_model in candidate_models.items():
        # Wrap in MultiOutputRegressor only if model doesn't natively support it
        model = MultiOutputRegressor(base_model, n_jobs=-1)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        for i, target in enumerate(TARGET_COLS):
            yt = y_test[:, i]
            yp = y_pred[:, i]
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(math.sqrt(mean_squared_error(yt, yp)))
            r2 = float(r2_score(yt, yp))
            all_metrics[target][model_name] = {"mae": mae, "rmse": rmse, "r2": r2}
            all_predictions[target][model_name] = (yt.tolist(), yp.tolist())

    # ── Select best model per target (highest R²) ─────────────────────────
    best_model_name = {}
    for target in TARGET_COLS:
        best = max(all_metrics[target], key=lambda m: all_metrics[target][m]["r2"])
        best_model_name[target] = best

    # ── Retrain the overall "best" model (majority vote) as a single model ─
    from collections import Counter
    votes = Counter(best_model_name.values())
    overall_best_name = votes.most_common(1)[0][0]

    best_base = candidate_models[overall_best_name]
    best_multi = MultiOutputRegressor(best_base, n_jobs=-1)
    best_multi.fit(X_train_sc, y_train)

    # Persist
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {"model": best_multi, "model_name": overall_best_name},
        MODELS_PATH,
    )
    joblib.dump(all_metrics, METRICS_PATH)
    joblib.dump(
        {target: all_predictions[target][best_model_name[target]]
         for target in TARGET_COLS},
        ACTUALS_PATH,
    )

    return {
        "metrics": all_metrics,
        "best_model": best_model_name,
        "overall_best": overall_best_name,
        "actuals": {
            target: all_predictions[target][best_model_name[target]]
            for target in TARGET_COLS
        },
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_position(tle_data: dict, target_dt) -> dict:
    """
    Predict satellite position at a future datetime using the trained model.

    Parameters
    ----------
    tle_data  : dict     — TLE dict
    target_dt : datetime — UTC datetime for prediction

    Returns
    -------
    dict: {latitude, longitude, altitude_km, speed_km_s}
    Raises RuntimeError if model has not been trained yet.
    """
    if not os.path.exists(MODELS_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            "No trained model found. Please train the model first."
        )

    scaler = joblib.load(SCALER_PATH)
    model_bundle = joblib.load(MODELS_PATH)
    model = model_bundle["model"]

    X_pred = build_prediction_features(tle_data, target_dt)
    X_scaled = scaler.transform(X_pred[FEATURE_COLS].values)
    y_pred = model.predict(X_scaled)[0]

    return {
        "latitude": round(float(y_pred[0]), 4),
        "longitude": round(float(y_pred[1]), 4),
        "altitude_km": round(float(y_pred[2]), 3),
        "speed_km_s": round(float(y_pred[3]), 4),
        "model_name": model_bundle.get("model_name", "Unknown"),
    }


# ── Load persisted results ────────────────────────────────────────────────────

def load_metrics() -> dict:
    """Load persisted evaluation metrics. Returns None if not available."""
    if os.path.exists(METRICS_PATH):
        try:
            return joblib.load(METRICS_PATH)
        except Exception:
            pass
    return None


def load_actuals() -> dict:
    """Load persisted actual vs predicted data. Returns None if not available."""
    if os.path.exists(ACTUALS_PATH):
        try:
            return joblib.load(ACTUALS_PATH)
        except Exception:
            pass
    return None


def load_model_name() -> str:
    """Return the name of the currently persisted best model."""
    if os.path.exists(MODELS_PATH):
        try:
            bundle = joblib.load(MODELS_PATH)
            return bundle.get("model_name", "Unknown")
        except Exception:
            pass
    return "Not trained"


def is_model_trained() -> bool:
    """Return True if a trained model exists on disk."""
    return os.path.exists(MODELS_PATH) and os.path.exists(SCALER_PATH)
