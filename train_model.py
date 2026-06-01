"""
grAIn ML Model Training
========================
Trains Random Forest and XGBoost regressors for rice grain drying prediction,
then evaluates a weighted ensemble for 30-minute moisture forecasting.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
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

TARGET_MOISTURE_30MIN = "moisture_30min"
TARGET_TIME_TO_TARGET = "time_to_target"


def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate training data."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples from {data_path}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Targets: [{TARGET_MOISTURE_30MIN}, {TARGET_TIME_TO_TARGET}]")
    return df


def regression_metrics(y_true, y_pred, include_cv=None) -> dict:
    """Compute common regression metrics."""
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }
    if include_cv is not None:
        metrics["cv_r2_mean"] = float(include_cv.mean())
        metrics["cv_r2_std"] = float(include_cv.std())
    return metrics


def print_metrics(name: str, train_metrics: dict, test_metrics: dict, importance: dict):
    """Print model metrics in the existing training-log style."""
    print(f"\n  Train RMSE: {train_metrics['rmse']:.4f} %")
    print(f"  Train MAE:  {train_metrics['mae']:.4f} %")
    print(f"  Train R2:   {train_metrics['r2']:.4f}")
    print(f"\n  Test RMSE:  {test_metrics['rmse']:.4f} %")
    print(f"  Test MAE:   {test_metrics['mae']:.4f} %")
    print(f"  Test R2:    {test_metrics['r2']:.4f}")
    print(f"  Test MAPE:  {test_metrics['mape']:.2f} %")
    print(f"\n  CV R2 (5-fold): {test_metrics['cv_r2_mean']:.4f} +/- {test_metrics['cv_r2_std']:.4f}")
    print(f"\n  {name} Feature Importance:")
    for feat, imp in importance.items():
        print(f"    {feat:30s} {imp:.4f}")


def sorted_importance(model) -> dict:
    """Map model feature importances to feature names."""
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))


def train_moisture_model(X_train, X_test, y_train, y_test):
    """Train Random Forest for 30-minute moisture prediction."""
    print("\n" + "=" * 60)
    print("Training Model 1A: Random Forest Moisture Prediction")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    metrics = {
        "train": regression_metrics(y_train, train_pred),
        "test": regression_metrics(y_test, test_pred, cv_scores),
    }
    importance = sorted_importance(model)
    print_metrics("Random Forest", metrics["train"], metrics["test"], importance)

    return model, metrics, importance


def train_xgboost_moisture_model(X_train, X_test, y_train, y_test):
    """Train XGBoost for 30-minute moisture prediction."""
    print("\n" + "=" * 60)
    print("Training Model 1B: XGBoost Moisture Prediction")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    metrics = {
        "train": regression_metrics(y_train, train_pred),
        "test": regression_metrics(y_test, test_pred, cv_scores),
    }
    importance = sorted_importance(model)
    print_metrics("XGBoost", metrics["train"], metrics["test"], importance)

    return model, metrics, importance


def evaluate_ensemble(rf_model, xgb_model, X_test, y_moisture_test):
    """Evaluate dynamic R2-weighted RF + XGBoost ensemble."""
    print("\n" + "=" * 60)
    print("Evaluating Weighted Ensemble")
    print("=" * 60)

    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    rf_r2 = r2_score(y_moisture_test, rf_pred)
    xgb_r2 = r2_score(y_moisture_test, xgb_pred)
    total = rf_r2 + xgb_r2
    if total <= 0:
        rf_weight = 0.5
        xgb_weight = 0.5
    else:
        rf_weight = rf_r2 / total
        xgb_weight = xgb_r2 / total

    ensemble_pred = (rf_weight * rf_pred) + (xgb_weight * xgb_pred)
    ensemble_metrics = regression_metrics(y_moisture_test, ensemble_pred)
    rf_metrics = regression_metrics(y_moisture_test, rf_pred)
    xgb_metrics = regression_metrics(y_moisture_test, xgb_pred)

    print(f"\n{'Model':<14} | {'R2':<6} | {'RMSE':<7} | {'MAE':<7}")
    print("-" * 45)
    print(f"{'RF':<14} | {rf_metrics['r2']:<6.4f} | {rf_metrics['rmse']:<7.4f} | {rf_metrics['mae']:<7.4f}")
    print(f"{'XGBoost':<14} | {xgb_metrics['r2']:<6.4f} | {xgb_metrics['rmse']:<7.4f} | {xgb_metrics['mae']:<7.4f}")
    print(f"{'Ensemble':<14} | {ensemble_metrics['r2']:<6.4f} | {ensemble_metrics['rmse']:<7.4f} | {ensemble_metrics['mae']:<7.4f}")
    print(f"\nEnsemble weights: RF={rf_weight:.4f}, XGB={xgb_weight:.4f}")

    return float(rf_weight), float(xgb_weight), ensemble_metrics


def train_time_model(X_train, X_test, y_train, y_test):
    """Train Random Forest for time-to-target prediction with log transform."""
    print("\n" + "=" * 60)
    print("Training Model 2: Time-to-Target Prediction")
    print("=" * 60)

    y_train_log = np.log1p(y_train)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train_log)
    train_pred = np.maximum(np.expm1(model.predict(X_train)), 0)
    test_pred = np.maximum(np.expm1(model.predict(X_test)), 0)

    metrics = {
        "train": {
            "rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "mae": float(mean_absolute_error(y_train, train_pred)),
            "r2": float(r2_score(y_train, train_pred)),
        },
        "test": {
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "mae": float(mean_absolute_error(y_test, test_pred)),
            "r2": float(r2_score(y_test, test_pred)),
        },
    }

    nonzero_mask = y_test > 1.0
    if nonzero_mask.sum() > 0:
        metrics["test"]["mape"] = float(
            mean_absolute_percentage_error(y_test[nonzero_mask], test_pred[nonzero_mask]) * 100
        )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"] = float(cv_scores.std())
    importance = sorted_importance(model)

    print(f"\n  Train RMSE: {metrics['train']['rmse']:.2f} min")
    print(f"  Train MAE:  {metrics['train']['mae']:.2f} min")
    print(f"  Train R2:   {metrics['train']['r2']:.4f}")
    print(f"\n  Test RMSE:  {metrics['test']['rmse']:.2f} min")
    print(f"  Test MAE:   {metrics['test']['mae']:.2f} min")
    print(f"  Test R2:    {metrics['test']['r2']:.4f}")
    if "mape" in metrics["test"]:
        print(f"  Test MAPE:  {metrics['test']['mape']:.2f} %")
    print(f"\n  CV R2 (5-fold): {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")

    return model, metrics, importance


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "training_data.csv")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)

    X = df[FEATURE_COLUMNS].values
    y_moisture = df[TARGET_MOISTURE_30MIN].values
    y_time = df[TARGET_TIME_TO_TARGET].values

    session_ids = df["session_id"].unique()
    train_sessions, test_sessions = train_test_split(
        session_ids, test_size=0.2, random_state=42
    )

    train_mask = df["session_id"].isin(train_sessions)
    test_mask = df["session_id"].isin(test_sessions)

    X_train, X_test = X[train_mask], X[test_mask]
    y_moisture_train, y_moisture_test = y_moisture[train_mask], y_moisture[test_mask]
    y_time_train, y_time_test = y_time[train_mask], y_time[test_mask]

    print(f"\nTrain set: {len(X_train):,} samples ({len(train_sessions)} sessions)")
    print(f"Test set:  {len(X_test):,} samples ({len(test_sessions)} sessions)")

    rf_model, rf_metrics, rf_importance = train_moisture_model(
        X_train, X_test, y_moisture_train, y_moisture_test
    )
    xgb_model, xgb_metrics, xgb_importance = train_xgboost_moisture_model(
        X_train, X_test, y_moisture_train, y_moisture_test
    )
    rf_weight, xgb_weight, ensemble_metrics = evaluate_ensemble(
        rf_model, xgb_model, X_test, y_moisture_test
    )
    time_model, time_metrics, time_importance = train_time_model(
        X_train, X_test, y_time_train, y_time_test
    )

    moisture_rf_path = os.path.join(model_dir, "moisture_predictor_rf.joblib")
    moisture_xgb_path = os.path.join(model_dir, "moisture_predictor_xgb.joblib")
    moisture_legacy_path = os.path.join(model_dir, "moisture_predictor.joblib")
    time_model_path = os.path.join(model_dir, "time_predictor.joblib")

    joblib.dump(rf_model, moisture_rf_path)
    joblib.dump(xgb_model, moisture_xgb_path)
    joblib.dump(rf_model, moisture_legacy_path)
    joblib.dump(time_model, time_model_path)

    print(f"\n\nModels saved:")
    print(f"  {moisture_rf_path}")
    print(f"  {moisture_xgb_path}")
    print(f"  {moisture_legacy_path}")
    print(f"  {time_model_path}")

    metadata = {
        "version": "2.0.0",
        "trained_at": datetime.now().isoformat(),
        "framework": "scikit-learn + xgboost",
        "algorithms": {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
            },
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "tree_method": "hist",
            },
        },
        "ensemble": {
            "method": "weighted_average",
            "rf_weight": rf_weight,
            "xgb_weight": xgb_weight,
        },
        "features": FEATURE_COLUMNS,
        "targets": {
            "moisture_30min": "Predicted moisture content 30 minutes ahead (% wet basis)",
            "time_to_target": "Estimated minutes until target moisture (14%) is reached",
        },
        "metrics": {
            "random_forest": rf_metrics["test"],
            "xgboost": xgb_metrics["test"],
            "ensemble": ensemble_metrics,
            "time_prediction": time_metrics,
        },
        "feature_importance": {
            "random_forest": rf_importance,
            "xgboost": xgb_importance,
            "time_model": time_importance,
        },
        "training_data": {
            "total_samples": len(df),
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "num_sessions": 50,
            "grain_type": "rice",
            "data_source": "synthetic (Page's equation)",
        },
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {metadata_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Model Performance Summary")
    print("=" * 60)
    print(f"\n{'Model':<30} {'R2':<8} {'RMSE':<10} {'MAE':<10}")
    print("-" * 58)
    print(f"{'Random Forest':<30} {rf_metrics['test']['r2']:<8.4f} {rf_metrics['test']['rmse']:<10.4f} {rf_metrics['test']['mae']:<10.4f}")
    print(f"{'XGBoost':<30} {xgb_metrics['test']['r2']:<8.4f} {xgb_metrics['test']['rmse']:<10.4f} {xgb_metrics['test']['mae']:<10.4f}")
    print(f"{'Ensemble (RF+XGBoost)':<30} {ensemble_metrics['r2']:<8.4f} {ensemble_metrics['rmse']:<10.4f} {ensemble_metrics['mae']:<10.4f}")
    print(f"{'Time-to-Target':<30} {time_metrics['test']['r2']:<8.4f} {time_metrics['test']['rmse']:<10.2f} {time_metrics['test']['mae']:<10.2f}")


if __name__ == "__main__":
    main()
