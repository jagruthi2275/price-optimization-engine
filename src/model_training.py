"""
Model training module for the Price Optimization Engine.
Trains LinearRegression, RandomForest, and XGBoost regressors.
Selects the best model by R² on the test set and saves it with Joblib.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.joblib")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def get_model_definitions() -> dict:
    """Return the three candidate models with their configurations."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),
    }


def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Train all models, compute metrics, pick the best by R², and save it.

    Returns a summary dict with per-model metrics and the best model name.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_model_definitions()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values if hasattr(y_test, "values") else y_test, y_pred)
        results[name] = {
            "model": model,
            "metrics": metrics,
        }
        print(f"  {name}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R²={metrics['R2']}")

    best_name = max(results, key=lambda k: results[k]["metrics"]["R2"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}  (R²={results[best_name]['metrics']['R2']})")

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Saved best model to {BEST_MODEL_PATH}")

    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = dict(zip(feature_names, importances.round(6).tolist()))

    summary = {
        "best_model_name": best_name,
        "feature_importance": feature_importance,
        "all_metrics": {k: v["metrics"] for k, v in results.items()},
    }
    joblib.dump(summary, METRICS_PATH)

    return summary


def load_best_model():
    """Load the best saved model from disk."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. Run the training pipeline first."
        )
    return joblib.load(BEST_MODEL_PATH)


def load_metrics() -> dict:
    """Load saved model comparison metrics."""
    if not os.path.exists(METRICS_PATH):
        return {}
    return joblib.load(METRICS_PATH)


def run_training_pipeline() -> dict:
    """Execute the full train pipeline (preprocessing → training → save)."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import preprocess_pipeline

    print("=== Starting Training Pipeline ===")
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess_pipeline(save=True)
    summary = train_models(X_train, X_test, y_train, y_test, feature_names)
    return summary


if __name__ == "__main__":
    summary = run_training_pipeline()
    print("\n=== Training Summary ===")
    print(f"Best model: {summary['best_model_name']}")
    print("\nAll model metrics:")
    for model, metrics in summary["all_metrics"].items():
        print(f"  {model}: {metrics}")
    if summary["feature_importance"]:
        print("\nTop feature importances:")
        sorted_fi = sorted(summary["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_fi[:5]:
            print(f"  {feat}: {imp:.4f}")
