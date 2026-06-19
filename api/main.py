"""
FastAPI backend for the Price Optimization Engine.

Endpoints:
  POST /predict-demand   — predict demand for a given price and context
  POST /optimize-price   — recommend revenue-maximizing price
  POST /retrain          — retrain all models (optional CSV upload)
  GET  /health           — liveness + model status probe
  GET  /model-info       — model metadata, version, and features
  GET  /elasticity       — price elasticity curve + recommendations
  GET  /simulate         — full price simulation table
  GET  /retrain-status   — current retraining job state
  GET  /retrain-history  — log of the last 50 retraining runs
"""

import io
import os
import sys
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("price_engine.api")

# ── Global state ───────────────────────────────────────────────────────────────
_model = None
_metrics_summary = None
_retrain_lock = threading.Lock()
_retrain_status: dict = {
    "state": "idle", "started_at": None, "finished_at": None, "error": None
}

# Env flags
AUTO_TRAIN = os.environ.get("AUTO_TRAIN_IF_MISSING", "true").lower() == "true"
RETRAIN_HISTORY_PATH = os.path.join(BASE_DIR, "models", "retrain_history.json")
REQUIRED_COLUMNS = {
    "price", "competitor_price", "promotion", "inventory_level",
    "season", "holiday", "weekday", "demand",
}


# ── Enums ──────────────────────────────────────────────────────────────────────
class Season(str, Enum):
    Spring = "Spring"
    Summer = "Summer"
    Fall = "Fall"
    Winter = "Winter"


# ── Model loading helpers ──────────────────────────────────────────────────────
def get_model():
    """
    Load the best trained model from disk.

    - If AUTO_TRAIN_IF_MISSING=true (local dev default): trains automatically when missing.
    - If AUTO_TRAIN_IF_MISSING=false (Docker/production): raises HTTP 503 with instructions.
    """
    global _model
    if _model is not None:
        return _model

    from src.model_training import load_best_model, BEST_MODEL_PATH

    if not os.path.exists(BEST_MODEL_PATH):
        if AUTO_TRAIN:
            logger.warning("No trained model found — running training pipeline automatically.")
            from src.model_training import run_training_pipeline
            run_training_pipeline()
        else:
            logger.error(
                "Model file not found at %s. "
                "Pre-train the model before starting in production mode: "
                "run `bash run.sh` locally, then mount the models/ directory.",
                BEST_MODEL_PATH,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model not loaded. Pre-train the model by running `bash run.sh` locally "
                    "and mounting the models/ directory into the container. "
                    f"Expected path: {BEST_MODEL_PATH}"
                ),
            )

    _model = load_best_model()
    logger.info("Model loaded: %s", BEST_MODEL_PATH)
    return _model


def get_metrics():
    global _metrics_summary
    if _metrics_summary is None:
        from src.model_training import load_metrics
        _metrics_summary = load_metrics()
    return _metrics_summary


# ── Retrain helpers ────────────────────────────────────────────────────────────
def _append_retrain_history(entry: dict):
    os.makedirs(os.path.dirname(RETRAIN_HISTORY_PATH), exist_ok=True)
    history: list = []
    if os.path.exists(RETRAIN_HISTORY_PATH):
        try:
            with open(RETRAIN_HISTORY_PATH) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    with open(RETRAIN_HISTORY_PATH, "w") as f:
        json.dump(history[-50:], f, indent=2)


def _do_retrain(csv_bytes: Optional[bytes], csv_filename: Optional[str]):
    """Background worker — runs full pipeline and reloads globals."""
    global _model, _metrics_summary, _retrain_status

    _retrain_status = {
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
    }
    metrics_before = (get_metrics() or {}).get("all_metrics", {})

    try:
        if csv_bytes:
            df_new = pd.read_csv(io.BytesIO(csv_bytes))
            missing = REQUIRED_COLUMNS - set(df_new.columns)
            if missing:
                raise ValueError(f"Uploaded CSV is missing required columns: {missing}")
            raw_dir = os.path.join(BASE_DIR, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            out_path = os.path.join(raw_dir, "retail_sales.csv")
            df_new.to_csv(out_path, index=False)
            logger.info("Saved uploaded CSV (%d rows) to %s", len(df_new), out_path)

        from src.model_training import run_training_pipeline, load_best_model, load_metrics
        summary = run_training_pipeline()
        _model = load_best_model()
        _metrics_summary = load_metrics()

        _append_retrain_history({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": csv_filename or "existing_data",
            "best_model": summary["best_model_name"],
            "metrics_before": metrics_before,
            "metrics_after": summary["all_metrics"],
        })
        _retrain_status = {
            "state": "done",
            "started_at": _retrain_status["started_at"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "best_model": summary["best_model_name"],
            "metrics_after": summary["all_metrics"],
        }
        logger.info("Retraining complete. Best model: %s", summary["best_model_name"])

    except Exception as exc:
        logger.exception("Retraining failed")
        _retrain_status = {
            "state": "error",
            "started_at": _retrain_status["started_at"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }


# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Price Optimization Engine API…")
    try:
        get_model()
        get_metrics()
        logger.info("Model ready.")
    except HTTPException as exc:
        logger.warning("Startup warning: %s", exc.detail)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Price Optimization Engine API",
    description=(
        "ML-powered demand forecasting and revenue optimization for retail pricing. "
        "Trains Linear Regression, Random Forest, and XGBoost; auto-selects the best model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class DemandRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product selling price (must be > 0)", example=100.0)
    stock: int = Field(500, ge=0, description="Current inventory level (≥ 0)", example=500)
    promotion: int = Field(0, ge=0, le=1, description="Promotion active: 0 or 1", example=0)
    competitor_price: float = Field(..., gt=0, description="Competitor's price (must be > 0)", example=95.0)
    season: Season = Field(Season.Summer, description="Season: Spring | Summer | Fall | Winter")
    holiday: int = Field(0, ge=0, le=1, description="Holiday indicator: 0 or 1", example=0)
    weekday: int = Field(2, ge=0, le=6, description="Day of week: 0=Mon … 6=Sun", example=2)


class DemandResponse(BaseModel):
    predicted_demand: float
    price: float
    season: str
    promotion: bool


class OptimizeRequest(BaseModel):
    price: float = Field(..., gt=0, example=100.0)
    stock: int = Field(500, ge=0, example=500)
    promotion: int = Field(0, ge=0, le=1, example=0)
    competitor_price: float = Field(..., gt=0, example=95.0)
    season: Season = Field(Season.Summer)
    holiday: int = Field(0, ge=0, le=1, example=0)
    weekday: int = Field(2, ge=0, le=6, example=2)
    price_range_lower: float = Field(0.5, gt=0, description="Lower bound as fraction of base price")
    price_range_upper: float = Field(1.5, gt=0, description="Upper bound as fraction of base price")
    n_simulations: int = Field(60, ge=10, le=200, description="Number of price points to evaluate")


class OptimizeResponse(BaseModel):
    recommended_price: float
    expected_revenue: float
    predicted_demand: float
    revenue_uplift_pct: float
    base_price: float
    base_revenue: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """
    Liveness probe. Returns model status so orchestrators can detect unhealthy containers.
    """
    from src.model_training import BEST_MODEL_PATH
    model_loaded = os.path.exists(BEST_MODEL_PATH) and _model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "service": "price-optimization-engine",
        "version": "1.0.0",
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """
    Structured metadata about the currently loaded model.
    """
    metrics = get_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="Model metrics not available. Run training first.")

    from src.model_training import BEST_MODEL_PATH, METRICS_PATH
    from src.data_preprocessing import FEATURE_COLS

    trained_date = None
    if os.path.exists(METRICS_PATH):
        trained_date = datetime.fromtimestamp(
            os.path.getmtime(METRICS_PATH), tz=timezone.utc
        ).isoformat()

    return {
        "model_type": metrics.get("best_model_name", "Unknown"),
        "version": "1.0",
        "features": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "trained_date": trained_date,
        "model_path": BEST_MODEL_PATH,
        "all_metrics": metrics.get("all_metrics", {}),
        "feature_importance": metrics.get("feature_importance", {}),
        "data_note": (
            "Metrics are from the most recent training run. "
            "For production use, train on a validated real-world retail dataset "
            "(e.g. UCI Online Retail, Dunnhumby)."
        ),
    }


@app.post("/predict-demand", response_model=DemandResponse, tags=["Prediction"])
def predict_demand(req: DemandRequest):
    """Predict units sold for a given price and market context."""
    try:
        model = get_model()
        from src.data_preprocessing import preprocess_single
        X = preprocess_single(
            price=req.price,
            competitor_price=req.competitor_price,
            promotion=req.promotion,
            inventory_level=req.stock,
            season=req.season.value,
            holiday=req.holiday,
            weekday=req.weekday,
        )
        demand = max(float(model.predict(X)[0]), 0)
        logger.info("Predicted demand=%.1f for price=%.2f season=%s", demand, req.price, req.season)
        return DemandResponse(
            predicted_demand=round(demand, 1),
            price=req.price,
            season=req.season.value,
            promotion=bool(req.promotion),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/optimize-price", response_model=OptimizeResponse, tags=["Optimization"])
def optimize_price(req: OptimizeRequest):
    """Find the revenue-maximizing price by simulating across a configurable price range."""
    try:
        model = get_model()
        from src.revenue_optimization import run_optimization
        result = run_optimization(
            base_price=req.price,
            competitor_price=req.competitor_price,
            promotion=req.promotion,
            inventory_level=req.stock,
            model=model,
            season=req.season.value,
            holiday=req.holiday,
            weekday=req.weekday,
            n_simulations=req.n_simulations,
        )
        logger.info(
            "Optimized price: base=%.2f → optimal=%.2f (uplift=%.1f%%)",
            req.price, result["recommended_price"], result["revenue_uplift_pct"],
        )
        return OptimizeResponse(
            recommended_price=result["recommended_price"],
            expected_revenue=result["expected_revenue"],
            predicted_demand=result["predicted_demand"],
            revenue_uplift_pct=result["revenue_uplift_pct"],
            base_price=result["base_price"],
            base_revenue=result["base_revenue"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/elasticity", tags=["Analysis"])
def get_elasticity(
    price: float = Query(..., gt=0, description="Current product price"),
    competitor_price: float = Query(..., gt=0, description="Competitor price"),
    promotion: int = Query(0, ge=0, le=1),
    stock: int = Query(500, ge=0),
    season: Season = Query(Season.Summer),
    holiday: int = Query(0, ge=0, le=1),
    weekday: int = Query(2, ge=0, le=6),
):
    """Return price elasticity curve and business recommendations."""
    try:
        model = get_model()
        from src.price_elasticity import generate_elasticity_report
        return generate_elasticity_report(
            base_price=price,
            competitor_price=competitor_price,
            promotion=promotion,
            inventory_level=stock,
            model=model,
            season=season.value,
            holiday=holiday,
            weekday=weekday,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Elasticity calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/simulate", tags=["Analysis"])
def simulate(
    price: float = Query(..., gt=0),
    competitor_price: float = Query(..., gt=0),
    promotion: int = Query(0, ge=0, le=1),
    stock: int = Query(500, ge=0),
    season: Season = Query(Season.Summer),
    holiday: int = Query(0, ge=0, le=1),
    weekday: int = Query(2, ge=0, le=6),
    n: int = Query(60, ge=10, le=200),
):
    """Return full price simulation table."""
    try:
        model = get_model()
        from src.revenue_optimization import run_optimization
        return run_optimization(
            base_price=price,
            competitor_price=competitor_price,
            promotion=promotion,
            inventory_level=stock,
            model=model,
            season=season.value,
            holiday=holiday,
            weekday=weekday,
            n_simulations=n,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrain", tags=["Model"])
async def retrain(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="Optional CSV with retail sales data"),
):
    """
    Retrain all models on new or existing data. Runs in the background.

    Without a file: retrains on the existing dataset on disk.
    With a file: replaces the raw dataset, then retrains.

    Required CSV columns: price, competitor_price, promotion, inventory_level,
    season, holiday, weekday, demand.

    Poll GET /retrain-status for progress.
    """
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining job is already running.")

    csv_bytes: Optional[bytes] = None
    csv_filename: Optional[str] = None
    try:
        if file and file.filename:
            if not file.filename.endswith(".csv"):
                _retrain_lock.release()
                raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
            csv_bytes = await file.read()
            csv_filename = file.filename
    except HTTPException:
        raise
    except Exception as exc:
        _retrain_lock.release()
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    def _run_and_release(csv_bytes, csv_filename):
        try:
            _do_retrain(csv_bytes, csv_filename)
        finally:
            _retrain_lock.release()

    background_tasks.add_task(_run_and_release, csv_bytes, csv_filename)
    return {
        "status": "accepted",
        "message": "Retraining started in the background.",
        "data_source": csv_filename or "existing retail_sales.csv",
        "poll": "GET /retrain-status",
    }


@app.get("/retrain-status", tags=["Model"])
def retrain_status():
    """Current state of the retraining job: idle | running | done | error."""
    return _retrain_status


@app.get("/retrain-history", tags=["Model"])
def retrain_history():
    """Log of the last 50 retraining runs (newest first)."""
    if not os.path.exists(RETRAIN_HISTORY_PATH):
        return {"history": [], "total": 0}
    try:
        with open(RETRAIN_HISTORY_PATH) as f:
            history = json.load(f)
        return {"history": list(reversed(history)), "total": len(history)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
