#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Price Optimization Engine — Local Development Startup Script
#
# Generates synthetic data, trains all ML models (first run only), then
# starts FastAPI and the Streamlit dashboard side-by-side.
#
# Usage:
#   bash run.sh                    # uses defaults
#   STREAMLIT_PORT=8501 bash run.sh  # override Streamlit port
#
# Ports (standard):
#   FastAPI   → http://localhost:8000   (API docs at /docs)
#   Streamlit → http://localhost:8501   (override via $STREAMLIT_PORT)
#
# For Docker multi-container setup, use docker-compose.yml instead.
# ─────────────────────────────────────────────────────────────────────────────

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export API_PORT="${API_PORT:-8000}"
export STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
export AUTO_TRAIN_IF_MISSING=true

echo "============================================"
echo "  Price Optimization Engine"
echo "============================================"

# Step 1: Generate raw data if not present
RAW_DATA="$ROOT_DIR/data/raw/retail_sales.csv"
LEGACY_DATA="$ROOT_DIR/data/retail_sales.csv"

mkdir -p "$ROOT_DIR/data/raw" "$ROOT_DIR/data/processed"

if [ ! -f "$RAW_DATA" ] && [ ! -f "$LEGACY_DATA" ]; then
    echo "[1/3] Generating synthetic retail dataset..."
    python data/generate_data.py
    # Copy to new raw/ location
    [ -f "$LEGACY_DATA" ] && cp "$LEGACY_DATA" "$RAW_DATA"
else
    echo "[1/3] Dataset found — skipping generation."
fi

# Step 2: Train models if not present
if [ ! -f "models/best_model.joblib" ]; then
    echo "[2/3] Training ML models (Linear Regression, Random Forest, XGBoost)..."
    python -c "
import sys; sys.path.insert(0, '.')
from src.model_training import run_training_pipeline
summary = run_training_pipeline()
print(f'Best model: {summary[\"best_model_name\"]}')
for name, m in summary['all_metrics'].items():
    print(f'  {name}: R²={m[\"R2\"]}, MAE={m[\"MAE\"]}, RMSE={m[\"RMSE\"]}')
"
else
    echo "[2/3] Trained model found — skipping training."
fi

# Step 3: Start services
echo "[3/3] Starting services..."
echo "  FastAPI    → http://localhost:${API_PORT}/docs"
echo "  Streamlit  → http://localhost:${STREAMLIT_PORT}"

uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --log-level info &
FASTAPI_PID=$!

sleep 1

streamlit run dashboard/app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#63b3ed" \
    --theme.backgroundColor "#0f1117" \
    --theme.secondaryBackgroundColor "#1a1f2e" \
    --theme.textColor "#e2e8f0"

wait $FASTAPI_PID
