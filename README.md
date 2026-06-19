# 💰 Price Optimization Engine

> ML-powered demand forecasting and revenue optimization for retail pricing

[![CI](https://github.com/your-username/price-optimization-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/price-optimization-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.137-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange.svg)](https://xgboost.ai)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

---

## 📌 Project Overview

The **Price Optimization Engine** is a production-grade ML application that:

1. **Forecasts product demand** using three competing regression models — Linear Regression, Random Forest, and XGBoost — with automatic best-model selection by R² score
2. **Recommends the optimal selling price** that maximizes expected revenue via configurable price simulation
3. **Models price elasticity of demand** to classify customer price sensitivity and generate business recommendations
4. **Explains predictions** using SHAP (SHapley Additive exPlanations) for full transparency
5. **Supports on-demand retraining** via REST API — upload a new CSV and hot-swap the model without downtime

> **Data note:** This project ships with a synthetic retail dataset for development and portfolio demonstration purposes. For production use, replace with a validated real-world dataset such as the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) or [Dunnhumby "The Complete Journey"](https://www.dunnhumby.com/source-files/). Accuracy metrics reflect performance on the current training data only.

---

## 🏗️ Architecture

```
User
 ↓
Streamlit Dashboard (port 8501)
 ↓  HTTP
FastAPI REST API (port 8000)
 ↓
ML Pipeline
 ├── data_preprocessing.py  — feature engineering + scaling
 ├── model_training.py      — train / compare / select best model
 ├── price_elasticity.py    — elasticity formula + curve builder
 └── revenue_optimization.py — price simulation engine
 ↓
XGBoost / Random Forest / Linear Regression
 ↓
Prediction Response
```

---

## 📦 Project Structure

```
price-optimization-engine/
├── data/
│   ├── raw/                     # Raw input data (supply your own real dataset here)
│   ├── processed/               # Cleaned / feature-engineered data
│   └── generate_data.py        # Synthetic dataset generator (dev only)
├── models/
│   ├── best_model.joblib        # Best trained model (auto-generated)
│   ├── scaler.joblib            # Feature scaler
│   ├── model_metrics.joblib     # Performance comparison
│   └── retrain_history.json    # Log of retraining runs
├── src/
│   ├── data_preprocessing.py   # Loading, cleaning, feature engineering
│   ├── model_training.py       # Train & compare 3 ML models
│   ├── price_elasticity.py     # Elasticity formula & curve builder
│   └── revenue_optimization.py # Price simulation & optimization
├── api/
│   └── main.py                 # FastAPI REST endpoints
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard (7 tabs)
├── tests/
│   └── test_pipeline.py        # Unit + integration tests
├── .env.example                # Environment variable template
├── docker-compose.yml          # Multi-container Docker setup
├── Dockerfile                  # Production image (code only, no training)
├── requirements.txt            # Pinned Python dependencies
└── run.sh                      # Local development startup script
```

---

## 🚀 Quick Start — Local Development

### Prerequisites
- Python 3.11+
- ~500 MB disk (ML dependencies)

```bash
# 1. Clone and enter the project
git clone https://github.com/your-username/price-optimization-engine.git
cd price-optimization-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start everything (generates data + trains models on first run)
bash run.sh
```

| Service | URL |
|---|---|
| 📊 Streamlit Dashboard | http://localhost:8501 |
| 🔌 FastAPI REST API | http://localhost:8000 |
| 📖 Interactive API Docs | http://localhost:8000/docs |

---

## 🐳 Docker Setup

The application is split into two independent containers:
- **`api`** — FastAPI backend
- **`dashboard`** — Streamlit frontend

> **Important:** Train your models locally before running Docker. Docker containers load pre-trained model files from a volume mount; they do **not** train during the build.

```bash
# Step 1: Train models locally (one-time)
bash run.sh
# Ctrl+C after training completes (or wait for it to finish)

# Step 2: Build and start all containers
docker compose up --build

# Step 3: Verify
curl http://localhost:8000/health        # {"status":"healthy","model_loaded":true}
open http://localhost:8000/docs          # FastAPI Swagger UI
open http://localhost:8501               # Streamlit dashboard
```

### docker-compose services

```yaml
services:
  api:       # FastAPI  → localhost:8000
  dashboard: # Streamlit → localhost:8501
```

The `dashboard` service waits for the `api` service to pass its health check before starting.

### Build a single image

```bash
docker build -t price-optimization-engine .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  price-optimization-engine
```

---

## 🔌 API Endpoints

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "service": "price-optimization-engine",
  "version": "1.0.0"
}
```

### `POST /predict-demand`

```bash
curl -X POST http://localhost:8000/predict-demand \
  -H "Content-Type: application/json" \
  -d '{
    "price": 100.0,
    "stock": 500,
    "promotion": 1,
    "competitor_price": 95.0,
    "season": "Summer",
    "holiday": 0,
    "weekday": 2
  }'
```

```json
{"predicted_demand": 1247.0, "price": 100.0, "season": "Summer", "promotion": true}
```

Valid `season` values: `Spring` | `Summer` | `Fall` | `Winter`

### `POST /optimize-price`

```bash
curl -X POST http://localhost:8000/optimize-price \
  -H "Content-Type: application/json" \
  -d '{"price": 100.0, "stock": 500, "promotion": 0, "competitor_price": 95.0}'
```

```json
{
  "recommended_price": 105.0,
  "expected_revenue": 132750.0,
  "predicted_demand": 1264.3,
  "revenue_uplift_pct": 5.2,
  "base_price": 100.0,
  "base_revenue": 126200.0
}
```

### `GET /model-info`

```json
{
  "model_type": "XGBoost",
  "version": "1.0",
  "features": ["price", "competitor_price", ...],
  "feature_count": 10,
  "trained_date": "2024-06-18T16:30:00+00:00",
  "all_metrics": {
    "XGBoost": {"MAE": 126.1, "RMSE": 187.4, "R2": 0.889},
    "Random Forest": {"MAE": 126.2, "RMSE": 189.1, "R2": 0.887},
    "Linear Regression": {"MAE": 176.9, "RMSE": 252.8, "R2": 0.798}
  }
}
```

### `POST /retrain` — on-demand retraining

```bash
# Retrain on existing data
curl -X POST http://localhost:8000/retrain

# Retrain with a new CSV
curl -X POST http://localhost:8000/retrain \
  -F "file=@/path/to/new_sales_data.csv"

# Poll status
curl http://localhost:8000/retrain-status

# View history
curl http://localhost:8000/retrain-history
```

### `GET /elasticity`

```bash
curl "http://localhost:8000/elasticity?price=100&competitor_price=95&promotion=0&stock=500"
```

### Input Validation

All endpoints validate inputs with **Pydantic v2**. Invalid inputs return HTTP `422 Unprocessable Entity`:

```json
{
  "detail": [{"loc": ["body", "price"], "msg": "Input should be greater than 0", "type": "greater_than"}]
}
```

---

## 🤖 ML Models

| Model | Description |
|---|---|
| Linear Regression | Baseline — fast, interpretable |
| Random Forest | Ensemble of 200 trees — handles non-linearities |
| XGBoost | Gradient-boosted trees — typically best accuracy |

Best model is selected by R² on an 80/20 held-out test split and saved automatically.

### Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — average unit prediction error |
| RMSE | Root Mean Squared Error — penalizes large errors |
| R² | Coefficient of determination — proportion of variance explained |

---

## 🧮 Price Elasticity

$$E = \frac{\% \Delta\ \text{Demand}}{\% \Delta\ \text{Price}} = \frac{(Q_2 - Q_1) / \bar{Q}}{(P_2 - P_1) / \bar{P}}$$

| Range | Classification | Implication |
|---|---|---|
| \|E\| < 0.5 | Highly Inelastic | Strong pricing power — raise prices |
| 0.5–1.0 | Inelastic | Moderate pricing power |
| \|E\| ≈ 1.0 | Unit Elastic | Revenue stable across prices |
| 1.0–2.5 | Elastic | Price-sensitive — discounts drive volume |
| \|E\| ≥ 2.5 | Highly Elastic | Very price-sensitive market |

---

## 🧠 Model Explainability (SHAP)

The **Explainability** tab uses SHAP (SHapley Additive exPlanations) to:

- Show global feature importance (mean |SHAP| across all samples)
- Display per-feature SHAP value distributions (violin plots)
- Render a SHAP heatmap across sample × feature
- Explain a single prediction's feature contributions

Tree models use `TreeExplainer` (fast); linear models use `LinearExplainer`.

---

## ⚙️ CI/CD Pipeline

```
Push to main
 ↓
GitHub Actions (.github/workflows/ci.yml)
 ├── flake8 lint
 ├── pytest (unit + integration)
 └── docker build check
```

---

## 🔧 Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | FastAPI server port |
| `STREAMLIT_PORT` | `8501` | Streamlit dashboard port |
| `AUTO_TRAIN_IF_MISSING` | `true` | Auto-train if model file absent (set `false` in Docker) |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `MODEL_PATH` | `models/best_model.joblib` | Path to trained model |

---

## 🔬 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- Data generation: shape, column presence, no negative demand
- Preprocessing: feature engineering, missing value handling
- Model training: metric computation, model persistence
- Elasticity: arc elasticity formula, category classification
- Revenue optimization: optimal price extraction
- API: health endpoint, input validation

---

## 🏆 Resume-Ready Achievements

- Designed an **end-to-end MLOps pipeline** comparing Linear Regression, Random Forest, and XGBoost with automated model selection by R²
- Implemented **price elasticity of demand** modeling with arc-elasticity formula and 5-tier business classification
- Built a **revenue optimization engine** simulating 60+ price candidates to find the revenue-maximizing price
- Exposed ML predictions via a **FastAPI REST API** with Pydantic v2 validation, structured logging, and OpenAPI docs
- Deployed a **7-tab Streamlit dashboard** with Plotly charts, SHAP explainability, CSV exports, and on-demand retraining
- Containerized with **Docker** and **Docker Compose** — API and dashboard run in independent containers
- Implemented **on-demand model retraining** via REST endpoint with background processing, status polling, and history tracking
- Added **GitHub Actions CI** pipeline: lint → test → docker build on every push

---

## 🔮 Future Improvements

- [ ] Scheduled retraining with model drift detection
- [ ] A/B pricing experiment tracking
- [ ] Real-time demand signals via streaming (Kafka / Kinesis)
- [ ] Multi-product portfolio optimization
- [ ] Bayesian price optimization with uncertainty bounds
- [ ] Model registry with versioning (MLflow)

---

## 📄 License

MIT License — free to use, modify, and distribute.
