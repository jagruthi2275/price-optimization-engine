"""
Unit and integration tests for the Price Optimization Engine pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


class TestDataGeneration:
    def test_generate_returns_dataframe(self):
        from data.generate_data import generate_retail_dataset
        df = generate_retail_dataset(n=100)
        assert len(df) == 100

    def test_required_columns_present(self):
        from data.generate_data import generate_retail_dataset
        df = generate_retail_dataset(n=50)
        required = {"price", "competitor_price", "promotion", "inventory_level",
                    "season", "holiday", "weekday", "demand"}
        assert required.issubset(df.columns)

    def test_no_negative_demand(self):
        from data.generate_data import generate_retail_dataset
        df = generate_retail_dataset(n=200)
        assert (df["demand"] >= 0).all()

    def test_promotion_binary(self):
        from data.generate_data import generate_retail_dataset
        df = generate_retail_dataset(n=200)
        assert df["promotion"].isin([0, 1]).all()

    def test_prices_positive(self):
        from data.generate_data import generate_retail_dataset
        df = generate_retail_dataset(n=200)
        assert (df["price"] > 0).all()
        assert (df["competitor_price"] > 0).all()


class TestPreprocessing:
    def test_feature_engineering_adds_columns(self):
        from data.generate_data import generate_retail_dataset
        from src.data_preprocessing import feature_engineering
        df = generate_retail_dataset(n=50)
        df_feat = feature_engineering(df)
        assert "log_price" in df_feat.columns
        assert "price_competitor_ratio" in df_feat.columns
        assert "log_inventory" in df_feat.columns

    def test_handle_missing_values_removes_nans(self):
        import pandas as pd
        from src.data_preprocessing import handle_missing_values
        from data.generate_data import generate_retail_dataset
        df = generate_retail_dataset(n=50)
        df.loc[0, "price"] = np.nan
        df.loc[1, "demand"] = np.nan
        df_clean = handle_missing_values(df)
        assert df_clean[["price", "demand"]].isnull().sum().sum() == 0

    def test_preprocess_single_returns_array(self):
        import joblib
        model_path = os.path.join(BASE_DIR, "models", "best_model.joblib")
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.joblib")
        if not os.path.exists(scaler_path):
            pytest.skip("Scaler not trained yet — run training pipeline first.")
        from src.data_preprocessing import preprocess_single
        X = preprocess_single(100.0, 95.0, 0, 500)
        assert X.shape[1] == 10


class TestModelTraining:
    def test_compute_metrics_returns_dict(self):
        from src.model_training import compute_metrics
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 195, 290])
        m = compute_metrics(y_true, y_pred)
        assert "MAE" in m and "RMSE" in m and "R2" in m
        assert 0 <= m["R2"] <= 1.1

    def test_compute_metrics_perfect_prediction(self):
        from src.model_training import compute_metrics
        y = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(y, y)
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert m["R2"] == pytest.approx(1.0)

    def test_load_best_model_raises_without_file(self, tmp_path, monkeypatch):
        import src.model_training as mt
        monkeypatch.setattr(mt, "BEST_MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
        with pytest.raises(FileNotFoundError):
            mt.load_best_model()


class TestPriceElasticity:
    def test_arc_elasticity_negative_for_normal_good(self):
        from src.price_elasticity import compute_arc_elasticity
        e = compute_arc_elasticity(100, 110, 500, 450)
        assert e < 0

    def test_arc_elasticity_zero_price_change(self):
        from src.price_elasticity import compute_arc_elasticity
        e = compute_arc_elasticity(100, 100, 500, 480)
        assert e == 0.0

    def test_elasticity_interpretation_categories(self):
        from src.price_elasticity import elasticity_interpretation
        assert "Inelastic" in elasticity_interpretation(-0.4)["category"]
        assert "Elastic" in elasticity_interpretation(-2.0)["category"]
        assert "Unit" in elasticity_interpretation(-1.0)["category"]


class TestRevenueOptimization:
    def test_find_optimal_price_returns_max_revenue(self):
        import pandas as pd
        from src.revenue_optimization import find_optimal_price
        df = pd.DataFrame({
            "price": [80, 100, 120, 90],
            "predicted_demand": [600, 500, 350, 560],
            "expected_revenue": [48000, 50000, 42000, 50400],
            "revenue_vs_base_pct": [0, 0, 0, 0],
            "is_optimal": [False, False, False, True],
        })
        result = find_optimal_price(df)
        assert result["recommended_price"] == 90
        assert result["expected_revenue"] == 50400


class TestAPIEndpoints:
    """Integration tests for FastAPI (requires running server or direct import)."""

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        try:
            from api.main import app
        except Exception:
            pytest.skip("Cannot import API without trained model.")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
