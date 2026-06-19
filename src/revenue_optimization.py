"""
Revenue Optimization Engine.
Simulates prices over a configurable range, predicts demand for each candidate,
and identifies the price that maximizes expected revenue.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def simulate_price_range(
    base_price: float,
    competitor_price: float,
    promotion: int,
    inventory_level: int,
    model,
    n_simulations: int = 60,
    lower_pct: float = 0.50,
    upper_pct: float = 1.50,
    season: str = "Summer",
    holiday: int = 0,
    weekday: int = 2,
) -> pd.DataFrame:
    """
    Simulate prices from lower_pct to upper_pct of base_price.
    For each candidate price: predict demand, calculate revenue.

    Returns a DataFrame with price, demand, revenue, revenue_vs_base (%).
    """
    from src.data_preprocessing import preprocess_single

    prices = np.linspace(
        base_price * lower_pct,
        base_price * upper_pct,
        n_simulations,
    )

    rows = []
    for p in prices:
        X = preprocess_single(
            p, competitor_price, promotion, inventory_level, season, holiday, weekday
        )
        demand = max(float(model.predict(X)[0]), 0)
        revenue = p * demand
        rows.append({"price": round(p, 2), "predicted_demand": round(demand, 1), "expected_revenue": round(revenue, 2)})

    df = pd.DataFrame(rows)

    base_idx = np.argmin(np.abs(df["price"] - base_price))
    base_revenue = df.iloc[base_idx]["expected_revenue"]
    df["revenue_vs_base_pct"] = ((df["expected_revenue"] - base_revenue) / max(base_revenue, 1) * 100).round(2)
    df["is_optimal"] = df["expected_revenue"] == df["expected_revenue"].max()

    return df


def find_optimal_price(simulation_df: pd.DataFrame) -> dict:
    """Extract the row with maximum expected revenue from simulation results."""
    opt_row = simulation_df.loc[simulation_df["expected_revenue"].idxmax()]
    return {
        "recommended_price": float(opt_row["price"]),
        "predicted_demand": float(opt_row["predicted_demand"]),
        "expected_revenue": float(opt_row["expected_revenue"]),
        "revenue_vs_base_pct": float(opt_row["revenue_vs_base_pct"]),
    }


def run_optimization(
    base_price: float,
    competitor_price: float,
    promotion: int,
    inventory_level: int,
    model,
    season: str = "Summer",
    holiday: int = 0,
    weekday: int = 2,
    n_simulations: int = 60,
) -> dict:
    """
    Full optimization run. Returns optimal price, demand, revenue, and comparison table.
    """
    sim_df = simulate_price_range(
        base_price, competitor_price, promotion, inventory_level, model,
        n_simulations=n_simulations, season=season, holiday=holiday, weekday=weekday,
    )
    optimal = find_optimal_price(sim_df)

    from src.data_preprocessing import preprocess_single
    X_base = preprocess_single(base_price, competitor_price, promotion, inventory_level, season, holiday, weekday)
    base_demand = max(float(model.predict(X_base)[0]), 0)
    base_revenue = base_price * base_demand

    top10 = (
        sim_df.nlargest(10, "expected_revenue")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    return {
        "recommended_price": optimal["recommended_price"],
        "predicted_demand": optimal["predicted_demand"],
        "expected_revenue": optimal["expected_revenue"],
        "revenue_uplift_pct": round(
            (optimal["expected_revenue"] - base_revenue) / max(base_revenue, 1) * 100, 2
        ),
        "base_price": base_price,
        "base_demand": round(base_demand, 1),
        "base_revenue": round(base_revenue, 2),
        "simulation_table": sim_df.to_dict(orient="records"),
        "top_10_prices": top10,
    }


def generate_optimization_report(result: dict) -> str:
    """Return a human-readable text summary of the optimization result."""
    lines = [
        "=" * 55,
        "  PRICE OPTIMIZATION REPORT",
        "=" * 55,
        f"  Base Price:          ${result['base_price']:.2f}",
        f"  Base Demand:         {result['base_demand']:.0f} units",
        f"  Base Revenue:        ${result['base_revenue']:,.2f}",
        "-" * 55,
        f"  Recommended Price:   ${result['recommended_price']:.2f}",
        f"  Predicted Demand:    {result['predicted_demand']:.0f} units",
        f"  Expected Revenue:    ${result['expected_revenue']:,.2f}",
        f"  Revenue Uplift:      {result['revenue_uplift_pct']:+.1f}%",
        "=" * 55,
    ]
    return "\n".join(lines)
