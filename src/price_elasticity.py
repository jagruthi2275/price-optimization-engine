"""
Price Elasticity of Demand module.
Calculates own-price elasticity and generates business recommendations.

Elasticity = (% Change in Demand) / (% Change in Price)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def compute_arc_elasticity(
    p1: float, p2: float, q1: float, q2: float
) -> float:
    """
    Arc (midpoint) price elasticity of demand.
    Avoids division-by-zero edge cases.
    """
    dp = p2 - p1
    dq = q2 - q1
    avg_p = (p1 + p2) / 2
    avg_q = (q1 + q2) / 2
    if avg_p == 0 or avg_q == 0:
        return 0.0
    return (dq / avg_q) / (dp / avg_p)


def compute_point_elasticity(
    price: float,
    competitor_price: float,
    promotion: int,
    inventory_level: int,
    model,
    scaler,
    delta_pct: float = 0.01,
    season: str = "Summer",
    holiday: int = 0,
    weekday: int = 2,
) -> float:
    """
    Estimate point elasticity via a small ±delta_pct price change.
    Uses the trained model to predict demand at p and p+delta.
    """
    from src.data_preprocessing import preprocess_single

    X1 = preprocess_single(price, competitor_price, promotion, inventory_level, season, holiday, weekday)
    q1 = float(model.predict(X1)[0])

    p2 = price * (1 + delta_pct)
    X2 = preprocess_single(p2, competitor_price, promotion, inventory_level, season, holiday, weekday)
    q2 = float(model.predict(X2)[0])

    return compute_arc_elasticity(price, p2, q1, q2)


def build_elasticity_curve(
    base_price: float,
    competitor_price: float,
    promotion: int,
    inventory_level: int,
    model,
    n_points: int = 40,
    price_range_pct: float = 0.50,
    season: str = "Summer",
    holiday: int = 0,
    weekday: int = 2,
) -> pd.DataFrame:
    """
    Build a demand curve and rolling elasticity across a price range.

    Returns DataFrame with columns: price, demand, elasticity, revenue.
    """
    from src.data_preprocessing import preprocess_single

    prices = np.linspace(
        base_price * (1 - price_range_pct),
        base_price * (1 + price_range_pct),
        n_points,
    )

    demands = []
    for p in prices:
        X = preprocess_single(p, competitor_price, promotion, inventory_level, season, holiday, weekday)
        d = max(float(model.predict(X)[0]), 0)
        demands.append(d)

    demands = np.array(demands)
    revenues = prices * demands

    elasticities = [np.nan]
    for i in range(1, len(prices)):
        e = compute_arc_elasticity(prices[i - 1], prices[i], demands[i - 1], demands[i])
        elasticities.append(e)

    df = pd.DataFrame({
        "price": np.round(prices, 2),
        "demand": np.round(demands, 1),
        "elasticity": np.round(elasticities, 4),
        "revenue": np.round(revenues, 2),
    })
    return df


def elasticity_interpretation(elasticity: float) -> dict:
    """Return a business interpretation of an elasticity value."""
    abs_e = abs(elasticity)
    if abs_e < 0.5:
        category = "Highly Inelastic"
        action = "Strong pricing power. You can raise prices with minimal demand loss."
        color = "#2ecc71"
    elif abs_e < 1.0:
        category = "Inelastic"
        action = "Moderate pricing power. Small price increases are viable."
        color = "#27ae60"
    elif abs_e < 1.5:
        category = "Unit Elastic"
        action = "Balanced market. Price changes proportionally affect revenue."
        color = "#f39c12"
    elif abs_e < 2.5:
        category = "Elastic"
        action = "Price-sensitive market. Discounts can significantly boost volume."
        color = "#e67e22"
    else:
        category = "Highly Elastic"
        action = "Very price-sensitive. Aggressive discounting may maximize revenue."
        color = "#e74c3c"

    return {
        "elasticity": round(elasticity, 4),
        "abs_elasticity": round(abs_e, 4),
        "category": category,
        "action": action,
        "color": color,
        "sign_explanation": (
            "Negative (normal good)" if elasticity < 0 else "Positive (Giffen/Veblen good)"
        ),
    }


def generate_elasticity_report(
    base_price: float,
    competitor_price: float,
    promotion: int,
    inventory_level: int,
    model,
    season: str = "Summer",
    holiday: int = 0,
    weekday: int = 2,
) -> dict:
    """
    Full elasticity report: curve data + interpretation + recommendations.
    """
    curve = build_elasticity_curve(
        base_price, competitor_price, promotion, inventory_level, model,
        season=season, holiday=holiday, weekday=weekday,
    )

    median_elasticity = float(curve["elasticity"].dropna().median())
    interpretation = elasticity_interpretation(median_elasticity)
    optimal_row = curve.loc[curve["revenue"].idxmax()]

    recommendations = []
    if median_elasticity < -1.5:
        recommendations.append("Consider promotional pricing — demand responds strongly to price cuts.")
    elif median_elasticity > -0.8:
        recommendations.append("Premium pricing viable — demand is relatively insensitive to price changes.")
    else:
        recommendations.append("Balanced pricing strategy recommended — test small increments.")

    if base_price > optimal_row["price"]:
        recommendations.append(
            f"Lowering price to ~${optimal_row['price']:.2f} may increase total revenue."
        )
    elif base_price < optimal_row["price"]:
        recommendations.append(
            f"There is room to raise price to ~${optimal_row['price']:.2f} for higher revenue."
        )
    else:
        recommendations.append("Current price is near the revenue-maximizing point.")

    return {
        "median_elasticity": median_elasticity,
        "interpretation": interpretation,
        "optimal_price": float(optimal_row["price"]),
        "optimal_revenue": float(optimal_row["revenue"]),
        "curve_data": curve.to_dict(orient="list"),
        "recommendations": recommendations,
    }
