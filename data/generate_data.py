"""
Generate synthetic retail sales dataset for the Price Optimization Engine.
Produces realistic demand-price relationships with noise.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SAMPLES = 6000
PRODUCTS = ["Widget_A", "Widget_B", "Gadget_X", "Gadget_Y", "Device_Z"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]


def generate_retail_dataset(n: int = N_SAMPLES, save_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic retail sales dataset with realistic price-demand dynamics.

    Features:
        - price: selling price of the product
        - competitor_price: rival product price
        - promotion: binary flag (0/1)
        - inventory_level: current stock units
        - season: categorical season
        - holiday: binary holiday indicator
        - weekday: day of week (0=Mon, 6=Sun)
        - product: product name
        - demand: units sold (target variable)
    """
    product = np.random.choice(PRODUCTS, n)
    base_prices = {"Widget_A": 80, "Widget_B": 120, "Gadget_X": 200, "Gadget_Y": 60, "Device_Z": 300}
    base_demands = {"Widget_A": 800, "Widget_B": 500, "Gadget_X": 300, "Gadget_Y": 1000, "Device_Z": 150}

    price_raw = np.array([base_prices[p] * np.random.uniform(0.7, 1.4) for p in product])
    price = np.round(price_raw, 2)

    competitor_price = np.round(price * np.random.uniform(0.85, 1.20, n), 2)
    promotion = np.random.binomial(1, 0.25, n)
    inventory_level = np.random.randint(50, 2000, n)
    season = np.random.choice(SEASONS, n)
    holiday = np.random.binomial(1, 0.08, n)
    weekday = np.random.randint(0, 7, n)

    season_multiplier = {"Spring": 1.05, "Summer": 1.20, "Fall": 0.95, "Winter": 0.90}

    base_demand = np.array([base_demands[p] for p in product])
    season_effect = np.array([season_multiplier[s] for s in season])

    price_elasticity = -2.1
    price_ratio = price / np.array([base_prices[p] for p in product])
    competitor_effect = np.where(price < competitor_price, 1.08, 0.94)

    demand = (
        base_demand
        * (price_ratio ** price_elasticity)
        * season_effect
        * np.where(promotion, 1.30, 1.0)
        * np.where(holiday, 1.20, 1.0)
        * np.where(weekday >= 5, 1.12, 1.0)
        * competitor_effect
        * np.random.uniform(0.82, 1.18, n)
    )
    demand = np.maximum(demand, 10).round().astype(int)

    df = pd.DataFrame({
        "product": product,
        "price": price,
        "competitor_price": competitor_price,
        "promotion": promotion,
        "inventory_level": inventory_level,
        "season": season,
        "holiday": holiday,
        "weekday": weekday,
        "demand": demand,
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}  ({len(df)} rows)")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, "data", "retail_sales.csv")
    df = generate_retail_dataset(save_path=out_path)
    print(df.describe())
    print("\nSeason distribution:\n", df["season"].value_counts())
    print("\nPromotion rate:", df["promotion"].mean().round(3))
