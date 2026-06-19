"""
Data preprocessing module for the Price Optimization Engine.
Handles loading, cleaning, feature engineering, and EDA.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "retail_sales.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.joblib")

FEATURE_COLS = [
    "price",
    "competitor_price",
    "promotion",
    "inventory_level",
    "holiday",
    "weekday",
    "season_encoded",
    "price_competitor_ratio",
    "log_price",
    "log_inventory",
]
TARGET_COL = "demand"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV; generate synthetic data if file does not exist."""
    if not os.path.exists(path):
        print(f"Raw data not found at {path}. Generating synthetic dataset...")
        sys.path.insert(0, DATA_DIR)
        from generate_data import generate_retail_dataset
        os.makedirs(DATA_DIR, exist_ok=True)
        df = generate_retail_dataset(save_path=path)
    else:
        df = pd.read_csv(path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw columns."""
    df = df.copy()

    df["price_competitor_ratio"] = (df["price"] / df["competitor_price"].replace(0, 1)).round(4)
    df["log_price"] = np.log1p(df["price"])
    df["log_inventory"] = np.log1p(df["inventory_level"])
    df["price_per_unit"] = df["price"]
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["promotion_holiday"] = df["promotion"] * df["holiday"]

    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> tuple:
    """
    Encode categorical columns.
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders = {}

    season_order = ["Spring", "Summer", "Fall", "Winter"]
    df["season_encoded"] = df["season"].map(
        {s: i for i, s in enumerate(season_order)}
    ).fillna(0).astype(int)

    if fit:
        le = LabelEncoder()
        df["product_encoded"] = le.fit_transform(df["product"].astype(str))
        encoders["product"] = le
    else:
        # Load saved encoders
        if os.path.exists(ENCODERS_PATH):
            encoders = joblib.load(ENCODERS_PATH)
            le = encoders.get("product")
            if le:
                known = set(le.classes_)
                df["product"] = df["product"].apply(lambda x: x if x in known else le.classes_[0])
                df["product_encoded"] = le.transform(df["product"].astype(str))
        else:
            df["product_encoded"] = 0

    return df, encoders


def scale_features(X: pd.DataFrame, fit: bool = True) -> tuple:
    """Scale numeric features; returns (scaled_array, scaler)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def run_eda(df: pd.DataFrame) -> dict:
    """Return basic EDA statistics for the dataset."""
    eda = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "demand_stats": df["demand"].describe().round(2).to_dict(),
        "price_stats": df["price"].describe().round(2).to_dict(),
        "promotion_rate": float(df["promotion"].mean().round(3)),
        "holiday_rate": float(df["holiday"].mean().round(3)),
        "season_counts": df["season"].value_counts().to_dict(),
        "demand_by_season": df.groupby("season")["demand"].mean().round(1).to_dict(),
        "demand_by_promotion": df.groupby("promotion")["demand"].mean().round(1).to_dict(),
        "price_demand_corr": float(df[["price", "demand"]].corr().loc["price", "demand"].round(4)),
    }
    return eda


def preprocess_pipeline(save: bool = True) -> tuple:
    """
    Full preprocessing pipeline.
    Returns (X_train, X_test, y_train, y_test, feature_names, df_clean).
    """
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df, encoders = encode_categoricals(df, fit=True)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(encoders, ENCODERS_PATH)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(CLEAN_DATA_PATH, index=False)
        print(f"Cleaned data saved to {CLEAN_DATA_PATH}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, scaler = scale_features(X_train_raw, fit=True)
    X_test_scaled, _ = scale_features(X_test_raw, fit=False)

    return X_train_scaled, X_test_scaled, y_train, y_test, FEATURE_COLS, df


def preprocess_single(
    price: float,
    competitor_price: float,
    promotion: int,
    inventory_level: int,
    season: str = "Summer",
    holiday: int = 0,
    weekday: int = 2,
) -> np.ndarray:
    """
    Preprocess a single prediction input using saved scaler.
    Returns scaled feature array ready for model prediction.
    """
    price_competitor_ratio = price / max(competitor_price, 1)
    log_price = np.log1p(price)
    log_inventory = np.log1p(inventory_level)
    season_map = {"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}
    season_encoded = season_map.get(season, 1)

    features = pd.DataFrame([[
        price,
        competitor_price,
        promotion,
        inventory_level,
        holiday,
        weekday,
        season_encoded,
        price_competitor_ratio,
        log_price,
        log_inventory,
    ]], columns=FEATURE_COLS)

    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(features)


if __name__ == "__main__":
    print("Running preprocessing pipeline...")
    X_tr, X_te, y_tr, y_te, feats, df_clean = preprocess_pipeline()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    eda = run_eda(df_clean)
    print("\nEDA Summary:")
    for k, v in eda.items():
        print(f"  {k}: {v}")
