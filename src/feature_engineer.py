# ============================================================
# Module: feature_engineer.py
# Purpose: Create meaningful features from raw sensor data.
#          Good features = better model accuracy.
#          This is one of the most important steps in ML.
# ============================================================

import pandas as pd
import numpy as np


def add_rolling_features(df: pd.DataFrame, windows: list = [3, 6, 12]) -> pd.DataFrame:
    """
    Rolling window statistics capture trends over time.
    Think of it as: "What was the AVERAGE temperature in the last 3 hours?"

    For each window size, compute:
      - Rolling mean  (trend)
      - Rolling std   (variability / instability)
      - Rolling max   (peak stress)
    """
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level"]

    df = df.sort_values(["machine_id", "timestamp"]).copy()

    for col in sensor_cols:
        for w in windows:
            # Group by machine so windows don't leak across machines
            grouped = df.groupby("machine_id")[col]

            df[f"{col}_roll_mean_{w}h"] = (
                grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            df[f"{col}_roll_std_{w}h"] = (
                grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            )
            df[f"{col}_roll_max_{w}h"] = (
                grouped.transform(lambda x: x.rolling(w, min_periods=1).max())
            )

    print(f"[✓] Rolling features added for windows: {windows} hours")
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 3, 6]) -> pd.DataFrame:
    """
    Lag features capture the PAST values of sensors.
    e.g., "What was the temperature 3 hours ago?"
    This helps the model learn temporal patterns.
    """
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level"]

    df = df.sort_values(["machine_id", "timestamp"]).copy()

    for col in sensor_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = (
                df.groupby("machine_id")[col]
                  .transform(lambda x: x.shift(lag))
            )

    # Fill NaN from lags with forward/backward fill
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df[lag_cols] = df[lag_cols].bfill().ffill()

    print(f"[✓] Lag features added for lags: {lags} hours")
    return df


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rate of change = how fast a sensor value is changing.
    A sudden spike in vibration rate → likely approaching failure.
    """
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level"]

    df = df.sort_values(["machine_id", "timestamp"]).copy()

    for col in sensor_cols:
        df[f"{col}_rate_of_change"] = (
            df.groupby("machine_id")[col]
              .transform(lambda x: x.diff().fillna(0))
        )

    print("[✓] Rate-of-change features added")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven interaction features:
    - temp × vibration  → thermal + mechanical stress combined
    - pressure × rpm    → hydraulic load
    - oil_ratio         → how depleted is the oil?
    """
    df["temp_vib_stress"]   = df["temperature"] * df["vibration"]
    df["pressure_rpm_load"] = df["pressure"] * df["rpm"] / 1000.0
    df["oil_ratio"]         = df["oil_level"] / 100.0  # normalized 0→1

    print("[✓] Interaction (domain) features added")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp.
    Machines may behave differently at different hours/days.
    """
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    print("[✓] Time-based features added (hour, day_of_week, month, is_weekend)")
    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Combines rolling, lag, rate-of-change, interaction, and time features.

    Returns enriched DataFrame for model training.
    """
    print("\n[→] Starting Feature Engineering Pipeline...")
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_rate_of_change(df)
    df = add_interaction_features(df)
    df = add_time_features(df)

    # Drop rows with any remaining NaN (from rolling at start of series)
    before = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    print(f"[✓] Dropped {before - df.shape[0]} NaN rows after feature engineering")
    print(f"[✓] Final feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print("[✓] Feature engineering complete.\n")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return only the feature columns (everything except IDs, timestamps, and label).
    """
    exclude = ["machine_id", "timestamp", "failure"]
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────
if __name__ == "__main__":
    from preprocessor import preprocess_pipeline
    df = preprocess_pipeline("data/sensor_data.csv")
    df = feature_engineering_pipeline(df)
    features = get_feature_columns(df)
    print(f"Total features: {len(features)}")
    print(features[:10])
