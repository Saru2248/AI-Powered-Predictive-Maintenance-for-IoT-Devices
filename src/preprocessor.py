# ============================================================
# Module: preprocessor.py
# Purpose: Clean and preprocess raw IoT sensor data.
#          Handles missing values, outliers, and data types.
# ============================================================

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load sensor CSV data and parse timestamps."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"[✓] Loaded data:  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    print("\n" + "="*55)
    print("  DATASET OVERVIEW")
    print("="*55)
    print(f"  Shape         : {df.shape}")
    print(f"  Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Machines      : {df['machine_id'].nunique()}")
    print(f"  Failure events: {df['failure'].sum()} ({df['failure'].mean()*100:.2f}%)")
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())
    print("="*55 + "\n")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward-fill (common for time-series sensor data).
    Forward-fill means: use the last known valid reading.
    """
    before = df.isnull().sum().sum()
    df = df.sort_values(["machine_id", "timestamp"]).copy()

    # Forward fill within each machine group (preserves sensor continuity)
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level"]
    df[sensor_cols] = (
        df.groupby("machine_id")[sensor_cols]
          .transform(lambda x: x.ffill().bfill())
    )
    after = df.isnull().sum().sum()
    print(f"[✓] Missing values:  {before} → {after} (filled with forward-fill)")
    return df


def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
    """
    Remove extreme outliers using Z-score method.
    Values > z_threshold standard deviations from mean are clipped.
    We CLIP (not drop) so we don't lose time-series continuity.
    """
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level"]
    before = df.shape[0]

    for col in sensor_cols:
        mean = df[col].mean()
        std  = df[col].std()
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std
        df[col] = df[col].clip(lower=lower, upper=upper)

    print(f"[✓] Outlier clipping done on {len(sensor_cols)} sensor columns")
    print(f"    Rows preserved: {df.shape[0]:,} (no rows removed — values clipped)")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply physical domain constraints to sensor values.
    (e.g., RPM can't be negative, oil level must be 0-100%)
    """
    df["rpm"]       = df["rpm"].clip(lower=0)
    df["oil_level"] = df["oil_level"].clip(lower=0, upper=100)
    df["vibration"] = df["vibration"].clip(lower=0)
    df["pressure"]  = df["pressure"].clip(lower=0)
    print("[✓] Physical range validation applied")
    return df


def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    Load → Inspect → Handle Nulls → Remove Outliers → Validate

    Returns cleaned DataFrame ready for feature engineering.
    """
    print("\n[→] Starting Preprocessing Pipeline...")
    df = load_data(filepath)
    inspect_data(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = validate_ranges(df)
    print("[✓] Preprocessing complete.\n")
    return df


# ─────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────
if __name__ == "__main__":
    df = preprocess_pipeline("data/sensor_data.csv")
    print(df.head())
