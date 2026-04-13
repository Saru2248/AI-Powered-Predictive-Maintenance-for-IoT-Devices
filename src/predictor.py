# ============================================================
# Module: predictor.py
# Purpose: Use the trained model to predict machine failures
#          on new sensor readings and generate ALERTS.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime


# ─────────────────────────────────────────
# Alert thresholds (domain knowledge)
# ─────────────────────────────────────────
ALERT_THRESHOLDS = {
    "temperature" : {"warning": 95.0,  "critical": 110.0},
    "vibration"   : {"warning": 1.5,   "critical": 2.0},
    "pressure"    : {"warning": 75.0,  "critical": 60.0},   # low pressure = bad
    "rpm"         : {"warning": 1200.0,"critical": 1000.0}, # low rpm = bad
    "oil_level"   : {"warning": 30.0,  "critical": 15.0},  # low oil = bad
}


def load_model_artifacts(model_dir: str = "models"):
    """Load trained model + scaler + feature names."""
    model        = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    scaler       = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
    return model, scaler, feature_cols


def predict_failure(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Run failure prediction on a processed & feature-engineered DataFrame.

    Adds columns:
      failure_prob   – probability of failure (0.0 → 1.0)
      predicted_fail – 1 = failure predicted, 0 = normal
      risk_level     – "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    """
    # ── Ensure all required feature columns exist ──────────────
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0   # safe default

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    probs       = model.predict_proba(X_scaled)[:, 1]
    predictions = (probs >= 0.5).astype(int)

    df = df.copy()
    df["failure_prob"]   = np.round(probs, 4)
    df["predicted_fail"] = predictions

    # ── Risk Level ────────────────────────────────────────────
    def risk_label(p):
        if p < 0.25:  return "LOW"
        if p < 0.50:  return "MEDIUM"
        if p < 0.75:  return "HIGH"
        return "CRITICAL"

    df["risk_level"] = df["failure_prob"].apply(risk_label)
    return df


def generate_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate maintenance alerts based on:
    1. Model prediction (ML-driven)
    2. Rule-based threshold breaches (classical)

    Returns a filtered DataFrame of alert records.
    """
    alerts = []

    for _, row in df.iterrows():
        alert_reasons = []

        # ── ML-based alert ────────────────────────────────────
        if row["predicted_fail"] == 1:
            alert_reasons.append(
                f"ML model predicts FAILURE (prob={row['failure_prob']:.1%})"
            )

        # ── Rule-based alerts ─────────────────────────────────
        if row["temperature"] >= ALERT_THRESHOLDS["temperature"]["critical"]:
            alert_reasons.append(f"⚠ CRITICAL: Temperature={row['temperature']}°C")
        elif row["temperature"] >= ALERT_THRESHOLDS["temperature"]["warning"]:
            alert_reasons.append(f"⚠ WARNING: Temperature={row['temperature']}°C")

        if row["vibration"] >= ALERT_THRESHOLDS["vibration"]["critical"]:
            alert_reasons.append(f"⚠ CRITICAL: Vibration={row['vibration']} mm/s")

        if row["oil_level"] <= ALERT_THRESHOLDS["oil_level"]["critical"]:
            alert_reasons.append(f"⚠ CRITICAL: Oil Level={row['oil_level']}%")
        elif row["oil_level"] <= ALERT_THRESHOLDS["oil_level"]["warning"]:
            alert_reasons.append(f"⚠ WARNING: Oil Level={row['oil_level']}%")

        if row["rpm"] <= ALERT_THRESHOLDS["rpm"]["critical"]:
            alert_reasons.append(f"⚠ CRITICAL: RPM={row['rpm']}")

        if alert_reasons:
            alerts.append({
                "machine_id"  : row["machine_id"],
                "timestamp"   : row["timestamp"],
                "risk_level"  : row["risk_level"],
                "failure_prob": row["failure_prob"],
                "alert_reason": " | ".join(alert_reasons),
            })

    alert_df = pd.DataFrame(alerts)
    if not alert_df.empty:
        alert_df = alert_df.sort_values("failure_prob", ascending=False)

    return alert_df


def print_alert_summary(alert_df: pd.DataFrame) -> None:
    """Print a formatted alert summary to the console."""
    if alert_df.empty:
        print("[✓] No alerts detected — all machines operating normally.")
        return

    print("\n" + "="*60)
    print("  🚨  PREDICTIVE MAINTENANCE ALERT SYSTEM  🚨")
    print("="*60)
    print(f"  Total alerts detected: {len(alert_df)}")

    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        subset = alert_df[alert_df["risk_level"] == level]
        if not subset.empty:
            print(f"\n  [{level}] — {len(subset)} alert(s)")
            print("  " + "─"*50)
            for _, a in subset.iterrows():
                print(f"  Machine {int(a['machine_id'])} | {a['timestamp']} | Prob: {a['failure_prob']:.1%}")
                print(f"  → {a['alert_reason']}")
                print()

    print("="*60 + "\n")


def save_alerts(alert_df: pd.DataFrame, save_path: str = "outputs/alerts.csv") -> None:
    """Save alerts to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    alert_df.to_csv(save_path, index=False)
    print(f"[✓] Alerts saved → {save_path}")


# ─────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────
if __name__ == "__main__":
    from preprocessor import preprocess_pipeline
    from feature_engineer import feature_engineering_pipeline, get_feature_columns

    df           = preprocess_pipeline("data/sensor_data.csv")
    df           = feature_engineering_pipeline(df)
    feature_cols = get_feature_columns(df)

    model, scaler, feature_cols = load_model_artifacts()
    df = predict_failure(df, model, scaler, feature_cols)

    # Show summary
    print(df[["machine_id", "timestamp", "failure_prob", "predicted_fail", "risk_level"]].tail(20))

    alerts = generate_alerts(df)
    print_alert_summary(alerts)
    save_alerts(alerts)
