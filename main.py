# ============================================================
# main.py
# AI-Powered Predictive Maintenance for IoT Devices
#
# Entry point — runs the full pipeline end-to-end:
#   1. Generate synthetic sensor data
#   2. Preprocess & clean
#   3. Feature engineering
#   4. Train models & evaluate
#   5. Predict failures & generate alerts
#   6. Visualize results
#
# Run: python main.py
# ============================================================

import os
import sys
import time

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_generator    import generate_sensor_data
from src.preprocessor      import preprocess_pipeline
from src.feature_engineer  import feature_engineering_pipeline, get_feature_columns
from src.model_trainer     import prepare_data, train_all_models, save_model
from src.predictor         import (
    predict_failure, generate_alerts,
    print_alert_summary, save_alerts
)
from src.visualizer        import generate_all_plots


def banner():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   AI-POWERED PREDICTIVE MAINTENANCE FOR IoT DEVICES      ║")
    print("║   Industry-Grade Machine Failure Prediction System        ║")
    print("╚" + "═"*58 + "╝\n")


def main():
    banner()
    start_total = time.time()

    # ──────────────────────────────────────────────────────────
    # PHASE 1 — Generate / Load Sensor Data
    # ──────────────────────────────────────────────────────────
    print("━"*60)
    print("PHASE 1 — DATA GENERATION (Virtual IoT Simulation)")
    print("━"*60)
    DATA_PATH = "data/sensor_data.csv"

    if os.path.exists(DATA_PATH):
        print(f"[→] Dataset found at {DATA_PATH} — skipping generation.")
        print("    (Delete data/sensor_data.csv to regenerate)")
    else:
        raw_df = generate_sensor_data(
            n_machines=5,
            days=60,
            samples_per_hour=1,
            failure_probability=0.05,
            save_path=DATA_PATH,
        )

    # ──────────────────────────────────────────────────────────
    # PHASE 2 — Preprocessing
    # ──────────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("PHASE 2 — DATA PREPROCESSING")
    print("━"*60)
    df = preprocess_pipeline(DATA_PATH)

    # ──────────────────────────────────────────────────────────
    # PHASE 3 — Feature Engineering
    # ──────────────────────────────────────────────────────────
    print("━"*60)
    print("PHASE 3 — FEATURE ENGINEERING")
    print("━"*60)
    df           = feature_engineering_pipeline(df)
    feature_cols = get_feature_columns(df)
    print(f"[✓] Total features created: {len(feature_cols)}")

    # ──────────────────────────────────────────────────────────
    # PHASE 4 — Model Training
    # ──────────────────────────────────────────────────────────
    print("━"*60)
    print("PHASE 4 — MODEL TRAINING & EVALUATION")
    print("━"*60)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, feature_cols)
    results, best_model, best_metrics, all_metrics = train_all_models(
        X_train, X_test, y_train, y_test
    )
    save_model(best_model, scaler, feature_cols)

    # ──────────────────────────────────────────────────────────
    # PHASE 5 — Failure Prediction & Alerts
    # ──────────────────────────────────────────────────────────
    print("━"*60)
    print("PHASE 5 — FAILURE PREDICTION & ALERT GENERATION")
    print("━"*60)

    # Only run predictions on test portion for speed
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Predict on entire dataset for visualization
    df_pred = predict_failure(df, best_model, scaler, feature_cols)
    alerts  = generate_alerts(df_pred)
    print_alert_summary(alerts)
    save_alerts(alerts)

    # Save prediction results
    os.makedirs("outputs", exist_ok=True)
    df_pred.to_csv("outputs/predictions.csv", index=False)
    print("[✓] Full predictions saved → outputs/predictions.csv")

    # ──────────────────────────────────────────────────────────
    # PHASE 6 — Visualization
    # ──────────────────────────────────────────────────────────
    print("━"*60)
    print("PHASE 6 — VISUALIZATION")
    print("━"*60)

    generate_all_plots(
        df=df_pred,
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        all_metrics=all_metrics,
    )

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   ✅  PIPELINE COMPLETE                                   ║")
    print(f"║   Best Model  : {best_metrics['model']:<40} ║")
    print(f"║   F1 Score    : {best_metrics['f1']:.4f}{'':<36} ║")
    print(f"║   ROC-AUC     : {best_metrics['roc_auc']:.4f}{'':<36} ║")
    print(f"║   Total Alerts: {len(alerts):<40} ║")
    print(f"║   Time Taken  : {elapsed:.1f}s{'':<40} ║")
    print("╚" + "═"*58 + "╝\n")
    print("📁 Check these folders for outputs:")
    print("   outputs/images/   → all plots and charts")
    print("   outputs/alerts.csv    → maintenance alerts")
    print("   outputs/predictions.csv → full predictions")
    print("   models/            → saved ML model\n")


if __name__ == "__main__":
    main()
