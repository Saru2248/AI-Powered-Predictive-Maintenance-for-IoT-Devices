# ============================================================
# Module: model_trainer.py
# Purpose: Train multiple ML models for failure prediction.
#          Compare models and select the best one.
#          Save the best model for deployment.
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE


def prepare_data(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    """
    Split data into train/test sets.
    Uses stratify=y to maintain class ratio in both splits.
    SMOTE is applied to balance the training set (failures are rare!).

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X = df[feature_cols].values
    y = df["failure"].values

    # ── Train / Test Split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Scale Features ────────────────────────────────────────
    # StandardScaler: mean=0, std=1 — helps Logistic Regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Handle Class Imbalance with SMOTE ─────────────────────
    # Failures are rare (5-10%). SMOTE creates synthetic failure
    # samples so the model sees equal numbers of both classes.
    print(f"[→] Before SMOTE — Class distribution: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"[✓] After  SMOTE — Class distribution: {np.bincount(y_train)}")

    print(f"\n[✓] Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    print(f"[✓] Features:   {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a model and return metrics as a dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model"    : model_name,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'─'*50}")
    print(f"  Model: {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}   ← crucial for maintenance!")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Failure','Failure'])}")

    return metrics


def train_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train three models and compare them:
    1. Logistic Regression   → fast baseline
    2. Random Forest         → handles non-linear patterns
    3. Gradient Boosting     → usually best on tabular data

    Returns list of (model, metrics) tuples + the best model.
    """

    models_to_train = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=42
        ),
    }

    results    = []
    all_metrics = []

    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    for name, model in models_to_train.items():
        print(f"\n[→] Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append((model, metrics))
        all_metrics.append(metrics)

    # ── Compare and select best model ────────────────────────
    # For predictive maintenance, RECALL is most important:
    # Missing a failure is worse than a false alarm.
    best_model, best_metrics = max(results, key=lambda x: x[1]["f1"])
    print(f"\n{'='*55}")
    print(f"  🏆 Best Model: {best_metrics['model']}")
    print(f"     F1 Score : {best_metrics['f1']:.4f}")
    print(f"     ROC-AUC  : {best_metrics['roc_auc']:.4f}")
    print(f"{'='*55}\n")

    return results, best_model, best_metrics, all_metrics


def save_model(model, scaler, feature_cols: list, save_dir: str = "models") -> None:
    """Save the trained model, scaler, and feature list for later use."""
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model,        os.path.join(save_dir, "best_model.joblib"))
    joblib.dump(scaler,       os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(save_dir, "feature_cols.joblib"))

    print(f"[✓] Model   saved → {save_dir}/best_model.joblib")
    print(f"[✓] Scaler  saved → {save_dir}/scaler.joblib")
    print(f"[✓] Features saved → {save_dir}/feature_cols.joblib")


def load_model(save_dir: str = "models"):
    """Load a previously saved model, scaler, and feature list."""
    model        = joblib.load(os.path.join(save_dir, "best_model.joblib"))
    scaler       = joblib.load(os.path.join(save_dir, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(save_dir, "feature_cols.joblib"))
    print("[✓] Model loaded from disk")
    return model, scaler, feature_cols


# ─────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────
if __name__ == "__main__":
    from preprocessor import preprocess_pipeline
    from feature_engineer import feature_engineering_pipeline, get_feature_columns

    df           = preprocess_pipeline("data/sensor_data.csv")
    df           = feature_engineering_pipeline(df)
    feature_cols = get_feature_columns(df)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df, feature_cols)
    results, best_model, best_metrics, all_metrics = train_all_models(
        X_train, X_test, y_train, y_test
    )
    save_model(best_model, scaler, feature_cols)
