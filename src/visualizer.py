# ============================================================
# Module: visualizer.py
# Purpose: Generate all charts and plots for the project.
#          Saves figures to outputs/images/ for GitHub proof.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (server/script safe)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ─────────────────────────────────────────
# Global plot style
# ─────────────────────────────────────────
SAVE_DIR = "outputs/images"
STYLE    = "dark_background"
PALETTE  = ["#00d4ff", "#ff4757", "#2ed573", "#ffa502", "#a29bfe"]

plt.rcParams.update({
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "figure.dpi"      : 120,
    "savefig.dpi"     : 150,
    "savefig.bbox"    : "tight",
})


def _savefig(fig, name: str) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[✓] Saved plot → {path}")
    return path


# ─────────────────────────────────────────
# 1. Sensor time-series for one machine
# ─────────────────────────────────────────
def plot_sensor_timeseries(df: pd.DataFrame, machine_id: int = 1) -> str:
    mdf = df[df["machine_id"] == machine_id].copy()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"Machine #{machine_id} — Sensor Readings Over Time",
            fontsize=14, color="white", fontweight="bold", y=1.02
        )

        sensors = [
            ("temperature", "°C",    PALETTE[0]),
            ("vibration",   "mm/s",  PALETTE[1]),
            ("pressure",    "PSI",   PALETTE[2]),
            ("rpm",         "RPM",   PALETTE[3]),
            ("oil_level",   "%",     PALETTE[4]),
        ]

        for ax, (col, unit, color) in zip(axes, sensors):
            ax.plot(mdf["timestamp"], mdf[col], color=color, linewidth=0.8, alpha=0.9)

            # Highlight failure events
            fail_mask = mdf["failure"] == 1
            ax.scatter(
                mdf.loc[fail_mask, "timestamp"],
                mdf.loc[fail_mask, col],
                color=PALETTE[1], s=20, zorder=5, label="Failure Event"
            )
            ax.set_ylabel(f"{col}\n({unit})", color=color, fontsize=9)
            ax.tick_params(colors="white")
            ax.set_facecolor("#1a1a2e")
            ax.grid(alpha=0.2, color="gray")

        axes[-1].set_xlabel("Timestamp", color="white")
        fig.tight_layout()

    return _savefig(fig, f"01_sensor_timeseries_machine_{machine_id}.png")


# ─────────────────────────────────────────
# 2. Failure distribution bar chart
# ─────────────────────────────────────────
def plot_failure_distribution(df: pd.DataFrame) -> str:
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Failure Distribution Analysis", fontsize=14,
                     color="white", fontweight="bold")

        # Overall counts
        counts = df["failure"].value_counts()
        axes[0].bar(["Normal", "Failure"], [counts.get(0, 0), counts.get(1, 0)],
                    color=[PALETTE[0], PALETTE[1]], edgecolor="white", linewidth=0.5)
        axes[0].set_title("Overall Class Distribution", color="white")
        axes[0].set_ylabel("Count", color="white")
        axes[0].tick_params(colors="white")
        axes[0].set_facecolor("#1a1a2e")
        for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
            axes[0].text(i, v + 50, str(v), ha="center", color="white", fontweight="bold")

        # Per machine
        per_machine = df.groupby("machine_id")["failure"].mean() * 100
        bars = axes[1].bar(
            [f"M{m}" for m in per_machine.index],
            per_machine.values,
            color=PALETTE[3], edgecolor="white", linewidth=0.5
        )
        axes[1].set_title("Failure Rate per Machine (%)", color="white")
        axes[1].set_ylabel("Failure Rate (%)", color="white")
        axes[1].tick_params(colors="white")
        axes[1].set_facecolor("#1a1a2e")
        axes[1].axhline(per_machine.mean(), color=PALETTE[1],
                        linestyle="--", linewidth=1.2, label=f"Avg: {per_machine.mean():.1f}%")
        axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

        fig.tight_layout()

    return _savefig(fig, "02_failure_distribution.png")


# ─────────────────────────────────────────
# 3. Correlation heatmap
# ─────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level", "failure"]
    corr = df[sensor_cols].corr()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="#333",
            ax=ax, annot_kws={"size": 10}
        )
        ax.set_title("Sensor Feature Correlation Heatmap", color="white",
                     fontweight="bold", pad=10)
        ax.tick_params(colors="white")
        fig.tight_layout()

    return _savefig(fig, "03_correlation_heatmap.png")


# ─────────────────────────────────────────
# 4. Model comparison bar chart
# ─────────────────────────────────────────
def plot_model_comparison(all_metrics: list) -> str:
    models   = [m["model"] for m in all_metrics]
    accuracy = [m["accuracy"]  for m in all_metrics]
    f1       = [m["f1"]        for m in all_metrics]
    recall   = [m["recall"]    for m in all_metrics]
    roc_auc  = [m["roc_auc"]   for m in all_metrics]

    x = np.arange(len(models))
    width = 0.2

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - 1.5*width, accuracy, width, label="Accuracy", color=PALETTE[0])
        ax.bar(x - 0.5*width, f1,       width, label="F1 Score", color=PALETTE[2])
        ax.bar(x + 0.5*width, recall,   width, label="Recall",   color=PALETTE[1])
        ax.bar(x + 1.5*width, roc_auc,  width, label="ROC-AUC",  color=PALETTE[3])

        ax.set_xlabel("Model", color="white")
        ax.set_ylabel("Score", color="white")
        ax.set_title("Model Performance Comparison", color="white", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.set_ylim(0, 1.1)
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.set_facecolor("#1a1a2e")
        ax.grid(axis="y", alpha=0.2, color="gray")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
        fig.tight_layout()

    return _savefig(fig, "04_model_comparison.png")


# ─────────────────────────────────────────
# 5. Confusion matrix
# ─────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Best Model") -> str:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            linewidths=1, linecolor="#333", ax=ax,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_xlabel("Predicted Label", color="white")
        ax.set_ylabel("True Label", color="white")
        ax.set_title(f"Confusion Matrix — {model_name}", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        fig.tight_layout()

    return _savefig(fig, "05_confusion_matrix.png")


# ─────────────────────────────────────────
# 6. ROC curve
# ─────────────────────────────────────────
def plot_roc_curve(model, X_test, y_test) -> str:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=PALETTE[0], lw=2,
                label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE[0])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", color="white")
        ax.set_ylabel("True Positive Rate", color="white")
        ax.set_title("ROC Curve — Failure Detection", color="white", fontweight="bold")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")
        ax.grid(alpha=0.2, color="gray")
        fig.tight_layout()

    return _savefig(fig, "06_roc_curve.png")


# ─────────────────────────────────────────
# 7. Feature importance (for tree-based)
# ─────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, top_n: int = 20) -> str:
    if not hasattr(model, "feature_importances_"):
        print("[!] Model does not support feature importances — skipping plot")
        return ""

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in indices]
    top_values   = importances[indices]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_values[::-1],
                       color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], color="white", fontsize=9)
        ax.set_xlabel("Feature Importance Score", color="white")
        ax.set_title(f"Top {top_n} Feature Importances", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")
        ax.grid(axis="x", alpha=0.2, color="gray")
        fig.tight_layout()

    return _savefig(fig, "07_feature_importance.png")


# ─────────────────────────────────────────
# 8. Predicted failure probability over time
# ─────────────────────────────────────────
def plot_failure_probability(df: pd.DataFrame, machine_id: int = 1) -> str:
    if "failure_prob" not in df.columns:
        print("[!] No failure_prob column — run predictor first")
        return ""

    mdf = df[df["machine_id"] == machine_id].copy()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(mdf["timestamp"], mdf["failure_prob"],
                        alpha=0.4, color=PALETTE[1])
        ax.plot(mdf["timestamp"], mdf["failure_prob"],
                color=PALETTE[1], linewidth=1.2, label="Failure Probability")

        # Threshold lines
        ax.axhline(0.25, color=PALETTE[3], linestyle="--", linewidth=0.8, label="Warning (25%)")
        ax.axhline(0.50, color=PALETTE[0], linestyle="--", linewidth=0.8, label="Alert (50%)")
        ax.axhline(0.75, color=PALETTE[1], linestyle=":",  linewidth=1.0, label="Critical (75%)")

        # Actual failures
        fail_mask = mdf["failure"] == 1
        ax.scatter(mdf.loc[fail_mask, "timestamp"], mdf.loc[fail_mask, "failure_prob"],
                   color="white", s=15, zorder=5, label="Actual Failure")

        ax.set_xlabel("Timestamp", color="white")
        ax.set_ylabel("Failure Probability", color="white")
        ax.set_title(f"Machine #{machine_id} — Predicted Failure Probability",
                     color="white", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")
        ax.grid(alpha=0.2, color="gray")
        fig.tight_layout()

    return _savefig(fig, f"08_failure_probability_machine_{machine_id}.png")


# ─────────────────────────────────────────
# 9. Sensor box plots by failure label
# ─────────────────────────────────────────
def plot_sensor_boxplots(df: pd.DataFrame) -> str:
    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "oil_level"]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        fig.suptitle("Sensor Distributions: Normal vs Failure",
                     color="white", fontweight="bold", fontsize=13)

        for ax, col in zip(axes, sensor_cols):
            data_normal  = df[df["failure"] == 0][col]
            data_failure = df[df["failure"] == 1][col]

            bp = ax.boxplot(
                [data_normal, data_failure],
                patch_artist=True,
                labels=["Normal", "Failure"]
            )
            bp["boxes"][0].set_facecolor(PALETTE[0])
            bp["boxes"][1].set_facecolor(PALETTE[1])
            for element in ["whiskers", "caps", "medians", "fliers"]:
                plt.setp(bp[element], color="white")

            ax.set_title(col.replace("_", " ").title(), color="white", fontsize=10)
            ax.tick_params(colors="white")
            ax.set_facecolor("#1a1a2e")
            ax.grid(axis="y", alpha=0.2, color="gray")

        fig.tight_layout()

    return _savefig(fig, "09_sensor_boxplots.png")


# ─────────────────────────────────────────
# 10. Dashboard summary (combined figure)
# ─────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, all_metrics: list = None) -> str:
    """Quick 4-panel dashboard for README demo."""
    sensor_cols = ["temperature", "vibration", "pressure", "oil_level"]
    mdf = df[df["machine_id"] == 1].copy()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("AI Predictive Maintenance — Dashboard Overview",
                     color="white", fontweight="bold", fontsize=14)

        for ax, (col, color) in zip(axes.flat, zip(sensor_cols, PALETTE)):
            ax.plot(mdf["timestamp"], mdf[col], color=color, linewidth=0.8)
            fail_m = mdf["failure"] == 1
            ax.scatter(mdf.loc[fail_m, "timestamp"], mdf.loc[fail_m, col],
                       color=PALETTE[1], s=15, zorder=5)
            ax.set_title(col.replace("_", " ").title(), color="white")
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=7)
            ax.grid(alpha=0.15, color="gray")

        fig.tight_layout()

    return _savefig(fig, "10_dashboard.png")


# ─────────────────────────────────────────
# Run all visualizations at once
# ─────────────────────────────────────────
def generate_all_plots(
    df: pd.DataFrame,
    model=None,
    X_test=None,
    y_test=None,
    feature_cols: list = None,
    all_metrics: list = None,
) -> None:
    print("\n[→] Generating all visualizations...")
    plot_sensor_timeseries(df)
    plot_failure_distribution(df)
    plot_correlation_heatmap(df)
    plot_sensor_boxplots(df)
    plot_dashboard(df)

    if all_metrics:
        plot_model_comparison(all_metrics)

    if model and X_test is not None and y_test is not None:
        plot_confusion_matrix(model, X_test, y_test)
        plot_roc_curve(model, X_test, y_test)

    if model and feature_cols:
        plot_feature_importance(model, feature_cols)

    if "failure_prob" in df.columns:
        plot_failure_probability(df)

    print(f"[✓] All plots saved to → {SAVE_DIR}/\n")
