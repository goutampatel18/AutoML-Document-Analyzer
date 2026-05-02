"""
model.py — Model Training & Evaluation Module
AutoML Document Analyzer
Trains multiple models, evaluates them, selects the best one.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score,
)
import joblib
from io import BytesIO
import json


# ─────────────────────────────────────────────
# Palette / style helpers  (mirrors eda.py)
# ─────────────────────────────────────────────
_PALETTE = ["#4F8EF7", "#F7874F", "#4FF7A8", "#F74F8E", "#A84FF7", "#F7D24F"]

def _dark_fig(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#12122A")
    return fig, ax

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1A1A2E")
    ax.set_title(title, color="#E0E0FF", fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color="#9090B0", fontsize=9)
    ax.set_ylabel(ylabel, color="#9090B0", fontsize=9)
    ax.tick_params(colors="#9090B0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2E4E")


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="saga"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_models(X_train, y_train, problem_type: str) -> dict:
    """
    Train all models for the given problem type.
    Returns dict: { model_name: fitted_estimator }
    """
    catalog = (CLASSIFICATION_MODELS if problem_type == "classification"
               else REGRESSION_MODELS)
    trained = {}
    for name, clf in catalog.items():
        clf.fit(X_train, y_train)
        trained[name] = clf
    return trained


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_models(trained: dict, X_test, y_test,
                    problem_type: str) -> tuple[dict, str]:
    """
    Evaluate all trained models on the test set.

    Returns:
      results  — dict of { model_name: { metric: value, ... } }
      best_name — name of the best-performing model
    """
    results = {}

    for name, model in trained.items():
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            avg = "weighted"
            results[name] = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(
                    y_test, y_pred, average=avg, zero_division=0), 4),
                "Recall": round(recall_score(
                    y_test, y_pred, average=avg, zero_division=0), 4),
                "F1 Score": round(f1_score(
                    y_test, y_pred, average=avg, zero_division=0), 4),
                "_cm": confusion_matrix(y_test, y_pred),
                "_y_pred": y_pred,
            }
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2   = float(r2_score(y_test, y_pred))
            results[name] = {
                "RMSE": round(rmse, 4),
                "R² Score": round(r2, 4),
                "_y_pred": y_pred,
            }

    # ── Pick best model ────────────────────────────────────────────────────
    if problem_type == "classification":
        best_name = max(results, key=lambda n: results[n]["F1 Score"])
    else:
        # Best R²; if negative prefer least negative
        best_name = max(results, key=lambda n: results[n]["R² Score"])

    return results, best_name


# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_labels=None):
    """Annotated confusion-matrix heatmap."""
    fig, ax = _dark_fig(figsize=(6, 5))
    labels = class_labels if class_labels is not None else list(range(cm.shape[0]))
    sns.heatmap(
        cm, ax=ax, annot=True, fmt="d",
        cmap="Blues", linewidths=0.5, linecolor="#2E2E4E",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 10, "color": "#E0E0FF"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_facecolor("#1A1A2E")
    ax.tick_params(colors="#9090B0", labelsize=8)
    ax.set_title("Confusion Matrix", color="#E0E0FF", fontsize=11, pad=10)
    ax.set_xlabel("Predicted", color="#9090B0", fontsize=9)
    ax.set_ylabel("Actual", color="#9090B0", fontsize=9)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """
    Horizontal bar chart of feature importances.
    Works for tree-based models (feature_importances_) and linear
    models (coef_). Returns None if neither attribute is available.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    # Sort and clip
    idx = np.argsort(importances)[::-1][:top_n]
    names  = [feature_names[i] for i in idx][::-1]
    values = importances[idx][::-1]

    fig, ax = _dark_fig(figsize=(8, max(4, len(names) * 0.4)))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
    bars = ax.barh(names, values, color=colors, edgecolor="#2E2E4E", linewidth=0.3)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left",
                color="#E0E0FF", fontsize=7)

    _style_ax(ax, title=f"Top {top_n} Feature Importances",
              xlabel="Importance", ylabel="")
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results: dict, problem_type: str):
    """
    Side-by-side bar chart comparing metrics across models.
    """
    metric_keys = (
        ["Accuracy", "Precision", "Recall", "F1 Score"]
        if problem_type == "classification"
        else ["R² Score"]          # RMSE scale differs, shown separately
    )

    model_names = list(results.keys())
    n_metrics = len(metric_keys)
    x = np.arange(len(model_names))
    width = 0.7 / n_metrics

    fig, ax = _dark_fig(figsize=(8, 5))
    for i, metric in enumerate(metric_keys):
        vals = [results[m].get(metric, 0) for m in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width,
               label=metric, color=_PALETTE[i],
               edgecolor="#2E2E4E", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color="#9090B0", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1A1A2E",
              labelcolor="#E0E0FF", edgecolor="#2E2E4E")
    _style_ax(ax, title="Model Performance Comparison",
              xlabel="Model", ylabel="Score")
    plt.tight_layout()
    return fig


def plot_regression_scatter(y_test, y_pred, model_name: str):
    """Actual vs Predicted scatter for regression."""
    fig, ax = _dark_fig(figsize=(6, 5))
    ax.scatter(y_test, y_pred, color=_PALETTE[0], alpha=0.5,
               edgecolors="#2E2E4E", linewidth=0.3, s=25)
    lo = min(float(np.min(y_test)), float(np.min(y_pred)))
    hi = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="#F7D24F", linewidth=1.5, linestyle="--")
    _style_ax(ax, title=f"{model_name} — Actual vs Predicted",
              xlabel="Actual", ylabel="Predicted")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────

def save_model_bytes(model) -> bytes:
    """Serialise model to bytes using joblib."""
    buf = BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def build_report(eda_overview: dict, preprocess_report: dict,
                 results: dict, best_name: str, problem_type: str) -> str:
    """Build a JSON-serialisable analysis report string."""
    # Strip non-serialisable internals (_cm, _y_pred)
    clean_results = {}
    for name, metrics in results.items():
        clean_results[name] = {k: v for k, v in metrics.items()
                               if not k.startswith("_")}

    report = {
        "problem_type": problem_type,
        "best_model": best_name,
        "dataset_overview": eda_overview,
        "preprocessing": preprocess_report,
        "model_results": clean_results,
    }
    return json.dumps(report, indent=2, default=str)
