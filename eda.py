"""
eda.py — Exploratory Data Analysis Module
AutoML Document Analyzer
Performs automated EDA: statistics, missing values, distributions, correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from io import BytesIO


# ─────────────────────────────────────────────
# Palette & style helpers
# ─────────────────────────────────────────────
_PALETTE = ["#4F8EF7", "#F7874F", "#4FF7A8", "#F74F8E", "#A84FF7", "#F7D24F"]

def _apply_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent dark-style formatting to a matplotlib Axes."""
    ax.set_facecolor("#1A1A2E")
    ax.set_title(title, color="#E0E0FF", fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color="#9090B0", fontsize=9)
    ax.set_ylabel(ylabel, color="#9090B0", fontsize=9)
    ax.tick_params(colors="#9090B0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2E4E")


def _dark_fig(figsize=(10, 5)):
    """Return a dark-background figure."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#12122A")
    return fig, ax


# ─────────────────────────────────────────────
# Core EDA functions
# ─────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame) -> dict:
    """Return basic dataset metadata."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column summary: dtype, missing count/%, unique values,
    and for numeric columns: mean, std, min, max.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isnull().sum())
        row = {
            "Column": col,
            "Dtype": str(series.dtype),
            "Missing": n_missing,
            "Missing %": round(n_missing / len(df) * 100, 1),
            "Unique": series.nunique(),
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update({
                "Mean": round(series.mean(), 4),
                "Std": round(series.std(), 4),
                "Min": round(series.min(), 4),
                "Max": round(series.max(), 4),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() rounded to 4 decimal places."""
    return df.describe(include="all").T.round(4)


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame):
    """Horizontal bar chart of missing value counts."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None

    fig, ax = _dark_fig(figsize=(8, max(3, len(missing) * 0.5)))
    bars = ax.barh(missing.index.tolist(), missing.values,
                   color=_PALETTE[0], edgecolor="#2E2E4E", linewidth=0.5)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(width)}", va="center", ha="left",
                color="#E0E0FF", fontsize=8)
    _apply_style(ax, title="Missing Values per Column",
                 xlabel="Count", ylabel="Column")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap for numeric columns."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()
    size = max(6, corr.shape[0] * 0.8)
    fig, ax = _dark_fig(figsize=(size, size))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.4, linecolor="#2E2E4E",
        annot_kws={"size": 8, "color": "#E0E0FF"},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_facecolor("#1A1A2E")
    ax.tick_params(colors="#9090B0", labelsize=8)
    ax.set_title("Correlation Heatmap", color="#E0E0FF", fontsize=11, pad=10)
    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, max_cols: int = 8):
    """Distribution (KDE + histogram) for each numeric column."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:max_cols]
    if not num_cols:
        return None

    n = len(num_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             facecolor="#12122A")
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, color=_PALETTE[i % len(_PALETTE)],
                alpha=0.55, edgecolor="#2E2E4E", linewidth=0.3, density=True)
        try:
            data.plot.kde(ax=ax, color="#FFFFFF", linewidth=1.5)
        except Exception:
            pass
        _apply_style(ax, title=col, xlabel="Value", ylabel="Density")

    # Hide unused axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", color="#E0E0FF",
                 fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_value_counts(df: pd.DataFrame, max_cols: int = 6):
    """Bar plots of top-10 value counts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()[:max_cols]
    if not cat_cols:
        return None

    n = len(cat_cols)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4 * nrows),
                             facecolor="#12122A")
    axes = np.array(axes).flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        counts = df[col].value_counts().head(10)
        bars = ax.bar(range(len(counts)), counts.values,
                      color=_PALETTE[i % len(_PALETTE)],
                      edgecolor="#2E2E4E", linewidth=0.4)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str),
                           rotation=30, ha="right",
                           color="#9090B0", fontsize=7)
        _apply_style(ax, title=f"{col} — Top Values",
                     xlabel="", ylabel="Count")

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Value Counts", color="#E0E0FF",
                 fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_target_distribution(df: pd.DataFrame, target: str):
    """Distribution / count plot of the chosen target column."""
    series = df[target].dropna()
    fig, ax = _dark_fig(figsize=(8, 4))

    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series, bins=30, color=_PALETTE[0],
                alpha=0.7, edgecolor="#2E2E4E")
        _apply_style(ax, title=f"Target Distribution: {target}",
                     xlabel=target, ylabel="Count")
    else:
        counts = series.value_counts().head(20)
        ax.bar(range(len(counts)), counts.values,
               color=_PALETTE[0], edgecolor="#2E2E4E")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str),
                           rotation=30, ha="right",
                           color="#9090B0", fontsize=8)
        _apply_style(ax, title=f"Target Class Counts: {target}",
                     xlabel="", ylabel="Count")

    plt.tight_layout()
    return fig
