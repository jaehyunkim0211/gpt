"""Plotting utilities including random sample viewers used in EDA notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str | None) -> None:
    if name:
        fig.tight_layout()
        fig.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")


# ───────────────────────── Random sample viewers ─────────────────────────

def plot_random_climate_window(
    df: pd.DataFrame,
    days: int = 7,
    columns: Sequence[str] = ("T (degC)", "rh (%)", "wv (m/s)", "p (mbar)"),
    seed: int | None = None,
    save_as: str | None = None,
):
    """Pick a random `days`-long window and plot the given columns."""
    rng = np.random.default_rng(seed)
    span = pd.Timedelta(days=days)
    valid_starts = df.index[df.index <= df.index.max() - span]
    start = valid_starts[rng.integers(0, len(valid_starts))]
    window = df.loc[start : start + span]

    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 2.2 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        ax.plot(window.index, window[col], lw=0.9)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"Random {days}-day window starting {start}")
    axes[-1].set_xlabel("time")
    _save(fig, save_as)
    return fig


def plot_random_har_samples(
    X: np.ndarray,
    y: np.ndarray,
    label_names: dict[int, str],
    channel_names: Sequence[str],
    n: int = 5,
    seed: int | None = None,
    save_as: str | None = None,
):
    """Plot n randomly chosen HAR windows. Each subplot shows all 9 channels overlaid."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=n, replace=False)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for k, idx in enumerate(idxs):
        ax = axes[k // cols][k % cols]
        for c in range(X.shape[-1]):
            ax.plot(X[idx, :, c], lw=0.7, label=channel_names[c] if k == 0 else None)
        ax.set_title(f"#{idx} · {label_names.get(int(y[idx]), str(y[idx]))}", fontsize=9)
        ax.set_xticks([])
        ax.grid(alpha=0.3)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"Random {n} HAR windows (9 channels overlaid)", y=1.02)
    _save(fig, save_as)
    return fig


def plot_random_har_class_signals(
    X: np.ndarray,
    y: np.ndarray,
    label_names: dict[int, str],
    channel_names: Sequence[str],
    seed: int | None = None,
    save_as: str | None = None,
):
    """Pick a random class, then a random window of that class, plot all 9 channels stacked."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    cls = int(classes[rng.integers(0, len(classes))])
    candidates = np.where(y == cls)[0]
    idx = int(candidates[rng.integers(0, len(candidates))])

    n_ch = X.shape[-1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 1.2 * n_ch), sharex=True)
    for c in range(n_ch):
        axes[c].plot(X[idx, :, c], lw=0.9)
        axes[c].set_ylabel(channel_names[c], fontsize=8)
        axes[c].grid(alpha=0.3)
    axes[0].set_title(f"Random window #{idx} · class={label_names.get(cls, cls)}")
    axes[-1].set_xlabel("timestep")
    _save(fig, save_as)
    return fig


def plot_har_class_distribution(
    y: np.ndarray, label_names: dict[int, str], title: str = "HAR class distribution",
    save_as: str | None = None,
):
    counts = pd.Series(y).map(label_names).value_counts()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    _save(fig, save_as)
    return fig


def plot_har_mean_signal_heatmap(
    X: np.ndarray, y: np.ndarray, label_names: dict[int, str],
    channel_names: Sequence[str], save_as: str | None = None,
):
    """For each (class, channel), compute mean absolute signal → heatmap."""
    classes = sorted(np.unique(y).tolist())
    rows = []
    for cls in classes:
        mean_abs = np.abs(X[y == cls]).mean(axis=(0, 1))  # (C,)
        rows.append(mean_abs)
    mat = np.stack(rows, axis=0)
    df = pd.DataFrame(mat, index=[label_names[c] for c in classes], columns=channel_names)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title("Mean |signal| per class × channel")
    _save(fig, save_as)
    return fig


# ───────────────────────── Forecasting plots ─────────────────────────

def plot_forecast_vs_actual(
    timestamps, actual: np.ndarray, predictions: dict[str, np.ndarray],
    title: str = "Forecast vs actual", save_as: str | None = None,
):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamps, actual, label="actual", color="black", lw=1.2)
    for name, pred in predictions.items():
        ax.plot(timestamps, pred, label=name, lw=1.0, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("T (degC)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, save_as)
    return fig


def plot_metric_bars(metrics: dict[str, dict[str, float]], metric_name: str = "MAE",
                     save_as: str | None = None):
    names = list(metrics.keys())
    vals = [metrics[n][metric_name] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, vals, color="steelblue")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by model (lower is better)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, save_as)
    return fig


# ───────────────────────── Classification plots ─────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray, label_names: list[str], title: str = "Confusion matrix",
    save_as: str | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names,
                yticklabels=label_names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    _save(fig, save_as)
    return fig


def plot_training_curves(history, title: str = "Training curves", save_as: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history.train_loss, label="train_loss")
    ax.plot(history.val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, save_as)
    return fig
