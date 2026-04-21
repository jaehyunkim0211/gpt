"""Preprocessing utilities: resampling, scaling, sliding windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def downsample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-minute Jena data to hourly mean.

    The raw series has a few gaps (largest ~3 days). After resampling these
    become NaN rows, which break Naive/SARIMA/LSTM downstream. We fill them
    by time-based linear interpolation so the index stays contiguous.
    """
    hourly = df.resample("1h").mean()
    hourly = hourly.interpolate(method="time", limit_direction="both")
    return hourly


def time_split(df: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """Chronological train/val/test split."""
    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train : n_train + n_val].copy(),
        df.iloc[n_train + n_val :].copy(),
    )


@dataclass
class StandardScaler1D:
    """Numpy-based scaler that fits per-feature mean/std on a 2D array."""

    mean_: np.ndarray = None
    std_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "StandardScaler1D":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_


def make_supervised_windows(
    series: np.ndarray,
    input_len: int,
    output_len: int,
    feature_cols: Sequence[int] | None = None,
    target_col: int = 0,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window across a 2D array (T, F) → (N, input_len, F_in), (N, output_len).

    Args:
        series: 2D numpy array shape (T, F).
        input_len: length of input window.
        output_len: length of forecast horizon.
        feature_cols: which columns to use as input features. None → all.
        target_col: column index of the forecast target.
        step: stride between consecutive windows.
    """
    if series.ndim != 2:
        raise ValueError(f"series must be 2D (T, F), got {series.shape}")
    T, F = series.shape
    feats = list(range(F)) if feature_cols is None else list(feature_cols)
    total_len = input_len + output_len
    n_windows = (T - total_len) // step + 1
    X = np.empty((n_windows, input_len, len(feats)), dtype=np.float32)
    y = np.empty((n_windows, output_len), dtype=np.float32)
    for i in range(n_windows):
        start = i * step
        X[i] = series[start : start + input_len, feats]
        y[i] = series[start + input_len : start + total_len, target_col]
    return X, y
