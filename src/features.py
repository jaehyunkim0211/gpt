"""Lag and time-encoding features for baseline models."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sin/cos encodings of hour-of-day and day-of-year."""
    out = df.copy()
    hour = out.index.hour + out.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = out.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: Sequence[int] = (1, 24, 168),
) -> pd.DataFrame:
    """Append lagged copies of `target` column."""
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    return out


def make_supervised_table(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    lags: Sequence[int] = (1, 24, 168),
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a flat (X, y) table for tree-based forecasting.

    y is the target shifted -horizon (predict `horizon` steps ahead).
    """
    feat = add_time_features(df)
    feat = add_lag_features(feat, target=target, lags=lags)
    feat["y"] = df[target].shift(-horizon)
    feat = feat.dropna()
    y = feat["y"]
    X = feat.drop(columns=["y"])
    return X, y
