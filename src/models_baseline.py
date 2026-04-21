"""Baseline models for forecasting and classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat last value `horizon` times. history shape (T,) → (horizon,)."""
    return np.full(horizon, history[-1], dtype=np.float32)


def fit_xgb_regressor(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> xgb.XGBRegressor:
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    )
    params.update(kwargs)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def fit_xgb_classifier(X_train, y_train, n_classes: int, **kwargs) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        random_state=42,
    )
    params.update(kwargs)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train, y_train, **kwargs) -> RandomForestClassifier:
    params = dict(n_estimators=300, n_jobs=-1, random_state=42)
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
