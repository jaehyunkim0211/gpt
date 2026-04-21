"""비-딥러닝 베이스라인 모델 팩토리 모듈.

노트북에서 반복되는 기본 하이퍼파라미터를 한 곳에 모으고, 실험마다 노트북을
다시 읽지 않아도 되도록 얇은 래퍼만 제공합니다. 실제 학습은 scikit-learn /
XGBoost가 수행하며, 이 함수들은 "기본값 + `**kwargs` 오버라이드 + `.fit` 호출"
패턴을 따릅니다.

파일 구조
---------
1. `naive_forecast(history, horizon)`
   시계열 예측의 최소 베이스라인 — 마지막 관측값을 `horizon`번 반복.

2. `fit_xgb_regressor(X_train, y_train, **kwargs)`
   회귀용 XGBoost. Jena 기온 multi-step 예측에 사용.

3. `fit_xgb_classifier(X_train, y_train, n_classes, **kwargs)`
   다중 클래스 분류용 XGBoost. UCI HAR의 561 feature에 사용.

4. `fit_random_forest(X_train, y_train, **kwargs)`
   분류용 RandomForest. XGBoost와 비교하는 두 번째 트리 베이스라인.

모든 팩토리는 `random_state=42`로 고정되어 재현성이 보장됩니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """직전 관측값을 `horizon` 스텝 동안 반복하는 최소 베이스라인.

    Naive는 "더 나은 모델이 이것보다 낮은 MAE를 달성해야 함"을 보여주는
    하한선 역할을 합니다. 특히 trend가 완만하고 자기상관이 강한 시계열
    (예: 기온)에서는 Naive가 의외로 높은 기준선이 됩니다.

    Args:
        history: 과거 관측 시계열, 1D array shape `(T,)`. 마지막 원소만 사용.
        horizon: 예측할 미래 스텝 수.

    Returns:
        shape `(horizon,)` float32 배열, 모든 값이 `history[-1]`.
    """
    return np.full(horizon, history[-1], dtype=np.float32)


def fit_xgb_regressor(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> xgb.XGBRegressor:
    """기본 하이퍼파라미터로 XGBoost 회귀 모델을 학습합니다.

    기본 파라미터(노트북 경험값):
        - `n_estimators=400` : 400그루, 성능·속도 균형
        - `max_depth=6`       : 과적합 억제
        - `learning_rate=0.05`: 낮은 lr + 많은 tree
        - `subsample=0.8`, `colsample_bytree=0.8` : stochastic 학습
        - `tree_method="hist"` : 히스토그램 기반, CPU에서도 빠름
        - `random_state=42`   : 재현성

    `**kwargs` 로 임의의 XGBoost 인자를 오버라이드할 수 있습니다
    (예: `fit_xgb_regressor(Xtr, ytr, n_estimators=800)`).

    Args:
        X_train: 피처 테이블 `(N, F)`.
        y_train: 타깃 시리즈 `(N,)`.
        **kwargs: `XGBRegressor` 생성자 인자 오버라이드.

    Returns:
        학습이 완료된 `xgb.XGBRegressor` 인스턴스.
    """
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
    """기본 하이퍼파라미터로 XGBoost 다중 클래스 분류기를 학습합니다.

    - `objective="multi:softprob"` : 클래스별 확률 출력
    - `num_class=n_classes`        : 클래스 수 지정 필수
    - `learning_rate=0.1`, `n_estimators=400` : 회귀보다 약간 공격적

    `y_train`은 **0-indexed 정수**여야 합니다(UCI HAR 원본은 1-indexed이므로
    `y - 1` 로 변환 후 전달).

    Args:
        X_train: 피처 배열/DataFrame `(N, F)`.
        y_train: 정수 라벨 `(N,)`, 값은 `[0, n_classes-1]` 범위.
        n_classes: 전체 클래스 수.
        **kwargs: `XGBClassifier` 생성자 인자 오버라이드.

    Returns:
        학습된 `xgb.XGBClassifier`.
    """
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
    """RandomForest 분류기를 기본 하이퍼파라미터로 학습합니다.

    - `n_estimators=300` : 트리 수
    - `n_jobs=-1`        : 모든 CPU 코어 사용(학습·추론 병렬화)
    - `random_state=42`  : 재현성

    XGBoost와 달리 전처리·스케일링·NaN 처리가 추가로 필요할 수 있지만,
    UCI HAR의 561개 handcrafted feature는 이미 정제되어 있어 그대로 입력합니다.

    Args:
        X_train: 피처 배열 `(N, F)`.
        y_train: 정수 라벨 `(N,)`.
        **kwargs: `RandomForestClassifier` 생성자 인자 오버라이드.

    Returns:
        학습된 `RandomForestClassifier`.
    """
    params = dict(n_estimators=300, n_jobs=-1, random_state=42)
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
