"""트리 기반 모델(XGBoost 등) 학습용 피처 엔지니어링 모듈.

딥러닝 모델은 raw 시계열을 그대로 받지만, XGBoost·RandomForest 같은 테이블형
알고리즘에는 "어떤 과거 값을 쓸지" + "시각의 주기성"을 수치로 떠먹여야 하므로
아래 헬퍼들을 조합해 입력 테이블을 만듭니다.

파일 구조
---------
1. `add_time_features(df)`
   datetime 인덱스로부터 시/일의 sin·cos 인코딩을 추가. 23시와 0시가
   서로 가깝다는 **순환 구조**를 모델에 알려줍니다.

2. `add_lag_features(df, target, lags)`
   타깃의 과거 값(`target.shift(lag)`)을 새 컬럼으로 추가.
   기본 lag는 `(1, 24, 168)` — 1시간 전, 1일 전, 1주일 전.

3. `make_supervised_table(df, target, horizon, lags)`
   위 둘을 합성해 학습용 `(X, y)` 테이블을 만듭니다.
   `y = target.shift(-horizon)` 로 미래 값을 타깃화하고, 시프트로 생긴
   NaN은 `dropna()`로 제거합니다(**XGBoost는 NaN을 수용하므로 lag NaN은
   유지**, shift로 미래가 없는 꼬리 행만 제거).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """datetime 인덱스의 hour-of-day, day-of-year을 sin/cos로 인코딩합니다.

    시간은 본질적으로 순환적이므로 단순 정수 인코딩(0~23)은 부적절합니다.
    (예: 23시와 0시는 수치상 가장 멀지만 실제 의미는 인접) 단위원 위의
    (sin, cos) 좌표로 바꾸면 이 인접성을 자연스럽게 표현할 수 있습니다.

    추가되는 컬럼:
        - `hour_sin`, `hour_cos` : 24시간 주기
        - `doy_sin`, `doy_cos`   : 365.25일 주기(계절성)

    Args:
        df: datetime 인덱스를 가진 DataFrame. 원본은 수정되지 않습니다.

    Returns:
        원본 컬럼 + 4개 인코딩 컬럼이 추가된 복사본.
    """
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
    """`target` 컬럼의 지정 lag들을 새 컬럼으로 추가합니다.

    `target.shift(lag)`는 `lag` 스텝 전의 값을 현재 행에 실어주므로,
    기본값 `(1, 24, 168)`은 각각 "1시간 전 / 1일 전 / 1주일 전"을 뜻합니다
    (시간당 1행 전제). 앞쪽 `max(lags)`개 행은 shift로 NaN이 됩니다.

    Args:
        df: 원본 DataFrame. 수정되지 않습니다.
        target: lag를 생성할 대상 컬럼 이름.
        lags: 생성할 lag 스텝 수들의 시퀀스.

    Returns:
        `{target}_lag{k}` 컬럼들이 추가된 복사본.
    """
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
    """XGBoost용 학습 테이블 `(X, y)`를 만듭니다.

    처리 순서:
        1. `add_time_features` 로 시/일 sin·cos 컬럼 부착
        2. `add_lag_features` 로 타깃의 lag 컬럼 부착
        3. `y = target.shift(-horizon)` — `horizon` 스텝 뒤의 값을 타깃으로
        4. `dropna()` — shift로 발생한 맨 앞(lag)과 맨 뒤(horizon) NaN 제거

    단일 horizon만 지원합니다(multi-step은 별도 처리). XGBoost는 중간 NaN을
    스스로 분기할 수 있지만, 여기서는 학습 안정성을 위해 통째로 제거합니다.

    Args:
        df: 시간당 1행 DataFrame(보통 `downsample_hourly` 결과).
        target: 예측 대상 컬럼 이름(예: `"T (degC)"`).
        horizon: 몇 스텝 뒤 값을 예측할지(예: 24 = 24시간 뒤).
        lags: 사용할 lag 스텝 리스트.

    Returns:
        `(X, y)` — X는 피처 테이블, y는 horizon 스텝 뒤 target 값.
    """
    feat = add_time_features(df)
    feat = add_lag_features(feat, target=target, lags=lags)
    feat["y"] = df[target].shift(-horizon)
    feat = feat.dropna()
    y = feat["y"]
    X = feat.drop(columns=["y"])
    return X, y
