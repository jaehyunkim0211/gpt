"""시계열 전처리 유틸리티: 리샘플링·분할·스케일링·윈도잉.

이 모듈은 데이터 로더와 모델 사이의 **정제 단계**를 담당합니다.
순수 numpy/pandas 연산만 사용하므로 GPU/torch 의존성이 없습니다.

파일 구조
---------
1. `downsample_hourly(df)`
   10분 간격 원본 → 1시간 평균 + 시간 기반 선형 보간으로 gap 채우기.
   (Naive·SARIMA·LSTM이 NaN을 만나면 학습 지표가 모두 NaN이 되는 사고 방지)

2. `time_split(df, ratios)`
   시간 순서를 유지한 채 train/val/test로 연속 블록 분할.
   셔플을 하지 않으므로 future leakage가 생기지 않습니다.

3. `StandardScaler1D`
   sklearn 없이 쓸 수 있는 얇은 numpy 스케일러.
   fit·transform·inverse_transform을 지원하여 LSTM 출력 역변환에 사용.

4. `make_supervised_windows(series, input_len, output_len, ...)`
   2D 시계열 `(T, F)`를 슬라이딩 윈도우로 잘라
   `X.shape=(N, input_len, F_in)`, `y.shape=(N, output_len)`로 변환.
   LSTM 입력 텐서 준비에 사용됩니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def downsample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """10분 간격 Jena 데이터를 1시간 평균으로 다운샘플링하고 gap을 보간합니다.

    원본에는 수 시간~3일 규모의 gap이 몇 군데 있습니다. 단순 `resample("1h").mean()`은
    gap 위치에 NaN 행을 만들고, 이후 Naive·SARIMA·LSTM이 이 NaN을 만나면 출력 전체가
    NaN이 되어 지표가 계산되지 않는 사고가 발생합니다. 따라서 리샘플링 직후
    **시간 기반 선형 보간**(`method="time"`)을 적용하여 인덱스가 끊김 없이 이어지도록
    만듭니다. 양 끝단 NaN은 `limit_direction="both"`로 동일하게 채웁니다.

    Args:
        df: datetime 인덱스를 가진 10분 간격 Jena DataFrame.

    Returns:
        시간당 1행, NaN 없는 DataFrame.
    """
    hourly = df.resample("1h").mean()
    hourly = hourly.interpolate(method="time", limit_direction="both")
    return hourly


def time_split(df: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """시계열 데이터를 시간순으로 train/val/test 연속 구간 3개로 분할합니다.

    **셔플 금지** — 시계열 예측에서 future → past leakage를 막기 위해 항상
    인덱스 순서를 유지합니다. 원본 DataFrame을 수정하지 않도록 각 슬라이스는
    `.copy()`로 반환됩니다.

    Args:
        df: 시간순 정렬된 DataFrame.
        ratios: `(train, val, test)` 비율. 합이 1에 가깝지 않아도 무방하며
            (예: `(0.7, 0.1, 0.15)`), 남은 부분은 test 쪽에 붙습니다.

    Returns:
        `(train_df, val_df, test_df)` 튜플.
    """
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
    """2D 배열 `(N, F)`에 대해 피처별 평균·표준편차로 정규화하는 얇은 스케일러.

    sklearn의 `StandardScaler`와 역할이 같지만 의존성을 줄이고 numpy만으로
    동작하도록 작성했습니다. 분모 0 방지를 위해 `std`에 `1e-8`을 더합니다.
    `inverse_transform`이 있어 LSTM 예측값을 원래 단위(°C)로 되돌리는 데
    사용됩니다.

    Attributes:
        mean_: `fit` 후 계산된 피처별 평균, shape `(F,)`.
        std_:  `fit` 후 계산된 피처별 표준편차 + epsilon, shape `(F,)`.
    """

    mean_: np.ndarray = None
    std_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "StandardScaler1D":
        """`X.mean(0)`와 `X.std(0) + eps`를 계산하여 내부 상태에 저장합니다.

        Args:
            X: 2D 배열 `(N, F)`.

        Returns:
            자기 자신(체이닝용).
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """`fit`된 통계로 `X`를 표준화합니다. `fit` 미호출 시 AttributeError."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """`fit(X).transform(X)`의 편의 메서드."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """표준화된 값을 원래 스케일로 되돌립니다.

        LSTM이 스케일된 공간에서 예측하므로, 성능 지표 계산 전 이 함수로
        역변환해 실제 단위(°C, mbar 등)로 만들어야 MAE 해석이 가능합니다.
        """
        return X * self.std_ + self.mean_


def make_supervised_windows(
    series: np.ndarray,
    input_len: int,
    output_len: int,
    feature_cols: Sequence[int] | None = None,
    target_col: int = 0,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """2D 시계열을 슬라이딩 윈도우로 잘라 supervised (X, y) 쌍을 만듭니다.

    각 윈도우는 연속된 `input_len + output_len` 타임스텝을 차지하며,
    앞쪽 `input_len`은 입력으로, 뒤쪽 `output_len`은 예측 목표로 사용됩니다.
    윈도우는 `step` 간격으로 전진합니다.

    Args:
        series: 2D numpy 배열, shape `(T, F)`.
        input_len: 입력 윈도우 길이(=과거 몇 스텝을 볼지).
        output_len: 예측 호라이즌(=미래 몇 스텝을 맞힐지).
        feature_cols: 입력으로 쓸 컬럼 인덱스 리스트. None이면 전체 F개 사용.
        target_col: 예측 대상 컬럼 인덱스(단일 타깃만 지원).
        step: 연속 윈도우 간 stride. 1이면 최대 밀도로 샘플을 생성.

    Returns:
        `X.shape = (N, input_len, len(feature_cols))`,
        `y.shape = (N, output_len)`.  둘 다 `dtype=float32`.

    Raises:
        ValueError: `series.ndim != 2` 인 경우.
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
