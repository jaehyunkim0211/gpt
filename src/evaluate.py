"""회귀·분류 평가 지표 계산 모듈.

노트북은 이 모듈의 함수만 호출하여 평가 로직을 통일합니다. 회귀는 MAE/RMSE/MAPE
세 지표를, 분류는 accuracy / macro-F1 / confusion matrix / sklearn 텍스트 리포트를
하나의 dict로 반환합니다.

파일 구조
---------
1. 스칼라 회귀 지표 : `mae`, `rmse`, `mape`
2. `regression_report(y_true, y_pred)`       : 세 지표를 dict로 묶음
3. `classification_summary(y_true, y_pred)`  : 분류 결과 종합 dict

MAPE 주의: Jena 기온은 0°C를 통과하므로 분모가 작아져 MAPE가 과대 계산될 수
있습니다. 이 경우 MAE/RMSE가 더 신뢰할 만한 지표입니다.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. 단위는 타깃과 동일합니다.

    Args:
        y_true: 실제값 배열.
        y_pred: 예측값 배열. shape은 `y_true`와 동일해야 합니다.

    Returns:
        모든 원소에 대한 `|y_true - y_pred|`의 평균, 파이썬 float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. 큰 오차에 더 민감합니다(제곱 후 루트).

    MAE와 함께 보면 "오차 분포의 꼬리 두께"를 가늠할 수 있습니다
    (RMSE / MAE 비율이 1에 가까우면 오차가 균일, 크면 몇몇 큰 오차가 존재).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%). 0 분모 방지를 위해 `eps`를 더합니다.

    주의: `y_true`가 0 근처 값을 가지면 폭발적으로 커집니다. 섭씨 기온처럼
    0을 통과하는 지표에서는 MAE를 우선으로 보세요.

    Args:
        y_true: 실제값 배열.
        y_pred: 예측값 배열.
        eps: `|y_true| + eps`로 분모 보정. 기본 `1e-6`.

    Returns:
        백분율 값(예: 12.5 → 12.5%).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """세 가지 회귀 지표를 한 번에 계산하여 dict로 반환합니다.

    Returns:
        `{"MAE": ..., "RMSE": ..., "MAPE": ...}`.
        노트북이 DataFrame으로 변환하기 좋게 문자열 키를 사용합니다.
    """
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, label_names: dict | None = None):
    """분류 결과의 주요 지표를 계산하여 dict로 묶어 반환합니다.

    반환 dict:
        - `accuracy`         : 정확도(0~1).
        - `macro_f1`         : 클래스별 F1의 산술 평균(클래스 불균형 보정).
        - `confusion_matrix` : `(K, K)` 정수 배열. 행=실제, 열=예측.
        - `labels`           : 등장한 클래스 정수 리스트, 정렬됨.
        - `report`           : sklearn `classification_report` 문자열.

    Args:
        y_true: 정수 라벨 `(N,)`.
        y_pred: 정수 예측 `(N,)`.
        label_names: `{int: str}` 매핑. 주어지면 report에 이름으로 표시.

    Returns:
        위 키를 가진 dict.
    """
    target_names = None
    # y_true / y_pred에 **실제 등장한** 라벨만 뽑아 정렬.
    # (전체 라벨 맵이 아니라 실제 데이터에 나타난 라벨만 써야
    #  confusion_matrix·report의 shape/순서가 일관되게 유지됩니다.)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if label_names is not None:
        target_names = [label_names[i] for i in labels]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
        "report": classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=3),
    }
