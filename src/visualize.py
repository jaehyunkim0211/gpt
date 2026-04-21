"""프로젝트 전반에서 사용하는 plot 헬퍼 모음.

모든 그림을 노트북에서 직접 그리지 않고 이 모듈을 경유하면:
    - 스타일이 일관되게 유지됨
    - PNG 저장 경로/DPI가 한 곳에서 관리됨
    - 노트북 자체가 슬림해져 로직 검토가 쉬움

파일 구조
---------
0. 공용 설정
   - `PROJECT_ROOT`, `FIG_DIR` : 저장 루트
   - `_save(fig, name)`        : DPI/레이아웃 통일 저장

1. 무작위 샘플 뷰어 (EDA 노트북용)
   - `plot_random_climate_window`     : Jena 임의 기간 다변수 line plot
   - `plot_random_har_samples`        : HAR 무작위 윈도우 n개를 격자로
   - `plot_random_har_class_signals`  : 무작위 클래스의 한 윈도우를 9채널 스택
   - `plot_har_class_distribution`    : 활동 라벨 분포 bar
   - `plot_har_mean_signal_heatmap`   : (class × channel) 평균 |신호| 히트맵

2. 예측 결과 plot
   - `plot_forecast_vs_actual`   : 예측/실제 overlay
   - `plot_metric_bars`          : 모델별 MAE/RMSE/MAPE bar chart

3. 분류 결과 plot
   - `plot_confusion_matrix`     : 혼동 행렬 heatmap
   - `plot_training_curves`      : train/val loss 곡선

모든 함수가 `seed` 인자를 받을 수 있으며 기본은 None → 셀 재실행마다 새 샘플.
"""
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
    """figure를 `FIG_DIR/name` 경로에 120dpi tight-layout으로 저장합니다.

    `name`이 None이면 아무 일도 하지 않습니다(화면에만 표시하고 싶을 때).

    Args:
        fig: 저장할 `matplotlib` Figure.
        name: 파일명(확장자 포함). 예: `"forecast_MAE.png"`.
    """
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
    """Jena DataFrame에서 임의의 `days`일 구간을 골라 지정 변수를 line plot합니다.

    동작:
        1. `df.index.max() - days` 이전의 시점 중 무작위로 시작점 선택.
        2. 시작점 ~ 시작점+days 구간을 slice.
        3. `columns` 각각을 한 서브플롯에 위아래로 쌓아 그림.

    `seed=None`이면 **셀 재실행마다 매번 다른 구간**이 나옵니다. 분석 중
    "오, 이 구간 흥미롭네" 싶을 때 `seed=42` 등 정수로 고정하면 재현 가능.

    Args:
        df: datetime 인덱스 DataFrame(보통 1시간 리샘플링된 Jena).
        days: 윈도우 길이(일).
        columns: 표시할 변수 이름들.
        seed: 재현용 난수 시드.
        save_as: 저장 파일명(`FIG_DIR` 아래). None이면 저장 안 함.

    Returns:
        Figure 객체.
    """
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
    """UCI HAR에서 무작위 `n`개 윈도우를 골라 격자로 표시합니다.

    각 서브플롯에는 9채널을 **모두 겹쳐** 그립니다. 제목에는 윈도우 인덱스와
    라벨 이름이 표시됩니다. `n`이 5 이하면 한 줄, 초과하면 5열 기준으로 줄바꿈.

    Args:
        X: `(N, 128, 9)` raw 신호 배열.
        y: `(N,)` 정수 라벨 배열(1-index 원본 또는 0-index 모두 허용).
        label_names: `{int: str}` — 라벨 → 활동 이름.
        channel_names: 9개 채널 이름(제목에만 사용되지 않음, 첫 subplot 범례).
        n: 뽑을 샘플 수.
        seed: 재현용 난수 시드. None이면 매번 다름.
        save_as: 저장 파일명.

    Returns:
        Figure 객체.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=n, replace=False)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for k, idx in enumerate(idxs):
        ax = axes[k // cols][k % cols]
        for c in range(X.shape[-1]):
            # 범례는 첫 subplot에만 표시(중복 방지).
            ax.plot(X[idx, :, c], lw=0.7, label=channel_names[c] if k == 0 else None)
        ax.set_title(f"#{idx} · {label_names.get(int(y[idx]), str(y[idx]))}", fontsize=9)
        ax.set_xticks([])
        ax.grid(alpha=0.3)
    # 격자에 비어 있는 subplot은 축을 숨겨 여백만 남김.
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
    """무작위로 클래스 1개 → 그 클래스의 무작위 윈도우 1개를 골라 9채널을 수직 스택합니다.

    `plot_random_har_samples`는 여러 윈도우를 9채널 겹침으로 보여주지만,
    이 함수는 **단일 윈도우의 채널별 차이**를 관찰하는 용도입니다.

    Args:
        X: `(N, 128, 9)` 신호 배열.
        y: `(N,)` 라벨.
        label_names: 라벨 이름 맵.
        channel_names: 9개 채널 이름(각 subplot y축 라벨).
        seed: 난수 시드.
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
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
    """UCI HAR 라벨 분포를 bar chart로 그립니다. 클래스 불균형을 한눈에 확인.

    Args:
        y: `(N,)` 라벨 배열.
        label_names: 정수 → 문자열 매핑.
        title: 그림 제목.
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
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
    """(클래스 × 채널) 격자에서 평균 |신호| 크기를 히트맵으로 시각화합니다.

    "어떤 활동이 어느 센서에 강하게 반응하는가"를 한 장으로 보여줍니다
    (예: WALKING은 body_acc_x의 평균 |신호|가 크고, LAYING은 작음).

    계산: `|X[y==cls]|`를 `(sample, time)` 축으로 평균 → 채널별 스칼라.

    Args:
        X: `(N, 128, 9)` 신호.
        y: `(N,)` 라벨.
        label_names: 라벨 이름 맵.
        channel_names: 채널 이름 리스트(열 라벨).
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
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
    """실제값(검정 굵은 선) 위에 여러 모델 예측을 overlay합니다.

    예측이 실제와 얼마나 추세·위상이 맞는지 직관적으로 비교하는 용도.
    메트릭 테이블과 쌍으로 보여주면 해석이 풍부해집니다.

    Args:
        timestamps: x축 값(시간 인덱스나 정수 인덱스 모두 허용).
        actual: 실제값 시퀀스 `(T,)`.
        predictions: `{"모델명": (T,) 배열}` 형태의 dict.
        title: 그림 제목.
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
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
    """모델별 단일 지표(예: MAE)를 bar chart로 비교합니다.

    Args:
        metrics: `{"모델명": {"MAE": ..., "RMSE": ..., "MAPE": ...}}` 형태.
        metric_name: 표시할 키(기본 "MAE").
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
    names = list(metrics.keys())
    vals = [metrics[n][metric_name] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, vals, color="steelblue")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by model (lower is better)")
    # 각 막대 위에 값 라벨을 표시하여 스크린샷만으로도 수치 판독이 가능하게 함.
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
    """`(K, K)` 혼동 행렬을 annotated heatmap으로 그립니다.

    행 = 실제 라벨, 열 = 예측 라벨. 대각이 진할수록 좋고, 비대각의 밝은 셀은
    혼동이 잦은 클래스 쌍을 의미합니다(예: WALKING ↔ WALKING_UPSTAIRS).

    Args:
        cm: `(K, K)` 정수 confusion matrix(행=true, 열=pred).
        label_names: 축에 표시할 K개 라벨 이름 리스트.
        title: 그림 제목.
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names,
                yticklabels=label_names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    _save(fig, save_as)
    return fig


def plot_training_curves(history, title: str = "Training curves", save_as: str | None = None):
    """`TrainHistory`의 train_loss·val_loss를 에폭 축에 그립니다.

    교차 지점이나 발산 여부로 과적합·학습률 문제를 진단합니다.

    Args:
        history: `train.TrainHistory` 인스턴스(`train_loss`, `val_loss` 속성 필요).
        title: 그림 제목.
        save_as: 저장 파일명.

    Returns:
        Figure.
    """
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
