"""시계열 변화점(Change-point) 탐지 모듈.

두 가지 탐지 방식을 제공합니다.

1. **단순 버전** — `ruptures.Pelt` 기반
   - 잘 알려진 PELT 알고리즘으로 `l2`/`rbf` cost를 최소화하며 최적 분할점을 찾음.
   - 학술적 레퍼런스와 호환성이 좋고, 파라미터가 `model`, `pen` 둘뿐이라 간단.

2. **커스텀 강건 버전** — `detect_change_points(...)`
   - 레벨 이동·변동성 이동·추세 이동을 동시에 합성한 discrepancy score를 계산.
   - MAD 기반 robust scaling + outlier clip + local maxima + non-maximum suppression.
   - transform(auto/level/diff/pct/log_return), smoothing, window, sensitivity를
     한 번에 노출하여 "어떤 시그널 형태로 볼지"를 실험할 수 있음.
   - `ruptures`에 의존하지 않고 순수 numpy/pandas로 동작.

파일 구조
---------
- 단순 API : `get_stock_series`, `smooth_series`,
            `pelt_change_points`, `plot_with_change_points`, `analyze_stock_pelt`
- 커스텀 API : `detect_change_points` (단일 진입점, dict 반환)

커스텀 버전이 가독성·해석력이 더 높으므로 주로 추천하고, 단순 버전은 비교·검증용
으로 유지합니다.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt


# ─────────────────────────── 단순(ruptures) 버전 ───────────────────────────

def get_stock_series(df: pd.DataFrame, code: int) -> pd.DataFrame:
    """종목 하나만 뽑아 `Date` 인덱스 시계열로 만듭니다.

    `src.jpx_loader.get_stock_series`와 동일한 로직을 두고, 본 모듈이 JPX 외의
    시계열에도 쓰일 수 있도록 독립 복제했습니다. 향후 중복이 거슬리면 jpx_loader
    쪽을 import하는 방향으로 합쳐도 됩니다.
    """
    temp = df[df["SecuritiesCode"] == code].copy()
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp = temp.sort_values("Date").set_index("Date")
    return temp


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """rolling mean으로 시리즈를 스무딩합니다.

    `min_periods=1`이라 시리즈 앞부분에서도 NaN이 나오지 않습니다(대신 초기 몇
    포인트는 실제보다 불안정할 수 있음).
    """
    return series.rolling(window=window, min_periods=1).mean()


def pelt_change_points(series: pd.Series, model: str = "l2", pen: float = 10) -> list[int]:
    """ruptures의 PELT로 단일 시리즈의 변화점 인덱스를 계산합니다.

    Args:
        series: 1D 수치 시리즈(numpy 배열로 변환되어 쓰임).
        model: ruptures cost 모델. `"l2"`(분산 가정), `"rbf"`(커널) 등.
        pen: 패널티 계수. 클수록 변화점을 적게 찾음(보수적).

    Returns:
        변화점 위치 인덱스 리스트. 마지막 원소는 전체 길이(끝점)가 포함됨.
    """
    signal = np.asarray(series.values, dtype=float)
    algo = rpt.Pelt(model=model).fit(signal)
    return algo.predict(pen=pen)


def plot_with_change_points(
    series: pd.Series, change_points: list[int], title: str = "",
):
    """시리즈 라인 플롯 위에 변화점을 수직 점선으로 표시합니다.

    Args:
        series: 날짜 인덱스 시리즈.
        change_points: `pelt_change_points`가 반환한 인덱스 리스트.
            마지막 `len(series)` 값은 끝점이라 그리지 않고 건너뜁니다.
        title: 그림 제목.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="signal")
    for cp in change_points:
        if cp < len(series):
            plt.axvline(series.index[cp], linestyle="--", alpha=0.7)
    plt.title(title)
    plt.legend()


def analyze_stock_pelt(
    df: pd.DataFrame,
    code: int,
    pen: float = 100,
    window: int = 20,
) -> list[int]:
    """단순 파이프라인: 한 종목의 수익률 → 스무딩 → `ruptures.Pelt(rbf)`.

    `Close.pct_change()`로 레벨 트렌드를 제거해 **수익률**을 보고, rolling mean으로
    노이즈를 줄인 뒤 변화점을 찾습니다. 내부적으로 plot도 같이 그립니다.

    Args:
        df: long-format stock_prices DataFrame.
        code: 종목 코드.
        pen: ruptures 패널티(기본 100 — 원본 노트북 값).
        window: 스무딩 rolling window 길이.

    Returns:
        변화점 인덱스 리스트.
    """
    temp = get_stock_series(df, code)
    series = temp["Close"].pct_change().dropna()
    series = series.rolling(window=window, min_periods=1).mean()

    algo = rpt.Pelt(model="rbf", min_size=20).fit(series.values)
    change_points = algo.predict(pen=pen)

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="Return (smoothed)")
    for cp in change_points:
        if cp < len(series):
            plt.axvline(series.index[cp], linestyle="--", alpha=0.7)
    plt.title(f"{code} Change Point (PELT)")
    plt.legend()
    return change_points


# ─────────────────────────── 커스텀 강건 버전 ───────────────────────────

def detect_change_points(
    data,
    value_cols: str | list[str] | None = None,
    time_col: str | None = None,
    transform: str = "auto",
    smooth: int = 5,
    window: int = 20,
    min_distance: int = 20,
    sensitivity: float = 0.5,
    plot: bool = True,
) -> dict[str, Any]:
    """레벨·변동성·추세 변화를 합성하여 강건하게 변화점을 찾습니다.

    `ruptures`에 의존하지 않고, 다음과 같은 단계로 점수를 계산한 뒤 임계값을
    넘는 지점을 변화점 후보로 고릅니다:

    1. **input 정리** : Series/DataFrame/ndarray를 DataFrame으로 통일.
    2. **time column 처리** : datetime 파싱·정렬·중복 제거.
    3. **value column 선택** : 숫자 컬럼 자동 선택 가능.
    4. **transform** : `auto`일 경우 양수 + dynamic range > 1.5 → `log_return`,
       lag1 자기상관 > 0.85 → `diff`, 그 외 `level`.
    5. **robust clip** : MAD 기반 z-score를 ±8로 클립해 이상치 영향 완화.
    6. **smoothing** : rolling median(`smooth`).
    7. **robust scaling** : MAD로 재차 정규화.
    8. **discrepancy score** : 각 시점 `t`에서 좌우 `window`의
       (median 차이) + 0.6 × (MAD 차이) + 0.8 × (선형 slope 차이 × window)를 합산.
    9. **thresholding** : `sensitivity`로 MAD 배수와 분위수 threshold를 계산해
       두 값 중 큰 쪽을 선택.
    10. **local max + non-max suppression** : `min_distance`보다 가까운 후보 제거.

    Args:
        data: pandas Series/DataFrame 또는 1D/2D ndarray.
        value_cols: 관찰할 컬럼 이름(들). None이면 숫자 컬럼 전부.
        time_col: 날짜 컬럼 이름. None이면 기존 인덱스 사용.
        transform: `"auto" | "level" | "diff" | "pct" | "log_return"`.
            `log_return`은 모든 값이 양수일 때만 유효합니다.
        smooth: rolling median 윈도우 길이(≥1).
        window: 좌우 비교 윈도우 길이(≥3).
        min_distance: 변화점 사이 최소 간격.
        sensitivity: 0.05~0.95. 클수록 더 많은 변화점을 뽑음.
        plot: True면 상단에 원 신호 + 변화점 수직선, 하단에 score + threshold 표시.

    Returns:
        dict:
            - `change_points` : 변화점 DataFrame(`proc_pos`, `time`, `raw_score`,
              `z_strength`, `strength_0_100`, `level_shift`, `volatility_shift`, `trend_shift`)
            - `score`         : cp_score 시리즈(시간 인덱스)
            - `processed`     : transform + smoothing 후 처리된 DataFrame
            - `threshold`     : 최종 임계값
            - `transform_used`: 실제 사용된 transform 이름

    Raises:
        ValueError: 입력 차원 불일치, 데이터 길이 부족, 잘못된 transform 등.
    """
    eps = 1e-12
    sensitivity = float(np.clip(sensitivity, 0.05, 0.95))
    smooth = max(1, int(smooth))
    window = max(3, int(window))
    min_distance = max(1, int(min_distance))

    # ---- 0) input → DataFrame 통일 ----
    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "value").copy()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        arr = np.asarray(data)
        if arr.ndim == 1:
            df = pd.DataFrame({"value": arr})
        elif arr.ndim == 2:
            df = pd.DataFrame(arr, columns=[f"value_{i}" for i in range(arr.shape[1])])
        else:
            raise ValueError("data는 Series / DataFrame / 1D or 2D ndarray 여야 합니다.")

    # ---- 1) 시간축 정리 ----
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = (
            df.dropna(subset=[time_col])
              .sort_values(time_col)
              .drop_duplicates(subset=[time_col], keep="last")
              .set_index(time_col)
        )
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        else:
            df = df.reset_index(drop=True)

    # ---- 2) 관찰 컬럼 선택 ----
    if value_cols is None:
        value_cols = df.select_dtypes(include=np.number).columns.tolist()
    elif isinstance(value_cols, str):
        value_cols = [value_cols]
    if not value_cols:
        raise ValueError("numeric value column이 없습니다.")

    work = df[value_cols].apply(pd.to_numeric, errors="coerce")
    # gap을 시간 선형 보간으로 채워 길이 손실을 방지.
    work = work.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if len(work) < (2 * window + 5):
        raise ValueError(f"데이터가 너무 짧습니다. 최소 {2 * window + 5}개 이상 필요합니다.")
    original_for_plot = work.copy()

    # ---- 내부 헬퍼들 ----
    def _mad_frame(frame: pd.DataFrame):
        """컬럼별 median과 MAD(둘 다 Series)를 돌려줍니다.

        MAD가 0에 가까우면 표준편차로, 그조차 0이면 1로 대체하여
        후속 분모 0을 회피합니다.
        """
        med = frame.median()
        mad = frame.sub(med).abs().median()
        fallback = frame.std(ddof=0).replace(0, np.nan)
        mad = mad.where(mad > eps, fallback).fillna(1.0).replace(0, 1.0)
        return med, mad

    def _lag1_autocorr(x) -> float:
        """lag-1 자기상관. transform 자동 선택에서 "차분이 유리한지" 판단에 사용."""
        x = pd.Series(x).dropna().values.astype(float)
        if len(x) < 3:
            return 0.0
        a, b = x[:-1], x[1:]
        if np.std(a) < eps or np.std(b) < eps:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _choose_transform(frame: pd.DataFrame) -> str:
        """`transform='auto'`일 때 첫 컬럼 특성으로 변환 방식을 추천합니다."""
        x = frame.iloc[:, 0].values.astype(float)
        positive = bool(np.all(x > 0))
        ac1 = abs(_lag1_autocorr(x))
        if positive:
            dynamic_range = np.nanmax(x) / max(np.nanmin(x), eps)
            if dynamic_range > 1.5:
                return "log_return"
        if ac1 > 0.85:
            return "diff"
        return "level"

    def _slope(seg_2d: np.ndarray) -> np.ndarray:
        """세그먼트 각 컬럼의 최소제곱 기울기. 트렌드 변화 계산에 사용."""
        idx = np.arange(seg_2d.shape[0], dtype=float)
        idx = idx - idx.mean()
        denom = (idx ** 2).sum() + eps
        centered = seg_2d - seg_2d.mean(axis=0, keepdims=True)
        return (idx[:, None] * centered).sum(axis=0) / denom

    # ---- 3) transform ----
    transform_used = _choose_transform(work) if transform == "auto" else transform
    if transform_used == "level":
        proc = work.copy()
    elif transform_used == "diff":
        proc = work.diff()
    elif transform_used == "pct":
        proc = work.pct_change()
    elif transform_used == "log_return":
        if (work <= 0).any().any():
            raise ValueError("log_return은 모든 값이 양수일 때만 사용할 수 있습니다.")
        proc = np.log(work).diff()
    else:
        raise ValueError("transform은 auto / level / diff / pct / log_return 중 하나여야 합니다.")
    proc = proc.replace([np.inf, -np.inf], np.nan).dropna()
    if len(proc) < (2 * window + 5):
        raise ValueError("transform 후 데이터가 너무 짧아졌습니다. window를 줄이거나 데이터를 늘리세요.")

    # ---- 4) robust outlier clip ----
    med1, mad1 = _mad_frame(proc)
    z1 = proc.sub(med1).div(mad1)
    proc = med1 + z1.clip(-8, 8).mul(mad1)

    # ---- 5) smoothing ----
    if smooth > 1:
        proc = proc.rolling(smooth, center=True, min_periods=1).median()

    # ---- 6) robust scaling ----
    med2, mad2 = _mad_frame(proc)
    proc = proc.sub(med2).div(mad2)

    X = proc.to_numpy(dtype=float)
    n, _ = X.shape

    # ---- 7) discrepancy score ----
    level_score = np.zeros(n)
    vol_score = np.zeros(n)
    trend_score = np.zeros(n)
    for t in range(window, n - window):
        L = X[t - window:t]
        R = X[t:t + window]
        level_shift = np.linalg.norm(np.median(R, axis=0) - np.median(L, axis=0))
        madL = np.median(np.abs(L - np.median(L, axis=0)), axis=0)
        madR = np.median(np.abs(R - np.median(R, axis=0)), axis=0)
        volatility_shift = np.linalg.norm(madR - madL)
        trend_shift = np.linalg.norm(_slope(R) - _slope(L)) * window
        level_score[t] = level_shift
        vol_score[t] = volatility_shift
        trend_score[t] = trend_shift
    total_score = level_score + 0.6 * vol_score + 0.8 * trend_score
    score = pd.Series(total_score, index=proc.index, name="cp_score")
    score = score.rolling(max(3, smooth), center=True, min_periods=1).mean()

    # ---- threshold ----
    valid = score.iloc[window:n - window]
    s_med = float(valid.median())
    s_mad = float((valid - s_med).abs().median()) + eps
    mad_k = 3.2 - 2.0 * sensitivity
    abs_threshold = s_med + mad_k * s_mad
    q = float(np.clip(0.98 - 0.40 * sensitivity, 0.55, 0.99))
    q_threshold = float(valid.quantile(q))
    threshold = max(abs_threshold, q_threshold)

    # ---- local maxima + non-max suppression ----
    y = score.values
    candidates = np.where(
        (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] >= threshold)
    )[0] + 1
    order = candidates[np.argsort(y[candidates])[::-1]]
    selected: list[int] = []
    for idx in order:
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(int(idx))
    selected = sorted(selected)

    # ---- 8) 결과 테이블 ----
    if selected:
        raw = y[selected]
        strength_pct = np.array([(valid <= r).mean() * 100 for r in raw])
        z_strength = (raw - s_med) / s_mad
        cp_df = pd.DataFrame({
            "proc_pos": selected,
            "time": score.index[selected],
            "raw_score": raw,
            "z_strength": z_strength,
            "strength_0_100": strength_pct,
            "level_shift": level_score[selected],
            "volatility_shift": vol_score[selected],
            "trend_shift": trend_score[selected],
        }).sort_values("proc_pos").reset_index(drop=True)
    else:
        cp_df = pd.DataFrame(columns=[
            "proc_pos", "time", "raw_score", "z_strength", "strength_0_100",
            "level_shift", "volatility_shift", "trend_shift",
        ])

    # ---- 9) 시각화 ----
    if plot:
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        if original_for_plot.shape[1] == 1:
            plot_series = original_for_plot.iloc[:, 0]
            axes[0].plot(plot_series.index, plot_series.values)
            axes[0].set_ylabel(original_for_plot.columns[0])
        else:
            medp, madp = _mad_frame(original_for_plot)
            composite = original_for_plot.sub(medp).div(madp).mean(axis=1)
            axes[0].plot(composite.index, composite.values)
            axes[0].set_ylabel("composite signal")
        for _, row in cp_df.iterrows():
            axes[0].axvline(row["time"], linestyle="--", alpha=0.7)
        _, ymax = axes[0].get_ylim()
        for _, row in cp_df.iterrows():
            axes[0].text(row["time"], ymax, f'{row["strength_0_100"]:.0f}',
                         rotation=90, va="top", fontsize=8)
        axes[0].set_title(
            f"Change points | transform={transform_used}, smooth={smooth}, "
            f"window={window}, min_distance={min_distance}, sensitivity={sensitivity:.2f}"
        )
        axes[1].plot(score.index, score.values, label="cp_score")
        axes[1].axhline(threshold, linestyle="--", label="threshold")
        if len(cp_df) > 0:
            axes[1].scatter(cp_df["time"], cp_df["raw_score"], s=40)
        axes[1].set_ylabel("score")
        axes[1].set_xlabel("time")
        axes[1].legend()
        plt.tight_layout()

    return {
        "change_points": cp_df,
        "score": score,
        "processed": proc,
        "threshold": threshold,
        "transform_used": transform_used,
    }
