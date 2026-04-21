"""JPX 대회 평가 지표: Target 계산, 일별 스프레드 수익, Sharpe 스코어.

대회 규칙(요약):
    1. 매일 각 종목마다 `Rank`(0~N-1)를 예측 — 내림차순: Rank 0이 가장 오를 것 같은 종목.
    2. 상위 200 종목 매수, 하위 200 종목 공매도 포트폴리오 구성.
    3. 가중치는 `linspace(2, 1, 200)` — 가장 확신 있는 종목에 2배 비중.
    4. 해당 일의 스프레드 수익 = (매수 누적) - (매도 누적).
    5. 최종 점수 = **기간 전체 스프레드 수익의 Sharpe ratio**(평균 / 표준편차).

파일 구조
---------
1. `compute_target_from_close(series)`
   원본 Target 정의: `(Close.shift(-2) - Close.shift(-1)) / Close.shift(-1)`.
   검증·데모용 재현 함수이며, 실제 `stock_prices.csv`에는 이미 `Target` 컬럼이
   계산되어 있습니다.

2. `daily_spread_return(df_day, portfolio_size=200, toprank_weight_ratio=2)`
   하루치 프레임에서 `Sup - Sdown`을 반환합니다.

3. `calc_spread_return_sharpe(df, portfolio_size=200, toprank_weight_ratio=2)`
   기간 전체의 Sharpe ratio. 대회 공식 채점 함수를 그대로 차용(Kaggle notebook
   `smeitoma/jpx-competition-metric-definition`의 레퍼런스).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_target_from_close(close: pd.Series) -> pd.Series:
    """단일 종목 종가 시리즈로부터 JPX Target을 재현합니다.

    정의:
        `Target_t = (Close_{t+2} - Close_{t+1}) / Close_{t+1}`

    즉 "오늘 종가 대비 다음날 종가의 다음날 변화율". 시프트 때문에 마지막 2일은
    NaN이 됩니다. **실제 사용 시점엔 `stock_prices.csv`에 이미 `Target`이 들어있으므로**
    이 함수는 주로 "대회 정의가 어떻게 생겼는지" 설명하기 위한 참조 구현입니다.

    Args:
        close: 날짜 오름차순 종가 시리즈(단일 종목).

    Returns:
        같은 인덱스의 Target 시리즈. 마지막 2개 값은 NaN.
    """
    shift1 = close.shift(-1)
    shift2 = close.shift(-2)
    return (shift2 - shift1) / shift1


def daily_spread_return(
    df_day: pd.DataFrame,
    portfolio_size: int = 200,
    toprank_weight_ratio: float = 2.0,
) -> float:
    """하루치 `Sup - Sdown` 스프레드 수익을 계산합니다.

    계산 순서:
        1. `weights = linspace(2, 1, portfolio_size)` — 랭킹 1위에 2배 가중, 꼬리로 갈수록 1배.
        2. `Sup`   = 상위 portfolio_size 종목(Target × weights 합) / weights 평균.
        3. `Sdown` = 하위 portfolio_size 종목(Target × weights 합) / weights 평균.
        4. `Spread = Sup - Sdown`.

    위쪽은 "매수 → 상승 수익", 아래쪽은 "공매도 → 하락 수익"으로 해석하므로 Spread가
    클수록 당일 포트폴리오가 잘 맞힌 것을 뜻합니다.

    Args:
        df_day: 하루치(=특정 Date) 종목들만 담은 DataFrame. `Target`과 `Rank` 컬럼 필수.
            `Rank`는 0~len-1의 정수, 0이 "가장 오를 것"으로 예측.
        portfolio_size: 위/아래 각각 몇 개 종목을 사용할지. 기본 200.
        toprank_weight_ratio: 1위 가중치 / 꼴찌 가중치 비율. 기본 2.

    Returns:
        단일 float — 해당 날짜의 스프레드 수익률.
    """
    weights = np.linspace(start=toprank_weight_ratio, stop=1.0, num=portfolio_size)
    # 매수(상위): Rank 오름차순으로 앞 N개 = 랭킹 최상위.
    purchase = (
        df_day.sort_values("Rank")["Target"].iloc[:portfolio_size] * weights
    ).sum() / weights.mean()
    # 공매도(하위): Rank 내림차순으로 앞 N개 = 랭킹 최하위.
    short = (
        df_day.sort_values("Rank", ascending=False)["Target"].iloc[:portfolio_size] * weights
    ).sum() / weights.mean()
    return float(purchase - short)


def calc_spread_return_sharpe(
    df: pd.DataFrame,
    portfolio_size: int = 200,
    toprank_weight_ratio: float = 2.0,
) -> float:
    """기간 전체에 대한 스프레드 수익의 Sharpe ratio를 계산합니다(대회 공식 점수).

    `Date`별로 `daily_spread_return`을 구한 뒤,
        `sharpe = mean(daily_returns) / std(daily_returns)`
    로 집계합니다. 값이 클수록 "평균 수익이 높으면서 변동성이 낮은" 예측임을 뜻합니다.

    Args:
        df: 여러 날짜를 포함한 DataFrame. `Date`, `SecuritiesCode`, `Target`, `Rank` 컬럼 필수.
            `Rank.min()==0`, `Rank.max()==len(group)-1` 불변식이 날짜마다 만족되어야 합니다.
        portfolio_size: 상·하위 각 사용할 종목 수.
        toprank_weight_ratio: 1위 종목의 상대 가중치.

    Returns:
        float Sharpe ratio. 기간 전체 표준편차가 0이면 NaN/Inf가 나올 수 있습니다.
    """
    def _per_day(group):
        # 대회 공식 채점 함수가 요구하는 불변식을 그대로 검사.
        assert group["Rank"].min() == 0, "Rank는 0부터 시작해야 합니다"
        assert group["Rank"].max() == len(group["Rank"]) - 1, "Rank는 연속 정수여야 합니다"
        return daily_spread_return(group, portfolio_size, toprank_weight_ratio)

    daily = df.groupby("Date").apply(_per_day, include_groups=False)
    return float(daily.mean() / daily.std())
