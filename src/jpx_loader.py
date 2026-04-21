"""JPX Tokyo Stock Exchange Prediction 데이터셋 로더.

Kaggle의 JPX Tokyo Market Prediction 대회 데이터(주가·거래량·재무·옵션)를
`data/raw/jpx/` 아래에서 읽어들입니다. 이 모듈은 **로딩 + 날짜 파싱**만
담당하고, 타깃 계산·랭킹·메트릭 등은 `jpx_metric.py`로 분리되어 있습니다.

파일 구조
---------
1. 상수
   - `JPX_DIR`                   : `data/raw/jpx/` 절대 경로
   - `STOCK_PRICE_COLS`          : stock_prices 숫자 컬럼 목록(참조용)

2. 공개 로더
   - `load_stock_prices(split)`           : 주가 시계열(핵심 파일)
   - `load_secondary_stock_prices(split)` : 거래량이 적은 부수 종목
   - `load_financials(split)`             : 분기 실적
   - `load_trades(split)`                 : 주간 거래 요약
   - `load_options(split)`                : 옵션 시세
   - `load_stock_list()`                  : SecuritiesCode ↔ 회사 이름
   - `load_spec(name)`                    : `data_specifications/` 안의 컬럼 설명
   - `load_sample_submission()`           : 제출 템플릿

3. 유틸
   - `get_stock_series(df, code)` : `SecuritiesCode` 필터 + 날짜 정렬 + 인덱스 설정
   - `add_rank_column(df)`        : 일자별 Target 내림차순 0-index Rank 부여

split 인자는 `"train"`(train_files), `"supplemental"`(supplemental_files),
`"example"`(example_test_files) 중 하나입니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# data_loader.py와 동일한 규약으로 프로젝트 루트를 계산합니다.
# (src/ 기준 한 단계 위)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JPX_DIR = PROJECT_ROOT / "data" / "raw" / "jpx"

# split 이름 → 실제 하위 디렉터리 매핑.
# Kaggle 대회 공식 분할 이름을 그대로 사용합니다.
_SPLIT_DIRS = {
    "train": JPX_DIR / "train_files",
    "supplemental": JPX_DIR / "supplemental_files",
    "example": JPX_DIR / "example_test_files",
}

# stock_prices.csv 숫자 컬럼. 로딩 후 dtype 검증이나 집계에 참고로 사용.
STOCK_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor",
                    "ExpectedDividend", "SupervisionFlag", "Target"]


def _split_dir(split: str) -> Path:
    """`split` 문자열을 하위 디렉터리 경로로 변환합니다.

    Args:
        split: `"train"`, `"supplemental"`, `"example"` 중 하나.

    Returns:
        해당 split 디렉터리의 절대 경로.

    Raises:
        KeyError: 알 수 없는 split 이름.
    """
    if split not in _SPLIT_DIRS:
        raise KeyError(f"unknown split={split!r}, expected one of {list(_SPLIT_DIRS)}")
    return _SPLIT_DIRS[split]


def _load_csv_with_date(path: Path) -> pd.DataFrame:
    """CSV를 읽고 `Date` 컬럼이 있으면 datetime으로 파싱합니다.

    대부분의 JPX CSV는 `Date`를 첫 컬럼으로 가지며 문자열 `YYYY-MM-DD` 형식입니다.
    pandas는 기본적으로 object로 읽어들이므로 후속 슬라이싱·정렬이 느려집니다.
    이 헬퍼가 **로딩 시 한 번만** 파싱하도록 강제합니다.
    """
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_stock_prices(split: str = "train") -> pd.DataFrame:
    """`stock_prices.csv`를 DataFrame으로 로드합니다(대회의 핵심 파일).

    주요 컬럼:
        - `Date`, `SecuritiesCode` : 복합 키.
        - `Open/High/Low/Close/Volume` : 일봉 OHLCV.
        - `AdjustmentFactor` : 주식 분할 등의 조정 계수.
        - `Target` : 다음날 종가 대비 이틀 뒤 종가의 변화율
                     `(Close_{t+2} - Close_{t+1}) / Close_{t+1}`.

    Args:
        split: `"train"`, `"supplemental"`, `"example"`.

    Returns:
        약 200만~300만 row DataFrame(split에 따라 다름). `Date`는 datetime64.
    """
    return _load_csv_with_date(_split_dir(split) / "stock_prices.csv")


def load_secondary_stock_prices(split: str = "train") -> pd.DataFrame:
    """덜 유동적인 부수 종목(secondary) 주가.

    메인 `stock_prices`에 없는 덜 거래되는 종목들을 담고 있어, 특징 엔지니어링이나
    대체 종목 탐색에 사용합니다.
    """
    return _load_csv_with_date(_split_dir(split) / "secondary_stock_prices.csv")


def load_financials(split: str = "train") -> pd.DataFrame:
    """분기 실적(`financials.csv`). 매출·영업이익·순이익 등의 fundamentals."""
    return _load_csv_with_date(_split_dir(split) / "financials.csv")


def load_trades(split: str = "train") -> pd.DataFrame:
    """주간 거래 요약(`trades.csv`). 직전 영업주의 누적 거래량·매수주체별 분포."""
    return _load_csv_with_date(_split_dir(split) / "trades.csv")


def load_options(split: str = "train") -> pd.DataFrame:
    """옵션 시세(`options.csv`). 암묵적 변동성을 포함한 전체 시장 지표."""
    return _load_csv_with_date(_split_dir(split) / "options.csv")


def load_stock_list() -> pd.DataFrame:
    """`stock_list.csv` — SecuritiesCode ↔ 회사명·섹터 정보."""
    return pd.read_csv(JPX_DIR / "stock_list.csv")


def load_spec(name: str) -> pd.DataFrame:
    """`data_specifications/{name}_spec.csv` 스펙 문서 로드.

    Args:
        name: `"stock_price"`, `"stock_fin"`, `"stock_list"`, `"options"`, `"trades"` 중 하나.

    Returns:
        각 컬럼의 의미·단위를 설명하는 DataFrame.
    """
    return pd.read_csv(JPX_DIR / "data_specifications" / f"{name}_spec.csv")


def load_sample_submission() -> pd.DataFrame:
    """`example_test_files/sample_submission.csv` 제출 템플릿."""
    return pd.read_csv(_split_dir("example") / "sample_submission.csv")


def get_stock_series(df: pd.DataFrame, code: int) -> pd.DataFrame:
    """`SecuritiesCode`로 한 종목만 잘라내 날짜 인덱스 시계열로 만듭니다.

    원본 `load_stock_prices`는 long-format(모든 종목이 같은 DataFrame)입니다.
    단일 종목 분석·시각화가 필요할 때 이 헬퍼로 각 종목을 **개별 시계열**로
    변환합니다.

    Args:
        df: `load_stock_prices()` 결과 같은 long-format DataFrame.
            `Date`와 `SecuritiesCode` 컬럼이 있어야 합니다.
        code: 종목 코드(정수). 예: `1301`.

    Returns:
        `Date`가 인덱스인 DataFrame, 해당 종목 행만 날짜 오름차순 정렬.
    """
    temp = df[df["SecuritiesCode"] == code].copy()
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp = temp.sort_values("Date").set_index("Date")
    return temp


def add_rank_column(df: pd.DataFrame, target_col: str = "Target") -> pd.DataFrame:
    """일자별로 `target_col`을 내림차순으로 정렬한 0-index Rank 컬럼을 추가합니다.

    JPX 대회의 평가 방식(상위 200·하위 200)에 필요한 Rank를 계산합니다.
    - 동점 tiebreak는 `method="first"`로 결정적으로 처리.
    - `-1` 오프셋으로 0-index를 만들어 `Rank.min()==0`, `Rank.max()==len-1` 불변식 유지
      (`calc_spread_return_sharpe`의 `assert`와 호환).

    Args:
        df: `Date`와 `target_col`을 포함한 long-format DataFrame.
        target_col: 랭킹 기준 컬럼(기본 `"Target"`).

    Returns:
        원본 + `Rank` 컬럼이 추가된 **복사본**.
    """
    out = df.copy()
    out["Rank"] = (
        out.groupby("Date")[target_col].rank(ascending=False, method="first") - 1
    )
    return out
