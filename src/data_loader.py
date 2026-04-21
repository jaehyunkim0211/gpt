"""Jena Climate / UCI HAR 데이터셋 다운로드 및 로딩 모듈.

이 모듈은 외부 소스에서 두 개의 공개 데이터셋을 받아 `data/raw/` 아래에 캐시하고,
후속 파이프라인(전처리·모델링)이 바로 쓸 수 있는 형태로 로드합니다.
네트워크 접근은 **최초 1회**만 발생하며, 이미 다운로드된 파일이 있으면 재사용합니다.

파일 구조
---------
1. 상수 정의
   - `PROJECT_ROOT`, `RAW_DIR`  : 절대 경로
   - `JENA_URL`, `HAR_URL`      : 다운로드 소스 URL
   - `JENA_CSV`, `HAR_DIR`      : 압축 해제 후 기대되는 경로
   - `HAR_CHANNELS`, `HAR_LABELS` : UCI HAR 메타데이터

2. 내부 헬퍼
   - `_download(url, dest_zip)`        : 진행바와 함께 URL → 로컬 ZIP 저장
   - `_load_har_signals(split_dir, s)` : HAR의 9채널 inertial signal 배열화

3. 공개 API (노트북에서 직접 호출)
   - `fetch_jena_climate(force=False)` : Jena CSV 확보, 경로 반환
   - `fetch_uci_har(force=False)`      : HAR 디렉터리 확보, 경로 반환
   - `fetch_all(force=False)`          : 둘 다 한 번에
   - `load_jena_climate()`             : DataFrame으로 로드(datetime 인덱스)
   - `load_uci_har()`                  : train/test 신호·피처·라벨 dict 반환

4. `__main__` 가드
   - 스크립트로 직접 실행하면 두 데이터셋을 모두 받아 경로를 출력
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

JENA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

JENA_CSV = RAW_DIR / "jena_climate_2009_2016.csv"
HAR_DIR = RAW_DIR / "UCI HAR Dataset"


def _download(url: str, dest_zip: Path) -> None:
    """URL에서 ZIP 파일을 내려받아 `dest_zip` 경로에 저장합니다.

    - 스트리밍 방식(8192 바이트 chunk)으로 저장하여 메모리 사용을 억제합니다.
    - `tqdm` 진행바에 content-length 기반 퍼센티지를 표시합니다.
    - 상위 디렉터리가 없으면 자동 생성합니다.

    Args:
        url: 다운로드할 파일의 전체 URL.
        dest_zip: 저장할 로컬 파일 경로. 존재 시 덮어씁니다.

    Raises:
        requests.HTTPError: HTTP 4xx/5xx 응답.
        requests.Timeout: 60초 내 응답 실패.
    """
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_zip, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_zip.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def fetch_jena_climate(force: bool = False) -> Path:
    """Jena Climate CSV를 확보하고 그 경로를 반환합니다.

    동작 순서:
        1. 이미 `JENA_CSV`가 있고 `force=False`이면 즉시 반환.
        2. ZIP이 없거나 `force=True`이면 새로 다운로드.
        3. ZIP을 `RAW_DIR`에 풀어 CSV가 존재하는지 assert.

    Args:
        force: True이면 캐시를 무시하고 재다운로드·재추출합니다.

    Returns:
        압축 해제된 CSV 파일의 절대 경로.
    """
    if JENA_CSV.exists() and not force:
        return JENA_CSV
    zip_path = RAW_DIR / "jena_climate_2009_2016.csv.zip"
    if not zip_path.exists() or force:
        _download(JENA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    assert JENA_CSV.exists(), f"Expected {JENA_CSV} after extraction"
    return JENA_CSV


def fetch_uci_har(force: bool = False) -> Path:
    """UCI HAR 데이터셋 디렉터리를 확보하고 그 경로를 반환합니다.

    UCI HAR은 train/test 두 split과 561개 handcrafted feature, 9채널 raw signal을
    같은 ZIP에 포함하므로, 이 함수는 최상위 디렉터리(`UCI HAR Dataset/`)만 반환하고
    내부 파일 로딩은 `load_uci_har()`가 처리합니다.

    Args:
        force: True이면 캐시를 무시하고 재다운로드·재추출합니다.

    Returns:
        `UCI HAR Dataset/` 루트 디렉터리의 절대 경로.
    """
    if HAR_DIR.exists() and not force:
        return HAR_DIR
    zip_path = RAW_DIR / "UCI_HAR_Dataset.zip"
    if not zip_path.exists() or force:
        _download(HAR_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    assert HAR_DIR.exists(), f"Expected {HAR_DIR} after extraction"
    return HAR_DIR


def fetch_all(force: bool = False) -> dict[str, Path]:
    """두 데이터셋을 모두 다운로드·추출하고 경로 맵을 반환합니다.

    주로 노트북을 돌리기 전 사전 다운로드 용도로 사용합니다::

        python -c "from src.data_loader import fetch_all; fetch_all()"

    Args:
        force: True이면 두 데이터셋 모두 강제 재다운로드.

    Returns:
        `{"jena": <CSV 경로>, "har": <디렉터리 경로>}`.
    """
    return {
        "jena": fetch_jena_climate(force=force),
        "har": fetch_uci_har(force=force),
    }


def load_jena_climate() -> pd.DataFrame:
    """Jena Climate CSV를 DataFrame으로 로드합니다.

    - `Date Time` 컬럼을 `DD.MM.YYYY HH:MM:SS` 포맷으로 파싱하여 인덱스로 설정.
    - 정렬을 보장(원본은 거의 정렬되어 있지만 안전하게 `sort_index`).
    - **결측 처리는 하지 않음** — 10분 간격 원본에는 몇 개의 gap이 존재하며,
      이들은 `preprocessing.downsample_hourly`에서 시간 기반 보간으로 채워집니다.

    Returns:
        14개 기상 변수 컬럼 + datetime 인덱스를 가진 DataFrame.
    """
    csv_path = fetch_jena_climate()
    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    return df


# UCI HAR inertial signal 파일 이름(확장자 제외). 이 순서로 load하여
# (N, 128, 9) 형태의 배열을 만드므로 **재배열 금지**.
HAR_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# 활동 라벨 매핑. UCI HAR은 1부터 시작하는 정수 라벨을 사용하므로
# sklearn/torch에 넣기 전에 -1을 빼 0-index로 변환해야 합니다.
HAR_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _load_har_signals(split_dir: Path, split: str) -> np.ndarray:
    """UCI HAR의 9채널 inertial signal을 하나의 배열로 합칩니다.

    각 채널 파일은 `(N, 128)` 형태의 공백 구분 텍스트이며, 9개 채널을 마지막
    축으로 쌓아 `(N, 128, 9)`를 만듭니다. 배열 순서는 `HAR_CHANNELS`를 따릅니다.

    Args:
        split_dir: `train/` 또는 `test/` 디렉터리 경로.
        split: `"train"` 또는 `"test"` 문자열. 파일 이름 접미사로 사용됩니다.

    Returns:
        `shape=(N, 128, 9)`, `dtype=float64` 의 numpy 배열.
    """
    signals = []
    for ch in HAR_CHANNELS:
        path = split_dir / "Inertial Signals" / f"{ch}_{split}.txt"
        arr = np.loadtxt(path)
        signals.append(arr)
    return np.stack(signals, axis=-1)


def load_uci_har() -> dict:
    """UCI HAR 데이터셋을 train/test 전체 로드하여 dict로 반환합니다.

    반환 dict 키:
        - `X_signals_train` / `X_signals_test` : `(N, 128, 9)` raw 센서 신호
        - `X_features_train` / `X_features_test` : `(N, 561)` handcrafted 피처
        - `y_train` / `y_test` : `(N,)` int 라벨(1~6)
        - `feature_names` : 561개 피처 이름 리스트
        - `label_names` : `HAR_LABELS` 그대로
        - `channel_names` : `HAR_CHANNELS` 그대로

    Returns:
        위 키들을 가진 일반 파이썬 dict.
    """
    base = fetch_uci_har()
    out = {}
    for split in ("train", "test"):
        split_dir = base / split
        X_signals = _load_har_signals(split_dir, split)
        X_features = np.loadtxt(split_dir / f"X_{split}.txt")
        y = np.loadtxt(split_dir / f"y_{split}.txt").astype(int)
        out[f"X_signals_{split}"] = X_signals
        out[f"X_features_{split}"] = X_features
        out[f"y_{split}"] = y
    feature_names = (base / "features.txt").read_text(encoding="utf-8").strip().splitlines()
    out["feature_names"] = [line.split(" ", 1)[1] for line in feature_names]
    out["label_names"] = HAR_LABELS
    out["channel_names"] = HAR_CHANNELS
    return out


if __name__ == "__main__":
    # 스크립트로 직접 실행하면 두 데이터셋을 받고 경로를 출력합니다.
    # 주 사용 시점: 노트북 실행 전에 `python -m src.data_loader` 로 사전 캐시.
    paths = fetch_all()
    print("Datasets ready:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
