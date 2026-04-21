"""Download and load Jena Climate and UCI HAR datasets."""
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
    """Download Jena Climate CSV. Returns path to extracted CSV."""
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
    """Download UCI HAR dataset. Returns path to extracted directory."""
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
    return {
        "jena": fetch_jena_climate(force=force),
        "har": fetch_uci_har(force=force),
    }


def load_jena_climate() -> pd.DataFrame:
    """Load Jena Climate as a DataFrame indexed by datetime."""
    csv_path = fetch_jena_climate()
    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    return df


HAR_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

HAR_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _load_har_signals(split_dir: Path, split: str) -> np.ndarray:
    """Load 9-channel inertial signals for one split. Returns (N, 128, 9)."""
    signals = []
    for ch in HAR_CHANNELS:
        path = split_dir / "Inertial Signals" / f"{ch}_{split}.txt"
        arr = np.loadtxt(path)
        signals.append(arr)
    return np.stack(signals, axis=-1)


def load_uci_har() -> dict:
    """Load UCI HAR. Returns dict with raw signals, handcrafted features, labels."""
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
    paths = fetch_all()
    print("Datasets ready:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
