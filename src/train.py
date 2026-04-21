"""공통 PyTorch 학습 루프와 추론 헬퍼.

이 모듈은 예측(LSTM)과 분류(1D-CNN) 양쪽에서 **같은** 학습 코드를 쓰기 위한
얇은 공통 레이어입니다. 모델을 몰라도 `loss_fn`과 `metric_fn`만 주입받으면
동작하므로, 새 모델을 추가할 때 학습 루프를 다시 짤 필요가 없습니다.

파일 구조
---------
1. `get_device()`           : CUDA 가용 시 `cuda`, 아니면 `cpu` 반환.
2. `TrainHistory`           : epoch별 train/val loss와 metric을 담는 dataclass.
3. `make_loader(X, y, ...)` : numpy → `TensorDataset` → `DataLoader` 변환.
4. `train_model(...)`       : 메인 학습 루프(AdamW, tqdm, per-epoch 평가).
5. `predict(model, loader)` : 추론. 모든 배치 결과를 numpy로 concat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def get_device() -> torch.device:
    """사용 가능한 PyTorch 디바이스를 반환합니다.

    CUDA가 켜져 있으면 `cuda` 디바이스(=첫 번째 GPU)를, 아니면 `cpu`를 돌려줍니다.
    프로젝트 전반에서 `model.to(get_device())` 패턴으로 일관된 디바이스 사용.

    Returns:
        `torch.device("cuda")` 또는 `torch.device("cpu")`.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainHistory:
    """에폭별 학습 로그를 담는 간단한 컨테이너.

    Attributes:
        train_loss: 에폭별 평균 train loss 리스트.
        val_loss:   에폭별 평균 val loss 리스트.
        val_metric: 에폭별 평균 val metric 리스트. `metric_fn=None`이면 NaN으로 채움.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """numpy 배열을 PyTorch `DataLoader`로 감쌉니다.

    `y`의 dtype에 따라 텐서 타입을 자동 결정합니다:
        - 실수 dtype  → `float` (회귀용, `MSELoss` 등과 호환)
        - 정수 dtype  → `long`  (분류용, `CrossEntropyLoss`와 호환)

    `X`는 항상 `float32`로 변환합니다.

    Args:
        X: 입력 배열 `(N, ...)`.
        y: 타깃 배열 `(N, ...)`.
        batch_size: 배치 크기.
        shuffle: 에폭마다 셔플 여부(train=True, val/test=False 권장).

    Returns:
        `DataLoader` 인스턴스. 내부에서 `TensorDataset(X_t, y_t)` 사용.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y)
    if y_t.dtype.is_floating_point:
        y_t = y_t.float()
    else:
        y_t = y_t.long()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainHistory:
    """모델을 주어진 에폭 동안 학습하고 `TrainHistory`를 돌려줍니다.

    구현 세부:
        - 옵티마이저: `AdamW(lr, weight_decay)` 고정. 스케줄러 없음.
        - 각 에폭 말에 `val_loader`로 검증하고 평균 loss·metric을 기록.
        - `tqdm` 진행바가 배치별 평균 loss를 실시간으로 보여줍니다.
        - 최선 모델 저장/early stopping 등은 **구현하지 않음** — 현재 프로젝트에서는
          노트북이 수동으로 epoch 수를 지정하므로 단순함을 우선.

    Args:
        model: 학습할 `nn.Module`. 이 함수가 `device`로 옮깁니다.
        train_loader: 학습 DataLoader. `shuffle=True` 권장.
        val_loader: 검증 DataLoader. `shuffle=False` 필수.
        loss_fn: 미분 가능한 손실 함수(`MSELoss`, `CrossEntropyLoss` 등).
        metric_fn: `(pred, y)` → float 스칼라. 예: 정확도 계산 람다.
            None이면 `val_metric`은 NaN으로 기록됩니다.
        epochs: 학습 에폭 수.
        lr: 학습률.
        weight_decay: AdamW의 weight decay 계수.
        device: 사용 디바이스. None이면 `get_device()`가 자동 선택.
        verbose: True면 epoch 요약을 print하고 tqdm 진행바 표시.

    Returns:
        `TrainHistory` — loss/metric을 에폭 단위로 기록한 객체. 시각화에 사용.
    """
    device = device or get_device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainHistory()

    for epoch in range(1, epochs + 1):
        # ── 학습 phase ────────────────────────────────────────────────────
        model.train()
        running = 0.0
        n_seen = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            # running sum을 가중 평균으로 유지(배치 크기가 들쑥날쑥해도 안전).
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            bar.set_postfix(loss=running / n_seen)
        train_loss = running / max(n_seen, 1)

        # ── 검증 phase ────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        v_n = 0
        v_metric_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_loss += loss_fn(pred, yb).item() * xb.size(0)
                if metric_fn is not None:
                    v_metric_sum += metric_fn(pred, yb) * xb.size(0)
                v_n += xb.size(0)
        v_loss /= max(v_n, 1)
        v_metric = (v_metric_sum / max(v_n, 1)) if metric_fn is not None else float("nan")

        history.train_loss.append(train_loss)
        history.val_loss.append(v_loss)
        history.val_metric.append(v_metric)
        if verbose:
            msg = f"epoch {epoch:>2}: train_loss={train_loss:.4f}  val_loss={v_loss:.4f}"
            if metric_fn is not None:
                msg += f"  val_metric={v_metric:.4f}"
            print(msg)
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> np.ndarray:
    """DataLoader 전체에 대해 추론하고 예측을 numpy로 합쳐 반환합니다.

    `model.eval()` 및 `torch.no_grad()`가 자동 적용되므로 BatchNorm/Dropout이
    비활성화됩니다. 라벨 `y`는 무시하고 입력만 사용합니다.

    Args:
        model: 이미 학습된 `nn.Module`.
        loader: 추론 대상 DataLoader.
        device: 디바이스. None이면 `get_device()`.

    Returns:
        모든 배치 결과를 axis=0에서 concat한 numpy 배열.
        회귀라면 `(N, horizon)`, 분류라면 `(N, n_classes)` 로짓.
    """
    device = device or get_device()
    model.to(device).eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)
