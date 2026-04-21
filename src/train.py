"""Generic training loops for forecasting (regression) and classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
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
    device = device or get_device()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainHistory()

    for epoch in range(1, epochs + 1):
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
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            bar.set_postfix(loss=running / n_seen)
        train_loss = running / max(n_seen, 1)

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
    device = device or get_device()
    model.to(device).eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)
