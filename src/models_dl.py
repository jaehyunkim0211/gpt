"""PyTorch models: LSTM forecaster and 1D-CNN classifier."""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Encoder LSTM → linear head producing `horizon` future values."""

    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2,
                 horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) → (B, horizon)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class CNN1DClassifier(nn.Module):
    """3-stack Conv1d → GlobalAvgPool → Dense for HAR classification."""

    def __init__(self, n_channels: int, n_classes: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) → (B, n_classes)
        x = x.transpose(1, 2)  # (B, C, T)
        return self.net(x)
