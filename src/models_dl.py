"""PyTorch 기반 딥러닝 모델 정의: 예측용 LSTM, 분류용 1D-CNN.

두 클래스 모두 `nn.Module` 서브클래스이며 `train.train_model`이 그대로 재사용
할 수 있도록 공통 `forward` 규약을 따릅니다(입력 2D/3D 텐서 → 출력 텐서).

파일 구조
---------
1. `LSTMForecaster`
   입력: `(B, T, F)` — 배치, 타임스텝, 피처 수
   출력: `(B, horizon)` — horizon 스텝의 단일 타깃 예측
   구조: `LSTM(multi-layer) → 마지막 hidden state → Linear(hidden, horizon)`

2. `CNN1DClassifier`
   입력: `(B, T, C)` — 배치, 타임스텝, 채널 수 (사용자 친화적 ordering)
   출력: `(B, n_classes)` — 클래스별 로짓
   구조: `3-stack Conv1d + BN + ReLU → GlobalAvgPool → Dropout → Linear`
   내부에서 입력을 `(B, C, T)`로 transpose하여 PyTorch Conv1d 규약에 맞춤.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """LSTM 인코더 + 선형 헤드로 multi-step 예측을 수행하는 모델.

    동작 개요:
        1. `(B, T, F)` 입력을 `nn.LSTM`에 통과시켜 시간별 hidden state를 얻음.
        2. 마지막 타임스텝의 hidden state `out[:, -1, :]` 만 사용(전역 요약).
        3. 선형 계층으로 `horizon` 스텝의 스칼라 타깃을 **동시에** 회귀.

    Multi-step 출력을 순차적으로 만들지 않고 한 번에 뽑는 방식이라 학습·추론
    모두 단일 forward pass로 끝납니다(auto-regressive 방식보다 오차 누적이 적음).

    Args:
        n_features: 입력 피처 수 F (시계열 채널 수).
        hidden: LSTM hidden size 및 헤드 입력 차원.
        num_layers: LSTM 층 수. 2 이상이면 층 사이에 `dropout` 적용.
        horizon: 예측할 미래 스텝 수(출력 차원).
        dropout: LSTM 층간 dropout 확률. `num_layers=1`이면 자동으로 0.
    """

    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2,
                 horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            # PyTorch는 num_layers=1일 때 dropout을 쓰지 말라고 경고하므로 명시적으로 0 처리.
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM 순전파.

        Args:
            x: `(B, T, F)` 실수 텐서.

        Returns:
            `(B, horizon)` 예측 텐서. 스케일된 공간에서 나오므로
            실제 값으로 쓰려면 `StandardScaler1D.inverse_transform`이 필요.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class CNN1DClassifier(nn.Module):
    """9채널 × 128 timestep 시계열을 받아 6가지 활동을 분류하는 1D-CNN.

    구조:
        `Conv1d(C→H, k=5) → BN → ReLU`
        `→ Conv1d(H→2H, k=5) → BN → ReLU`
        `→ Conv1d(2H→2H, k=3) → BN → ReLU`
        `→ AdaptiveAvgPool1d(1) → Flatten → Dropout → Linear(2H, n_classes)`

    padding을 `kernel_size // 2`로 맞춰 시간 축 길이는 유지되며, 마지막
    GlobalAvgPool 단계에서만 `T → 1`로 요약됩니다. BatchNorm이 매 Conv 뒤에
    붙어 학습 안정성을 높이고, 최종 Dropout이 과적합을 억제합니다.

    Args:
        n_channels: 입력 채널 수(UCI HAR은 9).
        n_classes: 출력 클래스 수(UCI HAR은 6).
        hidden: 첫 Conv의 출력 채널 수. 이후 `hidden * 2`로 증폭.
        dropout: 최종 Linear 직전의 dropout 확률.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1D-CNN 순전파. 사용자는 `(B, T, C)` 순서로 주입하지만
        PyTorch Conv1d는 `(B, C, T)`를 기대하므로 내부에서 transpose합니다.

        Args:
            x: `(B, T, C)` 실수 텐서.

        Returns:
            `(B, n_classes)` 로짓 텐서. 확률화는 호출 측에서 `softmax`.
        """
        x = x.transpose(1, 2)  # (B, T, C) → (B, C, T)
        return self.net(x)
