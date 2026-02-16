from __future__ import annotations

import torch
import torch.nn as nn


class SmallCnnRegressor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.net(x).squeeze(-1)


class SmallCnnClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


class Wav2Vec2Head(nn.Module):
    """A small head on top of a wav2vec2 encoder.

    Contract:
    - Input: raw waveform tensor of shape (B, T) at 16kHz
    - Output:
      - regression: (B, 1) float
      - classification: (B, n_classes) logits
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        task: str,
        n_classes: int = 3,
        head_hidden: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.task = task

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        out_dim = 1 if task == "regression" else n_classes
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T)
        # torchaudio wav2vec2 models often expose extract_features() returning a list of layers.
        if hasattr(self.encoder, "extract_features"):
            feats, _ = self.encoder.extract_features(wav)
            x = feats[-1]  # (B, T', C)
        else:
            x = self.encoder(wav)
            if isinstance(x, (list, tuple)):
                x = x[0]

        if x.dim() == 3:
            x = x.mean(dim=1)  # temporal mean pooling -> (B, C)
        return self.head(x)
