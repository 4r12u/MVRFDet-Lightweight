from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Paper-style channel attention: max/avg pooling -> shared 256->16->256 MLP."""

    def __init__(self, channels: int = 256, hidden: int = 16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,C,H,W]
        avg = torch.mean(x, dim=(2, 3))
        max_ = torch.amax(x, dim=(2, 3))
        weights = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(max_))
        weighted = x * weights[:, :, None, None]
        return weighted, weights
