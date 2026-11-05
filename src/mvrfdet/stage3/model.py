from __future__ import annotations

import torch
from torch import nn

from .cnn import ThreeLayerCNN
from .channel_attention import ChannelAttention
from .classifier import ClassificationMLP


class LightweightMVRFDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ThreeLayerCNN()
        self.channel_attention = ChannelAttention(256, 16)
        self.classifier = ClassificationMLP()

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        features = self.cnn(x)
        weighted, attention = self.channel_attention(features)
        logits = self.classifier(weighted)
        if return_attention:
            return logits, attention
        return logits
