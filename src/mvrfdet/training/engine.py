from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def run_epoch(model, loader, device, optimizer=None) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch
    return EpochMetrics(loss=total_loss / max(1, total), accuracy=total_correct / max(1, total))
