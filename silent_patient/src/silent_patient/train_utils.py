from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainMetrics:
    loss: float
    mae: float | None = None
    acc: float | None = None


@torch.no_grad()
def eval_regression(model: nn.Module, loader, device: str, input_key: str = "x") -> TrainMetrics:
    model.eval()
    losses = []
    abs_err = []
    loss_fn = nn.MSELoss()
    for batch in loader:
        x = batch[input_key].to(device)
        y = batch["y"].to(device).float()
        pred = model(x).float()
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        abs_err.append(torch.abs(pred - y).mean().item())
    return TrainMetrics(loss=float(np.mean(losses)), mae=float(np.mean(abs_err)))


@torch.no_grad()
def eval_classification(model: nn.Module, loader, device: str, input_key: str = "x") -> TrainMetrics:
    model.eval()
    losses = []
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch[input_key].to(device)
        y = batch["y"].to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    acc = correct / max(1, total)
    return TrainMetrics(loss=float(np.mean(losses)), acc=float(acc))


def fit_one_epoch(model, loader, opt, device: str, task: str) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    if task == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if task == "regression":
            y = y.float()

        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * int(y.shape[0])
        n += int(y.shape[0])
    return total_loss / max(1, n)
