from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import masked_laplace_nll


def make_loader(split, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(split.x),
        torch.from_numpy(split.y),
        torch.from_numpy(split.y_mask),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_epoch(model, loader, adjacency, optimizer, device="cpu"):
    model.train()
    total, examples = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, adjacency)
        loss = masked_laplace_nll(output, y, mask)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        examples += len(x)
    return total / examples


@torch.no_grad()
def predict(model, loader, adjacency, device="cpu"):
    model.eval()
    outputs = {key: [] for key in ("target", "mask", "location", "scale")}
    start = time.perf_counter()
    for x, y, mask in loader:
        result = model(x.to(device), adjacency)
        outputs["target"].append(y.numpy())
        outputs["mask"].append(mask.numpy())
        outputs["location"].append(result.location.cpu().numpy())
        outputs["scale"].append(result.scale.cpu().numpy())
    outputs = {key: np.concatenate(value) for key, value in outputs.items()}
    outputs["elapsed_seconds"] = time.perf_counter() - start
    return outputs


def save_predictions(path: str | Path, predictions: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **predictions)

