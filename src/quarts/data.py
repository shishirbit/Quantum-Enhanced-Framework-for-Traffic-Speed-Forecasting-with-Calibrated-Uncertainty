from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitData:
    x: np.ndarray  # [sample, input_steps, node, feature]
    y: np.ndarray  # [sample, horizon, node]
    x_mask: np.ndarray
    y_mask: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> "Standardizer":
        safe = np.where(mask, values, np.nan)
        mean = np.nanmean(safe, axis=(0, 1, 2), keepdims=True)
        std = np.nanstd(safe, axis=(0, 1, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * np.squeeze(self.std) + np.squeeze(self.mean)


def load_timeseries(path: str | Path) -> np.ndarray:
    """Load canonical traffic arrays as [time, node, feature]."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(path)
        key = "data" if "data" in archive.files else archive.files[0]
        data = archive[key]
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        try:
            import pandas as pd
            frame = pd.read_hdf(path)
            data = frame.to_numpy()[..., None]
        except (ImportError, ValueError, KeyError):
            import h5py
            with h5py.File(path, "r") as handle:
                key = "data" if "data" in handle else next(iter(handle.keys()))
                data = handle[key][...]
    else:
        raise ValueError(f"Unsupported dataset file: {path}")
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f"Expected [time,node,feature], got {data.shape}")
    return data.astype(np.float32)


def chronological_boundaries(length: int, ratios=(0.7, 0.1, 0.2)) -> tuple[tuple[int, int], ...]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to one")
    train_end = int(length * ratios[0])
    valid_end = train_end + int(length * ratios[1])
    return (0, train_end), (train_end, valid_end), (valid_end, length)


def make_windows(segment: np.ndarray, input_steps: int, horizon: int) -> SplitData:
    """Window one already-separated timeline, preventing cross-split leakage."""
    mask = np.isfinite(segment) & (segment != 0)
    clean = np.where(mask, segment, 0.0).astype(np.float32)
    count = len(segment) - input_steps - horizon + 1
    if count <= 0:
        raise ValueError("Segment is too short for requested window")
    xs, ys, xm, ym = [], [], [], []
    for start in range(count):
        middle = start + input_steps
        end = middle + horizon
        xs.append(clean[start:middle])
        ys.append(clean[middle:end, :, 0])
        xm.append(mask[start:middle])
        ym.append(mask[middle:end, :, 0])
    return SplitData(*(np.stack(part) for part in (xs, ys, xm, ym)))


def split_and_window(data: np.ndarray, input_steps: int = 12, horizon: int = 12):
    bounds = chronological_boundaries(len(data))
    segments = [data[a:b] for a, b in bounds]
    train = make_windows(segments[0], input_steps, horizon)
    standardizer = Standardizer.fit(train.x, train.x_mask)

    output = []
    for segment in segments:
        windows = make_windows(segment, input_steps, horizon)
        scaled_x = standardizer.transform(windows.x)
        scaled_x = np.where(windows.x_mask, scaled_x, 0.0).astype(np.float32)
        target_mean = float(np.squeeze(standardizer.mean)[0] if standardizer.mean.size > 1 else standardizer.mean.item())
        target_std = float(np.squeeze(standardizer.std)[0] if standardizer.std.size > 1 else standardizer.std.item())
        scaled_y = ((windows.y - target_mean) / target_std).astype(np.float32)
        output.append(SplitData(scaled_x, scaled_y, windows.x_mask, windows.y_mask))
    return tuple(output), standardizer


def synthetic_traffic(steps=480, nodes=8, features=1, seed=42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    t = np.arange(steps)[:, None]
    base = 55 + 10 * np.sin(2 * np.pi * t / 288 + phase)
    rush = -12 * np.exp(-((t % 288 - 96) / 18) ** 2)
    coupled = np.roll(base, 1, axis=1) * 0.12 + base * 0.88 + rush
    values = coupled + rng.normal(0, 1.2, coupled.shape)
    data = np.repeat(values[..., None], features, axis=-1).astype(np.float32)
    adjacency = np.eye(nodes, dtype=np.float32)
    for i in range(nodes):
        adjacency[i, (i - 1) % nodes] = adjacency[i, (i + 1) % nodes] = 1
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    return data, adjacency


def inject_missing(x: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0,1)")
    rng = np.random.default_rng(seed)
    observed = rng.random(x.shape) >= rate
    return np.where(observed, x, 0.0), observed

