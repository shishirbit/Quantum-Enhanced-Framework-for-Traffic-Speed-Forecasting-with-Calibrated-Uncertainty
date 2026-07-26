"""Small end-to-end run; it creates no publishable scientific result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from quarts.data import synthetic_traffic, split_and_window
from quarts.metrics import mae, rmse
from quarts.reproducibility import environment_manifest, set_seed


def main() -> None:
    set_seed(42)
    data, adjacency = synthetic_traffic(steps=480, nodes=8, seed=42)
    (train, valid, test), _ = split_and_window(data, input_steps=12, horizon=12)
    # Persistence validates shapes, masks, metrics and split integrity without
    # pretending that a model has been trained.
    prediction = np.repeat(test.x[:, -1, :, 0][:, None, :], 12, axis=1)
    report = {
        "status": "structural smoke test only",
        "train_windows": len(train.x),
        "valid_windows": len(valid.x),
        "test_windows": len(test.x),
        "adjacency_shape": list(adjacency.shape),
        "prediction_shape": list(prediction.shape),
        "persistence_scaled_mae": mae(test.y, prediction, test.y_mask),
        "persistence_scaled_rmse": rmse(test.y, prediction, test.y_mask),
        "environment": environment_manifest(),
    }
    output = ROOT / "artifacts" / "smoke_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
