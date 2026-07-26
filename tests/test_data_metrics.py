import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarts.data import chronological_boundaries, synthetic_traffic, split_and_window
from quarts.metrics import (
    interval_metrics,
    laplace_crps,
    laplace_nll,
    mae,
    rmse,
    weighted_interval_score,
)
from quarts.statistics import cliffs_delta, paired_block_bootstrap


def test_chronological_boundaries_do_not_overlap():
    bounds = chronological_boundaries(100)
    assert bounds == ((0, 70), (70, 80), (80, 100))


def test_window_shapes_and_split_isolation():
    data, _ = synthetic_traffic(steps=200, nodes=5)
    (train, valid, test), _ = split_and_window(data, input_steps=12, horizon=3)
    assert train.x.shape[1:] == (12, 5, 1)
    assert train.y.shape[1:] == (3, 5)
    assert len(train.x) == 140 - 12 - 3 + 1
    assert len(valid.x) == 20 - 12 - 3 + 1
    assert len(test.x) == 40 - 12 - 3 + 1


def test_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 3.0, 1.0])
    assert np.isclose(mae(y, pred), 1.0)
    assert np.isclose(rmse(y, pred), np.sqrt(5 / 3))
    intervals = interval_metrics(y, y - 1, y + 1)
    assert intervals["coverage"] == 1.0
    assert intervals["mean_width"] == 2.0


def test_probabilistic_metrics_keep_multidimensional_masks_aligned():
    y = np.arange(24, dtype=float).reshape(2, 3, 4)
    location = y + 0.5
    scale = np.ones_like(y)
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0, 0] = False
    assert np.isfinite(laplace_nll(y, location, scale, mask))
    result = interval_metrics(y, location - 1, location + 1, mask)
    assert result["coverage"] == 1.0


def test_crps_and_weighted_interval_score_known_values():
    y = np.array([0.0, 0.0])
    location = np.array([0.0, 0.0])
    scale = np.array([2.0, 2.0])
    assert np.isclose(laplace_crps(y, location, scale), 0.5)
    intervals = {
        0.1: (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        0.5: (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
    }
    assert weighted_interval_score(y, location, intervals) > 0


def test_statistics_helpers():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    assert cliffs_delta(a, b) < 0
    result = paired_block_bootstrap(a, b, np.array([0, 0, 1, 1]), iterations=100, seed=1)
    assert np.isclose(result["estimate"], -1.0)
