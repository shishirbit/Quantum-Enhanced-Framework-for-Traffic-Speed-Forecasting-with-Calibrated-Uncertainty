from __future__ import annotations

import numpy as np


def _valid_mask(y, *predictions, mask=None):
    y = np.asarray(y)
    valid = np.ones_like(y, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    valid &= np.isfinite(y)
    for prediction in predictions:
        prediction = np.asarray(prediction)
        if prediction.shape != y.shape:
            raise ValueError(f"Shape mismatch: target {y.shape}, prediction {prediction.shape}")
        valid &= np.isfinite(prediction)
    return valid


def _valid(y, pred, mask=None):
    y, pred = np.asarray(y), np.asarray(pred)
    valid = _valid_mask(y, pred, mask=mask)
    return y[valid], pred[valid]


def mae(y, pred, mask=None) -> float:
    y, pred = _valid(y, pred, mask)
    return float(np.mean(np.abs(y - pred)))


def rmse(y, pred, mask=None) -> float:
    y, pred = _valid(y, pred, mask)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mape(y, pred, mask=None, epsilon=1.0) -> float:
    y, pred = _valid(y, pred, mask)
    keep = np.abs(y) >= epsilon
    return float(100 * np.mean(np.abs((y[keep] - pred[keep]) / y[keep])))


def wape(y, pred, mask=None) -> float:
    y, pred = _valid(y, pred, mask)
    return float(100 * np.sum(np.abs(y - pred)) / np.sum(np.abs(y)))


def laplace_nll(y, location, scale, mask=None) -> float:
    y, location, scale = np.asarray(y), np.asarray(location), np.asarray(scale)
    valid = _valid_mask(y, location, scale, mask=mask)
    y, location, scale = y[valid], location[valid], np.maximum(scale[valid], 1e-6)
    return float(np.mean(np.log(2 * scale) + np.abs(y - location) / scale))


def laplace_crps(y, location, scale, mask=None) -> float:
    """Continuous ranked probability score for a Laplace forecast."""
    y, location, scale = np.asarray(y), np.asarray(location), np.asarray(scale)
    valid = _valid_mask(y, location, scale, mask=mask)
    error = np.abs(y[valid] - location[valid])
    scale = np.maximum(scale[valid], 1e-6)
    score = error + scale * np.exp(-error / scale) - 0.75 * scale
    return float(np.mean(score))


def interval_metrics(y, lower, upper, mask=None, alpha=0.1) -> dict[str, float]:
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    valid = _valid_mask(y, lower, upper, mask=mask)
    y, lower, upper = y[valid], lower[valid], upper[valid]
    width = upper - lower
    covered = (y >= lower) & (y <= upper)
    score = width + (2 / alpha) * (lower - y) * (y < lower) + (2 / alpha) * (y - upper) * (y > upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "interval_score": float(np.mean(score)),
    }


def weighted_interval_score(y, location, intervals, mask=None) -> float:
    """Weighted interval score using a median and central intervals.

    ``intervals`` maps alpha to ``(lower, upper)`` arrays. The conventional
    weights are 0.5 for the median error and alpha/2 for each interval score.
    """
    y, location = np.asarray(y), np.asarray(location)
    predictions = [location]
    for lower, upper in intervals.values():
        predictions.extend([lower, upper])
    valid = _valid_mask(y, *predictions, mask=mask)
    observed, median = y[valid], location[valid]
    aggregate = 0.5 * np.abs(observed - median)
    for alpha, (lower, upper) in intervals.items():
        lower, upper = np.asarray(lower)[valid], np.asarray(upper)[valid]
        score = (
            upper
            - lower
            + (2 / alpha) * (lower - observed) * (observed < lower)
            + (2 / alpha) * (observed - upper) * (observed > upper)
        )
        aggregate += (alpha / 2) * score
    return float(np.mean(aggregate / (len(intervals) + 0.5)))
