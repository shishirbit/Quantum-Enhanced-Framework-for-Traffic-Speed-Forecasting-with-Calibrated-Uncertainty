from __future__ import annotations

import numpy as np


def cliffs_delta(a, b) -> float:
    """Pairwise dominance effect size; positive means a tends to be larger."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    greater = sum(np.sum(x > b) for x in a)
    less = sum(np.sum(x < b) for x in a)
    return float((greater - less) / (len(a) * len(b)))


def paired_block_bootstrap(a, b, blocks, iterations=5000, seed=42):
    """Bootstrap mean paired difference by resampling pre-aggregated blocks."""
    a, b, blocks = np.asarray(a), np.asarray(b), np.asarray(blocks)
    if not (a.shape == b.shape == blocks.shape):
        raise ValueError("a, b and blocks must have equal shape")
    labels = np.unique(blocks)
    block_diff = np.array([np.mean(a[blocks == label] - b[blocks == label]) for label in labels])
    rng = np.random.default_rng(seed)
    samples = rng.choice(block_diff, size=(iterations, len(block_diff)), replace=True).mean(axis=1)
    return {
        "estimate": float(block_diff.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def wilcoxon_paired(a, b) -> dict[str, float]:
    from scipy.stats import wilcoxon

    result = wilcoxon(np.asarray(a), np.asarray(b), alternative="two-sided", zero_method="pratt")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}

