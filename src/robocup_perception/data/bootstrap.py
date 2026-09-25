"""Paired cluster bootstrap for supplied aggregate observations."""
import numpy as np


def paired_cluster_mean_difference(reference, candidate, groups, draws=10000, seed=0):
    """Return candidate-minus-reference mean; resample whole clusters together.

    This observation-weighted estimator is explicit and is not substituted for
    historical trajectory-median or hierarchical-seed estimators.
    """
    a, b, g = np.asarray(reference, float), np.asarray(candidate, float), np.asarray(groups)
    if a.ndim != 1 or a.shape != b.shape or a.shape != g.shape or len(a) == 0:
        raise ValueError('Expected nonempty paired one-dimensional arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite observation')
    keys = np.unique(g)
    if len(keys) < 2 or draws < 100:
        raise ValueError('At least two clusters and 100 resamples required')
    diffs = [b[g == k] - a[g == k] for k in keys]
    rng = np.random.default_rng(seed)
    values = [np.concatenate([diffs[i] for i in rng.integers(len(keys), size=len(keys))]).mean()
              for _ in range(draws)]
    return {'delta': float((b-a).mean()), 'ci95': np.percentile(values, [2.5, 97.5]).tolist(),
            'clusters': len(keys), 'draws': draws, 'seed': seed,
            'estimator': 'observation-weighted mean of paired candidate-minus-reference differences'}
