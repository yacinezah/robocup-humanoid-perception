"""Finite log-weight normalization and the existing clipped gate semantics."""
import numpy as np


def clipped_best(scores, floor=-10.0):
    """One observation's best same-type score. No candidates uses the floor."""
    values = np.asarray(scores, float)
    if not np.isfinite(values).all() or not np.isfinite(floor):
        raise ValueError('Nonfinite likelihood')
    return max(float(np.max(values)), floor) if values.size else float(floor)


def normalize_log_weights(log_weights):
    logw = np.asarray(log_weights, float)
    if logw.ndim != 1 or not len(logw) or not np.isfinite(logw).all():
        raise ValueError('Expected finite nonempty log-weight vector')
    w = np.exp(logw-logw.max())
    return w/w.sum()
