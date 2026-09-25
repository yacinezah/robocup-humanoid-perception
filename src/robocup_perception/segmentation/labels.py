"""Explicit trusted-mask contracts, independent of dataset storage."""
import numpy as np


def decode_mask(mask):
    """Map stored intensities 0/128/255 to background/line/field 0/1/2."""
    a = np.asarray(mask)
    if a.ndim != 2 or not np.isin(a, [0, 128, 255]).all():
        raise ValueError('Expected a two-dimensional 0/128/255 trusted mask')
    result = np.zeros(a.shape, dtype=np.uint8)
    result[a == 128] = 1
    result[a == 255] = 2
    return result


def field_union(mask):
    return (decode_mask(mask) > 0).astype(np.uint8)
