"""LOC-R6 native-grid sparse32 support; image pixels, not inferred map identities."""
import numpy as np


def support_pixels(probability, threshold=.30, max_points=32):
    prob = np.asarray(probability)
    if prob.shape != (320, 320) or not np.isfinite(prob).all():
        raise ValueError("Finite 320x320 probability required")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilities must be in [0,1]")
    if not 0 < threshold < 1 or max_points != 32:
        raise ValueError("The public reference uses exactly the sparse32 sampling budget")
    # 16 x 16 tiles, each containing 20 x 20 native pixels.
    tiles = prob.reshape(16, 20, 16, 20).transpose(0, 2, 1, 3).reshape(256, 400)
    local = tiles.argmax(axis=1)
    values = tiles[np.arange(256), local]
    keep = values >= threshold
    if not keep.any():
        return np.empty((0, 3), dtype=float)
    tile_id, index = np.arange(256)[keep], local[keep]
    candidates = np.column_stack([(tile_id % 16) * 20 + index % 20,
                                  (tile_id // 16) * 20 + index // 20, values[keep]]).astype(float)
    selected = [int(np.argmax(candidates[:, 2]))]
    distance = np.full(len(candidates), np.inf)
    while len(selected) < min(max_points, len(candidates)):
        delta = candidates[:, :2] - candidates[selected[-1], :2]
        distance = np.minimum(distance, np.sum(delta * delta, axis=1))
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return candidates[selected]


def source_coordinates(pixels, width, height):
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 3)
    if min(width, height) <= 0 or not np.isfinite(pixels).all():
        raise ValueError("Finite pixels and positive source dimensions required")
    return np.column_stack([(pixels[:, :2] + .5) * [width / 320, height / 320] - .5, pixels[:, 2]])
