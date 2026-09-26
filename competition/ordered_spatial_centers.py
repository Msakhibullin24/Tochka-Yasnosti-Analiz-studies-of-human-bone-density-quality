"""Joint heatmap decoding with an anatomical order prior, never verification.

Use each level's own scores. Do not sort independently decoded coordinates:
that would silently relabel vertebrae. No visibility inference is provided.
"""
import numpy as np


def decode_ordered(logits, levels=4):
    values = np.asarray(logits, dtype=float)
    if (values.ndim != 4 or not 1 <= levels <= values.shape[1]
            or values.shape[2] < levels or values.shape[3] < 1 or not np.isfinite(values).all()):
        raise ValueError('Invalid ordered heatmaps')
    n, _, h, w = values.shape
    result = np.zeros((n, levels, 2))
    for sample in range(n):
        # Best x conditional on row, still tied to its original level head.
        row_score = values[sample, :levels].max(axis=2)
        xs = values[sample, :levels].argmax(axis=2)
        cost = np.full((levels, h), -np.inf)
        previous = np.full((levels, h), -1, dtype=int)
        cost[0] = row_score[0]
        for level in range(1, levels):
            for y in range(level, h):
                parent = int(cost[level-1, :y].argmax())
                cost[level, y] = cost[level-1, parent]+row_score[level, y]
                previous[level, y] = parent
        y = int(cost[-1].argmax())
        for level in range(levels-1, -1, -1):
            result[sample, level] = [(xs[level, y]+.5)/w, (y+.5)/h]
            y = previous[level, y]
    return result
