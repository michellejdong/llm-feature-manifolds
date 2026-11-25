"""Low-dimensional geometry scores used in the blind ranking."""

from __future__ import annotations

import numpy as np


def adjacent_pc_planes(points: np.ndarray) -> list[np.ndarray]:
    centered = points - points.mean(axis=0)
    _, scales, axes = np.linalg.svd(centered, full_matrices=False)
    if not len(scales) or scales[0] == 0:
        return []
    rank = min(5, np.count_nonzero(scales > scales[0] * 1e-6))
    scores = centered @ axes[:rank].T
    return [scores[:, i:i + 2] for i in range(rank - 1)]


def arrangement(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.stack([np.cos(angles), np.sin(angles)])
    b = np.stack([-np.sin(angles), np.cos(angles)])
    return a, b


def separability(xy: np.ndarray, angles: int = 32, bins: int = 12) -> float:
    """Minimum two-axis binned mutual information across an angle grid."""
    scale = np.sqrt(np.square(xy).mean())
    if scale < 1e-10:
        return 0.0
    a, b = arrangement(np.linspace(0, np.pi / 2, angles, endpoint=False))
    x = np.clip(((xy @ a / scale + 3) * (bins / 6)).astype(np.int32), 0, bins - 1)
    y = np.clip(((xy @ b / scale + 3) * (bins / 6)).astype(np.int32), 0, bins - 1)
    keys = x + bins * y + bins ** 2 * np.arange(angles)
    joint = np.bincount(keys.ravel(), minlength=angles * bins ** 2).reshape(angles, bins, bins)
    actual = joint / len(xy)
    expected = actual.sum(axis=1)[:, None, :] * actual.sum(axis=2)[:, :, None]
    observations = actual > 0
    terms = np.zeros_like(actual)
    terms[observations] = actual[observations] * np.log2(actual[observations] / expected[observations])
    return float(terms.sum(axis=(1, 2)).min())


def mixture(xy: np.ndarray, epsilon: float = 0.10, angles: int = 32) -> float:
    """Largest mass in an affine band over grid directions and candidate centers."""
    a, _ = arrangement(np.linspace(0, np.pi, angles, endpoint=False))
    projected = np.sort(xy @ a, axis=0)
    variance = np.square(projected).mean(axis=0)
    max_count = 0
    for col, scale in zip(projected.T, variance, strict=True):
        middle = np.concatenate([col, (col[1:] + col[:-1]) / 2])
        width = epsilon * np.sqrt(scale + np.square(middle))
        lo = np.searchsorted(col, middle - width, side="left")
        hi = np.searchsorted(col, middle + width, side="right")
        max_count = max(max_count, int((hi - lo).max()))
    return max_count / len(xy)


def irreducibility(points: np.ndarray) -> float:
    """Average the Engels et al. score over adjacent planes in the first five PCs."""
    planes = adjacent_pc_planes(points)
    values = [(1 - mixture(xy)) * separability(xy) for xy in planes]
    return float(np.mean(values)) if values else 0.0
