from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_ecc(
    a_log: np.ndarray,
    k: int = 32,
    q_min: float = 0.05,
    q_max: float = 0.95,
) -> np.ndarray:
    n = a_log.shape[0]
    tri = np.triu_indices(n, k=1)
    vals = a_log[tri]
    pos = vals[vals > 0]
    if pos.size == 0:
        return np.zeros(k, dtype=np.float32)
    quantiles = np.linspace(q_min, q_max, k)
    thresholds = np.quantile(pos, quantiles).astype(np.float32)
    ecc = np.zeros(k, dtype=np.float32)
    for i, tau in enumerate(thresholds):
        b = (a_log >= tau).astype(np.float32)
        np.fill_diagonal(b, 0.0)
        e = float(np.sum(np.triu(b, k=1)))
        tri_count = float(np.trace(b @ b @ b) / 6.0)
        ecc[i] = float(n) - e + tri_count
    return ecc


def ecc_thresholds(
    a_log: np.ndarray,
    k: int = 32,
    q_min: float = 0.05,
    q_max: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    n = a_log.shape[0]
    tri = np.triu_indices(n, k=1)
    vals = a_log[tri]
    pos = vals[vals > 0]
    if pos.size == 0:
        return np.zeros(k, dtype=np.float32), np.zeros(k, dtype=np.float32)
    quantiles = np.linspace(q_min, q_max, k)
    thresholds = np.quantile(pos, quantiles).astype(np.float32)
    ecc = np.zeros(k, dtype=np.float32)
    for i, tau in enumerate(thresholds):
        b = (a_log >= tau).astype(np.float32)
        np.fill_diagonal(b, 0.0)
        e = float(np.sum(np.triu(b, k=1)))
        tri_count = float(np.trace(b @ b @ b) / 6.0)
        ecc[i] = float(n) - e + tri_count
    return thresholds, ecc
