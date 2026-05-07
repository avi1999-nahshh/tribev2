"""Pairwise diff between two TRIBE v2 prediction tensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rois import aggregate


@dataclass
class DiffReport:
    networks: list[str]
    delta: np.ndarray                    # (T, N) — B minus A
    abs_delta_per_network: np.ndarray    # (N,)  — sum |delta| per network
    top_moments: list[tuple[int, str, float]]


def diff(preds_a: np.ndarray, preds_b: np.ndarray, top_k: int = 5) -> DiffReport:
    # Truncates to the shorter timeline. DTW for unequal-length stimuli is future work.
    roi_a = aggregate(preds_a)
    roi_b = aggregate(preds_b)
    t = min(roi_a.activity.shape[0], roi_b.activity.shape[0])
    delta = roi_b.activity[:t] - roi_a.activity[:t]
    abs_per_network = np.abs(delta).sum(axis=0)

    flat = np.argsort(np.abs(delta).ravel())[::-1][:top_k]
    moments = [
        (
            int(idx // delta.shape[1]),
            roi_a.networks[int(idx % delta.shape[1])],
            float(delta.ravel()[idx]),
        )
        for idx in flat
    ]
    return DiffReport(
        networks=roi_a.networks,
        delta=delta,
        abs_delta_per_network=abs_per_network,
        top_moments=moments,
    )
