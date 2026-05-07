"""Collapse fsaverage5 vertex-level activity into the 7 Yeo cortical networks.

Uses the Schaefer-2018 400-parcel parcellation on fsaverage5, with each parcel
mapped to one of Yeo's 7 networks. Atlas annotation files are fetched once from
the CBIG GitHub repo and cached under `./cache/atlas/`.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

NETWORK_NAMES: tuple[str, ...] = (
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Salience / Ventral Attention",
    "Limbic / Affective",
    "Frontoparietal Control",
    "Default Mode Network",
)

# Yeo-7 short tokens as they appear in Schaefer label names -> our display names.
_YEO7_TO_DISPLAY: dict[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience / Ventral Attention",
    "Limbic": "Limbic / Affective",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode Network",
}

N_VERTICES_FSAVERAGE5 = 20484
N_VERTICES_PER_HEMI = 10242

_CBIG_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/label"
)
_ATLAS_FILES: dict[str, str] = {
    "lh": "lh.Schaefer2018_400Parcels_7Networks_order.annot",
    "rh": "rh.Schaefer2018_400Parcels_7Networks_order.annot",
}
_ATLAS_DIR = Path("./cache/atlas")


@dataclass
class ROIBreakdown:
    networks: list[str]
    activity: np.ndarray  # (n_timesteps, n_networks)


def _ensure_atlas() -> dict[str, Path]:
    _ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for hemi, fname in _ATLAS_FILES.items():
        local = _ATLAS_DIR / fname
        if not local.exists():
            url = f"{_CBIG_BASE}/{fname}"
            try:
                urllib.request.urlretrieve(url, local)
            except BaseException:
                # urlretrieve writes to the destination directly; remove partial files.
                local.unlink(missing_ok=True)
                raise
        paths[hemi] = local
    return paths


def _yeo7_index_from_label(name: str) -> int:
    # Schaefer label names look like "7Networks_LH_Vis_1" or "7Networks_LH_Default_PFC_1".
    # The Yeo-7 short token is whichever segment matches our display map.
    for token in name.split("_"):
        display = _YEO7_TO_DISPLAY.get(token)
        if display is not None:
            return NETWORK_NAMES.index(display)
    return -1


@lru_cache(maxsize=1)
def _vertex_to_network() -> np.ndarray:
    import nibabel.freesurfer.io as fs

    paths = _ensure_atlas()
    labels = np.full(N_VERTICES_FSAVERAGE5, -1, dtype=np.int32)

    for hemi_idx, hemi in enumerate(("lh", "rh")):
        vertex_labels, _ctab, names = fs.read_annot(str(paths[hemi]))
        net_for_parcel = np.fromiter(
            (
                _yeo7_index_from_label(n.decode("ascii") if isinstance(n, bytes) else n)
                for n in names
            ),
            dtype=np.int32,
            count=len(names),
        )
        offset = hemi_idx * N_VERTICES_PER_HEMI
        labels[offset : offset + len(vertex_labels)] = net_for_parcel[vertex_labels]

    return labels


def aggregate(preds: np.ndarray) -> ROIBreakdown:
    if preds.ndim != 2:
        raise ValueError(f"expected (T, n_vertices), got {preds.shape}")
    labels = _vertex_to_network()
    if preds.shape[1] != labels.shape[0]:
        raise ValueError(
            f"vertex count mismatch: preds has {preds.shape[1]}, atlas has {labels.shape[0]}"
        )
    n_networks = len(NETWORK_NAMES)
    activity = np.zeros((preds.shape[0], n_networks), dtype=preds.dtype)
    for net_idx in range(n_networks):
        mask = labels == net_idx
        if mask.any():
            activity[:, net_idx] = preds[:, mask].mean(axis=1)
    return ROIBreakdown(networks=list(NETWORK_NAMES), activity=activity)
