"""TRIBE v2 model singleton + thin predict helpers."""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE_DIR = Path("./cache")


@lru_cache(maxsize=1)
def _model():
    from tribev2.demo_utils import TribeModel

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
    )


def predict_video(video_path: Path) -> tuple[np.ndarray, list]:
    # preds.shape == (seconds, ~20484); offset 5s back to account for hemodynamic lag.
    model = _model()
    df = model.get_events_dataframe(video_path=str(video_path))
    return model.predict(events=df)


def predict_audio(audio_path: Path) -> tuple[np.ndarray, list]:
    model = _model()
    df = model.get_events_dataframe(audio_path=str(audio_path))
    return model.predict(events=df)


def predict_text(text: str) -> tuple[np.ndarray, list]:
    model = _model()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        text_path = Path(f.name)
    df = model.get_events_dataframe(text_path=str(text_path))
    return model.predict(events=df)
