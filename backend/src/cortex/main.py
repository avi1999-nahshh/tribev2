"""Cortex prediction API."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .diff import diff
from .inference import predict_audio, predict_text, predict_video
from .rois import aggregate
from .schemas import DiffMoment, DiffResponse, PredictResponse, ROISeries

app = FastAPI(
    title="Cortex",
    description=(
        "In-silico creative pre-test API on Meta TRIBE v2. "
        "Outputs are population-average predictions of cortical activity. "
        "Research / non-commercial use only (CC BY-NC weights)."
    ),
    version="0.1.0",
)

# Permissive CORS for the local POC. Tighten before any non-local exposure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_SUFFIXES = (".mp4", ".mov", ".webm")
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")


@app.get("/health")
def health():
    return {"status": "ok"}


def _save_upload(upload: UploadFile, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    with tmp as f:
        shutil.copyfileobj(upload.file, f)
    return Path(tmp.name)


def _to_response(preds) -> PredictResponse:
    roi = aggregate(preds)
    return PredictResponse(
        n_timesteps=int(roi.activity.shape[0]),
        n_vertices=int(preds.shape[1]),
        networks=[
            ROISeries(network=name, activity=roi.activity[:, i].tolist())
            for i, name in enumerate(roi.networks)
        ],
    )


def _check_suffix(filename: str | None, allowed: tuple[str, ...]) -> str:
    if not filename:
        raise HTTPException(400, "missing filename")
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"expected one of {allowed}, got {suffix or 'none'}")
    return suffix


@app.post("/predict/video", response_model=PredictResponse)
async def predict_video_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    suffix = _check_suffix(file.filename, VIDEO_SUFFIXES)
    path = _save_upload(file, suffix=suffix)
    try:
        preds, _ = predict_video(path)
    finally:
        path.unlink(missing_ok=True)
    return _to_response(preds)


@app.post("/predict/audio", response_model=PredictResponse)
async def predict_audio_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    suffix = _check_suffix(file.filename, AUDIO_SUFFIXES)
    path = _save_upload(file, suffix=suffix)
    try:
        preds, _ = predict_audio(path)
    finally:
        path.unlink(missing_ok=True)
    return _to_response(preds)


@app.post("/predict/text", response_model=PredictResponse)
async def predict_text_endpoint(text: str = Form(...)) -> PredictResponse:
    if not text.strip():
        raise HTTPException(400, "text must not be empty")
    preds, _ = predict_text(text)
    return _to_response(preds)


@app.post("/diff/video", response_model=DiffResponse)
async def diff_video_endpoint(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
) -> DiffResponse:
    paths: list[Path] = []
    try:
        for f in (file_a, file_b):
            suffix = _check_suffix(f.filename, VIDEO_SUFFIXES)
            paths.append(_save_upload(f, suffix=suffix))
        preds_a, _ = predict_video(paths[0])
        preds_b, _ = predict_video(paths[1])
    finally:
        for p in paths:
            p.unlink(missing_ok=True)

    report = diff(preds_a, preds_b)
    return DiffResponse(
        n_timesteps=int(report.delta.shape[0]),
        networks=[
            ROISeries(network=name, activity=report.delta[:, i].tolist())
            for i, name in enumerate(report.networks)
        ],
        abs_delta_total={
            name: float(v)
            for name, v in zip(report.networks, report.abs_delta_per_network.tolist())
        },
        top_moments=[
            DiffMoment(t_seconds=t, network=net, delta=d)
            for (t, net, d) in report.top_moments
        ],
    )
