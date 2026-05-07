# Cortex backend

FastAPI worker around Meta's [TRIBE v2](https://huggingface.co/facebook/tribev2). Predicts cortical activity for a video / audio / text stimulus, collapses it to ~7 named cortical networks, and exposes a paired-stimulus diff endpoint.

> **License note:** the TRIBE v2 weights are CC BY-NC. Treat this code and any deployment as research / non-commercial.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/health` | — | `{"status": "ok"}` |
| POST | `/predict/video` | `file` (mp4/mov/webm) | per-network activity over time |
| POST | `/predict/audio` | `file` (wav/mp3/flac) | per-network activity over time |
| POST | `/predict/text` | `text` (form field) | per-network activity over time |
| POST | `/diff/video` | `file_a`, `file_b` | per-network delta + top divergent moments |

All `/predict/*` responses use `(n_timesteps, n_networks)` where `n_timesteps` is the duration of the stimulus in seconds (TRIBE v2 outputs at 1 TR = 1 s, offset 5 s back to compensate for hemodynamic lag).

## Setup (Colab Pro+ on an A100, recommended)

```bash
git clone <this-repo>
cd backend
pip install -e .
huggingface-cli login   # account must have access to meta-llama/Llama-3.2-3B
```

## Run

```bash
uvicorn cortex.main:app --reload --port 8080
```

First request triggers a ~1 GB model download + ~10–30 s warm-up; subsequent requests reuse the cached singleton.

To expose the backend to a remote frontend (e.g. a Vercel-hosted Next.js page) when running on Colab, tunnel via `ngrok http 8080` or similar.

## Layout

| File | Role |
|---|---|
| `src/cortex/main.py` | FastAPI app and route handlers |
| `src/cortex/inference.py` | TribeModel singleton + thin `predict_*` helpers |
| `src/cortex/rois.py` | Collapse 20 484 vertices → 7 Yeo cortical networks via Schaefer-2018 (atlas auto-downloaded on first call, cached under `./cache/atlas/`) |
| `src/cortex/diff.py` | Pairwise diff between two prediction tensors |
| `src/cortex/schemas.py` | Pydantic request / response models |

## Known TODOs

- DTW alignment for `/diff/video` when the two clips have different durations (currently truncates to the shorter timeline).
- Auth + rate limiting before any non-local exposure.
