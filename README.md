# Cortex (TRIBE v2 POC)

Applications built on top of Meta's [TRIBE v2](https://huggingface.co/facebook/tribev2) — a tri-modal foundation model that predicts fMRI brain responses to video / audio / text. **Research / non-commercial use only** (CC BY-NC).

## Read first

| Doc | Why |
|---|---|
| [`docs/tribe-v2-overview.md`](docs/tribe-v2-overview.md) | What TRIBE v2 actually is and isn't |
| [`docs/tribe-v2-application-thesis.md`](docs/tribe-v2-application-thesis.md) | Problem-first research brief + recommended thesis |
| [`docs/notebook-walkthrough.md`](docs/notebook-walkthrough.md) | Cell-by-cell summary of the official Colab demo |
| [`docs/colab-runbook.md`](docs/colab-runbook.md) | Copy-pasteable cells to take Colab from zero to ROI traces in ~20 min |

## What's in this repo

| Path | What it is |
|---|---|
| `docs/` | Capability brief, application thesis, notebook walkthrough |
| `backend/` | FastAPI worker around TRIBE v2 — `/predict/{video,audio,text}` + `/diff/video`, ROI-aggregated |
| `frontend/` | Minimal Next.js page that posts two video uploads to `/diff/video` and renders the report |

## 2-week POC plan (from the thesis doc)

1. **Smoke test** — open the [official Colab](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb) on an A100 (Colab Pro+); confirm Sintel-trailer prediction works end-to-end.
2. **Stand up the backend** — `cd backend && pip install -e .` on the same A100; `uvicorn cortex.main:app`. Hit `/predict/video` with a short clip.
3. **Run the frontend locally** — `cd frontend && pnpm install && pnpm dev`. Set `NEXT_PUBLIC_BACKEND_URL` to the Colab ngrok tunnel.
4. **Demo pair** — pick a publicly known A/B ad/trailer pair and screenshot the divergence panel. That screenshot is the asset.

## Honest constraints

- **License.** CC BY-NC weights. A commercial offering needs a Meta licensing conversation. Until then, frame everything as research.
- **Population-average ≠ personalized.** Every UI label and claim must reflect this.
- **No moat at the model layer.** Weights are public. The durable edge has to come from a per-customer calibration layer (pattern G in the thesis doc) trained on real outcome data — collect it from day 1.
