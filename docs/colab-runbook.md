# Colab run-book — TRIBE v2 from zero to ROI traces

A copy-pasteable sequence to take you from "open Colab" to "I have predicted brain activity for my own video clip" in ~20 minutes. Works on Colab Pro+ with an A100.

## Prereqs (do these once, outside Colab)

1. **Hugging Face account** with access to [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) — click through the access form, wait for approval (usually instant). Without this, `model.predict()` will 401.
2. **Generate an HF read token** at [hf.co/settings/tokens](https://huggingface.co/settings/tokens). Save it somewhere you can paste from.
3. **Colab Pro+** ($50/mo) — gives you A100 access on demand. T4 (16 GB) **will OOM** when LLaMA loads.

## In Colab — open a fresh notebook

**Runtime → Change runtime type → A100 GPU.** Confirm with the cell below.

### Cell 1 — sanity-check the GPU

```python
!nvidia-smi
```

Expect: `NVIDIA A100-SXM4-40GB` (or 80 GB) and ~40 GB free memory. If you see T4/V100, change the runtime.

### Cell 2 — install TRIBE v2 (with plotting extras)

```python
!uv pip install --system "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

~2–3 min. **Then: Runtime → Restart session.** Don't skip — the new packages don't bind until you restart.

### Cell 3 — log in to Hugging Face (LLaMA is gated)

```python
from huggingface_hub import login
login()  # paste your read token when prompted
```

### Cell 4 — load the model + plotter

```python
from pathlib import Path
from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain

CACHE = Path("./cache")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
plotter = PlotBrain(mesh="fsaverage5")
```

First run downloads ~1 GB of weights and pulls LLaMA 3.2-3B. ~3–5 min. Reuses cache thereafter.

### Cell 5 — smoke test on the Sintel trailer

```python
from tribev2.demo_utils import download_file

video = CACHE / "sintel.mp4"
download_file("https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4", video)

df = model.get_events_dataframe(video_path=video)
preds, segments = model.predict(events=df)
print("preds:", preds.shape)   # (n_seconds, ~20484)
```

Expected: a `(seconds, 20484)` tensor. ~1–3 min for a 60-second clip on A100.

### Cell 6 — visualize first 15 seconds

```python
plotter.plot_timesteps(
    preds[:15], segments=segments[:15],
    cmap="fire", norm_percentile=99, vmin=.6, alpha_cmap=(0, .2),
    show_stimuli=True,
)
```

You should see the visual cortex light up around t=4s and the language network light up when characters speak (~t=12s). If you see this, **the model is working — that's the milestone.**

### Cell 7 — predict on your own clip

```python
from google.colab import files
uploaded = files.upload()                      # pick a short .mp4 (15–60s recommended)
my_video = Path(next(iter(uploaded.keys())))

df = model.get_events_dataframe(video_path=my_video)
preds, segments = model.predict(events=df)
print("preds:", preds.shape)
```

### Cell 8 — quick ROI aggregation (mirrors `backend/src/cortex/rois.py`)

```python
import numpy as np

NETWORKS = (
    "Visual", "Somatomotor", "Dorsal Attention",
    "Salience / Ventral Attention", "Limbic / Affective",
    "Frontoparietal Control", "Default Mode Network",
)
# STUB: equal slices. Replace with Schaefer-400 -> Yeo-7 for real numbers.
boundaries = np.linspace(0, preds.shape[1], len(NETWORKS) + 1, dtype=int)
roi = np.column_stack([
    preds[:, boundaries[i]:boundaries[i+1]].mean(axis=1)
    for i in range(len(NETWORKS))
])
print("roi:", roi.shape)  # (seconds, 7)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
for i, name in enumerate(NETWORKS):
    ax.plot(roi[:, i], label=name)
ax.set_xlabel("seconds (offset 5s back from stimulus)")
ax.set_ylabel("predicted activity")
ax.legend(loc="upper right", fontsize=8)
plt.show()
```

This is the same shape of output the FastAPI backend returns from `/predict/video`. If you want the *real* network labels, swap the stub for Schaefer-2018 (`pip install nilearn`, `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=400)` mapped to fsaverage5).

---

## Optional — serve the local backend from Colab via ngrok

For when you want the Next.js frontend to hit a real backend.

### Cell A — clone & install your repo

The repo is **private**, so cloning from Colab needs an HTTPS URL with a token. Generate a fine-scoped GitHub token (`Contents: read`) at [github.com/settings/personal-access-tokens](https://github.com/settings/personal-access-tokens) and paste it where indicated.

```python
import getpass
GH_TOKEN = getpass.getpass("GitHub token: ")
!git clone https://oauth2:{GH_TOKEN}@github.com/avi1999-nahshh/tribev2.git /content/tribev2
%cd /content/tribev2/backend
!uv pip install --system -e .
!pip install pyngrok
```

(If you flip the repo to public via `gh repo edit avi1999-nahshh/tribev2 --visibility public`, drop the token and clone with the plain HTTPS URL.)

### Cell B — start uvicorn + ngrok tunnel

```python
import subprocess, threading, time
from pyngrok import ngrok

# put your ngrok authtoken (free tier is fine) at ngrok.com/dashboard
!ngrok config add-authtoken YOUR_NGROK_TOKEN

threading.Thread(
    target=lambda: subprocess.run(
        ["uvicorn", "cortex.main:app", "--host", "0.0.0.0", "--port", "8080"]
    ),
    daemon=True,
).start()
time.sleep(3)
public_url = ngrok.connect(8080).public_url
print("backend:", public_url)
```

Paste `public_url` into your local `frontend/.env.local`:

```
NEXT_PUBLIC_BACKEND_URL=https://abcd-1234.ngrok-free.app
```

Run `pnpm dev` locally and the upload-two-videos flow now hits TRIBE on the Colab A100.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: 401` when calling `model.predict()` | LLaMA 3.2-3B access not granted, or token missing | Click through the access form; re-run `login()` with a token that has read access |
| `CUDA out of memory` on first predict | Wrong runtime — T4 instead of A100 | Runtime → Change runtime type → A100; restart session |
| `AttributeError: tribev2 has no module ...` immediately after install | Forgot to restart runtime after Cell 2 | Runtime → Restart session, re-run from Cell 3 |
| `model.predict()` hangs for >10 min on a 30s clip | First call also downloads ancillary models (V-JEPA2, Wav2Vec-BERT, WhisperX) | Be patient; subsequent calls are 1–3 min |
| ngrok tunnel disconnects after ~2 hours | Free-tier session limit | Restart Cell B, update `.env.local` with the new URL |

## What "done" looks like

1. Cell 6 produces a brain-surface plot where visual cortex visibly lights up.
2. Cell 8 produces a 7-line plot per network across your own clip.
3. (Optional) The Next.js frontend, pointed at the ngrok URL, returns a diff report for two uploaded clips.

That third one is the screenshot you want for any stakeholder demo.
