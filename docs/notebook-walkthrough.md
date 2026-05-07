# TRIBE v2 — Official Demo Notebook Walkthrough

A cell-by-cell summary of `facebookresearch/tribev2/blob/main/tribe_demo.ipynb`. Read this before forking; everything below is what the official demo actually does (verified against the raw `.ipynb` JSON, May 2026).

## What the notebook does, end-to-end

1. Installs the `tribev2` package (with the `[plotting]` extras) via `uv pip install`.
2. Loads the pretrained model + a `PlotBrain` visualizer.
3. Downloads the **Sintel trailer** (Blender open movie) and predicts cortical activity for it.
4. Visualizes 15 seconds of predicted activity on a 3D brain surface, with stimulus frames shown beneath.
5. Repeats the same flow for a passage from **Hamlet's** "To be or not to be" — text → gTTS speech → WhisperX transcription → predict → visualize.

Total: 18 cells. ~1 GB model download on first run.

## Cell-by-cell

| Cell | Type | What it does |
|---|---|---|
| 0 | md | Title / one-paragraph framing of TRIBE v2 (LLaMA 3.2 + V-JEPA2 + Wav2Vec-BERT → fsaverage5 ~20k vertices). |
| 1 | md | Colab setup note: enable GPU, run install cell, restart runtime. |
| 2 | code | Install: `!uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"`. Note `uv`, not plain `pip`. |
| 3 | md | Loading-the-model intro; ~1 GB checkpoint downloaded once and cached. |
| 4 | code | **Model load** — see snippet below. Also constructs a `PlotBrain(mesh="fsaverage5")` for 3D viz. |
| 5 | md | Explains the video pipeline: extract audio → **WhisperX** transcribes speech with word-level timestamps → visual features via **DINOv2 + V-JEPA2** → audio features via **Wav2Vec-BERT** → text features via **LLaMA 3.2** → predict fMRI at **1 TR = 1 second**. |
| 6 | code | Downloads `sintel_trailer-480p.mp4`, calls `model.get_events_dataframe(video_path=...)`, displays the events table (type, start, duration, filepath, text, context). |
| 7 | md | Notes the LLaMA 3.2 access requirement on Hugging Face. Defines output shape `(n_timesteps, n_vertices)` ≈ `(seconds, ~20k)`. |
| 8 | code | **Predict call** — see snippet below. |
| 9 | md | Visualization explainer. **Important:** "Predictions are offset by 5 seconds in the past, in order to compensate for the hemodynamic lag." Visual cortex lights up at t=4s; language network at t=12s when a character speaks. |
| 10 | code | **Visualization call** — see snippet below. Plots first 15 timesteps with stimulus frames. |
| 11 | md | Text-input pipeline intro: text → gTTS speech → WhisperX → events. |
| 12 | code | Writes the Hamlet passage to `cache/shakespeare.txt`, then `model.get_events_dataframe(text_path=...)`. |
| 13 | md | "Same as before — call `model.predict()`." |
| 14 | code | Re-runs `model.predict(events=df)`; same `(n_timesteps, n_vertices)` output. |
| 15 | md | Audio-only viz note: stimulus display shows spoken words at each timestep instead of a frame. |
| 16 | code | Repeats the `plotter.plot_timesteps(...)` call for the text/audio prediction. |
| 17 | code | Empty trailing cell. |

## Key code snippets (verbatim)

**Install (Colab)**
```bash
!uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

**Model load**
```python
from tribev2.demo_utils import TribeModel, download_file
from tribev2.plotting import PlotBrain
from pathlib import Path

CACHE_FOLDER = Path("./cache")

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_FOLDER,
)
plotter = PlotBrain(mesh="fsaverage5")
```

**Predict (video)**
```python
df = model.get_events_dataframe(video_path=video_path)
preds, segments = model.predict(events=df)
print(f"Predictions shape: {preds.shape}")  # (n_timesteps, n_vertices) -> (seconds, ~20k)
```

**Predict (text → speech → predict)**
```python
text_path = CACHE_FOLDER / "shakespeare.txt"
text_path.write_text(text)
df = model.get_events_dataframe(text_path=text_path)
preds, segments = model.predict(events=df)
```

**Visualization**
```python
fig = plotter.plot_timesteps(
    preds[:n_timesteps],
    segments=segments[:n_timesteps],
    cmap="fire",
    norm_percentile=99,
    vmin=.6,
    alpha_cmap=(0, .2),
    show_stimuli=True,
)
```

## Non-obvious gotchas

- **LLaMA 3.2 is gated.** First `model.predict()` will fail with a 401 unless you've run `huggingface-cli login` with a read token *and* clicked through the LLaMA 3.2 access form on Hugging Face.
- **GPU memory:** Colab T4 (16 GB) OOMs when LLaMA loads. Use **A100 40 GB** minimum.
- **Hemodynamic lag:** raw output is shifted 5 s back vs. the stimulus timeline. Bake the offset into any UI that maps "moments that matter" back to stimulus seconds.
- **Time resolution is 1 Hz.** A 60-second clip = 60 prediction rows. No sub-second resolution.
- **`uv` is the notebook's installer.** If forking outside Colab, `pip install -e .` from the cloned repo is the equivalent.
- **WhisperX runs implicitly** when you pass `video_path` or `text_path` — no separate setup needed, but it's a hidden dependency that adds to first-call latency.
- **Visualization needs the `[plotting]` extras** (PyVista + Nilearn). If you only want the prediction tensor (e.g., a backend service), the bare install is enough.

## What you'd need to fork this for a custom application

1. **Strip the visualization** for a backend/API path; keep only `from_pretrained`, `get_events_dataframe`, `predict`. Visualization belongs in the front-end.
2. **Add an ROI aggregation step** on top of `preds[:, :]` — collapse the 20k vertices to a small set of named regions using a Schaefer/Glasser atlas mapped to fsaverage5. This is what makes outputs interpretable to non-neuroscientists.
3. **Bake in the 5-second hemodynamic offset** when surfacing timestamps to users.
4. **Cache `model` and `plotter` instances** at process start — model load is slow (~10–30 s) and you do not want to repeat it per request.
5. **Decide your input contract** — which of `video_path` / `audio_path` / `text_path` your app accepts, and whether you'll auto-detect or expose all three. The official `get_events_dataframe` handles all three; pick the surface area that fits your use case.

## Suggested first run (smoke test)

1. Open the [official Colab](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb).
2. Runtime → Change runtime type → **A100**.
3. Run cell 2; **restart runtime** when prompted.
4. Run `huggingface-cli login` with a token that has access to `meta-llama/Llama-3.2-3B`.
5. Run cells 3 onward. The Sintel trailer prediction should complete in a few minutes; you should see a brain visualization with the visual cortex lighting up around t=4s.
