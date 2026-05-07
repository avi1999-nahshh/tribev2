# TRIBE v2 — Capability Brief

*A primer to help us decide what to build. ~1.5 pages.*

## What it is

**TRIBE v2** is a foundation model from Meta AI (released March 26, 2026) that predicts **how the human brain will respond** to a given stimulus — a video clip, an audio track, or a piece of text. It does **not** generate text, images, or audio. Think of it as a "digital twin" of fMRI brain activity: feed it stimuli, and it returns a high-resolution map of predicted neural responses over time.

Weights and code are public under a **CC BY-NC** license — research and personal use only, no commercial deployment.

## What it can do

- **Inputs (any of):** video (`.mp4`), audio (`.wav`), or text (`.txt`). Text is auto-converted to speech and aligned at the word level before encoding.
- **Output:** a tensor of shape `(n_timesteps, n_vertices)` — predicted fMRI activity across **~20,000 vertices** on the standard `fsaverage5` cortical surface, sampled at **1 TR = 1 second**. So a 60-second clip → 60 prediction rows. Output is offset 5 seconds back from the stimulus to compensate for hemodynamic lag.
- **Zero-shot generalization** to new subjects, new languages, and new task types — no per-user fine-tuning needed.
- **~70× higher spatial resolution** and **2–3× higher accuracy** than prior brain-response models, per Meta's benchmarks.
- Distinguishes how *adjacent* brain regions respond to different facets of the same stimulus (e.g. visual motion vs. dialogue vs. emotional tone).

## How it works (short version)

A Transformer-based architecture that fuses three pretrained encoders, one per modality:

| Modality | Encoder |
|---|---|
| Text | LLaMA 3.2-3B |
| Video | DINOv2 + V-JEPA2 |
| Audio | Wav2Vec-BERT |
| Speech transcription | WhisperX (word-level timings) |

Trained on **500+ hours of fMRI recordings from 700+ volunteers** watching/listening to/reading a wide variety of media. Meta calls the resulting use case "**in-silico neuroscience**" — running brain experiments in software instead of a scanner.

## License & access

- **License:** CC BY-NC (non-commercial, attribution required).
- **Weights:** [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) on Hugging Face.
- **Code:** [`facebookresearch/tribev2`](https://github.com/facebookresearch/tribev2) on GitHub.
- **Paper:** [A foundation model of vision, audition, and language for in-silico neuroscience](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/).
- **Try-before-installing:** [interactive demo](https://aidemos.atmeta.com/tribev2) and a Colab notebook linked from the HF model card.

## What you need to run it locally

**Software**
- [ ] Python 3.10+
- [ ] Clone `facebookresearch/tribev2`, then `pip install -e .`
  - `pip install -e ".[plotting]"` for brain-surface visualization (PyVista + Nilearn)
  - `pip install -e ".[training]"` if we ever want to fine-tune
- [ ] Hugging Face account + **read access token** — `huggingface-cli login`
  - Required because LLaMA 3.2-3B is a *gated* model and TRIBE v2 pulls it as a dependency.

**Hardware**
- [ ] **GPU with ≥40 GB VRAM** — community reports confirm a Colab **T4 (16 GB) runs out of memory** when `model.predict()` loads LLaMA 3.2-3B. Use an **A100 40 GB** (or A100 80 GB "High RAM" for headroom). Cheapest path: Colab Pro / Pro+ which gives access to A100s on demand, before deciding whether to commit to local hardware.

**Inputs to play with**
- [ ] A handful of short clips: `.mp4` (15–60 s), `.wav`, and `.txt` snippets.

**Minimal inference snippet** (from the HF model card):

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
# or: text_path="path/to/text.txt"
# or: audio_path="path/to/audio.wav"

preds, segments = model.predict(events=df)
print(preds.shape)   # (n_timesteps, ~20_000)
```

## Public notebooks & walkthroughs

Several runnable references already exist — useful both for first-touch validation and as code starting points:

- **Official demo** ([Colab](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb) / [GitHub source](https://github.com/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)) — loads the pretrained model, predicts brain responses to a video clip and to text-via-audio, and visualizes activity on a 3D cortical surface. Best starting point.
- **DataCamp tutorial** — [TRIBE v2 Tutorial: Deep Learning Brain Activity Simulation](https://www.datacamp.com/tutorial/tribe-v2-tutorial). End-to-end Colab walk-through; predicts activity from naturalistic stimuli and renders interactive 3D heatmaps.
- **YouTube setup walk-through** — [How to set up TRIBE v2 (Google Colab)](https://www.youtube.com/watch?v=zTAuaRH_Btc). Step-by-step for the Colab environment, HF login, and the gated-LLaMA gotcha.
- **Community write-up** — [Exploring Meta's TRIBE v2 Model](https://medium.com/@nmillrr/exploring-metas-tribe-v2-model-46d9524448cb) by Nathan Miller. Hands-on notes on running the model against social-media-style content.

**Suggested first move:** open the official Colab on an A100, run it end-to-end against a sample clip, and confirm we can produce a `(n_timesteps, ~20k vertices)` prediction tensor + a brain-surface visualization. That gives us a known-good baseline to fork for whichever app direction we pick.

## App directions worth exploring (research-only)

Because output is *predicted brain activity*, useful apps are ones where knowing the neural response is the value:

1. **Content-resonance scoring** — rank video / audio / text variants by predicted engagement signature in attention, language, or affective regions.
2. **In-silico A/B testing of creative** — compare two cuts of a clip and visualize *where and when* the brain responses diverge.
3. **Engagement heatmap over a timeline** — for a podcast or video, surface the moments that drive the strongest predicted response, and in which cortical regions.
4. **Pre-screening stimuli for real fMRI studies** — researchers narrow a candidate set before booking expensive scanner time.
5. **Cross-modal comparison** — feed the same content as audio vs. captioned text and ask: do they elicit similar predicted responses, or meaningfully different ones?

Once we agree on a direction, the next step is to confirm the local setup with the Colab notebook, then scaffold a small Python project around `TribeModel.predict(...)` plus a visualization layer.

## Sources

- [Meta AI blog — Introducing TRIBE v2](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/)
- [Hugging Face model card — facebook/tribev2](https://huggingface.co/facebook/tribev2)
- [GitHub — facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)
- [Research paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
- [Interactive demo](https://aidemos.atmeta.com/tribev2)
- [Official Colab demo notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
- [DataCamp tutorial](https://www.datacamp.com/tutorial/tribe-v2-tutorial)
