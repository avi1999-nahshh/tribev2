# Cortex frontend

Minimal Next.js (App Router) UI for the Cortex POC. One page: upload two video cuts → POST `/diff/video` to the backend → render the per-network divergence + top moments.

## Run locally

```bash
pnpm install   # or npm install / yarn / bun
pnpm dev
```

Visit <http://localhost:3000>.

## Backend URL

Set `NEXT_PUBLIC_BACKEND_URL` in `.env.local` if the FastAPI backend isn't on `http://localhost:8080`. When the backend runs on Colab, expose it with `ngrok http 8080` and point this env var at the ngrok URL.

```
NEXT_PUBLIC_BACKEND_URL=https://your-tunnel.ngrok-free.app
```

## What's intentionally not here yet

- Charts. Right now it just renders top-moments and per-network totals as lists. Add Recharts / `@nivo` when you've validated the inference flow end-to-end.
- A 3D brain-surface viewer. `react-three-fiber` + an `fsaverage5` mesh export from the backend would land that once you've decided it's worth the polish.
- Auth. Don't deploy without it.
