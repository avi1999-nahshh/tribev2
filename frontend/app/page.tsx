"use client";

import { useState } from "react";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8080";

type DiffMoment = { t_seconds: number; network: string; delta: number };
type ROISeries = { network: string; activity: number[] };
type DiffResponse = {
  n_timesteps: number;
  networks: ROISeries[];
  abs_delta_total: Record<string, number>;
  top_moments: DiffMoment[];
};

export default function Home() {
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiffResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!a || !b) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file_a", a);
      fd.append("file_b", b);
      const res = await fetch(`${BACKEND_URL}/diff/video`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`backend returned ${res.status}: ${text}`);
      }
      setResult((await res.json()) as DiffResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Cortex — neural diff</h1>
      <p>
        Upload two short video cuts. We predict cortical activity for each (via
        Meta&apos;s TRIBE v2) and surface where their predicted brain responses
        diverge.
      </p>
      <p className="muted">
        Population-average predictions, not personalized. Research /
        non-commercial use only (CC BY-NC weights).
      </p>

      <div className="row">
        <label>
          Cut A
          <input
            type="file"
            accept="video/mp4,video/webm,video/quicktime"
            onChange={(e) => setA(e.target.files?.[0] ?? null)}
          />
        </label>
        <label>
          Cut B
          <input
            type="file"
            accept="video/mp4,video/webm,video/quicktime"
            onChange={(e) => setB(e.target.files?.[0] ?? null)}
          />
        </label>
      </div>

      <button disabled={!a || !b || loading} onClick={submit}>
        {loading ? "predicting…" : "Run diff"}
      </button>

      {error && <p className="error">error: {error}</p>}

      {result && (
        <section>
          <h2>Result · {result.n_timesteps}s</h2>

          <h3>Top divergent moments</h3>
          <ul className="report">
            {result.top_moments.map((m, i) => (
              <li key={i}>
                t = {m.t_seconds}s · {m.network} · Δ = {m.delta.toFixed(3)}
              </li>
            ))}
          </ul>

          <h3>Total absolute divergence per network</h3>
          <ul className="report">
            {Object.entries(result.abs_delta_total)
              .sort((x, y) => y[1] - x[1])
              .map(([k, v]) => (
                <li key={k}>
                  {k}: {v.toFixed(3)}
                </li>
              ))}
          </ul>

          <p className="muted">
            Predicted activity is offset 5s back from the stimulus to compensate
            for hemodynamic lag — &quot;peak at t = 12s&quot; refers to stimulus
            second 7.
          </p>
        </section>
      )}
    </main>
  );
}
