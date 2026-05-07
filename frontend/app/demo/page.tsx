// Static preview of the result UI with mock data — no backend required.
// Useful for design review / sharing without spinning up the Colab worker.

const MOCK = {
  n_timesteps: 60,
  abs_delta_total: {
    "Visual": 2.413,
    "Somatomotor": 0.821,
    "Dorsal Attention": 1.972,
    "Salience / Ventral Attention": 1.443,
    "Limbic / Affective": 0.312,
    "Frontoparietal Control": 1.180,
    "Default Mode Network": 2.064,
  },
  top_moments: [
    { t_seconds: 12, network: "Visual", delta: 0.471 },
    { t_seconds: 47, network: "Default Mode Network", delta: -0.382 },
    { t_seconds: 8, network: "Dorsal Attention", delta: 0.298 },
    { t_seconds: 31, network: "Visual", delta: -0.265 },
    { t_seconds: 22, network: "Limbic / Affective", delta: 0.243 },
  ],
};

export default function Demo() {
  return (
    <main>
      <p
        className="muted"
        style={{
          background: "#fff8d6",
          border: "1px solid #f0d97a",
          padding: "8px 12px",
          borderRadius: 6,
        }}
      >
        Static preview with mock data — no backend call. The real flow is at <a href="/">/</a>.
      </p>

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

      <section>
        <h2>Result · {MOCK.n_timesteps}s</h2>

        <h3>Top divergent moments</h3>
        <ul className="report">
          {MOCK.top_moments.map((m, i) => (
            <li key={i}>
              t = {m.t_seconds}s · {m.network} · Δ = {m.delta.toFixed(3)}
            </li>
          ))}
        </ul>

        <h3>Total absolute divergence per network</h3>
        <ul className="report">
          {Object.entries(MOCK.abs_delta_total)
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
    </main>
  );
}
