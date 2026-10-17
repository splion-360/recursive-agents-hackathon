import { useEffect, useState } from "react";

type RunEvidence = {
  run_id: string; task_id: string; status: "completed" | "failed";
  proposal: string; review_decision: "PASS" | "REPLACE";
  accepted_action: string; tool_result: string; eligible_for_consolidation: boolean;
};
type VersionEvidence = {
  version_id: string; role: "ACCEPTED" | "CANDIDATE" | "REJECTED" | "PREVIOUS";
  card_count: number; content_hash: string;
};
type PromotionEvidence = {
  decision: "PROMOTE" | "REJECT"; accepted_before: string; candidate: string;
  accepted_after: string; capability_delta: number; negative_transfer_rate: number;
  failed_gates: string[];
};
type MetricEvidence = {
  condition: "C0" | "C1" | "C2" | "C3"; accuracy: number;
  intervention_rate: number; mean_latency_ms: number; mean_tokens: number;
};
type ReplayEvidence = {
  mode: "REPLAY"; synthetic: true; provenance: string;
  events: Array<{ sequence: number; type: string; task_id: string }>;
};
type Evidence = {
  runs: RunEvidence[]; versions: VersionEvidence[]; promotion: PromotionEvidence;
  metrics: MetricEvidence[]; replay: ReplayEvidence;
};

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Evidence API returned ${response.status}`);
  return response.json() as Promise<T>;
}

export function App() {
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchJson<RunEvidence[]>("/api/evidence/runs"),
      fetchJson<VersionEvidence[]>("/api/evidence/versions"),
      fetchJson<PromotionEvidence>("/api/evidence/promotion"),
      fetchJson<MetricEvidence[]>("/api/evidence/metrics"),
      fetchJson<ReplayEvidence>("/api/evidence/replay"),
    ]).then(([runs, versions, promotion, metrics, replay]) => {
      setEvidence({ runs, versions, promotion, metrics, replay });
    }).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : "Unable to load evidence");
    });
  }, []);

  if (error !== null) return <main className="state-card">Evidence unavailable: {error}</main>;
  if (evidence === null) return <main className="state-card">Loading evidence chain…</main>;
  const run = evidence.runs[0];
  if (run === undefined) return <main className="state-card">Replay contains no runs.</main>;

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Memory-distilled agent team</p>
          <h1>One run becomes<br /><span>a better next run.</span></h1>
          <p className="summary">Follow a Student action through private review, safe execution,
            distilled memory, automatic promotion, and a controlled C0–C3 comparison.</p>
        </div>
        <div className="replay-badge"><i /> SYNTHETIC · {evidence.replay.mode}</div>
      </header>

      <section className="panel trajectory">
        <div className="section-heading"><div><span>01</span><h2>Reviewed trajectory</h2></div><code>{run.task_id}</code></div>
        <div className="flow-grid">
          <article className="flow-card student"><small>STUDENT PROPOSAL</small><strong>Unverified action</strong><p>{run.proposal}</p></article>
          <div className="connector">→</div>
          <article className="flow-card reviewer"><small>PRIVATE REVIEW BOUNDARY</small><strong>{run.review_decision}</strong><p>{run.accepted_action}</p></article>
          <div className="connector">→</div>
          <article className="flow-card tool"><small>ACCEPTED ACTION ONLY</small><strong>Deterministic sandbox</strong><p>{run.tool_result}</p></article>
        </div>
        <p className="privacy-note">Private Reviewer reasoning is intentionally absent from this replay.</p>
      </section>

      <section className="split-layout">
        <div className="panel memory-panel">
          <div className="section-heading"><div><span>02</span><h2>Memory lifecycle</h2></div></div>
          <div className="memory-path">
            <div className="memory-node"><small>PERSIST</small><b>Reviewed Episode</b><em>{run.eligible_for_consolidation ? "eligible" : "ineligible"} · immutable provenance</em></div>
            <div className="down-arrow">↓</div>
            <div className="memory-node accent"><small>CONSOLIDATE</small><b>Behavior Card</b><em>answer-free procedural guidance</em></div>
          </div>
          <div className="versions">{evidence.versions.map((version) => (
            <div className={`version ${version.role.toLowerCase()}`} key={version.version_id}>
              <span>{version.role}</span><b>{version.version_id}</b>
              <small>{version.card_count} cards · {version.content_hash.slice(0, 8)}</small>
            </div>
          ))}</div>
        </div>
        <div className="panel promotion-panel">
          <div className="section-heading"><div><span>03</span><h2>Promotion gate</h2></div></div>
          <div className={`decision ${evidence.promotion.decision.toLowerCase()}`}>{evidence.promotion.decision}</div>
          <dl>
            <div><dt>Capability Δ</dt><dd>{formatSignedPoints(evidence.promotion.capability_delta)}</dd></div>
            <div><dt>Negative transfer</dt><dd>{(evidence.promotion.negative_transfer_rate * 100).toFixed(0)}%</dd></div>
            <div><dt>Accepted pointer</dt><dd>{evidence.promotion.accepted_after}</dd></div>
          </dl>
          <p className="gate-note">
            {evidence.promotion.decision === "PROMOTE"
              ? "All deterministic quality, safety, and efficiency gates passed."
              : `Rejected by: ${evidence.promotion.failed_gates.join(", ")}. Accepted Version unchanged.`}
          </p>
        </div>
      </section>

      <section className="panel metrics-panel">
        <div className="section-heading"><div><span>04</span><h2>Controlled experiment</h2></div><small>Illustrative fixture values — not measured results</small></div>
        <div className="metric-grid">{evidence.metrics.map((metric) => (
          <article className={`metric-card ${metric.condition === "C2" ? "featured" : ""}`} key={metric.condition}>
            <div><b>{metric.condition}</b><span>{conditionLabel(metric.condition)}</span></div>
            <strong>{(metric.accuracy * 100).toFixed(0)}<sup>%</sup></strong>
            <div className="bar"><i style={{ width: `${metric.accuracy * 100}%` }} /></div>
            <small>{(metric.intervention_rate * 100).toFixed(0)}% interventions · {metric.mean_tokens} tokens</small>
          </article>
        ))}</div>
      </section>
      <footer><span>REPLAY PROVENANCE</span>{evidence.replay.provenance}</footer>
    </main>
  );
}

function conditionLabel(condition: MetricEvidence["condition"]): string {
  return { C0: "Student only", C1: "+ Reviewer", C2: "+ Shared Brain", C3: "+ Irrelevant memory" }[condition];
}

function formatSignedPoints(value: number): string {
  const points = value * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(0)} pp`;
}
