import { useEffect, useState } from "react";

import {
  type DashboardSnapshot,
  type RunEvent,
  fetchCalibrationTask,
  fetchDashboard,
  subscribeToRunEvents,
} from "./api";

type View = "overview" | "live" | "calibration" | "memory" | "health";

const VIEW_LABELS: Record<View, string> = {
  overview: "Overview",
  live: "Live Run",
  calibration: "Calibration",
  memory: "Shared Brain",
  health: "System Health",
};

export function App() {
  const [view, setView] = useState<View>("overview");
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = () => {
      fetchDashboard()
        .then((value) => {
          if (!active) return;
          setSnapshot(value);
          setEvents(value.recent_events);
          setError(null);
        })
        .catch((reason: unknown) => {
          if (active) setError(reason instanceof Error ? reason.message : "Dashboard unavailable");
        });
    };
    refresh();
    const timer = window.setInterval(refresh, 2_000);
    const unsubscribe = subscribeToRunEvents(
      (event) => setEvents((current) => mergeEvents(current, [event])),
      setConnected,
    );
    return () => {
      active = false;
      window.clearInterval(timer);
      unsubscribe();
    };
  }, []);

  if (snapshot === null) {
    return <main className="boot-state">{error ?? "Connecting to experiment control plane…"}</main>;
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><BrandMark /><div><strong>Distill</strong><span>Agent laboratory</span></div></div>
        <nav>
          {(Object.keys(VIEW_LABELS) as View[]).map((item) => (
            <button className={view === item ? "active" : ""} key={item} onClick={() => setView(item)}>
              <NavIcon view={item} />{VIEW_LABELS[item]}
              {item === "live" && snapshot.calibration.state === "running" ? <i className="nav-pulse" /> : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className={`connection-dot ${connected ? "online" : ""}`} />
          <div><b>{connected ? "Event stream connected" : "Waiting for event stream"}</b><small>LIVE · FastAPI / SSE</small></div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div><p>MEMORY-DISTILLED AGENT TEAM</p><h1>{VIEW_LABELS[view]}</h1></div>
          <div className="top-actions"><span className="mode-pill"><i /> LIVE</span><span className="run-id">{snapshot.calibration.run_id ?? "NO ACTIVE RUN"}</span></div>
        </header>
        {error ? <div className="error-banner">Refresh warning: {error}</div> : null}
        {view === "overview" ? <Overview snapshot={snapshot} events={events} onNavigate={setView} /> : null}
        {view === "live" ? <LiveRun snapshot={snapshot} events={events} connected={connected} /> : null}
        {view === "calibration" ? <Calibration snapshot={snapshot} events={events} /> : null}
        {view === "memory" ? <SharedBrain snapshot={snapshot} /> : null}
        {view === "health" ? <SystemHealth snapshot={snapshot} /> : null}
      </main>
    </div>
  );
}

function Overview({ snapshot, events, onNavigate }: { snapshot: DashboardSnapshot; events: RunEvent[]; onNavigate: (view: View) => void }) {
  const status = snapshot.calibration;
  const completed = status.discovery_completed + status.regression_completed;
  const progress = Math.round((completed / 16) * 100);
  return <div className="page-stack">
    <section className="hero-card">
      <div className="hero-copy"><span className="kicker">VERSION N → N+1</span><h2>Watch experience become<br/><em>operational memory.</em></h2><p>One evidence chain from ChartQA trajectories through private review, frozen κ guidance, and deterministic promotion.</p><button onClick={() => onNavigate("live")}>{status.state === "running" ? "Watch current forward pass" : "Open live monitor"}<span>→</span></button></div>
      <div className="orb-wrap"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="brain-orb"><span>κ</span><small>{status.kappa_hash ? "FROZEN" : "CALIBRATING"}</small></div></div>
    </section>
    <section className="stat-row">
      <Stat label="Calibration progress" value={`${completed}/16`} hint={`${progress}% · ${pretty(status.stage)}`} tone="mint" />
      <Stat label="Public run events" value={String(snapshot.event_count)} hint={events.length ? pretty(events.at(-1)?.type ?? "waiting") : "Waiting for shell runner"} tone="violet" />
      <Stat label="Accepted memory" value={snapshot.accepted_version} hint={`${snapshot.accepted_card_count} cards · baseline`} tone="amber" />
      <Stat label="Measured results" value="Pending" hint="No synthetic claims" tone="slate" />
    </section>
    <section className="panel pipeline-panel">
      <PanelTitle eyebrow="FORWARD PASS" title="Evidence pipeline" detail="Every step is provider-backed or explicitly pending" />
      <div className="pipeline">
        <PipelineStep number="01" title="Student" label={shortModel(snapshot.models.student)} state={stageReached(status.stage, "discovery")} />
        <PipelineArrow />
        <PipelineStep number="02" title="Private Reviewer" label={shortModel(snapshot.models.reviewer)} state={stageReached(status.stage, "regression")} />
        <PipelineArrow />
        <PipelineStep number="03" title="OpenShell" label="Bounded execution" state={events.some((event) => event.type === "observation")} />
        <PipelineArrow />
        <PipelineStep number="04" title="Shared Brain" label="MemMachine · v000" state={false} />
      </div>
    </section>
    <section className="two-column">
      <ProgressCard snapshot={snapshot} />
      <RecentEvents events={events.slice(-5).reverse()} onNavigate={() => onNavigate("live")} />
    </section>
  </div>;
}

function LiveRun({ snapshot, events, connected }: { snapshot: DashboardSnapshot; events: RunEvent[]; connected: boolean }) {
  const currentTask = snapshot.calibration.current_task_id;
  const traces = taskTraces(events);
  const completed = traces.filter((trace) => trace.some((event) => event.type === "run_completed"));
  const scored = completed.filter((trace) => trace.at(-1)?.correct !== undefined && trace.at(-1)?.correct !== null);
  const correct = scored.filter((trace) => trace.at(-1)?.correct === true).length;
  return <div className="page-stack">
    <section className="live-banner panel"><div><span className={`live-signal ${connected ? "on" : ""}`} /><div><b>{snapshot.calibration.state === "running" ? "Forward pass in progress" : snapshot.calibration.state === "failed" ? "Run stopped" : "Live monitor ready"}</b><small>{currentTask ?? "Start Ticket 14 in the shell; task results will appear automatically."}</small></div></div><Metric label="Stage" value={pretty(snapshot.calibration.stage)} /><Metric label="Tasks" value={String(traces.length)} /><Metric label="Score" value={scored.length ? `${correct}/${scored.length} scored` : "—"} /></section>
    {snapshot.calibration.state === "failed" ? <div className="run-state-note failed"><b>Calibration stopped</b><span>{snapshot.calibration.message}</span></div> : null}
    <section className="panel trace-panel"><PanelTitle eyebrow="PUBLIC TASK RESULTS" title="Compact trajectory monitor" detail="One row per task · Student output, Reviewer decision, and official score" />
      {events.length === 0 ? <EmptyState title="No forward pass yet" body="The SSE connection is ready. Run the calibration command in your shell to populate this table." /> : <TaskTraceTable traces={traces} />}
    </section>
  </div>;
}

function Calibration({ snapshot, events }: { snapshot: DashboardSnapshot; events: RunEvent[] }) {
  const c = snapshot.calibration;
  const decisions = events.filter((event) => event.type === "action_reviewed");
  const replacements = decisions.filter((event) => event.review?.decision === "replace").length;
  return <div className="page-stack">
    <section className="calibration-head"><div><span className="kicker">H-STAR → κ</span><h2>Calibration laboratory</h2><p>Discovery can shape one bounded proposal. Regression may only accept or reject it.</p></div><div className={`state-stamp ${c.state}`}>{pretty(c.state)}<small>{pretty(c.stage)}</small></div></section>
    <section className="split-progress">
      <PhaseCard index="A" title="Discovery" count={c.discovery_completed} total={8} detail="Visible to H-star analysis" locked={false} />
      <div className="isolation-line"><span>ISOLATION BOUNDARY</span></div>
      <PhaseCard index="B" title="Regression" count={c.regression_completed} total={8} detail="Hidden until proposal freezes" locked={c.stage === "waiting" || c.stage === "discovery"} />
    </section>
    <section className="two-column calibration-grid">
      <div className="panel proposal-card"><PanelTitle eyebrow="ONE BOUNDED REVISION" title="H-star proposal" detail={shortModel(snapshot.models.hstar)} />
        {c.proposal_hash ? <><CodeLine label="Proposal" value={c.proposal_id ?? "revision-1"} /><CodeLine label="SHA-256" value={c.proposal_hash} /><RuleList accepted={c.accepted_rule_ids} rejected={c.rejected_rule_ids} /></> : <EmptyState title="Proposal not generated" body="It will appear after all eight discovery trajectories are classified." />}
      </div>
      <div className="panel proposal-card"><PanelTitle eyebrow="FROZEN REVIEW POLICY" title="κ guidance" detail={shortModel(snapshot.models.reviewer)} />
        {c.kappa_hash ? <><CodeLine label="Artifact" value={c.kappa_artifact_id ?? "kappa.v1"} /><CodeLine label="SHA-256" value={c.kappa_hash} /><p className="success-note">Regression screen passed. Exact artifact is frozen for measured Collection.</p></> : <EmptyState title="No κ artifact yet" body="Only rules with discovery benefit, positive regression evidence, action compatibility, and privacy clearance can enter κ." />}
      </div>
    </section>
    <section className="stat-row compact"><Stat label="Reviewer decisions" value={String(decisions.length)} hint="Public decisions only" tone="violet" /><Stat label="REPLACE" value={String(replacements)} hint="Smallest valid replacement" tone="amber" /><Stat label="Tokens" value={formatNumber(c.input_tokens + c.output_tokens)} hint={`${formatNumber(c.input_tokens)} in · ${formatNumber(c.output_tokens)} out`} tone="mint" /><Stat label="Estimated cost" value={`$${c.estimated_cost_usd.toFixed(4)}`} hint="Runner-reported" tone="slate" /></section>
  </div>;
}

function SharedBrain({ snapshot }: { snapshot: DashboardSnapshot }) {
  return <div className="page-stack"><section className="brain-hero panel"><div className="large-orb">N</div><div><span className="kicker">MEMMACHINE OPERATIONAL MEMORY</span><h2>{snapshot.accepted_version}</h2><p>The immutable baseline is active. It contains no Behavior Cards, so the Student retrieves no procedural guidance yet.</p><div className="truth-badge">0 accepted cards · no measured N+1 claim</div></div></section><section className="panel"><PanelTitle eyebrow="VERSION LIFECYCLE" title="Nothing promoted yet" detail="Candidate construction begins after Collection" /><div className="version-rail"><VersionNode role="ACCEPTED" id="accepted-v000" active/><VersionConnector/><VersionNode role="CANDIDATE" id="Pending Collection"/><VersionConnector/><VersionNode role="DECISION" id="Pending Promotion"/></div><div className="warning-box"><b>Atomic pointer limitation</b><p>Managed MemMachine storage is configured, but atomic accepted-pointer compare-and-set is unavailable. The dashboard keeps this limitation visible instead of claiming promotion readiness.</p></div></section></div>;
}

function SystemHealth({ snapshot }: { snapshot: DashboardSnapshot }) {
  return <div className="page-stack"><section className="panel health-panel"><PanelTitle eyebrow="PREFLIGHT" title="Runtime configuration" detail="Configured does not mean live-verified in this process" /><div className="health-list">{snapshot.integrations.map((item) => <article key={item.name}><span className={`health-icon ${item.state}`}>{item.state === "configured" ? "✓" : item.state === "limited" ? "!" : "×"}</span><div><b>{item.name}</b><p>{item.detail}</p></div><em className={item.state}>{item.state}</em></article>)}</div></section><section className="two-column"><div className="panel model-panel"><PanelTitle eyebrow="LOCKED ROSTER" title="Model assignments" detail="Loaded from settings" />{Object.entries(snapshot.models).map(([role, model]) => <CodeLine key={role} label={pretty(role)} value={model ?? "Not configured"} />)}</div><div className="panel model-panel"><PanelTitle eyebrow="DATA ISOLATION" title="Frozen manifests" detail="Actual local manifest hashes" />{snapshot.manifests.map((manifest) => <CodeLine key={manifest.name} label={`${pretty(manifest.name)} · ${manifest.task_count}`} value={manifest.content_sha256} />)}</div></section></div>;
}

function ProgressCard({ snapshot }: { snapshot: DashboardSnapshot }) { const c = snapshot.calibration; return <div className="panel progress-card"><PanelTitle eyebrow="TICKET 14" title="Calibration progress" detail={c.message} /><Progress label="Discovery trajectories" value={c.discovery_completed} total={8} /><Progress label="Regression trajectories" value={c.regression_completed} total={8} /><Progress label="PASS / REPLACE smoke" value={c.smoke_completed} total={c.smoke_total || 5} /><div className="current-step"><span>NOW</span><b>{pretty(c.stage)}</b><small>{c.current_task_id ?? "Waiting for shell command"}</small></div></div>; }
function RecentEvents({ events, onNavigate }: { events: RunEvent[]; onNavigate: () => void }) { return <div className="panel recent-card"><PanelTitle eyebrow="TRACE TAIL" title="Recent events" detail="Sanitized RunEvent projection" />{events.length ? events.map((event, index) => <MiniEvent event={event} key={`${event.task_id}-${event.sequence}-${index}`} />) : <EmptyState title="Stream is quiet" body="No calibration events have been written." />}<button className="text-button" onClick={onNavigate}>Open full trajectory →</button></div>; }
function TaskTraceTable({ traces }: { traces: RunEvent[][] }) {
  const occurrenceByTask = new Map<string, number>();
  const keyed = [...traces].reverse().map((trace) => {
    const start = trace.find((event) => event.type === "run_started");
    const base = `${start?.task_id ?? trace[0]?.task_id ?? "partial"}-${start?.seed ?? "unknown"}`;
    const occurrence = occurrenceByTask.get(base) ?? 0;
    occurrenceByTask.set(base, occurrence + 1);
    return { trace, key: `${base}-${occurrence}` };
  }).reverse();
  return <div className="task-trace"><div className="task-trace-head"><span>Task / policy</span><span>Student proposal</span><span>Reviewer</span><span>Official result</span></div>{keyed.map(({ trace, key }, index) => <TaskTraceRow events={trace} index={index + 1} key={key} />)}</div>;
}
function TaskTraceRow({ events, index }: { events: RunEvent[]; index: number }) {
  const start = events.find((event) => event.type === "run_started");
  const proposal = events.filter((event) => event.type === "action_proposed").at(-1);
  const review = events.filter((event) => event.type === "action_reviewed").at(-1)?.review;
  const completion = events.filter((event) => event.type === "run_completed").at(-1);
  const policy = start?.kappa_hash ? `κ ${start.kappa_hash.slice(0, 8)}` : review ? "H* candidate" : "Baseline";
  const result = resultLabel(completion);
  const taskId = start?.task_id ?? events[0]?.task_id ?? "Unknown task";
  const [expanded, setExpanded] = useState(false);
  const [question, setQuestion] = useState<string | null>(null);
  useEffect(() => {
    if (!expanded || question !== null || taskId === "Unknown task") return;
    let active = true;
    fetchCalibrationTask(taskId)
      .then((task) => { if (active) setQuestion(task.question); })
      .catch(() => { if (active) setQuestion("Question unavailable for this task."); });
    return () => { active = false; };
  }, [expanded, question, taskId]);
  return <article className={`task-trace-row ${result.tone}`}>
    <div className="task-cell identity"><span className="task-number">{String(index).padStart(2, "0")}</span><div><b title={taskId}>{shortTask(taskId)}</b><small>{policy}{start?.seed !== undefined ? ` · seed ${start.seed}` : ""}</small></div></div>
    <div className="task-cell"><small>{proposal ? pretty(proposal.action?.type ?? "action") : "Waiting"}</small><strong title={actionSummary(proposal?.action)}>{actionSummary(proposal?.action)}</strong></div>
    <div className="task-cell"><small>{review ? review.decision.toUpperCase() : "BASELINE"}</small><strong title={reviewSummary(review)}>{reviewSummary(review)}</strong></div>
    <div className="task-cell result"><span className={`result-dot ${result.tone}`} /> <div><strong>{result.label}</strong><small>{completion ? pretty(completion.failure_kind ?? completion.termination_reason ?? "completed") : `${events.length} events received`}</small></div></div>
    <details className="task-chart" onToggle={(event) => setExpanded(event.currentTarget.open)}><summary><span>View task</span><code>{taskId}</code></summary><div><div className="task-question"><small>QUESTION</small><p>{question ?? "Loading question…"}</p></div><img src={`/api/dashboard/tasks/${encodeURIComponent(taskId)}/image`} alt={`Chart for ${taskId}`} loading="lazy" /></div></details>
  </article>;
}
function MiniEvent({ event }: { event: RunEvent }) { return <div className="mini-event"><span>{eventIcon(event.type)}</span><div><b>{pretty(event.type)}</b><small>{eventDescription(event)}</small></div><code>{event.sequence}</code></div>; }
function PhaseCard({ index, title, count, total, detail, locked }: { index: string; title: string; count: number; total: number; detail: string; locked: boolean }) { const percentage = Math.round((count / total) * 100); return <article className={`phase-card ${locked ? "locked" : ""}`}><span>{index}</span><div><small>{locked ? "SEALED" : "CALIBRATION SPLIT"}</small><h3>{title}</h3><p>{detail}</p></div><strong>{count}<i>/{total}</i></strong><div className="phase-bar"><i style={{ width: `${percentage}%` }} /></div></article>; }
function Stat({ label, value, hint, tone }: { label: string; value: string; hint: string; tone: string }) { return <article className={`stat-card ${tone}`}><small>{label}</small><strong>{value}</strong><p>{hint}</p></article>; }
function Metric({ label, value }: { label: string; value: string }) { return <div className="banner-metric"><small>{label}</small><b>{value}</b></div>; }
function Progress({ label, value, total }: { label: string; value: number; total: number }) { const percentage = Math.min(100, Math.round((value / total) * 100)); return <div className="progress-item"><div><span>{label}</span><b>{value}/{total}</b></div><div><i style={{ width: `${percentage}%` }}/></div></div>; }
function PanelTitle({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) { return <div className="panel-title"><div><small>{eyebrow}</small><h3>{title}</h3></div><p>{detail}</p></div>; }
function PipelineStep({ number, title, label, state }: { number: string; title: string; label: string; state: boolean }) { return <article className={state ? "reached" : ""}><span>{number}</span><div className="step-icon">{state ? "✓" : "·"}</div><b>{title}</b><small>{label}</small></article>; }
function PipelineArrow() { return <div className="pipeline-arrow"><i/><span>→</span></div>; }
function CodeLine({ label, value }: { label: string; value: string }) { return <div className="code-line"><span>{label}</span><code title={value}>{value}</code></div>; }
function RuleList({ accepted, rejected }: { accepted: string[]; rejected: string[] }) { return <div className="rules"><small>RULE SCREEN</small>{accepted.map((rule) => <span className="accepted" key={rule}>✓ {rule}</span>)}{rejected.map((rule) => <span className="rejected" key={rule}>× {rule}</span>)}</div>; }
function VersionNode({ role, id, active = false }: { role: string; id: string; active?: boolean }) { return <div className={`version-node ${active ? "active" : ""}`}><small>{role}</small><b>{id}</b></div>; }
function VersionConnector() { return <div className="version-connector">→</div>; }
function EmptyState({ title, body }: { title: string; body: string }) { return <div className="empty-state"><div>◇</div><b>{title}</b><p>{body}</p></div>; }
function BrandMark() { return <svg viewBox="0 0 36 36" aria-hidden="true"><path d="M18 3 31 10.5v15L18 33 5 25.5v-15Z"/><path d="m11 15 7-4 7 4v7l-7 4-7-4Z"/></svg>; }
function NavIcon({ view }: { view: View }) { const icons: Record<View, string> = { overview: "⌂", live: "⌁", calibration: "κ", memory: "◈", health: "⊕" }; return <span className="nav-icon">{icons[view]}</span>; }

function mergeEvents(current: RunEvent[], incoming: RunEvent[]): RunEvent[] { const merged = [...current]; for (const event of incoming) { const serialized = JSON.stringify(event); if (!merged.some((item) => JSON.stringify(item) === serialized)) merged.push(event); } return merged.slice(-200); }
function shortModel(model: string | null): string { return model?.split("/").at(-1) ?? "Not configured"; }
function shortTask(task: string): string { return task.length > 35 ? `${task.slice(0, 17)}…${task.slice(-12)}` : task; }
function pretty(value: string): string { return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function formatNumber(value: number): string { return new Intl.NumberFormat("en-US").format(value); }
function formatDuration(milliseconds: number): string { if (milliseconds < 1_000) return `${milliseconds}ms`; const seconds = Math.round(milliseconds / 1_000); return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`; }
function stageReached(current: string, target: string): boolean { const order = ["waiting", "preflight", "discovery", "hstar_proposal", "regression", "kappa_freeze", "smoke_test", "complete"]; return order.indexOf(current) >= order.indexOf(target); }
function eventIcon(type: string): string { return ({ run_started: "▶", inference_requested: "◎", action_proposed: "◇", action_reviewed: "κ", observation: "▣", sandbox_failure: "!", run_completed: "✓" } as Record<string, string>)[type] ?? "·"; }
function eventDescription(event: RunEvent): string { if (event.type === "run_started") return "Run opened with frozen seed and policy identity."; if (event.type === "inference_requested") return `Student inference requested for turn ${event.turn_index ?? 0}.`; if (event.type === "action_proposed") return `Student proposed ${pretty(event.action?.type ?? "action")}.`; if (event.type === "action_reviewed") return event.review?.decision === "replace" ? `Reviewer replaced ${pretty(event.review.proposed_action.type)} with ${pretty(event.review.accepted_action.type)}.` : `Reviewer passed ${pretty(event.review?.accepted_action.type ?? "action")} unchanged.`; if (event.type === "observation") return event.observation?.content ?? "Sandbox observation returned."; if (event.type === "sandbox_failure") return `OpenShell failed closed: ${pretty(event.failure_kind ?? "unknown")}.`; if (event.type === "run_completed") return `Run ${pretty(event.termination_reason ?? "completed")}.`; return pretty(event.type); }

function taskTraces(events: RunEvent[]): RunEvent[][] { const traces: RunEvent[][] = []; for (const event of events) { if (event.type === "run_started" || traces.length === 0) traces.push([event]); else traces.at(-1)?.push(event); } return traces; }
function actionSummary(action: RunEvent["action"]): string { if (!action) return "Waiting for Student"; if (action.type === "final_answer") return action.answer; if (action.type === "run_python") return action.code; if (action.type === "crop") return `(${action.x0}, ${action.y0}) → (${action.x1}, ${action.y1})`; return "Inspect image"; }
function reviewSummary(review: RunEvent["review"]): string { if (!review) return "Reviewer intentionally disabled"; if (review.decision === "pass") return `${pretty(review.accepted_action.type)} unchanged`; return `${pretty(review.proposed_action.type)} → ${pretty(review.accepted_action.type)}`; }
function resultLabel(completion: RunEvent | undefined): { label: string; tone: "running" | "correct" | "incorrect" | "failed" } { if (!completion) return { label: "Running", tone: "running" }; if (completion.termination_reason !== "completed") return { label: "Failed", tone: "failed" }; if (completion.correct === true) return { label: "Correct", tone: "correct" }; if (completion.correct === false) return { label: "Incorrect", tone: "incorrect" }; return { label: "Completed", tone: "running" }; }
