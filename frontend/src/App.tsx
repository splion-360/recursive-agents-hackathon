import { useEffect, useState } from "react";

import {
  type DashboardSnapshot,
  type RunEvent,
  fetchCalibrationTask,
  fetchDashboard,
  fetchLifecycleTask,
  subscribeToLifecycleEvents,
  subscribeToRunEvents,
} from "./api";

type View = "overview" | "live" | "calibration" | "memory" | "health";
type TraceCategory = "all" | "discovery_baseline" | "discovery_candidate" | "regression_baseline" | "regression_candidate" | "smoke";

const IMPORTANT_STAGE_DESCRIPTIONS: Record<string, string> = {
  discovery: "The Student solves visible tasks while H-star learns one general review rule from the results.",
  hstar_proposal: "H-star freezes its final proposed rule before any held-out validation task is revealed.",
  regression: "A sealed test set is run twice—without and with the proposed rule—to check whether the rule improves results.",
  kappa_freeze: "The supported part of the H-star rule is locked into the final κ Reviewer guidance.",
};

const TRACE_CATEGORY_DESCRIPTIONS: Record<TraceCategory, string> = {
  all: "Every execution from every calibration phase and condition.",
  discovery_baseline: "Discovery tasks run without the proposed H-star rule. This is the comparison point.",
  discovery_candidate: "The same Discovery tasks run with the current H-star rule so its effect can be measured.",
  regression_baseline: "Sealed test tasks run without the proposed rule. H-star never sees these tasks while evolving.",
  regression_candidate: "The same sealed test tasks run with the proposed rule. This decides whether the rule enters κ.",
  smoke: "Pilot tasks run with final frozen κ to verify the Reviewer can safely PASS or REPLACE actions.",
};

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
  const [lifecycleEvents, setLifecycleEvents] = useState<RunEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [lifecycleConnected, setLifecycleConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = () => {
      fetchDashboard()
        .then((value) => {
          if (!active) return;
          setSnapshot(value);
          setEvents(value.run_events);
          setLifecycleEvents(value.lifecycle.run_events);
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
    const unsubscribeLifecycle = subscribeToLifecycleEvents(
      (event) => setLifecycleEvents((current) => mergeEvents(current, [event])),
      setLifecycleConnected,
    );
    return () => {
      active = false;
      window.clearInterval(timer);
      unsubscribe();
      unsubscribeLifecycle();
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
              {item === "memory" && snapshot.lifecycle.state === "running" ? <i className="nav-pulse" /> : null}
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
          <div className="top-actions"><span className="mode-pill"><i /> LIVE</span><span className="run-id">{view === "memory" ? snapshot.lifecycle.run_id ?? "NO LIFECYCLE RUN" : snapshot.calibration.run_id ?? "NO ACTIVE RUN"}</span></div>
        </header>
        {error ? <div className="error-banner">Refresh warning: {error}</div> : null}
        {view === "overview" ? <Overview snapshot={snapshot} events={events} onNavigate={setView} /> : null}
        {view === "live" ? <LiveRun snapshot={snapshot} events={events} connected={connected} /> : null}
        {view === "calibration" ? <Calibration snapshot={snapshot} events={events} /> : null}
        {view === "memory" ? <SharedBrain snapshot={snapshot} events={lifecycleEvents} connected={lifecycleConnected} /> : null}
        {view === "health" ? <SystemHealth snapshot={snapshot} /> : null}
      </main>
    </div>
  );
}

function Overview({ snapshot, events, onNavigate }: { snapshot: DashboardSnapshot; events: RunEvent[]; onNavigate: (view: View) => void }) {
  const status = snapshot.calibration;
  const completed = status.discovery_completed + status.regression_completed;
  const total = snapshot.split_totals.discovery + snapshot.split_totals.regression;
  const progress = Math.round((completed / total) * 100);
  return <div className="page-stack">
    <section className="hero-card">
      <div className="hero-copy"><span className="kicker">VERSION N → N+1</span><h2>Watch experience become<br/><em>operational memory.</em></h2><p>One evidence chain from ChartQA trajectories through private review, frozen κ guidance, and deterministic promotion.</p><button onClick={() => onNavigate("live")}>{status.state === "running" ? "Watch current forward pass" : "Open live monitor"}<span>→</span></button></div>
      <div className="orb-wrap"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="brain-orb"><span>κ</span><small>{status.kappa_hash ? "FROZEN" : "CALIBRATING"}</small></div></div>
    </section>
    <section className="stat-row">
      <Stat label="Calibration progress" value={`${completed}/${total}`} hint={`${progress}% · ${stageLabel(status.stage)}`} tone="mint" />
      <Stat label="Public run events" value={String(snapshot.event_count)} hint={events.length ? pretty(events.at(-1)?.type ?? "waiting") : "No experiment events recorded"} tone="violet" />
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
  const [category, setCategory] = useState<TraceCategory>("all");
  const calibration = snapshot.calibration;
  const currentTask = calibration.current_task_id;
  const traces = taskTraces(events);
  const phases = new Map(snapshot.calibration_tasks.map((task) => [task.task_id, task.phase]));
  const categorized = traces.map((trace) => ({ trace, category: traceCategory(trace, phases) }));
  const visibleTraces = categorized
    .filter((item) => category === "all" || item.category === category)
    .map((item) => item.trace);
  const completed = traces.filter((trace) => trace.some((event) => event.type === "run_completed"));
  const scored = completed.filter((trace) => trace.at(-1)?.correct !== undefined && trace.at(-1)?.correct !== null);
  const correct = scored.filter((trace) => trace.at(-1)?.correct === true).length;
  const incorrect = scored.length - correct;
  const active = traces.length - completed.length;
  const progress = liveProgress(calibration);
  return <div className="page-stack">
    <section className="live-banner panel"><div><span className={`live-signal ${connected ? "on" : ""}`} /><div><b>{calibration.state === "running" ? "Forward pass in progress" : calibration.state === "failed" ? "Run stopped" : "Live monitor ready"}</b><small>{currentTask ?? calibrationStatusDetail(calibration)}</small></div></div><Metric label="Stage" value={stageLabel(calibration.stage)} /><Metric label={progress.label} value={progress.value} detail={progress.detail} /><Metric label="All run results" value={scored.length ? `${correct} correct · ${incorrect} incorrect` : "No completed scores"} detail={`${traces.length} recorded runs${active ? ` · ${active} running` : ""}`} /></section>
    <StageGuide currentStage={calibration.stage} />
    {calibration.state === "failed" ? <div className="run-state-note failed"><b>Calibration stopped</b><span>{calibration.message}</span></div> : null}
    <section className="panel trace-panel"><PanelTitle eyebrow="PUBLIC TASK RESULTS" title="Complete trajectory history" detail={`${traces.length} recorded runs · Student output, Reviewer decision, and official score`} />
      {events.length === 0 ? <EmptyState title="No forward pass yet" body="Calibration task results will appear here when an experiment run begins." /> : <><TraceFilters selected={category} categorized={categorized} onSelect={setCategory} /><p className="filter-description"><b>{traceCategoryLabel(category)}</b>{TRACE_CATEGORY_DESCRIPTIONS[category]}</p><TaskTraceTable traces={visibleTraces} /></>}
    </section>
  </div>;
}

function StageGuide({ currentStage }: { currentStage: string }) {
  const description = IMPORTANT_STAGE_DESCRIPTIONS[currentStage];
  if (!description) return null;
  return <div className="stage-guide"><b>{stageLabel(currentStage)}</b><span>{description}</span></div>;
}

function TraceFilters({ selected, categorized, onSelect }: { selected: TraceCategory; categorized: Array<{ trace: RunEvent[]; category: TraceCategory }>; onSelect: (category: TraceCategory) => void }) {
  const filters: Array<{ id: TraceCategory; label: string }> = [
    { id: "all", label: "All runs" },
    { id: "discovery_baseline", label: "Discovery · Baseline" },
    { id: "discovery_candidate", label: "Discovery · H-star candidate" },
    { id: "regression_baseline", label: "Validation · Without rule" },
    { id: "regression_candidate", label: "Validation · With rule" },
    { id: "smoke", label: "Smoke · Frozen κ" },
  ];
  return <div className="trace-filters" aria-label="Trajectory filters">{filters.map((filter) => {
    const count = filter.id === "all" ? categorized.length : categorized.filter((item) => item.category === filter.id).length;
    return <button className={selected === filter.id ? "active" : ""} key={filter.id} onClick={() => onSelect(filter.id)}>{filter.label}<span>{count}</span></button>;
  })}</div>;
}

function Calibration({ snapshot, events }: { snapshot: DashboardSnapshot; events: RunEvent[] }) {
  const c = snapshot.calibration;
  const legacyArtifact = c.state === "complete" && (
    c.discovery_completed < snapshot.split_totals.discovery
    || c.regression_completed < snapshot.split_totals.regression
  );
  const decisions = events.filter((event) => event.type === "action_reviewed");
  const replacements = decisions.filter((event) => event.review?.decision === "replace").length;
  return <div className="page-stack">
    <section className="calibration-head"><div><span className="kicker">H-STAR → κ</span><h2>Calibration laboratory</h2><p>Three four-task Discovery rounds evolve H-star. Held-out validation tests only the frozen final revision.</p></div><div className={`state-stamp ${c.state}`}>{pretty(c.state)}<small>{stageLabel(c.stage)}</small></div></section>
    {legacyArtifact ? <div className="run-state-note"><b>Historical artifact</b><span>This completed run used the previous 8+8 contract. Its results remain visible, but a new calibration is required for the current 12+12 three-round design.</span></div> : null}
    <section className="split-progress">
      <PhaseCard index="A" title="Discovery" count={c.discovery_completed} total={snapshot.split_totals.discovery} detail="3 rounds × 4 tasks · visible to H-star" locked={false} />
      <div className="isolation-line"><span>ISOLATION BOUNDARY</span></div>
      <PhaseCard index="B" title="Held-out validation" count={c.regression_completed} total={snapshot.split_totals.regression} detail="Hidden until final H-star freezes" locked={c.stage === "waiting" || c.stage === "discovery"} />
    </section>
    <section className="two-column calibration-grid">
      <div className="panel proposal-card"><PanelTitle eyebrow="THREE VERSIONED REVISIONS" title="H-star proposal" detail={shortModel(snapshot.models.hstar)} />
        {c.proposal_hash ? <><CodeLine label="Final proposal" value={c.proposal_id ?? "revision-3"} /><CodeLine label="SHA-256" value={c.proposal_hash} /><RuleList accepted={c.accepted_rule_ids} rejected={c.rejected_rule_ids} /></> : <EmptyState title="Proposal not generated" body="A revision appears after each four-task Discovery round." />}
      </div>
      <div className="panel proposal-card"><PanelTitle eyebrow="FROZEN REVIEW POLICY" title="κ guidance" detail={shortModel(snapshot.models.reviewer)} />
        {c.kappa_hash ? <><CodeLine label="Artifact" value={c.kappa_artifact_id ?? "kappa.v1"} /><CodeLine label="SHA-256" value={c.kappa_hash} /><p className="success-note">Held-out validation passed. Exact artifact is frozen for measured Collection.</p></> : <EmptyState title="No κ artifact yet" body="Only rules with discovery benefit, positive held-out evidence, action compatibility, and privacy clearance can enter κ." />}
      </div>
    </section>
    <section className="harness-grid">{snapshot.harnesses.map((harness) => <details className="panel harness-card" key={harness.role} open={harness.role === "student" || harness.role === "kappa"}><summary><span>{pretty(harness.role)}</span><small>{shortModel(harness.model_id)}</small></summary><p>{dashboardCopy(harness.purpose)}</p><HarnessLine label="Sees" values={harness.sees.map(dashboardCopy)} /><HarnessLine label="Can do" values={harness.actions.map(dashboardCopy)} />{harness.revisions.length ? <HarnessLine label="Evolution" values={harness.revisions.map(dashboardCopy)} /> : null}{harness.guidance.length ? <HarnessLine label="Exact guidance" values={harness.guidance.map(dashboardCopy)} /> : null}</details>)}</section>
    <section className="stat-row compact"><Stat label="Reviewer decisions" value={String(decisions.length)} hint="Public decisions only" tone="violet" /><Stat label="REPLACE" value={String(replacements)} hint="Smallest valid replacement" tone="amber" /><Stat label="Tokens" value={formatNumber(c.input_tokens + c.output_tokens)} hint={`${formatNumber(c.input_tokens)} in · ${formatNumber(c.output_tokens)} out`} tone="mint" /><Stat label="Estimated cost" value={`$${c.estimated_cost_usd.toFixed(4)}`} hint="Runner-reported" tone="slate" /></section>
  </div>;
}

function SharedBrain({ snapshot, events, connected }: { snapshot: DashboardSnapshot; events: RunEvent[]; connected: boolean }) {
  const lifecycle = snapshot.lifecycle;
  const traces = taskTraces(events);
  const collectionTraces = traces.slice(0, 40);
  const pairs = lifecycle.promotion_pairs;
  const baselineCorrect = pairs.filter((pair) => pair.accepted_correct).length;
  const candidateCorrect = pairs.filter((pair) => pair.candidate_correct).length;
  const stageText = lifecycle.state === "idle" ? "Waiting for the measured lifecycle command." : lifecycle.state === "complete" ? `Measured lifecycle ${lifecycle.decision?.toLowerCase() ?? "completed"}.` : `${pretty(lifecycle.stage)} is in progress.`;
  return <div className="page-stack">
    <section className="brain-hero panel"><div className="large-orb">N+1</div><div><span className="kicker">MEMMACHINE OPERATIONAL MEMORY</span><h2>{lifecycle.state === "complete" ? lifecycle.accepted_version_id : "Building Shared Brain N+1"}</h2><p>Converting privately reviewed Collection episodes into answer-free Behavior Cards, then testing the frozen candidate against the current memory.</p><div className={`truth-badge ${lifecycle.decision?.toLowerCase() ?? ""}`}>{stageText}</div></div><span className={`lifecycle-signal ${connected ? "online" : ""}`}>{connected ? "EVENT STREAM LIVE" : "POLLING ARTIFACTS"}</span></section>
    <section className="panel lifecycle-progress"><PanelTitle eyebrow="MEASURED LIFECYCLE" title={pretty(lifecycle.stage)} detail="Collection → Consolidation → Candidate N+1 → Promotion" /><div className="lifecycle-stages"><LifecycleStage label="Collection" value={`${lifecycle.collection_completed}/40`} active={lifecycle.stage === "collection"} done={lifecycle.collection_completed === 40}/><LifecycleStage label="Consolidation" value={lifecycle.candidate_version_id ? "Frozen" : "Pending"} active={lifecycle.stage === "consolidation"} done={lifecycle.candidate_version_id !== null}/><LifecycleStage label="Promotion" value={`${Math.floor(lifecycle.promotion_runs_completed / 2)}/30 pairs`} active={lifecycle.stage === "promotion"} done={lifecycle.state === "complete"}/><LifecycleStage label="Decision" value={lifecycle.decision ?? "Pending"} active={false} done={lifecycle.state === "complete"}/></div></section>
    <section className="stat-row compact"><Stat label="Reviewed episodes" value={String(lifecycle.reviewed_episode_count)} hint="Written to MemMachine" tone="mint"/><Stat label="Eligible evidence" value={String(lifecycle.eligible_episode_count)} hint="Used for consolidation" tone="violet"/><Stat label="Behavior Cards" value={String(lifecycle.cards.length)} hint="Frozen in Candidate N+1" tone="amber"/><Stat label="Promotion pairs" value={`${pairs.length}/30`} hint="Same task and seed" tone="slate"/></section>
    <section className="panel"><PanelTitle eyebrow="VERSION EVOLUTION" title="Operational memory" detail="Immutable versions and deterministic decision"/><div className="version-rail"><VersionNode role="PARENT" id={lifecycle.parent_version_id} active/><VersionConnector/><VersionNode role="CANDIDATE N+1" id={lifecycle.candidate_version_id ?? "Pending consolidation"} active={lifecycle.candidate_version_id !== null}/><VersionConnector/><VersionNode role="DECISION" id={lifecycle.decision ?? "Pending promotion"} active={lifecycle.state === "complete"}/></div>{lifecycle.cards.length ? <div className="behavior-cards">{lifecycle.cards.map((card) => <details className="behavior-card" key={card.card_id}><summary><span>{card.card_id}</span><b>{card.trigger}</b></summary><p>{card.instruction}</p><small>{card.evidence_episode_ids.length} supporting episodes</small></details>)}</div> : <EmptyState title="No candidate cards yet" body="Cards appear after the consolidator processes eligible Collection episodes."/>}</section>
    <section className="panel"><PanelTitle eyebrow="COLLECTION" title="Reviewed task evidence" detail="κ-guided trajectories eligible to become operational memory"/>{collectionTraces.length ? <TaskTraceTable traces={collectionTraces} lifecycle/> : <EmptyState title="Collection has not started" body="Run the measured lifecycle command; task results will appear here automatically."/>}</section>
    <section className="panel"><PanelTitle eyebrow="PAIRED PROMOTION" title={lifecycle.decision ? `${lifecycle.decision}: ${lifecycle.accepted_version_id}` : "Awaiting candidate evaluation"} detail="Current memory and Candidate N+1 run on identical tasks and seeds"/>{pairs.length ? <><div className="promotion-summary"><Metric label="Current memory accuracy" value={`${baselineCorrect}/${pairs.length}`}/><Metric label="Candidate N+1 accuracy" value={`${candidateCorrect}/${pairs.length}`}/><Metric label="Capability delta" value={formatPercent(lifecycle.capability_delta)}/><Metric label="Negative transfer" value={formatPercent(lifecycle.negative_transfer_rate)}/></div><div className="promotion-table"><div className="promotion-row head"><span>Task</span><span>Current memory</span><span>Candidate N+1</span><span>Outcome</span></div>{pairs.map((pair) => <div className="promotion-row" key={pair.task_id}><code title={pair.task_id}>{shortTask(pair.task_id)}</code><span>{pair.accepted_correct ? "Correct" : "Incorrect"}</span><span>{pair.candidate_correct ? "Correct" : "Incorrect"}</span><b>{pair.accepted_correct === pair.candidate_correct ? "No change" : pair.candidate_correct ? "Improved" : "Negative transfer"}</b></div>)}</div>{lifecycle.failed_gates.length ? <div className="warning-box"><b>Failed promotion gates</b><p>{lifecycle.failed_gates.map(pretty).join(" · ")}</p></div> : null}</> : <EmptyState title="Promotion has not finished" body="The comparison table appears after all 30 frozen task pairs complete."/>}</section>
  </div>;
}

function SystemHealth({ snapshot }: { snapshot: DashboardSnapshot }) {
  return <div className="page-stack"><section className="panel health-panel"><PanelTitle eyebrow="PREFLIGHT" title="Runtime configuration" detail="Configured does not mean live-verified in this process" /><div className="health-list">{snapshot.integrations.map((item) => <article key={item.name}><span className={`health-icon ${item.state}`}>{item.state === "configured" ? "✓" : item.state === "limited" ? "!" : "×"}</span><div><b>{item.name}</b><p>{item.detail}</p></div><em className={item.state}>{item.state}</em></article>)}</div></section><section className="two-column"><div className="panel model-panel"><PanelTitle eyebrow="LOCKED ROSTER" title="Model assignments" detail="Loaded from settings" />{Object.entries(snapshot.models).map(([role, model]) => <CodeLine key={role} label={pretty(role)} value={model ?? "Not configured"} />)}</div><div className="panel model-panel"><PanelTitle eyebrow="DATA ISOLATION" title="Frozen manifests" detail="Actual local manifest hashes" />{snapshot.manifests.map((manifest) => <CodeLine key={manifest.name} label={`${pretty(manifest.name)} · ${manifest.task_count}`} value={manifest.content_sha256} />)}</div></section></div>;
}

function ProgressCard({ snapshot }: { snapshot: DashboardSnapshot }) { const c = snapshot.calibration; return <div className="panel progress-card"><PanelTitle eyebrow="CALIBRATION" title="Calibration progress" detail={dashboardCopy(c.message)} /><Progress label="Discovery trajectories" value={c.discovery_completed} total={snapshot.split_totals.discovery} /><Progress label="Held-out validation pairs" value={c.regression_completed} total={snapshot.split_totals.regression} /><Progress label="PASS / REPLACE smoke" value={c.smoke_completed} total={c.smoke_total || 5} /><div className="current-step"><span>NOW</span><b>{stageLabel(c.stage)}</b><small>{c.current_task_id ?? calibrationStatusDetail(c)}</small></div></div>; }
function RecentEvents({ events, onNavigate }: { events: RunEvent[]; onNavigate: () => void }) { return <div className="panel recent-card"><PanelTitle eyebrow="TRACE TAIL" title="Recent events" detail="Sanitized RunEvent projection" />{events.length ? events.map((event, index) => <MiniEvent event={event} key={`${event.task_id}-${event.sequence}-${index}`} />) : <EmptyState title="Stream is quiet" body="No calibration events have been written." />}<button className="text-button" onClick={onNavigate}>Open full trajectory →</button></div>; }
function TaskTraceTable({ traces, lifecycle = false }: { traces: RunEvent[][]; lifecycle?: boolean }) {
  const [page, setPage] = useState(1);
  const occurrenceByTask = new Map<string, number>();
  const keyed = [...traces].reverse().map((trace) => {
    const start = trace.find((event) => event.type === "run_started");
    const base = `${start?.task_id ?? trace[0]?.task_id ?? "partial"}-${start?.seed ?? "unknown"}`;
    const occurrence = occurrenceByTask.get(base) ?? 0;
    occurrenceByTask.set(base, occurrence + 1);
    return { trace, key: `${base}-${occurrence}` };
  }).reverse();
  const pageSize = 5;
  const pageCount = Math.max(1, Math.ceil(keyed.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const startIndex = (currentPage - 1) * pageSize;
  const pageItems = keyed.slice(startIndex, startIndex + pageSize);
  useEffect(() => { if (page !== currentPage) setPage(currentPage); }, [currentPage, page]);
  return <><div className="task-trace"><div className="task-trace-head"><span>Task / policy</span><span>Student proposal</span><span>Reviewer history</span><span>Official result</span></div>{pageItems.map(({ trace, key }, index) => <TaskTraceRow events={trace} index={startIndex + index + 1} key={key} lifecycle={lifecycle} />)}</div><div className="task-pagination"><span>Showing {keyed.length ? startIndex + 1 : 0}–{Math.min(startIndex + pageSize, keyed.length)} of {keyed.length}</span><div><button type="button" disabled={currentPage === 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>← Previous</button><b>Page {currentPage} of {pageCount}</b><button type="button" disabled={currentPage === pageCount} onClick={() => setPage((value) => Math.min(pageCount, value + 1))}>Next →</button></div></div></>;
}
function TaskTraceRow({ events, index, lifecycle = false }: { events: RunEvent[]; index: number; lifecycle?: boolean }) {
  const start = events.find((event) => event.type === "run_started");
  const proposal = events.filter((event) => event.type === "action_proposed").at(-1);
  const reviews = events
    .filter((event) => event.type === "action_reviewed")
    .flatMap((event) => event.review ? [event.review] : []);
  const review = reviews.at(-1);
  const replacementCount = reviews.filter((item) => item.decision === "replace").length;
  const replacements = reviews.filter((item) => item.decision === "replace");
  const completion = events.filter((event) => event.type === "run_completed").at(-1);
  const policy = start?.kappa_hash ? `κ ${start.kappa_hash.slice(0, 8)}` : review ? "H* candidate" : "Baseline";
  const result = resultLabel(completion);
  const taskId = start?.task_id ?? events[0]?.task_id ?? "Unknown task";
  const [expanded, setExpanded] = useState(false);
  const [question, setQuestion] = useState<string | null>(null);
  useEffect(() => {
    if (!expanded || question !== null || taskId === "Unknown task") return;
    let active = true;
    (lifecycle ? fetchLifecycleTask(taskId) : fetchCalibrationTask(taskId))
      .then((task) => { if (active) setQuestion(task.question); })
      .catch(() => { if (active) setQuestion("Question unavailable for this task."); });
    return () => { active = false; };
  }, [expanded, lifecycle, question, taskId]);
  return <article className={`task-trace-row ${result.tone}`}>
    <div className="task-cell identity"><span className="task-number">{String(index).padStart(2, "0")}</span><div><b title={taskId}>{shortTask(taskId)}</b><small>{policy}{start?.seed !== undefined ? ` · seed ${start.seed}` : ""}</small></div></div>
    <div className="task-cell"><small>{proposal ? pretty(proposal.action?.type ?? "action") : "Waiting"}</small><strong title={actionSummary(proposal?.action)}>{actionSummary(proposal?.action)}</strong></div>
    <div className="task-cell reviewer-cell"><small>{review ? `LATEST ${review.decision.toUpperCase()}` : start?.kappa_hash ? completion ? "NOT REACHED" : "PENDING" : "BASELINE"}</small><strong title={reviewSummary(review, Boolean(start?.kappa_hash), Boolean(completion))}>{reviewSummary(review, Boolean(start?.kappa_hash), Boolean(completion))}</strong>{review ? <span className="review-history">{reviews.length} {reviews.length === 1 ? "review" : "reviews"} · {replacementCount} {replacementCount === 1 ? "replacement" : "replacements"}</span> : null}</div>
    <div className="task-cell result"><span className={`result-dot ${result.tone}`} /> <div><strong>{result.label}</strong><small>{completion ? pretty(completion.failure_kind ?? completion.termination_reason ?? "completed") : `${events.length} events received`}</small></div></div>
    <details className="task-chart" onToggle={(event) => setExpanded(event.currentTarget.open)}><summary><span>View task</span><code>{taskId}</code></summary><div><div className="task-question"><small>QUESTION</small><p>{question ?? "Loading question…"}</p></div><img src={`${lifecycle ? "/api/dashboard/lifecycle/tasks" : "/api/dashboard/tasks"}/${encodeURIComponent(taskId)}/image`} alt={`Chart for ${taskId}`} loading="lazy" />{replacements.length ? <section className="task-replacements"><div className="replacement-title"><small>REVIEWER REPLACEMENTS</small><span>{replacements.length} {replacements.length === 1 ? "change" : "changes"}</span></div>{replacements.map((item, replacementIndex) => <article key={`${item.proposed_action.type}-${item.accepted_action.type}-${replacementIndex}`}><header><span>#{replacementIndex + 1}</span><b>{pretty(item.proposed_action.type)} → {pretty(item.accepted_action.type)}</b></header><div><label>Student proposed<pre>{actionDetail(item.proposed_action)}</pre></label><i>→</i><label>Reviewer accepted<pre>{actionDetail(item.accepted_action)}</pre></label></div></article>)}</section> : <div className="task-replacements empty"><small>REVIEWER REPLACEMENTS</small><p>No actions were replaced in this trajectory.</p></div>}</div></details>
  </article>;
}
function MiniEvent({ event }: { event: RunEvent }) { return <div className="mini-event"><span>{eventIcon(event.type)}</span><div><b>{pretty(event.type)}</b><small>{eventDescription(event)}</small></div><code>{event.sequence}</code></div>; }
function PhaseCard({ index, title, count, total, detail, locked }: { index: string; title: string; count: number; total: number; detail: string; locked: boolean }) { const percentage = Math.round((count / total) * 100); return <article className={`phase-card ${locked ? "locked" : ""}`}><span>{index}</span><div><small>{locked ? "SEALED" : "CALIBRATION SPLIT"}</small><h3>{title}</h3><p>{detail}</p></div><strong>{count}<i>/{total}</i></strong><div className="phase-bar"><i style={{ width: `${percentage}%` }} /></div></article>; }
function Stat({ label, value, hint, tone }: { label: string; value: string; hint: string; tone: string }) { return <article className={`stat-card ${tone}`}><small>{label}</small><strong>{value}</strong><p>{hint}</p></article>; }
function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) { return <div className="banner-metric"><small>{label}</small><b>{value}</b>{detail ? <span>{detail}</span> : null}</div>; }
function Progress({ label, value, total }: { label: string; value: number; total: number }) { const percentage = Math.min(100, Math.round((value / total) * 100)); return <div className="progress-item"><div><span>{label}</span><b>{value}/{total}</b></div><div><i style={{ width: `${percentage}%` }}/></div></div>; }
function PanelTitle({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) { return <div className="panel-title"><div><small>{eyebrow}</small><h3>{title}</h3></div><p>{detail}</p></div>; }
function PipelineStep({ number, title, label, state }: { number: string; title: string; label: string; state: boolean }) { return <article className={state ? "reached" : ""}><span>{number}</span><div className="step-icon">{state ? "✓" : "·"}</div><b>{title}</b><small>{label}</small></article>; }
function PipelineArrow() { return <div className="pipeline-arrow"><i/><span>→</span></div>; }
function CodeLine({ label, value }: { label: string; value: string }) { return <div className="code-line"><span>{label}</span><code title={value}>{value}</code></div>; }
function RuleList({ accepted, rejected }: { accepted: string[]; rejected: string[] }) { return <div className="rules"><small>RULE SCREEN</small>{accepted.map((rule) => <span className="accepted" key={rule}>✓ {rule}</span>)}{rejected.map((rule) => <span className="rejected" key={rule}>× {rule}</span>)}</div>; }
function HarnessLine({ label, values }: { label: string; values: string[] }) { return <div className="harness-line"><small>{label}</small>{values.map((value) => <code key={value}>{value}</code>)}</div>; }
function VersionNode({ role, id, active = false }: { role: string; id: string; active?: boolean }) { return <div className={`version-node ${active ? "active" : ""}`}><small>{role}</small><b>{id}</b></div>; }
function VersionConnector() { return <div className="version-connector">→</div>; }
function LifecycleStage({ label, value, active, done }: { label: string; value: string; active: boolean; done: boolean }) { return <article className={`${active ? "active" : ""} ${done ? "done" : ""}`}><span>{done ? "✓" : active ? "●" : "○"}</span><div><small>{label}</small><b>{value}</b></div></article>; }
function EmptyState({ title, body }: { title: string; body: string }) { return <div className="empty-state"><div>◇</div><b>{title}</b><p>{body}</p></div>; }
function BrandMark() { return <svg viewBox="0 0 36 36" aria-hidden="true"><path d="M18 3 31 10.5v15L18 33 5 25.5v-15Z"/><path d="m11 15 7-4 7 4v7l-7 4-7-4Z"/></svg>; }
function NavIcon({ view }: { view: View }) { const icons: Record<View, string> = { overview: "⌂", live: "⌁", calibration: "κ", memory: "◈", health: "⊕" }; return <span className="nav-icon">{icons[view]}</span>; }

function mergeEvents(current: RunEvent[], incoming: RunEvent[]): RunEvent[] { const merged = [...current]; for (const event of incoming) { const serialized = JSON.stringify(event); if (!merged.some((item) => JSON.stringify(item) === serialized)) merged.push(event); } return merged.slice(-200); }
function shortModel(model: string | null): string { return model?.split("/").at(-1) ?? "Not configured"; }
function shortTask(task: string): string { return task.length > 35 ? `${task.slice(0, 17)}…${task.slice(-12)}` : task; }
function pretty(value: string): string { return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function stageLabel(stage: string): string { return stage === "regression" ? "Held-out validation" : pretty(stage); }
function dashboardCopy(value: string): string { return value.replace(/regression/gi, "held-out validation"); }
function calibrationStatusDetail(status: DashboardSnapshot["calibration"]): string {
  if (status.state === "complete") return "Frozen κ is ready for measured Collection.";
  if (status.state === "failed") return "Calibration ended before a valid κ artifact was frozen.";
  if (status.state === "running") return "Preparing the next calibration task.";
  return "No calibration run is active.";
}
function traceCategoryLabel(category: TraceCategory): string {
  if (category === "regression_baseline") return "Validation · Without rule";
  if (category === "regression_candidate") return "Validation · With rule";
  return pretty(category);
}
function liveProgress(status: DashboardSnapshot["calibration"]): { label: string; value: string; detail: string } {
  if (status.stage === "discovery") return { label: "Discovery progress", value: `${status.discovery_completed}/12`, detail: "three rounds of four" };
  if (status.stage === "regression") return { label: "Validation progress", value: `${status.regression_completed}/12 pairs`, detail: "sealed test set" };
  if (status.stage === "smoke_test") return { label: "Smoke progress", value: `${status.smoke_completed}/${status.smoke_total}`, detail: "PASS / REPLACE checks" };
  if (["kappa_freeze", "complete"].includes(status.stage)) return { label: "Validation progress", value: `${status.regression_completed}/12 pairs`, detail: "sealed test set" };
  return { label: "Calibration progress", value: `${status.discovery_completed + status.regression_completed}/24`, detail: "discovery + validation" };
}
function formatNumber(value: number): string { return new Intl.NumberFormat("en-US").format(value); }
function formatPercent(value: number | null): string { return value === null ? "Pending" : `${(value * 100).toFixed(1)}%`; }
function formatDuration(milliseconds: number): string { if (milliseconds < 1_000) return `${milliseconds}ms`; const seconds = Math.round(milliseconds / 1_000); return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`; }
function stageReached(current: string, target: string): boolean { const order = ["waiting", "preflight", "discovery", "hstar_proposal", "regression", "kappa_freeze", "smoke_test", "complete"]; return order.indexOf(current) >= order.indexOf(target); }
function eventIcon(type: string): string { return ({ run_started: "▶", inference_requested: "◎", action_proposed: "◇", action_reviewed: "κ", observation: "▣", sandbox_failure: "!", run_completed: "✓" } as Record<string, string>)[type] ?? "·"; }
function eventDescription(event: RunEvent): string { if (event.type === "run_started") return "Run opened with frozen seed and policy identity."; if (event.type === "inference_requested") return `Student inference requested for turn ${event.turn_index ?? 0}.`; if (event.type === "action_proposed") return `Student proposed ${pretty(event.action?.type ?? "action")}.`; if (event.type === "action_reviewed") return event.review?.decision === "replace" ? `Reviewer replaced ${pretty(event.review.proposed_action.type)} with ${pretty(event.review.accepted_action.type)}.` : `Reviewer passed ${pretty(event.review?.accepted_action.type ?? "action")} unchanged.`; if (event.type === "observation") return event.observation?.content ?? "Sandbox observation returned."; if (event.type === "sandbox_failure") return `OpenShell failed closed: ${pretty(event.failure_kind ?? "unknown")}.`; if (event.type === "run_completed") return `Run ${pretty(event.termination_reason ?? "completed")}.`; return pretty(event.type); }

function taskTraces(events: RunEvent[]): RunEvent[][] { const traces: RunEvent[][] = []; for (const event of events) { if (event.type === "run_started" || traces.length === 0) traces.push([event]); else traces.at(-1)?.push(event); } return traces; }
function traceCategory(trace: RunEvent[], phases: Map<string, "discovery" | "regression">): TraceCategory {
  const start = trace.find((event) => event.type === "run_started");
  if (start?.kappa_hash) return "smoke";
  const phase = phases.get(start?.task_id ?? trace[0]?.task_id ?? "");
  const candidate = trace.some((event) => event.type === "action_reviewed");
  if (phase === "discovery") return candidate ? "discovery_candidate" : "discovery_baseline";
  if (phase === "regression") return candidate ? "regression_candidate" : "regression_baseline";
  return "all";
}
function actionSummary(action: RunEvent["action"]): string { if (!action) return "Waiting for Student"; if (action.type === "final_answer") return action.answer; if (action.type === "run_python") return action.code; if (action.type === "crop") return `(${action.x0}, ${action.y0}) → (${action.x1}, ${action.y1})`; return "Inspect image"; }
function actionDetail(action: NonNullable<RunEvent["action"]>): string { return `${pretty(action.type)}: ${actionSummary(action)}`; }
function reviewSummary(review: RunEvent["review"], reviewExpected = false, completed = false): string { if (!review) return reviewExpected ? completed ? "Reviewer not reached — Student output failed" : "Waiting for Reviewer" : "Reviewer intentionally disabled"; if (review.decision === "pass") return `${pretty(review.accepted_action.type)} unchanged`; return `${pretty(review.proposed_action.type)} → ${pretty(review.accepted_action.type)}`; }
function resultLabel(completion: RunEvent | undefined): { label: string; tone: "running" | "correct" | "incorrect" | "failed" } { if (!completion) return { label: "Running", tone: "running" }; if (completion.termination_reason !== "completed") return { label: "Failed", tone: "failed" }; if (completion.correct === true) return { label: "Correct", tone: "correct" }; if (completion.correct === false) return { label: "Incorrect", tone: "incorrect" }; return { label: "Completed", tone: "running" }; }
