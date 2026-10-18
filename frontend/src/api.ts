export type StudentAction =
  | { type: "inspect_image" }
  | { type: "crop"; x0: number; y0: number; x1: number; y1: number }
  | { type: "run_python"; code: string }
  | { type: "final_answer"; answer: string };

export type RunEvent = {
  type: string;
  sequence: number;
  task_id: string;
  seed?: number;
  kappa_hash?: string | null;
  turn_index?: number;
  action?: StudentAction;
  review?: {
    decision: "pass" | "replace";
    proposed_action: StudentAction;
    accepted_action: StudentAction;
    model_id: string;
    input_tokens: number;
    output_tokens: number;
    latency_ms: number;
  };
  observation?: { action_type: string; content: string };
  termination_reason?: string;
  correct?: boolean | null;
  failure_kind?: string;
};

export type CalibrationStatus = {
  run_id: string | null;
  state: "idle" | "running" | "complete" | "failed";
  stage: string;
  discovery_completed: number;
  regression_completed: number;
  smoke_completed: number;
  smoke_total: number;
  current_task_id: string | null;
  proposal_id: string | null;
  proposal_hash: string | null;
  kappa_artifact_id: string | null;
  kappa_hash: string | null;
  accepted_rule_ids: string[];
  rejected_rule_ids: string[];
  input_tokens: number;
  output_tokens: number;
  elapsed_ms: number;
  estimated_cost_usd: number;
  message: string;
  updated_at: string | null;
};

export type DashboardSnapshot = {
  mode: "LIVE";
  generated_at: string;
  accepted_version: string;
  accepted_card_count: number;
  measured_results_available: boolean;
  calibration: CalibrationStatus;
  manifests: Array<{ name: string; task_count: number; content_sha256: string }>;
  integrations: Array<{
    name: string;
    state: "configured" | "missing" | "limited";
    detail: string;
  }>;
  models: {
    student: string | null;
    reviewer: string | null;
    consolidator: string | null;
    hstar: string | null;
  };
  event_count: number;
  split_totals: { discovery: number; regression: number };
  harnesses: Array<{
    role: "student" | "reviewer" | "hstar" | "kappa";
    model_id: string | null;
    purpose: string;
    sees: string[];
    actions: string[];
    guidance: string[];
    revisions: string[];
  }>;
  calibration_tasks: Array<{
    task_id: string;
    phase: "discovery" | "regression";
  }>;
  run_events: RunEvent[];
  lifecycle: {
    state: "idle" | "running" | "complete";
    stage: "waiting" | "collection" | "consolidation" | "promotion" | "complete";
    run_id: string | null;
    collection_completed: number;
    collection_total: 40;
    promotion_runs_completed: number;
    promotion_runs_total: 60;
    reviewed_episode_count: number;
    eligible_episode_count: number;
    parent_version_id: string;
    candidate_version_id: string | null;
    accepted_version_id: string;
    cards: Array<{
      card_id: string;
      trigger: string;
      instruction: string;
      evidence_episode_ids: string[];
    }>;
    decision: "PROMOTE" | "REJECT" | null;
    failed_gates: string[];
    capability_delta: number | null;
    negative_transfer_rate: number | null;
    latency_regression: number | null;
    token_regression: number | null;
    promotion_pairs: Array<{
      task_id: string;
      accepted_correct: boolean;
      candidate_correct: boolean;
      accepted_latency_ms: number;
      candidate_latency_ms: number;
      accepted_input_tokens: number;
      candidate_input_tokens: number;
    }>;
    run_events: RunEvent[];
  };
};

export type CalibrationTaskDetail = {
  task_id: string;
  question: string;
};

export async function fetchDashboard(): Promise<DashboardSnapshot> {
  const response = await fetch("/api/dashboard");
  if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
  return response.json() as Promise<DashboardSnapshot>;
}

export async function fetchCalibrationTask(taskId: string): Promise<CalibrationTaskDetail> {
  const response = await fetch(`/api/dashboard/tasks/${encodeURIComponent(taskId)}`);
  if (!response.ok) throw new Error(`Task API returned ${response.status}`);
  return response.json() as Promise<CalibrationTaskDetail>;
}

export async function fetchLifecycleTask(taskId: string): Promise<CalibrationTaskDetail> {
  const response = await fetch(`/api/dashboard/lifecycle/tasks/${encodeURIComponent(taskId)}`);
  if (!response.ok) throw new Error(`Lifecycle task API returned ${response.status}`);
  return response.json() as Promise<CalibrationTaskDetail>;
}

export function subscribeToRunEvents(
  onEvent: (event: RunEvent) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  const source = new EventSource("/api/dashboard/events");
  source.onopen = () => onConnection(true);
  source.onerror = () => onConnection(false);
  source.addEventListener("run_event", (message) => {
    onEvent(JSON.parse((message as MessageEvent<string>).data) as RunEvent);
  });
  return () => source.close();
}

export function subscribeToLifecycleEvents(
  onEvent: (event: RunEvent) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  const source = new EventSource("/api/dashboard/lifecycle/events");
  source.onopen = () => onConnection(true);
  source.onerror = () => onConnection(false);
  source.addEventListener("run_event", (message) => {
    onEvent(JSON.parse((message as MessageEvent<string>).data) as RunEvent);
  });
  return () => source.close();
}
