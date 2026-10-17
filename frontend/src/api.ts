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
  accepted_version: "accepted-v000";
  accepted_card_count: 0;
  measured_results_available: false;
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
  recent_events: RunEvent[];
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
