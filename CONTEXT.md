# Memory-Distilled Agent Team

This context defines the language used for an agent team that turns reviewed failures into measured, reusable improvements stored in long-term memory.

## Language

**Version N+1**:
The project’s primary theme: a new version of the agent team’s operating procedure whose candidate behavior changes are retained only after measurable improvement on unseen tasks.
_Avoid_: Self-improvement, recursive learning

**Shared Brain**:
The MemMachine-backed memory substrate through which agent roles preserve reviewed experience and reusable behaviors across runs. It enables Version N+1 but is not the project’s primary theme.
_Avoid_: Vector store, shared database

**Primary Benchmark**:
ChartQA is the required task suite used to build and evaluate the complete Version N+1 loop.
_Avoid_: Core benchmark suite, required datasets

**Stretch Benchmark**:
MathVista is an optional generalization test attempted only after the primary experiment and demo are complete.
_Avoid_: Secondary required benchmark

**Execution Boundary**:
The NemoClaw-managed OpenShell sandbox in which the student’s generated Python actions execute under explicit filesystem, network, and runtime policies.
_Avoid_: Local Python runner, unrestricted shell

**Experiment Visualizer**:
The live and replayable interface that exposes action correction, memory formation, behavior retrieval, and measured improvement. It demonstrates the agent system but is not itself the product.
_Avoid_: Dashboard, Streamlit app

## Harness Artifacts

**Minimal Harness (H)**:
The fixed Student prompt, typed ChartQA action space, tools, decoding settings, and budgets shared by every compared condition. Memory context and Reviewer access vary only where a condition explicitly says so.
_Avoid_: Student model, baseline prompt

**Optimized Reference Harness (H-star)**:
An empirically selected ChartQA reference harness proposed and evaluated only on the Harness Calibration Set. It may contain richer tools, checks, skills, or middleware than H, but it never executes measured Collection, Promotion, or Final Held-Out tasks.
_Avoid_: Production harness, teacher model

**Kappa Guidance**:
The frozen, answer-free private review artifact adapted from accepted H-star components into checks, action recipes, review guidance, and failure patterns executable through H. The Reviewer consumes Kappa Guidance; Kappa is not a model or an additional agent role.
_Avoid_: Kappa model, mutable reviewer prompt

**Harness Calibration Set**:
A fixed 16-item ChartQA manifest disjoint by task and image from Collection, Promotion, and Final Held-Out Sets. Eight discovery items may inform one H-star proposal; a separate eight-item regression subset accepts or rejects it without producing Behavior Cards.
_Avoid_: Collection Set, Promotion Set

**Lifecycle Pass**:
One batched Version N to N+1 attempt: collect the fixed Collection Set against one Accepted Version, freeze its eligible Reviewed Episode snapshot, consolidate and screen one immutable Candidate Version, then accept or reject it on the complete Promotion Set. No per-task memory mutation becomes retrievable mid-pass.
_Avoid_: Online learning, continuous update

## Agent Roles

**Student**:
The smaller multimodal agent that solves ChartQA tasks through the minimal action space. It receives no Reviewer assistance in C0, C2, or C3; C1 deliberately measures online Reviewer guidance as the ceiling.
_Avoid_: Base model, worker

**Reviewer**:
The Nemotron multimodal agent that consumes frozen Kappa Guidance and inspects proposed Student actions during calibration, measured Collection, and the C1 ceiling condition. It either accepts the proposal or supplies the smallest valid replacement.
_Avoid_: Judge, critic, teacher model

**Consolidator**:
The agent that converts repeated reviewed corrections into structured candidate behaviors while preserving their evidence and provenance.
_Avoid_: Summarizer, memory writer

**H-star Analysis Model**:
An optional calibration-only model that may propose one bounded H-star revision. Its exact endpoint is discovered at the live-provider gate, and deterministic regression evidence—not the model—decides acceptance.
_Avoid_: Required fourth runtime agent, promotion judge

## Memory and Evidence

**Operational Memory**:
The managed MemMachine project that exclusively stores reviewed episodes, candidate behaviors, accepted behavior versions, and the records retrieved by the Student.
_Avoid_: Local memory, JSONL store

**Experiment Trace**:
A local, append-only record used only for measurement, debugging, and visibly labeled demo replay. No agent may retrieve from it or use it as improvement context.
_Avoid_: Memory mirror, fallback memory

**Candidate Version**:
A frozen batch of proposed behavior changes that must pass the promotion gate before any Student can use it as the accepted operating procedure.
_Avoid_: New memory, latest memory

**Accepted Version**:
The immutable version root eligible for Student retrieval: initially empty Accepted v000, then the most recent Candidate Version that passed the promotion gate. It remains active until a better candidate replaces it.
_Avoid_: Production memory, current prompt

**Accepted v000**:
The immutable empty initial Accepted Version. It provides a real comparison root before any candidate is promoted and makes an unsuccessful first lifecycle pass representable without inventing memory.
_Avoid_: Missing version, implicit baseline

**Negative Transfer**:
A regression where a Candidate Version causes the Student to fail a task that the Accepted Version solved.
_Avoid_: Bad retrieval, random error

## Evaluation Sets

**Collection Set**:
The ChartQA items whose reviewer-guided episodes may produce Candidate Versions.
_Avoid_: Training examples, memory set

**Promotion Set**:
The validation items used to compare a Candidate Version with the Accepted Version and decide promotion; they never produce memories.
_Avoid_: Validation data, development set

**Final Held-Out Set**:
The untouched test items evaluated once after the final Accepted Version is frozen.
_Avoid_: Evaluation set, testmini

## Memory Records

**Reviewed Episode**:
A Collection Set trajectory containing the Student’s proposed actions, Reviewer decisions, tool observations, outcome, and provenance.
_Avoid_: Training example, raw memory

**Behavior Card**:
A structured, answer-free procedure abstracted from one or more Reviewed Episodes and eligible for inclusion in a Candidate Version.
_Avoid_: Retrieved example, summary, prompt fragment

**Promotion Gate**:
The automated comparison that promotes or rejects a Candidate Version using capability, Negative Transfer, and efficiency thresholds on the Promotion Set.
_Avoid_: Human approval, manual review

**Primary Evaluation Run**:
The single paired run of every required condition on the complete Final Held-Out Set under identical budgets and lowest-variance decoding.
_Avoid_: Benchmark run, final demo run

**Robustness Subset**:
A fixed subset repeated when time and quota permit to estimate run-to-run variance without making full repeated evaluation a completion requirement.
_Avoid_: Second test set, retry set

**Closed-Book Evaluation**:
An evaluation in which the Student may use its fixed tools and accepted Behavior Cards but cannot search the web or access external task solutions.
_Avoid_: Web-grounded evaluation, open-book benchmark

**Retrieval Policy**:
The frozen rule that filters and selects a bounded number of accepted Behavior Cards for a task, with a no-memory fallback when none is sufficiently relevant.
_Avoid_: Memory search, RAG pipeline

**Irrelevant-Memory Control (C3)**:
The Reviewer-free condition that receives Behavior Cards from a predeclared Accepted-Version donor pool whose triggers do not match the task. Assignment uses a frozen seed, matches C2 card count exactly and token count within 10%, and is reported unavailable when no honest match exists.
_Avoid_: Random memory, shuffled answers

**Definition of Done**:
Verified completion of the autonomous version lifecycle, required integrations, controlled evaluation, and evidence-rich demo, regardless of whether the final hypothesis is supported.
_Avoid_: Positive result, benchmark win
