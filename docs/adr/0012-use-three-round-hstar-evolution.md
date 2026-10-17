# Use three-round H-star evolution before Kappa freeze

Status: Accepted

H-star evolution uses twelve answer-free ChartQA Discovery tasks in three ordered rounds of four. After each round, the H-star Analysis Model receives cumulative public outcomes, action types, and the prior H-star revision, then returns a complete next revision. It never receives reference answers, private reasoning, Regression inputs, or final-set data.

The third revision is frozen before the twelve-task Regression manifest is loaded. Regression may only screen that exact proposal; it cannot trigger another revision. Passing components are deterministically adapted into Kappa Guidance, canonically hashed, and frozen before measured Collection.

The Student retains the fixed Minimal Harness actions: `inspect_image`, `crop`, `run_python`, and `final_answer`. The private Reviewer may PASS a proposed action or REPLACE it with the smallest valid action. When rejecting a proposed `final_answer`, it must replace it with evidence gathering (`inspect_image`, `crop`, or `run_python`), not reveal a corrected answer directly.

Calibration produces no Reviewed Episodes or Behavior Cards. This decision supersedes ADR 0009's one-revision 8+8 schedule while retaining its isolation and freeze boundaries.
