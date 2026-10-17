# Crusoe capability verification — 2026-10-17

## Verdict

The three planned Crusoe roles passed bounded live contract probes against
`https://api.inference.crusoecloud.com/v1`. This verifies provider compatibility;
it is not a measured experiment or a model-quality claim.

The command admitted at most $0.05 and 500,000 estimated tokens for its client
lifetime. The successful three-call run used 3,009 provider-reported tokens and
cost $0.00084375. No final-set task, gold answer, Behavior Card, or managed-memory
write was used.

## Frozen role selection

| Role | Exact model ID | Modality | Context | Structured output | Catalog revision |
|---|---|---:|---:|---|---|
| Student | `Qwen/Qwen3.8-27B` | multimodal | 262,144 | `response_format` and `structured_outputs` advertised; typed action validated | Not exposed by `/models` |
| Reviewer | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` | multimodal | 262,144 | `response_format` and `structured_outputs` advertised; typed review validated | Not exposed by `/models` |
| Consolidator | `nvidia/Nemotron-3.5-Lightning-30B-A3B` | text | 262,144 | `response_format` and `structured_outputs` advertised; strict card operation validated | Not exposed by `/models` |

All three catalog entries also advertised `seed`, `reasoning_effort`, and tool
parameters. The adapter sends `reasoning_effort=low` when advertised.

## Successful live evidence

| Role | Validated outcome | Request ID | Input | Output | Latency | Retries | Cost |
|---|---|---|---:|---:|---:|---:|---:|
| Student | `final_answer` action from a calibration-discovery ChartQA image | `d600ee3054df4531893d5e7397b99768` | 641 | 105 | 1,103 ms | 0 | $0.00037025 |
| Reviewer | `replace:inspect_image` for an unsupported answer under Kappa-like guidance | `chatcmpl-84844aac6d24ac4b` | 660 | 472 | 1,630 ms | 0 | $0.00025480 |
| Consolidator | strict `create` Behavior Card operation from an answer-free failure pattern | `chatcmpl-5841785e` | 50 | 1,081 | 8,364 ms | 0 | $0.00021870 |

The account advertised a 500,000-token request limit and a 50,000-output-token
limit in live completion response headers. The final call reported 499,947
remaining tokens and 49,998 remaining output tokens; both values are included in
the command's machine-readable output.

## Observed constraints

- Reasoning models can spend the entire completion allowance before emitting
  structured content. The Reviewer required more than 384 completion tokens and
  the Consolidator required more than 1,024 in these probes. The gate now uses
  ceilings of 2,048 and 4,096 respectively while retaining aggregate admission.
- Crusoe did not expose immutable model revisions in its model catalog or
  completion response. Exact provider model IDs are therefore the strongest live
  identity available; every measured run must also record the code/configuration
  revision.
- The adapter retries only pre-submission connection failures and explicit 429
  rejections. Gateway and ambiguous transport failures are surfaced without a
  retry, and an uncertain client is sealed, to avoid silently duplicating a
  potentially charged request.

## Reproduction

From `backend/`, with the API key and frozen model IDs configured in `.env`:

```bash
uv run verify-crusoe --confirm-live \
  --dataset-root '../.local/chartqa-source/ChartQA Dataset' \
  --manifest-dir experiment_manifests
```

The command prints sanitized JSON only. It does not persist prompts, images,
private Reviewer reasoning, provider payloads, or credentials.
