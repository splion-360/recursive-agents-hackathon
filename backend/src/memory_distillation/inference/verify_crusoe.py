"""Bounded live verification for the configured Crusoe model roles."""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_distillation.benchmark import DevelopmentBenchmark
from memory_distillation.inference.crusoe import (
    CrusoeClient,
    CrusoeReviewer,
    CrusoeStructuredResult,
    CrusoeStudentInference,
    build_crusoe_client,
)
from memory_distillation.inference.protocol import InferenceRequest, InferenceResult
from memory_distillation.settings import Settings, get_settings
from memory_distillation.task_execution import FinalAnswerAction
from memory_distillation.task_execution.models import StudentAction
from memory_distillation.task_execution.review import ReviewRequest, ReviewResult


class ConsolidatorProbe(BaseModel):
    """Minimal answer-free Behavior Card operation used by the smoke gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["create"]
    title: str = Field(min_length=1, max_length=80)
    trigger: str = Field(min_length=1, max_length=200)
    instruction: str = Field(min_length=1, max_length=300)


Role = Literal["student", "reviewer", "consolidator"]


class CallEvidence(BaseModel):
    """Sanitized evidence for one live provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    model_id: str
    request_id: str
    outcome: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retry_count: int
    cost_usd: str


class ModelEvidence(BaseModel):
    """Live catalog facts relevant to one configured role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    model_id: str
    model_revision: str | None
    modality: str
    context_length: int
    supported_parameters: tuple[str, ...]


class VerificationEvidence(BaseModel):
    """Sanitized, machine-readable result of the bounded smoke gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["crusoe-verification.v1"] = "crusoe-verification.v1"
    calls: tuple[CallEvidence, ...]
    models: tuple[ModelEvidence, ...]
    total_tokens: int
    total_cost_usd: str
    remaining_tokens: int | None
    remaining_output_tokens: int | None


def verify(settings: Settings, *, dataset_root: Path, manifest_dir: Path) -> VerificationEvidence:
    """Run one bounded call per role against answer-free development inputs."""

    model_ids: dict[Role, str] = {
        "student": _required(settings.crusoe_student_model, "CRUSOE_STUDENT_MODEL"),
        "reviewer": _required(settings.crusoe_reviewer_model, "CRUSOE_REVIEWER_MODEL"),
        "consolidator": _required(settings.crusoe_consolidator_model, "CRUSOE_CONSOLIDATOR_MODEL"),
    }
    benchmark = DevelopmentBenchmark(dataset_root=dataset_root, manifest_dir=manifest_dir)
    task = benchmark.load_calibration().discovery[0]
    asset = benchmark.load_asset(task)

    with build_crusoe_client(settings) as client:
        models = tuple(
            _model_evidence(role, client, model_id) for role, model_id in model_ids.items()
        )
        student = CrusoeStudentInference(
            client=client,
            model_id=model_ids["student"],
            max_output_tokens=256,
        ).generate(
            InferenceRequest(
                task=task,
                asset=asset,
                observations=(),
                seed=20261017,
                turn_index=0,
            )
        )
        reviewer = CrusoeReviewer(
            client=client,
            model_id=model_ids["reviewer"],
            max_output_tokens=2_048,
        ).review(
            ReviewRequest(
                task=task,
                asset=asset,
                proposed_action=FinalAnswerAction(answer="unsupported capability probe"),
                observations=(),
                guidance=(
                    "REPLACE a final answer that has no supporting observation with inspect_image.",
                ),
                turn_index=0,
            )
        )
        _validate_reviewer_probe(reviewer)
        consolidator = client.complete_json(
            model_id=model_ids["consolidator"],
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Create one reusable answer-free Behavior Card operation from the "
                        "abstract failure pattern. Do not include task answers."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Failure pattern: the agent answered a chart question before "
                        "collecting visual evidence."
                    ),
                },
            ),
            output_type=ConsolidatorProbe,
            max_output_tokens=4_096,
            seed=20261017,
        )
        calls = (
            _student_evidence(student),
            _reviewer_evidence(reviewer),
            _consolidator_evidence(consolidator),
        )
        return VerificationEvidence(
            calls=calls,
            models=models,
            total_tokens=client.spent_tokens,
            total_cost_usd=str(client.spent_cost_usd),
            remaining_tokens=client.remaining_tokens,
            remaining_output_tokens=client.remaining_output_tokens,
        )


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required for live verification")
    return value


def _model_evidence(role: Role, client: CrusoeClient, model_id: str) -> ModelEvidence:
    model = client.get_model(model_id)
    return ModelEvidence(
        role=role,
        model_id=model.id,
        model_revision=model.model_revision,
        modality=model.architecture.modality,
        context_length=model.context_length,
        supported_parameters=model.supported_parameters,
    )


def _student_evidence(result: InferenceResult[StudentAction]) -> CallEvidence:
    return CallEvidence(
        role="student",
        model_id=result.model_id,
        request_id=result.request_id,
        outcome=result.output.type,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        retry_count=result.retry_count,
        cost_usd=str(result.actual_cost_usd),
    )


def _reviewer_evidence(result: ReviewResult) -> CallEvidence:
    return CallEvidence(
        role="reviewer",
        model_id=result.model_id,
        request_id=result.request_id,
        outcome=f"{result.decision}:{result.accepted_action.type}",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        retry_count=result.retry_count,
        cost_usd=str(result.actual_cost_usd),
    )


def _validate_reviewer_probe(result: ReviewResult) -> None:
    if result.decision != "replace" or result.accepted_action.type != "inspect_image":
        raise ValueError(
            "Crusoe Reviewer smoke required REPLACE with inspect_image for the unsupported answer"
        )


def _consolidator_evidence(
    result: CrusoeStructuredResult[ConsolidatorProbe],
) -> CallEvidence:
    return CallEvidence(
        role="consolidator",
        model_id=result.model_id,
        request_id=result.request_id,
        outcome=result.output.operation,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        retry_count=result.retry_count,
        cost_usd=str(result.actual_cost_usd),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Acknowledge that this command makes paid provider calls.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("--confirm-live is required for paid provider calls")
    evidence = verify(
        get_settings(), dataset_root=args.dataset_root, manifest_dir=args.manifest_dir
    )
    print(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
