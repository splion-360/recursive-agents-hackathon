"""Typed Crusoe Managed Inference adapter with bounded cost admission."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic, sleep
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

import httpx2
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from memory_distillation.benchmark import TaskAsset
from memory_distillation.calibration.models import (
    HStarAnalysisRequest,
    HStarAnalysisResult,
)
from memory_distillation.inference.protocol import (
    InferenceRequest,
    InferenceResult,
)
from memory_distillation.review_policy import HStarProposal
from memory_distillation.settings import Settings
from memory_distillation.task_execution.models import (
    CropAction,
    FinalAnswerAction,
    InspectImageAction,
    RunPythonAction,
    StudentAction,
)
from memory_distillation.task_execution.review import ReviewRequest, ReviewResult

if TYPE_CHECKING:
    from memory_distillation.operational_memory.models import ReviewedEpisode
    from memory_distillation.versioning.models import ConsolidationBatch, VersionView

OutputT = TypeVar("OutputT", bound=BaseModel)
_MILLION = Decimal(1_000_000)


class CrusoeBudget(BaseModel):
    """Aggregate token admission with an optional caller-selected cost cap."""

    model_config = ConfigDict(frozen=True)

    max_cost_usd: Decimal | None = Field(default=None, gt=0)
    max_total_tokens: int = Field(gt=0)


class CrusoeAdmissionError(RuntimeError):
    """The request exceeded a configured cost, token, or context limit."""


class CrusoeProviderError(RuntimeError):
    """Sanitized provider failure with explicit submission semantics."""

    def __init__(
        self,
        *,
        status_code: int | None,
        failure_kind: str,
        request_id: str | None,
        retry_count: int,
        validation_summary: tuple[str, ...] = (),
    ) -> None:
        validation_detail = (
            f", validation={';'.join(validation_summary)}" if validation_summary else ""
        )
        super().__init__(
            "Crusoe inference failed "
            f"(kind={failure_kind}, status={status_code}, request_id={request_id}, "
            f"retries={retry_count}{validation_detail})"
        )
        self.status_code = status_code
        self.failure_kind = failure_kind
        self.request_id = request_id
        self.retry_count = retry_count
        self.validation_summary = validation_summary


class CrusoePricing(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt: Decimal = Field(ge=0)
    completion: Decimal = Field(ge=0)


class CrusoeArchitecture(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    modality: str
    parameter_count: int | None = None


class CrusoeModel(BaseModel):
    """Cost and capability metadata returned by Crusoe's live catalog."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    model_revision: str | None = None
    context_length: int = Field(gt=0)
    supported_parameters: tuple[str, ...]
    pricing: CrusoePricing
    architecture: CrusoeArchitecture


class _ModelCatalog(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: tuple[CrusoeModel, ...]


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: str | None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    finish_reason: str | None
    message: _Message


class _Completion(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    model: str
    choices: tuple[_Choice, ...] = Field(min_length=1)
    usage: _Usage


@dataclass(frozen=True, slots=True)
class CrusoeStructuredResult[OutputT]:
    output: OutputT
    model_id: str
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retry_count: int
    actual_cost_usd: Decimal
    remaining_tokens: int | None
    remaining_output_tokens: int | None


class _StudentOutput(BaseModel):
    """Provider-friendly flat action shape converted to the domain union."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: Literal["inspect_image", "crop", "run_python", "final_answer"]
    answer: str | None = Field(default=None, min_length=1, max_length=2_000)
    code: str | None = Field(default=None, min_length=1, max_length=10_000)
    x0: int | None = Field(default=None, ge=0)
    y0: int | None = Field(default=None, ge=0)
    x1: int | None = Field(default=None, gt=0)
    y1: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_action_payload(self) -> _StudentOutput:
        required_fields = {
            "inspect_image": frozenset(),
            "crop": frozenset({"x0", "y0", "x1", "y1"}),
            "run_python": frozenset({"code"}),
            "final_answer": frozenset({"answer"}),
        }[self.action_type]
        payload = {
            "answer": self.answer,
            "code": self.code,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }
        supplied_fields = {name for name, value in payload.items() if value is not None}
        if supplied_fields != required_fields:
            raise ValueError(
                f"{self.action_type} requires exactly {sorted(required_fields)} payload fields"
            )
        return self

    def to_action(self) -> StudentAction:
        if self.action_type == "inspect_image":
            return InspectImageAction()
        if self.action_type == "run_python":
            assert self.code is not None
            return RunPythonAction(code=self.code)
        if self.action_type == "final_answer":
            assert self.answer is not None
            return FinalAnswerAction(answer=self.answer)
        assert self.x0 is not None
        assert self.y0 is not None
        assert self.x1 is not None
        assert self.y1 is not None
        return CropAction(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)


class _ReviewOutput(BaseModel):
    """Provider-compatible review shape converted into a typed Student action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["pass", "replace"]
    accepted_action_type: Literal["inspect_image", "crop", "run_python", "final_answer"]
    answer: str | None = Field(min_length=1, max_length=2_000)
    code: str | None = Field(min_length=1, max_length=10_000)
    x0: int | None = Field(ge=0)
    y0: int | None = Field(ge=0)
    x1: int | None = Field(gt=0)
    y1: int | None = Field(gt=0)
    private_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action_payload(self) -> _ReviewOutput:
        required_fields = {
            "inspect_image": frozenset(),
            "crop": frozenset({"x0", "y0", "x1", "y1"}),
            "run_python": frozenset({"code"}),
            "final_answer": frozenset({"answer"}),
        }[self.accepted_action_type]
        payload = {
            "answer": self.answer,
            "code": self.code,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }
        supplied_fields = {name for name, value in payload.items() if value is not None}
        if supplied_fields != required_fields:
            raise ValueError(
                f"{self.accepted_action_type} requires exactly "
                f"{sorted(required_fields)} payload fields"
            )
        return self

    def to_action(self) -> StudentAction:
        """Build the canonical discriminated action after strict payload validation."""

        if self.accepted_action_type == "inspect_image":
            return InspectImageAction()
        if self.accepted_action_type == "run_python":
            assert self.code is not None
            return RunPythonAction(code=self.code)
        if self.accepted_action_type == "final_answer":
            assert self.answer is not None
            return FinalAnswerAction(answer=self.answer)
        assert self.x0 is not None
        assert self.y0 is not None
        assert self.x1 is not None
        assert self.y1 is not None
        return CropAction(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)


class _CardOperationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CREATE", "REINFORCE", "REFINE", "RETIRE"]
    card_id: str
    trigger: str | None = None
    instruction: str | None = None
    evidence_episode_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_domain_operation(self) -> _CardOperationOutput:
        """Reject unusable card operations while provider retries remain available."""

        from memory_distillation.versioning.models import CardOperation

        CardOperation.model_validate(self.model_dump(mode="json"))
        return self


class _ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operations: tuple[_CardOperationOutput, ...] = Field(min_length=1, max_length=8)


class CrusoeClient:
    """Call Crusoe through one provider boundary and return validated output."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        budget: CrusoeBudget,
        transport: httpx2.BaseTransport | None = None,
        timeout_seconds: float = 60,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._budget = budget
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._catalog: dict[str, CrusoeModel] | None = None
        self._spent_cost = Decimal(0)
        self._spent_tokens = 0
        self._remaining_tokens: int | None = None
        self._remaining_output_tokens: int | None = None
        self._submission_uncertain = False
        self._client = httpx2.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx2.Timeout(timeout_seconds),
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @property
    def spent_cost_usd(self) -> Decimal:
        """Return successful-call spend observed during this client lifetime."""

        return self._spent_cost

    @property
    def spent_tokens(self) -> int:
        """Return successful-call tokens observed during this client lifetime."""

        return self._spent_tokens

    @property
    def remaining_tokens(self) -> int | None:
        """Return the last provider-advertised aggregate token allowance."""

        return self._remaining_tokens

    @property
    def remaining_output_tokens(self) -> int | None:
        """Return the last provider-advertised output-token allowance."""

        return self._remaining_output_tokens

    def list_models(self) -> tuple[CrusoeModel, ...]:
        if self._catalog is None:
            response = self._client.get("models")
            response.raise_for_status()
            models = _ModelCatalog.model_validate(response.json()).data
            self._catalog = {model.id: model for model in models}
        return tuple(self._catalog.values())

    def get_model(self, model_id: str) -> CrusoeModel:
        """Return one exact model capability or fail closed."""

        return self._model(model_id)

    def complete_json(
        self,
        *,
        model_id: str,
        messages: tuple[dict[str, object], ...],
        output_type: type[OutputT],
        max_output_tokens: int,
        seed: int,
    ) -> CrusoeStructuredResult[OutputT]:
        model = self._model(model_id)
        if "response_format" not in model.supported_parameters:
            raise ValueError(f"model {model_id!r} does not advertise structured output")
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "strict": True,
                    "schema": output_type.model_json_schema(),
                },
            },
        }
        if "reasoning_effort" in model.supported_parameters:
            payload["reasoning_effort"] = "low"
        started = monotonic()
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = Decimal(0)
        total_retries = 0
        structured_retries = 0
        while True:
            # A schema-invalid completion is a completed, billed inference. Vary
            # only its retry seed deterministically; transport retries retain the
            # current seed because the provider explicitly rejected submission.
            payload["seed"] = seed + structured_retries
            self._admit(model=model, payload=payload, max_output_tokens=max_output_tokens)
            response, transport_retries = self._post_with_retries(payload)
            total_retries += transport_retries
            self._remaining_tokens = _optional_int_header(
                response, "x-ratelimit-remaining-tokens"
            )
            self._remaining_output_tokens = _optional_int_header(
                response, "x-ratelimit-remaining-output-tokens"
            )
            try:
                completion = _Completion.model_validate(response.json())
            except ValueError as error:
                self._submission_uncertain = True
                raise CrusoeProviderError(
                    status_code=response.status_code,
                    failure_kind="invalid_response",
                    request_id=response.headers.get("x-request-id"),
                    retry_count=total_retries,
                ) from error

            actual_cost = self._cost(
                model,
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            )
            self._spent_cost += actual_cost
            self._spent_tokens += completion.usage.total_tokens
            total_cost += actual_cost
            total_input_tokens += completion.usage.prompt_tokens
            total_output_tokens += completion.usage.completion_tokens
            if completion.model != model_id:
                raise CrusoeProviderError(
                    status_code=response.status_code,
                    failure_kind="model_identity_mismatch",
                    request_id=completion.id,
                    retry_count=total_retries,
                )

            choice = completion.choices[0]
            content = choice.message.content
            failure_kind: str | None = None
            validation_summary: tuple[str, ...] = ()
            output: OutputT | None = None
            if content is None:
                failure_kind = "missing_structured_output"
            else:
                try:
                    output = output_type.model_validate_json(content)
                except ValidationError as error:
                    failure_kind = (
                        "output_truncated"
                        if choice.finish_reason == "length"
                        else "invalid_structured_output"
                    )
                    validation_summary = tuple(
                        f"{'.'.join(str(part) for part in issue['loc']) or '<root>'}:"
                        f"{issue['type']}"
                        for issue in error.errors(include_input=False, include_url=False)[:5]
                    )

            if output is not None:
                return CrusoeStructuredResult(
                    output=output,
                    model_id=completion.model,
                    request_id=completion.id,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=round((monotonic() - started) * 1000),
                    retry_count=total_retries,
                    actual_cost_usd=total_cost,
                    remaining_tokens=self._remaining_tokens,
                    remaining_output_tokens=self._remaining_output_tokens,
                )

            if structured_retries < self._max_retries:
                self._backoff(structured_retries)
                structured_retries += 1
                total_retries += 1
                continue
            raise CrusoeProviderError(
                status_code=response.status_code,
                failure_kind=failure_kind or "invalid_structured_output",
                request_id=completion.id,
                retry_count=total_retries,
                validation_summary=validation_summary,
            )

    def _admit(
        self,
        *,
        model: CrusoeModel,
        payload: dict[str, Any],
        max_output_tokens: int,
    ) -> None:
        if self._submission_uncertain:
            raise CrusoeAdmissionError("client is sealed after an uncertain submission outcome")
        if max_output_tokens <= 0:
            raise CrusoeAdmissionError("output token limit must be positive")
        serialized_bytes = len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        # Serialized request bytes are not tokens. Treating base64 image bytes as
        # one token each falsely rejects ordinary ChartQA images before Crusoe
        # can apply its multimodal tokenizer and authoritative context check.
        estimated_input_tokens = max(1, serialized_bytes // 4)
        estimated_tokens = estimated_input_tokens + max_output_tokens
        if estimated_tokens > model.context_length:
            raise CrusoeAdmissionError("request exceeds the model context limit")
        if self._spent_tokens + estimated_tokens > self._budget.max_total_tokens:
            raise CrusoeAdmissionError("request exceeds the aggregate token budget")
        if self._budget.max_cost_usd is not None:
            estimated_cost = self._cost(
                model,
                input_tokens=estimated_input_tokens,
                output_tokens=max_output_tokens,
            )
            if self._spent_cost + estimated_cost > self._budget.max_cost_usd:
                raise CrusoeAdmissionError("request exceeds the aggregate cost budget")

    def _post_with_retries(self, payload: dict[str, Any]) -> tuple[httpx2.Response, int]:
        retry_count = 0
        while True:
            try:
                response = self._client.post("chat/completions", json=payload)
            except httpx2.ConnectError as error:
                if retry_count < self._max_retries:
                    self._backoff(retry_count)
                    retry_count += 1
                    continue
                raise CrusoeProviderError(
                    status_code=None,
                    failure_kind="pre_submit",
                    request_id=None,
                    retry_count=retry_count,
                ) from error
            except (
                httpx2.ReadTimeout,
                httpx2.ReadError,
                httpx2.RemoteProtocolError,
            ) as error:
                # Chat completion is side-effect free. If the request was
                # accepted but its response was lost, replaying the frozen
                # payload preserves experiment semantics (with a possible
                # duplicate provider charge).
                if retry_count < self._max_retries:
                    self._backoff(retry_count)
                    retry_count += 1
                    continue
                self._submission_uncertain = True
                raise CrusoeProviderError(
                    status_code=None,
                    failure_kind="response_unavailable",
                    request_id=None,
                    retry_count=retry_count,
                ) from error
            except httpx2.RequestError as error:
                self._submission_uncertain = True
                raise CrusoeProviderError(
                    status_code=None,
                    failure_kind="uncertain",
                    request_id=None,
                    retry_count=retry_count,
                ) from error
            if response.status_code < 400:
                return response, retry_count
            request_id = response.headers.get("x-request-id")
            # Crusoe documents 429 and 503 as transient rejections that are safe
            # to retry. Other gateway failures remain submission-ambiguous.
            if response.status_code in {429, 503}:
                if retry_count < self._max_retries:
                    self._backoff(retry_count)
                    retry_count += 1
                    continue
                raise CrusoeProviderError(
                    status_code=response.status_code,
                    failure_kind=(
                        "rate_limited"
                        if response.status_code == 429
                        else "provider_unavailable"
                    ),
                    request_id=request_id,
                    retry_count=retry_count,
                )
            failure_kind = "provider_rejected"
            if response.status_code >= 500:
                self._submission_uncertain = True
                failure_kind = "uncertain"
            raise CrusoeProviderError(
                status_code=response.status_code,
                failure_kind=failure_kind,
                request_id=request_id,
                retry_count=retry_count,
            )

    def _backoff(self, retry_count: int) -> None:
        sleep(self._retry_backoff_seconds * (2**retry_count))

    def _model(self, model_id: str) -> CrusoeModel:
        models = {model.id: model for model in self.list_models()}
        try:
            return models[model_id]
        except KeyError as error:
            raise ValueError(f"Crusoe model is unavailable: {model_id}") from error

    @staticmethod
    def _cost(model: CrusoeModel, *, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            Decimal(input_tokens) * model.pricing.prompt
            + Decimal(output_tokens) * model.pricing.completion
        ) / _MILLION


class CrusoeStudentInference:
    """Student action adapter using one configured Crusoe multimodal model."""

    def __init__(
        self,
        *,
        client: CrusoeClient,
        model_id: str,
        max_output_tokens: int,
    ) -> None:
        model = client.get_model(model_id)
        if model.architecture.modality != "multimodal":
            raise ValueError("Crusoe Student model must be multimodal")
        self._client = client
        self._model_id = model_id
        self._max_output_tokens = max_output_tokens

    def generate(self, request: InferenceRequest) -> InferenceResult[StudentAction]:
        prompt = {
            "task_id": request.task.task_id,
            "question": request.task.question,
            "observations": request.observations,
            "behavior_cards": request.behavior_cards,
            "turn_index": request.turn_index,
        }
        result = self._client.complete_json(
            model_id=self._model_id,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Act through exactly one allowed ChartQA action. Return only the "
                        "requested flat structured JSON. Set action_type and populate only "
                        "the fields required by that action: answer for final_answer, code "
                        "for run_python, x0/y0/x1/y1 for crop, or none for inspect_image. "
                        "Set every unused payload field to null. Never invent observations "
                        "or hidden references."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(prompt, sort_keys=True)},
                        {
                            "type": "image_url",
                            "image_url": {"url": _asset_data_url(request.asset)},
                        },
                    ],
                },
            ),
            output_type=_StudentOutput,
            max_output_tokens=self._max_output_tokens,
            seed=request.seed,
        )
        return InferenceResult(
            output=result.output.to_action(),
            model_id=result.model_id,
            request_id=result.request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            retry_count=result.retry_count,
            actual_cost_usd=result.actual_cost_usd,
        )


class CrusoeReviewer:
    """Private PASS/REPLACE adapter constrained to a multimodal Nemotron model."""

    def __init__(
        self,
        *,
        client: CrusoeClient,
        model_id: str,
        max_output_tokens: int,
    ) -> None:
        model = client.get_model(model_id)
        if model.architecture.modality != "multimodal" or "nemotron" not in model_id.casefold():
            raise ValueError("Crusoe Reviewer must be a multimodal Nemotron model")
        self._client = client
        self._model_id = model_id
        self._max_output_tokens = max_output_tokens

    def review(self, request: ReviewRequest) -> ReviewResult:
        prompt = {
            "task_id": request.task.task_id,
            "question": request.task.question,
            "proposed_action": request.proposed_action.model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in request.observations],
            "kappa_guidance": request.guidance,
            "turn_index": request.turn_index,
        }
        result = self._client.complete_json(
            model_id=self._model_id,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Privately review one proposed ChartQA action. PASS it unchanged when "
                        "valid; otherwise REPLACE it with the smallest valid action. Return only "
                        "the requested flat structured JSON with a brief non-empty "
                        "private_reason. Set accepted_action_type and populate only the fields "
                        "needed by that action: answer for final_answer, code for run_python, "
                        "x0/y0/x1/y1 for crop, or none for inspect_image. Set every unused "
                        "action payload field to null. If the proposed action is an unsupported "
                        "final_answer, REPLACE it with inspect_image, crop, or run_python to "
                        "gather evidence; never reveal a corrected final_answer directly."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(prompt, sort_keys=True)},
                        {
                            "type": "image_url",
                            "image_url": {"url": _asset_data_url(request.asset)},
                        },
                    ],
                },
            ),
            output_type=_ReviewOutput,
            max_output_tokens=self._max_output_tokens,
            seed=request.turn_index,
        )
        accepted_action = result.output.to_action()
        if result.output.decision == "pass" and accepted_action != request.proposed_action:
            raise ValueError("Crusoe Reviewer changed an action while claiming PASS")
        if (
            result.output.decision == "replace"
            and request.proposed_action.type == "final_answer"
            and accepted_action.type == "final_answer"
        ):
            raise ValueError(
                "Crusoe Reviewer must replace an unsupported final answer with an "
                "evidence-gathering action"
            )
        return ReviewResult(
            decision=result.output.decision,
            accepted_action=accepted_action,
            private_reason=result.output.private_reason,
            model_id=result.model_id,
            request_id=result.request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            retry_count=result.retry_count,
            actual_cost_usd=result.actual_cost_usd,
        )


class CrusoeHStarAnalyst:
    """Evolve a bounded review-policy harness from discovery-only evidence."""

    def __init__(
        self,
        *,
        client: CrusoeClient,
        model_id: str,
        max_output_tokens: int,
    ) -> None:
        client.get_model(model_id)
        self._client = client
        self._model_id = model_id
        self._max_output_tokens = max_output_tokens

    def propose(self, request: HStarAnalysisRequest, *, seed: int) -> HStarAnalysisResult:
        result = self._client.complete_json(
            model_id=self._model_id,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Analyze only the supplied ChartQA discovery outcomes and prior "
                        "proposal. Return the full next H-star revision containing exactly "
                        "one general, answer-free review rule. Set revision to round_index "
                        "and parent_revision to round_index minus one. "
                        "Classify its source_component as tools, middleware, skills, or "
                        "failure_memory. "
                        "Rules must use only inspect_image, crop, run_python, or final_answer. "
                        "Do not include task IDs, questions, chart values, answers, paths, or "
                        "unsupported tools in rule instructions. Return the exact discovery "
                        "task IDs only in discovery_task_ids."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request.model_dump(mode="json"), sort_keys=True),
                },
            ),
            output_type=HStarProposal,
            max_output_tokens=self._max_output_tokens,
            seed=seed,
        )
        # Revision lineage and evidence visibility are orchestrator-owned
        # provenance, not model judgments. Bind them from the validated request
        # so a schema-valid stale model default cannot corrupt the chain.
        proposal = result.output.model_copy(
            update={
                "parent_revision": request.round_index - 1,
                "revision": request.round_index,
                "discovery_task_ids": request.discovery_task_ids,
            }
        )
        return HStarAnalysisResult(
            proposal=proposal,
            model_id=result.model_id,
            request_id=result.request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            actual_cost_usd=result.actual_cost_usd,
        )


class CrusoeConsolidator:
    """Propose answer-free Behavior Card operations from one frozen snapshot."""

    def __init__(
        self,
        *,
        client: CrusoeClient,
        model_id: str,
        max_output_tokens: int,
    ) -> None:
        client.get_model(model_id)
        self._client = client
        self._model_id = model_id
        self._max_output_tokens = max_output_tokens

    def consolidate(
        self,
        *,
        episodes: tuple[ReviewedEpisode, ...],
        parent_view: VersionView,
        seed: int,
    ) -> ConsolidationBatch:
        """Return one bounded operation batch; deterministic validators remain authoritative."""

        from memory_distillation.versioning.models import CardOperation, ConsolidationBatch

        if not episodes:
            raise ValueError("consolidation requires eligible Reviewed Episodes")
        summaries = tuple(
            {
                "episode_id": episode.episode_id,
                "question": episode.task.question,
                "correct": episode.correct,
                "termination_reason": episode.termination_reason,
                "action_types": [action.type for action in episode.actions],
                "review_decisions": [review.decision for review in episode.reviews],
            }
            for episode in episodes
        )
        result = self._client.complete_json(
            model_id=self._model_id,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Distill the supplied reviewed ChartQA episode summaries into one to "
                        "eight general procedural Behavior Card operations. For an empty parent, "
                        "use CREATE only. Every episode_id must appear in at least one operation's "
                        "evidence_episode_ids. CREATE and REFINE require non-empty trigger and "
                        "instruction fields. REINFORCE and RETIRE must omit those fields. Triggers "
                        "and instructions must be answer-free and must not copy task IDs, "
                        "questions, chart labels, numbers, or answers."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "parent_view": parent_view.model_dump(mode="json"),
                            "episodes": summaries,
                        },
                        sort_keys=True,
                    ),
                },
            ),
            output_type=_ConsolidationOutput,
            max_output_tokens=self._max_output_tokens,
            seed=seed,
        )
        return ConsolidationBatch(
            operations=tuple(
                CardOperation.model_validate(item.model_dump(mode="json"))
                for item in result.output.operations
            ),
            model_id=result.model_id,
            request_id=result.request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )


def _asset_data_url(asset: TaskAsset) -> str:
    encoded = base64.b64encode(asset.content).decode("ascii")
    return f"data:{asset.media_type};base64,{encoded}"


def _optional_int_header(response: httpx2.Response, name: str) -> int | None:
    value = response.headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def build_crusoe_client(settings: Settings) -> CrusoeClient:
    """Compose the live client exclusively from centralized application settings."""

    if settings.crusoe_api_key is None:
        raise ValueError("CRUSOE_API_KEY is required for live inference")
    return CrusoeClient(
        base_url=str(settings.crusoe_base_url),
        api_key=settings.crusoe_api_key,
        budget=CrusoeBudget(
            max_total_tokens=settings.crusoe_max_total_tokens,
        ),
        timeout_seconds=settings.crusoe_timeout_seconds,
        max_retries=settings.crusoe_max_retries,
    )
