"""Typed Crusoe Managed Inference adapter with bounded cost admission."""

import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic, sleep
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

import httpx2
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from memory_distillation.benchmark import TaskAsset
from memory_distillation.inference.protocol import (
    InferenceRequest,
    InferenceResult,
)
from memory_distillation.settings import Settings
from memory_distillation.task_execution.models import StudentAction
from memory_distillation.task_execution.review import ReviewRequest, ReviewResult

OutputT = TypeVar("OutputT", bound=BaseModel)
_MILLION = Decimal(1_000_000)


class CrusoeBudget(BaseModel):
    """Hard aggregate admission limits for one client lifetime."""

    model_config = ConfigDict(frozen=True)

    max_cost_usd: Decimal = Field(gt=0)
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
    ) -> None:
        super().__init__(
            "Crusoe inference failed "
            f"(kind={failure_kind}, status={status_code}, request_id={request_id}, "
            f"retries={retry_count})"
        )
        self.status_code = status_code
        self.failure_kind = failure_kind
        self.request_id = request_id
        self.retry_count = retry_count


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
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: StudentAction


class _ReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["pass", "replace"]
    accepted_action: StudentAction
    private_reason: str = Field(min_length=1)


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
        self._admit(model=model, payload=payload, max_output_tokens=max_output_tokens)
        started = monotonic()
        response, retry_count = self._post_with_retries(payload)
        latency_ms = round((monotonic() - started) * 1000)
        self._remaining_tokens = _optional_int_header(response, "x-ratelimit-remaining-tokens")
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
                retry_count=retry_count,
            ) from error
        actual_cost = self._cost(
            model,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        self._spent_cost += actual_cost
        self._spent_tokens += completion.usage.total_tokens
        if completion.model != model_id:
            raise CrusoeProviderError(
                status_code=response.status_code,
                failure_kind="model_identity_mismatch",
                request_id=completion.id,
                retry_count=retry_count,
            )
        content = completion.choices[0].message.content
        if content is None:
            raise CrusoeProviderError(
                status_code=response.status_code,
                failure_kind="missing_structured_output",
                request_id=completion.id,
                retry_count=retry_count,
            )
        try:
            output = output_type.model_validate_json(content)
        except ValueError as error:
            raise CrusoeProviderError(
                status_code=response.status_code,
                failure_kind="invalid_structured_output",
                request_id=completion.id,
                retry_count=retry_count,
            ) from error
        return CrusoeStructuredResult(
            output=output,
            model_id=completion.model,
            request_id=completion.id,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            latency_ms=latency_ms,
            retry_count=retry_count,
            actual_cost_usd=actual_cost,
            remaining_tokens=self._remaining_tokens,
            remaining_output_tokens=self._remaining_output_tokens,
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
        estimated_tokens = serialized_bytes + max_output_tokens
        if estimated_tokens > model.context_length:
            raise CrusoeAdmissionError("request exceeds the model context limit")
        if self._spent_tokens + estimated_tokens > self._budget.max_total_tokens:
            raise CrusoeAdmissionError("request exceeds the aggregate token budget")
        estimated_cost = self._cost(
            model,
            input_tokens=serialized_bytes,
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
            # Only an explicit rate-limit rejection is known not to have executed
            # inference. Gateway failures are ambiguous and must not be replayed.
            if response.status_code == 429 and retry_count < self._max_retries:
                self._backoff(retry_count)
                retry_count += 1
                continue
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
                        "structured action. Never invent observations or hidden references."
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
            output=result.output.action,
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
                        "valid; otherwise REPLACE it with the smallest valid action."
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
        if result.output.decision == "pass" and (
            result.output.accepted_action != request.proposed_action
        ):
            raise ValueError("Crusoe Reviewer changed an action while claiming PASS")
        return ReviewResult(
            decision=result.output.decision,
            accepted_action=result.output.accepted_action,
            private_reason=result.output.private_reason,
            model_id=result.model_id,
            request_id=result.request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            retry_count=result.retry_count,
            actual_cost_usd=result.actual_cost_usd,
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
            max_cost_usd=settings.crusoe_max_cost_usd,
            max_total_tokens=settings.crusoe_max_total_tokens,
        ),
        timeout_seconds=settings.crusoe_timeout_seconds,
        max_retries=settings.crusoe_max_retries,
    )
