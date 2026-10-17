import json
from decimal import Decimal
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel, SecretStr

from memory_distillation.benchmark import TaskAsset, TaskInput
from memory_distillation.calibration import DiscoveryObservation, HStarAnalysisRequest
from memory_distillation.inference.crusoe import (
    CrusoeAdmissionError,
    CrusoeBudget,
    CrusoeClient,
    CrusoeHStarAnalyst,
    CrusoeProviderError,
    CrusoeReviewer,
    CrusoeStudentInference,
    build_crusoe_client,
)
from memory_distillation.inference.protocol import InferenceRequest
from memory_distillation.inference.verify_crusoe import _validate_reviewer_probe
from memory_distillation.settings import Settings
from memory_distillation.task_execution import FinalAnswerAction, InspectImageAction
from memory_distillation.task_execution.review import ReviewRequest, ReviewResult


class FixtureOutput(BaseModel):
    status: str


def test_client_composition_requires_a_configured_api_key() -> None:
    settings = Settings(_env_file=None, crusoe_api_key=None)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="CRUSOE_API_KEY"):
        build_crusoe_client(settings)


def _catalog_response(
    *, model_id: str = "fixture/model", modality: str = "text"
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "context_length": 8192,
                    "supported_parameters": ["response_format", "seed"],
                    "pricing": {"prompt": "0.50", "completion": "1.00"},
                    "architecture": {"modality": modality, "parameter_count": 1},
                }
            ],
        },
    )


def _completion_response(
    *, content: str = '{"status":"ok"}', model_id: str = "fixture/model"
) -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={
            "x-request-id": "header-request-id",
            "x-ratelimit-remaining-tokens": "499975",
            "x-ratelimit-remaining-output-tokens": "9995",
        },
        json={
            "id": "chatcmpl-fixture",
            "model": model_id,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        },
    )


def test_structured_completion_records_identity_usage_and_cost() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer fixture-secret"
        if request.url.path.endswith("/models"):
            return _catalog_response()
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["seed"] == 17
        return _completion_response()

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        result = client.complete_json(
            model_id="fixture/model",
            messages=({"role": "user", "content": "Return ok."},),
            output_type=FixtureOutput,
            max_output_tokens=16,
            seed=17,
        )

    assert result.output == FixtureOutput(status="ok")
    assert result.model_id == "fixture/model"
    assert result.request_id == "chatcmpl-fixture"
    assert result.input_tokens == 20
    assert result.output_tokens == 5
    assert result.retry_count == 0
    assert result.actual_cost_usd == Decimal("0.000015")
    assert result.remaining_tokens == 499975
    assert result.remaining_output_tokens == 9995
    assert result.latency_ms >= 0
    assert client.spent_cost_usd == Decimal("0.000015")
    assert client.spent_tokens == 25
    assert client.remaining_tokens == 499975
    assert client.remaining_output_tokens == 9995
    assert [request.url.path for request in requests] == ["/v1/models", "/v1/chat/completions"]


def test_cost_admission_rejects_before_inference_submission() -> None:
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/models"):
            return _catalog_response()
        raise AssertionError("inference must not be submitted after admission rejection")

    with (
        CrusoeClient(
            base_url="https://inference.example.test/v1",
            api_key=SecretStr("fixture-secret"),
            budget=CrusoeBudget(max_cost_usd=Decimal("0.000001"), max_total_tokens=10),
            transport=httpx2.MockTransport(handler),
        ) as client,
        pytest.raises(CrusoeAdmissionError, match=r"cost|token"),
    ):
        client.complete_json(
            model_id="fixture/model",
            messages=({"role": "user", "content": "Return ok."},),
            output_type=FixtureOutput,
            max_output_tokens=16,
            seed=17,
        )

    assert paths == ["/v1/models"]


def test_retryable_provider_rejection_is_bounded_and_observable() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        if attempts == 1:
            return httpx2.Response(429, json={"errors": ["rate limited"]})
        return _completion_response()

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
        max_retries=1,
    ) as client:
        result = client.complete_json(
            model_id="fixture/model",
            messages=({"role": "user", "content": "Return ok."},),
            output_type=FixtureOutput,
            max_output_tokens=16,
            seed=17,
        )

    assert attempts == 2
    assert result.retry_count == 1


def test_model_identity_mismatch_fails_closed_after_accounting_usage() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response()
        return _completion_response(model_id="provider/fallback")

    with (
        CrusoeClient(
            base_url="https://inference.example.test/v1",
            api_key=SecretStr("fixture-secret"),
            budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
            transport=httpx2.MockTransport(handler),
        ) as client,
        pytest.raises(CrusoeProviderError, match="model_identity_mismatch"),
    ):
        client.complete_json(
            model_id="fixture/model",
            messages=({"role": "user", "content": "Return ok."},),
            output_type=FixtureOutput,
            max_output_tokens=16,
            seed=17,
        )

    assert client.spent_tokens == 25


def test_nonretryable_provider_failure_is_typed_and_secret_free() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response()
        return httpx2.Response(400, json={"errors": ["invalid request"]})

    with (
        CrusoeClient(
            base_url="https://inference.example.test/v1",
            api_key=SecretStr("fixture-secret"),
            budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
            transport=httpx2.MockTransport(handler),
        ) as client,
        pytest.raises(CrusoeProviderError) as captured,
    ):
        client.complete_json(
            model_id="fixture/model",
            messages=({"role": "user", "content": "Return ok."},),
            output_type=FixtureOutput,
            max_output_tokens=16,
            seed=17,
        )

    assert captured.value.status_code == 400
    assert captured.value.failure_kind == "provider_rejected"
    assert "fixture-secret" not in str(captured.value)


def test_uncertain_submission_prevents_further_requests() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        raise httpx2.ReadTimeout("response outcome unknown", request=request)

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        with pytest.raises(CrusoeProviderError) as captured:
            client.complete_json(
                model_id="fixture/model",
                messages=({"role": "user", "content": "Return ok."},),
                output_type=FixtureOutput,
                max_output_tokens=16,
                seed=17,
            )
        assert captured.value.failure_kind == "uncertain"
        with pytest.raises(CrusoeAdmissionError, match="uncertain submission"):
            client.complete_json(
                model_id="fixture/model",
                messages=({"role": "user", "content": "Return ok."},),
                output_type=FixtureOutput,
                max_output_tokens=16,
                seed=17,
            )

    assert attempts == 1


def test_ambiguous_gateway_failure_seals_the_client() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        return httpx2.Response(503, json={"errors": ["gateway unavailable"]})

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
        max_retries=1,
    ) as client:
        with pytest.raises(CrusoeProviderError) as captured:
            client.complete_json(
                model_id="fixture/model",
                messages=({"role": "user", "content": "Return ok."},),
                output_type=FixtureOutput,
                max_output_tokens=16,
                seed=17,
            )
        assert captured.value.failure_kind == "uncertain"
        with pytest.raises(CrusoeAdmissionError, match="uncertain submission"):
            client.complete_json(
                model_id="fixture/model",
                messages=({"role": "user", "content": "Return ok."},),
                output_type=FixtureOutput,
                max_output_tokens=16,
                seed=17,
            )

    assert attempts == 1


def test_student_serializes_one_image_and_returns_a_typed_action() -> None:
    submitted: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response(model_id="fixture/student-vl", modality="multimodal")
        submitted.update(json.loads(request.content))
        return _completion_response(
            content='{"action":{"type":"final_answer","answer":"42"}}',
            model_id="fixture/student-vl",
        )

    task = TaskInput(
        task_id="chartqa:fixture:001",
        question="What is the total?",
        image_asset_id="fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    asset = TaskAsset(
        task_id=task.task_id,
        media_type="image/png",
        image_sha256=task.image_sha256,
        content=b"synthetic-image",
    )
    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=5000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        result = CrusoeStudentInference(
            client=client,
            model_id="fixture/student-vl",
            max_output_tokens=128,
        ).generate(
            InferenceRequest(
                task=task,
                asset=asset,
                observations=("observed labels",),
                seed=17,
                turn_index=0,
                behavior_cards=("Check units before arithmetic.",),
            )
        )

    assert result.output == FinalAnswerAction(answer="42")
    assert result.retry_count == 0
    assert result.actual_cost_usd == Decimal("0.000015")
    user_content = submitted["messages"][1]["content"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_reviewer_uses_kappa_and_returns_a_typed_replacement() -> None:
    submitted: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response(model_id="nvidia/fixture-nemotron-vl", modality="multimodal")
        submitted.update(json.loads(request.content))
        return _completion_response(
            content=(
                '{"decision":"replace","accepted_action":{"type":"inspect_image"},'
                '"private_reason":"Need chart evidence first."}'
            ),
            model_id="nvidia/fixture-nemotron-vl",
        )

    task = TaskInput(
        task_id="chartqa:fixture:001",
        question="What is the total?",
        image_asset_id="fixture.png",
        image_sha256="a" * 64,
        source_split="train",
    )
    asset = TaskAsset(
        task_id=task.task_id,
        media_type="image/png",
        image_sha256=task.image_sha256,
        content=b"synthetic-image",
    )
    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=5000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        result = CrusoeReviewer(
            client=client,
            model_id="nvidia/fixture-nemotron-vl",
            max_output_tokens=128,
        ).review(
            ReviewRequest(
                task=task,
                asset=asset,
                proposed_action=FinalAnswerAction(answer="42"),
                observations=(),
                guidance=("Require visual evidence before answering.",),
                turn_index=0,
            )
        )

    assert result.decision == "replace"
    assert result.accepted_action == InspectImageAction()
    assert result.private_reason == "Need chart evidence first."
    serialized = json.dumps(submitted)
    assert "Require visual evidence before answering." in serialized
    assert "synthetic-image" not in serialized


def test_hstar_analyst_receives_discovery_only_and_returns_bounded_rules() -> None:
    submitted: dict[str, Any] = {}
    discovery_ids = tuple(f"d{index}" for index in range(8))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response(model_id="fixture/hstar")
        submitted.update(json.loads(request.content))
        return _completion_response(
            content=json.dumps(
                {
                    "proposal_id": "hstar-1",
                    "parent_revision": 0,
                    "revision": 1,
                    "rules": [
                        {
                            "rule_id": "verify-labels",
                            "source_component": "skills",
                            "action_types": ["inspect_image", "final_answer"],
                            "instruction": "Verify chart labels before answering.",
                        }
                    ],
                    "discovery_task_ids": discovery_ids,
                }
            ),
            model_id="fixture/hstar",
        )

    analysis_request = HStarAnalysisRequest(
        manifest_hash="m" * 64,
        discovery_task_ids=discovery_ids,
        observations=tuple(
            DiscoveryObservation(
                task_id=task_id,
                question="What is shown?",
                correct=False,
                termination_reason="completed",
                action_types=("final_answer",),
            )
            for task_id in discovery_ids
        ),
    )
    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=5000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        result = CrusoeHStarAnalyst(
            client=client,
            model_id="fixture/hstar",
            max_output_tokens=256,
        ).propose(analysis_request, seed=17)

    assert result.proposal.discovery_task_ids == discovery_ids
    assert result.proposal.rules[0].rule_id == "verify-labels"
    serialized = json.dumps(submitted)
    assert "What is shown?" in serialized
    assert '"regression"' not in serialized


def test_live_gate_rejects_a_reviewer_that_passes_the_unsupported_answer() -> None:
    result = ReviewResult(
        decision="pass",
        accepted_action=FinalAnswerAction(answer="unsupported capability probe"),
        private_reason="Looks fine.",
        model_id="nvidia/fixture-nemotron-vl",
        request_id="review-fixture",
        input_tokens=10,
        output_tokens=10,
        latency_ms=1,
    )

    with pytest.raises(ValueError, match="required REPLACE"):
        _validate_reviewer_probe(result)
