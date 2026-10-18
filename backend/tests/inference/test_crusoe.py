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
        budget=CrusoeBudget(max_total_tokens=1000),
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


def test_multimodal_payload_bytes_are_not_counted_as_tokens() -> None:
    submitted = False

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal submitted
        if request.url.path.endswith("/models"):
            response = _catalog_response()
            payload = response.json()
            payload["data"][0]["context_length"] = 262_144
            return httpx2.Response(200, json=payload)
        submitted = True
        return _completion_response()

    large_image = "A" * 256_296
    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_total_tokens=10_000_000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        client.complete_json(
            model_id="fixture/model",
            messages=(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the chart."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{large_image}"},
                        },
                    ],
                },
            ),
            output_type=FixtureOutput,
            max_output_tokens=16_384,
            seed=17,
        )

    assert submitted is True


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


def test_invalid_structured_output_is_retried_and_fully_accounted() -> None:
    attempts = 0
    seeds: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        seeds.append(json.loads(request.content)["seed"])
        if attempts == 1:
            return _completion_response(content='{"status":')
        return _completion_response()

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
        max_retries=1,
        retry_backoff_seconds=0,
    ) as client:
        result = client.complete_json(
            model_id="fixture/model",
            messages=({"role": "user", "content": "Return ok."},),
            output_type=FixtureOutput,
            max_output_tokens=16,
            seed=17,
        )

    assert attempts == 2
    assert seeds == [17, 18]
    assert result.output == FixtureOutput(status="ok")
    assert result.retry_count == 1
    assert result.input_tokens == 40
    assert result.output_tokens == 10
    assert result.actual_cost_usd == Decimal("0.000030")
    assert client.spent_tokens == 50
    assert client.spent_cost_usd == Decimal("0.000030")


def test_invalid_structured_output_retries_are_bounded() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        return _completion_response(content='{"status":')

    with (
        CrusoeClient(
            base_url="https://inference.example.test/v1",
            api_key=SecretStr("fixture-secret"),
            budget=CrusoeBudget(max_total_tokens=1000),
            transport=httpx2.MockTransport(handler),
            max_retries=1,
            retry_backoff_seconds=0,
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

    assert attempts == 2
    assert captured.value.failure_kind == "invalid_structured_output"
    assert captured.value.retry_count == 1
    assert captured.value.validation_summary == ("<root>:json_invalid",)
    assert client.spent_tokens == 50


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


def test_response_read_timeout_is_retried_for_pure_inference() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        if attempts == 1:
            raise httpx2.ReadTimeout("response outcome unknown", request=request)
        return _completion_response()

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
        max_retries=1,
        retry_backoff_seconds=0,
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


def test_partial_write_failure_prevents_further_requests() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        raise httpx2.WriteError("request outcome unknown", request=request)

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


def test_transient_provider_unavailable_is_retried() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        if request.url.path.endswith("/models"):
            return _catalog_response()
        attempts += 1
        if attempts == 1:
            return httpx2.Response(503, json={"errors": ["shared capacity unavailable"]})
        return _completion_response()

    with CrusoeClient(
        base_url="https://inference.example.test/v1",
        api_key=SecretStr("fixture-secret"),
        budget=CrusoeBudget(max_cost_usd=Decimal("0.01"), max_total_tokens=1000),
        transport=httpx2.MockTransport(handler),
        max_retries=1,
        retry_backoff_seconds=0,
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


def test_student_serializes_one_image_and_returns_a_typed_action() -> None:
    submitted: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response(model_id="fixture/student-vl", modality="multimodal")
        submitted.update(json.loads(request.content))
        return _completion_response(
            content=(
                '{"action_type":"final_answer","answer":"42","code":null,'
                '"x0":null,"y0":null,"x1":null,"y1":null}'
            ),
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
                '{"decision":"replace","accepted_action_type":"inspect_image",'
                '"answer":null,"code":null,"x0":null,"y0":null,"x1":null,"y1":null,'
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


def test_reviewer_rejects_direct_answer_to_direct_answer_replacement() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response(model_id="nvidia/fixture-nemotron-vl", modality="multimodal")
        return _completion_response(
            content=(
                '{"decision":"replace","accepted_action_type":"final_answer",'
                '"answer":"corrected","code":null,"x0":null,"y0":null,'
                '"x1":null,"y1":null,"private_reason":"Unsupported answer."}'
            ),
            model_id="nvidia/fixture-nemotron-vl",
        )

    task = TaskInput(
        task_id="chartqa:fixture:direct-answer",
        question="What is shown?",
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
        budget=CrusoeBudget(max_total_tokens=5000),
        transport=httpx2.MockTransport(handler),
    ) as client:
        reviewer = CrusoeReviewer(
            client=client,
            model_id="nvidia/fixture-nemotron-vl",
            max_output_tokens=128,
        )
        with pytest.raises(ValueError, match="evidence-gathering action"):
            reviewer.review(
                ReviewRequest(
                    task=task,
                    asset=asset,
                    proposed_action=FinalAnswerAction(answer="wrong"),
                    observations=(),
                    guidance=("Gather evidence before answering.",),
                    turn_index=0,
                )
            )


def test_hstar_analyst_binds_revision_lineage_from_the_trusted_request() -> None:
    submitted: dict[str, Any] = {}
    discovery_ids = tuple(f"d{index}" for index in range(8))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _catalog_response(model_id="fixture/hstar")
        submitted.update(json.loads(request.content))
        return _completion_response(
            content=json.dumps(
                {
                    "proposal_id": "hstar-2",
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
                    "discovery_task_ids": discovery_ids[:4],
                }
            ),
            model_id="fixture/hstar",
        )

    analysis_request = HStarAnalysisRequest(
        manifest_hash="m" * 64,
            round_index=2,
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
    assert result.proposal.parent_revision == 1
    assert result.proposal.revision == 2
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
