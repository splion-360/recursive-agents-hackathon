"""Behavior tests for the ChartQA verifier boundary."""

import pytest
from pydantic import ValidationError

from memory_distillation.benchmark import ScoringReference, TaskInput, score_chartqa


@pytest.mark.parametrize(
    ("prediction", "answer", "expected"),
    [
        ("104.9", "100", True),
        ("105.1", "100", False),
        ("50%", "0.5", True),
        ("0", "0", True),
        ("0.01", "0", False),
        ("  YES  ", "yes", True),
        ("New   York", "new york", True),
        ("blue", "green", False),
    ],
)
def test_score_chartqa_uses_relaxed_numeric_or_normalized_exact_matching(
    prediction: str,
    answer: str,
    expected: bool,
) -> None:
    """ChartQA accepts numeric answers within 5% and exact normalized text."""

    reference = ScoringReference(task_id="chartqa:test:001", answer=answer)

    assert score_chartqa(prediction, reference).correct is expected


def test_task_input_is_answer_free_and_rejects_reference_fields() -> None:
    """Agent-visible task serialization cannot carry verifier-only labels."""

    task = TaskInput(
        task_id="chartqa:test:001",
        question="What value is shown?",
        image_asset_id="chartqa-image-001",
        image_sha256="a" * 64,
        source_split="test",
    )

    assert set(task.model_dump()) == {
        "schema_version",
        "task_id",
        "question",
        "image_asset_id",
        "image_sha256",
        "source_split",
    }
    with pytest.raises(ValidationError):
        TaskInput.model_validate({**task.model_dump(), "answer": "42"})
