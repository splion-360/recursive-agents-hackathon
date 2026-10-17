"""ChartQA scoring contracts independent of dataset and provider adapters."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RELAXED_NUMERIC_TOLERANCE = Decimal("0.05")


@dataclass(frozen=True, slots=True)
class TaskAsset:
    """One current chart image, detached from the benchmark directory tree."""

    task_id: str
    media_type: Literal["image/png"]
    image_sha256: str
    content: bytes


class TaskInput(BaseModel):
    """Answer-free ChartQA input safe for agent-visible boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["chartqa-task.v1"] = "chartqa-task.v1"
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    image_asset_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_split: Literal["train", "val", "test"]


class ScoringReference(BaseModel):
    """Verifier-only answer for one task; never serialize into agent context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    answer: str


class Score(BaseModel):
    """Answer-free verifier result safe to expose to evaluation code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correct: bool
    method: Literal["numeric_relaxed", "text_exact"]


def score_chartqa(prediction: str, reference: ScoringReference) -> Score:
    """Apply ChartQA relaxed accuracy with a 5% numeric tolerance."""

    prediction_number = _to_decimal(prediction)
    reference_number = _to_decimal(reference.answer)

    if prediction_number is not None and reference_number is not None:
        tolerance = abs(reference_number) * RELAXED_NUMERIC_TOLERANCE
        return Score(
            correct=abs(prediction_number - reference_number) <= tolerance,
            method="numeric_relaxed",
        )

    return Score(
        correct=prediction.casefold() == reference.answer.casefold(),
        method="text_exact",
    )


def _to_decimal(value: str) -> Decimal | None:
    is_percent = value.endswith("%")
    numeric_value = value.rstrip("%") if is_percent else value
    try:
        parsed = Decimal(numeric_value)
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return None
    return parsed / Decimal(100) if is_percent else parsed
