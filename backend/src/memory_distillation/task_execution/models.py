"""Typed actions, policies, events, and results for one Student run."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_distillation.benchmark import Score


class InspectImageAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["inspect_image"] = "inspect_image"


class CropAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["crop"] = "crop"
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int = Field(gt=0)
    y1: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_region(self) -> "CropAction":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("crop must have positive width and height")
        return self


class RunPythonAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["run_python"] = "run_python"
    code: str = Field(min_length=1, max_length=10_000)


class FinalAnswerAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["final_answer"] = "final_answer"
    answer: str = Field(min_length=1, max_length=2_000)


StudentAction = Annotated[
    InspectImageAction | CropAction | RunPythonAction | FinalAnswerAction,
    Field(discriminator="type"),
]


class ExecutionPolicy(BaseModel):
    """Frozen Minimal Harness budgets for one task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_actions: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=8, ge=0)
    max_python_calls: int = Field(default=2, ge=0)
    max_total_tokens: int = Field(default=8_000, ge=1)
    max_elapsed_ms: int = Field(default=120_000, ge=1)
    sandbox_timeout_ms: int = Field(default=10_000, ge=1)
    max_crop_coordinate: int = Field(default=8_192, ge=1)


class Observation(BaseModel):
    """Deterministic public result returned to the next Student turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: Literal["inspect_image", "crop", "run_python"]
    content: str


class RunStartedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["run_started"] = "run_started"
    sequence: int
    task_id: str
    seed: int


class ActionProposedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["action_proposed"] = "action_proposed"
    sequence: int
    task_id: str
    action: StudentAction
    model_id: str
    request_id: str


class ObservationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["observation"] = "observation"
    sequence: int
    task_id: str
    observation: Observation


TerminationReason = Literal[
    "completed",
    "action_budget_exhausted",
    "tool_budget_exhausted",
    "python_budget_exhausted",
    "token_budget_exhausted",
    "time_budget_exhausted",
    "inference_failed",
    "sandbox_failed",
    "invalid_crop",
]


class RunCompletedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["run_completed"] = "run_completed"
    sequence: int
    task_id: str
    termination_reason: TerminationReason
    correct: bool | None


RunEvent = Annotated[
    RunStartedEvent | ActionProposedEvent | ObservationEvent | RunCompletedEvent,
    Field(discriminator="type"),
]


class TaskRun(BaseModel):
    """Complete deterministic evidence for one offline Student trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["task-run.v1"] = "task-run.v1"
    task_id: str
    seed: int
    termination_reason: TerminationReason
    actions: tuple[StudentAction, ...]
    observations: tuple[Observation, ...]
    events: tuple[RunEvent, ...]
    score: Score | None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    python_calls: int = Field(ge=0)
