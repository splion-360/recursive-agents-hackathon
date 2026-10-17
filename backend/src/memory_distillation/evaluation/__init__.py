"""Read-only experiment evaluation."""

from memory_distillation.evaluation.orchestrator import (
    EvaluationTask,
    run_offline_conditions,
)
from memory_distillation.evaluation.report import (
    C2RetrievalTarget,
    C3Assignment,
    C3Semantics,
    ConditionMetrics,
    ConditionPolicy,
    ConditionResult,
    ExperimentReport,
    IrrelevantDonorCard,
    PairedContrast,
    assign_irrelevant_memory,
    build_experiment_report,
    frozen_condition_policies,
    validate_condition_policies,
)

__all__ = [
    "C2RetrievalTarget",
    "C3Assignment",
    "C3Semantics",
    "ConditionMetrics",
    "ConditionPolicy",
    "ConditionResult",
    "EvaluationTask",
    "ExperimentReport",
    "IrrelevantDonorCard",
    "PairedContrast",
    "assign_irrelevant_memory",
    "build_experiment_report",
    "frozen_condition_policies",
    "run_offline_conditions",
    "validate_condition_policies",
]
