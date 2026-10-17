"""Isolated H-star/Kappa calibration lifecycle."""

from typing import TYPE_CHECKING, Any

from memory_distillation.calibration.models import (
    AdaptationDecision,
    CalibrationProgress,
    CalibrationProvenance,
    CalibrationReport,
    DiscoveryObservation,
    HStarAnalysisRequest,
    HStarAnalysisResult,
    HStarAnalyst,
    ModelIdentities,
    PilotIntervention,
    PilotSummary,
)

if TYPE_CHECKING:
    from memory_distillation.calibration.orchestrator import (
        calibrate_review_policy,
        combined_manifest_hash,
    )

__all__ = [
    "AdaptationDecision",
    "CalibrationProgress",
    "CalibrationProvenance",
    "CalibrationReport",
    "DiscoveryObservation",
    "HStarAnalysisRequest",
    "HStarAnalysisResult",
    "HStarAnalyst",
    "ModelIdentities",
    "PilotIntervention",
    "PilotSummary",
    "calibrate_review_policy",
    "combined_manifest_hash",
]


def __getattr__(name: str) -> Any:
    """Load orchestration lazily to keep provider imports cycle-free."""

    if name in {"calibrate_review_policy", "combined_manifest_hash"}:
        from memory_distillation.calibration import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(name)
