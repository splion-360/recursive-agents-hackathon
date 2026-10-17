"""Typed inference seam and deterministic offline adapter."""

from memory_distillation.inference.protocol import (
    InferenceClient,
    InferenceRequest,
    InferenceResult,
)
from memory_distillation.inference.scripted import DeterministicInference

__all__ = [
    "DeterministicInference",
    "InferenceClient",
    "InferenceRequest",
    "InferenceResult",
]
