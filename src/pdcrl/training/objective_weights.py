"""Objective-weight schedules for offline process-driven training."""

from __future__ import annotations

from collections.abc import Sequence


Weights = tuple[float, float, float, float]


def _weights(values: Sequence[float]) -> Weights:
    if len(values) != 4:
        raise ValueError(f"every objective anchor must contain four values, got {len(values)}")
    result = tuple(float(value) for value in values)
    if any(value < 0.0 for value in result):
        raise ValueError(f"objective weights must be non-negative, got {result!r}")
    return result  # type: ignore[return-value]


def continuous_stage_weights(
    anchors: Sequence[Sequence[float]],
    stage_index: int,
    elapsed_seconds: float,
    blend_seconds: float,
) -> Weights:
    """Linearly move between adjacent anchors during the existing pool crossfade."""
    normalized = tuple(_weights(anchor) for anchor in anchors)
    if not normalized:
        raise ValueError("at least one objective anchor is required")
    if stage_index < 0 or stage_index >= len(normalized):
        raise ValueError(
            f"stage_index must be in [0, {len(normalized) - 1}], got {stage_index}"
        )
    if blend_seconds <= 0.0:
        raise ValueError(f"blend_seconds must be positive, got {blend_seconds}")
    if stage_index == 0:
        return normalized[0]
    fraction = min(1.0, max(0.0, float(elapsed_seconds) / float(blend_seconds)))
    return tuple(
        left + fraction * (right - left)
        for left, right in zip(normalized[stage_index - 1], normalized[stage_index])
    )  # type: ignore[return-value]

