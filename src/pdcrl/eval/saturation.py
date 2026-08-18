"""Shared quality-saturation state and seeded proxy-start selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SaturationConfig:
    """Method-independent stagnation semantics for a minimization objective."""

    rel_tol: float = 0.001
    patience: int = 6
    floor_patience: int = 10
    lr_initial: float = 3e-4
    lr_floor: float = 1e-5
    lr_reduction_factor: float = 0.5
    divergence_guard_factor: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.rel_tol < 1.0:
            raise ValueError("rel_tol must be in [0, 1)")
        if self.patience <= 0 or self.floor_patience <= 0:
            raise ValueError("patience values must be positive")
        if not 0.0 < self.lr_floor <= self.lr_initial:
            raise ValueError("learning rates must satisfy 0 < lr_floor <= lr_initial")
        if not 0.0 < self.lr_reduction_factor < 1.0:
            raise ValueError("lr_reduction_factor must be in (0, 1)")
        if self.divergence_guard_factor <= 1.0:
            raise ValueError("divergence_guard_factor must be greater than 1")


@dataclass
class SaturationState:
    incumbent: float = math.inf
    learning_rate: float = math.nan
    stale_observations: int = 0
    floor_observations: int = 0
    observations: int = 0
    lr_reductions: int = 0
    saturated: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class SaturationDecision:
    event: str
    terminal: bool
    improved: bool
    learning_rate: float
    incumbent: float


class SaturationTracker:
    """Track qualifying improvements and LR-floor saturation confirmations."""

    def __init__(self, config: SaturationConfig, state: SaturationState | None = None):
        self.config = config
        self.state = state or SaturationState(learning_rate=config.lr_initial)

    def _decision(self, event: str, *, improved: bool = False) -> SaturationDecision:
        return SaturationDecision(
            event=event,
            terminal=self.state.stop_reason is not None,
            improved=improved,
            learning_rate=self.state.learning_rate,
            incumbent=self.state.incumbent,
        )

    def observe(self, candidate: float) -> SaturationDecision:
        """Consume one confirmed minimization result and return the required action."""
        if self.state.stop_reason is not None:
            raise RuntimeError(f"tracker already stopped: {self.state.stop_reason}")
        self.state.observations += 1
        if not math.isfinite(candidate):
            self.state.stop_reason = "non_finite"
            return self._decision("non_finite")

        incumbent = self.state.incumbent
        if math.isfinite(incumbent) and candidate > (
            incumbent * self.config.divergence_guard_factor
        ):
            return self._decision("divergence")

        threshold = incumbent * (1.0 - self.config.rel_tol)
        if not math.isfinite(incumbent) or candidate < threshold:
            self.state.incumbent = candidate
            self.state.stale_observations = 0
            self.state.floor_observations = 0
            return self._decision("improved", improved=True)

        at_floor = self.state.learning_rate <= self.config.lr_floor * (1.0 + 1e-12)
        if at_floor:
            self.state.floor_observations += 1
            if self.state.floor_observations >= self.config.floor_patience:
                self.state.saturated = True
                self.state.stop_reason = "saturated"
                return self._decision("saturated")
            return self._decision("stale")

        self.state.stale_observations += 1
        if self.state.stale_observations < self.config.patience:
            return self._decision("stale")

        self.state.learning_rate = max(
            self.config.lr_floor,
            self.state.learning_rate * self.config.lr_reduction_factor,
        )
        self.state.stale_observations = 0
        self.state.floor_observations = 0
        self.state.lr_reductions += 1
        return self._decision("reduce_lr")

    def stop_abnormally(self, reason: str) -> SaturationDecision:
        """Terminate a run without marking quality saturation."""
        if self.state.stop_reason is not None:
            raise RuntimeError(f"tracker already stopped: {self.state.stop_reason}")
        if reason == "saturated":
            raise ValueError("use observe() to reach normal saturation")
        self.state.stop_reason = reason
        self.state.saturated = False
        return self._decision(reason)

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state)

    @classmethod
    def from_state_dict(
        cls,
        config: SaturationConfig,
        values: dict[str, Any],
    ) -> "SaturationTracker":
        return cls(config, SaturationState(**values))


def _quantile_bins(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    boundaries = np.unique(np.quantile(values, quantiles))
    return np.digitize(values, boundaries, right=True)


def select_proxy_starts(instance, count: int, seed: int) -> list[int]:
    """Select a deterministic width/priority-stratified subset of start slabs."""
    n = int(instance.num_slabs)
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= n:
        return list(range(n))

    widths = np.asarray([slab.width_mm for slab in instance.slabs], dtype=np.float64)
    priorities = np.asarray([slab.priority for slab in instance.slabs], dtype=np.float64)
    width_bin = _quantile_bins(widths, (0.25, 0.5, 0.75))
    priority_bin = _quantile_bins(priorities, (0.5,))
    rng = np.random.default_rng(seed)

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(width_bin.tolist(), priority_bin.tolist())):
        buckets.setdefault(key, []).append(index)
    for members in buckets.values():
        rng.shuffle(members)

    width_order = sorted(set(width_bin.tolist()))
    priority_order = sorted(set(priority_bin.tolist()))
    rng.shuffle(width_order)
    rng.shuffle(priority_order)
    selected: list[int] = []
    round_index = 0
    while len(selected) < count:
        before = len(selected)
        for width_index, width_group in enumerate(width_order):
            priority_offset = (round_index + width_index) % len(priority_order)
            choices = priority_order[priority_offset:] + priority_order[:priority_offset]
            for priority_group in choices:
                members = buckets.get((width_group, priority_group), [])
                if members:
                    selected.append(members.pop())
                    break
            if len(selected) == count:
                break
        if len(selected) == before:
            break
        round_index += 1

    if len(selected) < count:
        remaining = [index for index in rng.permutation(n).tolist() if index not in selected]
        selected.extend(remaining[: count - len(selected)])
    return sorted(selected)
