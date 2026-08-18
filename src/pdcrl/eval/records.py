"""Validated, atomic records for quality-saturation benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from pdcrl.rewards.costs import evaluate_full, full_weights


SCHEMA_VERSION = "saturation-run-v1"
_COMPONENTS = ("transition_cost", "unscheduled_cost", "energy_cost", "rollchange_cost")


@dataclass(frozen=True)
class RunIdentity:
    protocol_version: str
    method: str
    instance: str
    method_seed: int
    weights: tuple[float, float, float, float]
    scale: int | None = None
    batch_replicate: int | None = None
    data_seed: int | None = None

    def key(self) -> tuple:
        return (
            self.protocol_version,
            self.method,
            self.instance,
            self.method_seed,
            tuple(float(value) for value in self.weights),
            self.scale,
            self.batch_replicate,
            self.data_seed,
        )


@dataclass(frozen=True)
class RuntimeBreakdown:
    search_seconds: float
    decode_seconds: float
    local_search_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.search_seconds + self.decode_seconds + self.local_search_seconds


@dataclass(frozen=True)
class ArtifactPaths:
    schedule: str
    curve: str
    config: str
    checkpoint: str | None = None


@dataclass(frozen=True)
class RunRecord:
    identity: RunIdentity
    manifest_sha256: str
    config_sha256: str
    stop_reason: str
    saturated: bool
    raw_metrics: dict[str, Any]
    final_metrics: dict[str, Any]
    runtime: RuntimeBreakdown
    counts: dict[str, int]
    diagnostics: dict[str, Any]
    artifacts: ArtifactPaths

    def to_dict(self) -> dict[str, Any]:
        identity = asdict(self.identity)
        identity["weights"] = list(self.identity.weights)
        runtime = asdict(self.runtime)
        runtime["total_seconds"] = self.runtime.total_seconds
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
            "stop_reason": self.stop_reason,
            "saturated": self.saturated,
            "raw_metrics": self.raw_metrics,
            "final_metrics": self.final_metrics,
            "runtime": runtime,
            "counts": self.counts,
            "diagnostics": self.diagnostics,
            "artifacts": asdict(self.artifacts),
        }


def _transition_subcomponents(schedule, instance, profile) -> tuple[float, float, float]:
    width_cost = 0.0
    gauge_cost = 0.0
    hardness_cost = 0.0
    for unit in schedule.units:
        for left_index, right_index in zip(unit, unit[1:]):
            left = instance.slabs[left_index]
            right = instance.slabs[right_index]
            if right.width_mm > left.width_mm:
                width_cost += profile.width_up_penalty * (right.width_mm - left.width_mm)
            else:
                width_cost += profile.width_down_penalty * max(
                    0.0,
                    left.width_mm - right.width_mm - profile.width_step_allow_mm,
                )
            gauge_cost += profile.gauge_penalty * max(
                0.0,
                abs(left.gauge_mm - right.gauge_mm) - profile.gauge_tol_mm,
            )
            hardness_cost += profile.hardness_penalty * max(
                0.0,
                abs(left.hardness - right.hardness) - profile.hardness_tol,
            )
    return width_cost, gauge_cost, hardness_cost


def build_schedule_metrics(schedule, instance, profile, weights=(1, 1, 1, 1)) -> dict[str, Any]:
    """Recompute objective components and operational KPIs from one schedule."""
    base = evaluate_full(schedule, instance, profile)
    objective_weights = full_weights(weights)
    components = np.asarray([base[name] for name in _COMPONENTS], dtype=np.float64)
    width_cost, gauge_cost, hardness_cost = _transition_subcomponents(
        schedule, instance, profile
    )

    scheduled = schedule.scheduled_ids()
    total_priority = sum(slab.priority for slab in instance.slabs)
    scheduled_priority = sum(instance.slabs[index].priority for index in scheduled)
    power_changes = [
        abs(
            instance.slabs[left].power_per_ton
            - instance.slabs[right].power_per_ton
        )
        for unit in schedule.units
        for left, right in zip(unit, unit[1:])
    ]
    campaign_lengths = [
        sum(instance.slabs[index].rolling_length_m for index in unit)
        for unit in schedule.units
        if unit
    ]
    num_units = len(campaign_lengths)
    capacity_utilization = (
        sum(campaign_lengths) / (num_units * profile.capacity_max_m) if num_units else 0.0
    )

    return {
        "objective_value": float(np.dot(objective_weights, components)),
        "transition_cost": float(base["transition_cost"]),
        "unscheduled_cost": float(base["unscheduled_cost"]),
        "energy_cost": float(base["energy_cost"]),
        "rollchange_cost": float(base["rollchange_cost"]),
        "width_cost": float(width_cost),
        "gauge_cost": float(gauge_cost),
        "hardness_cost": float(hardness_cost),
        "feasible": bool(base["feasible"]),
        "scheduled_slab_ratio": len(scheduled) / max(1, instance.num_slabs),
        "scheduled_priority_ratio": scheduled_priority / max(1, total_priority),
        "adjacent_power_change_mean": float(np.mean(power_changes)) if power_changes else 0.0,
        "adjacent_power_change_p95": float(np.percentile(power_changes, 95))
        if power_changes
        else 0.0,
        "rolling_unit_count": num_units,
        "mean_campaign_slabs": float(np.mean([len(unit) for unit in schedule.units if unit]))
        if num_units
        else 0.0,
        "mean_campaign_length_m": float(np.mean(campaign_lengths)) if num_units else 0.0,
        "capacity_utilization": float(capacity_utilization),
        "unscheduled_slab_count": len(schedule.unscheduled),
    }


def _payload(record: RunRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_dict() if isinstance(record, RunRecord) else record


def validate_run_record(record: RunRecord | dict[str, Any]) -> None:
    """Raise ValueError with all schema and consistency problems."""
    payload = _payload(record)
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    identity = payload.get("identity", {})
    weights = identity.get("weights", [])
    if len(weights) != 4:
        errors.append("identity.weights must contain four values")
    for name in ("scale", "batch_replicate", "data_seed"):
        value = identity.get(name)
        if value is not None and int(value) < 0:
            errors.append(f"identity.{name} must be non-negative when present")

    for phase in ("raw_metrics", "final_metrics"):
        metrics = payload.get(phase, {})
        for name in (*_COMPONENTS, "objective_value", "feasible"):
            if name not in metrics:
                errors.append(f"{phase} missing {name}")
        if len(weights) == 4 and all(name in metrics for name in _COMPONENTS):
            expected = sum(float(weight) * float(metrics[name]) for weight, name in zip(weights, _COMPONENTS))
            actual = float(metrics.get("objective_value", math.nan))
            tolerance = max(1e-6, abs(expected) * 1e-9)
            if not math.isfinite(actual) or abs(actual - expected) > tolerance:
                errors.append(
                    f"{phase} objective_value does not equal weighted component sum "
                    f"({actual} != {expected})"
                )

    runtime = payload.get("runtime", {})
    for name in ("search_seconds", "decode_seconds", "local_search_seconds"):
        if float(runtime.get(name, -1.0)) < 0.0:
            errors.append(f"runtime.{name} must be non-negative")

    for name, value in payload.get("artifacts", {}).items():
        if value is None:
            continue
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"artifact {name} must be a relative path")
    if errors:
        raise ValueError("invalid run record:\n- " + "\n- ".join(errors))


def write_run_record_atomic(record: RunRecord, run_dir: str | Path) -> Path:
    """Validate and atomically replace one run's record.json."""
    validate_run_record(record)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=run_dir,
        prefix=".record.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, run_dir / "record.json")
    return run_dir / "record.json"


def _identity_key(payload: dict[str, Any]) -> tuple:
    identity = payload["identity"]
    return (
        identity["protocol_version"],
        identity["method"],
        identity["instance"],
        int(identity["method_seed"]),
        tuple(float(value) for value in identity["weights"]),
    )


def merge_run_records(root: str | Path, output_path: str | Path) -> Path:
    """Validate and atomically merge per-run records without concurrent appends."""
    root = Path(root)
    output_path = Path(output_path)
    records: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for path in sorted(root.rglob("record.json")):
        payload = json.loads(path.read_text())
        validate_run_record(payload)
        key = _identity_key(payload)
        if key in seen:
            raise ValueError(f"duplicate run identity: {key!r}")
        seen.add(key)
        records.append(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, output_path)
    return output_path
