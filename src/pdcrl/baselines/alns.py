"""Problem-specific adaptive large-neighborhood search for HPCVRP schedules."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
import math
import time

import numpy as np

from pdcrl.baselines.heuristics import greedy_schedule
from pdcrl.baselines.nsga2 import split_permutation_dp
from pdcrl.data.loader import transition_matrices
from pdcrl.eval.metrics import hypervolume
from pdcrl.rewards.costs import evaluate_full, full_weights


DESTROY_OPERATORS = (
    "random_removal",
    "worst_transition",
    "related_width_gauge",
    "whole_unit",
    "priority_aware",
)
REPAIR_OPERATORS = (
    "random_insertion",
    "cheapest_insertion",
    "regret2_insertion",
    "unit_split_merge",
    "prize_aware_defer",
)


@dataclass(frozen=True)
class ALNSResult:
    best_cost: float
    best_schedule: object
    archive: np.ndarray
    archive_schedules: list
    stop_reason: str
    saturated: bool
    iterations: int
    segments: int
    curve: list[dict]
    operator_weights: dict[str, dict[str, float]]
    resume_state: dict


@dataclass(frozen=True)
class StagnationState:
    """Updated qualified incumbents and counter for one ALNS segment."""

    qualified_cost: float
    qualified_hypervolume: float
    stale: int
    event: str


def update_stagnation(
    *,
    best_cost: float,
    hypervolume: float,
    qualified_cost: float,
    qualified_hypervolume: float,
    stale: int,
    rel_tol: float,
    stop_metric: str,
) -> StagnationState:
    """Update ALNS stagnation without coupling scalar and Pareto stopping.

    ``cost_or_front`` preserves the historical behavior. ``scalar`` is used by
    the fixed-weight independent-batch benchmark: the Pareto archive is still
    tracked, but front expansion cannot prolong scalar optimization.
    """
    if stop_metric not in {"cost_or_front", "scalar"}:
        raise ValueError(
            "stop_metric must be either 'cost_or_front' or 'scalar', "
            f"got {stop_metric!r}"
        )
    cost_improved = best_cost < qualified_cost * (1.0 - rel_tol)
    front_improved = hypervolume > qualified_hypervolume * (1.0 + rel_tol)
    if cost_improved:
        qualified_cost = best_cost
    if front_improved:
        qualified_hypervolume = hypervolume
    improved = cost_improved or (
        stop_metric == "cost_or_front" and front_improved
    )
    return StagnationState(
        qualified_cost=float(qualified_cost),
        qualified_hypervolume=float(qualified_hypervolume),
        stale=0 if improved else int(stale) + 1,
        event="improved" if improved else "stale",
    )


def _weighted_cost(schedule, instance, profile, weights) -> float:
    score = evaluate_full(schedule, instance, profile)
    components = np.asarray(
        [
            score["transition_cost"],
            score["unscheduled_cost"],
            score["energy_cost"],
            score["rollchange_cost"],
        ]
    )
    return float(np.dot(full_weights(weights), components))


def _component_vector(schedule, instance, profile) -> np.ndarray:
    score = evaluate_full(schedule, instance, profile)
    return np.asarray(
        [
            score["transition_cost"],
            score["unscheduled_cost"],
            score["energy_cost"],
        ],
        dtype=np.float64,
    )


def _normalize_remove_count(n: int, n_remove: int) -> int:
    if n < 2:
        raise ValueError("ALNS requires at least two slabs")
    return min(n - 1, max(1, int(n_remove)))


def destroy_permutation(
    permutation,
    schedule,
    instance,
    profile,
    *,
    operator: str,
    n_remove: int,
    rng: np.random.Generator,
    D=None,
    PE=None,
) -> tuple[list[int], list[int]]:
    """Remove exactly ``n_remove`` slab ids while preserving survivor order."""
    if operator not in DESTROY_OPERATORS:
        raise ValueError(f"unknown destroy operator: {operator!r}")
    permutation = [int(value) for value in permutation]
    n_remove = _normalize_remove_count(len(permutation), n_remove)
    all_ids = set(permutation)

    if operator == "random_removal":
        removed = set(rng.choice(permutation, size=n_remove, replace=False).tolist())
    elif operator == "worst_transition":
        if D is None or PE is None:
            D, PE = transition_matrices(instance, profile)
        contribution = {slab: 0.0 for slab in permutation}
        for unit in schedule.units:
            for left, right in zip(unit, unit[1:]):
                value = float(D[left, right] + PE[left, right])
                contribution[left] += value / 2.0
                contribution[right] += value / 2.0
        ranked = sorted(permutation, key=lambda slab: (-contribution[slab], slab))
        removed = set(ranked[:n_remove])
    elif operator == "related_width_gauge":
        pivot = instance.slabs[int(rng.choice(permutation))]
        width_scale = max(1.0, profile.width_max_mm - profile.width_min_mm)
        gauge_scale = max(1.0, profile.gauge_max_mm - profile.gauge_min_mm)
        ranked = sorted(
            permutation,
            key=lambda slab: (
                abs(instance.slabs[slab].width_mm - pivot.width_mm) / width_scale
                + abs(instance.slabs[slab].gauge_mm - pivot.gauge_mm) / gauge_scale,
                slab,
            ),
        )
        removed = set(ranked[:n_remove])
    elif operator == "whole_unit":
        units = [unit for unit in schedule.units if unit]
        seed_unit = list(units[int(rng.integers(len(units)))]) if units else []
        rng.shuffle(seed_unit)
        chosen = seed_unit[:n_remove]
        if len(chosen) < n_remove:
            remaining = list(all_ids - set(chosen))
            rng.shuffle(remaining)
            chosen.extend(remaining[: n_remove - len(chosen)])
        removed = set(chosen)
    else:  # priority_aware
        ranked = sorted(
            permutation,
            key=lambda slab: (instance.slabs[slab].priority, rng.random()),
        )
        removed = set(ranked[:n_remove])

    kept = [slab for slab in permutation if slab not in removed]
    removed_ordered = [slab for slab in permutation if slab in removed]
    return kept, removed_ordered


def _insertion_delta(sequence: list[int], position: int, slab: int, C: np.ndarray) -> float:
    delta = 0.0
    if position > 0:
        delta += C[sequence[position - 1], slab]
    if position < len(sequence):
        delta += C[slab, sequence[position]]
    if 0 < position < len(sequence):
        delta -= C[sequence[position - 1], sequence[position]]
    return float(delta)


def _best_positions(sequence: list[int], slab: int, C: np.ndarray) -> list[tuple[float, int]]:
    deltas = _insertion_deltas(sequence, slab, C)
    order = np.lexsort((np.arange(len(deltas)), deltas))
    return [(float(deltas[position]), int(position)) for position in order]


def _insertion_deltas(sequence: list[int], slab: int, C: np.ndarray) -> np.ndarray:
    if not sequence:
        return np.asarray([0.0])
    values = np.asarray(sequence, dtype=int)
    deltas = np.empty(len(values) + 1, dtype=np.float64)
    deltas[0] = C[slab, values[0]]
    deltas[-1] = C[values[-1], slab]
    if len(values) > 1:
        deltas[1:-1] = (
            C[values[:-1], slab]
            + C[slab, values[1:]]
            - C[values[:-1], values[1:]]
        )
    return deltas


def _best_position(sequence: list[int], slab: int, C: np.ndarray) -> tuple[float, int]:
    deltas = _insertion_deltas(sequence, slab, C)
    position = int(np.lexsort((np.arange(len(deltas)), deltas))[0])
    return float(deltas[position]), position


def _two_best_positions(
    sequence: list[int], slab: int, C: np.ndarray
) -> list[tuple[float, int]]:
    deltas = _insertion_deltas(sequence, slab, C)
    order = np.lexsort((np.arange(len(deltas)), deltas))[:2]
    return [(float(deltas[position]), int(position)) for position in order]


def repair_permutation(
    kept,
    removed,
    instance,
    profile,
    *,
    weights,
    operator: str,
    rng: np.random.Generator,
    D=None,
    PE=None,
) -> np.ndarray:
    """Reinsert all removed ids using a problem-specific ordering rule."""
    if operator not in REPAIR_OPERATORS:
        raise ValueError(f"unknown repair operator: {operator!r}")
    if D is None or PE is None:
        D, PE = transition_matrices(instance, profile)
    weights_array = full_weights(weights)
    C = weights_array[0] * D + weights_array[2] * PE
    sequence = [int(value) for value in kept]
    pending = [int(value) for value in removed]

    if operator == "random_insertion":
        rng.shuffle(pending)
        for slab in pending:
            sequence.insert(int(rng.integers(len(sequence) + 1)), slab)
    elif operator == "cheapest_insertion":
        pending.sort(key=lambda slab: (-instance.slabs[slab].priority, slab))
        for slab in pending:
            _, position = _best_position(sequence, slab, C)
            sequence.insert(position, slab)
    elif operator == "regret2_insertion":
        while pending:
            choices = []
            for slab in pending:
                positions = _two_best_positions(sequence, slab, C)
                regret = positions[min(1, len(positions) - 1)][0] - positions[0][0]
                choices.append((regret, instance.slabs[slab].priority, -slab, slab, positions[0][1]))
            *_, slab, position = max(choices)
            sequence.insert(position, slab)
            pending.remove(slab)
    elif operator == "unit_split_merge":
        pending.sort(
            key=lambda slab: (
                -instance.slabs[slab].width_mm,
                instance.slabs[slab].gauge_mm,
                slab,
            )
        )
        for slab in pending:
            _, position = _best_position(sequence, slab, C)
            sequence.insert(position, slab)
    else:  # prize_aware_defer
        pending.sort(key=lambda slab: (-instance.slabs[slab].priority, slab))
        priority_cut = float(np.median([instance.slabs[slab].priority for slab in pending]))
        for slab in pending:
            if instance.slabs[slab].priority < priority_cut:
                sequence.append(slab)
            else:
                _, position = _best_position(sequence, slab, C)
                sequence.insert(position, slab)
    return np.asarray(sequence, dtype=int)


def accept_candidate(
    candidate_cost: float,
    current_cost: float,
    *,
    temperature: float,
    random_value: float,
) -> bool:
    if candidate_cost <= current_cost:
        return True
    if temperature <= 0.0:
        return False
    probability = math.exp(-(candidate_cost - current_cost) / temperature)
    return random_value < probability


def _archive_add(archive: list[dict], vector: np.ndarray, schedule, permutation) -> bool:
    for entry in archive:
        existing = entry["vector"]
        if np.array_equal(existing, vector):
            return False
        if np.all(existing <= vector) and np.any(existing < vector):
            return False
    archive[:] = [
        entry
        for entry in archive
        if not (np.all(vector <= entry["vector"]) and np.any(vector < entry["vector"]))
    ]
    archive.append(
        {"vector": vector.copy(), "schedule": schedule, "permutation": permutation.copy()}
    )
    return True


def _roulette(weights: dict[str, float], rng: np.random.Generator) -> str:
    names = list(weights)
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=values / values.sum()))]


def _archive_hv(archive: list[dict], ideal, scale, reference) -> float:
    points = np.asarray([entry["vector"] for entry in archive], dtype=np.float64)
    normalized = (points - ideal) / scale
    eligible = normalized[np.all(normalized < reference, axis=1)]
    return hypervolume(eligible, reference) if len(eligible) else 0.0


def alns_saturated(
    instance,
    profile,
    *,
    iterations_per_segment: int = 1000,
    stale_segments: int = 10,
    rel_tol: float = 0.001,
    seed: int = 0,
    weights=(1, 1, 1, 1),
    destroy_fraction=(0.1, 0.3),
    reaction_factor: float = 0.2,
    cooling_rate: float = 0.9995,
    initial_temperature_fraction: float = 0.02,
    max_segments: int = 1000,
    stop_metric: str = "cost_or_front",
    max_seconds: float | None = None,
    resume_state: dict | None = None,
    segment_callback: Callable[[dict], None] | None = None,
) -> ALNSResult:
    """Run adaptive destroy/repair segments until incumbent and archive stagnate."""
    config_key = (
        iterations_per_segment,
        stale_segments,
        rel_tol,
        seed,
        tuple(weights),
        tuple(destroy_fraction),
        reaction_factor,
        cooling_rate,
        initial_temperature_fraction,
        stop_metric,
        None if max_seconds is None else float(max_seconds),
    )
    started = time.perf_counter()
    D, PE = transition_matrices(instance, profile)
    if resume_state is None:
        rng = np.random.default_rng(seed)
        greedy = greedy_schedule(instance, profile, weights=weights)
        permutation = np.asarray(
            [slab for unit in greedy.units for slab in unit] + sorted(greedy.unscheduled),
            dtype=int,
        )
        current_schedule = split_permutation_dp(
            permutation, instance, profile, weights=weights, D=D, PE=PE
        )
        current_cost = _weighted_cost(current_schedule, instance, profile, weights)
        best_cost = current_cost
        best_schedule = current_schedule
        best_permutation = permutation.copy()
        archive: list[dict] = []
        initial_vector = _component_vector(current_schedule, instance, profile)
        _archive_add(archive, initial_vector, current_schedule, permutation)
        ideal = initial_vector.copy()
        scale = np.maximum(np.abs(initial_vector), 1.0)
        reference = np.full(3, 1.1, dtype=np.float64)
        qualified_cost = best_cost
        qualified_hv = _archive_hv(archive, ideal, scale, reference)
        temperature = max(1e-9, current_cost * initial_temperature_fraction)
        destroy_weights = {name: 1.0 for name in DESTROY_OPERATORS}
        repair_weights = {name: 1.0 for name in REPAIR_OPERATORS}
        iterations = segments = stale = 0
        curve: list[dict] = []
        prior_elapsed_seconds = 0.0
    else:
        if tuple(resume_state["config_key"]) != config_key:
            raise ValueError("resume_state was created with a different ALNS configuration")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(resume_state["rng_state"])
        permutation = resume_state["permutation"].copy()
        current_schedule = resume_state["current_schedule"]
        current_cost = float(resume_state["current_cost"])
        best_cost = float(resume_state["best_cost"])
        best_schedule = resume_state["best_schedule"]
        best_permutation = resume_state["best_permutation"].copy()
        archive = list(resume_state["archive"])
        ideal = resume_state["ideal"].copy()
        scale = resume_state["scale"].copy()
        reference = resume_state["reference"].copy()
        qualified_cost = float(resume_state["qualified_cost"])
        qualified_hv = float(resume_state["qualified_hv"])
        temperature = float(resume_state["temperature"])
        destroy_weights = dict(resume_state["destroy_weights"])
        repair_weights = dict(resume_state["repair_weights"])
        iterations = int(resume_state["iterations"])
        segments = int(resume_state["segments"])
        stale = int(resume_state["stale"])
        curve = list(resume_state["curve"])
        prior_elapsed_seconds = float(resume_state.get("elapsed_seconds", 0.0))

    def elapsed_seconds() -> float:
        return prior_elapsed_seconds + time.perf_counter() - started

    def build_resume_state() -> dict:
        return {
            "config_key": config_key,
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "permutation": permutation.copy(),
            "current_schedule": current_schedule,
            "current_cost": current_cost,
            "best_cost": best_cost,
            "best_schedule": best_schedule,
            "best_permutation": best_permutation.copy(),
            "archive": list(archive),
            "ideal": ideal.copy(),
            "scale": scale.copy(),
            "reference": reference.copy(),
            "qualified_cost": qualified_cost,
            "qualified_hv": qualified_hv,
            "temperature": temperature,
            "destroy_weights": dict(destroy_weights),
            "repair_weights": dict(repair_weights),
            "iterations": iterations,
            "segments": segments,
            "stale": stale,
            "curve": list(curve),
            "elapsed_seconds": elapsed_seconds(),
        }

    saturated = False
    time_capped = False
    while segments < max_segments:
        if max_seconds is not None and elapsed_seconds() >= max_seconds:
            time_capped = True
            break
        destroy_score = {name: 0.0 for name in DESTROY_OPERATORS}
        destroy_use = {name: 0 for name in DESTROY_OPERATORS}
        repair_score = {name: 0.0 for name in REPAIR_OPERATORS}
        repair_use = {name: 0 for name in REPAIR_OPERATORS}
        for _ in range(iterations_per_segment):
            destroy_name = _roulette(destroy_weights, rng)
            repair_name = _roulette(repair_weights, rng)
            fraction = rng.uniform(*destroy_fraction)
            n_remove = _normalize_remove_count(
                instance.num_slabs, round(instance.num_slabs * fraction)
            )
            kept, removed = destroy_permutation(
                permutation,
                current_schedule,
                instance,
                profile,
                operator=destroy_name,
                n_remove=n_remove,
                rng=rng,
                D=D,
                PE=PE,
            )
            candidate_permutation = repair_permutation(
                kept,
                removed,
                instance,
                profile,
                weights=weights,
                operator=repair_name,
                rng=rng,
                D=D,
                PE=PE,
            )
            candidate_schedule = split_permutation_dp(
                candidate_permutation,
                instance,
                profile,
                weights=weights,
                D=D,
                PE=PE,
            )
            candidate_cost = _weighted_cost(candidate_schedule, instance, profile, weights)
            archive_added = _archive_add(
                archive,
                _component_vector(candidate_schedule, instance, profile),
                candidate_schedule,
                candidate_permutation,
            )
            accepted = accept_candidate(
                candidate_cost,
                current_cost,
                temperature=temperature,
                random_value=float(rng.random()),
            )
            reward = 0.0
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_schedule = candidate_schedule
                best_permutation = candidate_permutation.copy()
                reward = 5.0
            elif accepted and candidate_cost < current_cost:
                reward = 3.0
            elif accepted:
                reward = 1.0
            if archive_added:
                reward += 1.0
            if accepted:
                permutation = candidate_permutation
                current_schedule = candidate_schedule
                current_cost = candidate_cost
            destroy_score[destroy_name] += reward
            repair_score[repair_name] += reward
            destroy_use[destroy_name] += 1
            repair_use[repair_name] += 1
            iterations += 1
            temperature *= cooling_rate

        for name in DESTROY_OPERATORS:
            if destroy_use[name]:
                destroy_weights[name] = max(
                    1e-6,
                    (1.0 - reaction_factor) * destroy_weights[name]
                    + reaction_factor * destroy_score[name] / destroy_use[name],
                )
        for name in REPAIR_OPERATORS:
            if repair_use[name]:
                repair_weights[name] = max(
                    1e-6,
                    (1.0 - reaction_factor) * repair_weights[name]
                    + reaction_factor * repair_score[name] / repair_use[name],
                )
        segments += 1
        hv = _archive_hv(archive, ideal, scale, reference)
        stagnation = update_stagnation(
            best_cost=best_cost,
            hypervolume=hv,
            qualified_cost=qualified_cost,
            qualified_hypervolume=qualified_hv,
            stale=stale,
            rel_tol=rel_tol,
            stop_metric=stop_metric,
        )
        qualified_cost = stagnation.qualified_cost
        qualified_hv = stagnation.qualified_hypervolume
        stale = stagnation.stale
        event = stagnation.event
        curve.append(
            {
                "segment": segments,
                "iterations": iterations,
                "best_cost": best_cost,
                "hypervolume": hv,
                "event": event,
            }
        )
        if segment_callback is not None:
            segment_callback(build_resume_state())
        if stale >= stale_segments:
            saturated = True
            break
        if max_seconds is not None and elapsed_seconds() >= max_seconds:
            time_capped = True
            break

    archive.sort(key=lambda entry: tuple(entry["vector"].tolist()))
    archive_vectors = np.asarray([entry["vector"] for entry in archive], dtype=np.float64)
    state = build_resume_state()
    return ALNSResult(
        best_cost=best_cost,
        best_schedule=best_schedule,
        archive=archive_vectors,
        archive_schedules=[entry["schedule"] for entry in archive],
        stop_reason=(
            "saturated"
            if saturated
            else "time_cap"
            if time_capped
            else "hard_cap"
        ),
        saturated=saturated,
        iterations=iterations,
        segments=segments,
        curve=list(curve),
        operator_weights={
            "destroy": dict(destroy_weights),
            "repair": dict(repair_weights),
        },
        resume_state=state,
    )
