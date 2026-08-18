"""Evaluation metrics for single- and multi-objective scheduling results.

Responsibility: scoring and comparison utilities used to produce the paper's tables/figures.
    - single-objective: weighted objective value, gap-to-reference, feasibility rate, runtime
    - multi-objective: hypervolume, IGD, Pareto-front extraction/coverage
    - statistics: aggregation across seeds, Wilcoxon significance test

Depends on: numpy, scipy, pymoo (indicators). Used by: experiments/evaluate.py, scripts/figures.
"""

from __future__ import annotations

import numpy as np


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of ``points`` (minimization).

    A point j dominates point i if j is ≤ i in all objectives and < in at least one.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j dominates i ?
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                keep[i] = False
                break
    return pts[keep]


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume of a Pareto front (minimization) w.r.t. a reference point."""
    from pymoo.indicators.hv import HV
    return float(HV(ref_point=np.asarray(ref_point, dtype=np.float64))(np.asarray(front, dtype=np.float64)))


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted Generational Distance to a reference front."""
    from pymoo.indicators.igd import IGD
    return float(IGD(np.asarray(reference_front, dtype=np.float64))(np.asarray(front, dtype=np.float64)))


def gap_to_reference(value: float, reference: float) -> float:
    """Relative optimality gap: (value - reference) / reference."""
    return (value - reference) / reference


def pooled_front_indicators(fronts_by_method: dict[str, np.ndarray]) -> dict:
    """Compute HV, IGD+, and epsilon under one pooled-front normalization."""
    if not fronts_by_method:
        raise ValueError("at least one method front is required")
    dimensions = {np.atleast_2d(front).shape[1] for front in fronts_by_method.values()}
    if len(dimensions) != 1:
        raise ValueError("all method fronts must have the same objective dimension")
    pooled = np.vstack(
        [np.asarray(front, dtype=np.float64) for front in fronts_by_method.values()]
    )
    reference_front = pareto_front(pooled)
    ideal = reference_front.min(axis=0)
    nadir = reference_front.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    normalized_reference = (reference_front - ideal) / scale
    hv_reference = np.full(reference_front.shape[1], 1.1, dtype=np.float64)

    from pymoo.indicators.epsilon import Epsilon
    from pymoo.indicators.igd_plus import IGDPlus

    methods = {}
    for method, front in sorted(fronts_by_method.items()):
        normalized = (pareto_front(np.asarray(front, dtype=np.float64)) - ideal) / scale
        eligible = normalized[np.all(normalized < hv_reference, axis=1)]
        methods[method] = {
            "hypervolume": hypervolume(eligible, hv_reference) if len(eligible) else 0.0,
            "igd_plus": float(IGDPlus(normalized_reference)(normalized)),
            "epsilon": float(Epsilon(normalized_reference)(normalized)),
            "front_size": len(normalized),
        }
    return {
        "normalization": {
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "hv_reference": hv_reference.tolist(),
        },
        "reference_front": reference_front.tolist(),
        "methods": methods,
    }
