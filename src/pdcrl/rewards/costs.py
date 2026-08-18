"""HPCVRP objective/cost engine (PLOS ONE 2020 formulation + scalar roll-change cost).

Pareto objectives (OBJECTIVE_NAMES, 3): S1 transition (width/gauge/hardness jumps, d_ij),
S2 unscheduled prize, S3 energy jump (P^e). The scalarised objective adds S4 = roll_change_cost
* number of rolling units (an operational setup cost, not a Pareto axis). Internal incremental
vectors carry 4 components (COST_COMPONENTS); opening a unit charges the roll change.
evaluate_full scores a full solution; incremental_cost gives the per-placement delta for a
constructive MDP (sums to evaluate_full's components; unscheduled handled by prize_delta).
"""

from __future__ import annotations

import numpy as np

from pdcrl.data.loader import transition_penalty

# Canonical Pareto objective order (fronts/HV/IGD) — unchanged by S4.
OBJECTIVE_NAMES = ("transition", "unscheduled", "energy")
# Internal cost-component order for incremental accounting (scalarisation weights align).
COST_COMPONENTS = ("transition", "unscheduled", "energy", "rollchange")


def full_weights(weights=None) -> np.ndarray:
    """Length-4 scalarisation weights (w_transition, w_unscheduled, w_energy, w_rollchange).
    None -> ones(4); a length-3 vector is padded with w_rollchange = 1.0."""
    if weights is None:
        return np.ones(4)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] == 3:
        return np.concatenate([w, [1.0]])
    if w.shape[0] != 4:
        raise ValueError(f"weights must have length 3 or 4, got {w.shape[0]}")
    return w


def evaluate_full(schedule, instance, profile) -> dict:
    """Return the objective components, their weighted sum, and a feasibility flag."""
    slabs = instance.slabs
    s1 = 0.0  # transition
    s3 = 0.0  # energy
    m = 0     # rolling units
    feasible = schedule.is_partition()
    for unit in schedule.units:
        if not unit:
            continue
        m += 1
        length = sum(instance.capacity(i) for i in unit)
        if not (profile.capacity_min_m <= length <= profile.capacity_max_m):
            feasible = False
        for k in range(len(unit) - 1):
            d, pe = transition_penalty(slabs[unit[k]], slabs[unit[k + 1]], profile)
            s1 += d
            s3 += pe
    s2 = sum(instance.prize(i, profile) for i in schedule.unscheduled)  # unscheduled
    s4 = profile.roll_change_cost * m                                    # roll changes

    cost = {
        "transition_cost": s1,
        "unscheduled_cost": s2,
        "energy_cost": s3,
        "rollchange_cost": s4,
        "feasible": feasible,
    }
    cost["objective_value"] = (
        profile.w_transition * s1 + profile.w_unscheduled * s2
        + profile.w_energy * s3 + profile.w_rollchange * s4
    )
    return cost


def incremental_cost(prev_idx, new_idx, instance, profile) -> np.ndarray:
    """Delta (len 4, in COST_COMPONENTS order) of appending slab ``new_idx`` after ``prev_idx``.

    Opening a unit (``prev_idx is None``) charges the roll change: [0, 0, 0, roll_change_cost].
    Unscheduled prize is accounted separately (use ``prize_delta``)."""
    d = np.zeros(4)
    if prev_idx is None:
        d[3] = profile.roll_change_cost
    else:
        dt, pe = transition_penalty(instance.slabs[prev_idx], instance.slabs[new_idx], profile)
        d[0] = dt
        d[2] = pe
    return d


def prize_delta(idx, instance, profile) -> np.ndarray:
    """Delta (len 4) for leaving slab ``idx`` unscheduled: [0, p_i, 0, 0]."""
    d = np.zeros(4)
    d[1] = instance.prize(idx, profile)
    return d
