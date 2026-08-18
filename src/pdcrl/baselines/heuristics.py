"""Non-learning baselines for the HPCVRP hot-rolling problem.

- greedy_schedule: nearest-neighbour construction (min weighted transition), capacity-aware,
  prize-collecting (leftover slabs that cannot form a >= Q_S unit are left unscheduled).
NSGA-II / multi-start-LS metaheuristics live in pdcrl.baselines.nsga2; the local-search
improver in pdcrl.baselines.local_search.
"""

from __future__ import annotations

import numpy as np

from pdcrl.data.loader import transition_penalty
from pdcrl.rewards.costs import full_weights
from pdcrl.schedule import Schedule


def greedy_schedule(instance, profile, weights=None, rng=None, rcl_k: int = 1) -> Schedule:
    """Greedy nearest-neighbour construction: start a unit with the widest remaining slab, append
    the min-weighted-transition feasible slab; once the unit reaches Q_S, keep appending while the
    cheapest continuation costs no more than one roll change (w4 * roll_change_cost), then close.
    Leftover slabs that cannot form a valid unit are left unscheduled (prize-collecting).

    With ``rng`` and ``rcl_k > 1`` the construction is GRASP-randomised: each choice is drawn
    uniformly from the top-``rcl_k`` candidates (restricted candidate list) instead of the argmin."""
    w = full_weights(weights)
    n = instance.num_slabs
    scheduled = np.zeros(n, dtype=bool)
    units: list[list[int]] = []

    def step_cost(prev, j):
        d, pe = transition_penalty(instance.slabs[prev], instance.slabs[j], profile)
        return w[0] * d + w[2] * pe

    def pick(cands, key):
        ranked = sorted(cands, key=key)
        if rng is None or rcl_k <= 1:
            return ranked[0]
        return ranked[int(rng.integers(min(rcl_k, len(ranked))))]

    while True:
        remaining = [j for j in range(n) if not scheduled[j]
                     and instance.capacity(j) <= profile.capacity_max_m]
        if not remaining:
            break
        start = pick(remaining, key=lambda j: -instance.slabs[j].width_mm)
        unit = [start]
        scheduled[start] = True
        length = instance.capacity(start)
        last = start
        w_rc = w[3] * profile.roll_change_cost
        while True:
            cands = [j for j in range(n) if not scheduled[j]
                     and length + instance.capacity(j) <= profile.capacity_max_m]
            if not cands:
                break
            best_cost = min(step_cost(last, j) for j in cands)
            if length >= profile.capacity_min_m and best_cost > w_rc:
                break   # cheapest continuation dearer than a fresh roll change -> close
            nxt = pick(cands, key=lambda j: (step_cost(last, j), j))
            unit.append(nxt)
            scheduled[nxt] = True
            length += instance.capacity(nxt)
            last = nxt
        if length >= profile.capacity_min_m:
            units.append(unit)
        else:  # too short to be valid -> release and stop (remaining left unscheduled)
            for j in unit:
                scheduled[j] = False
            break
    unsched = {j for j in range(n) if not scheduled[j]}
    return Schedule(n, units, unsched)
