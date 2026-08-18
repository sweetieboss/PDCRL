"""First-improvement local search over HPCVRP schedules (insert + relocate + swap + 2-opt).

Neighbourhoods, applied until a fixed point (or time budget):
  - insert: move an unscheduled slab into any capacity-feasible position whose transition
    delta is smaller than the slab's prize (net objective gain);
  - relocate: move a scheduled slab to a cheaper position, within or across units, keeping
    both units inside [Q_S, Q_L] (a unit may be emptied and removed);
  - swap: exchange two scheduled slabs (within or across units; cross-unit swaps keep both
    unit lengths inside [Q_S, Q_L] where relocate alone would violate them);
  - 2-opt: reverse a segment inside a unit (width/coffin costs are asymmetric, so reversal
    is a real move);
  - merge: concatenate two units (either orientation) when the combined length fits Q_L and
    the seam transition costs less than the saved roll change;

Used both to strengthen the metaheuristic baselines (greedy+LS, GRASP, NSGA-II+LS) and as a
post-processor on RL solutions (RL+LS hybrid). Costs use the same weighted objective as the
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from pdcrl.data.loader import transition_matrices
from pdcrl.rewards.costs import full_weights
from pdcrl.schedule import Schedule


@dataclass(frozen=True)
class LocalSearchResult:
    schedule: Schedule
    stop_reason: str
    saturated: bool
    elapsed_seconds: float
    passes: int
    accepted_moves: dict[str, int]


def local_search_with_stats(schedule: Schedule, instance, profile, weights=(1, 1, 1, 1),
                            max_sec: float = 60.0) -> LocalSearchResult:
    t0 = time.perf_counter()

    def expired() -> bool:
        return time.perf_counter() - t0 >= max_sec

    w = full_weights(weights)
    D, PE = transition_matrices(instance, profile)
    C = w[0] * D + w[2] * PE
    Q = np.array([s.rolling_length_m for s in instance.slabs])
    prize = w[1] * np.array([profile.prize_base * s.priority for s in instance.slabs])
    Qs, Ql = profile.capacity_min_m, profile.capacity_max_m
    w_rc = w[3] * profile.roll_change_cost

    units = [list(u) for u in schedule.units]
    unsched = set(schedule.unscheduled)
    accepted = {name: 0 for name in ("insert", "relocate", "merge", "swap", "two_opt")}
    passes = 0
    hit_cap = False

    # dissolve capacity-infeasible units (below Q_S) so the output is always feasible;
    # their slabs go to unscheduled and are re-inserted below where profitable
    keep = []
    for u in units:
        if Q[u].sum() < Qs:
            unsched.update(u)
        else:
            keep.append(u)
    units = keep

    def insert_delta(u, pos, s):
        """Transition-cost delta of inserting slab s at position pos of unit u."""
        d = 0.0
        if pos > 0:
            d += C[u[pos - 1], s]
        if pos < len(u):
            d += C[s, u[pos]]
        if 0 < pos < len(u):
            d -= C[u[pos - 1], u[pos]]
        return d

    def removal_delta(u, i):
        """Transition-cost delta of removing u[i] (negative of what it currently costs)."""
        d = 0.0
        if i > 0:
            d += C[u[i - 1], u[i]]
        if i < len(u) - 1:
            d += C[u[i], u[i + 1]]
        if 0 < i < len(u) - 1:
            d -= C[u[i - 1], u[i + 1]]
        return d

    def unit_cost(u):
        return sum(C[a, b] for a, b in zip(u, u[1:]))

    def swap_pass():
        """First improving swap; None means the safety deadline interrupted the scan."""
        for ui, u in enumerate(units):
            if expired():
                return None
            lu = Q[u].sum()
            for i in range(len(u)):
                a = u[i]
                for vi in range(ui, len(units)):
                    v = units[vi]
                    lv = Q[v].sum()
                    for j in range(i + 1 if vi == ui else 0, len(v)):
                        b = v[j]
                        if vi != ui:
                            if not (Qs <= lu - Q[a] + Q[b] <= Ql):
                                continue
                            if not (Qs <= lv - Q[b] + Q[a] <= Ql):
                                continue
                        old = unit_cost(u) + (0.0 if vi == ui else unit_cost(v))
                        u[i], v[j] = b, a
                        new = unit_cost(u) + (0.0 if vi == ui else unit_cost(v))
                        if new < old - 1e-9:
                            accepted["swap"] += 1
                            return True
                        u[i], v[j] = a, b   # revert
        return False

    def merge_pass():
        """First improving merge; None means the safety deadline interrupted the scan."""
        for ui in range(len(units)):
            if expired():
                return None
            for vi in range(ui + 1, len(units)):
                u, v = units[ui], units[vi]
                if Q[u].sum() + Q[v].sum() > Ql:
                    continue
                for a, b in ((u, v), (v, u)):
                    if C[a[-1], b[0]] < w_rc - 1e-9:
                        units[ui] = list(a) + list(b)
                        del units[vi]
                        accepted["merge"] += 1
                        return True
        return False

    def two_opt_pass():
        """First improving reversal; None means the safety deadline interrupted the scan."""
        for u in units:
            if expired():
                return None
            base = unit_cost(u)
            for i in range(len(u) - 1):
                for j in range(i + 1, len(u)):
                    u[i:j + 1] = u[i:j + 1][::-1]
                    if unit_cost(u) < base - 1e-9:
                        accepted["two_opt"] += 1
                        return True
                    u[i:j + 1] = u[i:j + 1][::-1]   # revert
        return False

    improved = True
    while improved:
        if expired():
            hit_cap = True
            break
        passes += 1
        improved = False
        # --- insert unscheduled slabs (best gain per slab) ---
        for s in sorted(unsched):
            if expired():
                hit_cap = True
                break
            best = None
            for ui, u in enumerate(units):
                if Q[u].sum() + Q[s] > Ql:
                    continue
                for pos in range(len(u) + 1):
                    gain = prize[s] - insert_delta(u, pos, s)
                    if gain > 1e-9 and (best is None or gain > best[0]):
                        best = (gain, ui, pos)
            if best is not None:
                _, ui, pos = best
                units[ui].insert(pos, s)
                unsched.discard(s)
                improved = True
                accepted["insert"] += 1
        if hit_cap:
            break
        # --- relocate one slab (first improvement, then rescan) ---
        moved = False
        for ui, u in enumerate(units):
            if expired():
                hit_cap = True
                break
            if moved:
                break
            src_len = Q[u].sum()
            for i in range(len(u)):
                s = u[i]
                # source unit must stay valid after removal (or become empty and be dropped)
                if len(u) > 1 and src_len - Q[s] < Qs:
                    continue
                rm = removal_delta(u, i)
                for vi, v in enumerate(units):
                    if vi != ui and Q[v].sum() + Q[s] > Ql:
                        continue
                    base = [x for x in v if x != s] if vi == ui else v
                    for pos in range(len(base) + 1):
                        if vi == ui and pos == (i if pos < i else i):
                            pass  # same-position no-ops are filtered by the delta check
                        add = insert_delta(base, pos, s)
                        bonus = w_rc if (vi != ui and len(u) == 1) else 0.0
                        if add < rm + bonus - 1e-9:
                            u.pop(i)
                            if vi == ui:
                                u.insert(pos, s)
                            else:
                                units[vi].insert(pos, s)
                            moved = True
                            improved = True
                            accepted["relocate"] += 1
                            break
                    if moved:
                        break
                if moved:
                    break
        if hit_cap:
            break
        if not moved:
            merge_result = merge_pass()
            if merge_result is None:
                hit_cap = True
            elif merge_result:
                improved = True
            else:
                swap_result = swap_pass()
                if swap_result is None:
                    hit_cap = True
                elif swap_result:
                    improved = True
                else:
                    two_opt_result = two_opt_pass()
                    if two_opt_result is None:
                        hit_cap = True
                    elif two_opt_result:
                        improved = True
        units = [u for u in units if u]
        if hit_cap:
            break
    result_schedule = Schedule(schedule.num_slabs, units, unsched)
    return LocalSearchResult(
        schedule=result_schedule,
        stop_reason="hard_cap" if hit_cap else "fixed_point",
        saturated=not hit_cap,
        elapsed_seconds=time.perf_counter() - t0,
        passes=passes,
        accepted_moves=accepted,
    )


def local_search(schedule: Schedule, instance, profile, weights=(1, 1, 1, 1),
                 max_sec: float = 60.0) -> Schedule:
    """Compatibility wrapper returning only the improved schedule."""
    return local_search_with_stats(schedule, instance, profile, weights, max_sec).schedule
