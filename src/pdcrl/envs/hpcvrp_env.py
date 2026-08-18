"""Constructive MDP environment for the HPCVRP hot-rolling scheduling problem.

The agent builds rolling units (routes) one at a time. At each step it either:
  - appends a remaining slab j to the current unit (feasible if unscheduled and the unit's
    rolling length stays <= Q_L),
  - closes the current unit and opens a new one (action = n; feasible if the current unit is
    non-empty and its length >= Q_S),
  - finishes (action = n+1; feasible if the current unit is empty or >= Q_S), leaving all
    remaining slabs unscheduled (prize-collecting).

Reward = -(w . delta), where delta is the 4-component incremental cost (COST_COMPONENTS order:
transition, unscheduled, energy, rollchange): append -> [d_ij, 0, pe_ij, C_roll if opening];
close -> [0,0,0,0]; finish -> [0, prize(remaining), 0, 0]. A `reset(start_slab=k)` opens a unit
immediately, so its opening roll-change is stashed as pending and folded into the first step's
cost_delta (the reset itself returns no reward/delta). Accumulated reward over an episode equals
-weighted evaluate_full of the produced Schedule.
"""

from __future__ import annotations

import numpy as np

from pdcrl.rewards.costs import incremental_cost, prize_delta, evaluate_full, full_weights
from pdcrl.schedule import Schedule


class HPCVRPEnv:
    def __init__(self, instance, profile, objective_weights=None):
        self.instance = instance
        self.profile = profile
        self.n = instance.num_slabs
        self.w = full_weights(objective_weights)
        self.CLOSE = self.n
        self.FINISH = self.n + 1
        self.n_actions = self.n + 2

    def reset(self, *, seed=None, start_slab=None):
        self._scheduled = np.zeros(self.n, dtype=bool)
        self._units: list[list[int]] = []
        self._cur: list[int] = []
        self._cur_len = 0.0
        self._last = None  # last slab idx in current unit
        self._done = False
        self._pending = np.zeros(4)   # opening delta from start_slab, folded into the 1st step
        if start_slab is not None:  # POMO multi-start: force the first slab
            self._pending = self._append(int(start_slab))
        return self._obs(), {}

    # ---- feasibility ----
    def action_mask(self) -> np.ndarray:
        m = np.zeros(self.n_actions, dtype=bool)
        if self._done:
            return m
        prof = self.profile
        for j in range(self.n):
            if not self._scheduled[j] and self._cur_len + self.instance.capacity(j) <= prof.capacity_max_m:
                m[j] = True
        # close current unit: non-empty and meets Q_S
        if self._cur and self._cur_len >= prof.capacity_min_m:
            m[self.CLOSE] = True
        # finish: current unit empty (nothing dangling) or closeable
        if (not self._cur) or self._cur_len >= prof.capacity_min_m:
            m[self.FINISH] = True
        return m

    # ---- transitions ----
    def _append(self, j):
        delta = incremental_cost(self._last, j, self.instance, self.profile)
        self._cur.append(j)
        self._cur_len += self.instance.capacity(j)
        self._last = j
        self._scheduled[j] = True
        return delta

    def _close(self):
        self._units.append(self._cur)
        self._cur = []
        self._cur_len = 0.0
        self._last = None
        return np.zeros(4)

    def step(self, action):
        if self._done:
            raise RuntimeError("step after done")
        action = int(action)
        if action < self.n and self._scheduled[action]:
            raise ValueError(f"slab {action} already scheduled")
        pending, self._pending = self._pending, np.zeros(4)
        if action == self.FINISH:
            return self._finish(pending)
        if action == self.CLOSE:
            delta = self._close() + pending
        else:
            delta = self._append(action) + pending
        # auto-finish if no slab can be appended to the current unit AND none can start a new one
        if not self._can_append_current() and not self._can_start_new():
            return self._finish(delta)
        return self._obs(), -float(self.w @ delta), False, False, {"cost_delta": delta}

    def _can_append_current(self):
        cap = self.profile.capacity_max_m
        return any(not self._scheduled[j] and self._cur_len + self.instance.capacity(j) <= cap
                   for j in range(self.n))

    def _can_start_new(self):
        cap = self.profile.capacity_max_m
        return any(not self._scheduled[j] and self.instance.capacity(j) <= cap for j in range(self.n))

    def _finish(self, step_delta):
        """Terminate. A valid open unit (>= Q_S) is closed; a short one is *released* — its slabs
        return to unscheduled (prizes paid below) and its internal transitions and the unit's
        roll-change are refunded, so the env never emits a capacity-infeasible schedule. The
        returned cost_delta bundles the triggering step's delta with refunds and prizes so
        per-step deltas sum to evaluate_full."""
        delta = np.array(step_delta, dtype=float)
        if self._cur:
            if self._cur_len >= self.profile.capacity_min_m:
                self._close()
            else:
                for a, b in zip([None] + self._cur, self._cur):
                    delta = delta - incremental_cost(a, b, self.instance, self.profile)
                for s in self._cur:
                    self._scheduled[s] = False
                self._cur = []
                self._cur_len = 0.0
                self._last = None
        for i in range(self.n):
            if not self._scheduled[i]:
                delta = delta + prize_delta(i, self.instance, self.profile)
        self._done = True
        return self._obs(), -float(self.w @ delta), True, False, self._final_info(delta)

    def _schedule(self) -> Schedule:
        unsched = {i for i in range(self.n) if not self._scheduled[i]}
        return Schedule(self.n, [u for u in self._units if u], unsched)

    def _final_info(self, last_delta):
        sched = self._schedule()
        return {"cost_delta": last_delta, "schedule": sched,
                "cost": evaluate_full(sched, self.instance, self.profile)}

    def _obs(self):
        unit_mask = np.zeros(self.n, dtype=bool)
        unit_mask[self._cur] = True
        return {
            "scheduled": self._scheduled.copy(),
            "unit_mask": unit_mask,
            "cur_len": self._cur_len,
            "last": self._last,
            "action_mask": self.action_mask(),
        }
