"""NSGA-II and GRASP metaheuristic baselines for HPCVRP.

Encoding: a permutation of all slabs, decoded by a capacity-window split (fill the open
unit while <= Q_L; a slab that does not fit is skipped to unscheduled if the unit cannot
close yet; a tail below Q_S is left unscheduled) — decoded schedules are always feasible.
NSGA-II (pymoo, order crossover + mixed inversion/swap mutation) evolves the permutation
against the 3 objectives (S1 transition, S2 unscheduled prize, S3 energy); the initial
population is *seeded* with the greedy solution and perturbations of it (vanilla random
initialisation does not converge at 151-331 slabs). GRASP restarts local search from
randomised-greedy (restricted-candidate-list) constructions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.ox import OrderCrossover

from pdcrl.baselines.heuristics import greedy_schedule
from pdcrl.baselines.local_search import local_search, local_search_with_stats
from pdcrl.data.loader import transition_matrices
from pdcrl.eval.metrics import hypervolume
from pdcrl.rewards.costs import evaluate_full
from pdcrl.rewards.costs import full_weights
from pdcrl.schedule import Schedule


@dataclass(frozen=True)
class SearchResult:
    best_cost: float
    best_schedule: Schedule
    stop_reason: str
    saturated: bool
    iterations: int
    curve: list[dict]
    resume_state: dict
    local_search_caps: int = 0


@dataclass(frozen=True)
class MultiObjectiveSearchResult:
    front: np.ndarray
    schedules: list[Schedule]
    best_cost: float
    best_schedule: Schedule
    stop_reason: str
    saturated: bool
    generations: int
    curve: list[dict]
    resume_state: dict


def _weighted_schedule_cost(schedule, instance, profile, weights) -> float:
    scored = evaluate_full(schedule, instance, profile)
    components = np.asarray(
        [
            scored["transition_cost"],
            scored["unscheduled_cost"],
            scored["energy_cost"],
            scored["rollchange_cost"],
        ],
        dtype=np.float64,
    )
    return float(np.dot(full_weights(weights), components))


class _SaturatedPermutationProblem(ElementwiseProblem):
    def __init__(self, instance, profile, weights, D, PE, prize):
        super().__init__(n_var=instance.num_slabs, n_obj=3, xl=0,
                         xu=instance.num_slabs - 1, vtype=int)
        self.instance = instance
        self.profile = profile
        self.weights = tuple(weights)
        self.D = D
        self.PE = PE
        self.prize = prize
        self.best_cost = float("inf")
        self.best_schedule = None

    def _evaluate(self, x, out, *args, **kwargs):
        schedule = split_permutation_dp(
            x,
            self.instance,
            self.profile,
            weights=self.weights,
            D=self.D,
            PE=self.PE,
        )
        objectives = _objectives(schedule, self.D, self.PE, self.prize)
        out["F"] = list(objectives)
        cost = float(np.dot(full_weights(self.weights)[:3], objectives)) + (
            full_weights(self.weights)[3]
            * self.profile.roll_change_cost
            * len(schedule.units)
        )
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_schedule = schedule


class _SeededPermutationSampling(Sampling):
    def __init__(self, greedy_permutation, seed: int):
        super().__init__()
        self.greedy_permutation = np.asarray(greedy_permutation, dtype=int)
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        n = problem.n_var
        samples = [self.greedy_permutation.copy()]
        while len(samples) < n_samples:
            if len(samples) % 2:
                candidate = self.greedy_permutation.copy()
                for _ in range(int(self.rng.integers(1, 9))):
                    left, right = self.rng.integers(0, n, 2)
                    candidate[left], candidate[right] = candidate[right], candidate[left]
            else:
                candidate = self.rng.permutation(n)
            samples.append(candidate)
        return np.asarray(samples)


class _SeededMixedMutation(Mutation):
    def __init__(self, n: int, seed: int):
        super().__init__()
        self.n = n
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        mutated = X.copy()
        for row in range(len(mutated)):
            left, right = sorted(self.rng.integers(0, self.n, 2))
            if left == right:
                continue
            if self.rng.random() < 0.5:
                mutated[row, left:right + 1] = mutated[row, left:right + 1][::-1]
            else:
                mutated[row, left], mutated[row, right] = (
                    mutated[row, right],
                    mutated[row, left],
                )
        return mutated


def _greedy_permutation(instance, profile) -> np.ndarray:
    """Flatten the greedy schedule into a permutation (units in order, unscheduled at the end)."""
    g = greedy_schedule(instance, profile)
    order = [s for u in g.units for s in u] + sorted(g.unscheduled)
    return np.asarray(order, dtype=int)


def split_permutation(perm, instance, profile) -> Schedule:
    Q = np.array([s.rolling_length_m for s in instance.slabs])
    Qs, Ql = profile.capacity_min_m, profile.capacity_max_m
    units: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0.0
    unscheduled: set[int] = set()
    for s in map(int, perm):
        if cur_len + Q[s] <= Ql:
            cur.append(s)
            cur_len += Q[s]
        elif cur_len >= Qs:
            units.append(cur)
            cur = [s]
            cur_len = Q[s]
        else:
            unscheduled.add(s)   # does not fit and the unit cannot close yet
    if cur and cur_len >= Qs:
        units.append(cur)
    else:
        unscheduled.update(cur)
    return Schedule(instance.num_slabs, units, unscheduled)


def split_permutation_dp(perm, instance, profile, weights=(1, 1, 1, 1), D=None, PE=None) -> Schedule:
    """Optimal split of a giant-tour permutation (Prins-style DP): choose unit boundaries —
    and per-slab skips — minimising the weighted objective, subject to [Q_S, Q_L] per unit.
    Each unit additionally charges w4 * roll_change_cost, so the DP trades off fewer/larger
    units against transition/energy cost. O(n * max_unit_len). Guarantees
    decoded(greedy permutation) <= greedy schedule cost."""
    if D is None or PE is None:
        D, PE = transition_matrices(instance, profile)
    w = full_weights(weights)
    C = w[0] * D + w[2] * PE
    Q = np.array([s.rolling_length_m for s in instance.slabs])
    prize_w = w[1] * np.array([profile.prize_base * s.priority for s in instance.slabs])
    Qs, Ql = profile.capacity_min_m, profile.capacity_max_m
    w_rc = w[3] * profile.roll_change_cost
    p = np.asarray(perm, dtype=int)
    n = len(p)
    f = np.full(n + 1, np.inf)
    f[0] = 0.0
    choice = [None] * (n + 1)   # ("skip",) or ("unit", i) meaning segment p[i:j]
    for j in range(1, n + 1):
        f[j] = f[j - 1] + prize_w[p[j - 1]]   # leave slab p[j-1] unscheduled
        choice[j] = ("skip",)
        seg_len = 0.0
        seg_cost = 0.0
        for i in range(j - 1, -1, -1):        # segment p[i:j], grown at the front
            s = p[i]
            if i < j - 1:
                seg_cost += C[s, p[i + 1]]
            seg_len += Q[s]
            if seg_len > Ql:
                break
            if seg_len >= Qs:
                cand = f[i] + seg_cost + w_rc
                if cand < f[j]:
                    f[j] = cand
                    choice[j] = ("unit", i)
    units: list[list[int]] = []
    unscheduled: set[int] = set()
    j = n
    while j > 0:
        ch = choice[j]
        if ch[0] == "skip":
            unscheduled.add(int(p[j - 1]))
            j -= 1
        else:
            i = ch[1]
            units.append([int(x) for x in p[i:j]])
            j = i
    units.reverse()
    return Schedule(instance.num_slabs, units, unscheduled)


def _objectives(sched: Schedule, D, PE, prize):
    s1 = sum(D[a, b] for u in sched.units for a, b in zip(u, u[1:]))
    s3 = sum(PE[a, b] for u in sched.units for a, b in zip(u, u[1:]))
    s2 = float(prize[list(sched.unscheduled)].sum()) if sched.unscheduled else 0.0
    return float(s1), s2, float(s3)


def nsga2_pareto(instance, profile, pop_size: int = 100, n_gen: int = 200, seed: int = 0,
                 weights=(1, 1, 1, 1)):
    """Evolve permutations with NSGA-II (greedy-seeded init, OX crossover, mixed
    inversion/swap mutation, optimal-split DP decoder).

    Returns (F (m,3) Pareto objectives, [Schedule]*m, (best_weighted_cost, best Schedule))
    where the best-weighted incumbent is tracked over *all* evaluations (crowding-distance
    truncation may drop it from the final front). The Pareto front stays strictly 3-D (S1/S2/S3);
    the incumbent's weighted cost additionally charges w4 * roll_change_cost per rolling unit."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.optimize import minimize

    D, PE = transition_matrices(instance, profile)
    prize = np.array([profile.prize_base * s.priority for s in instance.slabs])
    w = full_weights(weights)
    n = instance.num_slabs
    greedy_perm = _greedy_permutation(instance, profile)
    best = {"cost": float("inf"), "sched": None}

    class _Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=3, xl=0, xu=n - 1, vtype=int)

        def _evaluate(self, x, out, *args, **kwargs):
            sched = split_permutation_dp(x, instance, profile, weights=weights, D=D, PE=PE)
            objs = _objectives(sched, D, PE, prize)
            out["F"] = list(objs)
            c = float(np.dot(w[:3], objs)) + w[3] * profile.roll_change_cost * len(sched.units)
            if c < best["cost"]:
                best["cost"] = c
                best["sched"] = sched

    class _SeededSampling(Sampling):
        """Greedy permutation + swap-perturbed copies + random fill."""
        def _do(self, problem, n_samples, **kwargs):
            rng = np.random.default_rng(seed)
            X = [greedy_perm.copy()]
            while len(X) < n_samples:
                if len(X) % 2 == 1:      # perturbed greedy: a few random swaps
                    x = greedy_perm.copy()
                    for _ in range(int(rng.integers(1, 9))):
                        i, j = rng.integers(0, n, 2)
                        x[i], x[j] = x[j], x[i]
                else:                    # random permutation for diversity
                    x = rng.permutation(n)
                X.append(x)
            return np.asarray(X)

    class _MixedMutation(Mutation):
        """50/50 inversion or pairwise swap per offspring."""
        def _do(self, problem, X, **kwargs):
            rng = np.random.default_rng(seed + 1)
            X = X.copy()
            for r in range(len(X)):
                i, j = sorted(rng.integers(0, n, 2))
                if i == j:
                    continue
                if rng.random() < 0.5:
                    X[r, i:j + 1] = X[r, i:j + 1][::-1]
                else:
                    X[r, i], X[r, j] = X[r, j], X[r, i]
            return X

    algo = NSGA2(pop_size=pop_size, sampling=_SeededSampling(),
                 crossover=OrderCrossover(), mutation=_MixedMutation(),
                 eliminate_duplicates=False)
    res = minimize(_Problem(), algo, ("n_gen", n_gen), seed=seed, verbose=False)
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    scheds = [split_permutation_dp(x, instance, profile, weights=weights, D=D, PE=PE) for x in X]
    return F, scheds, (best["cost"], best["sched"])


def grasp(instance, profile, n_starts: int = 20, rcl_k: int = 3, seed: int = 0,
          weights=(1, 1, 1, 1), ls_sec_per_start: float = 60.0):
    """GRASP: randomised-greedy construction (top-``rcl_k`` restricted candidate list) +
    local search per restart; returns (best weighted cost, best Schedule)."""
    from pdcrl.rewards.costs import evaluate_full

    rng = np.random.default_rng(seed)
    best_cost, best_sched = float("inf"), None
    for _ in range(n_starts):
        s0 = greedy_schedule(instance, profile, weights=weights, rng=rng, rcl_k=rcl_k)
        s1 = local_search(s0, instance, profile, weights=weights, max_sec=ls_sec_per_start)
        c = evaluate_full(s1, instance, profile)["objective_value"]
        if c < best_cost:
            best_cost, best_sched = float(c), s1
    return best_cost, best_sched


def grasp_saturated(
    instance,
    profile,
    *,
    stale_restarts: int = 100,
    rel_tol: float = 0.001,
    rcl_k: int = 3,
    seed: int = 0,
    weights=(1, 1, 1, 1),
    ls_sec_per_start: float = 21600.0,
    max_restarts: int = 100000,
    resume_state: dict | None = None,
) -> SearchResult:
    """Run GRASP until restart-level quality saturation."""
    config_key = (stale_restarts, rel_tol, rcl_k, seed, tuple(weights), ls_sec_per_start)
    if resume_state is None:
        rng = np.random.default_rng(seed)
        best_cost = float("inf")
        best_schedule = None
        qualified_cost = float("inf")
        stale = 0
        iterations = 0
        curve: list[dict] = []
        local_search_caps = 0
    else:
        if tuple(resume_state["config_key"]) != config_key:
            raise ValueError("resume_state was created with a different GRASP configuration")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(resume_state["rng_state"])
        best_cost = float(resume_state["best_cost"])
        best_schedule = resume_state["best_schedule"]
        qualified_cost = float(resume_state["qualified_cost"])
        stale = int(resume_state["stale"])
        iterations = int(resume_state["iterations"])
        curve = list(resume_state["curve"])
        local_search_caps = int(resume_state["local_search_caps"])

    saturated = False
    while iterations < max_restarts:
        initial = greedy_schedule(instance, profile, weights=weights, rng=rng, rcl_k=rcl_k)
        improved = local_search_with_stats(
            initial,
            instance,
            profile,
            weights=weights,
            max_sec=ls_sec_per_start,
        )
        local_search_caps += int(improved.stop_reason == "hard_cap")
        cost = _weighted_schedule_cost(improved.schedule, instance, profile, weights)
        iterations += 1
        if cost < best_cost:
            best_cost = cost
            best_schedule = improved.schedule

        if not np.isfinite(qualified_cost) or cost < qualified_cost * (1.0 - rel_tol):
            qualified_cost = cost
            stale = 0
            event = "improved"
        else:
            stale += 1
            event = "stale"
        curve.append(
            {
                "restart": iterations,
                "candidate_cost": cost,
                "best_cost": best_cost,
                "event": event,
            }
        )
        if stale >= stale_restarts:
            saturated = True
            break

    state = {
        "config_key": config_key,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "best_cost": best_cost,
        "best_schedule": best_schedule,
        "qualified_cost": qualified_cost,
        "stale": stale,
        "iterations": iterations,
        "curve": list(curve),
        "local_search_caps": local_search_caps,
    }
    return SearchResult(
        best_cost=best_cost,
        best_schedule=best_schedule,
        stop_reason="saturated" if saturated else "hard_cap",
        saturated=saturated,
        iterations=iterations,
        curve=list(curve),
        resume_state=state,
        local_search_caps=local_search_caps,
    )


def _front_hypervolume(front: np.ndarray, ideal: np.ndarray, scale: np.ndarray,
                       reference: np.ndarray) -> float:
    normalized = (front - ideal) / scale
    eligible = normalized[np.all(normalized < reference, axis=1)]
    return hypervolume(eligible, reference) if len(eligible) else 0.0


def nsga2_saturated(
    instance,
    profile,
    *,
    pop_size: int = 150,
    generations_per_block: int = 100,
    stale_blocks: int = 10,
    rel_tol: float = 0.001,
    seed: int = 0,
    weights=(1, 1, 1, 1),
    max_blocks: int = 1000,
    resume_state: dict | None = None,
) -> MultiObjectiveSearchResult:
    """Advance one NSGA-II population in blocks until weighted and front stagnation."""
    config_key = (
        pop_size,
        generations_per_block,
        stale_blocks,
        rel_tol,
        seed,
        tuple(weights),
    )
    if resume_state is None:
        D, PE = transition_matrices(instance, profile)
        prize = np.asarray([profile.prize_base * slab.priority for slab in instance.slabs])
        problem = _SaturatedPermutationProblem(instance, profile, weights, D, PE, prize)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=_SeededPermutationSampling(_greedy_permutation(instance, profile), seed),
            crossover=OrderCrossover(),
            mutation=_SeededMixedMutation(instance.num_slabs, seed + 1),
            eliminate_duplicates=False,
        )
        algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
        blocks = 0
        generations = 0
        stale = 0
        qualified_cost = float("inf")
        qualified_hv = float("-inf")
        ideal = scale = reference = None
        curve: list[dict] = []
    else:
        if tuple(resume_state["config_key"]) != config_key:
            raise ValueError("resume_state was created with a different NSGA-II configuration")
        problem = resume_state["problem"]
        algorithm = resume_state["algorithm"]
        blocks = int(resume_state["blocks"])
        generations = int(resume_state["generations"])
        stale = int(resume_state["stale"])
        qualified_cost = float(resume_state["qualified_cost"])
        qualified_hv = float(resume_state["qualified_hv"])
        ideal = resume_state["ideal"]
        scale = resume_state["scale"]
        reference = resume_state["reference"]
        curve = list(resume_state["curve"])

    saturated = False
    while blocks < max_blocks:
        for _ in range(generations_per_block):
            algorithm.next()
            generations += 1
        blocks += 1
        front = np.atleast_2d(algorithm.opt.get("F")).astype(np.float64)
        if ideal is None:
            ideal = front.min(axis=0)
            scale = np.maximum(front.max(axis=0) - ideal, 1.0)
            reference = np.full(front.shape[1], 1.1, dtype=np.float64)
        hv = _front_hypervolume(front, ideal, scale, reference)
        weighted_improved = (
            not np.isfinite(qualified_cost)
            or problem.best_cost < qualified_cost * (1.0 - rel_tol)
        )
        front_improved = (
            not np.isfinite(qualified_hv)
            or hv > qualified_hv * (1.0 + rel_tol)
        )
        if weighted_improved:
            qualified_cost = problem.best_cost
        if front_improved:
            qualified_hv = hv
        if weighted_improved or front_improved:
            stale = 0
            event = "improved"
        else:
            stale += 1
            event = "stale"
        curve.append(
            {
                "block": blocks,
                "generations": generations,
                "best_cost": float(problem.best_cost),
                "hypervolume": float(hv),
                "event": event,
            }
        )
        if stale >= stale_blocks:
            saturated = True
            break

    X = np.atleast_2d(algorithm.opt.get("X"))
    front = np.atleast_2d(algorithm.opt.get("F")).astype(np.float64)
    schedules = [
        split_permutation_dp(
            permutation,
            instance,
            profile,
            weights=weights,
            D=problem.D,
            PE=problem.PE,
        )
        for permutation in X
    ]
    state = {
        "config_key": config_key,
        "problem": problem,
        "algorithm": algorithm,
        "blocks": blocks,
        "generations": generations,
        "stale": stale,
        "qualified_cost": qualified_cost,
        "qualified_hv": qualified_hv,
        "ideal": ideal,
        "scale": scale,
        "reference": reference,
        "curve": list(curve),
    }
    return MultiObjectiveSearchResult(
        front=front,
        schedules=schedules,
        best_cost=float(problem.best_cost),
        best_schedule=problem.best_schedule,
        stop_reason="saturated" if saturated else "hard_cap",
        saturated=saturated,
        generations=generations,
        curve=list(curve),
        resume_state=state,
    )
