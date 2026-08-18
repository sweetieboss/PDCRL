"""Curriculum POMO trainer + Python-env greedy decode for the HPCVRP policy.

Training runs on the batched rollout (pdcrl.agents.pomo_batched): all POMO starts in
parallel as tensor ops, dense prize reward, shared-baseline REINFORCE. This module adds
the curriculum schedule (stage pools + objective-weight stages) and the Python-env greedy
decode used for solution reconstruction and batched/env parity checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pdcrl.agents.am_policy import AMPolicy, encode_instance
from pdcrl.agents.pomo_batched import _Tensors, batched_rollout
from pdcrl.baselines.heuristics import greedy_schedule
from pdcrl.envs.hpcvrp_env import HPCVRPEnv
from pdcrl.eval.saturation import SaturationConfig, SaturationTracker, select_proxy_starts
from pdcrl.rewards.costs import evaluate_full, full_weights
from pdcrl.utils.seeding import seed_everything


@dataclass
class POMOConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 2
    clip: float = 10.0
    lr: float = 3e-4
    lr_final: float | None = 1e-5  # cosine-anneal to this over the run; None -> constant lr
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    num_updates: int = 600
    num_starts: int | None = None  # None -> all slabs
    device: str = "cpu"
    rich_ctx: bool = False


@dataclass
class Stage:
    pool: str            # "small" | "mid" | "full"
    weights: tuple       # (w_transition, w_unscheduled, w_energy, w_rollchange)


@dataclass(frozen=True)
class FullConfirmationDecision:
    event: str
    terminal: bool
    learning_rate: float
    restore_state: object | None = None


class FullConfirmationController:
    """Gate checkpoints and LR decisions on full all-start evaluations."""

    def __init__(self, config: SaturationConfig, proxy_patience: int):
        if proxy_patience <= 0:
            raise ValueError("proxy_patience must be positive")
        self.tracker = SaturationTracker(config)
        self.proxy_patience = proxy_patience
        self.best_proxy_cost = float("inf")
        self.proxy_stale = 0
        self.best_full_cost = float("inf")
        self.best_state = None
        self.checkpoint_id = 0
        self.proxy_diverged = False

    def observe_proxy(self, cost: float) -> bool:
        """Return True when the current policy needs a full confirmation."""
        self.proxy_diverged = np.isfinite(self.best_proxy_cost) and cost > (
            self.best_proxy_cost * self.tracker.config.divergence_guard_factor
        )
        if self.proxy_diverged:
            return True
        threshold = self.best_proxy_cost * (1.0 - self.tracker.config.rel_tol)
        if not np.isfinite(self.best_proxy_cost) or cost < threshold:
            self.best_proxy_cost = cost
            self.proxy_stale = 0
            return True
        self.proxy_stale += 1
        if self.proxy_stale >= self.proxy_patience:
            self.proxy_stale = 0
            return True
        return False

    def _rollback_decision(self) -> FullConfirmationDecision:
        tracker_state = self.tracker.state
        tracker_state.learning_rate = max(
            self.tracker.config.lr_floor,
            tracker_state.learning_rate * self.tracker.config.lr_reduction_factor,
        )
        tracker_state.stale_observations = 0
        tracker_state.floor_observations = 0
        tracker_state.lr_reductions += 1
        return FullConfirmationDecision(
            event="rollback",
            terminal=False,
            learning_rate=tracker_state.learning_rate,
            restore_state=copy.deepcopy(self.best_state),
        )

    def rollback_from_proxy(self) -> FullConfirmationDecision:
        if not self.proxy_diverged:
            raise RuntimeError("the last proxy observation was not divergent")
        self.proxy_diverged = False
        return self._rollback_decision()

    def confirm_full(self, cost: float, state) -> FullConfirmationDecision:
        """Observe a full decode and retain only full-confirmed checkpoints."""
        decision = self.tracker.observe(cost)
        if decision.event == "improved":
            self.best_full_cost = cost
            self.best_state = copy.deepcopy(state)
            self.checkpoint_id += 1
        if decision.event == "divergence":
            return self._rollback_decision()
        return FullConfirmationDecision(
            event=decision.event,
            terminal=decision.terminal,
            learning_rate=decision.learning_rate,
        )


def build_stages(kind: str, pool_names=("small", "mid", "full")) -> list:
    """Curriculum stages over an ordered scale ladder of pools (last = target scale).
    Process-driven follows the rolling process's quality hierarchy (form good units + cover
    slabs -> then energy) while scaling up; ablations vary only scale ('size') or nothing
    ('none' = target pool only). S2 (unscheduled prize) stays on in every stage, else
    'schedule nothing' is degenerate-optimal."""
    pool_names = list(pool_names)
    if kind == "process_driven":
        return [
            Stage(p, (1, 1, 0, 1) if i == 0 and len(pool_names) > 1 else (1, 1, 1, 1))
            for i, p in enumerate(pool_names)
        ]
    if kind == "size":
        return [Stage(p, (1, 1, 1, 1)) for p in pool_names]
    if kind == "process_staged":
        # Same scale ladder as size; the reward-side process components are ramped in smoothly
        # per update by staged_component_weights (the trainer overrides these weights).
        return [Stage(p, (1, 1, 1, 1)) for p in pool_names]
    if kind == "none":
        return [Stage(pool_names[-1], (1, 1, 1, 1))]
    raise ValueError(f"unknown curriculum kind: {kind!r}")


def staged_component_weights(stage_index: int, num_stages: int, frac: float) -> tuple:
    """Smooth process-staged component weights (w_width, w_energy, w_gauge, w_hardness).

    Stage 0 (smallest pool) is pure width (learn the coffin unit-formation skill undiluted).
    Across stages 1..num_stages-1 the components ramp in smoothly in process order —
    energy, then gauge, then hardness — reaching full weight by the end of the target stage.
    ``frac`` is the within-stage wall-clock progress in [0, 1]. This is a smooth ramp aligned to
    pool-stage boundaries with a smooth transition rather than a hard switch."""
    if num_stages <= 1:
        return (1.0, 1.0, 1.0, 1.0)
    if stage_index <= 0:
        return (1.0, 0.0, 0.0, 0.0)
    u = ((stage_index - 1) + min(1.0, max(0.0, frac))) / (num_stages - 1)  # 0..1 over stages 1..end

    def seg(lo: float, hi: float) -> float:
        if u <= lo:
            return 0.0
        if u >= hi:
            return 1.0
        return (u - lo) / (hi - lo)

    return (1.0, seg(0.0, 1.0 / 3), seg(1.0 / 3, 2.0 / 3), seg(2.0 / 3, 1.0))


def _pomo_update(policy, opt, T, starts, weights, cfg) -> float:
    """One batched POMO update; returns the mean rollout cost."""
    logp, ent, cost, ret = batched_rollout(policy, T, starts, weights)
    adv = (ret - ret.mean()) / (ret.std() + 1e-6)
    loss = -(adv.detach() * logp).mean() - cfg.entropy_coef * ent.mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    opt.step()
    return float(cost.mean())


def train_pomo_curriculum(kind: str, pools: dict, profile, cfg: POMOConfig,
                          total_updates: int, seed: int = 0):
    """Train ONE policy through the curriculum's stages (transfer across stages).
    The total update budget is split evenly across stages so all curricula are compared fairly
    (equal update count; 'none' spends every update on the largest instances and thus gets the
    most compute, which is conservative for the curriculum claim).

    Returns (policy, hist) with hist = {"cost": per-update mean rollout cost,
    "stage_bounds": update index where each stage starts}."""
    seed_everything(seed)
    policy = AMPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip, rich_ctx=getattr(cfg, "rich_ctx", False)).to(cfg.device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    lr_sched = None if cfg.lr_final is None else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=cfg.lr_final)
    rng = np.random.default_rng(seed)
    stages = build_stages(kind)
    per_stage = max(1, total_updates // len(stages))
    tensor_cache: dict = {}
    hist = {"cost": [], "stage_bounds": []}
    for stage in stages:
        hist["stage_bounds"].append(len(hist["cost"]))
        insts = pools[stage.pool]
        w = tuple(float(x) for x in stage.weights)
        for _ in range(per_stage):
            inst = insts[int(rng.integers(len(insts)))]
            T = tensor_cache.get(id(inst))
            if T is None:
                T = _Tensors(inst, profile, cfg.device)
                tensor_cache[id(inst)] = T
            n = inst.num_slabs
            starts = list(range(n)) if cfg.num_starts is None else \
                list(rng.choice(n, size=min(cfg.num_starts, n), replace=False))
            hist["cost"].append(_pomo_update(policy, opt, T, starts, w, cfg))
            if lr_sched is not None:
                lr_sched.step()
    return policy, hist


def train_pomo_curriculum_timed(kind: str, pools: dict, profile, cfg: POMOConfig,
                                total_seconds: float, seed: int = 0,
                                stage_time_frac=None, pool_names=None,
                                smooth: bool = False, lr_restart: bool = True):
    """Wall-clock-budget curriculum training over an ordered scale ladder of pools.

    Stage time fractions default to frac_i ∝ 2^i (later/larger stages get more time; the
    ladder spends most of its budget at the target scale). Two fixes for curriculum-specific
    failure modes:
      - ``lr_restart``: the LR cosine restarts *per stage* instead of decaying globally —
        otherwise the curriculum reaches the target scale with an already-cold LR and gets
        less high-plasticity training there than 'none' does.
      - ``smooth``: no hard stage switches; the sampled pool crossfades along the ladder
        (linear ramp between the midpoints of adjacent stages), removing the distribution
        shock / catastrophic-forgetting cliff. For 'process_driven' the energy weight w2
        ramps 0 -> 1 across the first stage's time share. Smooth uses a global LR cosine.

    Returns (policy, hist) like train_pomo_curriculum."""
    import math
    import time

    seed_everything(seed)
    policy = AMPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip, rich_ctx=getattr(cfg, "rich_ctx", False)).to(cfg.device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    if pool_names is None:
        pool_names = ("small", "mid", "full") if "full" in pools else tuple(pools.keys())
    stages = build_stages(kind, pool_names)
    k = len(stages)
    if stage_time_frac is None:
        raw = [2.0 ** i for i in range(k)]
        stage_time_frac = tuple(x / sum(raw) for x in raw)
    assert len(stage_time_frac) == k
    tensor_cache: dict = {}
    hist = {"cost": [], "stage_bounds": []}

    def set_lr(t_frac):
        if cfg.lr_final is None:
            return
        lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * min(1.0, t_frac)))
        for grp in opt.param_groups:
            grp["lr"] = lr

    def update_on(pool, w):
        insts = pools[pool]
        inst = insts[int(rng.integers(len(insts)))]
        T = tensor_cache.get(id(inst))
        if T is None:
            T = _Tensors(inst, profile, cfg.device)
            tensor_cache[id(inst)] = T
        n = inst.num_slabs
        starts = list(range(n)) if cfg.num_starts is None else \
            list(rng.choice(n, size=min(cfg.num_starts, n), replace=False))
        hist["cost"].append(_pomo_update(policy, opt, T, starts, w, cfg))

    t0 = time.time()
    if smooth:
        # stage midpoints on the cumulative-time axis; crossfade between adjacent pools
        cum = np.cumsum([0.0, *stage_time_frac])
        mids = [(cum[i] + cum[i + 1]) / 2 for i in range(k)]
        hist["stage_bounds"].append(0)
        first = True
        while first or (time.time() - t0) < total_seconds:
            first = False
            t = min(1.0, (time.time() - t0) / total_seconds)
            set_lr(t)
            i = int(np.searchsorted(mids, t))          # between mids[i-1] and mids[i]
            if i <= 0:
                pool_i = 0
            elif i >= k:
                pool_i = k - 1
            else:
                ramp = (t - mids[i - 1]) / max(1e-9, mids[i] - mids[i - 1])
                pool_i = i if rng.random() < ramp else i - 1
            w2 = min(1.0, t / max(1e-9, stage_time_frac[0])) if kind == "process_driven" else 1.0
            update_on(stages[pool_i].pool, (1.0, 1.0, w2, 1.0))
    else:
        for si, (stage, frac) in enumerate(zip(stages, stage_time_frac)):
            hist["stage_bounds"].append(len(hist["cost"]))
            stage_start = time.time() - t0
            stage_end = stage_start + frac * total_seconds
            w = tuple(float(x) for x in stage.weights)
            first = True
            while first or (time.time() - t0) < min(stage_end, total_seconds):
                first = False
                el = time.time() - t0
                if lr_restart:
                    set_lr((el - stage_start) / max(1e-9, frac * total_seconds))
                else:
                    set_lr(el / total_seconds)
                update_on(stage.pool, w)
    return policy, hist


def train_pomo_curriculum_converge(kind: str, pools: dict, profile, cfg: POMOConfig, eval_inst,
                                   seed: int = 0, pool_names=None,
                                   intermediate_cap_s: float = 1800.0, target_cap_s: float = 36000.0,
                                   eval_every_s: float = 480.0, patience_evals: int = 6,
                                   plateau_rel: float = 0.001, lr_peak: float = 1.5e-4,
                                   eval_starts: int = 64, blend_s: float = 600.0,
                                   guard_factor: float = 1.5,
                                   floor_confirmations: int = 10,
                                   eval_every_updates: int | None = None,
                                   objective_weights=(1, 1, 1, 1),
                                   intermediate_rule: str = "stage_saturation",
                                   mastery_cost_ratio: float = 0.95,
                                   mastery_confirmations: int = 3,
                                   blend_updates: int | None = None):
    """Train stages to stagnation and gate target checkpoints on full all-start decode.

    Frequent seeded proxy evaluations may nominate a policy, but only a full all-start target
    evaluation can replace the returned checkpoint or drive a target-stage learning-rate
    transition. Wall-clock caps are abnormal guards and are reported as ``hard_cap``.
    """
    import time

    seed_everything(seed)
    ent0 = cfg.entropy_coef
    policy = AMPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip, rich_ctx=getattr(cfg, "rich_ctx", False)).to(cfg.device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    if pool_names is None:
        pool_names = ("small", "mid", "full") if "full" in pools else tuple(pools.keys())
    stages = build_stages(kind, pool_names)
    objective_weights = tuple(float(value) for value in full_weights(objective_weights))
    if intermediate_rule not in {"adequate", "stage_saturation", "mastery_confirm"}:
        raise ValueError(f"unknown intermediate rule: {intermediate_rule!r}")
    k = len(stages)
    tensor_cache: dict = {}
    hist = {
        "cost": [],
        "stage_bounds": [],
        "curve": [],
        "stage_stop_reason": [],
        "lr_events": [],
        "confirmation_events": [],
        "proxy_events": [],
        "rollbacks": 0,
    }

    def T_of(inst):
        T = tensor_cache.get(id(inst))
        if T is None:
            T = _Tensors(inst, profile, cfg.device)
            tensor_cache[id(inst)] = T
        return T

    def update_on(pool, w):
        insts = pools[pool]
        inst = insts[int(rng.integers(len(insts)))]
        T = T_of(inst)
        n = inst.num_slabs
        starts = list(range(n)) if cfg.num_starts is None else \
            list(rng.choice(n, size=min(cfg.num_starts, n), replace=False))
        hist["cost"].append(_pomo_update(policy, opt, T, starts, w, cfg))

    proxy_starts = select_proxy_starts(eval_inst, eval_starts, seed)
    hist["proxy_starts"] = proxy_starts

    def eval_cost(inst, starts):
        T = T_of(inst)
        with torch.no_grad():
            _, _, c, _ = batched_rollout(
                policy, T, starts, objective_weights, greedy=True
            )
        return float(c.min())

    def target_proxy_cost():
        return eval_cost(eval_inst, proxy_starts)

    def target_full_cost():
        return eval_cost(eval_inst, list(range(eval_inst.num_slabs)))

    def cpu_state():
        return {key: value.detach().to("cpu").clone() for key, value in policy.state_dict().items()}

    t0 = time.time()
    target_controller = None

    for si, stage in enumerate(stages):
        final = si == k - 1
        hist["stage_bounds"].append(len(hist["cost"]))
        cap = target_cap_s if final else intermediate_cap_s
        stage_eval_every = eval_every_s if final else eval_every_s / 4.0
        stage_eval_every_updates = None if eval_every_updates is None else (
            eval_every_updates if final else max(1, eval_every_updates // 4)
        )
        w = objective_weights
        if kind == "process_driven" and si == 0 and k > 1:
            w = (objective_weights[0], objective_weights[1], 0.0, objective_weights[3])
        stage_inst = max(pools[stage.pool], key=lambda i: i.num_slabs)
        peak = cfg.lr if kind == "none" else lr_peak
        floor = cfg.lr_final if cfg.lr_final is not None else peak
        prev_pool = stages[si - 1].pool if si > 0 else None
        stage_t0 = time.time()
        lr = peak
        cfg.entropy_coef = ent0
        hist["lr_events"].append((round(time.time() - t0, 1), lr))
        best = float("inf")
        stale = 0
        mastered = False
        mastery_stale = 0
        mastery_threshold = None
        next_stage_eval = stage_eval_every
        next_stage_eval_update = stage_eval_every_updates
        stage_updates = 0

        def reduce_intermediate_lr():
            nonlocal lr, opt, stale
            lr = max(lr * 0.5, floor)
            cfg.entropy_coef *= 0.5
            opt = torch.optim.Adam(policy.parameters(), lr=lr)
            stale = 0
            hist["lr_events"].append((round(time.time() - t0, 1), lr))

        if final:
            target_controller = FullConfirmationController(
                SaturationConfig(
                    rel_tol=plateau_rel,
                    patience=1,
                    floor_patience=floor_confirmations,
                    lr_initial=peak,
                    lr_floor=floor,
                    lr_reduction_factor=0.5,
                    divergence_guard_factor=guard_factor,
                ),
                proxy_patience=patience_evals,
            )
            proxy_cost = target_proxy_cost()
            target_controller.observe_proxy(proxy_cost)
            full_cost = target_full_cost()
            initial = target_controller.confirm_full(full_cost, cpu_state())
            hist["confirmation_events"].append(
                {
                    "elapsed_seconds": round(time.time() - t0, 3),
                    "proxy_cost": proxy_cost,
                    "full_cost": full_cost,
                    "event": initial.event,
                    "learning_rate": initial.learning_rate,
                    "checkpoint_id": target_controller.checkpoint_id,
                }
            )
            hist["curve"].append(
                (round(time.time() - t0, 3), target_controller.best_full_cost)
            )
        elif intermediate_rule == "mastery_confirm":
            greedy_reference = greedy_schedule(stage_inst, profile, weights=w)
            greedy_score = evaluate_full(greedy_reference, stage_inst, profile)
            greedy_components = (
                greedy_score["transition_cost"],
                greedy_score["unscheduled_cost"],
                greedy_score["energy_cost"],
                greedy_score["rollchange_cost"],
            )
            mastery_threshold = mastery_cost_ratio * float(
                np.dot(full_weights(w), greedy_components)
            )

        while True:
            el = time.time() - stage_t0
            if el >= cap:
                hist["stage_stop_reason"].append("hard_cap")
                break
            for grp in opt.param_groups:
                grp["lr"] = lr
            pool = stage.pool
            if prev_pool is not None:
                if blend_updates is None:
                    blend_active = el < blend_s
                    blend_progress = el / max(1e-9, blend_s)
                else:
                    blend_active = stage_updates < blend_updates
                    blend_progress = stage_updates / max(1, blend_updates)
                if blend_active and rng.random() >= blend_progress:
                    pool = prev_pool
            update_on(pool, w)
            stage_updates += 1
            el = time.time() - stage_t0
            time_due = stage_eval_every_updates is None and el >= next_stage_eval
            update_due = (
                stage_eval_every_updates is not None
                and stage_updates >= next_stage_eval_update
            )
            if time_due or update_due:
                if time_due:
                    next_stage_eval += stage_eval_every
                else:
                    next_stage_eval_update += stage_eval_every_updates
                if not final:
                    stage_starts = select_proxy_starts(
                        stage_inst, eval_starts, seed + 1009 * (si + 1)
                    )
                    sc = eval_cost(stage_inst, stage_starts)
                    blend_complete = (
                        prev_pool is None
                        or (blend_updates is not None and stage_updates >= blend_updates)
                        or (blend_updates is None and el >= blend_s)
                    )
                    if blend_complete:
                        qualifying = sc < best * (1.0 - plateau_rel)
                        if qualifying:
                            best = sc
                            stale = 0
                        else:
                            stale += 1
                        if intermediate_rule == "adequate":
                            if stale >= patience_evals:
                                hist["stage_stop_reason"].append("adequate_stagnation")
                                break
                        elif intermediate_rule == "stage_saturation":
                            if stale >= patience_evals:
                                if lr <= floor * 1.001:
                                    hist["stage_stop_reason"].append("saturated")
                                    break
                                reduce_intermediate_lr()
                        else:
                            if sc <= mastery_threshold:
                                if not mastered:
                                    mastered = True
                                    mastery_stale = 0
                                elif qualifying:
                                    mastery_stale = 0
                                else:
                                    mastery_stale += 1
                            if mastered and mastery_stale >= mastery_confirmations:
                                hist["stage_stop_reason"].append("mastery_confirmed")
                                break
                    continue

                target_blend_complete = (
                    prev_pool is None
                    or (blend_updates is not None and stage_updates >= blend_updates)
                    or (blend_updates is None and el >= blend_s)
                )
                if not target_blend_complete:
                    hist["proxy_events"].append(
                        {
                            "elapsed_seconds": round(time.time() - t0, 3),
                            "cost": None,
                            "needs_full": False,
                            "diverged": False,
                            "in_blend": True,
                        }
                    )
                    continue
                proxy_cost = target_proxy_cost()
                needs_full = target_controller.observe_proxy(proxy_cost)
                hist["proxy_events"].append(
                    {
                        "elapsed_seconds": round(time.time() - t0, 3),
                        "cost": proxy_cost,
                        "needs_full": needs_full,
                        "diverged": target_controller.proxy_diverged,
                        "in_blend": False,
                    }
                )
                if target_controller.proxy_diverged:
                    decision = target_controller.rollback_from_proxy()
                    policy.load_state_dict(decision.restore_state)
                    hist["rollbacks"] += 1
                    cfg.entropy_coef *= 0.5
                    lr = decision.learning_rate
                    opt = torch.optim.Adam(policy.parameters(), lr=lr)
                    hist["lr_events"].append((round(time.time() - t0, 1), lr))
                    continue
                if not needs_full:
                    continue

                full_cost = target_full_cost()
                decision = target_controller.confirm_full(full_cost, cpu_state())
                hist["confirmation_events"].append(
                    {
                        "elapsed_seconds": round(time.time() - t0, 3),
                        "proxy_cost": proxy_cost,
                        "full_cost": full_cost,
                        "event": decision.event,
                        "learning_rate": decision.learning_rate,
                        "checkpoint_id": target_controller.checkpoint_id,
                    }
                )
                hist["curve"].append(
                    (round(time.time() - t0, 3), target_controller.best_full_cost)
                )
                if decision.event == "rollback":
                    policy.load_state_dict(decision.restore_state)
                    hist["rollbacks"] += 1
                if decision.event in ("rollback", "reduce_lr"):
                    cfg.entropy_coef *= 0.5
                    lr = decision.learning_rate
                    opt = torch.optim.Adam(policy.parameters(), lr=lr)
                    hist["lr_events"].append((round(time.time() - t0, 1), lr))
                if decision.terminal:
                    hist["stage_stop_reason"].append("saturated")
                    break

    if target_controller is None or target_controller.best_state is None:
        raise RuntimeError("target stage completed without a full-confirmed checkpoint")
    policy.load_state_dict(target_controller.best_state)
    cfg.entropy_coef = ent0
    hist["best_target"] = target_controller.best_full_cost
    hist["best_target_full"] = target_controller.best_full_cost
    hist["saturated"] = hist["stage_stop_reason"][-1] == "saturated"
    return policy, hist


def greedy_decode(policy, instance, profile, weights=None):
    """Multi-start greedy (argmax) decode in the Python env; return the best objective and Schedule."""
    env = HPCVRPEnv(instance, profile, objective_weights=weights)
    best_cost = float("inf")
    best_sched = None
    with torch.no_grad():
        enc = encode_instance(policy, instance, profile)
        for s in range(instance.num_slabs):
            obs, _ = env.reset(start_slab=s)
            terminated = False
            info = {}
            while not terminated:
                logits = policy.action_logits(enc, obs)
                obs, _, terminated, _, info = env.step(int(logits.argmax()))
            c = info["cost"]["objective_value"]
            if c < best_cost:
                best_cost = c
                best_sched = info["schedule"]
    return best_cost, best_sched
