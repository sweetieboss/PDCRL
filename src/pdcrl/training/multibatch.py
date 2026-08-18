"""Multi-batch trainers used by the paper's PDCRL and scale-only protocols."""

from __future__ import annotations

from collections import Counter
import time

import numpy as np
import torch

from pdcrl.agents.am_policy import AMPolicy
from pdcrl.agents.pomo import POMOConfig, _pomo_update, build_stages
from pdcrl.agents.pomo_batched import _Tensors, batched_rollout
from pdcrl.utils.seeding import seed_everything

from pdcrl.training.objective_weights import continuous_stage_weights


PDCRL_CONTROLLER_SOURCE = "paper PDCRL plateau controller"
SCALE_ONLY_REFERENCE = "paper scale-only fixed-stage controller"


def _cpu_state(policy):
    return {key: value.detach().to("cpu").clone() for key, value in policy.state_dict().items()}


def _validation_tools(policy, profile, cfg, references):
    cache = {}

    def tensors_for(instance):
        if id(instance) not in cache:
            cache[id(instance)] = _Tensors(instance, profile, cfg.device)
        return cache[id(instance)]

    def eval_cost(instance, eval_starts):
        tensors = tensors_for(instance)
        tensors.reset_s1()
        starts = list(range(min(eval_starts, instance.num_slabs)))
        with torch.no_grad():
            _, _, costs, _ = batched_rollout(
                policy, tensors, starts, (1, 1, 1, 1), greedy=True
            )
        return float(costs.min())

    def aggregate(instances, eval_starts):
        rows = []
        for instance in instances:
            cost = eval_cost(instance, eval_starts)
            reference = float(references[instance.name])
            if reference <= 0.0:
                raise ValueError(f"non-positive validation reference for {instance.name}")
            rows.append({"instance": instance.name, "cost": cost, "reference": reference,
                         "normalized_cost": cost / reference})
        if not rows:
            raise ValueError("validation aggregation requires at least one instance")
        return float(np.median([row["normalized_cost"] for row in rows])), rows

    return tensors_for, aggregate


def train_pdcrl_multibatch(
    kind,
    pools,
    profile,
    cfg: POMOConfig,
    *,
    seed,
    pool_names,
    stage_validation_pools,
    target_validation_instances,
    validation_reference_costs,
    intermediate_cap_s=1800.0,
    target_cap_s=36000.0,
    eval_every_s=480.0,
    patience_evals=6,
    plateau_rel=0.005,
    lr_peak=1.5e-4,
    eval_starts=64,
    blend_s=600.0,
    guard_factor=1.5,
    objective_weight_anchors=None,
    target_validation_scope="six-formal-scale-validation-pool",
):
    """Train the paper PDCRL controller with multi-instance curriculum pools."""
    if kind not in {"process_driven", "none"}:
        raise ValueError("offline historical trainer supports process_driven or none")
    seed_everything(seed)
    initial_entropy = cfg.entropy_coef
    policy = AMPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip, rich_ctx=cfg.rich_ctx).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    stages = build_stages(kind, pool_names)
    if objective_weight_anchors is not None and len(objective_weight_anchors) != len(stages):
        raise ValueError("objective_weight_anchors must contain exactly one anchor per stage")
    cache = {}
    updates = Counter()
    history = {
        "cost": [], "stage_bounds": [], "curve": [], "stage_stop_reason": [],
        "protocol_implementation": "pdcrl-plateau-controller",
        "protocol_source": PDCRL_CONTROLLER_SOURCE,
        "input_extension": "multi-instance-pool-sampling",
        "lr_events": [], "entropy_events": [], "rollbacks": 0,
        "restored_best": False, "instance_updates": {}, "total_rollouts": 0,
        "stage_elapsed_seconds": [], "validation_events": [],
        "validation_aggregation": {
            "normalization": "greedy_objective", "statistic": "median",
            "stage_scope": "current-stage-validation-pool",
            "target_scope": str(target_validation_scope),
        },
    }
    if objective_weight_anchors is not None:
        history["objective_weight_schedule"] = {
            "type": "linear-during-existing-pool-crossfade",
            "blend_seconds": float(blend_s),
            "anchors": [[float(value) for value in anchor] for anchor in objective_weight_anchors],
        }
        history["weight_trace"] = []

    def tensors_for(instance):
        if id(instance) not in cache:
            cache[id(instance)] = _Tensors(instance, profile, cfg.device)
        return cache[id(instance)]

    def update_on(pool, weights):
        instance = pools[pool][int(rng.integers(len(pools[pool])))]
        starts = (
            list(range(instance.num_slabs))
            if cfg.num_starts is None
            else list(rng.choice(instance.num_slabs, min(cfg.num_starts, instance.num_slabs), replace=False))
        )
        history["cost"].append(_pomo_update(policy, optimizer, tensors_for(instance), starts, weights, cfg))
        history["total_rollouts"] += len(starts)
        updates[instance.name] += 1
        return instance

    def aggregate(instances):
        rows = []
        for instance in instances:
            tensors = tensors_for(instance)
            tensors.reset_s1()
            starts = list(range(min(eval_starts, instance.num_slabs)))
            with torch.no_grad():
                _, _, costs, _ = batched_rollout(policy, tensors, starts, (1, 1, 1, 1), greedy=True)
            cost = float(costs.min())
            reference = float(validation_reference_costs[instance.name])
            rows.append({"instance": instance.name, "cost": cost, "reference": reference,
                         "normalized_cost": cost / reference})
        return float(np.median([row["normalized_cost"] for row in rows])), rows

    started = time.time()
    next_curve_eval = eval_every_s
    best_target = float("inf")
    best_state = None
    for stage_index, stage in enumerate(stages):
        final = stage_index == len(stages) - 1
        history["stage_bounds"].append(len(history["cost"]))
        cap = target_cap_s if final else intermediate_cap_s
        stage_eval_every = eval_every_s if final else eval_every_s / 4.0
        base_weights = tuple(float(value) for value in stage.weights)
        previous_pool = stages[stage_index - 1].pool if stage_index else None
        peak = cfg.lr if kind == "none" else lr_peak
        floor = cfg.lr_final if cfg.lr_final is not None else peak
        stage_started = time.time()
        learning_rate = peak
        cfg.entropy_coef = initial_entropy
        history["lr_events"].append([round(time.time() - started, 3), learning_rate])
        history["entropy_events"].append([round(time.time() - started, 3), cfg.entropy_coef])
        best_stage = float("inf")
        stale = 0
        next_stage_eval = stage_eval_every

        def reduce_lr():
            nonlocal learning_rate, optimizer, stale
            learning_rate = max(learning_rate * 0.5, floor)
            cfg.entropy_coef *= 0.5
            optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
            stale = 0
            history["lr_events"].append([round(time.time() - started, 3), learning_rate])
            history["entropy_events"].append([round(time.time() - started, 3), cfg.entropy_coef])

        while True:
            elapsed = time.time() - stage_started
            if elapsed >= cap:
                history["stage_stop_reason"].append("cap")
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            sampled_pool = stage.pool
            if previous_pool is not None and elapsed < blend_s and rng.random() >= elapsed / blend_s:
                sampled_pool = previous_pool
            weights = (
                continuous_stage_weights(objective_weight_anchors, stage_index, elapsed, blend_s)
                if objective_weight_anchors is not None else base_weights
            )
            sampled = update_on(sampled_pool, weights)
            if objective_weight_anchors is not None:
                history["weight_trace"].append({
                    "update": len(history["cost"]) - 1, "stage": stage_index,
                    "elapsed_stage_seconds": elapsed, "sampled_pool": sampled_pool,
                    "sampled_instance": sampled.name, "weights": list(weights),
                })
            elapsed = time.time() - stage_started
            if elapsed < next_stage_eval:
                continue
            next_stage_eval += stage_eval_every
            stage_cost, stage_rows = aggregate(stage_validation_pools[stage.pool])
            target_cost, target_rows = aggregate(target_validation_instances)
            history["validation_events"].append({
                "elapsed_seconds": round(time.time() - started, 3), "stage": stage_index,
                "stage_cost": stage_cost, "target_cost": target_cost,
                "stage_values": stage_rows, "target_values": target_rows,
            })
            if target_cost < best_target:
                best_target, best_state = target_cost, _cpu_state(policy)
            if time.time() - started >= next_curve_eval:
                next_curve_eval += eval_every_s
                history["curve"].append([round(time.time() - started, 3), target_cost])
            if final and best_state is not None and target_cost > guard_factor * best_target:
                policy.load_state_dict(best_state)
                history["rollbacks"] += 1
                if learning_rate <= floor * 1.001:
                    history["stage_stop_reason"].append("guard_floor")
                    break
                reduce_lr()
                continue
            if previous_pool is None or elapsed >= blend_s:
                if stage_cost < best_stage * (1.0 - plateau_rel):
                    best_stage, stale = stage_cost, 0
                else:
                    stale += 1
                    if stale >= patience_evals:
                        if learning_rate <= floor * 1.001:
                            history["stage_stop_reason"].append("plateau")
                            break
                        reduce_lr()
        history["stage_elapsed_seconds"].append(time.time() - stage_started)

    if best_state is not None:
        policy.load_state_dict(best_state)
        history["restored_best"] = True
    cfg.entropy_coef = initial_entropy
    history["best_target"] = best_target
    final_target, final_rows = aggregate(target_validation_instances)
    history["curve"].append([round(time.time() - started, 3), final_target])
    history["final_validation_values"] = final_rows
    history["instance_updates"] = dict(sorted(updates.items()))
    return policy, history


def train_scale_only_multibatch(
    pools,
    profile,
    cfg: POMOConfig,
    *,
    seed,
    pool_names,
    stage_validation_pools,
    target_validation_instances,
    validation_reference_costs,
    total_budget_s,
    eval_every_s=480.0,
    eval_starts=64,
):
    """Train the paper scale-only hard-stage controller with within-pool sampling."""
    stage_count = len(pool_names)
    if stage_count not in {4, 5}:
        raise ValueError("scale-only multibatch training requires four or five stages")
    if abs(cfg.lr - 1.0e-4) > 1e-12:
        raise ValueError("scale-only multibatch training requires constant lr=1e-4")
    seed_everything(seed)
    policy = AMPolicy(cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.clip, rich_ctx=cfg.rich_ctx).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    cache = {}
    updates = Counter()
    cap = float(total_budget_s) / stage_count
    history = {
        "cost": [], "stage_bounds": [], "stage_caps_s": [cap] * stage_count,
        "stage_stop_reason": [], "stage_pool_updates": [], "stage_elapsed_seconds": [],
        "curve": [], "validation_events": [], "instance_updates": {}, "total_rollouts": 0,
        "protocol_implementation": "classic-staged", "variant_id": "scale-only-multibatch",
        "reference_results": SCALE_ONLY_REFERENCE, "stabilization": False,
        "restored_best": False, "rollbacks": 0, "blend_s": 0.0,
        "adaptive_stage_termination": False, "lr_restart_per_stage": False,
        "stage_alloc": "equal", "lr_schedule": "constant", "lr_peak_used": cfg.lr,
        "objective_weights": [1.0, 1.0, 1.0, 1.0],
        "validation_aggregation": {"role": "diagnostic-only", "normalization": "greedy_objective",
                                   "statistic": "median"},
    }

    def tensors_for(instance):
        if id(instance) not in cache:
            cache[id(instance)] = _Tensors(instance, profile, cfg.device)
        return cache[id(instance)]

    def aggregate(instances):
        rows = []
        for instance in instances:
            starts = list(range(min(eval_starts, instance.num_slabs)))
            with torch.no_grad():
                _, _, costs, _ = batched_rollout(policy, tensors_for(instance), starts,
                                                 (1, 1, 1, 1), greedy=True)
            cost = float(costs.min())
            reference = float(validation_reference_costs[instance.name])
            rows.append({"instance": instance.name, "cost": cost, "reference": reference,
                         "normalized_cost": cost / reference})
        return float(np.median([row["normalized_cost"] for row in rows])), rows

    started = time.time()
    for stage_index, pool in enumerate(pool_names):
        history["stage_bounds"].append(len(history["cost"]))
        history["stage_pool_updates"].append({pool: 0})
        stage_started = time.time()
        final_stage = stage_index == stage_count - 1
        next_eval = eval_every_s if final_stage else eval_every_s / 4.0
        while time.time() - stage_started < cap:
            instance = pools[pool][int(rng.integers(len(pools[pool])))]
            starts = (
                list(range(instance.num_slabs)) if cfg.num_starts is None
                else list(rng.choice(instance.num_slabs, min(cfg.num_starts, instance.num_slabs), replace=False))
            )
            history["cost"].append(_pomo_update(policy, optimizer, tensors_for(instance), starts,
                                                (1, 1, 1, 1), cfg))
            history["total_rollouts"] += len(starts)
            history["stage_pool_updates"][-1][pool] += 1
            updates[instance.name] += 1
            elapsed = time.time() - stage_started
            if elapsed >= next_eval:
                next_eval += eval_every_s if final_stage else eval_every_s / 4.0
                stage_cost, stage_rows = aggregate(stage_validation_pools[pool])
                target_cost, target_rows = aggregate(target_validation_instances)
                history["validation_events"].append({
                    "stage": stage_index, "elapsed_seconds": round(time.time() - started, 3),
                    "stage_cost": stage_cost, "target_cost": target_cost,
                    "stage_values": stage_rows, "target_values": target_rows,
                })
        history["stage_stop_reason"].append("cap")
        history["stage_elapsed_seconds"].append(time.time() - stage_started)
    final_target, rows = aggregate(target_validation_instances)
    history["best_target"] = final_target
    history["curve"].append([round(time.time() - started, 3), final_target])
    history["final_validation_values"] = rows
    history["instance_updates"] = dict(sorted(updates.items()))
    return policy, history
