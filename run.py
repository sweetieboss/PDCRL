#!/usr/bin/env python
"""Single command-line entry point for the PDCRL paper release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdcrl.agents.am_policy import AMPolicy  # noqa: E402
from pdcrl.agents.pomo import POMOConfig  # noqa: E402
from pdcrl.agents.pomo_batched import greedy_decode_batched  # noqa: E402
from pdcrl.baselines.alns import alns_saturated  # noqa: E402
from pdcrl.baselines.heuristics import greedy_schedule  # noqa: E402
from pdcrl.baselines.local_search import local_search_with_stats  # noqa: E402
from pdcrl.baselines.nsga2 import grasp, nsga2_pareto  # noqa: E402
from pdcrl.data.benchmark import sha256_file, validate_manifest  # noqa: E402
from pdcrl.data.loader import load_instance  # noqa: E402
from pdcrl.data.release_data import (  # noqa: E402
    reproduce_release_data,
    validate_release_data_recipe,
)
from pdcrl.eval.records import build_schedule_metrics  # noqa: E402
from pdcrl.process import load_process_profile  # noqa: E402
from pdcrl.training.model_state import state_dict_sha256  # noqa: E402
from pdcrl.training.multibatch import (  # noqa: E402
    train_pdcrl_multibatch,
    train_scale_only_multibatch,
)
from pdcrl.training.pomo_baseline import (  # noqa: E402
    PublishedPOMOPolicy,
    train_published_pomo_fixed_budget,
)
from pdcrl.training.specialist_data import (  # noqa: E402
    rows_for_stage,
    validate_specialist_manifest,
)
from pdcrl.utils.seeding import seed_everything  # noqa: E402


CONFIG_PATH = ROOT / "configs/paper_experiments.yaml"
LEARNED_METHODS = ("pdcrl", "pomo", "direct", "scale_only")
BASELINES = ("greedy", "grasp", "nsga2", "alns")


def _load_config() -> dict:
    return OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_value(mapping: dict, key: int):
    value = mapping.get(key, mapping.get(str(key)))
    if value is None:
        raise KeyError(f"missing p{key} configuration")
    return value


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temporary, path)


def _write_schedule(path: Path, schedule) -> None:
    _write_json(
        path,
        {
            "num_slabs": schedule.num_slabs,
            "units": schedule.units,
            "unscheduled": sorted(schedule.unscheduled),
        },
    )


def _manifest_paths(config: dict) -> tuple[Path, Path, Path]:
    return (
        ROOT / config["training_manifest"],
        ROOT / config["validation_manifest"],
        ROOT / config["test_manifest"],
    )


def _scale_only_prefix_errors(config: dict, manifest: dict, root: Path) -> list[str]:
    """Validate the Scale-Only pre-target instances stored with the training split."""
    rows = manifest.get("scale_only_prefix_batches", [])
    configured = config["methods"]["scale_only"]["prefix_stage_instance_names"]
    expected_names = {
        str(name)
        for stages in configured.values()
        for stage in stages
        for name in stage
    }
    errors: list[str] = []
    if len(rows) != len(expected_names):
        errors.append(
            f"expected {len(expected_names)} Scale-Only prefix records, found {len(rows)}"
        )
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name", "<unnamed>"))
        if name in seen:
            errors.append(f"duplicate Scale-Only prefix name: {name}")
        seen.add(name)
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{name}: path must remain within the training root")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"{name}: missing file {relative.as_posix()}")
            continue
        if _sha256(path) != row.get("sha256"):
            errors.append(f"{name}: sha256 mismatch")
        try:
            instance = load_instance(name, root)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{name}: unreadable CSV: {exc}")
            continue
        if instance.num_slabs != int(row.get("scale", -1)):
            errors.append(f"{name}: row count does not match scale")
    if seen != expected_names:
        errors.append("Scale-Only prefix manifest and configuration names differ")
    return errors


def verify_release() -> dict[str, int]:
    """Validate the frozen data split, process profile, and checkpoint inventory."""
    config = _load_config()
    training_path, validation_path, test_path = _manifest_paths(config)
    training = _load_json(training_path)
    validation = _load_json(validation_path)
    test = _load_json(test_path)
    recipe_errors = validate_release_data_recipe(
        training,
        validation,
        test,
        ROOT / config["process"],
    )
    training_errors = validate_specialist_manifest(training, training_path.parent)
    prefix_errors = _scale_only_prefix_errors(config, training, training_path.parent)
    validation_errors = validate_specialist_manifest(validation, validation_path.parent)
    test_errors = validate_manifest(test, test_path.parent)
    if recipe_errors or training_errors or prefix_errors or validation_errors or test_errors:
        raise RuntimeError(
            "release data validation failed:\n"
            + "\n".join(
                recipe_errors
                + training_errors
                + prefix_errors
                + validation_errors
                + test_errors
            )
        )
    if {row["role"] for row in training["batches"]} != {"curriculum"}:
        raise RuntimeError("training manifest contains non-training records")
    if {row["role"] for row in validation["batches"]} != {"development"}:
        raise RuntimeError("validation manifest contains non-validation records")
    prefix_rows = training["scale_only_prefix_batches"]
    train_hashes = {
        row["sha256"] for row in training["batches"] + prefix_rows
    }
    validation_hashes = {row["sha256"] for row in validation["batches"]}
    test_hashes = {row["sha256"] for row in test["batches"]}
    if (
        train_hashes & validation_hashes
        or train_hashes & test_hashes
        or validation_hashes & test_hashes
    ):
        raise RuntimeError("training/validation/test content leakage detected")

    derived_scales = sum(
        len(group)
        for scale in config["scales"]
        for group in _mapping_value(config["stage_scale_groups"], int(scale))
    )
    expected_training = derived_scales * int(config["training_parents_per_scale"])
    expected_validation = derived_scales * int(config["development_parents_per_scale"])
    if len(training["batches"]) != expected_training:
        raise RuntimeError("unexpected training-grid size")
    if len(validation["batches"]) != expected_validation:
        raise RuntimeError("unexpected validation-grid size")

    profile = load_process_profile(ROOT / config["process"])
    if (profile.capacity_min_m, profile.capacity_max_m, profile.roll_change_cost) != (
        10000,
        50000,
        1000.0,
    ):
        raise RuntimeError("process profile differs from the paper")

    checkpoint_manifest = _load_json(ROOT / "checkpoints/manifest.json")
    seen = set()
    for row in checkpoint_manifest["checkpoints"]:
        identity = (row["method"], int(row["scale"]), int(row["seed"]))
        if identity in seen:
            raise RuntimeError(f"duplicate checkpoint identity: {identity}")
        seen.add(identity)
        path = ROOT / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {row['path']}")

    expected_test = len(config["scales"]) * int(config["test_batches_per_scale"])
    if len(test["batches"]) != expected_test:
        raise RuntimeError("unexpected test-grid size")
    return {
        "training_records": len(training["batches"]) + len(prefix_rows),
        "scale_only_prefix_records": len(prefix_rows),
        "validation_records": len(validation["batches"]),
        "test_records": len(test["batches"]),
        "checkpoints": len(checkpoint_manifest["checkpoints"]),
    }


def _test_case(config: dict, scale: int, replicate: int):
    if scale not in {int(value) for value in config["scales"]}:
        raise ValueError(f"unsupported scale p{scale}")
    if not 0 <= replicate < int(config["test_batches_per_scale"]):
        raise ValueError("replicate must be in [0, 9]")
    _, _, test_path = _manifest_paths(config)
    manifest = _load_json(test_path)
    name = f"target_n{scale:04d}_r{replicate:02d}"
    row = next((item for item in manifest["batches"] if item["name"] == name), None)
    if row is None:
        raise ValueError(f"test batch not found: {name}")
    return row, load_instance(name, test_path.parent)


def _validate_method_identity(config: dict, method: str, scale: int, seed: int) -> None:
    settings = config["methods"][method]
    if scale not in {int(value) for value in settings["scales"]}:
        raise ValueError(f"{method} was not reported at p{scale}")
    if seed not in {int(value) for value in settings["seeds"]}:
        raise ValueError(f"{method} p{scale} seed {seed} is outside the paper protocol")


def _checkpoint(method: str, scale: int, seed: int) -> Path:
    return ROOT / "checkpoints" / method / f"p{scale}_seed{seed}.pt"


def _policy(config: dict, method: str, device: str):
    if method == "pomo":
        settings = config["methods"]["pomo"]
        return PublishedPOMOPolicy(
            d_model=int(settings["d_model"]),
            n_heads=int(settings["n_heads"]),
            n_layers=int(settings["n_layers"]),
            clip=float(config["model"]["clip"]),
        ).to(device)
    model = config["model"]
    return AMPolicy(
        int(model["d_model"]),
        int(model["n_heads"]),
        int(model["n_layers"]),
        float(model["clip"]),
        rich_ctx=bool(model["rich_ctx"]),
    ).to(device)


def evaluate_checkpoint(
    *,
    method: str,
    scale: int,
    replicate: int,
    seed: int,
    device: str,
    raw_only: bool,
    output_root: Path,
) -> dict:
    config = _load_config()
    _validate_method_identity(config, method, scale, seed)
    device = _device(device)
    row, instance = _test_case(config, scale, replicate)
    profile = load_process_profile(ROOT / config["process"])
    policy = _policy(config, method, device)
    checkpoint = _checkpoint(method, scale, seed)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    before = state_dict_sha256(policy.state_dict())

    seed_everything(seed)
    _sync(device)
    started = time.perf_counter()
    decoded_cost, raw_schedule = greedy_decode_batched(
        policy,
        instance,
        profile,
        weights=tuple(config["objective_weights"]),
    )
    _sync(device)
    decode_seconds = time.perf_counter() - started
    after = state_dict_sha256(policy.state_dict())
    if after != before:
        raise RuntimeError("checkpoint state changed during evaluation")

    raw_metrics = build_schedule_metrics(
        raw_schedule, instance, profile, tuple(config["objective_weights"])
    )
    tolerance = max(1e-5, abs(float(decoded_cost)) * 1e-8)
    if not raw_metrics["feasible"] or not raw_schedule.is_partition():
        raise RuntimeError("decoder produced an infeasible schedule")
    if abs(float(decoded_cost) - float(raw_metrics["objective_value"])) > tolerance:
        raise RuntimeError("decoder cost and independent recomputation disagree")

    final_schedule = raw_schedule
    local_search_seconds = 0.0
    local_search = None
    if method == "pdcrl" and not raw_only:
        local_search = local_search_with_stats(
            raw_schedule,
            instance,
            profile,
            weights=tuple(config["objective_weights"]),
            max_sec=float(config["local_search"]["hard_cap_seconds"]),
        )
        final_schedule = local_search.schedule
        local_search_seconds = float(local_search.elapsed_seconds)
    final_metrics = build_schedule_metrics(
        final_schedule, instance, profile, tuple(config["objective_weights"])
    )
    if not final_metrics["feasible"] or not final_schedule.is_partition():
        raise RuntimeError("final schedule is infeasible")

    run_dir = output_root / "evaluation" / method / f"p{scale}" / row["name"] / f"seed_{seed}"
    _write_schedule(run_dir / "raw_schedule.json", raw_schedule)
    _write_schedule(run_dir / "schedule.json", final_schedule)
    record = {
        "protocol_version": config["protocol_version"],
        "method": method,
        "scale": scale,
        "batch_replicate": replicate,
        "model_seed": seed,
        "instance": row["name"],
        "data_seed": int(row["generator_seed"]),
        "objective_weights": config["objective_weights"],
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_state_unchanged": True,
        "zero_shot": True,
        "target_training_or_adaptation_seconds": 0.0,
        "raw_metrics": raw_metrics,
        "final_metrics": final_metrics,
        "runtime": {
            "decode_seconds": decode_seconds,
            "local_search_seconds": local_search_seconds,
            "total_seconds": decode_seconds + local_search_seconds,
        },
        "local_search": None
        if local_search is None
        else {
            "stop_reason": local_search.stop_reason,
            "passes": local_search.passes,
            "accepted_moves": local_search.accepted_moves,
        },
    }
    _write_json(run_dir / "record.json", record)
    return record


def _atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    os.replace(temporary, path)


def run_baseline(
    *, method: str, scale: int, replicate: int, output_root: Path
) -> dict:
    config = _load_config()
    row, instance = _test_case(config, scale, replicate)
    profile = load_process_profile(ROOT / config["process"])
    weights = tuple(config["objective_weights"])
    seed = replicate
    settings = config["baselines"]
    run_dir = output_root / "baselines" / method / f"p{scale}" / row["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    counts: dict[str, int] = {}
    stop_reason = "completed"

    if method == "greedy":
        schedule = greedy_schedule(instance, profile, weights=weights)
        counts = {"constructions": 1}
    elif method == "grasp":
        grasp_cfg = settings["grasp"]
        _, schedule = grasp(
            instance,
            profile,
            n_starts=int(grasp_cfg["starts"]),
            rcl_k=int(grasp_cfg["restricted_candidate_list"]),
            seed=seed,
            weights=weights,
            ls_sec_per_start=float(grasp_cfg["local_search_seconds_per_start"]),
        )
        counts = {"restarts": int(grasp_cfg["starts"])}
    elif method == "nsga2":
        nsga_cfg = settings["nsga2"]
        front, _, (_, schedule) = nsga2_pareto(
            instance,
            profile,
            pop_size=int(nsga_cfg["population"]),
            n_gen=int(nsga_cfg["generations"]),
            seed=seed,
            weights=weights,
        )
        counts = {
            "population": int(nsga_cfg["population"]),
            "generations": int(nsga_cfg["generations"]),
            "front_size": int(len(front)),
        }
    elif method == "alns":
        alns_cfg = settings["alns"]
        resume_path = run_dir / "resume.pkl"
        resume_state = None
        if resume_path.is_file():
            with resume_path.open("rb") as handle:
                resume_state = pickle.load(handle)

        def checkpoint(state: dict) -> None:
            _atomic_pickle(resume_path, state)

        result = alns_saturated(
            instance,
            profile,
            iterations_per_segment=int(alns_cfg["iterations_per_segment"]),
            stale_segments=int(alns_cfg["stale_segments"]),
            rel_tol=float(alns_cfg["relative_improvement"]),
            seed=seed,
            weights=weights,
            destroy_fraction=tuple(alns_cfg["destroy_fraction"]),
            reaction_factor=float(alns_cfg["reaction_factor"]),
            cooling_rate=float(alns_cfg["cooling_rate"]),
            initial_temperature_fraction=float(alns_cfg["initial_temperature_fraction"]),
            max_segments=int(alns_cfg["max_segments"]),
            stop_metric=str(alns_cfg["stop_metric"]),
            max_seconds=float(_mapping_value(alns_cfg["wall_clock_guard_seconds"], scale)),
            resume_state=resume_state,
            segment_callback=checkpoint,
        )
        checkpoint(result.resume_state)
        schedule = result.best_schedule
        stop_reason = result.stop_reason
        counts = {"iterations": result.iterations, "segments": result.segments}
    else:  # pragma: no cover - argparse enforces this
        raise ValueError(method)

    elapsed = time.perf_counter() - started
    metrics = build_schedule_metrics(schedule, instance, profile, weights)
    if not metrics["feasible"] or not schedule.is_partition():
        raise RuntimeError(f"{method} produced an infeasible schedule")
    _write_schedule(run_dir / "schedule.json", schedule)
    record = {
        "protocol_version": config["protocol_version"],
        "method": method,
        "scale": scale,
        "batch_replicate": replicate,
        "method_seed": seed,
        "instance": row["name"],
        "objective_weights": list(weights),
        "metrics": metrics,
        "runtime_seconds": elapsed,
        "stop_reason": stop_reason,
        "counts": counts,
    }
    _write_json(run_dir / "record.json", record)
    return record


def _anchors(stage_count: int) -> list[list[float]]:
    anchors = [[1.0, 0.8, 0.0, 0.5]]
    if stage_count >= 2:
        anchors.append([1.0, 1.0, 0.0, 1.0])
    anchors.extend([[1.0, 1.0, 1.0, 1.0] for _ in range(stage_count - len(anchors))])
    return anchors


def _load_rows(rows: list[dict], root: Path):
    return [load_instance(str(row["name"]), root) for row in rows]


def _specialist_pools(
    config: dict,
    training_manifest: dict,
    validation_manifest: dict,
    training_root: Path,
    validation_root: Path,
    method: str,
    scale: int,
):
    groups = _mapping_value(config["stage_scale_groups"], scale)
    if method in {"direct", "pomo"}:
        groups = [[scale]]
    training_pools = {}
    validation_pools = {}
    for index, group in enumerate(groups):
        pool = f"stage_{index}"
        training_pools[pool] = _load_rows(
            rows_for_stage(training_manifest, scale, "curriculum", list(group)),
            training_root,
        )
        validation_pools[pool] = _load_rows(
            rows_for_stage(validation_manifest, scale, "development", list(group)),
            validation_root,
        )
    target_validation = _load_rows(
        rows_for_stage(validation_manifest, scale, "development", [scale]),
        validation_root,
    )
    return training_pools, validation_pools, target_validation


def _scale_only_pools(
    config: dict,
    training_manifest: dict,
    validation_manifest: dict,
    training_root: Path,
    validation_root: Path,
    scale: int,
):
    stages = _mapping_value(
        config["methods"]["scale_only"]["prefix_stage_instance_names"], scale
    )
    training_pools = {}
    validation_pools = {}
    for index, names in enumerate(stages):
        pool = f"stage_{index}"
        instances = [load_instance(str(name), training_root) for name in names]
        training_pools[pool] = instances
        validation_pools[pool] = [max(instances, key=lambda value: value.num_slabs)]
    final_pool = f"stage_{len(stages)}"
    training_pools[final_pool] = _load_rows(
        rows_for_stage(training_manifest, scale, "curriculum", [scale]),
        training_root,
    )
    development = _load_rows(
        rows_for_stage(validation_manifest, scale, "development", [scale]),
        validation_root,
    )
    validation_pools[final_pool] = development
    return training_pools, validation_pools, development


def _training_config(config: dict, method: str, scale: int, device: str, smoke: bool):
    model = config["model"]
    if method == "pomo":
        settings = config["methods"]["pomo"]
        return POMOConfig(
            d_model=int(settings["d_model"]),
            n_heads=int(settings["n_heads"]),
            n_layers=int(settings["n_layers"]),
            clip=float(model["clip"]),
            lr=float(settings["learning_rate"]),
            lr_final=None,
            entropy_coef=float(settings["entropy_coef"]),
            grad_clip=float(model["grad_clip"]),
            num_starts=4 if smoke else int(settings["num_starts"]),
            device=device,
            rich_ctx=False,
        )
    learning_rate = (
        float(config["methods"]["scale_only"]["learning_rate"])
        if method == "scale_only"
        else float(model["lr"])
    )
    return POMOConfig(
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        clip=float(model["clip"]),
        lr=learning_rate,
        lr_final=None if method == "scale_only" else float(model["lr_final"]),
        entropy_coef=float(model["entropy_coef"]),
        grad_clip=float(model["grad_clip"]),
        num_starts=4 if smoke else int(_mapping_value(model["num_starts_by_scale"], scale)),
        device=device,
        rich_ctx=bool(model["rich_ctx"]),
    )


def train_checkpoint(
    *, method: str, scale: int, seed: int, device: str, smoke: bool, output_root: Path
) -> dict:
    config = _load_config()
    _validate_method_identity(config, method, scale, seed)
    device = _device(device)
    training_path, validation_path, _ = _manifest_paths(config)
    training_manifest = _load_json(training_path)
    validation_manifest = _load_json(validation_path)
    training_errors = validate_specialist_manifest(training_manifest, training_path.parent)
    prefix_errors = (
        _scale_only_prefix_errors(config, training_manifest, training_path.parent)
        if method == "scale_only"
        else []
    )
    validation_errors = validate_specialist_manifest(
        validation_manifest, validation_path.parent
    )
    if training_errors or prefix_errors or validation_errors:
        raise RuntimeError(
            "training or validation manifest is invalid:\n"
            + "\n".join(training_errors + prefix_errors + validation_errors)
        )
    profile = load_process_profile(ROOT / config["process"])

    if method == "scale_only":
        pools, validation_pools, target_validation = _scale_only_pools(
            config,
            training_manifest,
            validation_manifest,
            training_path.parent,
            validation_path.parent,
            scale,
        )
    else:
        pools, validation_pools, target_validation = _specialist_pools(
            config,
            training_manifest,
            validation_manifest,
            training_path.parent,
            validation_path.parent,
            method,
            scale,
        )
    if smoke:
        pools = {name: [values[0]] for name, values in pools.items()}
        validation_pools = {name: [values[0]] for name, values in validation_pools.items()}
        target_validation = target_validation[:1]

    validation_instances = {
        instance.name: instance
        for values in validation_pools.values()
        for instance in values
    }
    validation_instances.update({instance.name: instance for instance in target_validation})
    references = {
        name: float(
            build_schedule_metrics(
                greedy_schedule(instance, profile),
                instance,
                profile,
                tuple(config["objective_weights"]),
            )["objective_value"]
        )
        for name, instance in validation_instances.items()
    }
    policy_config = _training_config(config, method, scale, device, smoke)
    validation = config["validation"]
    eval_every = 0.02 if smoke else float(validation["eval_every_seconds"])
    eval_starts = 4 if smoke else int(validation["eval_starts"])
    settings = config["methods"][method]
    started = time.perf_counter()

    if method == "pdcrl":
        policy, history = train_pdcrl_multibatch(
            "process_driven",
            pools,
            profile,
            policy_config,
            seed=seed,
            pool_names=tuple(pools),
            intermediate_cap_s=0.04 if smoke else float(settings["intermediate_cap_seconds"]),
            target_cap_s=0.06 if smoke else float(_mapping_value(settings["target_cap_seconds"], scale)),
            eval_every_s=eval_every,
            patience_evals=1 if smoke else int(settings["patience_evals"]),
            plateau_rel=float(settings["plateau_relative_improvement"]),
            lr_peak=float(settings["curriculum_peak_lr"]),
            eval_starts=eval_starts,
            blend_s=0.01 if smoke else float(settings["crossfade_seconds"]),
            guard_factor=float(settings["divergence_guard_factor"]),
            objective_weight_anchors=_anchors(len(pools)),
            stage_validation_pools=validation_pools,
            target_validation_instances=target_validation,
            validation_reference_costs=references,
            target_validation_scope="three-same-scale-development-batches",
        )
    elif method == "direct":
        policy, history = train_pdcrl_multibatch(
            "none",
            pools,
            profile,
            policy_config,
            seed=seed,
            pool_names=tuple(pools),
            target_cap_s=0.06 if smoke else float(_mapping_value(settings["target_cap_seconds"], scale)),
            eval_every_s=eval_every,
            patience_evals=1 if smoke else int(settings["patience_evals"]),
            plateau_rel=float(settings["plateau_relative_improvement"]),
            eval_starts=eval_starts,
            guard_factor=float(settings["divergence_guard_factor"]),
            stage_validation_pools=validation_pools,
            target_validation_instances=target_validation,
            validation_reference_costs=references,
            target_validation_scope="three-same-scale-development-batches",
        )
    elif method == "pomo":
        instances = [instance for values in pools.values() for instance in values]
        policy, history = train_published_pomo_fixed_budget(
            instances,
            profile,
            policy_config,
            seed=seed,
            total_seconds=0.06 if smoke else float(settings["fixed_budget_seconds"]),
            validation_instances=target_validation,
            validation_reference_costs=references,
            eval_every_s=eval_every,
            weight_decay=float(settings["weight_decay"]),
        )
    else:
        policy, history = train_scale_only_multibatch(
            pools,
            profile,
            policy_config,
            seed=seed,
            pool_names=tuple(pools),
            total_budget_s=0.05 * len(pools)
            if smoke
            else float(_mapping_value(settings["total_budget_seconds"], scale)),
            eval_every_s=eval_every,
            eval_starts=eval_starts,
            stage_validation_pools=validation_pools,
            target_validation_instances=target_validation,
            validation_reference_costs=references,
        )

    elapsed = time.perf_counter() - started
    phase = "smoke" if smoke else "training"
    run_dir = output_root / phase / method / f"p{scale}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in policy.state_dict().items()}
    temporary = run_dir / ".policy.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, run_dir / "policy.pt")
    _write_json(run_dir / "training_history.json", history)
    record = {
        "protocol_version": config["protocol_version"],
        "method": method,
        "scale": scale,
        "seed": seed,
        "smoke": smoke,
        "training_seconds": elapsed,
        "training_batch_count": sum(len(values) for values in pools.values()),
        "development_batch_count": sum(len(values) for values in validation_pools.values()),
        "checkpoint": "policy.pt",
        "checkpoint_sha256": sha256_file(run_dir / "policy.pt"),
        "checkpoint_state_sha256": state_dict_sha256(state),
        "objective_weights": config["objective_weights"],
        "device": device,
    }
    _write_json(run_dir / "training_record.json", record)
    return record


def _print_record(record: dict) -> None:
    print(json.dumps(record, indent=2, sort_keys=True, default=_json_default))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="validate data splits and checkpoint hashes")

    generate_data_parser = subparsers.add_parser(
        "generate-data", help="regenerate all synthetic data into a new directory"
    )
    generate_data_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output directory; generation refuses to overwrite an existing path",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a frozen checkpoint")
    evaluate_parser.add_argument("--method", choices=LEARNED_METHODS, required=True)
    evaluate_parser.add_argument("--scale", type=int, required=True)
    evaluate_parser.add_argument("--replicate", type=int, default=0)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument(
        "--raw", action="store_true", help="skip PDCRL's bounded local search"
    )
    evaluate_parser.add_argument("--output", type=Path, default=ROOT / "artifacts")

    baseline_parser = subparsers.add_parser("baseline", help="run a paper baseline")
    baseline_parser.add_argument("--method", choices=BASELINES, required=True)
    baseline_parser.add_argument("--scale", type=int, required=True)
    baseline_parser.add_argument("--replicate", type=int, default=0)
    baseline_parser.add_argument("--output", type=Path, default=ROOT / "artifacts")

    train_parser = subparsers.add_parser("train", help="train one paper checkpoint")
    train_parser.add_argument("--method", choices=LEARNED_METHODS, required=True)
    train_parser.add_argument("--scale", type=int, required=True)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--output", type=Path, default=ROOT / "artifacts")

    args = parser.parse_args()
    if args.command == "verify":
        _print_record(verify_release())
    elif args.command == "generate-data":
        config = _load_config()
        _print_record(
            reproduce_release_data(
                reference_data_root=ROOT / "data",
                process_profile_path=ROOT / config["process"],
                output_root=args.output,
            )
        )
    elif args.command == "evaluate":
        _print_record(
            evaluate_checkpoint(
                method=args.method,
                scale=args.scale,
                replicate=args.replicate,
                seed=args.seed,
                device=args.device,
                raw_only=args.raw,
                output_root=args.output,
            )
        )
    elif args.command == "baseline":
        _print_record(
            run_baseline(
                method=args.method,
                scale=args.scale,
                replicate=args.replicate,
                output_root=args.output,
            )
        )
    else:
        _print_record(
            train_checkpoint(
                method=args.method,
                scale=args.scale,
                seed=args.seed,
                device=args.device,
                smoke=args.smoke,
                output_root=args.output,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
