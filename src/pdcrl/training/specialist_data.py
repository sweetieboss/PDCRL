"""Deterministic nested-subset construction for paper training batches."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import pandas as pd

from pdcrl.data.benchmark import sha256_file


ROLES = ("curriculum", "development")


def derive_subset_seed(namespace: str, target_scale: int, role: str, parent_name: str,
                       parent_sha256: str) -> int:
    """Derive the stable permutation seed for one nested curriculum ladder."""
    identity = (
        f"{namespace}\0{target_scale}\0{role}\0{parent_name}\0{parent_sha256}"
    )
    return int.from_bytes(sha256(identity.encode("utf-8")).digest()[:4], "big")


def _stage_groups(config: Mapping, target_scale: int) -> list[list[int]]:
    raw = config["stage_scale_groups"]
    groups = raw.get(target_scale, raw.get(str(target_scale)))
    if groups is None:
        raise KeyError(f"missing stage groups for target scale {target_scale}")
    result = [[int(value) for value in group] for group in groups]
    flat = [value for group in result for value in group]
    if not result or flat[-1] != target_scale or len(flat) != len(set(flat)):
        raise ValueError(f"invalid scale ladder for p{target_scale}: {result}")
    if any(value <= 0 or value > target_scale for value in flat):
        raise ValueError(f"out-of-range child scale in p{target_scale}: {result}")
    return result


def _source_rows(source_manifest: dict, target_scale: int, role: str,
                 expected_count: int) -> list[dict]:
    rows = sorted(
        (
            row for row in source_manifest.get("batches", [])
            if int(row.get("scale", -1)) == target_scale and row.get("role") == role
        ),
        key=lambda row: int(row["replicate"]),
    )
    if len(rows) != expected_count:
        raise ValueError(
            f"p{target_scale} {role}: expected {expected_count} parents, found {len(rows)}"
        )
    if [int(row["replicate"]) for row in rows] != list(range(expected_count)):
        raise ValueError(f"p{target_scale} {role}: parent replicates are not contiguous")
    return rows


def generate_specialist_benchmark(
    source_manifest_path: str | Path,
    training_output_dir: str | Path,
    validation_output_dir: str | Path,
    config: Mapping,
) -> tuple[dict, dict]:
    """Write separate training and validation ladders and manifests.

    A single permutation is derived for each parent. Every proper child is a prefix of
    that permutation; the full-size child is a byte-for-byte copy of the parent.
    """
    source_manifest_path = Path(source_manifest_path).resolve()
    source_root = source_manifest_path.parent
    output_dirs = {
        "curriculum": Path(training_output_dir).resolve(),
        "development": Path(validation_output_dir).resolve(),
    }
    if output_dirs["curriculum"] == output_dirs["development"]:
        raise ValueError("training and validation output directories must differ")
    for output_dir in output_dirs.values():
        (output_dir / "instances").mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(source_manifest_path.read_text())
    namespace = str(config["seed_namespace"])
    rows_by_role: dict[str, list[dict]] = {role: [] for role in ROLES}

    for target_scale in (int(value) for value in config["specialist_scales"]):
        groups = _stage_groups(config, target_scale)
        child_scales = [value for group in groups for value in group]
        for role in ROLES:
            output_dir = output_dirs[role]
            count = int(config[f"{role}_parent_count"])
            for parent in _source_rows(source_manifest, target_scale, role, count):
                parent_path = source_root / str(parent["path"])
                actual_parent_hash = sha256_file(parent_path)
                if actual_parent_hash != str(parent["sha256"]):
                    raise ValueError(f"source parent hash mismatch: {parent['name']}")
                frame = pd.read_csv(parent_path)
                if len(frame) != target_scale:
                    raise ValueError(
                        f"{parent['name']}: expected {target_scale} rows, found {len(frame)}"
                    )
                seed = derive_subset_seed(
                    namespace, target_scale, role, str(parent["name"]), actual_parent_hash
                )
                permutation = np.random.default_rng(seed).permutation(target_scale).tolist()
                for stage_index, group in enumerate(groups):
                    for derived_scale in group:
                        name = (
                            f"specialist_t{target_scale:04d}_{role}_"
                            f"r{int(parent['replicate']):02d}_n{derived_scale:04d}"
                        )
                        relative = Path("instances") / f"{name}.csv"
                        destination = output_dir / relative
                        if derived_scale == target_scale:
                            selected = list(range(target_scale))
                            shutil.copyfile(parent_path, destination)
                        else:
                            selected = permutation[:derived_scale]
                            child = frame.iloc[selected].copy()
                            child["slab_id"] = np.arange(derived_scale)
                            child.to_csv(destination, index=False, lineterminator="\n")
                        rows_by_role[role].append(
                            {
                                "name": name,
                                "scale": derived_scale,
                                "target_scale": target_scale,
                                "stage_index": stage_index,
                                "role": role,
                                "replicate": int(parent["replicate"]),
                                "parent_replicate": int(parent["replicate"]),
                                "parent_name": str(parent["name"]),
                                "parent_path": str(parent["path"]),
                                "parent_sha256": actual_parent_hash,
                                "subset_seed": seed,
                                "selected_parent_row_indices": selected,
                                "path": relative.as_posix(),
                                "sha256": sha256_file(destination),
                            }
                        )

    common = {
        "protocol_version": str(config["protocol_version"]),
        "seed_namespace": namespace,
        "source_manifest_name": source_manifest_path.name,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "stage_scale_groups": {
            str(scale): _stage_groups(config, int(scale))
            for scale in config["specialist_scales"]
        },
    }
    manifests = {
        "curriculum": {
            **common,
            "schema_version": "scale-specialist-training-v2",
            "batches": rows_by_role["curriculum"],
        },
        "development": {
            **common,
            "schema_version": "scale-specialist-validation-v1",
            "batches": rows_by_role["development"],
        },
    }
    for role, manifest in manifests.items():
        output_dir = output_dirs[role]
        temporary = output_dir / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_dir / "manifest.json")
    return manifests["curriculum"], manifests["development"]


def validate_specialist_manifest(manifest: dict, root: str | Path) -> list[str]:
    """Return structural, hash, size, and per-parent nesting errors."""
    root = Path(root)
    errors: list[str] = []
    identities: set[tuple] = set()
    grouped: dict[tuple, list[dict]] = {}
    for row in manifest.get("batches", []):
        name = str(row.get("name", "<unnamed>"))
        identity = (
            int(row.get("target_scale", -1)), str(row.get("role")),
            int(row.get("parent_replicate", -1)), int(row.get("scale", -1)),
        )
        if identity in identities:
            errors.append(f"{name}: duplicate derived identity {identity!r}")
        identities.add(identity)
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{name}: path must remain within benchmark root")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"{name}: missing file {relative.as_posix()}")
            continue
        actual = sha256_file(path)
        if actual != row.get("sha256"):
            errors.append(f"{name}: sha256 mismatch (expected {row.get('sha256')}, got {actual})")
        try:
            size = len(pd.read_csv(path))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{name}: unreadable CSV: {exc}")
            continue
        if size != int(row.get("scale", -1)):
            errors.append(f"{name}: row count {size} != scale {row.get('scale')}")
        selected = [int(value) for value in row.get("selected_parent_row_indices", [])]
        if len(selected) != int(row.get("scale", -1)) or len(selected) != len(set(selected)):
            errors.append(f"{name}: invalid selected_parent_row_indices")
        grouped.setdefault(identity[:3], []).append(row)

    for parent, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["scale"]))
        for left, right in zip(ordered, ordered[1:]):
            if not set(left["selected_parent_row_indices"]) < set(
                right["selected_parent_row_indices"]
            ):
                errors.append(f"{parent}: derived subsets are not strictly nested")
        if ordered:
            target = int(ordered[-1]["target_scale"])
            if int(ordered[-1]["scale"]) != target:
                errors.append(f"{parent}: missing full target-scale child")
            elif ordered[-1]["selected_parent_row_indices"] != list(range(target)):
                errors.append(f"{parent}: full target child must preserve parent order")
    return errors


def rows_for_stage(manifest: dict, target_scale: int, role: str,
                   stage_scales: list[int]) -> list[dict]:
    """Select deterministically ordered derived rows for one stage pool."""
    wanted = {int(value) for value in stage_scales}
    return sorted(
        (
            row for row in manifest["batches"]
            if int(row["target_scale"]) == int(target_scale)
            and row["role"] == role and int(row["scale"]) in wanted
        ),
        key=lambda row: (int(row["scale"]), int(row["parent_replicate"])),
    )
