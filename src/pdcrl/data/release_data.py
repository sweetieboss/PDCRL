"""Safe, byte-exact regeneration of every synthetic dataset in the paper release."""

from __future__ import annotations

from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from pdcrl.data.benchmark import derive_generator_seed, sha256_file
from pdcrl.data.generator import generate_instance
from pdcrl.process import load_process_profile
from pdcrl.training.specialist_data import derive_subset_seed


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _checked_destination(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"release data path must be relative: {relative_value!r}")
    return root / relative


def _write_checked_csv(
    root: Path, row: dict, payload: bytes, *, identity: str
) -> None:
    actual = sha256(payload).hexdigest()
    expected = str(row["sha256"])
    if actual != expected:
        raise RuntimeError(
            f"{identity}: regenerated sha256 mismatch (expected {expected}, got {actual})"
        )
    destination = _checked_destination(root, str(row["path"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def validate_release_data_recipe(
    training: dict,
    validation: dict,
    test: dict,
    process_profile_path: str | Path,
) -> list[str]:
    """Return provenance errors that would prevent exact data regeneration."""
    errors: list[str] = []
    actual_profile_hash = sha256_file(process_profile_path)
    for label, manifest in (
        ("training", training),
        ("validation", validation),
        ("test", test),
    ):
        if manifest.get("process_profile", {}).get("sha256") != actual_profile_hash:
            errors.append(f"{label}: process profile hash mismatch")

    namespace = training.get("seed_namespace")
    if not namespace or validation.get("seed_namespace") != namespace:
        errors.append("training/validation: missing or inconsistent subset namespace")
    parent_generation = training.get("parent_generation", {})
    if not parent_generation or validation.get("parent_generation") != parent_generation:
        errors.append("training/validation: inconsistent parent-generation recipe")
    else:
        protocol = str(parent_generation.get("protocol_version", ""))
        parent_namespace = str(parent_generation.get("seed_namespace", ""))
        for label, manifest in (("training", training), ("validation", validation)):
            seen: set[str] = set()
            for row in manifest.get("source_batches", []):
                name = str(row.get("name", "<unnamed>"))
                if name in seen:
                    errors.append(f"{label}: duplicate source batch {name}")
                seen.add(name)
                if "generator_seed" not in row:
                    errors.append(f"{label}/{name}: missing generator_seed")
                    continue
                expected = derive_generator_seed(
                    protocol,
                    parent_namespace,
                    str(row.get("role")),
                    int(row.get("scale", -1)),
                    int(row.get("replicate", -1)),
                )
                if int(row["generator_seed"]) != expected:
                    errors.append(f"{label}/{name}: generator seed mismatch")
            source_names = seen
            for row in manifest.get("batches", []):
                name = str(row.get("name", "<unnamed>"))
                if str(row.get("parent_name")) not in source_names:
                    errors.append(f"{label}/{name}: missing source-batch recipe")
                    continue
                expected = derive_subset_seed(
                    str(namespace),
                    int(row.get("target_scale", -1)),
                    str(row.get("role")),
                    str(row.get("parent_name")),
                    str(row.get("parent_sha256")),
                )
                if int(row.get("subset_seed", -1)) != expected:
                    errors.append(f"{label}/{name}: subset seed mismatch")

    for row in training.get("scale_only_prefix_batches", []):
        if "generator_seed" not in row:
            errors.append(f"training/{row.get('name', '<unnamed>')}: missing generator_seed")

    test_generation = test.get("batch_generation", {})
    test_protocol = str(test_generation.get("protocol_version", ""))
    test_namespace = str(test_generation.get("seed_namespace", ""))
    if not test_protocol or not test_namespace or test.get("seed_namespace") != test_namespace:
        errors.append("test: missing or inconsistent batch-generation namespace")
    else:
        for row in test.get("batches", []):
            name = str(row.get("name", "<unnamed>"))
            if "generator_seed" not in row:
                errors.append(f"test/{name}: missing generator_seed")
                continue
            expected = derive_generator_seed(
                test_protocol,
                test_namespace,
                str(row.get("role")),
                int(row.get("scale", -1)),
                int(row.get("replicate", -1)),
            )
            if int(row["generator_seed"]) != expected:
                errors.append(f"test/{name}: generator seed mismatch")
    return errors


def _generate_source_batches(
    manifest: dict,
    profile,
) -> dict[str, tuple[dict, pd.DataFrame, bytes]]:
    generation = manifest.get("parent_generation", {})
    protocol = str(generation.get("protocol_version", ""))
    namespace = str(generation.get("seed_namespace", ""))
    if not protocol or not namespace:
        raise ValueError("parent generation protocol and seed namespace are required")

    result: dict[str, tuple[dict, pd.DataFrame, bytes]] = {}
    for row in manifest.get("source_batches", []):
        name = str(row["name"])
        if name in result:
            raise ValueError(f"duplicate source batch: {name}")
        expected_seed = derive_generator_seed(
            protocol,
            namespace,
            str(row["role"]),
            int(row["scale"]),
            int(row["replicate"]),
        )
        if int(row["generator_seed"]) != expected_seed:
            raise ValueError(f"{name}: generator seed is inconsistent with its namespace")
        frame = generate_instance(profile, int(row["scale"]), expected_seed)
        payload = _csv_bytes(frame)
        actual = sha256(payload).hexdigest()
        if actual != str(row["sha256"]):
            raise RuntimeError(f"{name}: regenerated source-batch sha256 mismatch")
        result[name] = (row, frame, payload)
    if not result:
        raise ValueError("source_batches must not be empty")
    return result


def _generate_nested_split(
    manifest: dict,
    profile,
    output_root: Path,
) -> int:
    sources = _generate_source_batches(manifest, profile)
    namespace = str(manifest.get("seed_namespace", ""))
    if not namespace:
        raise ValueError("nested-subset seed_namespace is required")

    count = 0
    for row in manifest.get("batches", []):
        name = str(row["name"])
        parent_name = str(row["parent_name"])
        if parent_name not in sources:
            raise ValueError(f"{name}: missing source batch {parent_name}")
        parent_row, parent, parent_payload = sources[parent_name]
        target_scale = int(row["target_scale"])
        scale = int(row["scale"])
        if (
            int(parent_row["scale"]) != target_scale
            or str(parent_row["sha256"]) != str(row["parent_sha256"])
            or str(parent_row["role"]) != str(row["role"])
            or int(parent_row["replicate"]) != int(row["parent_replicate"])
        ):
            raise ValueError(f"{name}: source-batch identity is inconsistent")

        subset_seed = derive_subset_seed(
            namespace,
            target_scale,
            str(row["role"]),
            parent_name,
            str(row["parent_sha256"]),
        )
        if subset_seed != int(row["subset_seed"]):
            raise ValueError(f"{name}: subset seed is inconsistent with its namespace")
        if scale == target_scale:
            selected = list(range(target_scale))
            payload = parent_payload
        else:
            selected = np.random.default_rng(subset_seed).permutation(target_scale)[:scale].tolist()
            child = parent.iloc[selected].copy()
            child["slab_id"] = np.arange(scale)
            payload = _csv_bytes(child)
        if selected != [int(value) for value in row["selected_parent_row_indices"]]:
            raise ValueError(f"{name}: recorded nested subset does not match its seed")
        _write_checked_csv(output_root, row, payload, identity=name)
        count += 1
    return count


def _generate_explicit_seed_rows(
    rows: list[dict],
    profile,
    output_root: Path,
) -> int:
    for row in rows:
        name = str(row["name"])
        if "generator_seed" not in row:
            raise ValueError(f"{name}: generator_seed is required")
        frame = generate_instance(profile, int(row["scale"]), int(row["generator_seed"]))
        _write_checked_csv(output_root, row, _csv_bytes(frame), identity=name)
    return len(rows)


def reproduce_release_data(
    *,
    reference_data_root: str | Path,
    process_profile_path: str | Path,
    output_root: str | Path,
) -> dict[str, int | bool]:
    """Regenerate all release CSVs into a new directory and verify every hash.

    ``output_root`` must not exist.  Generation happens in a temporary sibling and is
    renamed into place only after every regenerated CSV matches its frozen SHA-256.
    """
    reference_data_root = Path(reference_data_root).resolve()
    process_profile_path = Path(process_profile_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")

    manifests = {
        split: _load_manifest(reference_data_root / split / "manifest.json")
        for split in ("training", "validation", "test")
    }
    recipe_errors = validate_release_data_recipe(
        manifests["training"],
        manifests["validation"],
        manifests["test"],
        process_profile_path,
    )
    if recipe_errors:
        raise ValueError("invalid release data recipe:\n" + "\n".join(recipe_errors))

    profile = load_process_profile(process_profile_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        training_count = _generate_nested_split(
            manifests["training"], profile, temporary / "training"
        )
        prefix_count = _generate_explicit_seed_rows(
            manifests["training"].get("scale_only_prefix_batches", []),
            profile,
            temporary / "training",
        )
        validation_count = _generate_nested_split(
            manifests["validation"], profile, temporary / "validation"
        )
        test_count = _generate_explicit_seed_rows(
            manifests["test"].get("batches", []), profile, temporary / "test"
        )
        for split, manifest in manifests.items():
            manifest_path = temporary / split / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if output_root.exists():
            raise FileExistsError(f"output directory already exists: {output_root}")
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "training_records": training_count + prefix_count,
        "scale_only_prefix_records": prefix_count,
        "validation_records": validation_count,
        "test_records": test_count,
        "all_sha256_match": True,
    }
