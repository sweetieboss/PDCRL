"""Reproducible independent-batch generation and provenance manifests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping

from pdcrl.data.generator import generate_instance
from pdcrl.process import load_process_profile


@dataclass(frozen=True)
class BatchSpec:
    """One generated order batch in a benchmark."""

    name: str
    scale: int
    role: str
    replicate: int


@dataclass(frozen=True)
class BenchmarkSpec:
    """Frozen inputs needed to generate a benchmark dataset."""

    protocol_version: str
    seed_namespace: str
    batches: tuple[BatchSpec, ...]


def benchmark_spec_from_mapping(
    config: Mapping,
    *,
    include_supplementary: bool = False,
) -> BenchmarkSpec:
    """Expand scale/count mappings from a YAML-compatible benchmark config."""
    roles = ["target", "curriculum", "development", "bound"]
    if include_supplementary:
        roles.append("supplementary")
    batches: list[BatchSpec] = []
    for role in roles:
        counts = config.get(f"{role}_counts", {})
        for raw_scale, raw_count in sorted(counts.items(), key=lambda item: int(item[0])):
            scale = int(raw_scale)
            count = int(raw_count)
            if count < 0:
                raise ValueError(f"negative batch count for role={role!r}, scale={scale}")
            for replicate in range(count):
                batches.append(
                    BatchSpec(
                        name=f"{role}_n{scale:04d}_r{replicate:02d}",
                        scale=scale,
                        role=role,
                        replicate=replicate,
                    )
                )
    return BenchmarkSpec(
        protocol_version=str(config["protocol_version"]),
        seed_namespace=str(config["seed_namespace"]),
        batches=tuple(batches),
    )


def derive_generator_seed(
    protocol_version: str,
    seed_namespace: str,
    role: str,
    scale: int,
    replicate: int,
) -> int:
    """Derive a stable uint32 seed without relying on Python's randomized hash."""
    identity = f"{protocol_version}\0{seed_namespace}\0{role}\0{scale}\0{replicate}"
    return int.from_bytes(sha256(identity.encode("utf-8")).digest()[:4], "big")


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest(manifest: dict) -> str:
    """Return the stable serialized representation used for equality and hashing."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_spec(spec: BenchmarkSpec) -> None:
    seen_identities: set[tuple[str, int, int]] = set()
    seen_names: set[str] = set()
    for batch in spec.batches:
        identity = (batch.role, batch.scale, batch.replicate)
        if identity in seen_identities:
            raise ValueError(f"duplicate batch identity: {identity!r}")
        if batch.name in seen_names:
            raise ValueError(f"duplicate batch name: {batch.name!r}")
        if batch.scale <= 0 or batch.replicate < 0:
            raise ValueError(f"invalid batch: {batch!r}")
        seen_identities.add(identity)
        seen_names.add(batch.name)


def generate_benchmark(
    spec: BenchmarkSpec,
    process_profile_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Generate all batches and write a path-independent manifest."""
    _validate_spec(spec)
    process_profile_path = Path(process_profile_path)
    output_dir = Path(output_dir)
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    profile = load_process_profile(process_profile_path)

    rows = []
    for batch in spec.batches:
        seed = derive_generator_seed(
            spec.protocol_version,
            spec.seed_namespace,
            batch.role,
            batch.scale,
            batch.replicate,
        )
        relative_path = Path("instances") / f"{batch.name}.csv"
        csv_path = output_dir / relative_path
        generate_instance(profile, batch.scale, seed).to_csv(csv_path, index=False)
        rows.append(
            {
                "name": batch.name,
                "scale": batch.scale,
                "role": batch.role,
                "replicate": batch.replicate,
                "generator_seed": seed,
                "path": relative_path.as_posix(),
                "sha256": sha256_file(csv_path),
            }
        )

    manifest = {
        "protocol_version": spec.protocol_version,
        "seed_namespace": spec.seed_namespace,
        "process_profile": {
            "name": profile.name,
            "source_name": process_profile_path.name,
            "sha256": sha256_file(process_profile_path),
        },
        "batches": rows,
    }
    temp_path = output_dir / ".manifest.json.tmp"
    temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, output_dir / "manifest.json")
    return manifest


def validate_manifest(manifest: dict, root: str | Path) -> list[str]:
    """Return all structural and file-integrity errors found in a manifest."""
    root = Path(root)
    errors: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for row in manifest.get("batches", []):
        name = str(row.get("name", "<unnamed>"))
        identity = (str(row.get("role")), int(row.get("scale", -1)), int(row.get("replicate", -1)))
        if identity in seen:
            errors.append(f"{name}: duplicate batch identity {identity!r}")
        seen.add(identity)

        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{name}: path must be relative to the benchmark root")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"{name}: missing file {relative.as_posix()}")
            continue
        actual = sha256_file(path)
        if actual != row.get("sha256"):
            errors.append(f"{name}: sha256 mismatch (expected {row.get('sha256')}, got {actual})")
    return errors
