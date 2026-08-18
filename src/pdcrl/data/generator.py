"""Seeded synthetic HPCVRP hot-rolling instance generator.

Produces slab tables (PLOS ONE 2020 schema) reproducibly: same seed -> byte-identical CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_STEEL_DENSITY = 7850.0  # kg/m^3
COLUMNS = [
    "slab_id", "width_mm", "gauge_mm", "hardness", "steel_grade",
    "rolling_length_m", "weight_t", "priority", "power_per_ton",
]


def _name_offset(name: str) -> int:
    return sum(ord(c) for c in name) % 10000


def generate_instance(profile, n_slabs: int, seed: int) -> pd.DataFrame:
    """Deterministic slab table with the HPCVRP schema, rounded for byte-stable CSVs."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_slabs):
        width = round(float(rng.integers(int(profile.width_min_mm), int(profile.width_max_mm) + 1) / 10) * 10, 1)
        gauge = round(float(rng.uniform(profile.gauge_min_mm, profile.gauge_max_mm)), 2)
        hardness = int(rng.integers(profile.hardness_min, profile.hardness_max + 1))
        grade_idx = int(rng.integers(0, profile.num_grades))
        weight = round(float(rng.uniform(15.0, 30.0)), 1)
        length = round(weight * 1000.0 / ((width / 1000.0) * (gauge / 1000.0) * _STEEL_DENSITY), 1)
        priority = int(rng.integers(1, profile.priority_max + 1))
        # power-per-ton: grade base + gauge term (deterministic), rounded.
        power = round(100.0 + grade_idx * 15.0 + gauge * 2.0, 2)
        rows.append([sid, width, gauge, hardness, f"G{grade_idx}", length, weight, priority, power])
    return pd.DataFrame(rows, columns=COLUMNS)


def write_dataset(profile, processed_dir, specs: dict, seed: int = 0) -> None:
    """Write each named instance CSV under <processed_dir>/instances/."""
    inst_dir = Path(processed_dir) / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    for name, n in specs.items():
        df = generate_instance(profile, n, seed + _name_offset(name))
        df.to_csv(inst_dir / f"{name}.csv", index=False)
