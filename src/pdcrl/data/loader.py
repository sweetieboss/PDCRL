"""Load HPCVRP hot-rolling instances (PLOS ONE 2020 formulation).

A slab is a node; scheduling assigns slabs to rolling units (routes) or leaves them
unscheduled (prize-collecting). This module parses slab tables and exposes the pairwise
transition penalty d_ij (width/gauge/hardness, coffin-asymmetric) and energy penalty P^e_ij,
plus per-slab prize p_i and capacity Q_i.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Slab:
    slab_id: int
    width_mm: float
    gauge_mm: float
    hardness: int
    steel_grade: str
    rolling_length_m: float  # Q_i, capacity contribution
    weight_t: float
    priority: int
    power_per_ton: float  # pe_i, drives energy penalty


@dataclass
class Instance:
    name: str
    slabs: list[Slab]

    @property
    def num_slabs(self) -> int:
        return len(self.slabs)

    def prize(self, i: int, profile) -> float:
        """Unscheduled-slab penalty p_i = prize_base * priority_i."""
        return profile.prize_base * self.slabs[i].priority

    def capacity(self, i: int) -> float:
        """Rolling length Q_i of slab i."""
        return self.slabs[i].rolling_length_m


def transition_component_penalty(a: Slab, b: Slab, profile) -> tuple[float, float, float, float]:
    """Split the a->b transition into process components: (width, gauge, hardness, energy).
    width+gauge+hardness == d_ij and energy == pe_ij from ``transition_penalty``. Used by the
    process_staged curriculum to introduce each process component into the reward separately."""
    if b.width_mm > a.width_mm:
        width = profile.width_up_penalty * (b.width_mm - a.width_mm)
    else:
        width = profile.width_down_penalty * max(0.0, (a.width_mm - b.width_mm) - profile.width_step_allow_mm)
    gauge = profile.gauge_penalty * max(0.0, abs(a.gauge_mm - b.gauge_mm) - profile.gauge_tol_mm)
    hardness = profile.hardness_penalty * max(0.0, abs(a.hardness - b.hardness) - profile.hardness_tol)
    energy = profile.energy_penalty * abs(a.power_per_ton - b.power_per_ton)
    return width, gauge, hardness, energy


def transition_penalty(a: Slab, b: Slab, profile) -> tuple[float, float]:
    """Return (d_ij, pe_ij): the (width+gauge+hardness) transition penalty and energy penalty a->b."""
    width, gauge, hardness, energy = transition_component_penalty(a, b, profile)
    return width + gauge + hardness, energy


def transition_matrices(instance: Instance, profile) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise (D, PE) matrices over real slabs, shape (n, n) each (routing 'distances')."""
    n = instance.num_slabs
    D = np.zeros((n, n), dtype=np.float64)
    PE = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            D[i, j], PE[i, j] = transition_penalty(instance.slabs[i], instance.slabs[j], profile)
    return D, PE


def transition_component_matrices(instance: Instance, profile):
    """Per-component (D_width, D_gauge, D_hardness, PE) matrices, shape (n, n) each.
    D_width+D_gauge+D_hardness equals ``transition_matrices``'s D; PE is identical."""
    n = instance.num_slabs
    Dw = np.zeros((n, n), dtype=np.float64)
    Dg = np.zeros((n, n), dtype=np.float64)
    Dh = np.zeros((n, n), dtype=np.float64)
    PE = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w, g, h, e = transition_component_penalty(instance.slabs[i], instance.slabs[j], profile)
            Dw[i, j], Dg[i, j], Dh[i, j], PE[i, j] = w, g, h, e
    return Dw, Dg, Dh, PE


def load_instance(name: str, processed_dir: str | Path) -> Instance:
    processed_dir = Path(processed_dir)
    df = pd.read_csv(processed_dir / "instances" / f"{name}.csv")
    slabs = [
        Slab(
            slab_id=int(r.slab_id), width_mm=float(r.width_mm), gauge_mm=float(r.gauge_mm),
            hardness=int(r.hardness), steel_grade=str(r.steel_grade),
            rolling_length_m=float(r.rolling_length_m), weight_t=float(r.weight_t),
            priority=int(r.priority), power_per_ton=float(r.power_per_ton),
        )
        for r in df.itertuples()
    ]
    return Instance(name=name, slabs=slabs)
