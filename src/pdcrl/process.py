"""Process profile for the hot-rolling HPCVRP formulation (PLOS ONE 2020).

Holds the penalty parameters that define the pairwise transition "distance" d_ij
(width/gauge/hardness jumps, coffin-asymmetric), the energy penalty P^e, the
unscheduled-slab prize scale, the rolling-unit capacity bounds [Q_S, Q_L], and the
ranges used by the synthetic data generator. Loaded from a YAML config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ProcessProfile:
    name: str
    # rolling-unit capacity (total rolling length per unit), Q_S..Q_L
    capacity_min_m: float
    capacity_max_m: float
    # width coffin penalty (asymmetric): widening penalised per mm; narrowing beyond allowance per mm
    width_up_penalty: float
    width_down_penalty: float
    width_step_allow_mm: float
    # gauge / hardness jump penalties (beyond a tolerance)
    gauge_penalty: float
    gauge_tol_mm: float
    hardness_penalty: float
    hardness_tol: float
    # energy: penalty per unit |Δ power_per_ton| between adjacent slabs
    energy_penalty: float
    # unscheduled-slab prize scale: p_i = prize_base * priority_i
    prize_base: float
    # data-generation ranges (used by the generator)
    width_min_mm: float
    width_max_mm: float
    gauge_min_mm: float
    gauge_max_mm: float
    hardness_min: int
    hardness_max: int
    num_grades: int
    priority_max: int
    # objective weights for the scalarised objective_value (S1, S2, S3, S4)
    w_transition: float
    w_unscheduled: float
    w_energy: float
    # roll-change (unit setup) cost: S4 = roll_change_cost * number of rolling units.
    # Calibrated by scripts/calibrate_rollchange.py; 0.0 disables the term.
    roll_change_cost: float = 0.0
    w_rollchange: float = 1.0


def load_process_profile(path: str | Path) -> ProcessProfile:
    cfg = OmegaConf.load(str(path))
    return ProcessProfile(
        name=str(cfg.name),
        capacity_min_m=float(cfg.capacity_min_m),
        capacity_max_m=float(cfg.capacity_max_m),
        width_up_penalty=float(cfg.width_up_penalty),
        width_down_penalty=float(cfg.width_down_penalty),
        width_step_allow_mm=float(cfg.width_step_allow_mm),
        gauge_penalty=float(cfg.gauge_penalty),
        gauge_tol_mm=float(cfg.gauge_tol_mm),
        hardness_penalty=float(cfg.hardness_penalty),
        hardness_tol=float(cfg.hardness_tol),
        energy_penalty=float(cfg.energy_penalty),
        prize_base=float(cfg.prize_base),
        width_min_mm=float(cfg.width_min_mm),
        width_max_mm=float(cfg.width_max_mm),
        gauge_min_mm=float(cfg.gauge_min_mm),
        gauge_max_mm=float(cfg.gauge_max_mm),
        hardness_min=int(cfg.hardness_min),
        hardness_max=int(cfg.hardness_max),
        num_grades=int(cfg.num_grades),
        priority_max=int(cfg.priority_max),
        w_transition=float(cfg.w_transition),
        w_unscheduled=float(cfg.w_unscheduled),
        w_energy=float(cfg.w_energy),
        roll_change_cost=float(cfg.get("roll_change_cost", 0.0)),
        w_rollchange=float(cfg.get("w_rollchange", 1.0)),
    )
