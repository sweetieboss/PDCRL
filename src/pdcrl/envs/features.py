from __future__ import annotations

import numpy as np

COIL_FEATURE_DIM = 8
CONTEXT_DIM = 6
_CARBON = ("LC", "MC", "HC")


def coil_features(instance, mill) -> np.ndarray:
    coils = instance.coils
    grades = sorted({c.grade for c in coils})
    gidx = {g: i for i, g in enumerate(grades)}
    gden = max(1, len(grades) - 1)
    wspan = max(1e-9, mill.width_max_mm - mill.width_min_mm)
    gspan = max(1e-9, mill.gauge_max_mm - mill.gauge_min_mm)
    lmax = max(1.0, max(c.length_m for c in coils))
    out = np.zeros((len(coils), COIL_FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(coils):
        out[i, 0] = (c.width_mm - mill.width_min_mm) / wspan
        out[i, 1] = (c.gauge_mm - mill.gauge_min_mm) / gspan
        out[i, 2] = (c.hardness_class - 1) / 4.0
        out[i, 3] = c.length_m / lmax
        out[i, 4 + _CARBON.index(c.carbon_class)] = 1.0
        out[i, 7] = gidx[c.grade] / gden
    return out


def context_features(
    state, mill, instance, num_placed: int, num_units: int, preference=None
) -> np.ndarray:
    c = np.zeros(CONTEXT_DIM, dtype=np.float32)
    if state.last_coil_idx is not None:
        last = instance.coils[state.last_coil_idx]
        wspan = max(1e-9, mill.width_max_mm - mill.width_min_mm)
        gspan = max(1e-9, mill.gauge_max_mm - mill.gauge_min_mm)
        c[0] = (last.width_mm - mill.width_min_mm) / wspan
        c[1] = (last.gauge_mm - mill.gauge_min_mm) / gspan
        c[2] = (last.hardness_class - 1) / 4.0
    c[3] = min(1.0, state.unit_km / max(1e-9, mill.roll_campaign_max_km))
    c[4] = min(1.0, state.run_len / max(1, mill.same_width_max_run))
    c[5] = num_placed / max(1, instance.num_coils)
    if preference is not None:
        return np.concatenate([c, np.asarray(preference, dtype=np.float32)])
    return c
