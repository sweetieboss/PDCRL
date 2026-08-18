"""HPCVRP solution representation for hot-rolling batch scheduling.

A solution partitions slabs into rolling units (routes) plus a set of unscheduled slabs
(prize-collecting). Each slab appears in at most one unit; slabs not in any unit are unscheduled.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Schedule:
    num_slabs: int
    units: list[list[int]] = field(default_factory=list)  # each unit = ordered slab ids (a route)
    unscheduled: set[int] = field(default_factory=set)

    @property
    def num_units(self) -> int:
        return len([u for u in self.units if u])

    def scheduled_ids(self) -> list[int]:
        return [s for u in self.units for s in u]

    def is_partition(self) -> bool:
        """True iff every slab is in exactly one unit or unscheduled (no dup, full coverage)."""
        sched = self.scheduled_ids()
        allids = sorted(sched + list(self.unscheduled))
        return len(sched) == len(set(sched)) and allids == list(range(self.num_slabs))
