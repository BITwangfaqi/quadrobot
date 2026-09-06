"""Fixed stand/trot contact schedules in FR, FL, BR, BL order.

The old controller's ``getMpcTable`` defines the phase pattern. We reuse it
through an explicit file import, then evaluate that pattern at physical times.
This matters when the MPC intervals grow along the horizon.
"""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np


def _load_legacy_table():
    source = Path(__file__).resolve().parent.parent / "my_BIGDOG4_MPC" / "getMpcTable.py"
    spec = importlib.util.spec_from_file_location("_inv_dyn_legacy_contact_table", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the reference gait utility: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.getMpcTable


_legacy_table = _load_legacy_table()
_TROT_CONTACTS = _legacy_table(0, 10, [0, 5, 5, 0], [5] * 4).reshape(10, 4).astype(bool)
LEG_ORDER = ("FR", "FL", "BR", "BL")


@dataclass(frozen=True)
class ContactSchedule:
    """A snapshot with t=0 at the MPC solve time.

    ``time`` is time since the gait was activated. All query arguments are
    seconds into the future from this snapshot, including horizon boundaries.
    """

    mode: str = "stand"
    time: float = 0.0
    period: float = 0.60

    def __post_init__(self):
        if self.mode not in ("stand", "trot"):
            raise ValueError("Only the fixed stand and trot schedules are implemented")
        if not np.isfinite(self.time) or not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("Schedule time must be finite and period must be positive")

    def contact(self, t=0.0):
        if self.mode == "stand":
            return np.ones(4, dtype=bool)
        phase = ((self.time + float(t)) / self.period) % 1.0
        index = int(np.floor(phase * 10.0 + 1e-10)) % 10
        return _TROT_CONTACTS[index].copy()

    def swing_phase(self, t=0.0):
        """Zero in stance, otherwise normalized progress through the swing."""
        if self.mode == "stand":
            return np.zeros(4)
        phase = ((self.time + float(t)) / self.period) % 1.0
        progress = np.mod(phase - np.array([0.0, 0.5, 0.5, 0.0]), 1.0)
        return np.where(progress >= 0.5, 2.0 * (progress - 0.5), 0.0)

    def horizon(self, times):
        """Contact flags at *cumulative* times (seconds), shape (N, 4)."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("Horizon times must be a finite nonnegative vector")
        if self.mode == 'stand':
            return np.ones((len(times), 4), dtype=bool)
        phase = ((self.time + times) / self.period) % 1.0
        indices = np.floor(phase * 10.0 + 1e-10).astype(int) % 10
        return _TROT_CONTACTS[indices].copy()

    def from_intervals(self, intervals):
        intervals = np.asarray(intervals, dtype=float)
        if intervals.ndim != 1 or np.any(intervals <= 0) or not np.all(np.isfinite(intervals)):
            raise ValueError("Horizon intervals must be finite and positive")
        return self.horizon(np.r_[0.0, np.cumsum(intervals)])
