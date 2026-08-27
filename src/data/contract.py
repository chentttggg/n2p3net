"""Canonical physical defaults shared by EEG ingestion and model front ends."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class EEGDataContract:
    """One source of truth for a model-ready P300 time axis."""

    name: str
    sample_rate_hz: float = 250.0
    l_freq: float | None = 0.1
    h_freq: float | None = None
    tmin_ms: float = -200.0
    tmax_ms: float = 1200.0
    baseline_mode: str = "trial"

    @property
    def n_times(self) -> int:
        return int(floor((self.tmax_ms - self.tmin_ms) * self.sample_rate_hz / 1000.0 + 1e-9))


DEFAULT_P300_DATA_CONTRACT = EEGDataContract(name="p300_performance_v1")
DEFAULT_GTN_DATA_CONTRACT = EEGDataContract(
    name="gtn_lmbc_v1",
    tmax_ms=800.0,
)
