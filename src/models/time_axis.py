"""Physical time-axis contract shared by N2P3-Net temporal modules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpochTimeAxis:
    """A sampled EEG epoch described explicitly in milliseconds.

    ``tmax_ms`` is the exclusive right edge. Therefore a 256-sample epoch at
    256 Hz spanning -200 ms has ``tmax_ms=800`` and its final sample is at
    796.09375 ms.
    """

    tmin_ms: float
    tmax_ms: float
    sfreq: float
    n_time: int

    def __post_init__(self) -> None:
        values = (self.tmin_ms, self.tmax_ms, self.sfreq)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("time-axis values must be finite.")
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}.")
        if int(self.n_time) != self.n_time or self.n_time < 2:
            raise ValueError(f"n_time must be an integer >=2, got {self.n_time}.")
        if self.tmax_ms <= self.tmin_ms:
            raise ValueError(f"tmax_ms must exceed tmin_ms, got {self.tmin_ms}/{self.tmax_ms} ms.")

        declared = self.duration_ms
        expected = self.expected_duration_ms
        # One sample accommodates data loaders that describe an inclusive last
        # sample instead of the exclusive right edge. A seconds-valued 0.8 ms
        # epoch still fails by three orders of magnitude.
        tolerance = self.sample_period_ms + 1e-6
        if abs(declared - expected) > tolerance:
            raise ValueError(
                "inconsistent physical time axis: "
                f"tmax_ms-tmin_ms={declared:.6g} ms, but n_time/sfreq="
                f"{expected:.6g} ms (tolerance one sample={tolerance:.6g} ms). "
                "N2P3-Net time bounds are milliseconds, not seconds."
            )

    @property
    def duration_ms(self) -> float:
        return float(self.tmax_ms - self.tmin_ms)

    @property
    def expected_duration_ms(self) -> float:
        return 1000.0 * int(self.n_time) / float(self.sfreq)

    @property
    def sample_period_ms(self) -> float:
        return 1000.0 / float(self.sfreq)

    def samples_ms(self) -> np.ndarray:
        return self.tmin_ms + np.arange(self.n_time, dtype=float) * self.sample_period_ms
