"""Canonical physical defaults shared by EEG ingestion and model front ends."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class EEGDataContract:
    """One source of truth for the derived model-ready P300 tensor.

    This is deliberately separate from an acquisition device's native sample
    rate, which remains attached to the raw recording and event ledger.
    """

    name: str
    sample_rate_hz: float = 128.0
    l_freq: float | None = 2.0
    h_freq: float | None = 30.0
    tmin_ms: float = -200.0
    tmax_ms: float = 800.0
    baseline_mode: str = "mean_only"
    signal_unit: str = "V"
    filter_method: str = "iir"
    filter_order: int = 4
    filter_phase: str = "zero"
    resample_domain: str = "epoched"
    resample_method: str = "fft"
    resample_npad: str = "auto"
    resample_window: str = "auto"
    resample_pad: str = "edge"

    @property
    def n_times(self) -> int:
        return int(floor((self.tmax_ms - self.tmin_ms) * self.sample_rate_hz / 1000.0 + 1e-9))


DEFAULT_P300_DATA_CONTRACT = EEGDataContract(name="p300_ms_eegnet_input_v2")
DEFAULT_GTN_DATA_CONTRACT = EEGDataContract(
    name="gtn_ms_eegnet_input_v2",
)


def assert_default_p300_input_contract(preprocessing: object) -> None:
    """Fail closed before mainline training on a physically incompatible cache."""

    expected = DEFAULT_P300_DATA_CONTRACT
    fields = {
        "sfreq": expected.sample_rate_hz,
        "l_freq": expected.l_freq,
        "h_freq": expected.h_freq,
        "tmin_ms": expected.tmin_ms,
        "tmax_ms": expected.tmax_ms,
        "n_times": expected.n_times,
        "baseline_mode": expected.baseline_mode,
        "signal_unit": expected.signal_unit,
        "filter_method": expected.filter_method,
        "filter_order": expected.filter_order,
        "filter_phase": expected.filter_phase,
        "resample_domain": expected.resample_domain,
        "resample_method": expected.resample_method,
        "resample_npad": expected.resample_npad,
        "resample_window": expected.resample_window,
        "resample_pad": expected.resample_pad,
    }
    mismatches: list[str] = []
    for field_name, expected_value in fields.items():
        actual = getattr(preprocessing, field_name, None)
        if isinstance(expected_value, float):
            matches = actual is not None and abs(float(actual) - expected_value) <= 1e-9
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(f"{field_name}={actual!r} (expected {expected_value!r})")
    if mismatches:
        raise ValueError(
            "Dataset cache does not match the executable P300 input contract: "
            + "; ".join(mismatches)
            + ". Regenerate the cache; legacy inputs cannot support current model claims."
        )


def assert_p300_source_provenance(dataset: object) -> None:
    """Require acquisition evidence needed to interpret a model-ready tensor."""

    provenance = getattr(dataset, "provenance", None)
    if not isinstance(provenance, dict):
        raise ValueError("Dataset lacks source provenance.")
    reference = provenance.get("source_reference")
    source_rate = provenance.get("source_sample_rate_hz")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Dataset provenance must declare one non-empty source_reference.")
    if source_rate is None:
        raise ValueError("Dataset provenance must declare source_sample_rate_hz.")
