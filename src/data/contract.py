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
    l_freq: float | None = 0.1
    h_freq: float | None = 30.0
    tmin_ms: float = -200.0
    tmax_ms: float = 1200.0
    baseline_mode: str = "mean_only"
    signal_unit: str = "V"
    filter_method: str = "iir"
    filter_order: int = 4
    filter_phase: str = "zero"
    causal_iir_initial_state: str = "not_applicable"
    resample_domain: str = "epoched"
    resample_method: str = "fft"
    resample_npad: str = "auto"
    resample_window: str = "auto"
    resample_pad: str = "edge"

    @property
    def n_times(self) -> int:
        return int(floor((self.tmax_ms - self.tmin_ms) * self.sample_rate_hz / 1000.0 + 1e-9))


DEFAULT_P300_DATA_CONTRACT = EEGDataContract(name="p300_ms_eegnet_input_v3")
CAUSAL_IIR_INITIAL_STATE = "steady_state_first_sample"
# GTN is a 7-17 year-old, 3-channel cohort: its P300 energy sits at 1-4 Hz and
# its positive component can extend to ~1000 ms. The controlled LOSO ablation
# of 2026-08-30 (doc/gtn_accuracy_gap_audit_20260830.zh.md) measured the
# previous 2 Hz / 800 ms defaults at dAUC -0.066 / dHit -0.102 versus 0.1 Hz /
# 1200 ms, so this cohort contract restores the literature-supported values.
PAPER_GTN_DATA_CONTRACT = EEGDataContract(
    name="gtn_paper_offline_v1",
    l_freq=0.5,
    tmax_ms=1200.0,
)
# Causal one-pass filtering for within-subject prefix/suffix protocols.
# Zero-phase filtering would smear future test-period samples into training
# epochs; a chronological single-subject split therefore requires this contract.
SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT = EEGDataContract(
    name="p300_single_subject_causal_v3",
    l_freq=0.1,
    tmax_ms=1200.0,
    filter_phase="forward",
    causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
)
# Paper-aligned GTN causal contract (the source study high-passes at 0.5 Hz);
# used only for SOTA-comparison anchors, not for our performance claims.
PAPER_GTN_CAUSAL_DATA_CONTRACT = EEGDataContract(
    name="gtn_paper_causal_v2",
    l_freq=0.5,
    tmax_ms=1200.0,
    filter_phase="forward",
    causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
)
SOURCE_COHORT_DATA_CONTRACTS = {
    "offline": DEFAULT_P300_DATA_CONTRACT,
    "causal": SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    "gtn_paper": PAPER_GTN_CAUSAL_DATA_CONTRACT,
}


def assert_p300_input_contract(
    preprocessing: object,
    expected: EEGDataContract = DEFAULT_P300_DATA_CONTRACT,
) -> None:
    """Fail closed when a cache does not match an executable input contract."""

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
        "causal_iir_initial_state": expected.causal_iir_initial_state,
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


def assert_default_p300_input_contract(preprocessing: object) -> None:
    """Assert the offline zero-phase LOSO contract."""

    assert_p300_input_contract(preprocessing, DEFAULT_P300_DATA_CONTRACT)


def assert_causal_p300_input_contract(preprocessing: object) -> None:
    """Assert the causal contract required for chronological single-subject folds."""

    assert_p300_input_contract(preprocessing, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)


def assert_paper_gtn_causal_input_contract(preprocessing: object) -> None:
    """Assert the paper-aligned GTN causal contract (0.5 Hz high-pass, 1200 ms window)."""

    assert_p300_input_contract(preprocessing, PAPER_GTN_CAUSAL_DATA_CONTRACT)


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
