"""Canonical physical defaults shared by EEG ingestion and model front ends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import floor, isfinite
from numbers import Real

BRAIN_SYNC_LAYOUT_SCHEMA = "brainsync-scalp-eog-layout/1"
BRAIN_SYNC_ACQUISITION_CHANNELS = ("Fz", "Cz", "P3", "Pz", "P4", "PO8", "EOG", "Oz")
BRAIN_SYNC_ACQUISITION_TYPES = ("EEG", "EEG", "EEG", "EEG", "EEG", "EEG", "EOG", "EEG")
BRAIN_SYNC_ACQUISITION_REFERENCE = "left earlobe"
BRAIN_SYNC_REFERENCE_SCHEMA = "brainsync-acquisition-reference/1"
BRAIN_SYNC_EEG_CHANNELS = tuple(
    name
    for name, kind in zip(
        BRAIN_SYNC_ACQUISITION_CHANNELS, BRAIN_SYNC_ACQUISITION_TYPES, strict=True
    )
    if kind == "EEG"
)


def brainsync_reference_record() -> dict[str, str]:
    """Physical REF placement, independent of any software rereferencing."""
    return {
        "schema": BRAIN_SYNC_REFERENCE_SCHEMA,
        "label": BRAIN_SYNC_ACQUISITION_REFERENCE,
        "side": "left",
        "site": "earlobe",
        "electrode": "ear_clip",
    }


def validate_zero_phase_edge_policy(policy: str, method: str, phase: str) -> None:
    if not isinstance(policy, str) or policy not in {"legacy", "mne_reflect_limited"}:
        raise ValueError("unknown zero_phase_edge_policy")
    if policy != "legacy" and (method != "iir" or phase != "zero"):
        raise ValueError("zero_phase_edge_policy requires zero-phase IIR")


def validate_notch_parameters(frequencies, quality_factor, *, sample_rate_hz=None) -> None:
    """Validate an explicit notch declaration against the source rate when known."""
    if (not isinstance(frequencies, (tuple, list))
            or any(isinstance(f, bool) or not isinstance(f, Real) or not isfinite(f) or f <= 0
                   for f in frequencies)
            or len(set(frequencies)) != len(frequencies)):
        raise ValueError("notch_freqs_hz must contain distinct positive finite frequencies")
    if (isinstance(quality_factor, bool) or not isinstance(quality_factor, Real)
            or not isfinite(quality_factor) or quality_factor <= 0):
        raise ValueError("notch_quality_factor must be positive and finite")
    if sample_rate_hz is not None:
        if not isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("notch source sample rate must be positive and finite")
        if any(f >= sample_rate_hz / 2 for f in frequencies):
            raise ValueError("notch frequencies must be below the source Nyquist frequency")



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
    notch_freqs_hz: tuple[float, ...] = ()
    notch_quality_factor: float = 30.0
    zero_phase_edge_policy: str = "legacy"
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
BRAIN_SYNC_CAUSAL_DATA_CONTRACT = replace(
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
    name="brainsync_causal_notch_v1",
    notch_freqs_hz=(50.0,),
)
BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT = replace(
    BRAIN_SYNC_CAUSAL_DATA_CONTRACT,
    name="brainsync_bidirectional_v1",
    filter_phase="zero",
    causal_iir_initial_state="not_applicable",
    zero_phase_edge_policy="mne_reflect_limited",
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


def epoch_support_record(
    contract: EEGDataContract = SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
) -> dict[str, str | float | int]:
    """Export the model geometry for an independently packaged acquisition app."""
    return {
        "schema": "n2p3-epoch-support/1",
        "tmin_ms": contract.tmin_ms,
        "tmax_ms": contract.tmax_ms,
        "model_sample_rate_hz": contract.sample_rate_hz,
        "n_times": contract.n_times,
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
        "notch_freqs_hz": expected.notch_freqs_hz,
        "notch_quality_factor": expected.notch_quality_factor,
        "zero_phase_edge_policy": expected.zero_phase_edge_policy,
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


def assert_personal_p300_input_contract(preprocessing: object) -> None:
    """Accept the complete-recording product recipe and frozen historical causal inputs."""
    if getattr(preprocessing, "filter_phase", None) == "zero":
        assert_p300_input_contract(preprocessing, BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
        return
    expected = (BRAIN_SYNC_CAUSAL_DATA_CONTRACT if getattr(preprocessing, "notch_freqs_hz", ())
                else SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    assert_p300_input_contract(preprocessing, expected)


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
