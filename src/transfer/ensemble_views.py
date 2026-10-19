"""Shared physical decisions, with separate labeled fitting and unlabeled scoring.

Raw observations retain their native reference and sampling grid. Expert-specific
epoch transformations are explicit frozen profiles; they never alter the shared
physical decision or read beyond its declared observations and availability.
"""
from __future__ import annotations

import hashlib
import io
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from numbers import Integral
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from data.bnci2014_008_candidate import (
    BNCI2014_008_CANDIDATE_TASK_CONTRACT,
    BNCI2014_008_CHANNELS,
    BNCI2014_008_DATASET_ID,
    BNCI2014_008_FLASHES_PER_SELECTION,
    BNCI2014_008_GRID_SHAPE,
    BNCI2014_008_REPETITIONS_PER_SELECTION,
    BNCI2014_008_SOURCE_REFERENCE,
    BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
    bnci2014_008_raw,
    build_bnci2014_008_subject_dataset,
    read_bnci2014_008_candidate_record,
)
from data.candidate_task import (
    CandidateTaskContract,
    MembershipDecision,
    validate_candidate_membership_metadata,
)
from data.channel import build_channel_identity, canonical_channel_name
from data.continuous_qc import signal_dependency_matrix
from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.domain import adapt_observed_montage
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    materialize_dataset_identity,
    materialize_dataset_lineage,
    parse_preprocessing_spec,
    validate_preprocessing_execution,
)
from data.events import ScheduledEventTimeline
from data.finite_filter_operator import FiniteFilterOperator
from data.identity import DatasetIdentityTable, origin_subject_key
from data.lineage import DataLineage
from data.preprocess import (
    RawSampleScope,
    design_continuous_sos,
    filter_bidirectional_segment,
    prefix_raw_scopes,
    source_epoch_support,
    transform_epoch_samples,
)
from research.contracts import semantic_sha256
from research.prefix_replay_inputs import immutable_input_array, immutable_input_array_geometry
from transfer.binary import (
    binary_time_split,
    calibration_budget_requests,
    resolve_calibration_splits,
)
from transfer.candidate_decision import parse_row_column_target, row_column_target
from transfer.continuous_decision import CalibrationRecord
from transfer.noise_model import array_digest, freeze_metadata, require_identifiers, require_sha256

_SELECTION_EXECUTION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _immutable_array(values, dtype):
    return immutable_input_array(values, dtype)


def observed_spatial_projection(native_channels, slot_channels, *, reference_channel=None):
    """One label-free measured-space matrix, with absent slots explicitly unobserved."""
    native = tuple(map(canonical_channel_name, native_channels))
    slots = tuple(slot_channels)
    matrix = signal_dependency_matrix(native_channels, slots, reference_channel=reference_channel)
    present = np.asarray([canonical_channel_name(name) in native for name in slots])
    observed = present & np.any(matrix != 0, axis=1)
    return matrix, observed


def project_raw_observations(samples, native_channels, native_reference, declaration, *, sample_rate_hz):
    """Raw covariance consumes only actual nonzero observed coordinates."""
    if declaration is None:
        return samples, tuple(native_channels), "native_unfiltered_volts/1", None
    mode = declaration["mode"]
    if mode == "native":
        if (tuple(declaration["channels"]) != tuple(native_channels)
                or declaration["reference"] != native_reference):
            raise ValueError("native raw spatial programme differs from actual observation authority")
        matrix, observed = observed_spatial_projection(native_channels, native_channels)
        channels = tuple(native_channels)
        reference = native_reference
    elif mode == "observed_measured_reference":
        matrix, observed = observed_spatial_projection(native_channels, declaration["slot_channels"],
                                                      reference_channel=declaration["reference_channel"])
        if declaration.get("observed_only_for_raw_covariance") is not True:
            raise ValueError("raw spatial programme must consume only real observed coordinates")
        if "matrix" in declaration and not np.array_equal(matrix, declaration["matrix"]):
            raise ValueError("raw spatial matrix differs from its actual measured dependencies")
        channels = tuple(name for name, present in zip(declaration["slot_channels"], observed, strict=True) if present)
        reference = "measured_channel:" + canonical_channel_name(declaration["reference_channel"])
    else:
        raise ValueError("unknown raw spatial programme")
    if declaration.get("native_sample_rate_hz", sample_rate_hz) != sample_rate_hz:
        raise ValueError("raw spatial programme cannot change the actual temporal sample grid")
    matrix = matrix[observed]
    if not len(matrix) or np.linalg.matrix_rank(matrix) != len(matrix):
        raise ValueError("raw spatial covariance requires independent actual observed coordinates")
    values = _immutable_array(np.asarray(samples) @ matrix.T, np.float64)
    receipt = {"schema": "n2p3_observed_raw_spatial_projection/1", "declaration": declaration,
        "native_channels": list(native_channels), "native_reference": native_reference,
        "native_values_sha256": array_digest(samples), "native_sample_rate_hz": sample_rate_hz,
        "matrix": matrix.tolist(), "observed_slots": observed.tolist(), "rank": len(matrix),
        "output_channels": list(channels), "output_reference": reference,
        "output_values_sha256": array_digest(values), "time_transform": "none"}
    return values, channels, "native_unfiltered_observed_spatial/1:" + semantic_sha256({
        "native_channels": native_channels, "native_reference": native_reference,
        "matrix": matrix.tolist(), "output_channels": channels, "reference": reference}), freeze_metadata(receipt)


def _immutable_decision(decision):
    # Copy at ingress: callers cannot mutate a view through their original ledger.
    result = MembershipDecision.from_record(decision.record())
    for field_name in ("membership", "stimulus_indices", "repetition_indices", "evidence_indices"):
        values = getattr(result, field_name)
        object.__setattr__(result, field_name, _immutable_array(values, values.dtype))
    return result


@dataclass(frozen=True)
class PredictionDecisionView:
    """Actual native EEG plus a public schedule; this type contains no truth."""

    record_id: str
    participant_key: str
    decision_id: str
    samples: np.ndarray | None
    onsets: np.ndarray
    decision: MembershipDecision
    source_sha256: str
    raw_scope: tuple[int, int]
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    reference: str
    source_relative_path: str
    source_dataset_id: str
    source_subject_id: str
    available_sample: int
    signal_unit: str = DEFAULT_P300_DATA_CONTRACT.signal_unit
    unavailable_reason: str | None = None

    def __post_init__(self):
        for field_name in ("record_id", "participant_key", "decision_id", "reference",
                           "source_relative_path", "source_dataset_id", "source_subject_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} requires an exact nonempty identity")
        require_sha256(self.source_sha256, "original raw source")
        channels = tuple(self.channel_names)
        require_identifiers(channels, "physical EEG channels")
        if len(set(map(canonical_channel_name, channels))) != len(channels):
            raise ValueError("physical EEG channels must be distinct after canonical normalization")
        kinds = tuple(self.channel_types)
        if (len(kinds) != len(channels) or any(kind != "EEG" for kind in kinds)
                or any(name.upper() in {"EOG", "VEOG", "HEOG"} for name in channels)):
            raise ValueError("prediction samples must contain measured EEG channels only")
        if (isinstance(self.sample_rate_hz, bool) or not np.isfinite(self.sample_rate_hz)
                or self.sample_rate_hz <= 0 or self.signal_unit != DEFAULT_P300_DATA_CONTRACT.signal_unit):
            raise ValueError("native sample rate must be positive and actual samples must be volts")
        scope = tuple(self.raw_scope)
        if (len(scope) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in scope)
                or not 0 <= scope[0] < scope[1]):
            raise ValueError("actual raw scope must be an absolute half-open sample interval")
        if (isinstance(self.available_sample, bool) or not isinstance(self.available_sample, Integral)
                or self.available_sample < scope[1]):
            raise ValueError("availability cannot precede the last actual raw observation")
        samples = None if self.samples is None else np.asarray(self.samples)
        if samples is None and not self.unavailable_reason:
            raise ValueError("unobserved physical decisions require an explicit unavailable reason")
        if samples is not None and (samples.shape != (scope[1] - scope[0], len(channels))
                or samples.dtype.kind != "f" or not np.isfinite(samples).all()):
            raise ValueError("actual raw scope and finite samples-by-EEG-channel values must agree")
        if not isinstance(self.decision, MembershipDecision):
            raise TypeError("public physical membership requires a MembershipDecision")
        onset = np.asarray(self.onsets)
        if (onset.ndim != 1 or len(onset) != len(self.decision.event_ids)
                or onset.dtype.kind not in "iu" or np.any(onset[1:] <= onset[:-1])
                or not len(onset) or onset[0] < 0 or onset[-1] >= scope[1] - scope[0]):
            raise ValueError("public event onsets must be ordered integer samples within the actual raw scope")
        object.__setattr__(self, "samples", None if samples is None else _immutable_array(samples, np.float64))
        object.__setattr__(self, "onsets", _immutable_array(onset, np.int64))
        object.__setattr__(self, "decision", _immutable_decision(self.decision))
        object.__setattr__(self, "raw_scope", tuple(int(value) for value in scope))
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "channel_types", kinds)
        object.__setattr__(self, "available_sample", int(self.available_sample))

    def physical_record(self):
        record = {"schema": "n2p3_prediction_decision_physical_support/1",
                "record_id": self.record_id, "participant_key": self.participant_key,
                "decision_id": self.decision_id, "source_sha256": self.source_sha256,
                "source_relative_path": self.source_relative_path,
                "source_dataset_id": self.source_dataset_id, "source_subject_id": self.source_subject_id,
                "raw_scope": list(self.raw_scope), "sample_rate_hz": self.sample_rate_hz,
                "channel_names": list(self.channel_names), "channel_types": list(self.channel_types),
                "reference": self.reference, "signal_unit": self.signal_unit,
                "onsets": self.onsets.tolist(), "decision": self.decision.record(),
                "available_sample": self.available_sample}
        if self.unavailable_reason is not None:
            record["unavailable_reason"] = self.unavailable_reason
        return record

    @property
    def physical_support_digest(self):
        return semantic_sha256(self.physical_record())

    @property
    def raw_input_digest(self):
        return semantic_sha256({"physical_support_digest": self.physical_support_digest,
                                "raw_samples_sha256": None if self.samples is None else array_digest(self.samples)})

    def record(self):
        return {**self.physical_record(), "physical_support_digest": self.physical_support_digest,
                "raw_input_digest": self.raw_input_digest}

    @classmethod
    def from_record(cls, record, *, samples):
        values = dict(record)
        if values.pop("schema") != "n2p3_prediction_decision_physical_support/1":
            raise ValueError("unknown prediction decision view schema")
        physical_digest = values.pop("physical_support_digest")
        input_digest = values.pop("raw_input_digest")
        values["decision"] = MembershipDecision.from_record(values["decision"])
        values["onsets"] = np.asarray(values["onsets"], dtype=np.int64)
        result = cls(**values, samples=samples)
        if result.physical_support_digest != physical_digest or result.raw_input_digest != input_digest:
            raise ValueError("prediction decision receipt differs from actual physical or numerical input")
        return result


@dataclass(frozen=True)
class CalibrationSelection:
    """Authorized labels stay on the fit side of the prediction interface."""

    view: PredictionDecisionView
    labels: np.ndarray
    target_candidate: str

    def __post_init__(self):
        if not isinstance(self.view, PredictionDecisionView):
            raise TypeError("calibration requires a validated prediction view")
        labels = np.asarray(self.labels)
        if (labels.dtype != bool or labels.shape != (len(self.view.onsets),)
                or self.target_candidate not in self.view.decision.candidate_vocabulary):
            raise ValueError("calibration requires authorized binary labels and an exact candidate truth")
        target_index = self.view.decision.candidate_vocabulary.index(self.target_candidate)
        if not np.array_equal(labels, self.view.decision.membership[:, target_index]):
            raise ValueError("calibration labels disagree with public physical membership")
        object.__setattr__(self, "labels", _immutable_array(labels, bool))

    def record(self):
        return {"schema": "n2p3_calibration_selection/1", "view": self.view.record(),
                "target_candidate": self.target_candidate,
                "authorized_label_count": len(self.labels), "labels_sha256": array_digest(self.labels)}


def _profile_operator(preprocessing, source_rate):
    sos = design_continuous_sos(
        source_rate, l_freq=preprocessing.l_freq, h_freq=preprocessing.h_freq,
        order=preprocessing.filter_order, notch_freqs_hz=preprocessing.notch_freqs_hz,
        notch_quality_factor=preprocessing.notch_quality_factor,
    )
    if not len(sos):
        return None
    padding = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    return FiniteFilterOperator(tuple(map(tuple, sos)), padding)


@dataclass(frozen=True)
class SelectionEpochProfile:
    """One frozen shared-owner transform on an explicitly observed raw segment.

    ``reference`` identifies the native observations. A measured re-reference is
    a separate, explicit projection and never overwrites the acquisition field.
    Geometry comes from the complete preprocessing contract, including its
    closed source endpoint; 178 and 179 samples represent distinct profiles.
    """

    preprocessing: PreprocessingSpec
    source_sample_rate_hz: float
    channel_names: tuple[str, ...]
    reference: str
    operator: FiniteFilterOperator | None
    reference_channel: str | None = None
    output_dtype: str = "float32"
    allow_missing_channels: bool = False
    _source_start: int = field(init=False, repr=False, compare=False)
    _source_stop: int = field(init=False, repr=False, compare=False)
    _resampled_samples: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.preprocessing, PreprocessingSpec):
            raise TypeError("epoch profiles require a complete PreprocessingSpec")
        self.preprocessing.validate()
        if self.preprocessing.filter_phase != "zero":
            raise ValueError("independent selection profiles require an explicit finite zero-phase filter")
        if (self.preprocessing.notch_freqs_hz
                and self.preprocessing.zero_phase_edge_policy != "mne_reflect_limited"):
            raise ValueError("finite selection filtering cannot execute the declared legacy notch edge policy")
        channels = tuple(self.channel_names)
        require_identifiers(channels, "declared epoch EEG channels")
        canonical = tuple(map(canonical_channel_name, channels))
        if len(set(canonical)) != len(canonical) or any(name in {"EOG", "VEOG", "HEOG"} for name in canonical):
            raise ValueError("epoch profiles require distinct measured EEG channels")
        if not isinstance(self.reference, str) or not self.reference or self.reference != self.reference.strip():
            raise ValueError("epoch profiles must identify the actual native acquisition reference")
        if self.reference_channel is not None:
            anchor = canonical_channel_name(self.reference_channel)
            if anchor in canonical or anchor in {"EOG", "VEOG", "HEOG"}:
                raise ValueError("the measured EEG reference cannot also be an independent epoch channel")
            object.__setattr__(self, "reference_channel", anchor)
        if self.output_dtype not in {"float32", "float64"}:
            raise ValueError("epoch output dtype must be explicitly float32 or float64")
        starts, stops = source_epoch_support(np.array([[0, 0, 1]]), self.source_sample_rate_hz,
                                             self.preprocessing)
        source_samples = int(stops[0] - starts[0])
        resampled = (int(round(source_samples * self.preprocessing.sfreq / self.source_sample_rate_hz))
                     if not np.isclose(self.preprocessing.sfreq, self.source_sample_rate_hz) else source_samples)
        if resampled < self.preprocessing.n_times:
            raise ValueError("actual source epoch geometry cannot produce the declared output samples")
        expected = _profile_operator(self.preprocessing, self.source_sample_rate_hz)
        if ((self.operator is None) != (expected is None)
                or expected is not None and (not isinstance(self.operator, FiniteFilterOperator)
                                            or self.operator.record() != expected.record())):
            raise ValueError("frozen finite filter differs from the preprocessing declaration")
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "_source_start", int(starts[0]))
        object.__setattr__(self, "_source_stop", int(stops[0]))
        object.__setattr__(self, "_resampled_samples", resampled)

    @classmethod
    def from_preprocessing(cls, preprocessing, *, source_sample_rate_hz, channel_names,
                           reference, reference_channel=None, output_dtype="float32", allow_missing_channels=False):
        return cls(preprocessing, source_sample_rate_hz, tuple(channel_names), reference,
                   _profile_operator(preprocessing, source_sample_rate_hz), reference_channel, output_dtype, allow_missing_channels)

    @property
    def epoch_geometry(self):
        return {"source_sample_rate_hz": self.source_sample_rate_hz,
                "source_start_inclusive": self._source_start,
                "source_stop_exclusive": self._source_stop,
                "source_last_inclusive": self._source_stop - 1,
                "source_samples": self._source_stop - self._source_start,
                "actual_source_start_ms": self._source_start / self.source_sample_rate_hz * 1000,
                "actual_source_last_ms": (self._source_stop - 1) / self.source_sample_rate_hz * 1000,
                "fft_resampled_samples": self._resampled_samples,
                "output_sample_rate_hz": self.preprocessing.sfreq,
                "output_samples": self.preprocessing.n_times,
                "first_T_discarded_output_samples": self._resampled_samples - self.preprocessing.n_times,
                "declared_output_start_ms": self.preprocessing.tmin_ms,
                "declared_output_last_ms": self.preprocessing.tmin_ms
                + (self.preprocessing.n_times - 1) / self.preprocessing.sfreq * 1000}

    def record(self):
        record = {"schema": "n2p3_selection_epoch_profile/1", "preprocessing": self.preprocessing.record(),
                "source_sample_rate_hz": self.source_sample_rate_hz,
                "channel_names": list(self.channel_names), "reference": self.reference,
                "reference_channel": self.reference_channel, "output_dtype": self.output_dtype,
                "operator": None if self.operator is None else self.operator.record(),
                "epoch_geometry": self.epoch_geometry}
        if self.allow_missing_channels:
            record["allow_missing_channels"] = True
        return record

    @property
    def digest(self):
        return semantic_sha256(self.record())

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        if values.pop("schema") != "n2p3_selection_epoch_profile/1":
            raise ValueError("unknown selection epoch profile schema")
        geometry = values.pop("epoch_geometry")
        values["preprocessing"] = parse_preprocessing_spec(values["preprocessing"])
        if values["operator"] is not None:
            values["operator"] = FiniteFilterOperator.from_record(values["operator"])
        result = cls(**values)
        if result.epoch_geometry != geometry:
            raise ValueError("saved epoch geometry differs from the actual preprocessing contract")
        return result

    def validate_observation(self, view: PredictionDecisionView):
        """Check the frozen physical projection before deciding data availability."""
        if not isinstance(view, PredictionDecisionView):
            raise TypeError("epoch prediction transforms consume unlabeled PredictionDecisionView inputs")
        if view.sample_rate_hz != self.source_sample_rate_hz or view.reference != self.reference:
            raise ValueError("epoch profile differs from the actual native rate or acquisition reference")
        native = tuple(map(canonical_channel_name, view.channel_names))
        required = (*map(canonical_channel_name, self.channel_names),)
        if self.reference_channel is not None:
            required += (self.reference_channel,)
        missing = set(required)-set(native)
        if missing and (not self.allow_missing_channels or self.reference_channel in missing):
            raise ValueError("epoch projection requests EEG or measured-reference channels absent from actual observations")

    def transform(self, view: PredictionDecisionView):
        """Return actual epochs, their observation masks and an execution receipt."""
        self.validate_observation(view)
        starts, stops = view.onsets + self._source_start, view.onsets + self._source_stop
        if starts.min() < 0 or stops.max() > len(view.samples):
            raise ValueError("declared epoch endpoints extend beyond observed raw support")
        filtered = view.samples
        filter_receipt = {"operator": None, "padding_samples": 0,
                          "left_state": [], "right_state": [], "effective_filter_states_sha256": None}
        if self.operator is not None:
            executed = filter_bidirectional_segment(np.asarray(self.operator.sos), view.samples.T,
                                                    padding_limit=self.operator.padding_limit)
            filtered = executed.data.T
            filter_receipt = {"operator": self.operator.record(), "padding_samples": executed.padding_samples,
                              "left_state": executed.left_state.tolist(), "right_state": executed.right_state.tolist(),
                              "effective_filter_states_sha256": array_digest(executed.left_state, executed.right_state)}
        raw_epochs = np.stack([filtered[start:stop].T for start, stop in zip(starts, stops, strict=True)])
        epochs = transform_epoch_samples(raw_epochs, view.sample_rate_hz, self.preprocessing,
                                         require_linear=True).astype(self.output_dtype)
        projection, spatial_observed = observed_spatial_projection(view.channel_names, self.channel_names,
                                                                   reference_channel=self.reference_channel)
        projection = projection.astype(self.output_dtype)
        epochs = np.einsum("nct,oc->not", epochs, projection, optimize=True)
        mask = (np.broadcast_to(view.decision.evidence_indices[:, None] >= 0,
                               (len(view.onsets), len(self.channel_names))) & spatial_observed).copy()
        epochs[~mask] = 0
        epochs = _immutable_array(epochs, self.output_dtype)
        mask = _immutable_array(mask, bool)
        numerical_digest = semantic_sha256({"profile_digest": self.digest,
            "physical_support_digest": view.physical_support_digest, "values_sha256": array_digest(epochs, mask)})
        receipt = {"schema": "n2p3_selection_epoch_execution/1", "profile": self.record(),
                   "profile_digest": self.digest, "physical_support_digest": view.physical_support_digest,
                   "raw_input_digest": view.raw_input_digest, "numerical_input_digest": numerical_digest,
                   "epoch_geometry": self.epoch_geometry, "raw_scope": list(view.raw_scope),
                   "source_epoch_scopes": (np.column_stack((starts, stops)) + view.raw_scope[0]).tolist(),
                   "available_sample": view.available_sample, "filter": filter_receipt,
                   "output": {"shape": list(epochs.shape), "dtype": str(epochs.dtype),
                              "channel_names": list(self.channel_names), "signal_unit": view.signal_unit,
                              "source_reference": view.reference,
                              "reference": (view.reference if self.reference_channel is None
                                            else "measured_channel:" + self.reference_channel)}}
        return epochs, mask, freeze_metadata(receipt)

    def execution_capability(self):
        """A compiled inference capability, distinct from an executed raw receipt."""
        if self.output_dtype != "float32" or (self.preprocessing.baseline_mode == "trial_reference"
                and self.preprocessing.trial_reference_center != "mean"):
            raise ValueError("selection checkpoint capability requires the actual linear float32 transform")
        return {"schema": "n2p3_selection_epoch_capability/1", "profile": self.record(),
                "profile_digest": self.digest, "producer_source_sha256": _SELECTION_EXECUTION_SOURCE_SHA256,
                "operator_family": "finite_segment_sos_closed_grid_fft_first_T_linear_baseline_float32/1"}


def _source_sos_family_relation(actual_sos, recompiled_sos, preprocessing, *, source_rate_hz,
                                actual_padding, recompiled_padding):
    """Admit the declared source family with one named scalar-gain relation.

    Saved source coefficients retain their exact identity. This policy concerns
    source-recipe recompilation only; target execution and replay stay exact.
    """
    actual = np.asarray(actual_sos, dtype=np.float64)
    compiled = np.asarray(recompiled_sos, dtype=np.float64)
    if actual.shape != compiled.shape or actual_padding != recompiled_padding:
        raise ValueError("source finite SOS programme differs from the compiled selection family")
    receipt = {"schema": "n2p3_source_sos_family_relation/1", "source_rate_hz": source_rate_hz,
        "source_sos_sha256": array_digest(actual), "recompiled_sos_sha256": array_digest(compiled),
        "padding_limit_samples": actual_padding, "coefficient_replacement_performed": False}
    if np.array_equal(actual, compiled):
        return {**receipt, "policy": "exact_coefficients/1"}
    gain_row = len(preprocessing.notch_freqs_hz)
    supported = (preprocessing.filter_method == "iir" and preprocessing.filter_order == 4
        and preprocessing.l_freq is not None and preprocessing.h_freq is not None
        and 0 < preprocessing.l_freq < preprocessing.h_freq < source_rate_hz / 2
        and actual.shape == (gain_row + 4, 6))
    if not supported:
        raise ValueError("source finite SOS programme differs from the compiled selection family")
    old_gain, new_gain = actual[gain_row, 0], compiled[gain_row, 0]
    tiny = np.finfo(np.float64).tiny
    normal_gains = np.isfinite(old_gain) and np.isfinite(new_gain) and old_gain >= tiny and new_gain >= tiny
    scalar_shape = (normal_gains and np.array_equal(actual[gain_row, :3], [old_gain, 2 * old_gain, old_gain])
                    and np.array_equal(compiled[gain_row, :3], [new_gain, 2 * new_gain, new_gain]))
    mask = np.ones(actual.shape, dtype=bool)
    mask[gain_row, :3] = False
    if not scalar_shape or not np.array_equal(actual[mask], compiled[mask]):
        raise ValueError("source finite SOS programme differs from the compiled selection family")
    # Positive normal IEEE-754 values have an ordered uint64 encoding. Count
    # representable scalar-gain steps directly, including exponent boundaries.
    ulps = abs(int(old_gain.view(np.uint64)) - int(new_gain.view(np.uint64)))
    if ulps > 4:
        raise ValueError("source finite SOS programme differs from the compiled selection family")
    return {**receipt, "policy": "butterworth_bandpass_order4_scalar_gain_four_ulp/1",
        "gain_section": gain_row, "source_gain": float(old_gain), "recompiled_gain": float(new_gain),
        "scalar_gain_ulp_distance": ulps, "maximum_scalar_gain_ulp_distance": 4,
        "all_other_coefficients_exact": True, "normalized_gain_shape": "positive_normal_binary64_k_2k_k"}


def validate_selection_checkpoint_execution(dataset, source_execution, preprocessing):
    """Prove numerical-family compatibility without relabelling either ingress.

    Segment lengths and source grids remain their actual recorded supports;
    they never claim the same finite matrix or the same physical observations.
    """
    capability = dataset.provenance.get("selection_epoch_capability")
    if not isinstance(capability, dict) or capability.get("schema") != "n2p3_selection_epoch_capability/1":
        raise ValueError("checkpoint requires actual preprocessing execution evidence or compiled selection capability")
    profile = SelectionEpochProfile.from_record(capability["profile"])
    require_sha256(capability["producer_source_sha256"], "selection execution implementation")
    if (capability["profile_digest"] != profile.digest or profile.preprocessing != preprocessing
            or profile.output_dtype != "float32" or capability["operator_family"] != profile.execution_capability()["operator_family"]
            or tuple(map(canonical_channel_name, dataset.channel_names)) != tuple(map(canonical_channel_name, profile.channel_names))):
        raise ValueError("selection execution capability differs from the actual checkpoint numerical contract")
    sources = validate_preprocessing_execution(source_execution, preprocessing)
    if not sources or any(receipt["filter"]["backend"] not in {"identity", "mne_reflect_limited"} for receipt in sources):
        raise ValueError("source execution family is not supported by the compiled selection transform")
    operators, relations = {}, {}
    for receipt in sources:
        rate = receipt["source_context"]["sample_rate_hz"]
        if rate not in operators:
            operators[rate] = _profile_operator(preprocessing, rate)
        operator = operators[rate]
        actual_sos = np.asarray(receipt["filter"]["sos"])
        if (operator is None) != (not actual_sos.size):
            raise ValueError("source finite SOS programme differs from the compiled selection family")
        if operator is not None:
            relation = _source_sos_family_relation(actual_sos, operator.sos, preprocessing,
                source_rate_hz=rate, actual_padding=receipt["filter"]["padding_limit_samples"],
                recompiled_padding=operator.padding_limit)
            relations[semantic_sha256(relation)] = relation
    if isinstance(dataset, TrialCheckpointTarget):
        # No calibration or preprocessing is claimed here. Every inference
        # transforms an actual PredictionDecisionView through this same profile.
        return {"source_ingress": "mne_raw", "target_ingress": "selection_future_capability",
                "family": capability["operator_family"], "same_finite_support_claimed": False,
                "source_sos_family_relations": list(relations.values())}
    receipts = dataset.provenance.get("selection_epoch_execution")
    records = dataset.provenance.get("source_records")
    if not receipts or not records or len(receipts) != len(records):
        raise ValueError("selection calibration requires its actual executed raw observations")
    observed = (np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]) if dataset.trial_channel_mask is None
                else dataset.trial_channel_mask)
    offset = 0
    for receipt, raw in zip(receipts, records, strict=True):
        count = receipt["output"]["shape"][0]
        values, mask = dataset.X[offset:offset + count], observed[offset:offset + count]
        output = receipt["output"]
        physical = {key: value for key, value in raw.items() if key not in {"physical_support_digest", "raw_input_digest"}}
        expected_reference = profile.reference if profile.reference_channel is None else "measured_channel:" + profile.reference_channel
        if (receipt["schema"] != "n2p3_selection_epoch_execution/1" or receipt["profile_digest"] != profile.digest
                or semantic_sha256(receipt["profile"]) != profile.digest
                or receipt["physical_support_digest"] != raw["physical_support_digest"]
                or receipt["raw_input_digest"] != raw["raw_input_digest"]
                or tuple(output["shape"]) != (count, len(profile.channel_names), profile.preprocessing.n_times)
                or output["dtype"] != "float32" or tuple(output["channel_names"]) != profile.channel_names
                or output["reference"] != expected_reference or output["source_reference"] != raw["reference"]
                or semantic_sha256(physical) != raw["physical_support_digest"]
                or receipt["numerical_input_digest"] != semantic_sha256({"profile_digest": profile.digest,
                    "physical_support_digest": raw["physical_support_digest"], "values_sha256": array_digest(values, mask)})
                or (receipt["filter"]["operator"] is None) != (profile.operator is None)
                or profile.operator is not None and semantic_sha256(receipt["filter"]["operator"]) != profile.operator.digest):
            raise ValueError("selection execution receipt differs from its actual transformed calibration bytes")
        scope = raw["raw_scope"]
        spans = np.asarray(receipt["source_epoch_scopes"])
        expected_spans = np.column_stack((np.asarray(raw["onsets"]) + profile._source_start,
                                         np.asarray(raw["onsets"]) + profile._source_stop)) + scope[0]
        if (spans.shape != (count, 2) or np.any(spans[:, 0] < scope[0]) or np.any(spans[:, 1] > scope[1])
                or not np.array_equal(spans, expected_spans) or receipt["epoch_geometry"] != profile.epoch_geometry
                or receipt["available_sample"] != raw["available_sample"]
                or profile.operator is not None and receipt["filter"]["padding_samples"] != min(
                    profile.operator.padding_limit, scope[1] - scope[0] - 1)):
            raise ValueError("selection execution extends outside its actual authorized observation support")
        offset += count
    if offset != dataset.n_epochs:
        raise ValueError("selection execution receipts do not cover the actual calibration rows")
    return {"source_ingress": "mne_raw", "target_ingress": "selection_executed_segments",
            "family": capability["operator_family"], "same_finite_support_claimed": False,
            "source_sos_family_relations": list(relations.values())}


@dataclass(frozen=True)
class CalibrationGroupViews:
    groups: tuple[CalibrationSelection, ...]
    ledger: dict

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "ledger", freeze_metadata(self.ledger))


@dataclass(frozen=True)
class TrialCheckpointTarget:
    """Physical target input metadata for K0, with no target values or statistics."""

    preprocessing: PreprocessingSpec
    channel_names: tuple[str, ...]
    identity_table: DatasetIdentityTable
    provenance: dict

    @property
    def n_channels(self):
        return len(self.channel_names)

    @property
    def n_times(self):
        return self.preprocessing.n_times


class _FrozenPrefixDataset(EpochDataset):
    """Validated fit input: sealed numeric arrays and copy-on-read table records."""

    @classmethod
    def from_dataset(cls, data):
        materialize_dataset_identity(data)
        materialize_dataset_lineage(data)
        result = object.__new__(cls)
        for item in fields(EpochDataset):
            if item.name in {"metadata", "provenance"}:
                continue
            value = getattr(data, item.name)
            if isinstance(value, np.ndarray):
                value = _immutable_array(value.astype(str) if value.dtype.hasobject else value,
                                         str if value.dtype.hasobject else value.dtype)
            object.__setattr__(result, item.name, value)
        timeline = replace(data.event_timeline)
        for item in fields(timeline):
            value = getattr(timeline, item.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(timeline, item.name, _immutable_array(value, value.dtype))
        if result.qc_features is not None:
            quality = result.qc_features
            object.__setattr__(result, "qc_features", replace(quality, **{
                item.name: _immutable_array(getattr(quality, item.name), getattr(quality, item.name).dtype)
                for item in fields(quality) if isinstance(getattr(quality, item.name), np.ndarray)}))
        object.__setattr__(result, "event_timeline", timeline)
        object.__setattr__(result, "_prefix_metadata", data.metadata.copy(deep=True))
        object.__setattr__(result, "_prefix_provenance", deepcopy(data.provenance))
        return result

    def __setattr__(self, name, value):
        raise TypeError("authorized prefix input is frozen")

    @property
    def metadata(self):
        return self._prefix_metadata.copy(deep=True)

    @property
    def provenance(self):
        return deepcopy(self._prefix_provenance)


@dataclass(frozen=True)
class CalibrationRecordScope:
    """A real acquisition record's C prefix, in that record's native clock.

    Global label indices order permission across files; they never turn files
    into one continuous sample axis or authorize a noise/flat-state carryover.
    """

    record_id: str
    source_sha256: str
    source_relative_path: str
    source_dataset_id: str
    source_subject_id: str
    read_sample_support: tuple[int, int]
    global_label_indices: tuple[int, int]
    available_sample: int
    authorized_native_onsets: tuple[int, ...]
    native_clock: str = "actual_record_native_sample"

    def __post_init__(self):
        for name in ("record_id", "source_relative_path", "source_dataset_id", "source_subject_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("record calibration scope requires exact physical source identities")
        require_sha256(self.source_sha256, "original acquisition record")
        start, stop = self.read_sample_support
        lo, hi = self.global_label_indices
        if (any(type(x) is not int for x in (start, stop, lo, hi, self.available_sample))
                or start != 0 or stop <= 0 or not 0 <= lo < hi
                or self.available_sample != stop or self.native_clock != "actual_record_native_sample"
                or len(self.authorized_native_onsets) != hi-lo
                or any(type(x) is not int for x in self.authorized_native_onsets)
                or any(a >= b for a, b in zip(self.authorized_native_onsets[:-1], self.authorized_native_onsets[1:], strict=True))
                or not 0 <= self.authorized_native_onsets[0] < self.authorized_native_onsets[-1]+1 <= stop):
            raise ValueError("actual record clock/read boundary and global C-label authority differ")

    def record(self):
        return {"schema": "n2p3_calibration_record_scope/1", **{
            item.name: list(value) if isinstance(value, tuple) else value
            for item in fields(self) for value in (getattr(self, item.name),)}}

    @classmethod
    def from_record(cls, value):
        values = dict(value)
        if values.pop("schema") != "n2p3_calibration_record_scope/1":
            raise ValueError("unknown actual acquisition C-scope schema")
        for name in ("read_sample_support", "global_label_indices", "authorized_native_onsets"):
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class CalibrationPrefix:
    """One authorized physical prefix, with distinct raw and epoch label use."""

    budget: int
    raw: CalibrationRecord | None
    trial_datasets: dict[str, EpochDataset]
    complete_groups: tuple[CalibrationSelection, ...]
    target_contract: dict
    label_use: dict
    native_records: tuple[CalibrationRecord, ...] = ()
    _prepared_capabilities: dict | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._validate_authority()
        if self.raw is not None:
            object.__setattr__(self, "raw", replace(self.raw,
                samples=_immutable_array(self.raw.samples, self.raw.samples.dtype),
                onsets=_immutable_array(self.raw.onsets, self.raw.onsets.dtype),
                target_mask=_immutable_array(self.raw.target_mask, bool)))
        object.__setattr__(self, "native_records", tuple(replace(raw,
            samples=_immutable_array(raw.samples, raw.samples.dtype), onsets=_immutable_array(raw.onsets, raw.onsets.dtype),
            target_mask=_immutable_array(raw.target_mask, bool)) for raw in self.native_records))
        object.__setattr__(self, "trial_datasets", freeze_metadata({
            key: _FrozenPrefixDataset.from_dataset(value) for key, value in self.trial_datasets.items()}))
        object.__setattr__(self, "complete_groups", tuple(self.complete_groups))
        object.__setattr__(self, "target_contract", freeze_metadata(self.target_contract))
        object.__setattr__(self, "label_use", freeze_metadata(self.label_use))

    def validate(self):
        """Construction checked and froze authority; repeated fits share that result."""
        return self

    def _validate_authority(self):
        """Bind actual dosage, row identity and read support once at ingress."""
        target = self.target_contract
        if target.get("authority_kind") == "record_aware_calibration_prefix/1":
            return self._validate_record_authority()
        if (target.get("authority_kind") is not None or self.native_records
                or {"record_scopes", "source_record_roster"}.intersection(target)
                or {"record_scopes", "unavailable_records"}.intersection(self.label_use["raw"])):
            raise ValueError("legacy prefix cannot authorize record-only numerical inputs or scopes")
        start, cutoff = target["raw_scope"]
        if type(self.budget) is not int or self.budget < 0 or start != 0:
            raise ValueError("prefix authorization requires a chronological nonnegative K")
        if self.budget == 0:
            if self.raw is not None or self.trial_datasets or cutoff != 0 or self.complete_groups:
                raise ValueError("K0 has no target calibration observations, labels or groups")
            if (self.label_use["raw"]["used_labels"] or self.label_use["raw"]["requested_labels"]
                    or self.label_use["raw"]["read_sample_support"] is not None
                    or any(use["used_labels"] or use["requested_labels"]
                        or use["used_label_indices"] for use in self.label_use["trial"].values())):
                raise ValueError("K0 label ledger cannot authorize target observations")
            return self
        raw = self.raw
        if (raw is None or not raw.right_censored or raw.samples.shape != (cutoff, len(target["channel_names"]))
                or raw.record_id != f"{target['participant_key']}:C{self.budget}:{target['source_sha256']}"
                or raw.samples.dtype.kind not in "fi" or not np.isfinite(raw.samples).all()
                or len(raw.target_mask) != self.budget or raw.onsets.shape != (self.budget,)
                or raw.target_mask.dtype != bool or raw.onsets.dtype.kind not in "iu"
                or np.any(np.diff(raw.onsets) <= 0) or raw.onsets[0] < 0 or raw.onsets[-1] >= cutoff):
            raise ValueError("actual raw calibration labels/onsets/read support differ from authorized K/cutoff")
        raw_use = self.label_use["raw"]
        authorization = RawSampleScope.from_record(raw_use["authorization_scope"])
        if (raw_use["requested_labels"] != self.budget or raw_use["used_labels"] != self.budget
                or tuple(raw_use["read_sample_support"]) != (0, cutoff)
                or raw_use["available_sample"] != cutoff or not raw_use["right_censored"]
                or authorization.event_indices != tuple(range(self.budget))
                or authorization.start_sample != 0
                or cutoff != min(authorization.stop_sample, raw_use["next_unknown_flash_sample"])):
            raise ValueError("raw label ledger differs from actual authorized prefix")
        for digest, data in self.trial_datasets.items():
            use = self.label_use["trial"][digest]
            profile = SelectionEpochProfile.from_record(use["profile"])
            physical = self.trial_target(profile)
            data.validate(require_labels=True)
            actual = np.asarray(data.metadata["flash_sample_matlab_1based"], dtype=np.int64) - 1
            expected = (raw.onsets + profile._source_start >= 0) & (raw.onsets + profile._source_stop <= cutoff)
            indices = np.flatnonzero(expected)
            identity = materialize_dataset_identity(data)
            execution = data.provenance["preprocessing_execution"]["source_context"]["raw_scope"]["declaration"]
            if (digest != profile.digest or not np.array_equal(actual, raw.onsets[indices])
                    or not np.array_equal(data.y, raw.target_mask[indices])
                    or tuple(use["used_label_indices"]) != tuple(indices) or use["used_labels"] != len(indices)
                    or tuple(use["unused_label_indices"]) != tuple(np.flatnonzero(~expected))
                    or use["requested_labels"] != self.budget or tuple(use["read_sample_support"]) != (0, cutoff)
                    or use["available_sample"] != cutoff
                    or tuple(map(tuple, use["source_epoch_scopes"])) != tuple(zip(
                        raw.onsets[indices] + profile._source_start,
                        raw.onsets[indices] + profile._source_stop, strict=True))
                    or execution["start_sample"] != 0 or execution["stop_sample"] != cutoff
                    or tuple(execution["event_indices"]) != tuple(range(self.budget))
                    or semantic_sha256(use["preprocessing_execution"]) != semantic_sha256(data.provenance["preprocessing_execution"])
                    or data.preprocessing.record() != profile.preprocessing.record()
                    or data.channel_names != physical.channel_names
                    or data.provenance["source_reference"] != physical.provenance["source_reference"]
                    or data.provenance["source_file_sha256"] != target["source_sha256"]
                    or identity.authority_keys() != (target["participant_key"],)
                    or use["numerical_input_digest"] != array_digest(data.X, data.y, data.trial_channel_mask)):
                raise ValueError("actual trial rows/labels/profile/source identity exceed or differ from prefix authorization")
        for group in self.complete_groups:
            view = group.view
            indices = np.searchsorted(raw.onsets, view.onsets + view.raw_scope[0])
            if (view.source_sha256 != target["source_sha256"] or view.participant_key != target["participant_key"]
                    or not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0)
                    or view.raw_scope[0] < 0 or view.raw_scope[1] > cutoff or view.samples is None
                    or np.any(indices >= self.budget) or not np.array_equal(raw.onsets[indices], view.onsets + view.raw_scope[0])
                    or not np.array_equal(group.labels, raw.target_mask[indices])
                    or not np.array_equal(view.samples, raw.samples[slice(*view.raw_scope)])):
                raise ValueError("temperature groups must be complete observations inside actual C authority")
        return self

    @property
    def calibration_records(self):
        return self.native_records if self.native_records else (() if self.raw is None else (self.raw,))

    def fit_receipts(self):
        if self.target_contract.get("authority_kind") != "record_aware_calibration_prefix/1":
            return [] if self.raw is None else [{**self.target_contract, "record_id": self.raw.record_id,
                "raw_input_digest": array_digest(self.raw.samples, self.raw.onsets, self.raw.target_mask),
                "label_use": self.label_use}]
        scopes = self.target_contract["record_scopes"]
        return [{**self.target_contract, **scopes[raw.record_id], "record_id": raw.record_id,
            "raw_scope": list(scopes[raw.record_id]["read_sample_support"]),
            "raw_input_digest": array_digest(raw.samples, raw.onsets, raw.target_mask),
            "label_use": self.label_use} for raw in self.calibration_records]

    def _validate_record_authority(self):
        target = self.target_contract
        if type(self.budget) is not int or self.budget < 0 or self.raw is not None or target["raw_scope"] is not None:
            raise ValueError("record-aware C uses global label K and separate actual native records")
        scopes = {key: CalibrationRecordScope.from_record(value) for key, value in target["record_scopes"].items()}
        if set(scopes) != {r.record_id for r in self.native_records}:
            raise ValueError("C numerical records differ from their actual acquisition scope roster")
        used = []
        for raw in self.native_records:
            scope = scopes[raw.record_id]
            if (raw.samples.shape != (scope.read_sample_support[1], len(target["channel_names"]))
                    or raw.target_mask.dtype != bool or not np.isfinite(raw.samples).all()
                    or not np.array_equal(raw.onsets, scope.authorized_native_onsets)
                    or raw.target_mask.shape != raw.onsets.shape):
                raise ValueError("actual record C samples/onsets/labels exceed its source-owned permission")
            used.extend(range(*scope.global_label_indices))
        raw_use = self.label_use["raw"]
        if (raw_use["requested_labels"] != self.budget or raw_use["used_labels"] != len(used)
                or tuple(raw_use["used_label_indices"]) != tuple(used) or any(i >= self.budget for i in used)):
            raise ValueError("global requested C-label dose differs from actual per-record authorized rows")
        for digest, data in self.trial_datasets.items():
            use = self.label_use["trial"][digest]
            data.validate(require_labels=True)
            indices = tuple(map(int, data.metadata["global_label_index"]))
            if (indices != tuple(use["used_label_indices"]) or use["used_labels"] != len(indices)
                    or use["requested_labels"] != self.budget or any(i not in used for i in indices)
                    or use["numerical_input_digest"] != array_digest(data.X, data.y, data.trial_channel_mask)):
                raise ValueError("record-aware epoch labels/numerical rows differ from original C permissions")
        for group in self.complete_groups:
            key = group.view.source_sha256
            candidates = [r for r in self.native_records if scopes[r.record_id].source_sha256 == key]
            if len(candidates) != 1:
                raise ValueError("C group belongs to another acquisition record")
            raw = candidates[0]
            absolute = group.view.onsets+group.view.raw_scope[0]
            indices = np.searchsorted(raw.onsets, absolute)
            if (group.view.samples is None or not group.view.decision.schedule_complete
                    or np.any(indices >= len(raw.onsets)) or not np.array_equal(raw.onsets[indices], absolute)
                    or not np.array_equal(group.labels, raw.target_mask[indices])
                    or group.view.raw_scope[1] > len(raw.samples)
                    or not np.array_equal(group.view.samples, raw.samples[slice(*group.view.raw_scope)])):
                raise ValueError("complete C group differs from its actual record-local fit observations")
        return self

    def trial_target(self, profile):
        return _trial_target_from_contract(self.target_contract, profile, capabilities=self._prepared_capabilities)

    def record(self):
        return {"budget": self.budget, "target_contract": self.target_contract,
                "label_use": self.label_use,
                "temperature_group_records": [group.record() for group in self.complete_groups]}


@dataclass(frozen=True)
class IndependentSubjectViews:
    calibration: dict[int, CalibrationPrefix]
    decisions: tuple[PredictionDecisionView, ...]
    scoring_truth: dict[str, str]
    ledger: dict
    prefix_compiler: CalibrationPrefixCompiler | None = None


def _trial_target_from_contract(contract, profile, *, capabilities=None):
    """Compile the same metadata-only checkpoint target at fit and package load."""
    channels = tuple(map(canonical_channel_name, profile.channel_names))
    native = tuple(map(canonical_channel_name, contract["channel_names"]))
    required = set(channels)
    if profile.reference_channel is not None:
        required.add(profile.reference_channel)
    missing = required-set(native)
    if (profile.source_sample_rate_hz != contract["sample_rate_hz"]
            or profile.reference != contract["reference"]
            or missing and (not profile.allow_missing_channels or profile.reference_channel in missing)):
        raise ValueError("prefix profile must use the actual measured native channels/reference")
    reference = ("measured_channel:" + profile.reference_channel if profile.reference_channel is not None else
                 contract["reference"] if channels == native else "native:" + contract["reference"])
    capability = profile.execution_capability() if capabilities is None else capabilities[profile.digest]
    return TrialCheckpointTarget(profile.preprocessing, channels,
        DatasetIdentityTable.from_source_rows([contract["source_subject_id"]], [contract["source_dataset_id"]]),
        {"source_reference": reference, "selection_epoch_capability": capability})


def _bnci_prediction_view(record, indices, start, stop, *, available_sample, reason=None,
                          evidence_indices=None):
    decision = MembershipDecision.row_column(record.candidate_id[indices],
        n_rows=BNCI2014_008_GRID_SHAPE[0], n_columns=BNCI2014_008_GRID_SHAPE[1],
        event_ids=tuple(f"{BNCI2014_008_DATASET_ID}:{record.subject_id}:{int(row)}" for row in indices),
        repetition_indices=record.repetition_index[indices], repetitions=BNCI2014_008_REPETITIONS_PER_SELECTION,
        evidence_indices=evidence_indices)
    decision_id = str(record.selection_id[indices[0]])
    return PredictionDecisionView(
        record_id=f"{BNCI2014_008_DATASET_ID}:{decision_id}:{record.source_sha256}",
        participant_key=origin_subject_key(BNCI2014_008_DATASET_ID, record.subject_id),
        decision_id=decision_id,
        samples=None if reason else np.asarray(record.eeg_uv[start:stop], dtype=np.float64) * 1e-6,
        onsets=record.flash_sample[indices] - start, decision=decision, source_sha256=record.source_sha256,
        raw_scope=(start, stop), sample_rate_hz=BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        channel_names=BNCI2014_008_CHANNELS, channel_types=("EEG",) * len(BNCI2014_008_CHANNELS),
        reference=BNCI2014_008_SOURCE_REFERENCE, source_relative_path=record.source_relative_path,
        source_dataset_id=BNCI2014_008_DATASET_ID, source_subject_id=record.subject_id,
        available_sample=available_sample, unavailable_reason=reason)


def _compile_bnci_calibration_prefix(record, raw, events, budget, *, maximum, profiles,
                                     authorization_preprocessing, width, preroll, target):
    source_sha256 = record.source_sha256
    selections = [np.flatnonzero(record.selection_index == selection)
                  for selection in np.unique(record.selection_index)]
    datasets, groups = {}, []
    authorized, _ = prefix_raw_scopes(raw, events, authorization_preprocessing,
        calibration_events=budget, maximum_calibration_events=maximum) if budget else (None, None)
    cutoff = min(authorized.stop_sample, int(record.flash_sample[budget])) if budget else 0
    prefix_contract = {**target, "raw_scope": [0, cutoff], "available_sample": cutoff}
    fitted_raw = (CalibrationRecord(f"{target['participant_key']}:C{budget}:{source_sha256}",
        np.asarray(record.eeg_uv[:cutoff], dtype=np.float64) * 1e-6,
        record.flash_sample[:budget], record.raw_is_target[:budget], right_censored=True) if budget else None)
    label_use = {"raw": {"requested_labels": budget, "used_labels": budget, "unused_label_indices": [],
        "authorization_scope": None if authorized is None else authorized.record(),
        "next_unknown_flash_sample": int(record.flash_sample[budget]) if budget else None,
        "read_sample_support": [0, cutoff] if budget else None, "available_sample": cutoff,
        "right_censored": True, "observed_response_samples":
            np.minimum(width, cutoff - record.flash_sample[:budget]).tolist()}, "trial": {}}
    prefix = CalibrationPrefix(budget, fitted_raw, datasets, tuple(groups), prefix_contract, label_use)
    for digest, profile in profiles.items():
        prefix.trial_target(profile)
        epoch_starts = record.flash_sample[:budget] + profile._source_start
        epoch_stops = record.flash_sample[:budget] + profile._source_stop
        available = (epoch_starts >= 0) & (epoch_stops <= cutoff)
        used = np.flatnonzero(available)
        receipt = {"requested_labels": budget, "used_labels": len(used), "used_label_indices": used.tolist(),
            "unused_label_indices": np.flatnonzero(~available).tolist(), "profile": profile.record(),
            "read_sample_support": [0, cutoff] if len(used) else None, "available_sample": cutoff,
            "source_epoch_scopes": np.column_stack((epoch_starts[available], epoch_stops[available])).tolist()}
        if len(used):
            scope = RawSampleScope(f"recording:C{budget}-before-next-flash", 0, cutoff, tuple(range(budget)))
            data = build_bnci2014_008_subject_dataset(record, preprocessing=profile.preprocessing,
                raw_scope=scope, raw=raw, output_dtype=profile.output_dtype)
            if tuple(map(canonical_channel_name, profile.channel_names)) != data.channel_names or profile.reference_channel is not None:
                data = adapt_observed_montage(data, profile.channel_names, reference_channel=profile.reference_channel)
            materialize_dataset_identity(data)
            if not np.array_equal(data.metadata["flash_sample_matlab_1based"].to_numpy() - 1,
                                  record.flash_sample[used]):
                raise ValueError("actual trial preprocessing evidence differs from prefix availability")
            datasets[digest] = data
            receipt.update(numerical_input_digest=array_digest(data.X, data.y, data.trial_channel_mask),
                preprocessing_execution=data.provenance["preprocessing_execution"],
                actual_reference=data.provenance["source_reference"])
        label_use["trial"][digest] = receipt
    for indices in selections:
        first, last = int(record.flash_sample[indices[0]]), int(record.flash_sample[indices[-1]])
        start, stop = first - preroll, last + width
        if indices[-1] >= budget or start < 0 or stop > cutoff:
            continue
        if ((indices[0] > 0 and record.flash_sample[indices[0] - 1] + width > start)
                or (indices[-1] + 1 < len(events) and record.flash_sample[indices[-1] + 1] < stop)):
            continue
        view = _bnci_prediction_view(record, indices, start, stop, available_sample=stop)
        truth = row_column_target(int(record.target_row[indices[0]]), int(record.target_col[indices[0]]))
        groups.append(CalibrationSelection(view, record.raw_is_target[indices], truth))
    return CalibrationPrefix(budget, fitted_raw, datasets, tuple(groups), prefix_contract, label_use)


def calibration_prefix_identity(prefix):
    """Actual immutable observations and labels, beyond a presentation digest."""
    identity = {"prefix": prefix.record(),
        "raw": None if prefix.raw is None else array_digest(
            prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask),
        "trial": {key: array_digest(data.X, data.y, data.trial_channel_mask)
                  for key, data in prefix.trial_datasets.items()}}
    if prefix.native_records:
        identity["native_records"] = [{"record_id": raw.record_id,
            "input_sha256": array_digest(raw.samples, raw.onsets, raw.target_mask)} for raw in prefix.native_records]
    return semantic_sha256(identity)


def _prefix_owner_snapshot(prefix):
    """Public frozen records and array headers, without reading numeric payloads."""
    arrays = []

    def snapshot(value):
        if isinstance(value, np.ndarray):
            arrays.append(value)
            return ("array", len(arrays) - 1)
        if isinstance(value, dict):
            return (type(value), tuple((key, snapshot(item)) for key, item in value.items()))
        if isinstance(value, (tuple, list)):
            return (type(value), tuple(snapshot(item) for item in value))
        if is_dataclass(value):
            # Tables and provenance are exposed only by copy-on-read. Their
            # authorization relationship already lives in immutable label_use.
            omitted = {"metadata", "provenance"} if isinstance(value, _FrozenPrefixDataset) else set()
            return (type(value), tuple((item.name, snapshot(getattr(value, item.name)))
                                      for item in fields(value) if item.name not in omitted))
        return (type(value), value)

    return snapshot(prefix), tuple(arrays)


class _OwnedCalibrationPrefix:
    """The compiler's original numerical identity and immutable ingress geometry."""

    def __init__(self, prefix, identity):
        self.prefix, self.identity = prefix, identity
        self.structure, arrays = _prefix_owner_snapshot(prefix)
        self.arrays = tuple((array, *immutable_input_array_geometry(array)) for array in arrays)
        self._group_digests = None

    def assert_unchanged(self, prefix):
        structure, arrays = _prefix_owner_snapshot(prefix)
        if prefix is not self.prefix or structure != self.structure or len(arrays) != len(self.arrays):
            raise ValueError("owned prefix physical or label authority changed after compiler admission")
        for array, (original, backing, geometry) in zip(arrays, self.arrays, strict=True):
            current_backing, current_geometry = immutable_input_array_geometry(array)
            if array is not original or current_backing is not backing or current_geometry != geometry:
                raise ValueError("owned prefix array header or immutable backing changed after compiler admission")

    def assert_group_match(self, held):
        """Match complete record digests after the caller's owned geometry gate."""
        if self._group_digests is None:
            owned, occurrences = {}, {}
            for group in self.prefix.complete_groups:
                cached = owned.get(id(group))
                digest = semantic_sha256(group.record()) if cached is None else cached[1]
                owned[id(group)] = (group, digest)
                occurrences[digest] = occurrences.get(digest, 0) + 1
            self._group_digests = owned, occurrences
        owned, occurrences = self._group_digests
        if not occurrences:
            raise ValueError("held group differs from original parent group authority")
        cached = owned.get(id(held))
        digest = (cached[1] if cached is not None and cached[0] is held
                  else semantic_sha256(held.record()))
        if occurrences.get(digest, 0) != 1:
            raise ValueError("held group differs from original parent group authority")


class CalibrationPrefixCompiler:
    """Run-local original-source owner for genuinely earlier C prefixes.

    The public builder installs the decoded MAT and its declared input programme.
    Every derived scope is compiled before preprocessing; a filtered parent epoch
    cache is never cropped to impersonate an earlier experiment.
    """

    def __init__(self, payload, *, source_sha256, source_relative_path, budgets, profiles,
                 authorization_preprocessing, response_seconds=1.2, pre_roll_seconds=.2):
        if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != source_sha256:
            raise ValueError("prefix compiler requires the actual original verified MAT bytes")
        record = read_bnci2014_008_candidate_record(io.BytesIO(payload),
            source_relative_path=source_relative_path, source_sha256=source_sha256)
        profiles = {profile.digest: profile for profile in profiles}
        rate = BNCI2014_008_SOURCE_SAMPLE_RATE_HZ
        width = max([round(response_seconds * rate), *(profile._source_stop for profile in profiles.values())])
        preroll = max([round(pre_roll_seconds * rate), *(-profile._source_start for profile in profiles.values())])
        raw, events, _ = bnci2014_008_raw(record)
        starts, stops = source_epoch_support(events, rate, authorization_preprocessing)
        splits, adjustments = resolve_calibration_splits(budgets,
            lambda count, maximum: binary_time_split(events[:, 0] / rate, stops / rate,
                calibration_trials=count, maximum_calibration_trials=maximum,
                tmin_s=authorization_preprocessing.tmin_ms / 1000),
            minimum_test_trials=1, available_trials=len(events))
        target = {"participant_key": origin_subject_key(BNCI2014_008_DATASET_ID, record.subject_id),
            "candidate_vocabulary": [row_column_target(row, col) for row in range(6) for col in range(6)],
            "sample_rate_hz": rate, "channel_names": list(BNCI2014_008_CHANNELS),
            "channel_types": ["EEG"] * len(BNCI2014_008_CHANNELS), "reference": BNCI2014_008_SOURCE_REFERENCE,
            "signal_unit": "V", "source_sha256": source_sha256, "source_dataset_id": BNCI2014_008_DATASET_ID,
            "source_subject_id": record.subject_id, "source_relative_path": record.source_relative_path}
        self._record, self._raw, self._events = record, raw, events
        self._starts, self._stops = starts, stops
        self._splits, self._adjustments = splits, adjustments
        self._arguments = {"maximum": max(splits), "profiles": profiles,
            "authorization_preprocessing": authorization_preprocessing,
            "width": width, "preroll": preroll, "target": target}
        self._prefixes, self._prefix_ownership = {}, {}
        self._history_source_payload = payload
        self._history_source_identity = array_digest(record.eeg_uv, record.flash_sample,
            record.candidate_id, record.selection_index, record.repetition_index)
        self._history_ingress = freeze_metadata({"source_sha256": source_sha256,
            "source_relative_path": source_relative_path, "budgets": list(budgets),
            "profiles": [profile.record() for profile in profiles.values()],
            "authorization_preprocessing": authorization_preprocessing.record(),
            "response_seconds": response_seconds, "pre_roll_seconds": pre_roll_seconds})

    def compile(self, budget):
        if type(budget) is not int or not 0 <= budget <= self._arguments["maximum"]:
            raise ValueError("derived prefix exceeds original parent authorization")
        if budget not in self._prefixes:
            prefix = _compile_bnci_calibration_prefix(
                self._record, self._raw, self._events, budget, **self._arguments)
            ownership = _OwnedCalibrationPrefix(prefix, calibration_prefix_identity(prefix))
            self._prefixes[budget], self._prefix_ownership[budget] = prefix, ownership
        else:
            self._prefix_ownership[budget].assert_unchanged(self._prefixes[budget])
        return self._prefixes[budget]

    @classmethod
    def from_dataset_source(cls, declaration, **arguments):
        """Dispatch only registered actual source formats into the shared issuer."""
        from data.candidate_input import resolve_candidate_dataset_adapter

        return RecordCalibrationPrefixCompiler(resolve_candidate_dataset_adapter(declaration), **arguments)

    def assert_original_prefix(self, prefix):
        if not isinstance(prefix, CalibrationPrefix):
            raise TypeError("prefix compiler requires original CalibrationPrefix authority")
        actual = self.compile(prefix.budget)
        ownership = self._prefix_ownership[prefix.budget]
        if prefix is not actual and calibration_prefix_identity(prefix) != ownership.identity:
            raise ValueError("actual parent differs from original source prefix owner")
        return actual

    @property
    def history_owner_source(self):
        return _SELECTION_EXECUTION_SOURCE_SHA256

    def history_scope(self, view, *, prefix, model, law):
        """Opt-in history from this exact decoder and original C/held authority.

        Existing current-only views/gates are unchanged. Future actual clocks
        and targets never leave this owner; the new capability owns only the
        already occurred public event prefix and the original observed block.
        """
        from research.history_authority import _issue_history_scope

        if (hashlib.sha256(self._history_source_payload).hexdigest() != self._record.source_sha256
                or self._history_source_identity != array_digest(self._record.eeg_uv,
                    self._record.flash_sample, self._record.candidate_id, self._record.selection_index,
                    self._record.repetition_index)):
            raise ValueError("original history decoder operands changed after source-byte admission")
        return _issue_history_scope(self, view, prefix=prefix, model=model, law=law)

    def before_group(self, parent, held):
        parent = self.assert_original_prefix(parent)
        self._prefix_ownership[parent.budget].assert_group_match(held)
        absolute = held.view.onsets + held.view.raw_scope[0]
        indices = np.searchsorted(parent.raw.onsets, absolute)
        if (np.any(indices >= parent.budget)
                or not np.array_equal(parent.raw.onsets[indices], absolute)
                or not np.array_equal(parent.raw.target_mask[indices], held.labels)):
            raise ValueError("held group labels/onsets differ from actual original parent")
        earlier = self.compile(int(indices[0]))
        if earlier.target_contract["raw_scope"][1] > held.view.raw_scope[0]:
            raise ValueError("earlier prefix raw observations overlap held group support")
        if any(group.view.decision_id == held.view.decision_id for group in earlier.complete_groups):
            raise ValueError("earlier prefix retains held selection labels")
        return earlier

    def c_latent_fit_scope(self, prefix, *, dictionary, prior, noise, plan, innovation_covariance):
        """Original native C labels/samples for a new finite joint producer."""
        from research.multichannel_c_latent import _issue_c_latent_fit_scope

        return _issue_c_latent_fit_scope(self, prefix, dictionary=dictionary, prior=prior,
            noise=noise, plan=plan, innovation_covariance=innovation_covariance)

    def c_latent_history_scope(self, view, *, prefix, model, scoring_law=None):
        """Already observed unlabelled context, with score in original held scope."""
        from research.multichannel_c_latent import _issue_c_latent_history_scope

        return _issue_c_latent_history_scope(self, view, prefix=prefix, model=model, scoring_law=scoring_law)

    def record_read_scope(self, view, *, prefix, conditioning_stop, available_stop=None,
                          program="record_prequential_public_history/1"):
        """Issue a history cap, or retain the exact original typed refusal."""
        from research.history_authority import HistoryUnavailableError

        try:
            return self._issue_record_read_scope(view, prefix=prefix, conditioning_stop=conditioning_stop,
                available_stop=available_stop, program=program)
        except HistoryUnavailableError as error:
            _register_record_history_refusal(self, view, prefix=prefix, conditioning_stop=conditioning_stop,
                available_stop=available_stop, program=program, error=error)
            raise

    def _issue_record_read_scope(self, view, *, prefix, conditioning_stop, available_stop=None,
                                 program="record_prequential_public_history/1"):
        """Issue one original MAT acquisition's C/P/T history without changing legacy C IDs."""
        prefix = self.assert_original_prefix(prefix)
        record = self._record
        stop = view.available_sample if available_stop is None else available_stop
        if (program != "record_prequential_public_history/1" or type(conditioning_stop) is not int
                or type(stop) is not int or not 0 < conditioning_stop < stop == view.available_sample
                or stop > len(record.eeg_uv) or view.source_sha256 != record.source_sha256
                or view.source_relative_path != record.source_relative_path
                or view.participant_key != self._arguments["target"]["participant_key"]
                or prefix.target_contract["raw_scope"][1] > conditioning_stop):
            raise ValueError("original MAT C/P/T request differs from its record and label authority")
        rows = np.flatnonzero(record.selection_id == view.decision_id)
        if not len(rows):
            raise ValueError("history current selection is absent from original MAT physical schedule")
        expected = _bnci_prediction_view(record, rows, *view.raw_scope,
            available_sample=stop, reason="physical_geometry_only_no_signal_read")
        original_decision, actual_decision = expected.decision.record(), view.decision.record()
        original_decision.pop("evidence_indices")
        actual_decision.pop("evidence_indices")
        if (original_decision != actual_decision or not np.array_equal(expected.onsets, view.onsets)
                or expected.channel_names != view.channel_names or expected.reference != view.reference
                or expected.sample_rate_hz != view.sample_rate_hz):
            raise ValueError("original MAT public schedule/native coordinates differ from current history view")
        if (hashlib.sha256(self._history_source_payload).hexdigest() != record.source_sha256
                or self._history_source_identity != array_digest(record.eeg_uv, record.flash_sample,
                    record.candidate_id, record.selection_index, record.repetition_index)):
            raise ValueError("original MAT history decoder operands changed after byte admission")
        samples = np.asarray(record.eeg_uv[:stop], dtype=np.float64)*1e-6
        if view.samples is not None and not np.array_equal(samples[slice(*view.raw_scope)], view.samples):
            raise ValueError("current history samples differ from the original MAT acquisition")
        physical_id = f"{view.participant_key}:original_record:{record.source_sha256}"
        scopes = {}
        mapping = {}
        for raw in prefix.calibration_records:
            scope = CalibrationRecordScope(raw.record_id, record.source_sha256, record.source_relative_path,
                BNCI2014_008_DATASET_ID, record.subject_id, (0, len(raw.samples)), (0, len(raw.onsets)),
                len(raw.samples), tuple(map(int, raw.onsets)))
            scopes[raw.record_id] = scope.record()
            mapping[raw.record_id] = physical_id
        public = []
        for selected in np.unique(record.selection_index):
            rows = np.flatnonzero(record.selection_index == selected)
            occurred = record.flash_sample[rows] < stop
            if not occurred.any():
                continue
            decision = _bnci_prediction_view(record, rows, 0, stop,
                available_sample=stop, reason="physical_geometry_only_no_signal_read").decision
            public.append({"selection_id": str(record.selection_id[rows[0]]),
                "candidate_vocabulary": list(decision.candidate_vocabulary),
                "event_ids": list(np.asarray(decision.event_ids)[occurred]),
                "onsets": record.flash_sample[rows[occurred]].tolist(), "membership": decision.membership[occurred].tolist(),
                "complete_requested_current": str(record.selection_id[rows[0]]) == view.decision_id})
        binding = {"schema": "n2p3_authorized_record_public_history/1", "program": program,
            "record_id": physical_id, "source_sha256": record.source_sha256,
            "native_clock": "actual_record_native_sample", "participant_key": view.participant_key,
            "prefix_identity": calibration_prefix_identity(prefix), "C_record_scopes": scopes,
            "C_record_to_original_record": mapping, "C_record_input_sha256": {
                raw.record_id: array_digest(raw.samples, raw.onsets, raw.target_mask) for raw in prefix.calibration_records},
            "read_sample_support": [0, stop], "conditioning_sample_support": [0, conditioning_stop],
            "score_sample_support": [conditioning_stop, stop], "available_sample": stop,
            "current_decision_id": view.decision_id, "public_selections": public,
            "channel_names": list(view.channel_names), "sample_rate_hz": view.sample_rate_hz,
            "reference": view.reference, "signal_unit": "V", "samples_sha256": array_digest(samples),
            "current_view": view.record(), "source_owner_sha256": _SELECTION_EXECUTION_SOURCE_SHA256,
            "original_decoder_receipt": record.source_provenance, "future_truth_consumed": False}
        return _register_record_history(self, samples, prefix, binding)

    def calibration_record_physical_map(self, prefix):
        prefix = self.assert_original_prefix(prefix)
        physical = f"{prefix.target_contract['participant_key']}:original_record:{self._record.source_sha256}"
        return freeze_metadata({"schema": "n2p3_calibration_physical_record_map/1",
            "prefix_identity": calibration_prefix_identity(prefix), "source_owner_sha256": _SELECTION_EXECUTION_SOURCE_SHA256,
            "records": {raw.record_id: {"physical_record_id": physical, "source_sha256": self._record.source_sha256,
                "source_relative_path": self._record.source_relative_path,
                "input_sha256": array_digest(raw.samples, raw.onsets, raw.target_mask)} for raw in prefix.calibration_records}})


class _PreparedCalibrationPrefixCompiler(CalibrationPrefixCompiler):
    """Only the actual K admitted by the external original-producer capsule."""

    def __init__(self, prefixes, *, identities, producer_source_sha256, consumer_source_sha256):
        self._prefixes = dict(prefixes)
        self._prefix_ownership = {budget: _OwnedCalibrationPrefix(prefix, identities[budget])
                                  for budget, prefix in prefixes.items()}
        self.producer_source_sha256 = producer_source_sha256
        self.consumer_source_sha256 = consumer_source_sha256

    def compile(self, budget):
        if type(budget) is not int or budget not in self._prefixes:
            raise ValueError("requested actual K was not compiled by the admitted original producer")
        self._prefix_ownership[budget].assert_unchanged(self._prefixes[budget])
        return self._prefixes[budget]

    def record_read_scope(self, view, **arguments):
        return _prepared_record_read_scope(self, view, **arguments)

    def calibration_record_physical_map(self, prefix):
        prefix = self.assert_original_prefix(prefix)
        maps = getattr(self, "_physical_record_maps", {})
        if prefix.budget not in maps:
            raise ValueError("physical record map was not issued by the admitted original compiler")
        return maps[prefix.budget]


def prepare_bnci_independent_prefixes(path, *, source_sha256, budgets, profiles,
                                      authorization_preprocessing, response_seconds=1.2, pre_roll_seconds=.2):
    """Native MAT → strict C prefixes and one independent requested future queue.

    The original prefix owner determines C authorization and common Kmax embargo.
    Both expert views clip C before the next unknown flash. Boundary trial rows
    remain scheduled but unavailable; raw retains every K label, right censored.
    """
    budgets = tuple(budgets)
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(type(k) is not int or k < 0 for k in budgets)):
        raise ValueError("independent calibration budgets must be distinct nonnegative integers")
    compiler = CalibrationPrefixCompiler(Path(path).read_bytes(), source_sha256=source_sha256,
        source_relative_path=Path(path).name, budgets=budgets, profiles=profiles,
        authorization_preprocessing=authorization_preprocessing,
        response_seconds=response_seconds, pre_roll_seconds=pre_roll_seconds)
    record, raw, events = compiler._record, compiler._raw, compiler._events
    starts, stops = compiler._starts, compiler._stops
    splits, adjustments = compiler._splits, compiler._adjustments
    maximum = compiler._arguments["maximum"]
    width, preroll = compiler._arguments["width"], compiler._arguments["preroll"]
    requested_to_actual = {budget: row["actual_K"] for budget, row in
        calibration_budget_requests(budgets, sorted(splits), adjustments).items()}
    _, test_scope = prefix_raw_scopes(raw, events, authorization_preprocessing,
        calibration_events=0, maximum_calibration_events=maximum) if maximum else (None,
            RawSampleScope("recording:K0-heldout", 0, len(record.eeg_uv), tuple(range(len(events)))))
    later = np.arange(len(events)) >= maximum
    test_available = later & (starts >= test_scope.start_sample) & (stops <= test_scope.stop_sample)
    selections = [np.flatnonzero(record.selection_index == selection) for selection in np.unique(record.selection_index)]
    calibration = {}
    for budget in splits:
        calibration[budget] = compiler.compile(budget)
    decisions, truths = [], {}
    for indices in selections:
        if not np.any(later[indices]):
            continue
        first, last = int(record.flash_sample[indices[0]]), int(record.flash_sample[indices[-1]])
        start, stop = first - preroll, last + width
        reason = None
        if not test_available[indices].all():
            reason = "partial_decision_after_fixed_Kmax_and_original_epoch_support_purge"
        elif start < test_scope.start_sample or stop > test_scope.stop_sample:
            reason = "decision_response_outside_authorized_test_scope"
        elif indices[0] > 0 and record.flash_sample[indices[0] - 1] + width > start:
            reason = "previous_unknown_decision_overlaps_finite_response_support"
        elif indices[-1] + 1 < len(events) and record.flash_sample[indices[-1] + 1] < stop:
            reason = "next_unknown_decision_overlaps_finite_response_support"
        evidence = np.where(test_available[indices], np.arange(len(indices)), -1)
        view = _bnci_prediction_view(record, indices, start, stop,
            available_sample=max(stop, test_scope.start_sample), reason=reason, evidence_indices=evidence)
        decisions.append(view)
        truths[view.decision_id] = row_column_target(int(record.target_row[indices[0]]), int(record.target_col[indices[0]]))
    return IndependentSubjectViews(calibration, tuple(decisions), truths, {
        "source_sha256": source_sha256, "requested_Ks": list(budgets), "maximum_calibration_events": maximum,
        "requested_to_actual_K": {str(key): value for key, value in requested_to_actual.items()},
        "budget_adjustments": {str(key): value for key, value in adjustments.items()},
        "test_budget": splits[maximum].test_budget.record(),
        "calibration": {str(k): prefix.record() for k, prefix in calibration.items()}, "test_scope": test_scope.record(),
        "requested_decisions": len(decisions), "scorable_decisions": sum(view.samples is not None for view in decisions),
        "test_views": [view.record() for view in decisions], "operator_identity": "native_unfiltered_volts/1",
        "filtering_scope": "strict_C_prefix_or_each_independent_test_decision",
        "same_physical_calibration_cutoff_for_all_experts": True}, prefix_compiler=compiler)


def prepare_bnci_calibration_groups(path, *, source_sha256, authorized_labels=800, group_count=6,
                                    response_seconds=1.2, pre_roll_seconds=.2):
    """Build the requested earliest complete groups inside an attested label prefix.

    The prediction half has no labels. Calibrators receive only the explicitly
    authorized labels on ``CalibrationSelection``; an outer evaluator owns any
    held group's truth independently. A small K does not acquire extra groups.
    ``response_seconds`` declares the common observations, so a source profile
    requiring one extra actual sample can request it before any comparison.
    """
    if (isinstance(authorized_labels, bool) or not isinstance(authorized_labels, Integral)
            or authorized_labels < 1 or isinstance(group_count, bool)
            or not isinstance(group_count, Integral) or group_count < 1):
        raise ValueError("authorized labels and complete independent groups must be positive integers")
    if (isinstance(response_seconds, bool) or isinstance(pre_roll_seconds, bool)
            or not np.isfinite((response_seconds, pre_roll_seconds)).all()
            or response_seconds <= 0 or pre_roll_seconds < 0):
        raise ValueError("declared response and preroll support must be finite nonnegative durations")
    require_sha256(source_sha256, "original raw source")
    path = Path(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source_sha256:
        raise ValueError("original raw bytes differ from the frozen BNCI source declaration")
    raw = read_bnci2014_008_candidate_record(io.BytesIO(payload), source_relative_path=path.name,
                                             source_sha256=source_sha256)
    if authorized_labels > len(raw.flash_sample):
        raise ValueError("authorized labels exceed the original physical flash ledger")
    rate = BNCI2014_008_SOURCE_SAMPLE_RATE_HZ
    width, preroll = round(response_seconds * rate), round(pre_roll_seconds * rate)
    if width < 1:
        raise ValueError("declared response support must include at least one actual sample")
    authorized_stop = (int(raw.flash_sample[authorized_labels]) if authorized_labels < len(raw.flash_sample)
                       else len(raw.eeg_uv))
    groups, receipts, used_indices = [], [], []
    for selection in np.unique(raw.selection_index):
        indices = np.flatnonzero(raw.selection_index == selection)
        if indices[-1] >= authorized_labels:
            break
        if len(indices) != BNCI2014_008_FLASHES_PER_SELECTION:
            raise ValueError("physical selection does not contain the complete original flash schedule")
        first, last = int(raw.flash_sample[indices[0]]), int(raw.flash_sample[indices[-1]])
        start, stop = first - preroll, last + width
        if start < 0 or stop > min(authorized_stop, len(raw.eeg_uv)):
            raise ValueError("selection response extends outside the authorized actual raw observations")
        if indices[0] > 0 and raw.flash_sample[indices[0] - 1] + width > start:
            raise ValueError("previous unknown flash response overlaps the independent selection observations")
        if indices[-1] + 1 < len(raw.flash_sample) and raw.flash_sample[indices[-1] + 1] < stop:
            raise ValueError("next unknown flash overlaps the independent selection observations")
        if receipts and receipts[-1]["raw_scope"][1] > start:
            raise ValueError("independent calibration selections reuse actual raw sample positions")
        event_ids = tuple(f"{BNCI2014_008_DATASET_ID}:{raw.subject_id}:{int(index)}" for index in indices)
        decision = MembershipDecision.row_column(
            raw.candidate_id[indices], n_rows=BNCI2014_008_GRID_SHAPE[0],
            n_columns=BNCI2014_008_GRID_SHAPE[1], event_ids=event_ids,
            repetition_indices=raw.repetition_index[indices], repetitions=BNCI2014_008_REPETITIONS_PER_SELECTION,
        )
        decision_id = str(raw.selection_id[indices[0]])
        view = PredictionDecisionView(
            record_id=f"{BNCI2014_008_DATASET_ID}:{decision_id}:{source_sha256}",
            participant_key=origin_subject_key(BNCI2014_008_DATASET_ID, raw.subject_id),
            decision_id=decision_id, samples=np.asarray(raw.eeg_uv[start:stop], dtype=np.float64) * 1e-6,
            onsets=raw.flash_sample[indices] - start, decision=decision, source_sha256=source_sha256,
            raw_scope=(start, stop), sample_rate_hz=rate, channel_names=BNCI2014_008_CHANNELS,
            channel_types=("EEG",) * len(BNCI2014_008_CHANNELS), reference=BNCI2014_008_SOURCE_REFERENCE,
            source_relative_path=raw.source_relative_path, source_dataset_id=BNCI2014_008_DATASET_ID,
            source_subject_id=raw.subject_id, available_sample=stop,
        )
        truth = row_column_target(int(raw.target_row[indices[0]]), int(raw.target_col[indices[0]]))
        groups.append(CalibrationSelection(view, raw.raw_is_target[indices], truth))
        receipts.append({"decision_id": decision_id, "raw_scope": [start, stop], "label_indices": indices.tolist(),
                         "physical_support_digest": view.physical_support_digest,
                         "raw_input_digest": view.raw_input_digest, "available_sample": stop})
        used_indices.extend(indices.tolist())
        if len(groups) == group_count:
            break
    if len(groups) != group_count:
        raise ValueError("authorized prefix cannot provide the requested complete independent groups")
    unused = sorted(set(range(authorized_labels)) - set(used_indices))
    return CalibrationGroupViews(tuple(groups), {
        "schema": "n2p3_bnci_calibration_group_scope/1", "source_sha256": source_sha256,
        "source_relative_path": raw.source_relative_path,
        "participant_key": groups[0].view.participant_key, "authorized_label_count": int(authorized_labels),
        "authorized_stop_sample": authorized_stop, "requested_group_count": int(group_count),
        "actual_group_count": len(groups), "used_label_count": len(used_indices),
        "unused_authorized_labels": len(unused), "unused_authorized_label_indices": unused,
        "response_seconds": float(response_seconds), "pre_roll_seconds": float(pre_roll_seconds),
        "response_samples": width, "preroll_samples": preroll,
        "groups": receipts, "raw_route": "native_unfiltered_volts",
        "filtering_scope": "each_independent_selection", "different_groups_use_disjoint_sample_positions": True,
        "group_scoping_does_not_claim_statistical_independence": True,
        "evidence_scope": "public_calibration_only_development",
    })


def build_calibration_epoch_dataset(groups, profile: SelectionEpochProfile, *, task_contract=None,
                                    identity_table=None, name="authorized_selection_epochs"):
    """Lower authorized complete groups into the existing physical data contract.

    This preserves source event identities and recording availability. Its
    provenance names the actual selection operator; it does not manufacture a
    legacy ``mne_raw`` execution certificate for a different ingress route.
    """
    groups = tuple(groups)
    if not groups or any(not isinstance(group, CalibrationSelection) for group in groups):
        raise ValueError("epoch calibration datasets require authorized CalibrationSelection groups")
    views = tuple(group.view for group in groups)
    if len({view.participant_key for view in views}) != 1:
        raise ValueError("one target calibration dataset requires one explicit physical participant")
    if any(not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0) for view in views):
        raise ValueError("calibration epoch datasets require complete observed physical selections")
    source_ids = [view.source_dataset_id for view in views]
    subject_ids = [view.source_subject_id for view in views]
    table = (DatasetIdentityTable.from_source_rows(subject_ids, source_ids)
             if identity_table is None else identity_table)
    for view in views:
        if table.record_for(view.source_subject_id).authority_key() != view.participant_key:
            raise ValueError("calibration participant key disagrees with the shared identity contract")
    protocols = {view.decision.protocol for view in views}
    if protocols == {"row_column"}:
        if task_contract is None and set(source_ids) == {BNCI2014_008_DATASET_ID}:
            task_contract = BNCI2014_008_CANDIDATE_TASK_CONTRACT
        if not isinstance(task_contract, CandidateTaskContract) or set(source_ids) != {task_contract.dataset_id}:
            raise ValueError("row/column epoch metadata requires its actual dataset task contract")
    elif len(protocols) != 1 or task_contract is not None:
        raise ValueError("calibration group protocols and task contract must agree")
    arrays, masks, receipts, rows, timeline_rows, parents = [], [], [], [], [], []
    for group in groups:
        view, decision = group.view, group.view.decision
        X, mask, receipt = profile.transform(view)
        arrays.append(X)
        masks.append(mask)
        receipts.append(receipt)
        absolute_onsets = view.onsets + view.raw_scope[0]
        stimulus_ids = tuple(decision.stimulus_vocabulary[index] for index in decision.stimulus_indices)
        metadata = {"subject": np.repeat(view.source_subject_id, len(view.onsets)),
                    "selection_id": np.repeat(view.decision_id, len(view.onsets)),
                    "event_id": decision.event_ids, "repetition_index": decision.repetition_indices,
                    "selection_repetitions": np.full(len(view.onsets), decision.repetitions),
                    "candidate_id": stimulus_ids, "target_candidate": np.repeat(group.target_candidate, len(view.onsets)),
                    "is_target": group.labels, "acquisition_time_s": absolute_onsets / view.sample_rate_hz}
        if decision.protocol == "row_column":
            truth = parse_row_column_target(group.target_candidate, task_contract)
            target_row, target_col = (int(part.partition(":")[2]) for part in truth.split("|"))
            codes = np.asarray(stimulus_ids, dtype=np.int64)
            metadata.update(candidate_id=codes,
                            row_code=np.where(codes < task_contract.n_rows, codes, -1),
                            col_code=np.where(codes >= task_contract.n_rows, codes - task_contract.n_rows, -1),
                            target_row=np.full(len(codes), target_row), target_col=np.full(len(codes), target_col))
        rows.append(pd.DataFrame(metadata))
        timeline_rows.extend({"event_id": event, "group_id": view.decision_id, "subject_id": view.source_subject_id,
                              "stimulus_id": int(decision.stimulus_indices[index]),
                              "onset_sample": int(absolute_onsets[index]),
                              "onset_time_s": float(absolute_onsets[index] / view.sample_rate_hz),
                              "evidence_available_time_s": float(view.available_sample / view.sample_rate_hz),
                              "dataset_id": view.source_dataset_id, "session_id": "0",
                              "run_id": view.source_relative_path, "selection_id": view.decision_id,
                              "candidate_id": stimulus_ids[index],
                              "target_candidate_id": group.target_candidate if decision.protocol == "direct" else "",
                              "repetition_index": int(decision.repetition_indices[index])}
                             for index, event in enumerate(decision.event_ids))
        parents.append(DataLineage.source(parameters={**view.record(),
            "source_dataset_ids": [view.source_dataset_id]}))
    count = len(timeline_rows)
    timeline = ScheduledEventTimeline(
        event_ids=np.array([row["event_id"] for row in timeline_rows]),
        group_ids=np.array([row["group_id"] for row in timeline_rows]),
        subject_ids=np.array([row["subject_id"] for row in timeline_rows]),
        stimulus_ids=np.array([row["stimulus_id"] for row in timeline_rows], dtype=np.int64),
        onset_samples=np.array([row["onset_sample"] for row in timeline_rows], dtype=np.int64),
        onset_times_s=np.array([row["onset_time_s"] for row in timeline_rows]),
        evidence_available_times_s=np.array([row["evidence_available_time_s"] for row in timeline_rows]),
        evidence_indices=np.arange(count), statuses=np.repeat("available", count), status_details=np.repeat("", count),
        dataset_ids=np.array([row["dataset_id"] for row in timeline_rows]),
        session_ids=np.array([row["session_id"] for row in timeline_rows]),
        run_ids=np.array([row["run_id"] for row in timeline_rows]),
        selection_ids=np.array([row["selection_id"] for row in timeline_rows]),
        candidate_ids=np.array([row["candidate_id"] for row in timeline_rows]),
        target_candidate_ids=np.array([row["target_candidate_id"] for row in timeline_rows]),
        repetition_indices=np.array([row["repetition_index"] for row in timeline_rows], dtype=np.int64),
        complete=True, online_causal=False, timing_source="attested_raw_native_sample_grid;independent_selection_end",
    )
    X, mask = np.concatenate(arrays), np.concatenate(masks)
    labels = np.concatenate([group.labels for group in groups]).astype(np.int64)
    metadata = pd.concat(rows, ignore_index=True)
    if task_contract is not None:
        validate_candidate_membership_metadata(metadata, task_contract, labels=labels)
    geometry = build_channel_identity(profile.channel_names, allow_missing_positions=False)
    provenance = {"source": "attested_original_continuous_selection_views",
                  "source_reference": (profile.reference if profile.reference_channel is None
                                       else "measured_channel:" + profile.reference_channel),
                  "parent_source_reference": profile.reference, "signal_unit": DEFAULT_P300_DATA_CONTRACT.signal_unit,
                  "source_sample_rate_hz": profile.source_sample_rate_hz,
                  "source_records": [view.record() for view in views], "selection_epoch_profile": profile.record(),
                  "selection_epoch_execution": receipts, "authorized_label_count": len(labels)}
    provenance["selection_epoch_capability"] = profile.execution_capability()
    if len({view.source_sha256 for view in views}) == 1:
        provenance["source_file_sha256"] = views[0].source_sha256
    else:
        provenance["source_files_sha256"] = sorted({view.source_sha256 for view in views})
    if task_contract is not None:
        provenance["candidate_task_contract"] = task_contract.record()
    dataset = EpochDataset(
        name=name, X=X, y=labels, subject_ids=timeline.subject_ids, channel_names=geometry.names,
        channel_positions_m=geometry.coords, channel_mask=np.ones(len(profile.channel_names), dtype=bool),
        trial_channel_mask=mask, preprocessing=profile.preprocessing, event_timeline=timeline,
        metadata=metadata, provenance=provenance, identity_table=table,
        lineage=DataLineage.derive(parents, operation="selection_epoch_transform",
            parameters={"profile": profile.record(), "receipts": receipts,
                        "actual_epoch_values_sha256": array_digest(X, mask), "authorized_labels_sha256": array_digest(labels)}),
    )
    dataset.validate(require_labels=True)
    return dataset


class RecordCalibrationPrefixCompiler(CalibrationPrefixCompiler):
    """The original multi-file source owner; no stitched continuous recording."""

    def __init__(self, adapter, *, budgets, profiles, authorization_preprocessing,
                 response_seconds=1.203125, pre_roll_seconds=.2, _source_training=False):
        from data.candidate_input import CandidateDatasetAdapter

        if not isinstance(adapter, CandidateDatasetAdapter):
            raise TypeError("shared prefix issuer requires its registered original dataset adapter")
        if _source_training and type(self) is not RecordSourceTrainingCompiler:
            raise TypeError("full source permission uses its named original-record issuer")
        self._source_training = _source_training
        if _source_training and adapter.requested_future_record_ids is not None:
            raise ValueError("a source-only issuer does not attach future evaluation acquisitions")
        self.adapter, self.sources = adapter, adapter.recordings
        self.profiles = {p.digest: p for p in profiles}
        self.budgets = tuple(sorted(set(budgets)))
        total = sum(len(s.native_onsets) for s in self.sources)
        if not self.budgets or any(type(k) is not int or k < 0 for k in self.budgets):
            raise ValueError("global requested K must declare nonnegative ordered label doses")
        first = self.sources[0]
        self.rate, self.reference = first.sample_rate_hz, first.reference
        self.response_samples = max([round(response_seconds*self.rate), *(p._source_stop for p in self.profiles.values())])
        self.preroll_samples = max([round(pre_roll_seconds*self.rate), *(-p._source_start for p in self.profiles.values())])
        self.authorization_preprocessing = authorization_preprocessing
        self._prefixes, self._prefix_ownership = {}, {}
        self._offsets = np.r_[0, np.cumsum([len(s.native_onsets) for s in self.sources])]
        from transfer.binary import BinaryTimeSplit

        def split(k, maximum):
            if not 0 <= k <= maximum < total:
                raise ValueError("requested global C labels would consume the original later event pool")
            test, purged = [], []
            for source, base in zip(self.sources, self._offsets[:-1], strict=True):
                if (adapter.requested_future_record_ids is not None
                        and source.record_id not in adapter.requested_future_record_ids):
                    continue
                count = min(len(source.native_onsets), max(0, maximum-int(base)))
                last = source.native_onsets[:count]
                last = last[last >= 0]
                boundary = int(last[-1])+round(authorization_preprocessing.tmax_ms/1000*self.rate)+1 if len(last) else 0
                for local in range(count, len(source.native_onsets)):
                    value = int(source.native_onsets[local])
                    start = value+round(authorization_preprocessing.tmin_ms/1000*self.rate)
                    stop = value+round(authorization_preprocessing.tmax_ms/1000*self.rate)+1
                    admissible = (source.native_sample_count is not None and value >= 0 and start >= max(0, boundary)
                                  and stop <= source.native_sample_count)
                    (test if admissible else purged).append(int(base)+local)
            return BinaryTimeSplit(np.arange(k, dtype=np.int64), np.asarray(test, dtype=np.int64),
                                   np.asarray(purged, dtype=np.int64))
        if _source_training:
            if self.budgets != (total,):
                raise ValueError("the full source issuer retains its exact complete declared label roster")
            self._splits = {total: BinaryTimeSplit(np.arange(total, dtype=np.int64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))}
            self._adjustments = {}
        else:
            self._splits, self._adjustments = resolve_calibration_splits(self.budgets, split,
                minimum_test_trials=1, available_trials=total,
                separate_test_recordings=adapter.requested_future_record_ids is not None)
        self.actual_budgets, self.maximum = tuple(sorted(self._splits)), max(self._splits)
        if adapter.requested_future_record_ids is not None:
            for source, base in zip(self.sources, self._offsets[:-1], strict=True):
                if source.record_id in adapter.requested_future_record_ids and int(base) < self.maximum:
                    raise ValueError("declared evaluation acquisition record overlaps global maximum C-label permission")
        self._read_cache = {}
        self.target = {"participant_key": adapter.participant_key, "source_dataset_id": adapter.dataset_id,
            "source_subject_id": adapter.source_subject_id, "source_sha256": adapter.digest,
            "source_relative_path": "ordered_original_candidate_acquisitions",
            "sample_rate_hz": self.rate, "channel_names": list(first.channel_names),
            "channel_types": ["EEG"]*len(first.channel_names), "reference": self.reference,
            "signal_unit": "V", "candidate_vocabulary": list(first.selections[0].decision.candidate_vocabulary),
            "candidate_task_contract": adapter.candidate_task_contract, "endpoint": adapter.endpoint,
            "requested_future_record_ids": None if adapter.requested_future_record_ids is None else list(adapter.requested_future_record_ids),
            "authority_kind": "record_aware_calibration_prefix/1", "raw_scope": None,
            "source_record_roster": {s.record_id: {"record_id": s.record_id, "source_sha256": s.source_sha256,
                "source_relative_path": s.source_relative_path, "native_sample_count": s.native_sample_count,
                "sample_rate_hz": s.sample_rate_hz, "source_subject_id": s.source_subject_id,
                "session_id": s.session_id, "run_id": s.run_id} for s in self.sources}}
        if _source_training:
            self.target.update({"issuer_role": "full_original_source_training_prefix/1",
                "source_training_only": True, "source_requested_labels": total,
                "target_groups_materialized": False,
                "source_selection_ledger": [{"record_id": source.record_id,
                    "selections": [{**selection.public_record(),
                        "stimulus_schedule_complete": selection.decision.schedule_complete,
                        "response_support_right_censored": source.native_sample_count is not None and
                            int(source.native_onsets[selection.event_indices[-1]])+self.response_samples > source.native_sample_count}
                        for selection in source.selections],
                    "requested_selections": len(source.selections),
                    "complete_stimulus_schedules": sum(s.decision.schedule_complete for s in source.selections)}
                    for source in self.sources]})
        for source in self.sources:
            if (source.channel_names,source.reference,source.sample_rate_hz) != (first.channel_names,self.reference,self.rate):
                raise ValueError("original participant records have incompatible measured native layouts/reference/clock")
            geometry = self._current_view(source, 0, read_samples=False)
            for profile in self.profiles.values():
                profile.validate_observation(geometry)
        self._history_ingress = {"dataset_id": adapter.dataset_id, "source_format": adapter.source_format,
            "record_sources": list(adapter.source_artifacts), "budgets": list(self.budgets),
            "profiles": [p.record() for p in self.profiles.values()],
            "authorization_preprocessing": authorization_preprocessing.record(),
            "response_seconds": response_seconds, "pre_roll_seconds": pre_roll_seconds,
            "original_adapter_digest": adapter.digest, "original_factory_source_sha256": adapter.original_factory_source_sha256}
        if _source_training:
            self._history_ingress["issuer_role"] = "full_original_source_training_prefix/1"
        self._arguments = {"maximum": self.maximum, "profiles": self.profiles,
            "authorization_preprocessing": authorization_preprocessing, "width": self.response_samples,
            "preroll": self.preroll_samples, "target": self.target}

    def _read(self, source, cutoff):
        key = source.record_id, cutoff
        if key not in self._read_cache:
            masters = [(n, value) for (record, n), value in self._read_cache.items()
                       if record == source.record_id and n >= cutoff]
            self._read_cache[key] = (min(masters, key=lambda item: item[0])[1][:cutoff] if masters else
                                    _immutable_array(source.read_native(0, cutoff), np.float64))
        return self._read_cache[key]

    def _current_view(self, source, selection, *, permitted_event_stop=None, maximum_read_stop=None,
                      minimum_read_start=0, read_samples=True, forced_unavailable=None):
        declared = source.selections[selection]
        rows = np.asarray(declared.event_indices, dtype=np.int64)
        decision = declared.decision
        onsets = source.native_onsets[rows]
        reason = forced_unavailable
        if source.native_sample_count is None or np.any(onsets < 0):
            reason = "original_EEG_or_optical_clock_unavailable"
            if source.unavailable_public_onsets is None:
                raise ValueError("unavailable native clocks need their declared physical-unavailable view adapter")
            onsets = source.unavailable_public_onsets[rows]
        start = max(0, int(onsets[0])-self.preroll_samples)
        stop = int(onsets[-1])+self.response_samples
        if source.native_sample_count is not None and stop > source.native_sample_count:
            reason = "original_response_support_right_censored"
        if rows[0] > 0 and source.native_onsets[rows[0]-1] >= 0 and source.native_onsets[rows[0]-1]+self.response_samples > start:
            reason = "previous_unknown_decision_overlaps_finite_response_support"
        if rows[-1]+1 < len(source.native_onsets) and 0 <= source.native_onsets[rows[-1]+1] < stop:
            reason = "next_unknown_decision_overlaps_finite_response_support"
        if permitted_event_stop is not None and rows[-1] >= permitted_event_stop:
            reason = "partial_selection_at_global_C_label_boundary"
        if maximum_read_stop is not None and stop > maximum_read_stop:
            reason = "response_outside_original_C_read_cutoff"
        if start < minimum_read_start:
            reason = "response_overlaps_common_C_test_embargo"
        if not read_samples:
            reason = "physical_geometry_only_no_signal_read"
        if reason:
            decision = replace(decision, evidence_indices=np.full(len(rows), -1, dtype=np.int64))
        values = None if reason else source.read_native(start, stop)
        return PredictionDecisionView(f"{source.record_id}:selection:{selection}", source.participant_key,
            declared.selection_id, values, onsets-start, decision, source.source_sha256, (start, stop),
            self.rate, source.channel_names, ("EEG",)*len(source.channel_names), self.reference,
            source.source_relative_path, self.adapter.dataset_id, source.source_subject_id, stop, unavailable_reason=reason)

    def compile(self, budget):
        if type(budget) is not int or not 0 <= budget <= self.maximum:
            raise ValueError("derived record-aware C prefix exceeds original global label authority")
        if budget in self._prefixes:
            self._prefix_ownership[budget].assert_unchanged(self._prefixes[budget])
            return self._prefixes[budget]
        records, scopes, groups, unavailable, trial_parts = [], {}, [], [], {p: [] for p in self.profiles}
        for source, base in zip(self.sources, self._offsets[:-1], strict=True):
            count = min(len(source.native_onsets), max(0, budget-int(base)))
            if not count:
                continue
            onsets = source.native_onsets[:count]
            if source.native_sample_count is None or np.any(onsets < 0):
                unavailable.append({"record_id": source.record_id, "global_label_indices": [int(base), int(base)+count],
                    "reason": "original_EEG_or_C_optical_clock_unavailable"})
                continue
            next_onset = int(source.native_onsets[count]) if count < len(source.native_onsets) else source.native_sample_count
            if next_onset < 0:
                unavailable.append({"record_id": source.record_id, "global_label_indices": [int(base), int(base)+count],
                    "reason": "original_next_unknown_optical_cutoff_unavailable"})
                continue
            cutoff = min(next_onset, int(onsets[-1])+self.response_samples, source.native_sample_count)
            samples = self._read(source, cutoff)
            labels = source.authorized_binary_labels[:count]
            raw = CalibrationRecord(source.record_id, samples, onsets, labels, right_censored=True)
            records.append(raw)
            scope = CalibrationRecordScope(source.record_id, source.source_sha256,
                source.source_relative_path, self.adapter.dataset_id, source.source_subject_id, (0, cutoff), (int(base), int(base)+count),
                cutoff, tuple(map(int, onsets)))
            scopes[source.record_id] = scope.record()
            # The source-only role fits complete native records. Target-only
            # decision calibration groups would reread each same source range.
            if not self._source_training:
                for selection, declared in enumerate(source.selections):
                    event_rows = np.asarray(declared.event_indices, dtype=np.int64)
                    if event_rows[0] >= count:
                        break
                    group = self._current_view(source, selection, permitted_event_stop=count, maximum_read_stop=cutoff)
                    if group.samples is not None and group.raw_scope[1] <= cutoff and declared.parent_target_candidate is not None:
                        groups.append(CalibrationSelection(group, labels[event_rows], declared.parent_target_candidate))
            for digest, profile in self.profiles.items():
                profile.validate_observation(self._current_view(source, 0, read_samples=False))
                available = (onsets+profile._source_start >= 0) & (onsets+profile._source_stop <= cutoff)
                indices = np.flatnonzero(available)
                if not len(indices):
                    continue
                m = np.vstack([s.decision.membership for s in source.selections])[indices]
                all_events = tuple(e for s in source.selections for e in s.decision.event_ids)
                event_ids = tuple(all_events[i] for i in indices)
                physical = MembershipDecision(tuple(self.target["candidate_vocabulary"]), m, event_ids, event_ids,
                    np.arange(len(indices)), np.zeros(len(indices), dtype=np.int64), np.arange(len(indices)),
                    1, source.selections[0].decision.protocol, False)
                view = PredictionDecisionView(f"{source.record_id}:C{count}", source.participant_key,
                    f"{source.record_id}:authorized_C", samples, onsets[indices], physical,
                    source.source_sha256, (0, cutoff), self.rate,
                    tuple(self.target["channel_names"]), tuple(self.target["channel_types"]), self.reference,
                    source.source_relative_path, self.adapter.dataset_id, source.source_subject_id, cutoff)
                X, mask, receipt = profile.transform(view)
                trial_parts[digest].append((X, mask, receipt, view, indices+int(base), labels[indices], source))
        datasets, trial_use = {}, {}
        for digest, profile in self.profiles.items():
            parts = trial_parts[digest]
            used = np.concatenate([p[4] for p in parts]).tolist() if parts else []
            use = {"requested_labels": budget, "used_labels": len(used), "used_label_indices": used,
                "unused_label_indices": sorted(set(range(budget))-set(used)), "profile": profile.record(),
                "record_scopes": scopes}
            if parts:
                data = _record_epoch_prefix(parts, profile, self.target, budget)
                datasets[digest] = data
                use["numerical_input_digest"] = array_digest(data.X, data.y, data.trial_channel_mask)
            trial_use[digest] = use
        used_raw = [i for s in scopes.values() for i in range(*s["global_label_indices"])]
        prefix = CalibrationPrefix(budget, None, datasets, tuple(groups), {**self.target, "record_scopes": scopes},
            {"raw": {"requested_labels": budget, "used_labels": len(used_raw), "used_label_indices": used_raw,
                "unused_label_indices": sorted(set(range(budget))-set(used_raw)), "record_scopes": scopes,
                "unavailable_records": unavailable}, "trial": trial_use}, native_records=tuple(records))
        self._prefixes[budget] = prefix
        self._prefix_ownership[budget] = _OwnedCalibrationPrefix(prefix, calibration_prefix_identity(prefix))
        return prefix

    def before_group(self, parent, held):
        parent = self.assert_original_prefix(parent)
        self._prefix_ownership[parent.budget].assert_group_match(held)
        matches = [(s, int(base)) for s, base in zip(self.sources, self._offsets[:-1], strict=True)
                   if s.source_sha256 == held.view.source_sha256]
        if len(matches) != 1:
            raise ValueError("held C group has no unique actual acquisition identity")
        source, base = matches[0]
        absolute = held.view.onsets+held.view.raw_scope[0]
        positions = np.searchsorted(source.native_onsets, absolute)
        if not np.array_equal(source.native_onsets[positions], absolute):
            raise ValueError("held group clocks differ from the original record event ledger")
        # Its preroll must also stay out of earlier C observations. Original
        # next-event geometry is applied before any filtering/statistics.
        earlier_count = int(np.searchsorted(source.native_onsets, held.view.raw_scope[0], side="left"))
        earlier = self.compile(base+earlier_count)
        scope = earlier.target_contract["record_scopes"].get(source.record_id)
        if scope is not None and scope["read_sample_support"][1] > held.view.raw_scope[0]:
            raise ValueError("earlier record-aware prefix overlaps actual held raw observations")
        return earlier

    def _issue_record_read_scope(self, view, *, prefix, conditioning_stop, available_stop=None,
                                 program="record_prequential_public_history/1"):
        """New named P/T history permission; the independent holdout stays strict.

        C may overlap the observed conditioning prefix. C must end before P's
        cutoff, scored T follows P, and no held binary/target label leaves this
        issuer. Original acquisition boundaries remain explicit.
        """
        from research.history_authority import HistoryUnavailableError

        prefix = self.assert_original_prefix(prefix)
        stop = view.available_sample if available_stop is None else available_stop
        matches = [r for r in self.sources if r.source_sha256 == view.source_sha256
                   and r.source_relative_path == view.source_relative_path]
        if len(matches) != 1 or program != "record_prequential_public_history/1":
            raise ValueError("history scope requires one actual record and the named prequential programme")
        source = matches[0]
        selections = [s for s in source.selections if s.selection_id == view.decision_id]
        if (len(selections) != 1 or view.participant_key != source.participant_key
                or type(conditioning_stop) is not int or type(stop) is not int
                or not 0 < conditioning_stop < stop == view.available_sample
                or source.native_sample_count is None or stop > source.native_sample_count
                or view.sample_rate_hz != self.rate or view.channel_names != source.channel_names
                or view.reference != source.reference):
            raise ValueError("actual record P/T native clock, reference, request or cutoff changed")
        declared = selections[0]
        native = source.native_onsets[np.asarray(declared.event_indices)]
        actual_decision, original_decision = view.decision.record(), declared.decision.record()
        actual_decision.pop("evidence_indices")
        original_decision.pop("evidence_indices")
        if (np.any(native < 0) or not np.array_equal(native, view.onsets+view.raw_scope[0])
                or original_decision != actual_decision
                or view.unavailable_reason not in {None,
                    "previous_unknown_decision_overlaps_finite_response_support",
                    "next_unknown_decision_overlaps_finite_response_support",
                    "response_overlaps_common_C_test_embargo"}):
            raise ValueError("current history view differs from original event-set/native-clock authority")
        c_scope = prefix.target_contract["record_scopes"].get(source.record_id)
        if c_scope is not None and c_scope["read_sample_support"][1] > conditioning_stop:
            raise ValueError("C observations exceed the new past-conditioning cutoff")
        public = []
        for selection in source.selections:
            rows = np.asarray(selection.event_indices, dtype=np.int64)
            known = source.native_onsets[rows] >= 0
            possible_past = (source.unavailable_public_onsets[rows] < stop if source.unavailable_public_onsets is not None
                             else np.ones(len(rows), dtype=bool))
            if np.any(~known & possible_past):
                raise HistoryUnavailableError("record_history_unresolved_actual_event_clock")
            occurred = known & (source.native_onsets[rows] < stop)
            if not occurred.any():
                continue
            public.append({"selection_id": selection.selection_id,
                "candidate_vocabulary": list(selection.decision.candidate_vocabulary),
                "event_ids": list(np.asarray(selection.decision.event_ids)[occurred]),
                "onsets": source.native_onsets[rows[occurred]].tolist(),
                "membership": selection.decision.membership[occurred].tolist(),
                "complete_requested_current": selection.selection_id == view.decision_id})
        samples = source.read_native(0, stop)
        if view.samples is not None and not np.array_equal(samples[slice(*view.raw_scope)], view.samples):
            raise ValueError("current history samples differ from actual original record read")
        binding = {"schema": "n2p3_authorized_record_public_history/1", "program": program,
            "record_id": source.record_id, "source_sha256": source.source_sha256,
            "native_clock": "actual_record_native_sample", "participant_key": source.participant_key,
            "prefix_identity": calibration_prefix_identity(prefix), "C_record_scopes": prefix.target_contract["record_scopes"],
            "C_record_to_original_record": {raw.record_id: raw.record_id for raw in prefix.calibration_records},
            "C_record_input_sha256": {raw.record_id: array_digest(raw.samples, raw.onsets, raw.target_mask)
                for raw in prefix.calibration_records},
            "read_sample_support": [0, stop], "conditioning_sample_support": [0, conditioning_stop],
            "score_sample_support": [conditioning_stop, stop], "available_sample": stop,
            "current_decision_id": view.decision_id, "public_selections": public,
            "channel_names": list(source.channel_names), "sample_rate_hz": self.rate,
            "reference": self.reference, "signal_unit": "V", "samples_sha256": array_digest(samples),
            "current_view": view.record(), "original_decoder_receipt": source.source_receipt,
            "original_factory_source_sha256": self.adapter.original_factory_source_sha256,
            "source_owner_sha256": _SELECTION_EXECUTION_SOURCE_SHA256, "future_truth_consumed": False}
        return _register_record_history(self, samples, prefix, binding)

    def calibration_record_physical_map(self, prefix):
        prefix = self.assert_original_prefix(prefix)
        scopes = prefix.target_contract["record_scopes"]
        return freeze_metadata({"schema": "n2p3_calibration_physical_record_map/1",
            "prefix_identity": calibration_prefix_identity(prefix), "source_owner_sha256": _SELECTION_EXECUTION_SOURCE_SHA256,
            "records": {raw.record_id: {"physical_record_id": raw.record_id,
                "source_sha256": scopes[raw.record_id]["source_sha256"],
                "source_relative_path": scopes[raw.record_id]["source_relative_path"],
                "input_sha256": array_digest(raw.samples, raw.onsets, raw.target_mask)} for raw in prefix.calibration_records}})

    def independent_views(self):
        decisions, truths, test_scopes = [], {}, {}
        for source, base in zip(self.sources, self._offsets[:-1], strict=True):
            if (self.adapter.requested_future_record_ids is not None
                    and source.record_id not in self.adapter.requested_future_record_ids):
                continue
            count = min(len(source.native_onsets), max(0, self.maximum-int(base)))
            c_scope = self.compile(self.maximum).target_contract["record_scopes"].get(source.record_id)
            earlier = source.native_onsets[:count]
            earlier = earlier[earlier >= 0]
            epoch_boundary = (int(earlier[-1])+round(self.authorization_preprocessing.tmax_ms/1000*self.rate)+1
                              if len(earlier) else 0)
            earliest = max(epoch_boundary, 0 if c_scope is None else c_scope["read_sample_support"][1])
            for selection, declared in enumerate(source.selections):
                if declared.event_indices[-1] < count:
                    continue
                view = self._current_view(source, selection, minimum_read_start=earliest,
                    forced_unavailable="partial_selection_after_common_Kmax" if declared.event_indices[0] < count else None)
                decisions.append(view)
                truths[view.decision_id] = declared.parent_target_candidate
            valid = source.native_onsets[count:]
            valid = valid[valid >= 0]
            test_scopes[source.record_id] = {"source_sha256": source.source_sha256,
                "start_sample": earliest,
                "stop_sample": source.native_sample_count or 0}
        ledger = {"schema": "n2p3_independent_record_prefixes/1", "dataset_id": self.adapter.dataset_id,
            "source_sha256": self.target["source_sha256"], "requested_Ks": list(self.budgets),
            "requested_to_actual_K": {str(k): min(k, self.maximum) for k in self.budgets}, "maximum_calibration_events": self.maximum,
            "budget_adjustments": {str(k): value for k, value in self._adjustments.items()},
            "test_scope": {"record_scopes": test_scopes}, "test_budget": self._splits[self.maximum].test_budget.record(),
            "calibration": {str(k): self.compile(k).record() for k in self.actual_budgets},
            "requested_decisions": len(decisions), "scorable_decisions": sum(v.samples is not None for v in decisions),
            "test_views": [v.record() for v in decisions], "operator_identity": "native_unfiltered_volts/1",
            "endpoint": self.adapter.endpoint,
            "input_records": [s.public_record() for s in self.sources], "truth_consumed_by_prediction": False}
        return IndependentSubjectViews({k: self.compile(k) for k in self.actual_budgets}, tuple(decisions), truths, ledger, self)


class RecordSourceTrainingCompiler(RecordCalibrationPrefixCompiler):
    """Full source labels from the registered original adapter, with no future.

    The ordinary target compiler keeps its independent later-test requirement.
    This named issuer reuses the same native reader, prefix ownership, clocks,
    labels, geometry, right-censoring and optional epoch-profile implementation.
    """

    def __init__(self, adapter, *, profiles, authorization_preprocessing,
                 response_seconds, pre_roll_seconds):
        total = sum(len(source.native_onsets) for source in adapter.recordings)
        super().__init__(adapter, budgets=(total,), profiles=profiles,
            authorization_preprocessing=authorization_preprocessing, response_seconds=response_seconds,
            pre_roll_seconds=pre_roll_seconds, _source_training=True)

    def source_fit_operands(self):
        """Issue every exact source record and its original per-record controller."""
        prefix = self.assert_original_prefix(self.compile(self.maximum))
        use = prefix.label_use["raw"]
        if use["used_labels"] != self.maximum or use["unavailable_records"]:
            raise ValueError("the complete requested source programme has unavailable native records or clocks")
        records = prefix.calibration_records
        if {record.record_id for record in records} != {source.record_id for source in self.sources}:
            raise ValueError("the complete source issuer lost an original declared acquisition")
        # assert_original_prefix just verified this compiler-owned immutable
        # prefix. Reuse its admitted identity within this read-only call.
        prefix_identity = self._prefix_ownership[prefix.budget].identity
        receipts = []
        for record in records:
            scope = prefix.target_contract["record_scopes"][record.record_id]
            count = len(record.onsets)
            receipts.append({**prefix.target_contract, "record_id": record.record_id,
                "source_sha256": scope["source_sha256"], "raw_scope": scope["read_sample_support"],
                "raw_input_digest": array_digest(record.samples, record.onsets, record.target_mask),
                "source_prefix_identity": prefix_identity,
                "original_compiler_source_sha256": _SELECTION_EXECUTION_SOURCE_SHA256,
                "label_use": {"raw": {"requested_labels": count, "used_labels": count,
                    "read_sample_support": scope["read_sample_support"], "right_censored": record.right_censored}}})
        return prefix, records, tuple(receipts)

    def independent_views(self):
        raise TypeError("the full source-only issuer has no held-out target evaluation programme")


def _record_epoch_prefix(parts, profile, target, budget):
    arrays, masks, labels = [np.concatenate([p[i] for p in parts]) for i in (0, 1, 5)]
    metadata, timeline = [], []
    for _X, _, _, view, global_indices, _, source in parts:
        for j, (event, index) in enumerate(zip(view.decision.event_ids, global_indices, strict=True)):
            local = int(np.searchsorted(source.native_onsets, view.onsets[j]))
            declared = next(s for s in source.selections if local in s.event_indices)
            metadata.append({"subject": source.source_subject_id, "source_dataset_id": target["source_dataset_id"],
                "source_subject_id": source.source_subject_id, "source_record_id": source.record_id,
                "global_label_index": int(index), "source_native_sample": int(view.onsets[j]),
                "selection_id": declared.selection_id, "repetition_index": 0, "candidate_id": local})
            timeline.append({"event": event, "sample": int(view.onsets[j]), "source": source,
                "selection": declared.selection_id, "available": view.available_sample})
    count = len(timeline)
    t = ScheduledEventTimeline(event_ids=np.asarray([r["event"] for r in timeline]),
        group_ids=np.asarray([r["selection"] for r in timeline]), subject_ids=np.repeat(target["source_subject_id"], count),
        stimulus_ids=np.arange(count), onset_samples=np.asarray([r["sample"] for r in timeline]),
        onset_times_s=np.asarray([r["sample"] for r in timeline] )/target["sample_rate_hz"],
        evidence_available_times_s=np.asarray([r["available"] for r in timeline] )/target["sample_rate_hz"],
        evidence_indices=np.arange(count), statuses=np.repeat("available", count), status_details=np.repeat("", count),
        dataset_ids=np.repeat(target["source_dataset_id"], count), session_ids=np.asarray([r["source"].session_id for r in timeline]),
        run_ids=np.asarray([r["source"].run_id for r in timeline]), selection_ids=np.asarray([r["selection"] for r in timeline]),
        candidate_ids=np.asarray([r["event"] for r in timeline]), target_candidate_ids=np.repeat("", count),
        repetition_indices=np.zeros(count, dtype=np.int64), complete=True, online_causal=False,
        timing_source="actual_original_candidate_record_native_sample")
    geometry = build_channel_identity(profile.channel_names, allow_missing_positions=False)
    data = EpochDataset(name=f"{target['source_dataset_id']}-C{budget}", X=arrays, y=labels.astype(np.int64),
        subject_ids=np.repeat(target["source_subject_id"], count), channel_names=geometry.names,
        channel_positions_m=geometry.coords, channel_mask=np.any(masks, axis=0).any(axis=0) if masks.ndim == 3 else masks.any(axis=0),
        trial_channel_mask=masks, preprocessing=profile.preprocessing, metadata=pd.DataFrame(metadata), event_timeline=t,
        identity_table=DatasetIdentityTable.from_source_rows([target["source_subject_id"]], [target["source_dataset_id"]]),
        provenance={"source": "original_candidate_record_aware_authorized_prefix", "source_sha256": target["source_sha256"],
            "source_reference": target["reference"] if profile.reference_channel is None else "measured_channel:"+profile.reference_channel,
            "parent_source_reference": target["reference"], "signal_unit": "V", "source_sample_rate_hz": target["sample_rate_hz"],
            "selection_epoch_capability": profile.execution_capability(), "selection_epoch_profile": profile.record(),
            "selection_epoch_execution": [p[2] for p in parts], "source_records": [p[3].record() for p in parts],
            "authorized_label_count": count, "candidate_task_contract": target["candidate_task_contract"]})
    data.validate(require_labels=True)
    return data


@dataclass(frozen=True, init=False)
class AuthorizedRecordPublicHistory:
    record_id: str
    samples: np.ndarray
    calibration_records: tuple[CalibrationRecord, ...]
    binding: dict

    @property
    def digest(self):
        return semantic_sha256(self.binding)


@dataclass(frozen=True, init=False)
class RecordHistoryRefusal:
    """Original issuer refusal for one exact request; no reading capability."""
    binding: dict

    @property
    def digest(self):
        return semantic_sha256(self.binding)


def _record_history_refusal_request(compiler, view, *, prefix, conditioning_stop, available_stop=None,
                                    program="record_prequential_public_history/1"):
    prefix = compiler.assert_original_prefix(prefix)
    return {"program": program, "prefix_identity": calibration_prefix_identity(prefix),
        "current_view": view.record(), "conditioning_stop": conditioning_stop,
        "available_stop": view.available_sample if available_stop is None else available_stop,
        "target_contract_sha256": semantic_sha256(prefix.target_contract),
        "C_record_input_sha256": {raw.record_id: array_digest(raw.samples, raw.onsets, raw.target_mask)
                                  for raw in prefix.calibration_records},
        "source_owner_sha256": getattr(compiler, "producer_source_sha256", _SELECTION_EXECUTION_SOURCE_SHA256)}


def _record_history_refusal_key(binding):
    return semantic_sha256(binding["request"])


def _register_record_history_refusal(compiler, view, *, prefix, conditioning_stop, available_stop,
                                     program, error):
    if type(compiler) not in {CalibrationPrefixCompiler, RecordCalibrationPrefixCompiler}:
        raise TypeError("history refusal must be issued by its original acquisition compiler")
    clock_roster = [{"record_id": source.record_id, "source_sha256": source.source_sha256,
        "native_sample_count": source.native_sample_count,
        "native_onsets_sha256": array_digest(source.native_onsets),
        "unavailable_public_onsets_sha256": None if source.unavailable_public_onsets is None
            else array_digest(source.unavailable_public_onsets),
        "public_record_sha256": semantic_sha256(source.public_record())} for source in compiler.sources]
    binding = {"schema": "n2p3_record_history_refusal/1",
        "request": _record_history_refusal_request(compiler, view, prefix=prefix,
            conditioning_stop=conditioning_stop, available_stop=available_stop, program=program),
        "original_record_clock_roster": clock_roster, "error": error.record(),
        "raw_read_permission_granted": False, "future_labels_granted": False}
    return _store_record_history_refusal(compiler, binding)


def _store_record_history_refusal(compiler, binding):
    refusal = object.__new__(RecordHistoryRefusal)
    object.__setattr__(refusal, "binding", freeze_metadata(binding))
    if not hasattr(compiler, "_record_history_refusals"):
        compiler._record_history_refusals = {}
    compiler._record_history_refusals[_record_history_refusal_key(binding)] = refusal
    return refusal


def _lookup_record_history_refusal(compiler, view, *, prefix, conditioning_stop, available_stop=None,
                                   program="record_prequential_public_history/1"):
    request = _record_history_refusal_request(compiler, view, prefix=prefix,
        conditioning_stop=conditioning_stop, available_stop=available_stop, program=program)
    refusal = getattr(compiler, "_record_history_refusals", {}).get(semantic_sha256(request))
    if refusal is None or semantic_sha256(refusal.binding["request"]) != semantic_sha256(request):
        raise ValueError("history refusal was not issued for this exact original request")
    return refusal


def _restore_record_history_refusal(compiler, binding, *, future_views=()):
    if (not isinstance(compiler, _PreparedCalibrationPrefixCompiler)
            or binding["schema"] != "n2p3_record_history_refusal/1"
            or binding["raw_read_permission_granted"] is not False or binding["future_labels_granted"] is not False
            or binding["error"]["kind"] != "authorized_history_unavailable"):
        raise ValueError("prepared refusal requires its original typed issuer record, without reading permissions")
    request = binding["request"]
    prefixes = [p for p in compiler._prefixes.values() if calibration_prefix_identity(p) == request["prefix_identity"]]
    if len(prefixes) != 1:
        raise ValueError("prepared history refusal changed its original prefix identity")
    prefix = prefixes[0]
    views = [*future_views, *(group.view for p in compiler._prefixes.values() for group in p.complete_groups)]
    matches = [view for view in views if semantic_sha256(view.record()) == semantic_sha256(request["current_view"])]
    if not matches or semantic_sha256(_record_history_refusal_request(compiler, matches[0], prefix=prefix,
            conditioning_stop=request["conditioning_stop"], available_stop=request["available_stop"],
            program=request["program"])) != semantic_sha256(request):
        raise ValueError("prepared history refusal changed its exact view, C inputs, issuer or clock request")
    roster = prefix.target_contract["source_record_roster"]
    clock_roster = binding["original_record_clock_roster"]
    if (len(clock_roster) != len(roster) or {row["record_id"] for row in clock_roster} != set(roster)
            or any(any(row[key] != roster[row["record_id"]][key] for key in
                ("source_sha256", "native_sample_count")) for row in clock_roster)):
        raise ValueError("prepared refusal changed the original acquisition roster")
    key = _record_history_refusal_key(binding)
    if key in getattr(compiler, "_record_history_refusals", {}):
        raise ValueError("prepared input capsule repeats an original history refusal")
    return _store_record_history_refusal(compiler, binding)


def _record_history_key(binding):
    return semantic_sha256({key: binding[key] for key in ("source_sha256", "record_id", "current_decision_id",
        "prefix_identity", "conditioning_sample_support", "score_sample_support")})


def _register_record_history(compiler, samples, prefix, binding):
    result = object.__new__(AuthorizedRecordPublicHistory)
    for name, value in {"record_id": binding["record_id"], "samples": _immutable_array(samples, np.float64),
        "calibration_records": prefix.calibration_records, "binding": freeze_metadata(binding)}.items():
        object.__setattr__(result, name, value)
    if not hasattr(compiler, "_record_history_scopes"):
        compiler._record_history_scopes = {}
    compiler._record_history_scopes[_record_history_key(binding)] = result
    return result


def _prepared_record_read_scope(compiler, view, *, prefix, conditioning_stop, available_stop=None,
                                program="record_prequential_public_history/1"):
    prefix = compiler.assert_original_prefix(prefix)
    stop = view.available_sample if available_stop is None else available_stop
    request = _record_history_refusal_request(compiler, view, prefix=prefix, conditioning_stop=conditioning_stop,
        available_stop=available_stop, program=program)
    refusal = getattr(compiler, "_record_history_refusals", {}).get(semantic_sha256(request))
    if refusal is not None:
        from research.history_authority import HistoryUnavailableError

        error = refusal.binding["error"]
        raise HistoryUnavailableError(error["reason"], failure_kind=error["failure_kind"], evidence=error.get("evidence"))
    scopes = getattr(compiler, "_record_history_scopes", {})
    matches = [cap for cap in scopes.values() if cap.binding["source_sha256"] == view.source_sha256
        and cap.binding["current_decision_id"] == view.decision_id
        and cap.binding["prefix_identity"] == calibration_prefix_identity(prefix)
        and tuple(cap.binding["conditioning_sample_support"]) == (0, conditioning_stop)
        and tuple(cap.binding["score_sample_support"]) == (conditioning_stop, stop)]
    if len(matches) != 1 or program != "record_prequential_public_history/1" or stop != view.available_sample:
        raise ValueError("prepared history was not issued by the admitted original decoder for these exact C/P/T inputs")
    cap = matches[0]
    original_view = dict(cap.binding["current_view"])
    actual_view = view.record()
    if (semantic_sha256(original_view) != semantic_sha256(actual_view)
            or cap.binding["C_record_input_sha256"] != {raw.record_id:
                array_digest(raw.samples, raw.onsets, raw.target_mask) for raw in prefix.calibration_records}
            or cap.binding["samples_sha256"] != array_digest(cap.samples)):
        raise ValueError("prepared original history cap changed its native samples, C authority or current physical view")
    return cap

class _PreparedRecordCalibrationPrefixCompiler(_PreparedCalibrationPrefixCompiler):
    """The same original record-aware before-group join from admitted bytes."""

    def before_group(self, parent, held):
        parent = self.assert_original_prefix(parent)
        self._prefix_ownership[parent.budget].assert_group_match(held)
        scopes = parent.target_contract["record_scopes"]
        records = [r for r in parent.calibration_records if scopes[r.record_id]["source_sha256"] == held.view.source_sha256]
        if len(records) != 1:
            raise ValueError("prepared held group has no unique original physical record")
        raw = records[0]
        base = scopes[raw.record_id]["global_label_indices"][0]
        absolute = held.view.onsets+held.view.raw_scope[0]
        indices = np.searchsorted(raw.onsets, absolute)
        if (np.any(indices >= len(raw.onsets)) or not np.array_equal(raw.onsets[indices], absolute)
                or not np.array_equal(raw.target_mask[indices], held.labels)):
            raise ValueError("prepared C group differs from its original clock/label authority")
        count = int(np.searchsorted(raw.onsets, held.view.raw_scope[0], side="left"))
        earlier = self.compile(base+count)
        scope = earlier.target_contract["record_scopes"].get(raw.record_id)
        if scope is not None and scope["read_sample_support"][1] > held.view.raw_scope[0]:
            raise ValueError("prepared earlier prefix overlaps actual held raw observations")
        return earlier

    record_read_scope = _prepared_record_read_scope


def prepare_dataset_independent_prefixes(declaration, *, budgets, profiles, authorization_preprocessing,
                                          response_seconds=1.2, pre_roll_seconds=.2):
    """One original-data resolver shared by public preparation and replay."""
    if declaration.get("dataset_id", BNCI2014_008_DATASET_ID) == BNCI2014_008_DATASET_ID:
        return prepare_bnci_independent_prefixes(declaration["path"], source_sha256=declaration["sha256"],
            budgets=budgets, profiles=profiles, authorization_preprocessing=authorization_preprocessing,
            response_seconds=response_seconds, pre_roll_seconds=pre_roll_seconds)
    compiler = CalibrationPrefixCompiler.from_dataset_source(declaration, budgets=budgets, profiles=profiles,
        authorization_preprocessing=authorization_preprocessing, response_seconds=response_seconds,
        pre_roll_seconds=pre_roll_seconds)
    return compiler.independent_views()
