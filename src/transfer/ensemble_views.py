"""Shared physical decisions, with separate labeled fitting and unlabeled scoring.

Raw observations retain their native reference and sampling grid. Expert-specific
epoch transformations are explicit frozen profiles; they never alter the shared
physical decision or read beyond its declared observations and availability.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
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
from data.epochs import EpochDataset, PreprocessingSpec, parse_preprocessing_spec
from data.events import ScheduledEventTimeline
from data.finite_filter_operator import FiniteFilterOperator
from data.identity import DatasetIdentityTable, origin_subject_key
from data.lineage import DataLineage
from data.preprocess import (
    design_continuous_sos,
    filter_bidirectional_segment,
    source_epoch_support,
    transform_epoch_samples,
)
from research.contracts import semantic_sha256
from transfer.candidate_decision import parse_row_column_target, row_column_target
from transfer.noise_model import array_digest, freeze_metadata, require_identifiers, require_sha256


def _immutable_array(values, dtype):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


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
    samples: np.ndarray
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
        samples = np.asarray(self.samples)
        if (samples.shape != (scope[1] - scope[0], len(channels))
                or samples.dtype.kind != "f" or not np.isfinite(samples).all()):
            raise ValueError("actual raw scope and finite samples-by-EEG-channel values must agree")
        if not isinstance(self.decision, MembershipDecision):
            raise TypeError("public physical membership requires a MembershipDecision")
        onset = np.asarray(self.onsets)
        if (onset.ndim != 1 or len(onset) != len(self.decision.event_ids)
                or onset.dtype.kind not in "iu" or np.any(onset[1:] <= onset[:-1])
                or not len(onset) or onset[0] < 0 or onset[-1] >= len(samples)):
            raise ValueError("public event onsets must be ordered integer samples within the actual raw scope")
        object.__setattr__(self, "samples", _immutable_array(samples, np.float64))
        object.__setattr__(self, "onsets", _immutable_array(onset, np.int64))
        object.__setattr__(self, "decision", _immutable_decision(self.decision))
        object.__setattr__(self, "raw_scope", tuple(int(value) for value in scope))
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "channel_types", kinds)
        object.__setattr__(self, "available_sample", int(self.available_sample))

    def physical_record(self):
        return {"schema": "n2p3_prediction_decision_physical_support/1",
                "record_id": self.record_id, "participant_key": self.participant_key,
                "decision_id": self.decision_id, "source_sha256": self.source_sha256,
                "source_relative_path": self.source_relative_path,
                "source_dataset_id": self.source_dataset_id, "source_subject_id": self.source_subject_id,
                "raw_scope": list(self.raw_scope), "sample_rate_hz": self.sample_rate_hz,
                "channel_names": list(self.channel_names), "channel_types": list(self.channel_types),
                "reference": self.reference, "signal_unit": self.signal_unit,
                "onsets": self.onsets.tolist(), "decision": self.decision.record(),
                "available_sample": self.available_sample}

    @property
    def physical_support_digest(self):
        return semantic_sha256(self.physical_record())

    @property
    def raw_input_digest(self):
        return semantic_sha256({"physical_support_digest": self.physical_support_digest,
                                "raw_samples_sha256": array_digest(self.samples)})

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
                           reference, reference_channel=None, output_dtype="float32"):
        return cls(preprocessing, source_sample_rate_hz, tuple(channel_names), reference,
                   _profile_operator(preprocessing, source_sample_rate_hz), reference_channel, output_dtype)

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
        return {"schema": "n2p3_selection_epoch_profile/1", "preprocessing": self.preprocessing.record(),
                "source_sample_rate_hz": self.source_sample_rate_hz,
                "channel_names": list(self.channel_names), "reference": self.reference,
                "reference_channel": self.reference_channel, "output_dtype": self.output_dtype,
                "operator": None if self.operator is None else self.operator.record(),
                "epoch_geometry": self.epoch_geometry}

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

    def transform(self, view: PredictionDecisionView):
        """Return actual epochs, their observation masks and an execution receipt."""
        if not isinstance(view, PredictionDecisionView):
            raise TypeError("epoch prediction transforms consume unlabeled PredictionDecisionView inputs")
        if view.sample_rate_hz != self.source_sample_rate_hz or view.reference != self.reference:
            raise ValueError("epoch profile differs from the actual native rate or acquisition reference")
        native = tuple(map(canonical_channel_name, view.channel_names))
        required = (*map(canonical_channel_name, self.channel_names),)
        if self.reference_channel is not None:
            required += (self.reference_channel,)
        if set(required) - set(native):
            raise ValueError("epoch projection requests EEG or measured-reference channels absent from actual observations")
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
        projection = signal_dependency_matrix(view.channel_names, self.channel_names,
                                               reference_channel=self.reference_channel).astype(self.output_dtype)
        epochs = np.einsum("nct,oc->not", epochs, projection, optimize=True)
        mask = np.broadcast_to(view.decision.evidence_indices[:, None] >= 0,
                               (len(view.onsets), len(self.channel_names))).copy()
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


@dataclass(frozen=True)
class CalibrationGroupViews:
    groups: tuple[CalibrationSelection, ...]
    ledger: dict

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "ledger", freeze_metadata(self.ledger))


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
        parents.append(DataLineage.source(parameters=view.record()))
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
