"""Original-path BrainSync raw and independent calibration-label capabilities.

The issuer owns paths, full manifests and confirmed targets. Downstream receives
only registered immutable numerical/public operands. No fit or preprocessing.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.brainsync import IndependentRecordingScope
from data.brainsync_contract import (
    BRAIN_SYNC_SESSION_SCHEMA,
    validate_brainsync_bids_recording,
    validate_brainsync_bids_session,
)
from data.channel import canonical_channel_name
from data.contract import (
    BRAIN_SYNC_ACQUISITION_REFERENCE,
    BRAIN_SYNC_EEG_CHANNELS,
    brainsync_reference_record,
)
from data.dataset import read_raw
from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.noise_model import array_digest, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VOCABULARY = tuple(str(q) for q in range(1, 10))
_RAW_ISSUED, _LABELS_ISSUED = WeakKeyDictionary(), WeakKeyDictionary()
_ISSUER_KEY = object()


@dataclass(frozen=True)
class BrainSyncRecordingScope:
    """Declared original-record separation, including genuine empty-C inference.

    A nonempty held roster is explicit. Calibration may be empty; no dummy
    record is introduced. The existing IndependentRecordingScope remains an
    accepted input with its own original constraints and record semantics.
    """

    calibration_session_ids: tuple[str, ...]
    evaluation_session_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("calibration_session_ids", "evaluation_session_ids"):
            values = tuple(getattr(self, name))
            if (any(not isinstance(value, str) or not value.strip() for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError("recording scope requires unique original session identities")
            object.__setattr__(self, name, values)
        if not self.evaluation_session_ids:
            raise ValueError("recording scope requires its explicit held recording roster")
        if set(self.calibration_session_ids) & set(self.evaluation_session_ids):
            raise ValueError("calibration and inference recordings must be disjoint")

    def validate_application(self, session_id, fit_sessions=()):
        if session_id not in self.calibration_session_ids+self.evaluation_session_ids:
            raise ValueError("recording is outside its declared original recording scope")
        if not set(fit_sessions) <= set(self.calibration_session_ids):
            raise ValueError("fit scope exceeds the actual declared calibration recordings")

    def record(self):
        return {"mode": "independent_recording_holdout",
            "calibration_session_ids": list(self.calibration_session_ids),
            "evaluation_session_ids": list(self.evaluation_session_ids)}


def _readonly_bool(value):
    array = np.ascontiguousarray(value, dtype=bool)
    return np.frombuffer(array.tobytes(), dtype=bool).reshape(array.shape)


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_sha(path):
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            state.update(block)
    return state.hexdigest()


@dataclass(frozen=True)
class PublicFlash:
    event_id: str
    sample: int
    segment_sample: int
    block_id: int
    trial_index: int
    candidate_id: str


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class RawCapability:
    """Only original authorized rows/public fields; no issuer/path/manifest ref."""

    eeg: np.ndarray
    eog: np.ndarray
    events: tuple[PublicFlash, ...]
    membership: np.ndarray
    candidate_vocabulary: tuple[str, ...]
    public_binding: dict
    identity_sha256: str


@dataclass(frozen=True)
class AuthorizedLabel:
    event_id: str
    sample: int
    block_id: int
    trial_index: int
    binary_label: int


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class CalibrationLabels:
    labels: tuple[AuthorizedLabel, ...]
    public_binding: dict
    identity_sha256: str


def _raw_identity(capability):
    return metadata_digest({"public_binding": capability.public_binding,
        "eeg_sha256": array_digest(capability.eeg), "eog_sha256": array_digest(capability.eog),
        "membership_sha256": array_digest(capability.membership),
        "events": [vars(event) for event in capability.events],
        "candidate_vocabulary": capability.candidate_vocabulary})


def _labels_identity(capability):
    return metadata_digest({"public_binding": capability.public_binding,
        "labels": [vars(label) for label in capability.labels]})


def require_raw_capability(capability):
    receipt = _RAW_ISSUED.get(capability) if type(capability) is RawCapability else None
    if (receipt is None or capability.identity_sha256 != receipt
            or _raw_identity(capability) != receipt):
        raise ValueError("raw capability was not issued from original admitted BrainSync bytes")
    return capability


def require_calibration_labels(capability):
    receipt = _LABELS_ISSUED.get(capability) if type(capability) is CalibrationLabels else None
    if (receipt is None or capability.identity_sha256 != receipt
            or _labels_identity(capability) != receipt):
        raise ValueError("calibration labels were not issued by the original confirmed-session owner")
    return capability


def allowed_candidates(raw_capabilities, label_capability):
    """Join only granted C labels to original event identity and public membership."""
    caps = (raw_capabilities,) if type(raw_capabilities) is RawCapability else tuple(raw_capabilities)
    if not caps:
        raise ValueError("C label constraints require their original raw/public event capabilities")
    caps = tuple(require_raw_capability(cap) for cap in caps)
    raw = caps[0]
    labels = require_calibration_labels(label_capability)
    binding, label_binding = raw.public_binding, labels.public_binding
    if (any(cap.public_binding["role"] != "calibration"
            or cap.public_binding["record_id"] != label_binding["record_id"]
            or cap.public_binding["independent_recording_scope_sha256"] != label_binding["independent_recording_scope_sha256"]
            for cap in caps)
            or len({cap.public_binding["segment_id"] for cap in caps}) != len(caps)):
        raise ValueError("C labels and raw must belong to the same declared calibration record")
    lookup = {event.event_id: (cap.membership[row], event) for cap in caps
        for row, event in enumerate(cap.events)}
    allowed = np.ones(len(raw.candidate_vocabulary), dtype=bool)
    used = 0
    for label in labels.labels:
        if label.event_id not in lookup:
            raise ValueError("authorized C label is beyond this raw capability's actual availability")
        membership, event = lookup[label.event_id]
        if (event.sample != label.sample or event.block_id != label.block_id
                or event.trial_index != label.trial_index):
            raise ValueError("C labels and public events have different original native identities")
        allowed &= membership == label.binary_label
        used += 1
    if not allowed.any():
        raise ValueError("authorized labels contradict the original public membership")
    return {"candidate_vocabulary": raw.candidate_vocabulary,
        "allowed_candidates": tuple(q for q, keep in zip(raw.candidate_vocabulary, allowed, strict=True) if keep),
        "authorized_global_event_count": len(labels.labels), "labels_used": used,
        "requested_candidate_count": len(raw.candidate_vocabulary),
        "target_lock_group": binding["target_lock_group"]}


class _RecordingIssuer:
    def __init__(self, validated, scope, role, source_receipt, raw, *, key):
        if key is not _ISSUER_KEY:
            raise TypeError("open original BrainSync paths through the data authority")
        self._validated, self._scope, self._role = validated, scope, role
        self._source_receipt, self._raw = source_receipt, raw
        self._clock = validated.manifest["recording"]["clock"]
        self._segments = tuple(self._clock["segments"])
        self._scope_sha = metadata_digest(scope.record())
        self._record_id = metadata_digest({"session_id": validated.session_id,
            "participant_id": validated.subject_id,
            "acquisition_sha256": source_receipt[str(validated.bids.raw_path)]["sha256"]})
        self._eeg_indices = tuple(index for index, kind in enumerate(validated.bids.channel_types) if kind == "EEG")
        self._eog_indices = tuple(index for index, kind in enumerate(validated.bids.channel_types) if kind == "EOG")
        self._exposures = self._read_original_exposures()

    def _check_original_files(self):
        for name, receipt in self._source_receipt.items():
            if _signature(Path(name)) != receipt["file_identity"]:
                raise ValueError("original admitted source files changed before authorized raw/label access")

    def _event_id(self, event):
        return f"{self._record_id}:{event.trial_id}"

    def _read_original_exposures(self):
        """Original completed and interrupted attempts, not target-label fields."""
        result = [{"event_id": self._event_id(event), "trial_id": event.trial_id,
            "sample": event.sample, "duration_seconds": event.duration_seconds,
            "block_id": event.block_id, "trial_index": event.trial_index,
            "candidate_id": event.candidate_id, "exposure_role": "completed_protocol_stimulus",
            "original_TSV_row": event.row_number} for event in self._validated.bids.stimuli]
        if self._validated.bids.excluded_intervals:
            with self._validated.bids.events_path.open(encoding="utf8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            for interval in self._validated.bids.excluded_intervals:
                row = rows[interval.row_number-2]
                if row["trial_type"] != "interrupted_stimulus":
                    raise ValueError("original excluded interval does not identify its actual interrupted TSV row")
                candidate = row["digit"].strip()
                trial_text = row["trial_index"].strip()
                trial_index = int(trial_text) if trial_text.isdecimal() and 1 <= int(trial_text) <= 45 else None
                result.append({"event_id": f"{self._record_id}:interrupted-row:{interval.row_number}:{row['trial_id']}",
                    "trial_id": row["trial_id"], "sample": interval.sample,
                    "duration_seconds": interval.duration_seconds, "block_id": interval.block_id,
                    "trial_index": trial_index,
                    "candidate_id": candidate if candidate in VOCABULARY else None,
                    "exposure_role": "interrupted_physical_stimulus", "original_TSV_row": interval.row_number})
        return tuple(sorted(result, key=lambda event: (event["sample"], event["original_TSV_row"])))

    def _public_history(self, available_stop):
        """Keep both saved clock programs; assign no unrecorded timing certainty."""
        pauses = self._validated.manifest["recording"]["timeline"].get("acquisition_pauses", [])
        visible_pauses = [{key: pause[key] for key in
            ("pause_id", "boundary_sample", "start_wall_seconds", "end_wall_seconds") if key in pause}
            for pause in pauses if pause["boundary_sample"] < available_stop]
        segments = []
        for segment in self._segments:
            if segment["stored_start"] >= available_stop:
                continue
            matching = [pause for pause in pauses if pause["boundary_sample"] == segment["stored_start"]]
            wall_origin = (Fraction.from_float(float(matching[-1]["end_wall_seconds"]))
                if segment["segment_id"] else Fraction(0))
            record = {"segment_id": segment["segment_id"], "stored_start": segment["stored_start"],
                "stored_stop": min(segment["stored_stop"], available_stop),
                "timeline_wall_origin_seconds_exact_ratio": [wall_origin.numerator, wall_origin.denominator]}
            for key in ("first_sample_estimate_monotonic_ns", "anchor_source", "relative_rate_assumption",
                    "delivery_residual_min_seconds", "delivery_residual_max_seconds"):
                if key in segment:
                    record[key] = segment[key]
            if segment["stored_stop"] <= available_stop and "closed_monotonic_ns" in segment:
                record["closed_monotonic_ns"] = segment["closed_monotonic_ns"]
            segments.append(record)
        exposures = []
        for original in self._exposures:
            if original["sample"] >= available_stop:
                continue
            segment = next((item for item in segments if item["stored_start"] <= original["sample"] < item["stored_stop"]), None)
            if segment is None:
                raise ValueError("an actual physical exposure has no admitted acquisition segment")
            local = original["sample"]-segment["stored_start"]
            wall_origin = Fraction(*segment["timeline_wall_origin_seconds_exact_ratio"])
            wall = wall_origin+Fraction(local, 250)
            duration = Fraction.from_float(float(original["duration_seconds"]))
            exposure = {**original, "segment_id": segment["segment_id"], "segment_sample": local,
                "timeline_wall_onset_seconds_exact_ratio": [wall.numerator, wall.denominator],
                "duration_seconds_exact_ratio": [duration.numerator, duration.denominator],
                "membership": ([original["candidate_id"] == q for q in VOCABULARY]
                    if original["candidate_id"] is not None else None),
                "candidate_options": (original["candidate_id"],) if original["candidate_id"] is not None else VOCABULARY,
                "membership_role": "observed_public_digit" if original["candidate_id"] is not None else "unknown_exposure_candidate_no_prior_assigned"}
            if "first_sample_estimate_monotonic_ns" in segment:
                physical = Fraction(segment["first_sample_estimate_monotonic_ns"], 10**9)+Fraction(local, 250)
                exposure["delivery_anchored_monotonic_seconds_exact_ratio"] = [physical.numerator, physical.denominator]
            exposures.append(exposure)
        return {"schema": "brainsync_public_physical_exposure_history/1",
            "native_sample_rate_hz": 250, "segments": segments,
            "acquisition_pauses": visible_pauses, "exposures": exposures,
            "timeline_clock_program": "original_stored_time_inverse_per_actual_segment_nominal250Hz",
            "delivery_clock_program": "saved_first_batch_delivery_anchor_plus_within_segment_nominal250Hz",
            "clock_error_role": "original_delivery_estimate_and_pause_mapping_no_absolute_optical_bound_added",
            "prior_segment_responses_reset_to_zero": False,
            "acquisition_segment_role": "original_recorded_IDs_and_boundaries_only",
            "cross_segment_noise_law_owned_by_model": True,
            "unknown_exposure_count": sum(exposure["candidate_id"] is None for exposure in exposures),
            "requested_completed_flashes": self._validated.manifest["experiment"]["blocks"]*45}

    def private_admission_receipt(self):
        """For the trusted parent audit; never attached to an inference capability."""
        return {"role": self._role, "original_files": self._source_receipt,
            "issuer_implementation_sha256": IMPLEMENTATION_SHA256,
            "actual_raw_reader": "data.dataset.read_raw -> MNE lazy get_data",
            "original_session_validator": "data.brainsync_contract",
            "original_clock_validator": "data.recording_clock.validate_recording_clock"}

    def issue_raw(self, segment_id, *, prefix_stop, full_stop, available_stop):
        self._check_original_files()
        if type(segment_id) is not int or not 0 <= segment_id < len(self._segments):
            raise ValueError("raw request must identify one actual acquisition segment")
        segment = self._segments[segment_id]
        start, end = segment["stored_start"], segment["stored_stop"]
        if (any(type(stop) is not int for stop in (prefix_stop, full_stop, available_stop))
                or not start <= prefix_stop < full_stop <= available_stop <= end):
            raise ValueError("P/F read support must remain inside one actual segment and availability")
        values = self._raw.get_data(start=start, stop=full_stop)
        self._check_original_files()
        if values.shape != (8, full_stop-start) or not np.isfinite(values).all():
            raise ValueError("actual native raw read does not contain its finite eight acquired slots")
        eeg, eog = frozen_array(values[self._eeg_indices, :].T), frozen_array(values[self._eog_indices, :].T)
        public_events = tuple(PublicFlash(self._event_id(event), event.sample, event.sample-start,
            event.block_id, event.trial_index, event.candidate_id)
            for event in self._validated.bids.stimuli if start <= event.sample < available_stop)
        membership = _readonly_bool([[event.candidate_id == q for q in VOCABULARY] for event in public_events]
            if public_events else np.empty((0, len(VOCABULARY)), dtype=bool))
        experiment = self._validated.manifest["experiment"]
        block_rows = []
        for block_id in range(1, experiment["blocks"]+1):
            visible = [event for event in public_events if event.block_id == block_id]
            block_rows.append({"block_id": block_id, "requested_flash_count": 45,
                "requested_repetitions": 5, "visible_public_event_count": len(visible),
                "complete45_in_this_segment_and_availability": len(visible) == 45})
        clock_fields = ("schema", "sample_rate_hz", "alignment_level", "absolute_offset_bound_seconds",
            "acquired_samples", "edf_samples", "padding_samples", "sequence_observed",
            "sequence_continuity_verified", "known_missing_samples")
        segment_fields = ("segment_id", "stored_start", "stored_stop", "first_sample_estimate_monotonic_ns",
            "anchor_source", "relative_rate_assumption", "delivery_residual_min_seconds",
            "delivery_residual_max_seconds", "closed_monotonic_ns")
        binding = freeze_metadata({"schema": "brainsync_public_raw_capability/1",
            "issuer_implementation_sha256": IMPLEMENTATION_SHA256, "role": self._role,
            "participant_id": self._validated.subject_id, "session_id": self._validated.session_id,
            "record_id": self._record_id, "segment_id": segment_id,
            "record_segment_id": f"{self._record_id}:segment:{segment_id}",
            "segment_start": start, "segment_stop": end,
            "sample_start": start, "prefix_stop": prefix_stop, "full_stop": full_stop,
            "available_stop": available_stop, "sample_rate_hz": 250., "signal_unit": "V",
            "eeg_channels": tuple(self._validated.bids.channel_names[index] for index in self._eeg_indices),
            "eog_channels": tuple(self._validated.bids.channel_names[index] for index in self._eog_indices),
            "channel_statuses": self._validated.bids.channel_statuses,
            "reference": self._validated.bids.source_reference,
            "reference_details": self._validated.bids.source_reference_details,
            "eog_derivation": {channel: {key: record[key] for key in ("kind", "placement", "reference")}
                for channel, record in self._validated.bids.eog_derivation.items()},
            "operator": "original_native_acquisition_voltage_no_filter_reference_change_or_regression",
            "public_clock": {key: self._clock[key] for key in clock_fields if key in self._clock},
            "public_acquisition_segment": {key: segment[key] for key in segment_fields if key in segment},
            "target_lock_program": "brainsync-gtn-session/4_one_confirmed_target_per_session",
            "target_lock_group": f"{self._record_id}:v4-session-target",
            "public_protocol": {"schema": BRAIN_SYNC_SESSION_SCHEMA,
                "blocks": experiment["blocks"], "digits_per_block": 45, "repetitions_per_digit": 5,
                "sequence_policy": experiment["sequence_policy"]},
            "public_physical_history": self._public_history(available_stop),
            "independent_recording_scope_sha256": self._scope_sha,
            "block_requests": block_rows, "source_prior_or_fit_authority_granted": False})
        cap = RawCapability(eeg, eog, public_events, membership, VOCABULARY, binding, "")
        identity = _raw_identity(cap)
        object.__setattr__(cap, "identity_sha256", identity)
        _RAW_ISSUED[cap] = identity
        return cap

    def issue_labels(self, count, *, ready_at_timestamp_s):
        self._check_original_files()
        if self._role != "calibration":
            raise ValueError("an inference recording issuer has no C-label authority")
        if (type(count) is not int or not 0 <= count <= len(self._validated.bids.stimuli)
                or not np.isfinite(ready_at_timestamp_s)
                or ready_at_timestamp_s < self._validated.calibration_available_timestamp_s):
            raise ValueError("C prefix requires its actual original event count and labels-ready time")
        labels = tuple(AuthorizedLabel(self._event_id(event), event.sample, event.block_id, event.trial_index,
            int(int(event.candidate_id) == self._validated.thought_digit))
            for event in self._validated.bids.stimuli[:count])
        binding = freeze_metadata({"schema": "brainsync_original_C_labels_capability/1",
            "record_id": self._record_id, "participant_id": self._validated.subject_id,
            "session_id": self._validated.session_id, "authorized_original_event_count": count,
            "labels_ready_timestamp_s": self._validated.calibration_available_timestamp_s,
            "requested_ready_at_timestamp_s": float(ready_at_timestamp_s),
            "independent_recording_scope_sha256": self._scope_sha,
            "issuer_implementation_sha256": IMPLEMENTATION_SHA256})
        cap = CalibrationLabels(labels, binding, "")
        identity = _labels_identity(cap)
        object.__setattr__(cap, "identity_sha256", identity)
        _LABELS_ISSUED[cap] = identity
        return cap

    def close(self):
        self._raw.close()


def _open(path, scope, role):
    if not isinstance(scope, (IndependentRecordingScope, BrainSyncRecordingScope)):
        raise TypeError("the original independent calibration/evaluation scope is required")
    root = Path(path).expanduser().resolve()
    session_path = root/"session.json"
    session_bytes = session_path.read_bytes()
    manifest = json.loads(session_bytes)
    # The public original session identity decides whether supervised validation
    # is authorized. Do not inspect a held record's target fields before this.
    public_session_id = manifest.get("session_id")
    scope.validate_application(public_session_id)
    permitted = scope.calibration_session_ids if role == "calibration" else scope.evaluation_session_ids
    if public_session_id not in permitted:
        raise ValueError("original session is outside its declared calibration/inference role")
    paths = {session_path}
    for value in manifest["recording"]["bids"].values():
        candidate = (root/value).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("BIDS source must remain inside its original session directory")
        paths.add(candidate)
    for key in ("metadata_path", "journal_path"):
        value = (manifest["recording"].get("clock") or {}).get(key)
        if value is not None:
            candidate = (root/value).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError("clock source must remain inside its original session directory")
            paths.add(candidate)
    receipts = {}
    for source in sorted(paths):
        before = _signature(source)
        digest = _file_sha(source)
        if _signature(source) != before:
            raise ValueError("original source changed during byte admission")
        receipts[str(source)] = {"sha256": digest, "file_identity": before}
    validator = validate_brainsync_bids_session if role == "calibration" else validate_brainsync_bids_recording
    validated = validator(root)
    scope.validate_application(validated.session_id)
    permitted = scope.calibration_session_ids if role == "calibration" else scope.evaluation_session_ids
    if validated.session_id not in permitted:
        raise ValueError("original session is outside its declared calibration/inference role")
    experiment, bids = validated.manifest["experiment"], validated.bids
    if (bids.sample_rate_hz != 250. or bids.source_reference != BRAIN_SYNC_ACQUISITION_REFERENCE
            or dict(bids.source_reference_details) != brainsync_reference_record()
            or experiment["digits_per_block"] != 45 or experiment["repetitions_per_digit"] != 5
            or validated.manifest["schema"] != BRAIN_SYNC_SESSION_SCHEMA):
        raise ValueError("current BrainSync input requires native250Hz/left-earlobe/sevenEEG/complete45 protocol")
    clock = validated.manifest["recording"].get("clock")
    if clock is None:
        raise ValueError("current continuous inference requires the original finalized sample clock")
    if session_path.read_bytes() != session_bytes:
        raise ValueError("original manifest changed during admission")
    for name, receipt in receipts.items():
        if _signature(Path(name)) != receipt["file_identity"]:
            raise ValueError("original metadata/EEG changed during admission")
    raw = read_raw(bids.raw_path, preload=False)
    try:
        if (float(raw.info["sfreq"]) != bids.sample_rate_hz or raw.n_times != clock["edf_samples"]
                or tuple(map(canonical_channel_name, raw.ch_names)) != tuple(map(canonical_channel_name, bids.channel_names))):
            raise ValueError("actual EEG reader and admitted BIDS rate/channel/native sample extent disagree")
        eeg_names = tuple(bids.channel_names[index] for index, kind in enumerate(bids.channel_types) if kind == "EEG")
        if tuple(map(canonical_channel_name, eeg_names)) != tuple(map(canonical_channel_name, BRAIN_SYNC_EEG_CHANNELS)):
            raise ValueError("raw EEG coordinates do not match the original seven EEG roles")
        for filename in raw.filenames:
            source = Path(filename).resolve()
            if not source.is_relative_to(root):
                raise ValueError("actual EEG reader opened a companion outside the original recording")
            if str(source) not in receipts:
                before = _signature(source)
                receipts[str(source)] = {"sha256": _file_sha(source), "file_identity": before}
                if _signature(source) != before:
                    raise ValueError("actual EEG companion changed during admission")
        return _RecordingIssuer(validated, scope, role, receipts, raw, key=_ISSUER_KEY)
    except BaseException:
        raw.close()
        raise


def open_brainsync_recording(path, *, scope):
    return _open(path, scope, "inference")


def open_brainsync_calibration(path, *, scope):
    return _open(path, scope, "calibration")


def public_record(capability):
    cap = require_raw_capability(capability)
    return {"public_binding": plain_metadata(cap.public_binding),
        "candidate_vocabulary": list(cap.candidate_vocabulary),
        "events": [vars(event) for event in cap.events], "membership": cap.membership.tolist(),
        "eeg_sha256": array_digest(cap.eeg), "eog_sha256": array_digest(cap.eog),
        "eeg_shape": list(cap.eeg.shape), "eog_shape": list(cap.eog.shape),
        "identity_sha256": cap.identity_sha256}


@dataclass(frozen=True)
class PhysicalClockChart:
    """Small immutable chart compiled once after raw capability admission.

    It retains public metadata only. Querying a native grid does not rehash EEG
    for each row/continuous point. This numerical chart grants no raw read.
    """

    public_binding: dict
    capability_identity_sha256: str

    def _sample_times(self, native_samples, clock_program):
        scalar = isinstance(native_samples, (int, np.integer)) and not isinstance(native_samples, (bool, np.bool_))
        rows = (native_samples,) if scalar else tuple(native_samples)
        binding = self.public_binding
        if any(not isinstance(row, (int, np.integer)) or isinstance(row, (bool, np.bool_))
                or not binding["sample_start"] <= row < binding["full_stop"] for row in rows):
            raise ValueError("physical response query must use actually granted native observation rows")
        history = binding["public_physical_history"]
        segment = next(item for item in history["segments"] if item["segment_id"] == binding["segment_id"])
        if clock_program == "timeline_wall":
            origin = Fraction(*segment["timeline_wall_origin_seconds_exact_ratio"])
        elif clock_program == "delivery_anchor":
            if "first_sample_estimate_monotonic_ns" not in segment:
                raise ValueError("the original clock lacks an anchor required by this declared delivery-clock query")
            origin = Fraction(segment["first_sample_estimate_monotonic_ns"], 10**9)
        else:
            raise ValueError("choose one explicit original physical clock program")
        return scalar, tuple(origin+Fraction(int(row)-segment["stored_start"], 250) for row in rows)

    def sample_time_seconds(self, native_samples, *, clock_program, reference_origin=Fraction(0)):
        """Physical row times, including a baseline with no exposure.

        Timeline wall uses the original recording-relative origin. Delivery
        anchor uses original monotonic seconds. A model can explicitly subtract
        one common reference_origin for its drift group; no exposure is chosen
        to define time and this translation does not rescale physical seconds.
        """
        scalar, times = self._sample_times(native_samples, clock_program)
        reference = Fraction(reference_origin)
        values = tuple(time-reference for time in times)
        return values[0] if scalar else values

    def elapsed_seconds(self, native_samples, exposure_id, *, clock_program):
        scalar, times = self._sample_times(native_samples, clock_program)
        history = self.public_binding["public_physical_history"]
        exposure = next((event for event in history["exposures"] if event["event_id"] == exposure_id), None)
        if exposure is None:
            raise ValueError("response query needs its actual visible physical exposure identity")
        if clock_program == "timeline_wall":
            event_time = Fraction(*exposure["timeline_wall_onset_seconds_exact_ratio"])
        elif clock_program == "delivery_anchor":
            if "delivery_anchored_monotonic_seconds_exact_ratio" not in exposure:
                raise ValueError("the original clock lacks an anchor required by this declared delivery-clock query")
            event_time = Fraction(*exposure["delivery_anchored_monotonic_seconds_exact_ratio"])
        else:
            raise ValueError("choose one explicit original physical clock program")
        values = tuple(time-event_time for time in times)
        return values[0] if scalar else values


def compile_physical_clock(capability):
    cap = require_raw_capability(capability)
    return PhysicalClockChart(cap.public_binding, cap.identity_sha256)


def physical_elapsed_seconds(capability, native_sample, exposure_id, *, clock_program):
    """One-off convenience; repeated/grid consumers compile the chart once."""
    return compile_physical_clock(capability).elapsed_seconds(native_sample, exposure_id, clock_program=clock_program)
