"""Original BI2014a MAT/CSV byte admission and one physical schedule decoder.

The MAT route retains the original doubles. CSV retains its published rounding;
both expose the original sample-index clock and audited row/column flash fields.
The label-K adapter exports public memberships and keeps answers on the parent.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from data.bi2014a_candidate import (
    BI2014A_CANDIDATE_TASK_CONTRACT,
    BI2014A_CHANNELS,
    BI2014A_SFREQ,
    BI2014ACandidateRecord,
    recover_bi2014a_schedule,
)
from data.bi2014a_schedule import BI2014A_FLASH_SCHEDULE
from data.candidate_task import MembershipDecision
from data.exclusions import select_allowed_subjects
from data.identity import origin_subject_key
from transfer.noise_model import array_digest

BI2014A_ORIGINAL_FORMATS = frozenset({"original_mat", "original_csv", "original_mat_zip", "original_csv_zip"})
_FACTORY_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BI2014aOriginalRecord:
    subject_id: str
    source_sha256: str
    source_relative_path: str
    source_format: str
    eeg_uv: np.ndarray
    recorded_times_s: np.ndarray
    schedule: BI2014ACandidateRecord
    source_receipt: dict
    selection_boundary_samples: np.ndarray


def _original_payload(path, expected_sha256, source_format, source_member):
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("BI original source bytes differ from the declared acquisition")
    if source_format not in BI2014A_ORIGINAL_FORMATS:
        raise ValueError("BI original source format must be explicitly declared")
    kind = "mat" if source_format.startswith("original_mat") else "csv"
    subject = re.fullmatch(r"subject_([0-9]{2})", path.stem)
    if subject is None:
        raise ValueError("BI participant identity requires its original subject_NN filename")
    subject_id = subject[1]
    if int(subject_id) not in select_allowed_subjects("BI2014a", [int(subject_id)], explicit=True):
        raise ValueError("BI participant is outside the admitted original cohort")
    expected_member = f"subject_{subject_id}.{kind}"
    artifact = {"path": str(path.resolve()), "sha256": digest, "bytes": len(payload)}
    if source_format.endswith("_zip"):
        if path.suffix.lower() != ".zip" or source_member not in {None, expected_member}:
            raise ValueError("BI original format/member differs from its original participant acquisition")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = [entry for entry in archive.infolist() if entry.filename == expected_member]
            if len(entries) != 1:
                raise ValueError("BI archive must contain one declared original participant member")
            entry = entries[0]
            member = archive.read(entry)
        receipt = {"name": expected_member, "sha256": hashlib.sha256(member).hexdigest(),
                   "bytes": len(member), "crc32": f"{entry.CRC:08x}"}
        relative = f"{path.name}::{expected_member}"
    else:
        if path.name != expected_member or source_member is not None:
            raise ValueError("BI original format differs from its original participant acquisition")
        member, receipt, relative = payload, None, path.name
    return subject_id, member, kind, artifact, receipt, relative


def _clock_record(times, kind):
    native = np.arange(len(times), dtype=np.float64) / BI2014A_SFREQ
    exact = np.array_equal(times, native)
    if kind == "mat":
        valid = exact
    else:
        # The original CSV uses five significant digits. Its rounded timestamps
        # never move physical flashes off the original integer sample axis.
        rounding = np.zeros_like(native)
        positive = native > 0
        rounding[positive] = .5 * 10. ** (np.floor(np.log10(native[positive])) - 4)
        valid = np.all(np.abs(times - native) <= rounding + 1e-12) and np.all(np.diff(times) >= 0)
    if not valid:
        raise ValueError("BI recorded clock differs from its original 512 Hz sample-index clock")
    return {"schema": "n2p3_bi2014a_original_clock/1", "sample_rate_hz": BI2014A_SFREQ,
        "sample_count": len(times), "event_clock": "original_integer_sample_index",
        "recorded_time_exact_on_sample_grid": bool(exact),
        "recorded_time_sha256": array_digest(times),
        "maximum_recorded_time_rounding_s": float(np.max(np.abs(times - native)))}


def read_bi2014a_original_record(path, *, source_sha256, source_format, source_member=None,
                                subject_id=None):
    """Read one original file/member without writing an extracted EEG or CSV."""
    actual_subject, payload, kind, artifact, member, relative = _original_payload(
        path, source_sha256, source_format, source_member)
    if subject_id is not None and subject_id != actual_subject:
        raise ValueError("BI declared participant differs from the original member identity")
    values = (loadmat(io.BytesIO(payload), variable_names=["samples"]).get("samples") if kind == "mat"
              else pd.read_csv(io.BytesIO(payload), header=None).to_numpy())
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] != 19 or not len(values):
        raise ValueError("BI original table must be samples by its 19 declared physical fields")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("BI original table contains a non-finite clock, EEG, marker or label")
    clock = _clock_record(values[:, 0], kind)
    markers, labels = values[:, 17], values[:, 18]
    flash = (markers >= 20) & (markers <= 85)
    if (np.any(markers != np.floor(markers)) or not np.isin(labels, [0, 1, 2]).all()
            or np.any(labels[~flash] != 0)):
        raise ValueError("BI original marker/label fields differ from the physical event contract")
    schedule = recover_bi2014a_schedule(f"subject_{actual_subject}", markers, labels,
                                       stage=f"{source_format}_before_preprocessing")
    # One immutable decoded backing owns EEG and its original time columns.
    # Original flash/label arrays stay in the parent-side schedule object.
    compact = np.ascontiguousarray(values[:, :17])
    frozen = np.frombuffer(compact.tobytes(), dtype=np.float64).reshape(compact.shape)
    receipt = {"schema": "n2p3_bi2014a_original_record/1", "original_format": source_format,
        "artifact": artifact, "member": member, "variable": "samples" if kind == "mat" else None,
        "field_shape": list(values.shape), "field_axes": {"time": 0, "EEG": list(range(1, 17)),
            "original_marker": 17, "parent_label": 18}, "EEG_unit": "microvolt", "clock": clock,
        "numeric_fields_sha256": array_digest(frozen), "original_fields_sha256": array_digest(values),
        "decoder_source_sha256": _FACTORY_SOURCE_SHA256}
    return BI2014aOriginalRecord(actual_subject, source_sha256, relative, source_format,
        frozen[:, 1:17], frozen[:, 0], schedule, receipt,
        np.frombuffer(np.flatnonzero(np.isin(markers, [100, 104])).astype(np.int64).tobytes(), dtype=np.int64))


def _parent_target(record, rows):
    """Use actual target-coded flashes; a new incomplete episode can lack truth."""
    schedule = record.schedule
    codes = schedule.scheduled_flash_code[rows]
    decoded = [BI2014A_FLASH_SCHEDULE.decode(code) for code in codes]
    targets = {axis: {event.candidate_index for event in decoded if event.is_target and event.axis == axis}
               for axis in ("row", "column")}
    if any(len(values) > 1 for values in targets.values()):
        raise ValueError("BI physical episode contains conflicting actual target-coded membership")
    if not all(len(values) == 1 for values in targets.values()):
        return None
    from transfer.candidate_decision import row_column_target

    return row_column_target(next(iter(targets["row"])), next(iter(targets["column"])))


def bi2014a_dataset_adapter(record_sources, *, source_format):
    """Issue BI label-K recordings for the shared original-source prefix owner.

    One archive/file is one participant acquisition. Target codes, raw labels,
    feedback markers and truth stay in this original decoder's private state.
    """
    from data.candidate_input import (
        CandidateDatasetAdapter,
        OriginalCandidateRecording,
        OriginalCandidateSelection,
    )

    if len(record_sources) != 1:
        raise ValueError("BI label-K requires the one original continuous participant recording")
    declaration = record_sources[0]
    record = read_bi2014a_original_record(declaration["path"], source_sha256=declaration["sha256"],
        source_format=source_format, source_member=declaration.get("member"), subject_id=declaration.get("subject"))
    schedule = record.schedule
    public_ids = np.asarray([event.candidate_index + (6 if event.axis == "column" else 0)
        for event in map(BI2014A_FLASH_SCHEDULE.decode, schedule.scheduled_flash_code)], dtype=np.int64)
    participant = origin_subject_key("BI2014a", record.subject_id)
    selections = []
    partitions = np.searchsorted(record.selection_boundary_samples, schedule.scheduled_flash_sample, side="left")
    for episode, partition in enumerate(dict.fromkeys(partitions.tolist()), start=1):
        rows = np.flatnonzero(partitions == partition)
        repetitions = np.arange(len(rows), dtype=np.int64) // 12
        decision = MembershipDecision.row_column(public_ids[rows], n_rows=6, n_columns=6,
            event_ids=tuple(f"BI2014a:{record.subject_id}:{int(row)}" for row in rows),
            repetition_indices=repetitions, repetitions=int(repetitions.max()) + 1,
            evidence_indices=np.arange(len(rows)), timeline_complete=True)
        name = f"subject_{record.subject_id}:episode{episode}"
        selections.append(OriginalCandidateSelection(name, tuple(rows.tolist()), decision, _parent_target(record, rows)))

    def read_native(start, stop):
        if not 0 <= start < stop <= len(record.eeg_uv):
            raise ValueError("BI original sample request exceeds the actual native acquisition")
        return np.asarray(record.eeg_uv[start:stop], dtype=np.float64) * 1e-6

    recording = OriginalCandidateRecording(record_id=f"BI2014a:{record.subject_id}:original-continuous",
        source_sha256=record.source_sha256, source_relative_path=record.source_relative_path,
        participant_key=participant, source_subject_id=record.subject_id, session_id="0", run_id="0",
        sample_rate_hz=BI2014A_SFREQ, channel_names=tuple(name.upper() for name in BI2014A_CHANNELS),
        reference="right earlobe", native_sample_count=len(record.eeg_uv),
        native_onsets=schedule.scheduled_flash_sample,
        authorized_binary_labels=BI2014A_FLASH_SCHEDULE.target_mask(schedule.scheduled_flash_code),
        selections=tuple(selections), source_receipt=record.source_receipt, read_native=read_native)
    artifact = {**record.source_receipt["artifact"], "subject": record.subject_id,
                "original_format": source_format}
    if record.source_receipt["member"] is not None:
        artifact.update(member=record.source_receipt["member"]["name"],
                        member_sha256=record.source_receipt["member"]["sha256"],
                        member_crc32=record.source_receipt["member"]["crc32"])
    return CandidateDatasetAdapter("BI2014a", participant, record.subject_id, source_format,
        (recording,), (artifact,), BI2014A_CANDIDATE_TASK_CONTRACT.record(),
        "label_K_then_independent_later_row_column_decisions", _FACTORY_SOURCE_SHA256)
