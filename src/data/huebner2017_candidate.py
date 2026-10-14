"""Huebner original physical flash sets joined to NEMAR native EEG clocks.

The original marker stream owns the 42 visual positions, 32 selectable choices
and 68-frame character programme. Target-coded markers stay with preparation
and the evaluator; prediction receives their physical sets with that code bit
removed. Each acquisition file keeps its own native clock and noise boundary.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data.candidate_task import CandidateTaskContract, MembershipDecision
from research.contracts import semantic_sha256
from transfer.noise_model import freeze_metadata

DATASET_ID = "Huebner2017"
SOURCE_SAMPLE_RATE_HZ = 1000.
SOURCE_REFERENCE = "nose"
SELECTABLE_POSITIONS = (1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22,
                        24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 37, 39, 40, 41, 42)
BLANK_POSITIONS = (3, 5, 8, 14, 18, 23, 28, 33, 36, 38)
FRAMES_PER_SELECTION = 68
SELECTIONS_PER_RUN = 7
CANDIDATE_TASK = CandidateTaskContract(DATASET_ID, "llp_42_visual_positions_32_choices",
    {"label": "healthy volunteers in original copy-spelling programme"},
    {"stage": "public_processed_dataset_development", "product_confirmation": False},
    "set_membership", (6, 7), SELECTABLE_POSITIONS, (), (), "physical_selectable_position",
    physical_stimulus_ids=tuple(range(1, 43)), blank_position_ids=BLANK_POSITIONS,
    schedule_kind="original_68_frame_character_programme")


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _file_identity(stat):
    # Windows path stat exposes creation time as ctime while fstat exposes
    # change time. Their explicit birth time is the same physical-file field.
    stamp = stat.st_birthtime_ns if os.name == "nt" else stat.st_ctime_ns
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stamp


def read_local_marker_member(path, receipt):
    """Read only one complete local ZIP member and verify its header/CRC/SHA.

    A missing central directory is irrelevant to this exact local-member read.
    Neither archive repair nor EEG extraction is performed.
    """
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        stream.seek(receipt["offset"])
        header = stream.read(30)
        signature, _, flags, method, _, _, crc, packed, raw, name_length, extra = struct.unpack("<IHHHHHIIIHH", header)
        name = stream.read(name_length).decode("utf-8" if flags & 0x800 else "cp437")
        stream.read(extra)
        data_offset = stream.tell()
        if (signature != 0x04034b50 or flags & 8 or method not in {0, 8}
                or name != receipt["name"] or data_offset != receipt["data_offset"]
                or packed != receipt["compressed"] or raw != receipt["raw"]
                or crc != int(receipt["crc32"], 16)):
            raise ValueError("original marker local header differs from its declared byte range")
        compressed = stream.read(packed)
    content = compressed if method == 0 else zlib.decompress(compressed, -15)
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or len(compressed) != packed or len(content) != raw
            or zlib.crc32(content) & 0xffffffff != crc or _sha(content) != receipt["sha256"]):
        raise ValueError("original marker member changed or has incomplete/corrupt bytes")
    return content


@dataclass(frozen=True)
class PhysicalFlash:
    marker_id: str
    sequence_type: int
    software_sample: int
    optical_sample: int | None
    clock_status: str
    physical_positions: tuple[int, ...]
    target_positions: tuple[int, ...]

    def public_record(self):
        return {"marker_id": self.marker_id, "sequence_type": self.sequence_type,
            "software_sample": self.software_sample, "optical_sample": self.optical_sample,
            "clock_status": self.clock_status, "physical_positions": list(self.physical_positions)}


@dataclass(frozen=True)
class HuebnerMarkerRecord:
    flashes: tuple[PhysicalFlash, ...]
    target_positions: tuple[int, ...]
    marker_sha256: str
    source_authority: dict


def decode_original_markers(payload, *, source_authority):
    rows = []
    for line in payload.decode("utf-8-sig").splitlines():
        if not re.match(r"Mk\d+=", line):
            continue
        name, value = line.split("=", 1)
        fields = value.split(",")
        code = re.fullmatch(r"[SO]\s*(\d+)", fields[1].strip())
        rows.append((name, fields[0], None if code is None else int(code[1]), int(fields[2])-1))
    boundaries = [i for i, r in enumerate(rows) if r[1] == "Stimulus" and r[2] in {10, 21, 22}]
    flashes = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=False):
        group = rows[start:stop]
        if group[0][2] not in {21, 22}:
            continue
        symbols = [r[2] for r in group if r[1] == "Stimulus" and r[2] is not None
                   and (101 <= r[2] <= 142 or 201 <= r[2] <= 242)]
        positions = tuple(code-100 if code < 200 else code-200 for code in symbols)
        targets = tuple(code-100 for code in symbols if code < 200)
        optical = [i for i, r in enumerate(group) if r[1] == "Optic"]
        feedback = len(optical) == 2 and optical[1] > 0 and group[optical[1]-1][2] in {50, 51, 11}
        admitted = len(optical) == 1 or feedback
        if len(positions) != 12 or len(set(positions)) != 12 or len(targets) > 1:
            raise ValueError("original Huebner flash does not contain its exact twelve physical positions")
        flashes.append(PhysicalFlash(group[0][0], group[0][2], group[0][3],
            group[optical[0]][3] if admitted else None,
            "second_optic_is_feedback" if feedback else "one_optical_onset" if admitted else "unresolved_original_optical_onset",
            positions, targets))
    if len(flashes) != SELECTIONS_PER_RUN*FRAMES_PER_SELECTION:
        raise ValueError("original Huebner copy-spelling run must contain seven 68-frame selections")
    truths = []
    for ordinal in range(SELECTIONS_PER_RUN):
        group = flashes[ordinal*FRAMES_PER_SELECTION:(ordinal+1)*FRAMES_PER_SELECTION]
        target = {p for flash in group for p in flash.target_positions}
        if len(target) != 1 or next(iter(target)) not in SELECTABLE_POSITIONS:
            raise ValueError("original target code must identify one selectable position within a character")
        truth = next(iter(target))
        for flash in group:
            if bool(flash.target_positions) != (truth in flash.physical_positions):
                raise ValueError("original target bit disagrees with physical set membership")
        membership = np.asarray([[p in flash.physical_positions for p in SELECTABLE_POSITIONS] for flash in group])
        if not np.all(membership.sum(axis=0) == 16) or len({tuple(c) for c in membership.T}) != len(SELECTABLE_POSITIONS):
            raise ValueError("full Huebner programme must retain equal sixteen exposures and distinguish all 32 choices")
        for kind, width, count, exposure in ((21, 8, 4, 3), (22, 18, 2, 2)):
            typed = [flash for flash in group if flash.sequence_type == kind]
            if len(typed) != width*count:
                raise ValueError("actual sequence-type ledger differs from original Huebner programme")
            for offset in range(0, len(typed), width):
                m = [[p in flash.physical_positions for p in SELECTABLE_POSITIONS] for flash in typed[offset:offset+width]]
                if not np.all(np.sum(m, axis=0) == exposure):
                    raise ValueError("ordered type-specific original sequence has incorrect choice exposure")
        truths.append(truth)
    return HuebnerMarkerRecord(tuple(flashes), tuple(truths), _sha(payload), freeze_metadata(source_authority))


def join_nemar_events(record, payload, *, sample_rate_hz):
    """Infer this run's clock offset by the complete actual optical event join."""
    if sample_rate_hz != SOURCE_SAMPLE_RATE_HZ:
        raise ValueError("Huebner original/native clock requires actual1000Hz")
    table = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t"))
    valid = [f for f in record.flashes if f.optical_sample is not None]
    if len(valid) != len(table):
        raise ValueError("NEMAR retained event count differs from original admitted optical markers")
    offsets = []
    for flash, event in zip(valid, table, strict=True):
        if event["trial_type"] not in {"Target", "NonTarget"}:
            raise ValueError("NEMAR event has an undeclared original label category")
        sample = int(event["sample"])
        if (abs(float(event["onset"])-sample/sample_rate_hz) > .5/sample_rate_hz
                or bool(flash.target_positions) != (event["trial_type"] == "Target")):
            raise ValueError("NEMAR onset/target bit differs from original physical marker join")
        offsets.append(flash.optical_sample-sample)
    if len(set(offsets)) != 1:
        raise ValueError("complete original-to-NEMAR clock is not one exact run-specific offset")
    offset = offsets[0]
    return np.asarray([f.optical_sample-offset if f.optical_sample is not None else -1 for f in record.flashes],
                      dtype=np.int64), freeze_metadata({"program": "all_admitted_original_optical_events_to_actual_NEMAR_samples/1",
        "marker_sha256": record.marker_sha256, "events_sha256": _sha(payload), "clock_offset_samples": offset,
        "valid_optical_rows": len(valid), "scheduled_physical_rows": len(record.flashes),
        "unresolved_optical_ordinals": [i for i, f in enumerate(record.flashes) if f.optical_sample is None],
        "binary_label_join_exact": True})


def membership_decision(record, selection, *, event_prefix, evidence_indices=None):
    start = selection*FRAMES_PER_SELECTION
    flashes = record.flashes[start:start+FRAMES_PER_SELECTION]
    if len(flashes) != FRAMES_PER_SELECTION:
        raise ValueError("requested selection is outside original marker programme")
    events = tuple(f"{event_prefix}:{f.marker_id}" for f in flashes)
    return MembershipDecision(tuple(f"position:{p}" for p in SELECTABLE_POSITIONS),
        np.asarray([[p in f.physical_positions for p in SELECTABLE_POSITIONS] for f in flashes], dtype=bool),
        events, tuple(f"physical_set:{semantic_sha256(f.public_record())}" for f in flashes),
        np.arange(FRAMES_PER_SELECTION, dtype=np.int64), np.zeros(FRAMES_PER_SELECTION, dtype=np.int64),
        np.arange(FRAMES_PER_SELECTION, dtype=np.int64) if evidence_indices is None else evidence_indices,
        1, "set_membership", True)


@dataclass(frozen=True)
class HuebnerNativeSource:
    """One verified NEMAR acquisition and its original physical marker law."""

    record_id: str
    participant_key: str
    subject_id: str
    session_id: str
    run_id: str
    bdf_path: Path | None
    bdf_sha256: str | None
    channels: tuple[str, ...]
    marker: HuebnerMarkerRecord
    native_onsets: np.ndarray
    readout: dict
    _admitted_file_identity: tuple[int, ...] | None = field(default=None, repr=False, compare=False)

    @property
    def source_sha256(self):
        return self.bdf_sha256 if self.bdf_sha256 is not None else self.marker.marker_sha256

    @property
    def selection_ids(self):
        return tuple(f"{self.record_id}:character:{i}" for i in range(SELECTIONS_PER_RUN))

    def read_native(self, start, stop):
        """Read the actual requested original range without preloading the file."""
        import mne

        if self.bdf_path is None:
            raise ValueError("original acquisition payload is unavailable")
        if _file_identity(self.bdf_path.stat()) != self._admitted_file_identity:
            raise ValueError("original BDF file identity changed after byte admission")
        raw = mne.io.read_raw_bdf(self.bdf_path, preload=False, verbose=False)
        try:
            if tuple(raw.ch_names) != self.channels or raw.info["sfreq"] != SOURCE_SAMPLE_RATE_HZ:
                raise ValueError("actual BDF layout/native sample clock changed after source admission")
            if not 0 <= start < stop <= raw.n_times:
                raise ValueError("native read range exceeds the original acquisition file")
            values = raw.get_data(start=start, stop=stop).T
        finally:
            raw.close()
        if _file_identity(self.bdf_path.stat()) != self._admitted_file_identity:
            raise ValueError("original BDF file identity changed during native range read")
        return values


def load_huebner_native_source(declaration):
    """Join the original byte-range marker with the independently verified BDF."""
    from data.identity import origin_subject_key
    from transfer.noise_model import require_sha256

    subject = str(declaration["subject"])
    block_number, run_number = declaration["block"], declaration["run"]
    if (re.fullmatch(r"[1-9]|1[0-3]", subject) is None
            or type(block_number) is not int or not 0 <= block_number < 3
            or type(run_number) is not int or not 0 <= run_number < 9):
        raise ValueError("original Huebner source requires its exact thirteen-subject/session/run identities")
    block, run = str(block_number), str(run_number)
    record_id = f"{DATASET_ID}:{subject}:session:{block}:run:{run}"
    authority = declaration["marker"]
    if "archive_path" in authority:
        marker_bytes = read_local_marker_member(authority["archive_path"], authority["member"])
        original_member = authority["member"]["name"]
    else:
        marker_bytes = Path(authority["path"]).read_bytes()
        if _sha(marker_bytes) != authority["sha256"]:
            raise ValueError("original external marker bytes differ from declared source admission")
        original_member = authority["original_member"]
        member_receipt = authority["original_member_receipt"]
        receipt_bytes = Path(member_receipt["path"]).read_bytes()
        if _sha(receipt_bytes) != member_receipt["sha256"]:
            raise ValueError("original external marker member receipt differs from its frozen authority")
        receipt = json.loads(receipt_bytes)
        if (receipt["member"] != original_member or receipt["sha256"] != _sha(marker_bytes)
                or receipt["subject"] != int(subject) or receipt["block"] != block_number+1
                or receipt["run"] != run_number+1 or receipt["kind"] != "vmrk" or not receipt["crc_pass"]
                or receipt["raw_bytes"] != len(marker_bytes)
                or zlib.crc32(marker_bytes) & 0xffffffff != int(receipt["crc32"], 16)):
            raise ValueError("original external marker is not the admitted physical member/run")
    basename = f"matrixSpeller_Block{block_number+1}_Run{run_number+1}"
    member_match = re.fullmatch(r"subject(\d{2})/(?:block_without_optical/)?(matrixSpeller_Block\d+_Run\d+)\.vmrk", original_member)
    if (member_match is None or int(member_match[1]) != int(subject) or member_match[2] != basename
            or re.search(r"^DataFile="+re.escape(basename)+r"\.eeg\s*$", marker_bytes.decode("utf-8-sig"), re.MULTILINE) is None):
        raise ValueError("original marker subject/session/run/member identity differs from the declared acquisition")
    if "source_header" in authority:
        header = authority["source_header"]
        content = Path(header["path"]).read_bytes()
        if (_sha(content) != header["sha256"]
                or re.search(r"^DataFile="+re.escape(basename)+r"\.eeg\s*$", content.decode("utf-8-sig"), re.MULTILINE) is None
                or re.search(r"^MarkerFile="+re.escape(basename)+r"\.vmrk\s*$", content.decode("utf-8-sig"), re.MULTILINE) is None):
            raise ValueError("original BrainVision header does not link this exact physical marker/run")
    marker = decode_original_markers(marker_bytes, source_authority=authority)
    bdf = declaration.get("bdf")
    if bdf is None:
        return HuebnerNativeSource(record_id, origin_subject_key(DATASET_ID, subject), subject, block, run,
            None, None, (), marker, np.full(len(marker.flashes), -1, dtype=np.int64),
            freeze_metadata({"status": "original_EEG_unavailable", "native_sample_count": None,
                             "parent_truth_authority": "original_marker_target_position"}))
    import mne

    path = Path(bdf["path"])
    nemar_basename = f"sub-{subject}_ses-{block}_task-p300_run-{run}"
    if path.name != nemar_basename+"_eeg.bdf":
        raise ValueError("original NEMAR BDF subject/session/run differs from the physical marker acquisition")
    companion_paths = {}
    for kind in ("channels", "events", "eeg"):
        companion_path = Path(declaration[kind]["path"])
        suffix = "_eeg.json" if kind == "eeg" else f"_{kind}.tsv"
        if companion_path.name != nemar_basename+suffix or companion_path.parent.resolve() != path.parent.resolve():
            raise ValueError("NEMAR sidecar identity differs from its exact physical EEG acquisition")
        companion_paths[kind] = companion_path
    require_sha256(bdf["sha256"], "original NEMAR EEG")
    with path.open("rb") as stream:
        admitted_file_identity = _file_identity(os.fstat(stream.fileno()))
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if _file_identity(os.fstat(stream.fileno())) != admitted_file_identity:
            raise ValueError("original BDF file changed during one byte-hash admission")
    if _file_identity(path.stat()) != admitted_file_identity:
        raise ValueError("original BDF path no longer resolves to its admitted file identity")
    if actual != bdf["sha256"]:
        raise ValueError("NEMAR EEG differs from actual frozen acquisition bytes")
    companion = {}
    for kind in ("channels", "events", "eeg"):
        row = declaration[kind]
        payload = companion_paths[kind].read_bytes()
        if _sha(payload) != row["sha256"]:
            raise ValueError("NEMAR companion differs from its actual declared bytes")
        companion[kind] = payload
    channels = list(csv.DictReader(io.StringIO(companion["channels"].decode("utf-8-sig")), delimiter="\t"))
    eeg = json.loads(companion["eeg"])
    if (len(channels) != 31 or any(c["type"] != "EEG" or c["status"] != "good" or c["units"] != "V" for c in channels)
            or eeg["SamplingFrequency"] != SOURCE_SAMPLE_RATE_HZ or eeg["EEGReference"] != SOURCE_REFERENCE):
        raise ValueError("actual Huebner EEG geometry/reference/type differs from the admitted NEMAR law")
    raw = mne.io.read_raw_bdf(path, preload=False, verbose=False)
    try:
        names, samples = tuple(raw.ch_names), int(raw.n_times)
        if names != tuple(c["name"] for c in channels) or raw.info["sfreq"] != SOURCE_SAMPLE_RATE_HZ:
            raise ValueError("raw BDF disagrees with actual channel/clock sidecars")
    finally:
        raw.close()
    if _file_identity(path.stat()) != admitted_file_identity:
        raise ValueError("original BDF file identity changed during header admission")
    onsets, join = join_nemar_events(marker, companion["events"], sample_rate_hz=SOURCE_SAMPLE_RATE_HZ)
    if np.any(onsets >= samples):
        raise ValueError("actual joined marker lies beyond original EEG observations")
    return HuebnerNativeSource(record_id, origin_subject_key(DATASET_ID, subject), subject, block, run,
        path, actual, names, marker, onsets, freeze_metadata({"status": "original_clock_joined",
        "native_sample_count": samples, "clock_join": join, "original_inputs": declaration,
        "native_clock": "record_local_NEMAR_sample", "source_reference": SOURCE_REFERENCE,
        "native_signal_unit": "V", "acquisition_boundary": "independent_original_BDF_record"}),
        _admitted_file_identity=admitted_file_identity)


def huebner_dataset_adapter(declarations):
    from data.candidate_input import (
        CandidateDatasetAdapter,
        OriginalCandidateRecording,
        OriginalCandidateSelection,
    )

    declarations = tuple(declarations)
    order = [(str(d["subject"]), int(d["block"]), int(d["run"])) for d in declarations]
    if order != sorted(order) or len(set(order)) != len(order) or len({r[0] for r in order}) != 1:
        raise ValueError("original Huebner participant records must have unique ordered session/run authority")
    sources = tuple(load_huebner_native_source(d) for d in declarations)
    available = next((s for s in sources if s.bdf_path is not None), None)
    if available is None:
        raise ValueError("original participant roster has no available measured EEG acquisition")
    recordings = []
    for source in sources:
        selections = tuple(OriginalCandidateSelection(source.selection_ids[i],
            tuple(range(i*FRAMES_PER_SELECTION, (i+1)*FRAMES_PER_SELECTION)),
            membership_decision(source.marker, i, event_prefix=source.record_id),
            f"position:{source.marker.target_positions[i]}") for i in range(SELECTIONS_PER_RUN))
        recordings.append(OriginalCandidateRecording(source.record_id, source.source_sha256,
            source.bdf_path.name if source.bdf_path is not None else source.record_id,
            source.participant_key, source.subject_id, source.session_id, source.run_id, SOURCE_SAMPLE_RATE_HZ,
            available.channels if source.bdf_path is None else source.channels, SOURCE_REFERENCE,
            source.readout["native_sample_count"], source.native_onsets,
            np.asarray([bool(f.target_positions) for f in source.marker.flashes], dtype=bool),
            selections, source.readout, source.read_native,
            np.asarray([f.software_sample-(source.readout.get("clock_join", {}).get("clock_offset_samples", 0))
                        for f in source.marker.flashes], dtype=np.int64)))
    return CandidateDatasetAdapter(DATASET_ID, available.participant_key, available.subject_id,
        "original_vmrk_plus_NEMAR_BDF", tuple(recordings), declarations, CANDIDATE_TASK.record(),
        "one_original_68_frame_selection_R1", _sha(Path(__file__).read_bytes()))
