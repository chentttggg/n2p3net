"""One decoded prefix input capsule, shared through kernel-sealed numeric storage.

Preparation executes the original MAT compiler. The external request and actual
CloudExecutor retirement admit its capsule; neither a cache hash nor a read-only
mapping grants input authority. Consumers restore the original capabilities and
keep their own implementation identity separate.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import mmap
import os
import sys
import threading
import weakref
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np
import psutil

from research.contracts import semantic_sha256

_CAPSULE_SCHEMA = "n2p3_prefix_replay_inputs/1"
_REQUEST_SCHEMA = "n2p3_prefix_replay_input_request/1"
_SEALED_MAPPINGS = weakref.WeakKeyDictionary()
_MAPPING_LOCK = threading.RLock()
_CHUNK_BYTES = 8 * 2**20


def _sha256_buffer(buffer):
    digest = hashlib.sha256()
    view = memoryview(buffer)
    if not view.nbytes:
        return digest.hexdigest()
    view = view.cast("B")
    for start in range(0, len(view), _CHUNK_BYTES):
        digest.update(view[start:start + _CHUNK_BYTES])
    return digest.hexdigest()


def _immutable_backing(value):
    """Only terminal bytes and personally verified sealed mappings are immutable."""
    while isinstance(value, (np.ndarray, memoryview)):
        value = value.base if isinstance(value, np.ndarray) else value.obj
    if isinstance(value, bytes):
        return True
    if isinstance(value, mmap.mmap):
        with _MAPPING_LOCK:
            return value in _SEALED_MAPPINGS
    return False


def immutable_input_array(values, dtype):
    """Share immutable backing while returning a new, independent ndarray header."""
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("immutable numeric inputs cannot contain Python object pointers")
    if _immutable_backing(array):
        return np.ndarray(array.shape, dtype=array.dtype, buffer=array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _artifact(path):
    payload = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}


def _read_artifact(receipt):
    if set(receipt) != {"path", "sha256", "bytes"}:
        raise ValueError("prepared input artifact requires exact path/hash/byte receipt")
    payload = Path(receipt["path"]).read_bytes()
    if len(payload) != receipt["bytes"] or hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise ValueError("prepared input artifact bytes differ from the actual preparation receipt")
    return payload


def _validate_request(request):
    from transfer.ensemble_views import SelectionEpochProfile
    from transfer.noise_model import require_sha256

    expected = {"schema", "preparation_task_id", "original_source_sha256", "producer_source_sha256",
                "participant_key", "source_relative_path", "requested_Ks", "actual_Ks", "profiles",
                "authorization_preprocessing", "observation_response_seconds", "pre_roll_seconds"}
    if set(request) != expected or request["schema"] != _REQUEST_SCHEMA:
        raise ValueError("prepared prefix request must bind the original compiler and physical programme")
    for name in ("original_source_sha256", "producer_source_sha256"):
        require_sha256(request[name], name)
    for name in ("preparation_task_id", "participant_key", "source_relative_path"):
        if not isinstance(request[name], str) or not request[name] or request[name] != request[name].strip():
            raise ValueError("prepared prefix request requires exact task and physical source identities")
    for name in ("requested_Ks", "actual_Ks"):
        values = request[name]
        if (not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values)
                or any(type(value) is not int or value < 0 for value in values)):
            raise ValueError("prepared prefix request requires distinct nonnegative actual/requested K")
    profiles = tuple(SelectionEpochProfile.from_record(row) for row in request["profiles"])
    if len({profile.digest for profile in profiles}) != len(profiles):
        raise ValueError("prepared prefix profiles must be distinct actual programmes")
    return profiles


class _ArrayWriter:
    """Uncompressed arrays; identical arrays and verified raw spans share storage."""

    def __init__(self, stream):
        self.stream, self.arrays, self.identical = stream, [], {}
        self.raw_master = None
        self.raw_master_entry = None

    def array(self, values, *, raw_scope=None):
        value = np.asarray(values)
        if value.dtype.hasobject:
            value = value.astype(str)
        value = np.ascontiguousarray(value)
        if value.dtype.kind not in "biufUS":
            raise TypeError("prefix capsules support plain numeric and Unicode input arrays")
        if raw_scope is not None and self.raw_master is not None:
            start, stop = raw_scope
            expected = self.raw_master[start:stop]
            if value.shape != expected.shape or value.dtype != expected.dtype or not np.array_equal(value, expected):
                raise ValueError("prepared raw slice differs from the original native master observations")
            master = self.arrays[self.raw_master_entry]
            return self._entry(value, master["offset"] + start * value.shape[1] * value.dtype.itemsize)
        key = (value.dtype.str, value.shape, _sha256_buffer(value))
        if key in self.identical:
            return {"array": self.identical[key]}
        # Array offsets use item-size alignment, including UTF-32 strings.
        padding = (-self.stream.tell()) % max(8, value.dtype.alignment)
        self.stream.write(b"\0" * padding)
        offset = self.stream.tell()
        if value.nbytes:
            self.stream.write(memoryview(value).cast("B"))
        entry = self._entry(value, offset)
        self.identical[key] = entry["array"]
        return entry

    def _entry(self, value, offset):
        record = {"dtype": value.dtype.str, "shape": list(value.shape), "offset": offset,
                  "nbytes": value.nbytes}
        try:
            index = self.arrays.index(record)
        except ValueError:
            index = len(self.arrays)
            self.arrays.append(record)
        return {"array": index}


def _pack_dataset(data, writer):
    from data.epochs import EpochDataset, materialize_dataset_identity, materialize_dataset_lineage

    record = {}
    for item in fields(EpochDataset):
        if not item.init:
            continue
        value = getattr(data, item.name)
        if isinstance(value, np.ndarray):
            record[item.name] = writer.array(value)
        elif item.name == "metadata":
            record[item.name] = {"columns": list(value.columns), "index": writer.array(value.index.to_numpy()),
                                 "values": [writer.array(value[name].to_numpy()) for name in value.columns]}
        elif item.name == "preprocessing":
            record[item.name] = value.record()
        elif item.name == "event_timeline":
            record[item.name] = {field.name: writer.array(getattr(value, field.name))
                                if isinstance(getattr(value, field.name), np.ndarray) else getattr(value, field.name)
                                for field in fields(value)}
        elif item.name == "identity_table":
            record[item.name] = materialize_dataset_identity(data).payload()
        elif item.name == "lineage":
            record[item.name] = materialize_dataset_lineage(data).payload()
        elif item.name == "qc_features":
            record[item.name] = None if value is None else {
                field.name: writer.array(getattr(value, field.name))
                if isinstance(getattr(value, field.name), np.ndarray) else getattr(value, field.name)
                for field in fields(value)}
        elif item.name == "verified_cache_attestation":
            if value is not None:
                raise ValueError("native MAT prepared inputs cannot impersonate a verified epoch cache")
            record[item.name] = None
        else:
            record[item.name] = value
    return record


def _pack_view(view, writer, *, calibration=False):
    return {"record": view.record(), "samples": None if view.samples is None else
            writer.array(view.samples, raw_scope=view.raw_scope if calibration else None)}


def export_prepared_views(views, profiles, directory, request):
    """Export actual old-compiler prefixes, including all recursively earlier K.

    Run this function in the frozen producer process. The request pins its real
    ``ensemble_views.py`` bytes, so a newly compiled current result cannot be
    relabelled with an old implementation hash.
    """
    from transfer.ensemble_views import calibration_prefix_identity

    requested_profiles = _validate_request(request)
    compiler = views.prefix_compiler
    if compiler is None:
        raise ValueError("prepared prefix inputs require the original verified MAT compiler")
    producer_module = sys.modules[type(compiler).__module__]
    actual_sha = hashlib.sha256(Path(inspect.getfile(type(compiler))).read_bytes()).hexdigest()
    if (actual_sha != request["producer_source_sha256"]
            or getattr(producer_module, "_SELECTION_EXECUTION_SOURCE_SHA256", None) != actual_sha):
        raise ValueError("prepared prefix producer differs from the original frozen implementation")
    profile_records = [profile.record() for profile in profiles]
    if semantic_sha256(profile_records) != semantic_sha256([profile.record() for profile in requested_profiles]):
        raise ValueError("prepared prefix profiles differ from the external preparation request")
    target = compiler._arguments["target"]
    width = max([round(request["observation_response_seconds"] * target["sample_rate_hz"]),
                 *(profile._source_stop for profile in profiles)])
    preroll = max([round(request["pre_roll_seconds"] * target["sample_rate_hz"]),
                   *(-profile._source_start for profile in profiles)])
    if (target["source_sha256"] != request["original_source_sha256"]
            or target["participant_key"] != request["participant_key"]
            or target["source_relative_path"] != request["source_relative_path"]
            or views.ledger["requested_Ks"] != list(request["requested_Ks"])
            or compiler._arguments["authorization_preprocessing"].record() != request["authorization_preprocessing"]
            or compiler._arguments["width"] != width or compiler._arguments["preroll"] != preroll
            or set(compiler._arguments["profiles"]) != {profile.digest for profile in profiles}):
        raise ValueError("original MAT compiler arguments differ from external physical/profile authority")
    prefixes = dict(views.calibration)
    pending = list(set(request["actual_Ks"]) | set(prefixes))
    while pending:
        budget = pending.pop()
        if budget in prefixes and budget not in compiler._prefixes:
            raise ValueError("prepared calibration shell was not compiled by the original MAT owner")
        prefix = compiler.compile(budget)
        compiler.assert_original_prefix(prefix)
        prefixes[budget] = prefix
        for group in prefix.complete_groups:
            earlier = compiler.before_group(prefix, group)
            if earlier.budget not in prefixes and earlier.budget not in pending:
                pending.append(earlier.budget)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    blob_path = directory / "arrays.bin"
    with blob_path.open("xb") as stream:
        writer = _ArrayWriter(stream)
        largest = prefixes[max(prefixes)]
        if largest.raw is not None:
            master = writer.array(largest.raw.samples)
            writer.raw_master, writer.raw_master_entry = largest.raw.samples, master["array"]
        records = {}
        for budget, prefix in sorted(prefixes.items()):
            raw = prefix.raw
            capabilities = {profile.digest: prefix.trial_target(profile).provenance["selection_epoch_capability"]
                            for profile in profiles}
            if any(value["producer_source_sha256"] != actual_sha for value in capabilities.values()):
                raise ValueError("actual original prefix capability differs from the executed compiler")
            records[str(budget)] = {"budget": budget, "target_contract": prefix.target_contract,
                "label_use": prefix.label_use, "identity": calibration_prefix_identity(prefix),
                "capabilities": capabilities, "raw": None if raw is None else {
                    "record_id": raw.record_id, "samples": writer.array(raw.samples, raw_scope=(0, len(raw.samples))),
                    "onsets": writer.array(raw.onsets), "target_mask": writer.array(raw.target_mask),
                    "right_censored": raw.right_censored},
                "trial_datasets": {key: _pack_dataset(data, writer) for key, data in prefix.trial_datasets.items()},
                "complete_groups": [{"view": _pack_view(group.view, writer, calibration=True),
                                     "labels": writer.array(group.labels), "target_candidate": group.target_candidate}
                                    for group in prefix.complete_groups]}
        decisions = [_pack_view(view, writer) for view in views.decisions]
    blob_digest = hashlib.sha256()
    with blob_path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            blob_digest.update(block)
    document = {"schema": _CAPSULE_SCHEMA, "request": request, "request_sha256": semantic_sha256(request),
        "producer_source_sha256": actual_sha, "blob": {"name": blob_path.name, "sha256": blob_digest.hexdigest(),
                                                       "bytes": blob_path.stat().st_size},
        "arrays": writer.arrays, "prefixes": records,
        "calibration_Ks": list(views.calibration), "decisions": decisions, "ledger": views.ledger}
    capsule_path = directory / "capsule.json"
    capsule_path.write_bytes(_json_bytes(document))
    # Future truth belongs only to the parent scoring owner, never reader capsules.
    truth_path = directory / "scoring_truth.json"
    truth_path.write_bytes(_json_bytes({"schema": "n2p3_prefix_scoring_truth/1",
        "request_sha256": document["request_sha256"], "capsule_sha256": _artifact(capsule_path)["sha256"],
        "source_sha256": request["original_source_sha256"], "scoring_truth": views.scoring_truth,
        "future_support": [view.record() for view in views.decisions]}))
    return _artifact(capsule_path)


def _validate_capsule(payload, *, request_sha):
    document = json.loads(payload)
    if (document["schema"] != _CAPSULE_SCHEMA or semantic_sha256(document["request"]) != request_sha
            or document["request_sha256"] != request_sha):
        raise ValueError("prepared input capsule differs from the externally admitted request")
    profiles = _validate_request(document["request"])
    if document["producer_source_sha256"] != document["request"]["producer_source_sha256"]:
        raise ValueError("prepared input capsule changed original producer identity")
    if not set(document["request"]["actual_Ks"]).issubset(map(int, document["prefixes"])):
        raise ValueError("prepared input capsule omitted requested actual K")
    if set(document["calibration_Ks"]) != set(document["ledger"]["requested_to_actual_K"].values()):
        raise ValueError("prepared input capsule changed original requested-to-actual K")
    size = document["blob"]["bytes"]
    if type(size) is not int or size < 0 or document["blob"]["name"] != "arrays.bin":
        raise ValueError("prepared input capsule requires a local bounded numeric blob")
    for layout in document["arrays"]:
        dtype = np.dtype(layout["dtype"])
        shape = layout["shape"]
        offset, nbytes = layout["offset"], layout["nbytes"]
        if (dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufUS"
                or any(type(value) is not int or value < 0 for value in shape)
                or type(offset) is not int or offset < 0 or type(nbytes) is not int
                or nbytes != int(np.prod(shape, dtype=object)) * dtype.itemsize
                or offset + nbytes > size or offset % max(1, dtype.alignment)):
            raise ValueError("prepared input array shape/dtype/byte span is outside its immutable blob")
    for key, record in document["prefixes"].items():
        if record["budget"] != int(key) or set(record["capabilities"]) != {profile.digest for profile in profiles}:
            raise ValueError("prepared input prefix lacks complete actual profile capabilities")
        for profile in profiles:
            capability = record["capabilities"][profile.digest]
            expected = profile.execution_capability()
            expected["producer_source_sha256"] = document["producer_source_sha256"]
            if semantic_sha256(capability) != semantic_sha256(expected):
                raise ValueError("prepared input capability differs from the original executed profile")
    return document


def _validate_preparation(descriptor, request_sha, outcome, result_key):
    from research.cloud_execution import CloudResult

    if not isinstance(outcome, CloudResult):
        raise TypeError("prepared inputs require an actual CloudExecutor completion outcome")
    returned = outcome.value
    if result_key is not None:
        if (not isinstance(result_key, str) or not result_key or not isinstance(returned, Mapping)
                or result_key not in returned):
            raise ValueError("prepared input result key is absent from the actual producer artifact bundle")
        returned = returned[result_key]
    worker = outcome.worker or {}
    remaining = worker.get("process_tree_retirement", {}).get("active_owned_processes_after_exit")
    if (outcome.status != "completed" or outcome.failure is not None
            or worker.get("alive") is not False or type(worker.get("exit_code")) is not int
            or worker["exit_code"] != 0
            or type(remaining) is not int or remaining != 0
            or returned != descriptor):
        raise ValueError("prepared input producer requires actual successful exit and full tree retirement")
    payload = _read_artifact(descriptor)
    document = _validate_capsule(payload, request_sha=request_sha)
    if outcome.task_id != document["request"]["preparation_task_id"]:
        raise ValueError("prepared input artifact was not returned by its actual requested producer task")
    return payload, document


def _required_seals():
    import fcntl

    return fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


class PrefixReplayInputHolder:
    """Parent-owned sealed backing and resident lease, retained through reader exit."""

    def __init__(self, capsule, capsule_sha, backing, lease, *, request_sha, fd=None, capsule_fd=None):
        self._request_sha = request_sha
        self._capsule = capsule if capsule_fd is None else None
        self._capsule_sha = capsule_sha
        self._backing, self._lease, self._fd, self._capsule_fd = backing, lease, fd, capsule_fd
        self._closed = False

    @classmethod
    def from_prepared(cls, descriptor, *, request_sha, prep_outcome, admission, result_key=None):
        payload, document = _validate_preparation(descriptor, request_sha, prep_outcome, result_key)
        lease = admission.try_resident_input(document["blob"]["bytes"] + len(payload))
        if lease is None:
            return None
        fd = capsule_fd = None
        try:
            digest, total = hashlib.sha256(), 0
            path = Path(descriptor["path"]).parent / document["blob"]["name"]
            if sys.platform == "linux":
                import fcntl

                fd = os.memfd_create("n2p3-prefix-replay", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                        total += len(block)
                        if total > document["blob"]["bytes"]:
                            raise ValueError("prepared input numeric blob grew beyond its admitted byte budget")
                        digest.update(block)
                        view = memoryview(block)
                        while view:
                            view = view[os.write(fd, view):]
                backing = None
            else:
                with path.open("rb") as stream:
                    backing = stream.read(document["blob"]["bytes"])
                    if stream.read(1):
                        raise ValueError("prepared input numeric blob grew beyond its admitted byte budget")
                total, checksum = len(backing), hashlib.sha256(backing).hexdigest()
                digest = None
            checksum = digest.hexdigest() if digest is not None else checksum
            if total != document["blob"]["bytes"] or checksum != document["blob"]["sha256"]:
                raise ValueError("prepared input numeric bytes differ from the admitted producer capsule")
            if fd is not None:
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, _required_seals())
                if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != _required_seals():
                    raise RuntimeError("prepared input backing did not acquire all immutable kernel seals")
                capsule_fd = os.memfd_create("n2p3-prefix-capsule", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                view = memoryview(payload)
                while view:
                    view = view[os.write(capsule_fd, view):]
                fcntl.fcntl(capsule_fd, fcntl.F_ADD_SEALS, _required_seals())
                if fcntl.fcntl(capsule_fd, fcntl.F_GET_SEALS) != _required_seals():
                    raise RuntimeError("prepared input capsule did not acquire all immutable kernel seals")
            return cls(payload, descriptor["sha256"], backing, lease, request_sha=request_sha,
                       fd=fd, capsule_fd=capsule_fd)
        except BaseException:
            if capsule_fd is not None:
                os.close(capsule_fd)
            if fd is not None:
                os.close(fd)
            lease.release_unused()
            raise

    @property
    def descriptor(self):
        if self._closed:
            raise RuntimeError("prepared input holder has retired")
        result = {"schema": "n2p3_prefix_replay_shared_inputs/1",
                  "capsule_sha256": self._capsule_sha, "request_sha256": self._request_sha}
        if self._fd is None:
            result.update(backing="immutable_bytes", payload=self._backing, capsule=self._capsule)
        else:
            result.update(**_sealed_descriptor(self._fd), capsule=_sealed_descriptor(self._capsule_fd))
        return result

    def close(self):
        """Caller invokes this after all borrowed-reader/reducer trees retire."""
        if self._closed:
            return
        if self._fd is not None:
            os.close(self._fd)
            os.close(self._capsule_fd)
        self._backing = self._capsule = None
        self._lease.release_unused()
        self._closed = True


def _sealed_descriptor(fd):
    info = os.fstat(fd)
    return {"backing": "sealed_memfd", "parent_pid": os.getpid(),
            "parent_birth": psutil.Process().create_time(), "fd": fd,
            "device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size, "seals": _required_seals()}


def _open_backing(descriptor, document):
    if descriptor["backing"] == "immutable_bytes":
        value = descriptor["payload"]
        if not isinstance(value, bytes):
            raise TypeError("prepared Windows backing must terminate in immutable bytes")
        if len(value) != document["blob"]["bytes"] or _sha256_buffer(value) != document["blob"]["sha256"]:
            raise ValueError("prepared immutable input bytes differ from the admitted capsule")
        return value
    return _open_sealed_backing(descriptor, expected_bytes=document["blob"]["bytes"],
                                expected_sha=document["blob"]["sha256"])


def _open_sealed_backing(descriptor, *, expected_bytes, expected_sha):
    if descriptor["backing"] != "sealed_memfd" or sys.platform != "linux":
        raise ValueError("prepared input backing is unavailable on this execution host")
    import fcntl

    pid, birth = descriptor["parent_pid"], descriptor["parent_birth"]
    if type(pid) is not int or type(descriptor["fd"]) is not int or descriptor["fd"] < 0:
        raise ValueError("sealed input descriptor requires exact parent and file descriptor identities")
    if psutil.Process(pid).create_time() != birth:
        raise ValueError("sealed input parent identity has changed")
    fd = os.open(f"/proc/{pid}/fd/{descriptor['fd']}", os.O_RDONLY | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
        if (psutil.Process(pid).create_time() != birth
                or info.st_dev != descriptor["device"] or info.st_ino != descriptor["inode"]
                or info.st_size != descriptor["bytes"] or info.st_size != expected_bytes
                or seals != _required_seals() or descriptor["seals"] != seals):
            raise ValueError("sealed input parent/fd/inode/byte extent or immutable seals changed")
        if info.st_size == 0:
            if _sha256_buffer(b"") != expected_sha:
                raise ValueError("empty sealed input differs from its admitted digest")
            return b""
        value = mmap.mmap(fd, info.st_size, access=mmap.ACCESS_READ)
        if _sha256_buffer(value) != expected_sha:
            value.close()
            raise ValueError("sealed numeric backing differs from the admitted original input digest")
        with _MAPPING_LOCK:
            _SEALED_MAPPINGS[value] = (info.st_dev, info.st_ino, seals)
        return value
    finally:
        os.close(fd)


class _ArrayReader:
    def __init__(self, buffer, layouts):
        self.buffer, self.layouts = buffer, layouts

    def array(self, reference):
        if set(reference) != {"array"} or type(reference["array"]) is not int:
            raise ValueError("prepared input arrays require one exact layout reference")
        index = reference["array"]
        if not 0 <= index < len(self.layouts):
            raise ValueError("prepared input array reference is outside admitted layouts")
        layout = self.layouts[index]
        return np.ndarray(tuple(layout["shape"]), dtype=np.dtype(layout["dtype"]),
                          buffer=self.buffer, offset=layout["offset"])


def _unpack_dataset(record, reader):
    import pandas as pd

    from data.epochs import EpochDataset, parse_preprocessing_spec
    from data.events import ScheduledEventTimeline
    from data.identity import DatasetIdentityTable
    from data.lineage import DataLineage
    from data.qc_features import EpochQCFeatures

    values = dict(record)
    for name, value in tuple(values.items()):
        if isinstance(value, dict) and set(value) == {"array"}:
            values[name] = reader.array(value)
    table = values["metadata"]
    values["metadata"] = pd.DataFrame({name: reader.array(value)
        for name, value in zip(table["columns"], table["values"], strict=True)}, index=reader.array(table["index"]))
    values["preprocessing"] = parse_preprocessing_spec(values["preprocessing"])
    values["channel_names"] = tuple(values["channel_names"])
    values["event_timeline"] = ScheduledEventTimeline(**{
        name: reader.array(value) if isinstance(value, dict) and set(value) == {"array"} else value
        for name, value in values["event_timeline"].items()})
    values["identity_table"] = DatasetIdentityTable.from_payload(values["identity_table"])
    values["lineage"] = DataLineage.from_payload(values["lineage"])
    if values["qc_features"] is not None:
        values["qc_features"] = EpochQCFeatures(**{
            name: reader.array(value) if isinstance(value, dict) and set(value) == {"array"} else value
            for name, value in values["qc_features"].items()})
    return EpochDataset(**values)


def _unpack_view(record, reader):
    from transfer.ensemble_views import PredictionDecisionView

    return PredictionDecisionView.from_record(record["record"], samples=None if record["samples"] is None else
                                               reader.array(record["samples"]))


def _restore_views(document, reader):
    from transfer.continuous_decision import CalibrationRecord
    from transfer.ensemble_views import (
        CalibrationPrefix,
        CalibrationSelection,
        IndependentSubjectViews,
        _PreparedCalibrationPrefixCompiler,
        calibration_prefix_identity,
    )
    from transfer.noise_model import freeze_metadata

    prefixes = {}
    for key, record in document["prefixes"].items():
        raw = record["raw"]
        if raw is not None:
            raw = CalibrationRecord(raw["record_id"], reader.array(raw["samples"]), reader.array(raw["onsets"]),
                                    reader.array(raw["target_mask"]), right_censored=raw["right_censored"])
        prefix = CalibrationPrefix(record["budget"], raw,
            {name: _unpack_dataset(data, reader) for name, data in record["trial_datasets"].items()},
            tuple(CalibrationSelection(_unpack_view(group["view"], reader), reader.array(group["labels"]),
                                       group["target_candidate"]) for group in record["complete_groups"]),
            record["target_contract"], record["label_use"])
        object.__setattr__(prefix, "_prepared_capabilities", freeze_metadata(record["capabilities"]))
        if calibration_prefix_identity(prefix) != record["identity"]:
            raise ValueError("restored prefix numeric input/authority differs from the actual original compilation")
        prefixes[int(key)] = prefix

    compiler = _PreparedCalibrationPrefixCompiler(prefixes,
        producer_source_sha256=document["producer_source_sha256"],
        consumer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Re-run the unchanged original structural authority, never synthesize missing K.
    for prefix in prefixes.values():
        for group in prefix.complete_groups:
            compiler.before_group(prefix, group)
    return IndependentSubjectViews({budget: prefixes[budget] for budget in document["calibration_Ks"]},
        tuple(_unpack_view(row, reader) for row in document["decisions"]), {}, document["ledger"], compiler)


class BorrowedPrefixInputs:
    """Reader lifetime; arrays keep the immutable map alive if a caller retains them."""

    def __init__(self, views, backing, receipt):
        self.views, self._backing, self.receipt = views, backing, receipt

    def close(self):
        self.views = None
        self._backing = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_prepared_inputs(descriptor, *, expected_request_sha, expected_capsule_sha):
    """Worker personally verifies immutable storage and restores original inputs."""
    payload = descriptor["capsule"]
    if isinstance(payload, dict):
        mapping = _open_sealed_backing(payload, expected_bytes=payload["bytes"], expected_sha=expected_capsule_sha)
        try:
            payload = bytes(mapping)
        finally:
            if isinstance(mapping, mmap.mmap):
                mapping.close()
    if (descriptor["schema"] != "n2p3_prefix_replay_shared_inputs/1" or not isinstance(payload, bytes)
            or hashlib.sha256(payload).hexdigest() != expected_capsule_sha
            or descriptor["capsule_sha256"] != expected_capsule_sha
            or descriptor["request_sha256"] != expected_request_sha):
        raise ValueError("shared input descriptor differs from external request/capsule admission")
    document = _validate_capsule(payload, request_sha=expected_request_sha)
    backing = _open_backing(descriptor, document)
    views = _restore_views(document, _ArrayReader(backing, document["arrays"]))
    receipt = {"schema": "n2p3_prefix_replay_input_consumer/1", "request_sha256": expected_request_sha,
        "capsule_sha256": expected_capsule_sha, "producer_source_sha256": document["producer_source_sha256"],
        "consumer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "consumer_selection_source_sha256": hashlib.sha256(Path(inspect.getfile(type(views.calibration[
            next(iter(views.calibration))]))).read_bytes()).hexdigest(),
        "numeric_bytes": document["blob"]["bytes"], "capsule_bytes": len(payload),
        "resident_bytes": document["blob"]["bytes"] + len(payload),
        "prepared_actual_Ks": sorted(map(int, document["prefixes"])),
        "backing": descriptor["backing"], "future_truth_present": False}
    return BorrowedPrefixInputs(views, backing, receipt)


def read_scoring_truth(descriptor, *, capsule_sha, request_sha, future_views):
    """Parent-only scoring truth, bound to the same original future observations."""
    document = json.loads(_read_artifact(descriptor))
    if (document["schema"] != "n2p3_prefix_scoring_truth/1" or document["capsule_sha256"] != capsule_sha
            or document["request_sha256"] != request_sha
            or semantic_sha256(document["future_support"]) != semantic_sha256([view.record() for view in future_views])
            or any(view.source_sha256 != document["source_sha256"] for view in future_views)
            or set(document["scoring_truth"]) != {view.decision_id for view in future_views}
            or any(value not in view.decision.candidate_vocabulary
                   for view in future_views for key, value in document["scoring_truth"].items() if key == view.decision_id)):
        raise ValueError("parent scoring truth differs from the original admitted future scope")
    return document["scoring_truth"]
