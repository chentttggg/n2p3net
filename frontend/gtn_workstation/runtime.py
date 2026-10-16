from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import operator
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore

from .device_state import DeviceLifecycle, DeviceState
from .models import (
    DEFAULT_SAMPLE_RATE_HZ,
    DIGIT_VOCAB,
    EDF_UNKNOWN_BIRTHDATE,
    MODEL_EPOCH_SUPPORT,
    MODEL_TASK_KEYS,
    ChannelMontage,
    DeviceSnapshot,
    ExperimentConfig,
    ModelTaskConfig,
)
from .recording_clock import SampleClockLedger
from .storage import SessionStore, atomic_write_json, load_model_input_session
from .transport import EEG_CALLBACK_BATCH_SIZE, transport_spec, validate_acquisition_rate

MODEL_TASKS = MODEL_TASK_KEYS
EEG_ADC_FULL_SCALE = float(1 << 23)
EEG_REFERENCE_VOLTAGE = 4.5
PREVIEW_CLEANUP_RETRY_COUNT = 3
SDK_CLEANUP_TIMEOUT_SECONDS = 3.0
EDF_FINALIZE_TIMEOUT_SECONDS = 30.0
EEG_STATS_STARTUP_SETTLE_SECONDS = 1.0
RECORDING_DATA_READY_TIMEOUT_SECONDS = 8.0
RECORDING_DATA_POLL_SECONDS = 0.05
IMPEDANCE_MEASUREMENT_SECONDS = 10
IMPEDANCE_SETUP_TIMEOUT_SECONDS = 8.0
RESUME_COUNTDOWN_SECONDS = 3


def sample_block_digits(rng: np.random.Generator, digits_per_block: int) -> np.ndarray:
    """Shuffle complete 1-9 cycles so every candidate has equal repetitions."""
    count = int(digits_per_block)
    vocabulary = np.asarray(DIGIT_VOCAB, dtype=np.int64)
    if count <= 0 or count % len(vocabulary):
        raise ValueError(f"每个 Block 的数字数必须是 {len(vocabulary)} 的正整数倍。")
    cycles = [rng.permutation(vocabulary) for _ in range(count // len(vocabulary))]
    return np.concatenate(cycles)


async def sleep_until(deadline_monotonic: float) -> None:
    """Wait for an absolute deadline without accumulating prior scheduling delay."""

    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining > 0.0:
        await asyncio.sleep(remaining)
PREVIEW_CLEANUP_RETRY_DELAY_SECONDS = 0.2
PREVIEW_TRANSFER_SETTLE_SECONDS = 0.1


class ImpedanceOperationError(RuntimeError):
    """A user-facing error with the failed impedance lifecycle stage."""

    _STAGE_LABELS = {
        "callback": "注册回调",
        "noise": "配置噪声滤波",
        "enable": "启用检测模式",
        "run": "执行检测",
        "data": "接收数据",
        "cleanup": "清理检测资源",
    }

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = str(detail)
        label = self._STAGE_LABELS.get(stage, stage)
        super().__init__(f"阻抗检测{label}失败: {self.detail}")


def split_process_arguments(text: str) -> list[str]:
    """Parse the user argument field with the host OS command-line rules."""

    if not text.strip():
        return []
    if os.name != "nt":
        return shlex.split(text, posix=True)

    # shlex.split(posix=False) preserves quote characters. Windows programs
    # receive arguments through CommandLineToArgvW semantics instead.
    import ctypes
    from ctypes import wintypes

    command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    command_line_to_argv.restype = ctypes.POINTER(wintypes.LPWSTR)
    local_free = ctypes.windll.kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    argc = ctypes.c_int()
    argv = command_line_to_argv(text, ctypes.byref(argc))
    if not argv:
        raise ValueError("算法参数无法按 Windows 命令行规则解析。")
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        local_free(argv)


def expand_process_arguments(text: str, replacements: dict[str, str]) -> list[str]:
    """Parse arguments first so substituted paths remain single argv items."""

    arguments = split_process_arguments(text)
    for index, argument in enumerate(arguments):
        for token, value in replacements.items():
            argument = argument.replace(token, str(value))
        arguments[index] = argument
    return arguments


class DeviceSignals(QtCore.QObject):
    eeg_batch = QtCore.Signal(object)
    impedance = QtCore.Signal(object)
    state_changed = QtCore.Signal(object)
    error = QtCore.Signal(str)
    acquisition_fault = QtCore.Signal(str)


def write_edf_plus_file(
    path: Path,
    data: np.ndarray,
    labels: list[str] | tuple[str, ...],
    sample_rate_hz: int,
    *,
    subject_id: str = "BrainSync",
    sex: str = "X",
    birthdate: str = EDF_UNKNOWN_BIRTHDATE,
    participant_name: str = "BrainSync GTN",
    annotations: tuple = ([], [], []),
) -> str:
    """Write an EDF+ file atomically; shared by finalize and crash recovery."""

    import pyedflib

    channel_count = len(labels)
    limits = np.maximum(5_000.0, np.max(np.abs(data), axis=0) * 1.1)
    headers = [
        {
            "label": label,
            "dimension": "uV",
            "sample_frequency": int(sample_rate_hz),
            "physical_min": -float(limit),
            "physical_max": float(limit),
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        }
        for label, limit in zip(labels, limits, strict=True)
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        writer = pyedflib.EdfWriter(
            str(temporary), channel_count, file_type=pyedflib.FILETYPE_EDFPLUS
        )
        try:
            record_count = max(1, int(np.ceil(len(data) / float(sample_rate_hz))))
            annotation_signals = max(1, int(np.ceil(len(annotations[0]) / record_count)))
            if annotation_signals > 64:
                raise ValueError("EDF annotations exceed the supported per-record capacity.")
            writer.set_number_of_annotation_signals(annotation_signals)
            writer.setSignalHeaders(headers)
            writer.setHeader(
                {
                    "technician": "BrainSync",
                    "recording_additional": "Callback acquisition",
                    "patientname": participant_name,
                    "patient_additional": "",
                    "patientcode": subject_id,
                    "sex": sex,
                    "birthdate": datetime.strptime(birthdate.upper(), "%d-%b-%Y").date(),
                    "startdate": datetime.now(),
                    "admincode": "",
                    "equipment": "BrainSync_8CH_EEG",
                }
            )
            writer.writeSamples([np.ascontiguousarray(data[:, index]) for index in range(channel_count)])
            for onset, duration, description in zip(*annotations, strict=True):
                writer.writeAnnotation(onset, duration, description)
        finally:
            writer.close()
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return str(path)


def load_recording_cache(checkpoint_path: str | Path) -> list[np.ndarray]:
    """Read every complete batch from a crash-recovery checkpoint stream.

    A truncated trailing record (process killed mid-write) is dropped.
    """

    path = Path(checkpoint_path)
    sidecar_path = path.with_name(path.name + ".json")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
    commit = metadata.get("checkpoint_commit")
    if metadata.get("checkpoint_schema") == "brainsync-joint-checkpoint/1" and not isinstance(commit, dict):
        raise ValueError("Joint recording checkpoint has no committed boundary.")
    limit = path.stat().st_size if commit is None else int(commit["voltage_bytes"])
    if limit < 0 or limit > path.stat().st_size:
        raise ValueError("Committed voltage checkpoint is truncated.")
    batches: list[np.ndarray] = []
    with path.open("rb") as fp:
        while fp.tell() < limit:
            try:
                array = np.lib.format.read_array(fp)
            except ValueError:
                if commit is not None:
                    raise ValueError("Committed voltage checkpoint contains an incomplete record.") from None
                break
            if fp.tell() > limit:
                raise ValueError("Voltage record crosses the committed checkpoint boundary.")
            if isinstance(array, np.ndarray) and array.ndim == 2:
                batches.append(array)
    if commit is not None:
        journal_path = path.with_name(path.name.removesuffix(".part") + ".clock.jsonl")
        with journal_path.open("rb") as fp:
            journal = fp.read(int(commit["journal_bytes"]))
        if len(journal) != commit["journal_bytes"]:
            raise ValueError("Committed clock journal is truncated.")
        rows = [json.loads(line) for line in journal.splitlines()]
        stored = [row for row in rows if row["n_stored"]]
        if (len(rows) != commit["clock_batches"] or len(batches) != commit["voltage_batches"]
                or len(stored) != len(batches)):
            raise ValueError("Committed clock and voltage batch counts disagree.")
        offset = 0
        for row, array in zip(stored, batches):
            if row["stored_start"] != offset or row["n_stored"] != len(array):
                raise ValueError("Committed clock rows do not match the voltage samples.")
            offset += len(array)
        if offset != commit["stored_samples"]:
            raise ValueError("Committed sample count disagrees with voltage data.")
    return batches


def rebuild_edf_from_cache(checkpoint_path: str | Path, output_path: str | Path | None = None) -> str:
    """Rebuild an EDF+ file from a checkpoint cache after a crash/force kill."""

    path = Path(checkpoint_path).expanduser().resolve()
    sidecar_path = path.with_name(path.name + ".json")
    metadata: dict[str, Any] = {}
    if sidecar_path.is_file():
        try:
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, json.JSONDecodeError):
            metadata = {}
    batches = load_recording_cache(path)
    if not batches:
        raise RuntimeError(f"检查点缓存没有可恢复的 EEG 数据: {path}")
    data = np.concatenate(batches, axis=0)
    if output_path is None:
        declared = str(metadata.get("edf_path") or "").strip()
        output_path = (
            path.parent / declared
            if declared and not Path(declared).is_absolute()
            else path.with_suffix(".recovered.edf")
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_edf_plus_file(
        output_path,
        data,
        list(metadata.get("labels") or ChannelMontage().eeg),
        int(metadata.get("sample_rate_hz") or DEFAULT_SAMPLE_RATE_HZ),
        subject_id=str(metadata.get("subject_id") or "BrainSync"),
        sex=str(metadata.get("sex") or "X"),
        birthdate=str(metadata.get("birthdate") or EDF_UNKNOWN_BIRTHDATE),
        participant_name=str(metadata.get("participant_name") or "BrainSync GTN"),
    )


class LocalEdfRecorder:
    """Write either transport's callback samples with joint voltage/clock checkpoints.

    The same callback buffer owns serial and BLE recording samples. This recorder:

    * appends every batch to a ``.part`` crash-recovery stream while recording
      (bounded by ``CHECKPOINT_INTERVAL_SECONDS``);
    * writes the final EDF+ file from memory when recording stops — BEFORE any
      SDK stop command is attempted — so a hung device cannot cost the data;
    * keeps the checkpoint stream until the EDF is safely on disk, allowing
      ``rebuild_edf_from_cache`` to recover a session after a force kill.
    """

    CHANNEL_COUNT = 8
    CHECKPOINT_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        output_dir: str | Path,
        labels: list[str] | tuple[str, ...],
        sample_rate_hz: int,
        subject_id: str,
        sex: str,
        birthdate: str,
        participant_name: str,
    ) -> None:
        normalized_labels = [str(label).strip() for label in labels]
        if len(normalized_labels) != self.CHANNEL_COUNT or any(not label for label in normalized_labels):
            raise ValueError("本地 EDF 记录器必须配置 8 个非空通道标签。")
        if int(sample_rate_hz) <= 0:
            raise ValueError("本地 EDF 记录器采样率必须大于 0。")
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = root / f"streaming_EEG_{timestamp}.edf"
        self.labels = normalized_labels
        self.sample_rate_hz = int(sample_rate_hz)
        self.subject_id = str(subject_id).strip() or "BrainSync"
        self.sex = str(sex).strip().upper() if str(sex).strip().upper() in {"M", "F", "X"} else "X"
        self.birthdate = str(birthdate).strip() or EDF_UNKNOWN_BIRTHDATE
        self.participant_name = str(participant_name).strip() or "BrainSync GTN"
        self.checkpoint_path = root / f"{self.path.name}.part"
        self.clock_journal_path = root / f"{self.path.name}.clock.jsonl"
        self.clock_metadata_path = root / f"{self.path.name}.clock.json"
        self.clock = SampleClockLedger(self.sample_rate_hz)
        self.integrity_failed = False
        self._clock_file = None
        self._clock_batches_written = 0
        self._clock_bytes_written = 0
        self._checkpoint_bytes_written = 0
        self._checkpoint_metadata = {
            "checkpoint_schema": "brainsync-joint-checkpoint/1",
            "edf_path": self.path.name, "labels": self.labels,
            "sample_rate_hz": self.sample_rate_hz, "subject_id": self.subject_id,
            "sex": self.sex, "birthdate": self.birthdate,
            "participant_name": self.participant_name,
            "checkpoint_commit": {
                "voltage_bytes": 0, "journal_bytes": 0, "voltage_batches": 0,
                "clock_batches": 0, "stored_samples": 0,
            },
        }
        self._checkpoint_initialized = False
        self._discard_next_packet_batch = False
        self._batches: list[np.ndarray] = []
        self._annotations: list[tuple[float, float, str]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._finalized = False
        self._finalize_started = False
        self._finalize_done = threading.Event()
        self._finalize_error: BaseException | None = None
        self._checkpoint_file = None
        self._checkpoint_batches_written = 0
        self._last_checkpoint_monotonic = time.monotonic()
        # Wall-clock (monotonic) time of the last batch actually appended to
        # the EDF buffer. Unlike snapshot.last_eeg_monotonic this can never be
        # refreshed by SDK-level counters, so it is a trustworthy "data is
        # reaching the file" signal for stall detection.
        self.last_append_monotonic: float | None = None
        self.paused = False

    @property
    def sample_count(self) -> int:
        with self._lock:
            return sum(batch.shape[0] for batch in self._batches)

    def append(
        self, data: np.ndarray, *, packet_metadata: list[dict] | None = None,
        receive_monotonic_ns: int | None = None,
    ) -> None:
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.CHANNEL_COUNT:
            raise ValueError(f"本地 EDF 数据形状为 {values.shape}，预期为 (样本数, 8)。")
        if not np.isfinite(values).all():
            raise ValueError("本地 EDF 数据包含 NaN 或无穷值。")
        receive_ns = time.monotonic_ns() if receive_monotonic_ns is None else receive_monotonic_ns
        metadata = packet_metadata if packet_metadata is not None else [{} for _ in values]
        if len(metadata) != len(values):
            raise ValueError("Packet metadata must align with every received sample.")
        with self._lock:
            if self._closed:
                return
            batch = self.clock.observe_batch(
                [row.get("seq_num") for row in metadata],
                [row.get("delta_time") for row in metadata], int(receive_ns),
            )
            if self.paused or self.integrity_failed:
                batch["storage_status"] = "paused" if self.paused else "integrity_failed"
                self.clock.commit_batch(batch, stored_start=None)
                return
            if not batch["sequence_valid"]:
                batch["storage_status"] = "sequence_failure"
                self.clock.commit_batch(batch, stored_start=None)
                self.integrity_failed = True
                raise ValueError("EEG sequence discontinuity; batch retained in ledger, not stored.")
            if packet_metadata is not None and self._discard_next_packet_batch:
                batch["storage_status"] = "uncertain_resume_batch"
                self.clock.commit_batch(batch, stored_start=None)
                self._discard_next_packet_batch = False
                return
            batch["storage_status"] = "buffered"
            start = sum(item.shape[0] for item in self._batches)
            self._batches.append(values.copy())
            self.clock.commit_batch(batch, stored_start=start)
            self.last_append_monotonic = time.monotonic()

    def reject_batch(self, metadata: list[dict], receive_ns: int, reason: str) -> None:
        with self._lock:
            if self._closed:
                return
            batch = self.clock.observe_batch(
                [row.get("seq_num") for row in metadata],
                [row.get("delta_time") for row in metadata], receive_ns,
            )
            batch.update(storage_status="conversion_failure", conversion_error=str(reason))
            self.clock.commit_batch(batch, stored_start=None)
            self.integrity_failed = True

    def clock_record(self) -> dict:
        with self._lock:
            return {**self.clock.record(), "integrity_failed": self.integrity_failed,
                    "journal_path": str(self.clock_journal_path),
                    "metadata_path": str(self.clock_metadata_path)}

    def pause(self) -> int:
        with self._lock:
            self.paused = True
            self.clock.close_segment(time.monotonic())
            return sum(batch.shape[0] for batch in self._batches)

    def resume(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("EDF recorder is already closed.")
            self.paused = False
            self._discard_next_packet_batch = True
            self.last_append_monotonic = time.monotonic()
            return sum(batch.shape[0] for batch in self._batches)

    def add_annotation(self, onset_seconds: float, description: str) -> None:
        with self._lock:
            if not self._closed:
                self._annotations.append((float(onset_seconds), 0.0, str(description)))

    def replace_annotations(self, annotations: list[tuple[float, float, str]]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot change annotations after EDF finalization.")
            self._annotations = list(annotations)

    def checkpoint(self, *, force: bool = False) -> None:
        """Append batches that are not yet persisted to the crash-recovery stream."""

        with self._lock:
            if self._finalized:
                return
            now = time.monotonic()
            if not force and now - self._last_checkpoint_monotonic < self.CHECKPOINT_INTERVAL_SECONDS:
                return
            pending = self._batches[self._checkpoint_batches_written:]
            pending_clock = self.clock.batches[self._clock_batches_written:]
            if not pending and not pending_clock:
                return
            sidecar_path = self.checkpoint_path.with_name(self.checkpoint_path.name + ".json")
            if not self._checkpoint_initialized:
                # Publish an empty boundary before appending either stream, so
                # recovery cannot mistake an interrupted first write for legacy data.
                atomic_write_json(sidecar_path, self._checkpoint_metadata)
                self._checkpoint_initialized = True
            if self._checkpoint_file is None:
                self._checkpoint_file = self.checkpoint_path.open(
                    "r+b" if self.checkpoint_path.exists() else "w+b"
                )
            if self._clock_file is None:
                self._clock_file = self.clock_journal_path.open(
                    "r+b" if self.clock_journal_path.exists() else "w+b"
                )
            streams = ((self._checkpoint_file, self._checkpoint_bytes_written),
                       (self._clock_file, self._clock_bytes_written))
            for stream, offset in streams:
                stream.seek(offset)
                stream.truncate()
            try:
                for batch in pending:
                    np.save(self._checkpoint_file, batch, allow_pickle=False)
                for batch in pending_clock:
                    self._clock_file.write(
                        (json.dumps(batch, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
                    )
                for stream, _ in streams:
                    stream.flush()
                    os.fsync(stream.fileno())
                commit = {
                    "voltage_bytes": self._checkpoint_file.tell(),
                    "journal_bytes": self._clock_file.tell(),
                    "voltage_batches": len(self._batches),
                    "clock_batches": len(self.clock.batches),
                    "stored_samples": self.clock.stored_samples,
                }
                metadata = {**self._checkpoint_metadata, "checkpoint_commit": commit}
                atomic_write_json(sidecar_path, metadata)
            except BaseException:
                # The published commit remains authoritative even when rollback
                # itself fails (for example, because the disk has disappeared).
                for stream, offset in streams:
                    try:
                        stream.seek(offset)
                        stream.truncate()
                        stream.flush()
                        os.fsync(stream.fileno())
                    except OSError:
                        pass
                raise
            self._checkpoint_metadata = metadata
            self._checkpoint_bytes_written = commit["voltage_bytes"]
            self._clock_bytes_written = commit["journal_bytes"]
            self._checkpoint_batches_written = len(self._batches)
            self._clock_batches_written = len(self.clock.batches)
            self._last_checkpoint_monotonic = now

    def emergency_save(self) -> str | None:
        """Synchronous last-resort save. Never raises; safe from atexit/any thread."""

        try:
            self.checkpoint(force=True)
        except Exception:
            pass
        with self._lock:
            if self._finalized:
                return str(self.path)
            if self._finalize_started:
                return None
        try:
            return self.finalize()
        except Exception:
            return None

    def close_empty(self) -> bool:
        """Release an aborted, empty writer without deleting clock evidence."""
        with self._lock:
            if self._batches or self._finalize_started:
                return False
            self._closed = True
            self._close_checkpoint_file_unlocked()
            return True

    def finalize(self) -> str:
        with self._lock:
            if self._finalized:
                return str(self.path)
            waiting = self._finalize_started
            if not waiting:
                self._finalize_started = True
                self._finalize_done = threading.Event()
                self._finalize_error = None
                self._closed = True
            completion = self._finalize_done
        # The writer needs the same lock to publish its result. Never hold
        # it while another caller waits for that result.
        if waiting:
            # The owner performs synchronous EDF serialization and verification;
            # its configured timeout applies to UI-side waits, not to this
            # hand-off. A short concurrent wait can otherwise report a healthy
            # export as failed while the owner is still publishing the file.
            if not completion.wait(timeout=max(EDF_FINALIZE_TIMEOUT_SECONDS, 1.0)):
                raise TimeoutError("本地 EDF export is still in progress.")
            with self._lock:
                if self._finalized:
                    return str(self.path)
                error = self._finalize_error
            raise RuntimeError("Concurrent 本地 EDF export failed; recording remains recoverable.") from error

        try:
            self.checkpoint(force=True)
            with self._lock:
                if not self._batches:
                    raise RuntimeError("本地 EDF 录制没有收到任何 EEG 回调数据。")
                batches = tuple(self._batches)
                annotations = tuple(zip(*self._annotations, strict=True)) if self._annotations else ([], [], [])
            data = np.concatenate(batches, axis=0)
            write_edf_plus_file(
                self.path,
                data,
                self.labels,
                self.sample_rate_hz,
                subject_id=self.subject_id,
                sex=self.sex,
                birthdate=self.birthdate,
                participant_name=self.participant_name,
                annotations=annotations,
            )
            import pyedflib

            reader = pyedflib.EdfReader(str(self.path))
            try:
                serialized_counts = reader.getNSamples().tolist()
                serialized_rates = reader.getSampleFrequencies().tolist()
            finally:
                reader.close()
            if (len(serialized_counts) != len(self.labels)
                    or len(set(serialized_counts)) != 1
                    or serialized_rates != [self.sample_rate_hz] * len(self.labels)
                    or serialized_counts[0] < len(data)):
                raise ValueError("Finalized EDF geometry disagrees with acquired samples.")
            edf_samples = int(serialized_counts[0])
            clock_record = self.clock_record()
            clock_record.update(
                finalized=True, voltage_samples=len(data),
                acquired_samples=len(data), edf_samples=edf_samples,
                padding_samples=edf_samples - len(data),
                journal_sha256=hashlib.sha256(self.clock_journal_path.read_bytes()).hexdigest(),
            )
            atomic_write_json(self.clock_metadata_path, clock_record)
        except BaseException as error:
            with self._lock:
                self._finalize_error = error
                self._finalize_started = False
                self._close_checkpoint_file_unlocked()
                completion.set()
            raise
        with self._lock:
            self._finalized = True
            self._finalize_started = False
            self._close_checkpoint_file_unlocked()
            completion.set()
        self._remove_checkpoint()
        return str(self.path)

    def _close_checkpoint_file_unlocked(self) -> None:
        clock_file = self._clock_file
        self._clock_file = None
        if clock_file is not None:
            clock_file.close()
        fp = self._checkpoint_file
        self._checkpoint_file = None
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass

    def _remove_checkpoint(self) -> None:
        for candidate in (
            self.checkpoint_path,
            self.checkpoint_path.with_name(self.checkpoint_path.name + ".json"),
        ):
            try:
                candidate.unlink()
            except OSError:
                pass


class DeviceAdapter:
    """Small, UI-independent contract around the BrainSync SDK."""

    def __init__(self) -> None:
        self.signals = DeviceSignals()
        self.lifecycle = DeviceLifecycle()
        self._recovery_source_state: DeviceState | None = None
        self.snapshot = DeviceSnapshot()
        self.handle: Any = None
        self._sdk: Any = None
        self._gain_enum: Any = None
        self._gain_value = 24.0
        self._eeg_callback_registered = False
        self._stream_recovery_active = False
        self._recording_operation_lock = asyncio.Lock()
        self._recording_start_task: asyncio.Task | None = None
        self._recording_start_operation: asyncio.Future | None = None
        self._recording_start_cleanup: asyncio.Task | None = None
        self._sdk_worker_done: threading.Event | None = None
        self._impedance_callback_registered = False
        self._impedance_mode_active = False
        self._impedance_transfer_started = False
        self._impedance_last_data_monotonic: float | None = None
        self._impedance_worker_done: threading.Event | None = None
        self._impedance_data_received = False
        self._impedance_controller_task: asyncio.Task | None = None
        self._loss_reset_task: asyncio.Task | None = None
        self._eeg_unit_guard_triggered = False
        self._recording_transfer_started = False
        self._recording_started_monotonic: float | None = None
        self._local_edf_recorder: LocalEdfRecorder | None = None
        self._disconnect_requested = False
        self._ble_connection_state = ""
        self._ble_device_id = ""
        self._eeg_readback_available: bool | None = None
        self._connection_operation: asyncio.Task | None = None
        self._connection_abort = False
        self._connection_generation = 0
        self._eeg_configuration_dirty = False
        self._requested_eeg_configuration = (DEFAULT_SAMPLE_RATE_HZ, 24)
    @property
    def recording_started_monotonic(self) -> float | None:
        """Monotonic timestamp at which the SDK recording transfer was enabled."""

        return self._recording_started_monotonic

    @property
    def sdk(self) -> Any | None:
        return self._sdk

    @property
    def state(self) -> DeviceState:
        return self.lifecycle.state

    @property
    def recovery_reason(self) -> str:
        return self.lifecycle.recovery_reason

    @property
    def recovery_may_have_preview(self) -> bool:
        return bool(
            self.state is DeviceState.RECOVERY_REQUIRED
            and (self._recovery_source_state is DeviceState.PREVIEWING or self._eeg_callback_registered)
        )

    @property
    def recovery_may_have_recording(self) -> bool:
        return bool(
            self.state is DeviceState.RECOVERY_REQUIRED
            and (self._recovery_source_state is DeviceState.RECORDING
                 or self._local_edf_recorder is not None or self._recording_start_cleanup is not None)
        )

    @property
    def recovery_may_have_impedance(self) -> bool:
        return bool(
            self.state is DeviceState.RECOVERY_REQUIRED
            and (
                self._recovery_source_state is DeviceState.IMPEDANCE
                or self._impedance_mode_active
                or self._impedance_transfer_started
                or self._impedance_callback_registered
            )
        )

    @property
    def impedance_data_received(self) -> bool:
        return self._impedance_data_received

    def _set_state(self, state: DeviceState, recovery_reason: str = "") -> None:
        """Apply a lifecycle transition and publish a derived UI snapshot."""

        previous_state = self.lifecycle.state
        self.lifecycle.transition(state, recovery_reason)
        if state is DeviceState.RECOVERY_REQUIRED:
            if previous_state is not DeviceState.RECOVERY_REQUIRED:
                self._recovery_source_state = previous_state
        else:
            self._recovery_source_state = None
        if state in {DeviceState.PREVIEWING, DeviceState.RECORDING} and previous_state is not state:
            # Counters are scoped to one transfer window. Keeping the previous
            # timestamp would make a newly started stream look stale before it
            # has delivered its first packet.
            self.snapshot.packets = 0
            self.snapshot.lost_packets = 0
            self.snapshot.last_eeg_monotonic = None
        if state not in {DeviceState.PREVIEWING, DeviceState.RECORDING}:
            self.snapshot.stream_started_monotonic = None
        self.signals.state_changed.emit(self.snapshot)

    def _enter_recovery(self, reason: Exception | str) -> None:
        if self.handle is None and self.state is DeviceState.DISCONNECTED:
            self.snapshot.last_error = str(reason)
            self.signals.error.emit(str(reason))
            return
        self._set_state(DeviceState.RECOVERY_REQUIRED, str(reason))

    def _has_live_resources(self) -> bool:
        return bool(
            (self._sdk_worker_done is not None and not self._sdk_worker_done.is_set())
            or self._eeg_callback_registered
            or self._impedance_callback_registered
            or self._impedance_mode_active
            or self._impedance_transfer_started
            or self._recording_transfer_started
            or self._local_edf_recorder is not None
        )

    def _on_sdk_connection_state(self, device_id: Any, state: Any) -> None:
        self._ble_device_id = str(device_id)
        self._ble_connection_state = str(state).strip()
        if self._ble_connection_state.casefold() != "disconnected":
            return
        if self._disconnect_requested or (
            self.handle is None and self.state is DeviceState.DISCONNECTED
        ):
            return

        reason = f"BLE 设备意外断开: {self._ble_device_id or 'unknown'}"
        self.handle = None
        self._eeg_callback_registered = False
        self._impedance_callback_registered = False
        self._impedance_mode_active = False
        self._impedance_transfer_started = False
        self._recording_transfer_started = False
        self._recording_started_monotonic = None
        self.snapshot.stream_started_monotonic = None
        self.snapshot.last_error = reason
        self.lifecycle.force_disconnected()
        # The lost handle cannot stop a later session from starting. Persist
        # this recorder now; retain its owner if export fails so it cannot be
        # silently replaced after reconnecting.
        self.emergency_save_recording()
        self.signals.error.emit(reason)
        self.signals.state_changed.emit(self.snapshot)

    @staticmethod
    def _is_readback_unavailable(error: Exception) -> bool:
        return "timeout" in str(error).casefold()

    def _settle_cleanup(self, errors: list[Exception]) -> None:
        if self.handle is None:
            self.lifecycle.force_disconnected()
            self.signals.state_changed.emit(self.snapshot)
            return
        if errors or self._has_live_resources():
            detail = "; ".join(str(error) for error in errors) or "设备资源未完全释放。"
            self._enter_recovery(detail)
        else:
            self._set_state(DeviceState.READY)

    async def connect(self, port_name: str | None = None, transport: str = "serial") -> dict[str, Any]:
        self._require_sdk_worker_idle()
        self.lifecycle.require("连接设备", DeviceState.DISCONNECTED)
        spec = transport_spec(transport)
        try:
            import brainsync_sdk as sdk
        except ImportError as exc:
            raise RuntimeError("未找到 brainsync_sdk，请使用 SDK 环境启动工作站。") from exc
        self._sdk = sdk
        self._gain_enum = getattr(getattr(sdk, "EegGain", None), "Gain24", None)
        self._gain_value = 24.0
        self._eeg_unit_guard_triggered = False
        self._disconnect_requested = False
        self._ble_connection_state = ""
        self._ble_device_id = ""
        self._eeg_readback_available = None
        self._connection_abort = False
        self._connection_generation += 1
        self._connection_operation = asyncio.create_task(self._open_connection(sdk, spec, port_name))
        self._connection_operation.add_done_callback(self._connection_finished)
        try:
            return await asyncio.shield(self._connection_operation)
        except asyncio.CancelledError:
            # The native opener's Future is a receipt, not cancellation of the
            # hardware operation. Keep its owner until any late handle is closed.
            self._connection_abort = True
            try:
                await self._wait_connection_operation()
            except TimeoutError as exc:
                self.signals.error.emit(str(exc))
            raise

    @property
    def connection_pending(self) -> bool:
        return self._connection_operation is not None and not self._connection_operation.done()

    def _connection_finished(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()  # A cancelled caller may no longer be awaiting the result.
        # Late cleanup must unlock the UI after the native owner has settled.
        self.signals.state_changed.emit(self.snapshot)

    async def wait_for_connection_cleanup(self) -> None:
        """Keep the event loop alive until a cancelled opener releases its handle.

        Application shutdown must await this owner; ordinary cancellation and
        disconnect remain bounded so the operator can see pending cleanup.
        """
        if self.connection_pending:
            self._connection_abort = True
        await self._wait_connection_operation(bounded=False)

    async def _wait_connection_operation(self, *, bounded: bool = True) -> None:
        task = self._connection_operation
        if task is None or task is asyncio.current_task():
            return
        # Gather retrieves a late failure even if the bounded waiter times out.
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(task, return_exceptions=True)),
                timeout=SDK_CLEANUP_TIMEOUT_SECONDS if bounded else None,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("连接原生请求尚未结束，正在等待句柄释放；暂不能重新连接。") from exc

    async def _open_connection(self, sdk, spec, port_name) -> dict[str, Any]:
        try:
            if spec.connection_callback:
                connection_callback = getattr(sdk, "set_connection_state_callback", None)
                if callable(connection_callback):
                    generation = self._connection_generation

                    def on_connection_state(device_id, state):
                        if generation == self._connection_generation:
                            self._on_sdk_connection_state(device_id, state)

                    connection_callback(on_connection_state)
            self.handle = await getattr(sdk, spec.opener)(port_name or spec.default_address)
            if self._connection_abort:
                raise asyncio.CancelledError()
            if spec.settle_seconds:
                await asyncio.sleep(spec.settle_seconds)
                if self._ble_connection_state.casefold() == "disconnected":
                    raise RuntimeError("SDK reported Disconnected after returning a BLE handle.")
            if self.handle is None:
                raise RuntimeError("SDK 未返回有效设备句柄。")
            opened_handle = self.handle
            firmware = await sdk.get_firmware_version(opened_handle)
            if self.handle != opened_handle:
                raise RuntimeError("连接验证期间设备已断开，不能使用迟到的固件应答。")
            if self._connection_abort:
                raise asyncio.CancelledError()
        except BaseException:
            if self.handle is not None:
                try:
                    self._disconnect_requested = True
                    handle = self.handle
                    await self._run_sdk_cleanup_operation("关闭设备", lambda: sdk.close_device(handle))
                except Exception as cleanup_error:
                    self._disconnect_requested = False
                    self._set_state(DeviceState.RECOVERY_REQUIRED, f"连接失败后句柄未释放: {cleanup_error}")
                    raise
            self.handle = None
            self._sdk = None
            self._gain_enum = None
            self._gain_value = 24.0
            self._disconnect_requested = False
            raise
        self.snapshot.transport = spec.key
        self.snapshot.firmware = self._format_firmware(firmware)
        self._set_state(DeviceState.READY)
        return firmware if isinstance(firmware, dict) else {"raw": str(firmware)}

    async def configure(self, sample_rate: int = DEFAULT_SAMPLE_RATE_HZ, gain: int = 24) -> None:
        self._require_sdk_worker_idle()
        self._require_handle()
        self.lifecycle.require("修改采样率或增益", DeviceState.READY)
        spec = transport_spec(self.snapshot.transport)
        validate_acquisition_rate(sample_rate)
        sample_enum = getattr(self._sdk.EegSampleRate, f"Hz{sample_rate}")
        gain_enum = getattr(self._sdk.EegGain, f"Gain{gain}")
        self._requested_eeg_configuration = (sample_rate, gain)
        self._invalidate_eeg_configuration("配置尚未完成，开始 EEG 前必须重新应用。")
        await self._sdk.set_eeg_sample_rate(self.handle, sample_enum)
        await self._sdk.set_eeg_gain(self.handle, gain_enum)
        await self._sdk.set_eeg_signal_type(self.handle, self._sdk.EegSignalType.Normal)
        verification = "readback_confirmed"
        verification_note = ""
        try:
            if spec.allow_readback_timeout and self._eeg_readback_available is False:
                raise RuntimeError("Request timeout (BLE readback previously unavailable)")
            actual_sample_rate = await self._read_eeg_sample_rate()
        except Exception as exc:
            if not spec.allow_readback_timeout or not self._is_readback_unavailable(exc):
                raise
            try:
                await self._sdk.get_firmware_version(self.handle)
            except Exception as liveness_exc:
                raise RuntimeError(
                    f"BLE EEG 参数读回失败，且设备链路存活确认失败: {liveness_exc}"
                ) from exc
            actual_sample_rate = int(sample_rate)
            verification = "write_acknowledged_readback_unavailable"
            verification_note = str(exc)
            self._eeg_readback_available = False
            self.signals.error.emit(
                "BLE 固件未响应 EEG 参数读回；写命令已接受且固件链路仍存活，"
                "本次配置按请求值记录。"
            )
        else:
            self._eeg_readback_available = True
            if actual_sample_rate != int(sample_rate):
                raise RuntimeError(
                    f"设备采样率读回为 {actual_sample_rate} Hz，配置请求为 {sample_rate} Hz。"
                )
        self._gain_enum = gain_enum
        self._gain_value = float(gain)
        self.snapshot.sample_rate_hz = actual_sample_rate
        self.snapshot.gain = f"Gain{gain}"
        self.snapshot.configuration_verification = verification
        self.snapshot.configuration_note = verification_note
        self._eeg_configuration_dirty = False
        self._set_state(DeviceState.READY)

    def _invalidate_eeg_configuration(self, note: str) -> None:
        self._eeg_configuration_dirty = True
        self.snapshot.configuration_verification = "configuration_required"
        self.snapshot.configuration_note = note

    async def _ensure_eeg_configuration(self) -> None:
        if self._eeg_configuration_dirty:
            await self.configure(*self._requested_eeg_configuration)

    async def start_stream(self, batch_size: int = EEG_CALLBACK_BATCH_SIZE) -> None:
        self._require_sdk_worker_idle()
        self._require_handle()
        self.lifecycle.require("启动 EEG 预览", DeviceState.READY)
        await self._ensure_eeg_configuration()
        if self._eeg_callback_registered:
            self._enter_recovery("检测到未释放的 EEG 预览订阅。")
            raise RuntimeError("EEG 预览订阅未释放，请先断开设备恢复。")
        if batch_size <= 0 or batch_size % EEG_CALLBACK_BATCH_SIZE:
            raise ValueError(
                f"SDK EEG batch_size 必须是 {EEG_CALLBACK_BATCH_SIZE} 的正整数倍。"
            )
        clear_buffer = getattr(self._sdk, "clear_receive_buffer", None)
        if callable(clear_buffer):
            await self._maybe_await(clear_buffer(self.handle))
        transfer_started = False
        transfer_stopped = True
        try:
            await self._sdk.subscribe_eeg_data(self.handle, batch_size, self._on_eeg_batch)
            self._eeg_callback_registered = True
            transfer_started = True
            await self._sdk.set_eeg_transfer(self.handle, True)
        except Exception:
            if transfer_started:
                try:
                    await self._retry_eeg_cleanup_operation(
                        "关闭 EEG 传输",
                        lambda: self._sdk.set_eeg_transfer(self.handle, False),
                    )
                    transfer_stopped = True
                except Exception:
                    transfer_stopped = False
            if self._eeg_callback_registered:
                try:
                    await self._retry_eeg_cleanup_operation(
                        "取消 EEG 订阅",
                        lambda: self._sdk.unsubscribe_eeg_data(self.handle),
                    )
                    self._eeg_callback_registered = False
                except Exception:
                    pass
            if not transfer_stopped or self._eeg_callback_registered:
                self._enter_recovery("启动 EEG 预览失败且 SDK 清理未完成。")
            raise
        self.snapshot.stream_started_monotonic = time.monotonic()
        self._set_state(DeviceState.PREVIEWING)
        self._loss_reset_task = asyncio.create_task(self._reset_loss_stats_task())

    async def stop_stream(self) -> None:
        cancelled_start = await self._cancel_recording_start()
        if cancelled_start and self.state is DeviceState.READY:
            return
        if self.handle is None:
            return
        if self._recording_start_cleanup is not None:
            await self._wait_recording_start_cleanup()
            if self.state is DeviceState.READY:
                return
        self.lifecycle.require("停止 EEG 预览", DeviceState.PREVIEWING, DeviceState.RECOVERY_REQUIRED)
        if (
            self.lifecycle.state is DeviceState.RECOVERY_REQUIRED
            and not self._eeg_callback_registered
            and not self.recovery_may_have_preview
        ):
            return
        await self._cancel_loss_stats_reset()
        errors: list[Exception] = []
        # A failed cleanup can leave the subscription registered while the
        # UI snapshot already says stopped. Always send a stop command while
        # either part of the preview lifecycle is still active.
        transfer_active = self.state is DeviceState.PREVIEWING or self.recovery_may_have_preview
        transfer_stopped = not transfer_active
        if transfer_active:
            try:
                await self._retry_eeg_cleanup_operation(
                    "关闭 EEG 传输",
                    lambda: self._sdk.set_eeg_transfer(self.handle, False),
                )
                transfer_stopped = True
                # BLE firmware may need a short interval to commit the stop
                # command before accepting the subscription cleanup.
                await asyncio.sleep(PREVIEW_TRANSFER_SETTLE_SECONDS)
            except Exception as exc:
                errors.append(exc)
        callback_cleared = not self._eeg_callback_registered
        if self._eeg_callback_registered:
            try:
                await self._retry_eeg_cleanup_operation(
                    "取消 EEG 订阅",
                    lambda: self._sdk.unsubscribe_eeg_data(self.handle),
                )
                self._eeg_callback_registered = False
                callback_cleared = True
            except Exception as exc:
                errors.append(exc)
        if not transfer_stopped or not callback_cleared:
            errors.append(RuntimeError("EEG 预览资源仍处于活动状态。"))
        self._settle_cleanup(errors)
        if errors:
            raise RuntimeError("停止预览 EEG 流失败: " + "; ".join(str(error) for error in errors)) from errors[0]

    async def start_recording(
        self,
        output_dir: str,
        subject_id: str,
        channel_config: Any = None,
        sex: str = "X",
        birthdate: str = EDF_UNKNOWN_BIRTHDATE,
        participant_name: str = "BrainSync GTN",
    ) -> str:
        if self._recording_start_task is not None and not self._recording_start_task.done():
            raise RuntimeError("已有 EDF 录制正在启动。")
        self._recording_start_task = asyncio.current_task()
        try:
            return await self._start_recording(
                output_dir, subject_id, channel_config, sex, birthdate, participant_name,
            )
        finally:
            self._recording_start_task = None

    async def _cancel_recording_start(self) -> bool:
        task = self._recording_start_task
        if task is None or task.done() or task is asyncio.current_task():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def _start_recording(
        self, output_dir, subject_id, channel_config, sex, birthdate, participant_name,
    ) -> str:
        self._require_sdk_worker_idle()
        self._require_handle()
        self.lifecycle.require("启动 EDF 录制", DeviceState.READY)
        await self._ensure_eeg_configuration()
        if self._local_edf_recorder is not None:
            self._enter_recovery("上一份 EDF 录制尚未保存，必须先恢复保存后才能开始新录制。")
            raise RuntimeError("上一份 EDF 录制尚未保存。")
        self._recording_start_cleanup = None
        if self._eeg_callback_registered:
            self._enter_recovery("检测到未释放的 EEG 订阅。")
            raise RuntimeError("请先释放上一条 EEG 订阅再开始录制。")
        return await self._start_callback_recording(
            output_dir, subject_id, channel_config, sex, birthdate, participant_name)

    async def _recording_start_step(self, request):
        """Keep the native receipt alive until resource ownership is settled.

        SDK 0.2.0 returns native-backed asyncio Futures. Cancelling that
        receipt does not establish that the native side effect was undone.
        """
        self._require_handle()
        handle = self.handle
        outcome = request()
        if not inspect.isawaitable(outcome):
            if handle is None or self.handle != handle:
                raise RuntimeError("录制启动期间设备连接已断开或改变。")
            return outcome
        operation = asyncio.ensure_future(outcome)
        self._recording_start_operation = operation
        try:
            result = await asyncio.shield(operation)
            if handle is None or self.handle != handle:
                raise RuntimeError("录制启动期间设备连接已断开或改变。")
            return result
        finally:
            if operation.done():
                self._recording_start_operation = None

    async def _abort_recording_start(self, error: BaseException) -> None:
        self._enter_recovery(f"录制启动未完成，正在清理: {error}")
        self._recording_start_cleanup = asyncio.create_task(self._finish_recording_start_cleanup())
        try:
            await self._wait_recording_start_cleanup()
        except (TimeoutError, asyncio.CancelledError):
            # Keep the owner alive. It will clean up after the native request
            # settles; stop/disconnect may not race that pending request.
            self._enter_recovery("录制启动操作尚未完成，等待原生请求结束后清理")

    async def _wait_recording_start_cleanup(self) -> None:
        cleanup = self._recording_start_cleanup
        if cleanup is not None:
            try:
                await asyncio.wait_for(asyncio.shield(cleanup), timeout=SDK_CLEANUP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as error:
                raise TimeoutError("录制启动的原生请求尚未结束，暂不能再次清理或关闭设备") from error
            self._recording_start_cleanup = None

    async def _finish_recording_start_cleanup(self) -> None:
        operation = self._recording_start_operation
        if operation is not None:
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                # A terminal SDK cancellation still needs reverse cleanup.
                # Cancellation of this owner cannot bypass a pending native request.
                if not operation.done():
                    raise
            except Exception:
                pass  # A failed request can still have partially acquired resources.
            self._recording_start_operation = None
        try:
            async with self._recording_operation_lock:
                await self._stop_recording_locked()
        except Exception as error:
            self._enter_recovery(f"录制启动清理未完成: {error}")

    async def _start_callback_recording(
        self,
        output_dir: str,
        subject_id: str,
        channel_config: Any,
        sex: str,
        birthdate: str,
        participant_name: str,
    ) -> str:
        """Record either SDK transport through one callback, clock and EDF writer."""

        labels = getattr(channel_config, "labels", None) or list(ChannelMontage().eeg)
        recorder = LocalEdfRecorder(
            output_dir,
            list(labels),
            int(self.snapshot.sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ),
            subject_id,
            sex,
            birthdate,
            participant_name,
        )
        self._local_edf_recorder = recorder
        self._last_recording_clock = None
        try:
            # Both transports stop transfer and allow the command to settle before
            # registering a new callback. An inactive transfer is harmless.
            try:
                await self._recording_start_step(lambda: self._sdk.set_eeg_transfer(self.handle, False))
            except Exception as exc:
                if not self._is_already_inactive_error(exc):
                    raise
            await asyncio.sleep(PREVIEW_TRANSFER_SETTLE_SECONDS)
            for name in ("clear_receive_buffer", "reset_eeg_loss_stats"):
                operation = getattr(self._sdk, name, None)
                if callable(operation):
                    await self._recording_start_step(lambda operation=operation: operation(self.handle))
            self._require_handle()
            self._eeg_callback_registered = True
            await self._recording_start_step(lambda: self._sdk.subscribe_eeg_data(
                self.handle, EEG_CALLBACK_BATCH_SIZE, self._on_eeg_batch
            ))
            # Some SDK builds may dispatch the first callback synchronously
            # from set_eeg_transfer(True). Enter RECORDING before that call so
            # the lifecycle counter reset cannot erase an already received
            # batch.
            self._set_state(DeviceState.RECORDING)
            self._recording_transfer_started = True
            await self._recording_start_step(lambda: self._sdk.set_eeg_transfer(self.handle, True))
            self._recording_started_monotonic = time.monotonic()
            self.snapshot.stream_started_monotonic = self._recording_started_monotonic
            self._recording_started_monotonic = await self._wait_for_recording_data()
            self.snapshot.stream_started_monotonic = self._recording_started_monotonic
            return str(recorder.path)
        except (Exception, asyncio.CancelledError) as start_error:
            await self._abort_recording_start(start_error)
            raise

    async def _read_eeg_sample_rate(self) -> int:
        getter = getattr(self._sdk, "get_eeg_params", None)
        converter = getattr(self._sdk, "get_eeg_sample_rate_value", None)
        if not callable(getter) or not callable(converter):
            raise RuntimeError("当前 SDK 无法读回 EEG 采样率。")
        params = await self._maybe_await(getter(self.handle))
        if isinstance(params, dict):
            raw_rate = next(
                (
                    params[name]
                    for name in ("sample_rate", "sampling_rate", "sample_rate_hz")
                    if name in params
                ),
                None,
            )
        elif isinstance(params, (tuple, list)) and params:
            raw_rate = params[0]
        else:
            raw_rate = next(
                (
                    getattr(params, name)
                    for name in ("sample_rate", "sampling_rate", "sample_rate_hz")
                    if hasattr(params, name)
                ),
                None,
            )
        if raw_rate is None:
            raise RuntimeError("SDK EEG 参数缺少 sample_rate。")
        value = await self._maybe_await(converter(raw_rate))
        try:
            actual = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"SDK 返回了无效 EEG 采样率: {value!r}") from exc
        if actual <= 0:
            raise RuntimeError(f"SDK 返回了无效 EEG 采样率: {actual}")
        return actual

    async def _wait_for_recording_data(self) -> float:
        deadline = time.monotonic() + RECORDING_DATA_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            recorder = self._local_edf_recorder
            if recorder is None:
                raise RuntimeError("录制回调记录器不可用。")
            if recorder.integrity_failed:
                raise RuntimeError("EEG acquisition integrity failed before recording became ready.")
            if recorder.sample_count and recorder.clock.first_sample_monotonic is not None:
                self.snapshot.last_eeg_monotonic = recorder.last_append_monotonic
                return recorder.clock.first_sample_monotonic
            await asyncio.sleep(RECORDING_DATA_POLL_SECONDS)
        raise TimeoutError(f"EDF 已启动，但 {RECORDING_DATA_READY_TIMEOUT_SECONDS:.0f} 秒内未收到 EEG 数据。")

    async def stop_recording(self) -> None:
        cancelled_start = await self._cancel_recording_start()
        if cancelled_start and self.state is DeviceState.READY:
            return
        if self._recording_start_cleanup is not None:
            await self._wait_recording_start_cleanup()
            if self.state is DeviceState.READY:
                return
        async with self._recording_operation_lock:
            await self._stop_recording_locked()

    @property
    def recording_paused(self) -> bool:
        return self._local_edf_recorder is not None and self._local_edf_recorder.paused

    def pause_recording(self) -> int:
        self.lifecycle.require("暂停 EDF 写入", DeviceState.RECORDING)
        if self._local_edf_recorder is None:
            raise RuntimeError("This recording has no pausable callback recorder.")
        return self._local_edf_recorder.pause()

    def resume_recording(self) -> int:
        self.lifecycle.require("继续 EDF 写入", DeviceState.RECORDING)
        if self._local_edf_recorder is None:
            raise RuntimeError("This recording has no pausable callback recorder.")
        # Mock's sample clock must not backfill the discarded pause interval.
        if hasattr(self, "_append_mock_samples"):
            self._append_mock_samples(time.monotonic())
        return self._local_edf_recorder.resume()

    def replace_recording_annotations(self, annotations) -> None:
        """Publish the finalized stored-sample marker timeline for a pausable recording."""
        if self._local_edf_recorder is None:
            raise RuntimeError("This recording has no pausable callback recorder.")
        self._local_edf_recorder.replace_annotations(annotations)

    async def _stop_recording_locked(self) -> None:
        """Persist received samples before releasing either transport's subscription."""
        if self.handle is not None:
            self.lifecycle.require("停止 EDF 录制", DeviceState.RECORDING, DeviceState.RECOVERY_REQUIRED)
        await self._cancel_loss_stats_reset()
        recorder = self._local_edf_recorder
        errors: list[Exception] = []

        self._recording_started_monotonic = None
        if recorder is not None:
            # Persist the EDF from the in-memory buffer before touching the
            # device. This step does not depend on the hardware, so even a
            # fully stalled transport cannot cost the session its data.
            try:
                if not recorder.close_empty():
                    await asyncio.wait_for(
                        asyncio.to_thread(recorder.finalize),
                        timeout=EDF_FINALIZE_TIMEOUT_SECONDS,
                    )
            except Exception as exc:
                errors.append(RuntimeError(f"本地 EDF 文件写入失败: {exc}"))
            else:
                self._last_recording_clock = recorder.clock_record()
                self._local_edf_recorder = None

        if self.handle is not None and self._recording_transfer_started:
            try:
                await self._retry_eeg_cleanup_operation(
                    "关闭 EEG 传输",
                    lambda: self._sdk.set_eeg_transfer(self.handle, False),
                )
                self._recording_transfer_started = False
                await asyncio.sleep(PREVIEW_TRANSFER_SETTLE_SECONDS)
            except Exception as exc:
                errors.append(exc)
        if self.handle is not None and self._eeg_callback_registered:
            try:
                await self._retry_eeg_cleanup_operation(
                    "取消 EEG 订阅",
                    lambda: self._sdk.unsubscribe_eeg_data(self.handle),
                )
                self._eeg_callback_registered = False
            except Exception as exc:
                errors.append(exc)

        self._settle_cleanup(errors)
        if errors:
            raise RuntimeError(
                "停止 本地 EDF 录制或数据传输失败: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    def checkpoint_recording(self) -> None:
        """Persist recent EEG samples to the crash-recovery stream.

        Cheap, thread-safe, and never raises; call periodically (e.g. from a
        UI monitor timer) so a force kill loses at most a few seconds.
        """

        recorder = self._local_edf_recorder
        if recorder is None:
            return
        try:
            recorder.checkpoint()
        except Exception:
            pass

    @property
    def recording_last_sample_monotonic(self) -> float | None:
        """Monotonic time of the last sample that reached the EDF buffer.

        For callback recordings this reflects the callback->recorder path only;
        SDK-level counters cannot refresh it. No sample is implied when there
        is no active recorder.
        """

        recorder = self._local_edf_recorder
        if recorder is not None:
            return recorder.last_append_monotonic
        return None

    @property
    def recording_sample_count(self) -> int | None:
        """Samples currently buffered for the active callback recording (or None)."""

        recorder = self._local_edf_recorder
        if recorder is None:
            return None
        try:
            return recorder.sample_count
        except Exception:
            return None

    @property
    def recording_clock_record(self) -> dict | None:
        recorder = self._local_edf_recorder
        if recorder is not None:
            return recorder.clock_record()
        return getattr(self, "_last_recording_clock", None)

    def recording_source_sample(self, wall_monotonic: float) -> int | None:
        position = self.recording_source_position(wall_monotonic)
        return None if position is None else position[0]

    def recording_source_position(self, wall_monotonic: float) -> tuple[int, int] | None:
        if hasattr(self, "_append_mock_samples"):
            self._append_mock_samples(wall_monotonic)
        recorder = self._local_edf_recorder
        if recorder is None or recorder.clock.first_sample_monotonic is None:
            return None
        with recorder._lock:
            return recorder.clock.source_position(wall_monotonic)

    async def wait_for_recorded_sample(
        self, sample_index: int, *, start_sample: int, segment_id: int,
    ) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (sample_index, start_sample, segment_id)) or start_sample > sample_index:
            raise ValueError("Required EEG support and segment must be valid non-negative indices.")
        deadline = time.monotonic() + RECORDING_DATA_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            recorder = self._local_edf_recorder
            if recorder is None or recorder.integrity_failed:
                raise RuntimeError("Cannot confirm the required EEG tail samples.")
            with recorder._lock:
                segments = recorder.clock.segments
                segment = segments[segment_id] if segment_id < len(segments) else None
                if segment is None or start_sample < segment["stored_start"]:
                    raise RuntimeError("Required EEG support has no continuous acquisition segment.")
                complete = segment["stored_stop"] > sample_index
                if not complete and "closed_monotonic_ns" in segment:
                    raise RuntimeError("Required EEG support crosses a continuous acquisition segment boundary.")
            if complete:
                recorder.checkpoint(force=True)
                return
            if hasattr(self, "_append_mock_samples"):
                self._append_mock_samples(time.monotonic())
            await asyncio.sleep(RECORDING_DATA_POLL_SECONDS)
        raise TimeoutError("The complete EEG response tail was not received and persisted.")

    async def wait_for_recorded_baseline(self, *, not_before_monotonic: float | None = None) -> dict:
        if not_before_monotonic is not None and not np.isfinite(not_before_monotonic):
            raise ValueError("Baseline boundary must be a finite monotonic timestamp.")
        requested_at = time.monotonic()
        deadline = requested_at + RECORDING_DATA_READY_TIMEOUT_SECONDS
        required = None
        while time.monotonic() < deadline:
            recorder = self._local_edf_recorder
            if recorder is None or recorder.integrity_failed:
                raise RuntimeError("Cannot confirm the required EEG baseline samples.")
            with recorder._lock:
                segments = recorder.clock.segments
                segment = segments[-1] if segments and "closed_monotonic_ns" not in segments[-1] else None
                if required is not None and (segment is None or segment["segment_id"] != required["segment_id"]):
                    raise RuntimeError("Required EEG baseline crosses a continuous acquisition segment boundary.")
                if segment is not None and not recorder.paused:
                    rate = recorder.sample_rate_hz
                    count = int(np.ceil(-MODEL_EPOCH_SUPPORT.tmin_ms * rate / 1000.0))
                    first_time = segment["first_sample_estimate_monotonic_ns"] / 1e9
                    if required is None:
                        earliest = segment["stored_start"]
                        if not_before_monotonic is not None:
                            earliest += max(0, int(np.ceil((not_before_monotonic - first_time) * rate)))
                        requested_sample, _ = recorder.clock.source_position(max(requested_at, first_time))
                        stop = max(requested_sample, earliest + count)
                        required = {"segment_id": segment["segment_id"],
                                    "start_sample": stop - count, "stop_sample_exclusive": stop,
                                    "sample_rate_hz": rate}
                    complete = segment["stored_stop"] >= required["stop_sample_exclusive"]
                else:
                    complete = False
            if complete:
                recorder.checkpoint(force=True)
                return {**required, "checkpointed": True}
            if hasattr(self, "_append_mock_samples"):
                self._append_mock_samples(time.monotonic())
            await asyncio.sleep(RECORDING_DATA_POLL_SECONDS)
        raise TimeoutError("The complete EEG baseline was not received and persisted.")

    async def attempt_stream_recovery(self) -> bool:
        if self._recording_operation_lock.locked():
            return False
        async with self._recording_operation_lock:
            return await self._attempt_stream_recovery_locked()

    async def _attempt_stream_recovery_locked(self) -> bool:
        """Attempt to restore a stalled EEG callback stream by re-subscribing.

        P004-P006 diagnosis: the device kept sending data (SDK counters kept
        rising) while the SDK's native callback dispatch stopped delivering
        batches to `_on_eeg_batch`. The only Python-side remedy is to
        unsubscribe and re-subscribe the EEG callback. Safe to call whenever
        the stream looks stalled: it never raises, only acts on an active callback
        recording, and every SDK call is thread-isolated with a timeout so a
        wedged device cannot hang the UI.
        """

        if (
            self.handle is None or self.lifecycle.state is not DeviceState.RECORDING
        ):
            return False
        if self._local_edf_recorder is None or not self._eeg_callback_registered:
            return False
        if self._stream_recovery_active:
            return False
        self._stream_recovery_active = True
        try:
            batch_size = EEG_CALLBACK_BATCH_SIZE
            # 1) Stop the transfer; "already inactive" is the expected state
            #    when the dispatch died, not an error.
            try:
                await self._run_sdk_cleanup_operation(
                    "关闭 EEG 传输",
                    lambda: self._sdk.set_eeg_transfer(self.handle, False),
                )
                self._recording_transfer_started = False
            except Exception as exc:
                if not self._is_already_inactive_error(exc):
                    raise
                self._recording_transfer_started = False
            await asyncio.sleep(PREVIEW_TRANSFER_SETTLE_SECONDS)
            # 2) Drop the dead subscription.
            try:
                await self._run_sdk_cleanup_operation(
                    "取消 EEG 订阅",
                    lambda: self._sdk.unsubscribe_eeg_data(self.handle),
                )
            except Exception as exc:
                if not self._is_already_inactive_error(exc):
                    raise
            self._eeg_callback_registered = False
            await asyncio.sleep(PREVIEW_TRANSFER_SETTLE_SECONDS)
            # 3) Re-register the callback and restart the transfer.
            self._eeg_callback_registered = True
            await self._run_sdk_cleanup_operation(
                "重新订阅 EEG 数据",
                lambda: self._sdk.subscribe_eeg_data(
                    self.handle, batch_size, self._on_eeg_batch
                ),
            )
            self._recording_transfer_started = True
            await self._run_sdk_cleanup_operation(
                "重启 EEG 传输",
                lambda: self._sdk.set_eeg_transfer(self.handle, True),
            )
            return True
        except Exception as error:
            self._enter_recovery(error)
            return False
        finally:
            self._stream_recovery_active = False

    def emergency_save_recording(self) -> str | None:
        """Synchronous last-resort EDF export for the active callback recording.

        Never raises; returns the EDF path on success. Safe to call from any
        thread, including atexit handlers and Qt's aboutToQuit.
        """

        recorder = self._local_edf_recorder
        if recorder is None:
            return None
        path = recorder.emergency_save()
        if path is not None:
            self._last_recording_clock = recorder.clock_record()
            if not self._recording_transfer_started and not self._eeg_callback_registered:
                self._local_edf_recorder = None
        return path

    async def add_annotation(self, onset_seconds: float, description: str) -> None:
        if self.lifecycle.state is DeviceState.RECORDING:
            if self._local_edf_recorder is None:
                raise RuntimeError("录制回调记录器不可用。")
            self._local_edf_recorder.add_annotation(onset_seconds, description)

    async def run_impedance(self, duration_sec: int | None = None) -> None:
        self._require_handle()
        if duration_sec is not None and int(duration_sec) <= 0:
            raise ValueError("阻抗检测时长必须大于 0 秒。")
        self.lifecycle.require("启动阻抗检测", DeviceState.READY)
        self._impedance_controller_task = asyncio.current_task()
        self._impedance_data_received = False
        self._impedance_last_data_monotonic = None
        self._set_state(DeviceState.IMPEDANCE)
        primary_error: BaseException | None = None
        try:
            if not self._impedance_callback_registered:
                # Mark ownership before a native call that may partially succeed.
                self._impedance_callback_registered = True
                sdk, handle = self._sdk, self.handle
                await self._impedance_sdk_operation(
                    "callback", lambda: sdk.set_impedance_callback(handle, self._on_impedance)
                )
            sdk, handle = self._sdk, self.handle
            await self._impedance_sdk_operation(
                "noise", lambda: sdk.set_impedance_noise_type(handle, sdk.NoiseType.Both)
            )
            self._impedance_mode_active = True
            # The SDK sets 250 Hz/Gain24 on enable and does not promise to
            # restore the prior EEG configuration when disabling impedance.
            self._invalidate_eeg_configuration("阻抗检测修改了 EEG 参数，开始 EEG 前将恢复所选配置。")
            await self._impedance_sdk_operation(
                "enable", lambda: sdk.enable_impedance_detection_mode(handle)
            )
            self._impedance_transfer_started = True
            await self._impedance_sdk_operation(
                "run", lambda: sdk.set_eeg_transfer(handle, True)
            )
            started = time.monotonic()
            # The SDK convenience helper toggles transfer around a sleep.
            # Own the continuous transfer here so cancellation cannot leave a
            # native timer that later stops a new preview or recording.
            while True:
                now = time.monotonic()
                last_data = self._impedance_last_data_monotonic
                if now - (last_data if last_data is not None else started) >= IMPEDANCE_MEASUREMENT_SECONDS:
                    raise ImpedanceOperationError("data", "检测期间未收到设备阻抗回调数据。")
                if duration_sec is not None and now - started >= int(duration_sec):
                    if not self._impedance_data_received:
                        raise ImpedanceOperationError("data", "检测期间未收到设备阻抗回调数据。")
                    break
                await asyncio.sleep(0.05)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error = await self._cleanup_impedance_resources()
            self._impedance_controller_task = None
            if cleanup_error is None:
                self._settle_cleanup([])
            else:
                self._enter_recovery(cleanup_error)
            if cleanup_error is not None and primary_error is None:
                raise ImpedanceOperationError("cleanup", cleanup_error)

    async def stop_impedance(self) -> None:
        if self.handle is None:
            return
        self.lifecycle.require("停止阻抗检测", DeviceState.IMPEDANCE, DeviceState.RECOVERY_REQUIRED)
        controller = self._impedance_controller_task
        current = asyncio.current_task()
        if controller is not None and controller is not current and not controller.done():
            controller.cancel()
            await asyncio.gather(controller, return_exceptions=True)
            self._impedance_controller_task = None
        cleanup_error = await self._cleanup_impedance_resources()
        if cleanup_error is not None:
            self._enter_recovery(cleanup_error)
            raise ImpedanceOperationError("cleanup", cleanup_error) from cleanup_error
        self._settle_cleanup([])

    async def clear_impedance(self) -> None:
        if self._impedance_callback_registered or self._impedance_mode_active or self._impedance_transfer_started:
            await self.stop_impedance()

    async def _impedance_sdk_operation(self, stage: str, request: Any) -> None:
        previous = self._impedance_worker_done
        if previous is not None and not previous.is_set():
            raise ImpedanceOperationError(stage, "上一条设备命令仍未返回，请断开设备后重连。")
        finished = threading.Event()
        self._impedance_worker_done = finished

        async def invoke() -> None:
            try:
                await self._maybe_await(request())
            finally:
                finished.set()

        label = ImpedanceOperationError._STAGE_LABELS.get(stage, stage)
        logger = logging.getLogger(__name__)
        logger.info("Impedance command start: %s", label)
        operation = asyncio.create_task(self._run_sdk_cleanup_operation(
            f"阻抗{label}", invoke,
            timeout_seconds=(SDK_CLEANUP_TIMEOUT_SECONDS if stage == "cleanup" else IMPEDANCE_SETUP_TIMEOUT_SECONDS),
            allow_loop_fallback=False,
        ))
        try:
            # Finish or time out the native command before releasing resources.
            await asyncio.shield(operation)
            logger.info("Impedance command complete: %s", label)
        except asyncio.CancelledError:
            await asyncio.gather(operation, return_exceptions=True)
            raise
        except Exception as exc:
            logger.exception("Impedance command failed: %s", label)
            raise ImpedanceOperationError(stage, exc) from exc

    async def _cleanup_impedance_resources(self) -> Exception | None:
        """Release impedance mode before its callback, as required by the SDK."""

        cleanup_errors: list[Exception] = []
        if self.handle is None:
            return None
        sdk, handle = self._sdk, self.handle
        if self._impedance_transfer_started:
            try:
                await self._impedance_sdk_operation(
                    "cleanup", lambda: sdk.set_eeg_transfer(handle, False)
                )
                self._impedance_transfer_started = False
            except Exception as exc:
                return exc
        if self._impedance_mode_active:
            try:
                await self._impedance_sdk_operation(
                    "cleanup", lambda: sdk.disable_impedance_detection_mode(handle)
                )
                self._impedance_mode_active = False
            except Exception as exc:
                if self._is_already_inactive_error(exc):
                    self._impedance_mode_active = False
                else:
                    # Preserve the resource flag. A recovery retry must issue
                    # the disable command again; clearing it here would make
                    # the state machine claim a physical cleanup it never
                    # verified.
                    cleanup_errors.append(exc)

        # The SDK requires mode shutdown before callback removal. If mode
        # shutdown is uncertain, retain both resources for a later retry.
        if not self._impedance_mode_active and self._impedance_callback_registered:
            try:
                await self._impedance_sdk_operation(
                    "cleanup", lambda: sdk.clear_impedance_callback(handle)
                )
                self._impedance_callback_registered = False
            except Exception as exc:
                cleanup_errors.append(exc)

        if not cleanup_errors:
            return None
        return RuntimeError("; ".join(str(error) for error in cleanup_errors))

    @staticmethod
    def _is_already_inactive_error(error: Exception) -> bool:
        message = str(error).casefold()
        return any(
            marker in message
            for marker in (
                "already disabled",
                "already stopped",
                "not active",
                "not subscribed",
                "no active subscription",
                "已关闭",
                "未订阅",
            )
        )

    async def _retry_eeg_cleanup_operation(self, operation: str, callback: Any) -> None:
        """Retry idempotent EEG cleanup before declaring hardware state uncertain."""

        last_error: Exception | None = None
        for attempt in range(1, PREVIEW_CLEANUP_RETRY_COUNT + 1):
            try:
                await self._run_sdk_cleanup_operation(operation, callback)
                return
            except Exception as exc:
                if self._is_already_inactive_error(exc):
                    return
                if self._sdk_worker_done is not None and not self._sdk_worker_done.is_set():
                    raise  # The timed-out command still owns the handle; do not retry it.
                last_error = exc
                if attempt < PREVIEW_CLEANUP_RETRY_COUNT:
                    await asyncio.sleep(PREVIEW_CLEANUP_RETRY_DELAY_SECONDS)
        assert last_error is not None
        raise RuntimeError(
            f"{operation}连续 {PREVIEW_CLEANUP_RETRY_COUNT} 次失败: {last_error}"
        ) from last_error

    @staticmethod
    def _settle_worker_future(
        future: asyncio.Future, error: BaseException | None, result: Any
    ) -> None:
        """Complete the caller's future; runs on the event-loop thread."""

        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    @staticmethod
    def _is_loop_binding_error(error: BaseException) -> bool:
        """Detect SDK awaitables that refuse to run on a foreign event loop."""

        message = str(error).casefold()
        return any(
            marker in message
            for marker in (
                "attached to a different loop",
                "different event loop",
                "no running event loop",
                "event loop is closed",
            )
        )

    def _require_sdk_worker_idle(self) -> None:
        if self.connection_pending and self._connection_operation is not asyncio.current_task():
            raise RuntimeError("连接原生请求尚未结束，暂不能开始其他设备操作。")
        if self._sdk_worker_done is not None and not self._sdk_worker_done.is_set():
            raise RuntimeError("上一条原生设备命令尚未结束，暂不能重试或启动新设备操作。")

    async def _run_sdk_cleanup_operation(
        self, operation: str, request: Any, *, timeout_seconds: float | None = None,
        allow_loop_fallback: bool = True,
    ) -> None:
        """Bound SDK shutdown calls so a device failure cannot hang the UI.

        The native SDK can block synchronously inside a stop command (for
        example a stalled BLE stack). ``asyncio.wait_for`` on the UI loop can
        only cancel a *suspended* awaitable — a call that is already executing
        would freeze the qasync loop and, with it, every Qt control. The call
        therefore runs on a daemon worker thread that drives its own event
        loop; the UI side merely awaits the settle future under the usual
        timeout. If the SDK object is bound to the caller's loop and rejects
        the worker loop, the call is retried inline (legacy mock-SDK path).
        """

        self._require_sdk_worker_idle()
        timeout = SDK_CLEANUP_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        is_factory = callable(request) and not inspect.isawaitable(request)

        def factory() -> Any:
            return request() if is_factory else request

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        finished = threading.Event()
        self._sdk_worker_done = finished

        def worker() -> None:
            error: BaseException | None = None
            result: Any = None
            try:
                outcome = factory()
                if inspect.isawaitable(outcome):

                    async def drive() -> Any:
                        return await outcome

                    result = asyncio.run(drive())
                else:
                    result = outcome
            except BaseException as exc:  # relayed verbatim to the caller
                error = exc
            finally:
                # A caller timeout only cancels its receipt. The native worker
                # may still mutate this handle, so track its actual lifetime.
                finished.set()
            try:
                loop.call_soon_threadsafe(
                    self._settle_worker_future, future, error, result
                )
            except RuntimeError:
                # Event loop already closed during app teardown; the daemon
                # worker's outcome is no longer needed.
                pass

        threading.Thread(
            target=worker, name=f"brainsync-sdk-{operation}", daemon=True
        ).start()

        try:
            await asyncio.wait_for(future, timeout=timeout)
            return
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{operation}超时（>{timeout:.1f} 秒）"
            ) from exc
        except Exception as exc:
            if not (allow_loop_fallback and is_factory and self._is_loop_binding_error(exc)):
                raise
        # The SDK object rejected the worker's event loop. Build a fresh
        # awaitable from the factory and run it on the caller's loop.
        try:
            finished.clear()
            try:
                fallback = asyncio.ensure_future(self._maybe_await(factory()))
            except BaseException:
                finished.set()
                raise

            def fallback_finished(task):
                if not task.cancelled():
                    task.exception()  # Retrieve late failures after a caller timeout.
                finished.set()

            fallback.add_done_callback(fallback_finished)
            await asyncio.wait_for(asyncio.shield(fallback), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{operation}超时（>{timeout:.1f} 秒）"
            ) from exc

    async def loss_stats(self) -> dict[str, Any]:
        if self.handle is None:
            return {}
        stats = await self._sdk.get_eeg_loss_stats(self.handle)
        if isinstance(stats, dict):
            previous = self.snapshot.packets
            received = int(stats.get("total_received", previous))
            lost = int(stats.get("total_lost", self.snapshot.lost_packets))
            if received < 0 or lost < 0:
                raise ValueError(f"SDK 返回了无效的丢包统计: {stats!r}")
            # SDK counters can reset shortly after a transfer starts. UI
            # callback counts are already scoped to this transfer and must not
            # move backwards when that reset completes.
            self.snapshot.packets = max(previous, received)
            self.snapshot.lost_packets = max(self.snapshot.lost_packets, lost)
            if received > previous:
                self.snapshot.last_eeg_monotonic = time.monotonic()
            self.signals.state_changed.emit(self.snapshot)
        return stats if isinstance(stats, dict) else {"raw": str(stats)}

    async def disconnect(self) -> None:
        if self.connection_pending:
            self._connection_abort = True
        await self._wait_connection_operation()
        await self._cancel_recording_start()
        self._require_sdk_worker_idle()
        if self.handle is None:
            return
        # A late native subscription may not land after we close or reuse its
        # device handle. Keep recovery explicit while that request is pending.
        await self._wait_recording_start_cleanup()
        errors: list[Exception] = []
        self._disconnect_requested = True
        try:
            if (self.lifecycle.state is DeviceState.RECORDING or self._recording_transfer_started
                    or self._local_edf_recorder is not None):
                try:
                    await self.stop_recording()
                except Exception as exc:
                    errors.append(exc)
            if self.lifecycle.state is DeviceState.IMPEDANCE or self._impedance_callback_registered or self._impedance_mode_active:
                try:
                    await self.stop_impedance()
                except Exception as exc:
                    errors.append(exc)
            if self.lifecycle.state is DeviceState.PREVIEWING or self._eeg_callback_registered:
                try:
                    await self.stop_stream()
                except Exception as exc:
                    errors.append(exc)
        finally:
            try:
                self._require_sdk_worker_idle()
                await self._run_sdk_cleanup_operation(
                    "关闭设备",
                    lambda: self._sdk.close_device(self.handle),
                )
            except Exception as exc:
                errors.append(exc)
                self._disconnect_requested = False
                self._enter_recovery(exc)
                raise RuntimeError("设备句柄尚未确认释放，断开需要稍后重试。") from exc
            self.handle = None
            self._sdk = None
            self._gain_enum = None
            self._gain_value = 24.0
            self._eeg_unit_guard_triggered = False
            self._eeg_callback_registered = False
            self._impedance_callback_registered = False
            self._impedance_mode_active = False
            self._impedance_transfer_started = False
            self._impedance_data_received = False
            self._impedance_controller_task = None
            self._recording_transfer_started = False
            self._recording_started_monotonic = None
            self._disconnect_requested = False
            self._ble_connection_state = ""
            self._ble_device_id = ""
            self._eeg_readback_available = None
            self.snapshot = DeviceSnapshot()
            self.lifecycle.force_disconnected()
            self.signals.state_changed.emit(self.snapshot)
        if errors:
            raise RuntimeError("设备清理失败: " + "; ".join(str(error) for error in errors)) from errors[0]

    def _on_eeg_batch(self, packets: list[Any]) -> None:
        receive_ns = time.monotonic_ns()
        metadata = []
        for packet in packets:
            entry = {}
            for name in ("seq_num", "delta_time"):
                value = getattr(packet, name, None)
                if value is not None:
                    try:
                        value = operator.index(value) if not isinstance(value, bool) else -1
                    except TypeError:
                        value = str(value)
                entry[name] = value
            metadata.append(entry)
        rows: list[list[float]] = []
        for index, packet in enumerate(packets):
            try:
                rows.append(self._convert_eeg_packet(packet))
            except Exception as exc:
                reason = f"EEG 数据包转换失败 (row {index}): {exc}"
                self._record_acquisition_fault(metadata, receive_ns, reason)
                return
        if rows:
            try:
                data = np.asarray(rows, dtype=np.float64)
                if data.ndim != 2 or data.shape[1] != 8:
                    raise ValueError(f"数据形状为 {data.shape}，预期为 (样本数, 8)")
                if not np.isfinite(data).all():
                    raise ValueError("数据包含 NaN 或无穷值")
            except (TypeError, ValueError) as exc:
                self._record_acquisition_fault(metadata, receive_ns, f"EEG 数据批次形状无效: {exc}")
                return
            if self._local_edf_recorder is not None:
                try:
                    self._local_edf_recorder.append(
                        data, packet_metadata=metadata, receive_monotonic_ns=receive_ns,
                    )
                except Exception as exc:
                    reason = f"本地 EDF 数据写入缓冲失败: {exc}"
                    self.signals.error.emit(reason)
                    self.signals.acquisition_fault.emit(reason)
                    self.checkpoint_recording()
                    return
            # Count the batch only after it is safely buffered/recorded, so
            # the readiness poll in _wait_for_recording_data cannot treat a
            # dropped batch as received data.
            self.snapshot.packets += len(rows)
            self.snapshot.last_eeg_monotonic = time.monotonic()
            self.signals.eeg_batch.emit(data)

    def _record_acquisition_fault(self, metadata: list[dict], receive_ns: int, reason: str) -> None:
        self.signals.error.emit(reason)
        if self._local_edf_recorder is not None:
            try:
                self._local_edf_recorder.reject_batch(metadata, receive_ns, reason)
                self._local_edf_recorder.checkpoint(force=True)
            except Exception as error:
                self.signals.error.emit(f"EEG clock journal failure: {error}")
            self.signals.acquisition_fault.emit(reason)

    def _convert_eeg_packet(self, packet: Any) -> list[float]:
        """Return one EEG packet in microvolts, never raw ADC units."""

        gain_enum = self._gain_enum
        if gain_enum is None and self._sdk is not None:
            gain_enum = getattr(getattr(self._sdk, "EegGain", None), "Gain24", None)

        adc_values = getattr(packet, "adc_values", None)
        manual = self._adc_to_microvolts(adc_values) if adc_values is not None else None
        converted: list[float] | None = None

        to_microvolts = getattr(packet, "to_microvolts", None)
        if callable(to_microvolts) and gain_enum is not None:
            try:
                converted = [float(value) for value in to_microvolts(gain_enum)]
            except Exception:
                # Some SDK revisions expose the method but fail to convert a
                # packet shape that can still be handled from adc_values.
                pass

        if converted is None:
            to_volts = getattr(packet, "to_volts", None)
            if callable(to_volts) and gain_enum is not None:
                try:
                    converted = [float(value) * 1_000_000.0 for value in to_volts(gain_enum)]
                except Exception:
                    pass

        if converted is not None:
            return self._correct_eeg_units_per_channel(converted, manual) if manual is not None else converted
        if manual is not None:
            return manual
        raise ValueError("SDK 数据包没有可转换的 EEG 通道字段")

    def _correct_eeg_units_per_channel(
        self, converted: list[float], manual: list[float]
    ) -> list[float]:
        """Correct only channels whose SDK conversion is clearly in raw ADC units."""

        if len(converted) != len(manual):
            raise ValueError(
                f"SDK 转换通道数 {len(converted)} 与 ADC 通道数 {len(manual)} 不一致"
            )
        corrected = list(converted)
        corrected_channels: list[int] = []
        for index, (sdk_value, adc_value) in enumerate(zip(converted, manual)):
            if self._is_channel_conversion_error(sdk_value, adc_value):
                corrected[index] = adc_value
                corrected_channels.append(index + 1)
        if corrected_channels and not self._eeg_unit_guard_triggered:
            self._eeg_unit_guard_triggered = True
            labels = ", ".join(f"CH{index}" for index in corrected_channels)
            self.signals.error.emit(
                f"SDK EEG 转换结果与 ADC 校准值严重不符（{labels}），已按 ADC 校准公式修正为 μV。"
            )
        return corrected

    def _adc_to_microvolts(self, adc_values: Any) -> list[float]:
        gain = float(self._gain_value)
        if gain <= 0:
            raise ValueError(f"EEG 增益无效: {gain}")
        return [
            (float(value) / EEG_ADC_FULL_SCALE)
            * (EEG_REFERENCE_VOLTAGE / gain)
            * 1_000_000.0
            for value in adc_values
        ]

    @staticmethod
    def _is_channel_conversion_error(converted_value: float, manual_value: float) -> bool:
        if not np.isfinite(converted_value) or not np.isfinite(manual_value):
            return not np.isfinite(converted_value) and np.isfinite(manual_value)
        if abs(manual_value) <= 1e-6:
            return abs(converted_value) >= 10_000.0
        ratio = abs(converted_value / manual_value)
        # A normal SDK calibration may differ slightly from the fallback
        # reference voltage. A 10x+ difference paired with a 10 mV+ display
        # value is raw ADC counts or a unit conversion fault, not physiology.
        return ratio >= 10.0 and abs(converted_value) >= 10_000.0

    def _on_impedance(self, data: Any) -> None:
        try:
            values = [float(v) for v in data.impedances]
            if not values:
                raise ValueError("设备返回了空的阻抗通道列表")
            self._impedance_data_received = True
            self._impedance_last_data_monotonic = time.monotonic()
            self.signals.impedance.emit(values)
        except Exception as exc:
            self.signals.error.emit(f"阻抗数据解析失败: {exc}")

    def _require_handle(self) -> None:
        if self.handle is None:
            raise RuntimeError("设备尚未连接。")

    async def _reset_loss_stats_after_start(self) -> None:
        """Reset startup sequence statistics after the first transfer window."""

        await asyncio.sleep(EEG_STATS_STARTUP_SETTLE_SECONDS)
        if self.handle is None or self.state not in {DeviceState.PREVIEWING, DeviceState.RECORDING}:
            return
        reset_loss = getattr(self._sdk, "reset_eeg_loss_stats", None)
        if reset_loss is None:
            return
        try:
            await self._maybe_await(reset_loss(self.handle))
            self.snapshot.packets = 0
            self.snapshot.lost_packets = 0
            self.snapshot.last_eeg_monotonic = time.monotonic()
            self.signals.state_changed.emit(self.snapshot)
        except Exception as exc:
            self.signals.error.emit(f"重置 EEG 丢包统计失败: {exc}")

    async def _reset_loss_stats_task(self) -> None:
        try:
            await self._reset_loss_stats_after_start()
        finally:
            if self._loss_reset_task is asyncio.current_task():
                self._loss_reset_task = None

    async def _cancel_loss_stats_reset(self) -> None:
        task = self._loss_reset_task
        if task is None:
            return
        self._loss_reset_task = None
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value

    @staticmethod
    def _format_firmware(info: Any) -> str:
        if isinstance(info, dict):
            sw = info.get("sw_version", "--")
            hw = info.get("hw_version", "--")
            return f"SW {sw} / HW {hw}"
        return str(info)


class MockDeviceAdapter(DeviceAdapter):
    """Offline device with the same signals and lifecycle as the real adapter."""

    def __init__(self) -> None:
        super().__init__()
        self._mock_task: asyncio.Task | None = None
        self._impedance_task: asyncio.Task | None = None
        self._mock_stream_started: float | None = None
        self._mock_samples_generated = 0
        self._mock_rng = np.random.default_rng(7)

    async def connect(self, port_name: str | None = None, transport: str = "serial") -> dict[str, Any]:
        self.lifecycle.require("连接设备", DeviceState.DISCONNECTED)
        self.handle = "MOCK_DEVICE"
        self.snapshot.transport = "mock"
        self.snapshot.firmware = "SIMULATOR / UI TEST"
        self._set_state(DeviceState.READY)
        return {"device_type": "BrainSync Mock", "sw_version": "sim", "hw_version": "sim"}

    async def configure(self, sample_rate: int = DEFAULT_SAMPLE_RATE_HZ, gain: int = 24) -> None:
        self.lifecycle.require("修改采样率或增益", DeviceState.READY)
        validate_acquisition_rate(sample_rate)
        self.snapshot.sample_rate_hz = sample_rate
        self.snapshot.gain = f"Gain{gain}"
        self.snapshot.configuration_verification = "mock"
        self.snapshot.configuration_note = "offline simulator"
        self._set_state(DeviceState.READY)

    async def start_stream(self, batch_size: int = EEG_CALLBACK_BATCH_SIZE) -> None:
        self.lifecycle.require("启动 EEG 预览", DeviceState.READY)
        if batch_size <= 0 or batch_size % EEG_CALLBACK_BATCH_SIZE:
            raise ValueError(
                f"SDK EEG batch_size 必须是 {EEG_CALLBACK_BATCH_SIZE} 的正整数倍。"
            )
        self.snapshot.stream_started_monotonic = time.monotonic()
        self._reset_mock_sample_clock(self.snapshot.stream_started_monotonic)
        self._set_state(DeviceState.PREVIEWING)
        if self._mock_task is None:
            self._mock_task = asyncio.create_task(self._stream_loop())

    async def stop_stream(self) -> None:
        self.lifecycle.require("停止 EEG 预览", DeviceState.PREVIEWING)
        self.snapshot.stream_started_monotonic = None
        if self._mock_task is not None:
            task = self._mock_task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._mock_task = None
        self._set_state(DeviceState.READY)

    async def start_recording(
        self,
        output_dir: str,
        subject_id: str,
        channel_config: Any = None,
        sex: str = "X",
        birthdate: str = EDF_UNKNOWN_BIRTHDATE,
        participant_name: str = "BrainSync GTN",
    ) -> str:
        self.lifecycle.require("启动 EDF 录制", DeviceState.READY)
        if self._mock_task is not None:
            await self.stop_stream()
        labels = getattr(channel_config, "labels", None) or list(ChannelMontage().eeg)
        self._local_edf_recorder = LocalEdfRecorder(
            output_dir, labels, self.snapshot.sample_rate_hz, subject_id,
            sex, birthdate, participant_name,
        )
        self._local_edf_recorder.add_annotation(0.0, "SIMULATED_DATA|NO_HARDWARE")
        self._recording_transfer_started = True
        self._recording_started_monotonic = time.monotonic()
        self.snapshot.stream_started_monotonic = self._recording_started_monotonic
        self._reset_mock_sample_clock(self._recording_started_monotonic)
        self._set_state(DeviceState.RECORDING)
        if self._mock_task is None:
            self._mock_task = asyncio.create_task(self._stream_loop())
        return str(self._local_edf_recorder.path)

    async def stop_recording(self) -> None:
        async with self._recording_operation_lock:
            self.lifecycle.require("停止 EDF 录制", DeviceState.RECORDING, DeviceState.RECOVERY_REQUIRED)
            if self._mock_task is not None:
                task = self._mock_task
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._mock_task = None
                self._append_mock_samples(time.monotonic())
            self._recording_transfer_started = False
            recorder = self._local_edf_recorder
            try:
                if recorder is not None:
                    await asyncio.wait_for(
                        asyncio.to_thread(recorder.finalize), timeout=EDF_FINALIZE_TIMEOUT_SECONDS
                    )
            except Exception as error:
                self._enter_recovery(error)
                raise
            self._last_recording_clock = None if recorder is None else recorder.clock_record()
            self._local_edf_recorder = None
            self._recording_started_monotonic = None
            self.snapshot.stream_started_monotonic = None
            self._set_state(DeviceState.READY)

    async def add_annotation(self, onset_seconds: float, description: str) -> None:
        if self._local_edf_recorder is not None:
            self._local_edf_recorder.add_annotation(onset_seconds, description)

    async def run_impedance(self, duration_sec: int | None = None) -> None:
        if duration_sec is not None and int(duration_sec) <= 0:
            raise ValueError("阻抗检测时长必须大于 0 秒。")
        self.lifecycle.require("启动阻抗检测", DeviceState.READY)
        self._set_state(DeviceState.IMPEDANCE)
        self._impedance_task = asyncio.create_task(self._impedance_loop(duration_sec))
        try:
            await self._impedance_task
        finally:
            self._impedance_task = None
            self._set_state(DeviceState.READY)

    async def stop_impedance(self) -> None:
        self.lifecycle.require("停止阻抗检测", DeviceState.IMPEDANCE)
        if self._impedance_task is not None:
            task = self._impedance_task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._impedance_task = None
        self._set_state(DeviceState.READY)

    async def loss_stats(self) -> dict[str, Any]:
        return {"total_received": self.snapshot.packets, "total_lost": 0, "loss_rate": 0.0}

    async def disconnect(self) -> None:
        if self.lifecycle.state is DeviceState.RECORDING or self.recovery_may_have_recording:
            await self.stop_recording()
        if self.lifecycle.state is DeviceState.IMPEDANCE:
            await self.stop_impedance()
        if self.lifecycle.state is DeviceState.PREVIEWING:
            await self.stop_stream()
        self.snapshot = DeviceSnapshot()
        self.handle = None
        self.lifecycle.force_disconnected()
        self.signals.state_changed.emit(self.snapshot)

    def _reset_mock_sample_clock(self, started: float) -> None:
        self._mock_stream_started = started
        self._mock_samples_generated = 0
        self._mock_rng = np.random.default_rng(7)

    def _append_mock_samples(self, now: float) -> None:
        if self._mock_stream_started is None:
            return
        rate = float(self.snapshot.sample_rate_hz)
        due = int(max(0.0, now - self._mock_stream_started) * rate)
        count = due - self._mock_samples_generated
        if count <= 0:
            return
        times = np.arange(self._mock_samples_generated, due, dtype=np.float64)[:, None] / rate
        channels = np.arange(LocalEdfRecorder.CHANNEL_COUNT, dtype=np.float64)[None, :]
        samples = (
            18.0 * np.sin(2 * np.pi * 10 * times + channels * 0.22)
            + 4.0 * np.sin(2 * np.pi * 1.2 * times)
            + self._mock_rng.normal(0, 2.2, size=(count, LocalEdfRecorder.CHANNEL_COUNT))
        )
        if self._local_edf_recorder is not None:
            self._local_edf_recorder.append(samples, packet_metadata=[
                {"seq_num": index % 65536, "delta_time": round(100000 / rate)}
                for index in range(self._mock_samples_generated, due)
            ], receive_monotonic_ns=time.monotonic_ns())
        self._mock_samples_generated = due
        self.snapshot.packets += count
        self.snapshot.last_eeg_monotonic = now
        self.signals.eeg_batch.emit(samples)

    async def _stream_loop(self) -> None:
        while True:
            await asyncio.sleep(0.04)
            self._append_mock_samples(time.monotonic())

    async def _impedance_loop(self, duration_sec: int | None) -> None:
        step = 0
        while duration_sec is None or step < max(1, int(duration_sec)):
            values = [
                2.6 + 0.7 * np.sin(step * 0.4 + idx * 0.6) + (idx % 4) * 0.45
                for idx in range(8)
            ]
            self.signals.impedance.emit(values)
            await asyncio.sleep(1.0)
            step += 1


def make_channel_config(sdk: Any, montage: ChannelMontage | None = None) -> Any:
    """Build channel metadata without coupling the UI to SDK internals."""
    montage = montage or ChannelMontage()
    montage.validate()
    return sdk.ChannelConfig(
        labels=list(montage.eeg),
        active_mask=montage.active_mask,
        ref_label=montage.ref_label,
        gnd_label=montage.gnd_label,
        # The SDK channel field is zero-based and follows the physical sample
        # array: channel 5 is PO8 EEG, channel 6 is PO7 EOG.
        eeg=[sdk.ElectrodeAssignment(label, index) for index, label in enumerate(montage.eeg)],
        stim=[
            sdk.StimElectrodeAssignment(
                str(item["label"]), int(item["slot"]), int(item["channel"]), str(item["polarity"])
            )
            for item in montage.stim
            if {"label", "slot", "channel", "polarity"}.issubset(item)
        ],
    )


class ExperimentController(QtCore.QObject):
    state_changed = QtCore.Signal(str)
    trial_changed = QtCore.Signal(object)
    progress_changed = QtCore.Signal(int, int)
    present_digit = QtCore.Signal(object)
    present_text = QtCore.Signal(str)
    stimulus_event = QtCore.Signal(object)
    message = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    practice_finished = QtCore.Signal(str)
    paused = QtCore.Signal()
    resuming = QtCore.Signal()

    def __init__(self, store: SessionStore, *, recording_support=None):
        super().__init__()
        self.store = store
        self.recording_support = recording_support
        self._task: asyncio.Task | None = None
        self._stop_requested = False
        self._recording_mode = False
        self._pause_requested = asyncio.Event()
        self._resume_requested = asyncio.Event()
        self.is_paused = False
        self._baseline_required = True
        self._baseline_not_before: float | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, config: ExperimentConfig) -> None:
        if self.running:
            return
        self._stop_requested = False
        self._recording_mode = True
        self._baseline_required = True
        self._baseline_not_before = None
        self.is_paused = False
        self._pause_requested.clear()
        self._resume_requested.clear()
        self._task = asyncio.create_task(self._run(config))
        self._task.add_done_callback(lambda task: self._completed(task, recording_mode=True))

    def start_practice(self, config: ExperimentConfig) -> None:
        if self.running:
            return
        self._stop_requested = False
        self._recording_mode = False
        self._pause_requested.clear()
        self.is_paused = False
        self._resume_requested.clear()
        practice = replace(config, blocks=1, digits_per_block=9, rest_sec=0)
        self._task = asyncio.create_task(self._run(practice))
        self._task.add_done_callback(lambda task: self._completed(task, recording_mode=False))

    def _completed(self, task: asyncio.Task, *, recording_mode: bool) -> None:
        """Publish completion only after scheduling and its waiters have stopped."""
        try:
            state = task.result()
        except asyncio.CancelledError:
            state = "aborted"
            if recording_mode:
                self._write_event_best_effort("protocol_end", {
                    "status": state, "completed_trials": 0, "cancelled": True,
                })
        except Exception as error:
            state = "error"
            self.message.emit(f"实验执行失败: {error}")
        self.is_paused = False
        self.present_digit.emit(None)
        self.present_text.emit("")
        self.state_changed.emit(state)
        (self.finished if recording_mode else self.practice_finished).emit(state)

    def _write_event(self, name: str, payload: dict[str, Any]) -> None:
        self.store.event(name, payload)

    def _write_event_best_effort(self, name: str, payload: dict[str, Any]) -> None:
        try:
            self.store.event(name, payload)
        except Exception as exc:
            self.message.emit(f"事件写入失败: {exc}")

    def stop(self) -> None:
        self._stop_requested = True
        if self._task is not None and not self._task.done() and self._task is not asyncio.current_task():
            self._task.cancel()

    async def wait(self) -> None:
        """Wait until scheduling and all sample-support waiters have stopped."""
        if self._task is not None and self._task is not asyncio.current_task():
            await asyncio.gather(self._task, return_exceptions=True)

    def pause(self) -> None:
        if self.running and not self.is_paused:
            self._pause_requested.set()

    def resume(self) -> None:
        if self.is_paused:
            self._resume_requested.set()

    async def _wait_or_pause(self, deadline: float) -> bool:
        if self._pause_requested.is_set():
            return True
        sleeper = asyncio.create_task(sleep_until(deadline))
        request = asyncio.create_task(self._pause_requested.wait())
        try:
            await asyncio.wait((sleeper, request), return_when=asyncio.FIRST_COMPLETED)
            if sleeper.done():
                sleeper.result()
            return self._pause_requested.is_set()
        finally:
            sleeper.cancel()
            request.cancel()
            await asyncio.gather(sleeper, request, return_exceptions=True)

    async def _rest_from_menu(self) -> None:
        self.is_paused = True
        self._resume_requested.clear()
        self.present_digit.emit(None)
        self.state_changed.emit("paused")
        self.paused.emit()
        if self._stop_requested:
            raise asyncio.CancelledError
        await self._resume_requested.wait()
        self._pause_requested.clear()
        self.is_paused = False
        self.resuming.emit()
        self._baseline_required = True
        self._baseline_not_before = None
        if self._stop_requested:
            raise asyncio.CancelledError
        self.state_changed.emit("resuming")
        for remaining in range(RESUME_COUNTDOWN_SECONDS, 0, -1):
            self.present_text.emit(f"请重新注视屏幕中央\n准备继续：{remaining}")
            if await self._wait_or_pause(time.monotonic() + 1.0):
                await self._rest_from_menu()
                return
        self.state_changed.emit("running" if self._recording_mode else "practice")

        if self._recording_mode:
            self._write_event("experiment_resume_ready", {"countdown_seconds": RESUME_COUNTDOWN_SECONDS})

    async def _wait_for_support(self, stage: str, request: dict) -> bool:
        if self.recording_support is None:
            raise RuntimeError("Recording requires an actual EEG sample-support waiter.")
        support = asyncio.create_task(self.recording_support(stage, request))
        pause = asyncio.create_task(self._pause_requested.wait())
        try:
            await asyncio.wait((support, pause), return_when=asyncio.FIRST_COMPLETED)
            if support.done():
                support.result()
            return self._pause_requested.is_set()
        finally:
            support.cancel()
            pause.cancel()
            await asyncio.gather(support, pause, return_exceptions=True)

    async def _run(self, config: ExperimentConfig) -> str:
        config.validate_protocol()
        rng = np.random.default_rng(config.random_seed)
        total = config.blocks * config.digits_per_block
        completed = 0
        active_stimulus: dict[str, Any] | None = None

        def close_active_stimulus(*, interrupted: bool) -> None:
            nonlocal active_stimulus
            if active_stimulus is None:
                return
            self.present_digit.emit(None)
            offset_payload = {
                **active_stimulus,
                "software_offset_monotonic": time.monotonic(),
            }
            if interrupted:
                offset_payload["interrupted"] = True
            if self._recording_mode:
                self._write_event_best_effort("stimulus_offset", {
                    key: value for key, value in offset_payload.items()
                    if not key.endswith("_monotonic")
                })
            self.stimulus_event.emit({"kind": "offset", **offset_payload})
            active_stimulus = None

        try:
            self.state_changed.emit("running" if self._recording_mode else "practice")
            if self._recording_mode:
                self._write_event("protocol_start", {"total_trials": total})
            self.present_text.emit(config.instruction_text.strip() or "请注视屏幕中央的数字。")
            if await self._wait_or_pause(time.monotonic() + 3.0):
                await self._rest_from_menu()
            for block in range(1, config.blocks + 1):
                if self._stop_requested:
                    break
                digits = sample_block_digits(rng, config.digits_per_block)
                if self._recording_mode:
                    self._write_event("block_start", {"block_id": block, "count": len(digits)})
                for index, digit in enumerate(digits, start=1):
                    if self._stop_requested:
                        break
                    attempt = 0
                    while True:
                        if self._pause_requested.is_set():
                            await self._rest_from_menu()
                        if self._recording_mode and self._baseline_required:
                            if await self._wait_for_support("baseline", {"not_before_monotonic": self._baseline_not_before}):
                                await self._rest_from_menu()
                                continue
                            self._baseline_required = False
                        attempt += 1
                        trial_id = f"B{block:03d}-T{index:03d}-A{attempt:03d}"
                        identity = {
                            "trial_id": trial_id, "block_id": block,
                            "trial_index": index, "digit": int(digit), "attempt": attempt,
                        }
                        if self._recording_mode:
                            self._write_event("trial_start", identity)
                        self.present_digit.emit(int(digit))
                        onset = time.monotonic()
                        active_stimulus = identity
                        if self._recording_mode:
                            self._write_event("stimulus_onset", identity)
                        self.stimulus_event.emit({
                            "kind": "onset", **identity, "software_onset_monotonic": onset,
                        })
                        self.trial_changed.emit({
                            "trial_id": trial_id, "block": block, "trial": index, "digit": int(digit),
                        })
                        interrupted = await self._wait_or_pause(onset + config.stimulus_ms / 1000.0)
                        close_active_stimulus(interrupted=interrupted)
                        if not interrupted:
                            interrupted = await self._wait_or_pause(onset + config.soa_ms / 1000.0)
                        if not interrupted and self._recording_mode and index == len(digits):
                            interrupted = await self._wait_or_pause(
                                onset + MODEL_EPOCH_SUPPORT.response_guard_seconds()
                            )
                            if not interrupted:
                                interrupted = await self._wait_for_support("response", {})
                        if not interrupted:
                            break
                        self.stimulus_event.emit({
                            "kind": "invalidated", **identity,
                            "software_invalidated_monotonic": time.monotonic(),
                            "reason": "operator_pause", "valid": False,
                        })
                        await self._rest_from_menu()
                    completed += 1
                    self.progress_changed.emit(completed, total)
                if self._recording_mode:
                    self._write_event("block_end", {"block_id": block, "completed_trials": completed})
                if block < config.blocks and not self._stop_requested:
                    self.state_changed.emit("rest")
                    rest_start = time.monotonic()
                    rest_payload = {
                        "block_id": block,
                        "software_rest_start_monotonic": rest_start,
                    }
                    if self._recording_mode:
                        self._write_event("rest_start", {"block_id": block})
                    self.stimulus_event.emit({"kind": "rest_start", **rest_payload})
                    for elapsed, remaining in enumerate(
                        range(max(0, config.rest_sec), 0, -1), start=1
                    ):
                        self.present_text.emit(f"休息\n{remaining} 秒")
                        if await self._wait_or_pause(time.monotonic() + 1.0):
                            await self._rest_from_menu()
                    rest_end = time.monotonic()
                    rest_end_payload = {
                        "block_id": block,
                        "software_rest_end_monotonic": rest_end,
                    }
                    if self._recording_mode:
                        self._write_event("rest_end", {"block_id": block})
                    self.stimulus_event.emit({"kind": "rest_end", **rest_end_payload})
                    if self._recording_mode:
                        self.present_digit.emit(None)
                        self._baseline_required = True
                        self._baseline_not_before = rest_end
                        if await self._wait_or_pause(
                            rest_end + MODEL_EPOCH_SUPPORT.baseline_guard_seconds()
                        ):
                            await self._rest_from_menu()
                    self.state_changed.emit("running")
            final_state = "aborted" if self._stop_requested else "completed"
            if self._recording_mode:
                self._write_event("protocol_end", {"status": final_state, "completed_trials": completed})
            return final_state
        except asyncio.CancelledError:
            self.is_paused = False
            close_active_stimulus(interrupted=True)
            if self._recording_mode:
                self._write_event_best_effort("protocol_end", {
                    "status": "aborted",
                    "completed_trials": completed,
                    "cancelled": True,
                })
            return "aborted"
        except Exception as exc:
            close_active_stimulus(interrupted=True)
            if self._recording_mode:
                self._write_event_best_effort("protocol_end", {
                    "status": "error",
                    "completed_trials": completed,
                    "error": str(exc),
                })
            self.message.emit(f"实验执行失败: {exc}")
            return "error"


class ModelJobRunner(QtCore.QObject):
    progress_changed = QtCore.Signal(int)
    output_line = QtCore.Signal(str)
    state_changed = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, store: SessionStore):
        super().__init__()
        self.store = store
        self._task: asyncio.Task | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._tree_kill_task: asyncio.Task | None = None
        self._stop_requested = False
        self.closed = False
        self._shutdown_task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def cancel_requested(self) -> bool:
        return self._stop_requested

    def start(
        self,
        config: ModelTaskConfig,
        session_dir: Path | None,
        mock: bool = False,
        base_dir: Path | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("Algorithm runner is closed.")
        if self.running:
            return
        self._stop_requested = False
        self._task = asyncio.create_task(self._run(config, session_dir, mock, base_dir))
        self._task.add_done_callback(self._completed)

    def start_backend(
        self,
        program: str,
        arguments: list[str],
        *,
        task: str,
        output_dir: Path,
    ) -> None:
        """Start a structured backend command whose CLI owns its input contract."""
        if self.closed:
            raise RuntimeError("Algorithm runner is closed.")
        if self.running:
            raise RuntimeError("An algorithm task is already running.")
        executable = Path(program).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Backend executable does not exist: {executable}")
        if not isinstance(arguments, list) or any(
            not isinstance(value, str) or "\x00" in value for value in arguments
        ):
            raise ValueError("Backend arguments must be a list of strings without NUL bytes.")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Backend task must be a non-empty name.")
        self._stop_requested = False
        self._task = asyncio.create_task(
            self._run_backend([str(executable), *arguments], task, Path(output_dir).resolve())
        )
        self._task.add_done_callback(self._completed)

    def _completed(self, task: asyncio.Task) -> None:
        """A finished subscriber may immediately launch the next job."""
        try:
            state, ok, message = task.result()
        except asyncio.CancelledError:
            state, ok, message = "stopped", False, "算法任务已停止"
        except Exception as error:
            state, ok, message = "failed", False, f"算法任务失败: {error}"
        self.state_changed.emit(state)
        self.finished.emit(ok, message)

    async def wait(self) -> None:
        if self._task is not None and self._task is not asyncio.current_task():
            await asyncio.shield(self._task)

    async def shutdown(self) -> None:
        """Reject new work and reap the complete process tree before loop closure."""
        self.closed = True
        if self._shutdown_task is None:
            self.stop()
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=3.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    async def _run_backend(self, command: list[str], task: str, output_dir: Path) -> tuple[str, bool, str]:
        record = {
            "schema": "brainsync-backend-task/1",
            "task": task,
            "command": command,
            "working_directory": str(output_dir),
            "output_dir": str(output_dir),
        }
        log_file = None

        def record_status(status: str, extra: dict[str, Any] | None = None) -> None:
            record.update({"status": status, **(extra or {})})
            atomic_write_json(output_dir / "backend_task.json", record)

        def emit_line(line: str) -> None:
            if log_file is not None:
                log_file.write(line + "\n")
                log_file.flush()
            self.output_line.emit(line)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = (output_dir / "backend.log").open("w", encoding="utf-8")
            self.state_changed.emit("running")
            env = {key: value for key, value in os.environ.items() if not key.startswith("BRAIN_SYNC_")}
            for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
                env.pop(key, None)
            env["PYTHONNOUSERSITE"] = "1"
            env["BRAIN_SYNC_TASK"] = task
            env["BRAIN_SYNC_OUTPUT_DIR"] = str(output_dir)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            code = await self._execute_command(command, str(output_dir), env, emit_line, record_status)
            state = "stopped" if self._stop_requested else ("completed" if code == 0 else "failed")
            record_status(state, {"returncode": code})
            return state, state == "completed", f"算法任务已结束，返回码 {code}"
        except asyncio.CancelledError:
            record_status("stopped", {"cancelled": True})
            return "stopped", False, "算法任务已停止"
        except Exception as error:
            try:
                record_status("failed", {"error": str(error)})
            except OSError:
                pass
            return "failed", False, f"算法任务失败: {error}"
        finally:
            if log_file is not None:
                log_file.close()

    def stop(self) -> None:
        self._stop_requested = True
        process = self._process
        if process is None:
            # The task may still be validating its manifest or resolving its
            # executable. _run() checks this flag immediately before and after
            # process creation, so a stop request cannot be lost in that gap.
            return
        if os.name == "nt":
            if self._tree_kill_task is None or self._tree_kill_task.done():
                self._tree_kill_task = asyncio.create_task(self._kill_process_tree(process.pid))
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        """Terminate the child and its descendants, then wait for reaping."""

        if os.name == "nt":
            if self._tree_kill_task is None or self._tree_kill_task.done():
                self._tree_kill_task = asyncio.create_task(self._kill_process_tree(process.pid))
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if self._tree_kill_task is not None:
            await asyncio.gather(self._tree_kill_task, return_exceptions=True)
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    async def _kill_process_tree(self, pid: int) -> None:
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(pid), "/T", "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        except (OSError, asyncio.CancelledError):
            pass

    async def _execute_command(
        self,
        command: list[str],
        cwd: str,
        env: dict[str, str],
        emit_line: Any,
        record_status: Any,
    ) -> int | None:
        try:
            if self._stop_requested:
                return None
            emit_line("$ " + subprocess.list2cmdline(command))
            process_options: dict[str, Any] = {}
            if os.name == "nt":
                process_options["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            else:
                process_options["start_new_session"] = True
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **process_options,
            )
            if self._stop_requested:
                await self._terminate_process(self._process)
                return self._process.returncode
            record_status("started")
            assert self._process.stdout is not None
            async for raw_line in self._process.stdout:
                line = raw_line.decode(errors="replace").rstrip()
                emit_line(line)
                if line.startswith("PROGRESS "):
                    try:
                        self.progress_changed.emit(max(0, min(100, int(line.split()[1]))))
                    except (IndexError, ValueError):
                        pass
            return await self._process.wait()
        except BaseException:
            if self._process is not None and self._process.returncode is None:
                await self._terminate_process(self._process)
            raise
        finally:
            if self._tree_kill_task is not None:
                await asyncio.gather(self._tree_kill_task, return_exceptions=True)
                self._tree_kill_task = None
            self._process = None

    async def _run(
        self,
        config: ModelTaskConfig,
        session_dir: Path | None,
        mock: bool,
        base_dir: Path | None,
    ) -> tuple[str, bool, str]:
        log_file = None
        if session_dir is not None:
            session_dir = session_dir.resolve()
        task_extra: dict[str, Any] = {"session_dir": str(session_dir or "")}

        def record_status(status: str, extra: dict[str, Any] | None = None) -> None:
            try:
                self.store.model_task(config, status, {**task_extra, **(extra or {})}, session_dir=session_dir)
            except Exception as exc:
                self.output_line.emit(f"模型任务状态写入失败: {exc}")

        try:
            self.state_changed.emit("running")
            if config.task not in MODEL_TASKS:
                raise ValueError(f"不支持的算法任务类型: {config.task!r}")
            recording_path: Path | None = None
            session_payload: dict[str, Any] | None = None
            if session_dir is not None:
                session_payload, recording_path = load_model_input_session(session_dir)
                task_extra["recording_path"] = str(recording_path)
                task_extra["session_schema"] = session_payload["schema"]
            output_dir = Path(config.output_dir).expanduser()
            if config.task in {"infer", "evaluate"} and config.output_dir.strip() == "models/runs" and session_dir is not None:
                output_dir = session_dir / "analysis" / "model_outputs"
            if not output_dir.is_absolute():
                output_dir = (base_dir or Path.cwd()) / output_dir
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            if session_dir is not None:
                log_dir = session_dir / "analysis" / "model_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = (log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{config.task}.log").open("a", encoding="utf-8")

            task_extra["output_dir"] = str(output_dir)

            def emit_line(line: str) -> None:
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()
                self.output_line.emit(line)

            if mock or config.dry_run:
                record_status("started", {"mode": "mock"})
                for value in range(0, 101, 10):
                    if self._stop_requested:
                        record_status("stopped", {"mode": "mock"})
                        return "stopped", False, "模拟模型任务已停止"
                    await asyncio.sleep(0.12)
                    self.progress_changed.emit(value)
                    emit_line(f"[模拟任务] progress={value}%")
                record_status("completed", {"mode": "mock"})
                return "completed", True, "模拟模型任务完成"
            executable = config.python_executable.strip() or sys.executable
            executable_path = Path(executable).expanduser()
            if not executable_path.is_absolute():
                executable_path = (base_dir or Path.cwd()) / executable_path
            executable_path = executable_path.resolve()
            if config.python_executable.strip() and not executable_path.is_file():
                raise FileNotFoundError(f"Python 解释器不存在: {executable_path}")
            executable = str(executable_path)
            manifest_path: Path | None = None
            manifest_payload: dict[str, Any] | None = None
            if config.model_manifest.strip():
                manifest_path = Path(config.model_manifest).expanduser()
                if not manifest_path.is_absolute():
                    manifest_path = (base_dir or Path.cwd()) / manifest_path
                manifest_path = manifest_path.resolve()
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"模型 manifest 不存在: {manifest_path}")
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest_payload, dict):
                    raise ValueError("模型 manifest 顶层必须是 JSON 对象")
                if manifest_payload.get("schema") != "brainsync-model-manifest/1":
                    raise ValueError("模型 manifest schema 不受支持")
                declared_tasks = manifest_payload.get("tasks")
                if declared_tasks is not None:
                    if not isinstance(declared_tasks, list) or any(not isinstance(item, str) for item in declared_tasks):
                        raise ValueError("模型 manifest 的 tasks 必须是字符串数组")
                    if config.task not in declared_tasks:
                        raise ValueError(f"manifest 未声明任务 {config.task!r}")
            entrypoint_text = config.entrypoint.strip()
            if not entrypoint_text and manifest_payload is not None:
                manifest_entrypoint = manifest_payload.get("entrypoint", "")
                if not isinstance(manifest_entrypoint, str):
                    raise ValueError("模型 manifest 的 entrypoint 必须是字符串")
                entrypoint_text = manifest_entrypoint.strip()
            if not entrypoint_text:
                message = "尚未配置算法 entrypoint；当前只建立通用任务接口。"
                emit_line(message)
                record_status("not_configured")
                return "ready", False, message
            entrypoint = Path(entrypoint_text).expanduser()
            if not entrypoint.is_absolute():
                entrypoint = (
                    manifest_path.parent
                    if manifest_path is not None
                    else (base_dir or Path.cwd())
                ) / entrypoint
                entrypoint = entrypoint.resolve()
            if entrypoint.suffix.lower() == ".py" or entrypoint.exists():
                if not entrypoint.is_file():
                    raise FileNotFoundError(f"算法脚本不存在: {entrypoint}")
                command = [executable, str(entrypoint)]
                cwd = str(entrypoint.parent)
            else:
                command = [executable, "-m", entrypoint_text]
                cwd = str(base_dir or Path.cwd())
            replacements = {
                "{session_dir}": str(session_dir or ""),
                "{output_dir}": str(output_dir),
                "{task}": config.task,
                "{manifest}": str(manifest_path or ""),
            }
            command.extend(expand_process_arguments(config.arguments, replacements))
            env = os.environ.copy()
            if session_dir is not None:
                env["BRAIN_SYNC_SESSION_DIR"] = str(session_dir)
                env["BRAIN_SYNC_SESSION_MANIFEST"] = str(session_dir / "session.json")
                recording = session_payload["recording"]
                bids = recording["bids"]
                env["BRAIN_SYNC_BIDS_ROOT"] = str(session_dir)
                env["BRAIN_SYNC_EVENTS_TSV"] = str(session_dir / bids["events_tsv"])
                env["BRAIN_SYNC_SOURCE_DATA_DIR"] = str(session_dir / "sourcedata" / "brainsync")
                env["BRAIN_SYNC_ANALYSIS_DIR"] = str(session_dir / "analysis")
                assert recording_path is not None
                env["BRAIN_SYNC_RECORDING_PATH"] = str(recording_path)
                channel_types = session_payload.get("channel_types")
                eeg_channel_indices = session_payload.get("eeg_channels")
                eog_channel_indices = session_payload.get("eog_channels")
                if not isinstance(channel_types, list):
                    montage_payload = session_payload.get("montage")
                    session_montage = (
                        ChannelMontage.from_dict(montage_payload)
                        if isinstance(montage_payload, dict) and montage_payload
                        else ChannelMontage()
                    )
                    channel_types = list(session_montage.channel_types)
                    eeg_channel_indices = list(session_montage.eeg_indices)
                    eog_channel_indices = list(session_montage.eog_indices)
                elif not isinstance(eeg_channel_indices, list) or not isinstance(eog_channel_indices, list):
                    eeg_channel_indices = [
                        index for index, value in enumerate(channel_types) if str(value).casefold() == "eeg"
                    ]
                    eog_channel_indices = [
                        index for index, value in enumerate(channel_types) if str(value).casefold() == "eog"
                    ]
                env["BRAIN_SYNC_CHANNEL_TYPES"] = json.dumps(channel_types, ensure_ascii=True)
                env["BRAIN_SYNC_EEG_CHANNEL_INDICES"] = json.dumps(eeg_channel_indices)
                env["BRAIN_SYNC_EOG_CHANNEL_INDICES"] = json.dumps(eog_channel_indices)
                task_extra["channel_types"] = channel_types
                task_extra["eeg_channel_indices"] = eeg_channel_indices
                task_extra["eog_channel_indices"] = eog_channel_indices
            else:
                env.pop("BRAIN_SYNC_SESSION_DIR", None)
                for key in (
                    "BRAIN_SYNC_SESSION_MANIFEST",
                    "BRAIN_SYNC_BIDS_ROOT",
                    "BRAIN_SYNC_EVENTS_TSV",
                    "BRAIN_SYNC_SOURCE_DATA_DIR",
                    "BRAIN_SYNC_ANALYSIS_DIR",
                    "BRAIN_SYNC_RECORDING_PATH",
                    "BRAIN_SYNC_CHANNEL_TYPES",
                    "BRAIN_SYNC_EEG_CHANNEL_INDICES",
                    "BRAIN_SYNC_EOG_CHANNEL_INDICES",
                ):
                    env.pop(key, None)
            env["BRAIN_SYNC_TASK"] = config.task
            env["BRAIN_SYNC_OUTPUT_DIR"] = str(output_dir)
            if config.model_manifest.strip():
                env["BRAIN_SYNC_MODEL_MANIFEST"] = str(manifest_path)
            else:
                env.pop("BRAIN_SYNC_MODEL_MANIFEST", None)
            task_extra["command"] = command
            task_extra["working_directory"] = cwd or str(base_dir or Path.cwd())
            if manifest_payload is not None:
                task_extra["manifest_schema"] = manifest_payload.get("schema")
            if self._stop_requested:
                record_status("stopped", {"cancelled": True})
                return "stopped", False, "算法任务已停止"
            code = await self._execute_command(command, cwd, env, emit_line, record_status)
            state = "stopped" if self._stop_requested else ("completed" if code == 0 else "failed")
            record_status(state, {"returncode": code})
            return state, state == "completed", f"算法任务已结束，返回码 {code}"
        except asyncio.CancelledError:
            record_status("stopped", {"cancelled": True})
            return "stopped", False, "算法任务已停止"
        except Exception as exc:
            record_status("failed", {"error": str(exc)})
            return "failed", False, f"算法任务启动失败: {exc}"
        finally:
            if log_file is not None:
                log_file.close()


def create_device_adapter(mock: bool) -> DeviceAdapter:
    return MockDeviceAdapter() if mock else DeviceAdapter()
