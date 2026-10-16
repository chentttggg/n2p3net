from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    BRAIN_SYNC_STORAGE_CONTRACT,
    DEFAULT_SAMPLE_RATE_HZ,
    ChannelMontage,
    ExperimentConfig,
    ModelTaskConfig,
    utc_now,
)
from .recording_clock import SEGMENTED_TIME_BASE

SESSION_STATUSES = frozenset({"running", "completed", "aborted", "recording_error", "error", "aborted_recovered", "discarded"})
SESSION_SCHEMA = BRAIN_SYNC_STORAGE_CONTRACT.session_schema
ANALYSIS_READY_SESSION_STATUSES = frozenset({"completed"})
RAW_TIME_BASE = BRAIN_SYNC_STORAGE_CONTRACT.raw_time_base
CONFIRMATION_PENDING = ".target-confirmation.pending.json"
SESSION_ACTIVE_LOCK = ".session-active.lock"


def _manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@contextmanager
def _session_file_lock(session: Path, name: str):
    """Take a non-blocking OS lock, automatically released after process exit."""
    with (session / name).open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("会话仍处于活动状态或被其他操作占用，请稍后重试。") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _confirmation_lock(session: Path):
    return _session_file_lock(session, ".target-confirmation.lock")


def _active_session_lock(session: Path):
    return _session_file_lock(session, SESSION_ACTIVE_LOCK)


def _recover_confirmation(session: Path) -> None:
    """Finish a durable intent, preserving history and rejecting concurrent edits.

    The pending marker blocks readers until both the journal and manifest are
    durable. Repeating recovery never appends a second confirmation event.
    """
    pending = session / CONFIRMATION_PENDING
    if not pending.exists():
        return
    intent = json.loads(pending.read_text(encoding="utf-8"))
    if not isinstance(intent, dict) or intent.get("schema") != "brainsync-target-confirmation/1":
        raise ValueError("Unsupported target confirmation recovery record")
    manifest_path = session / "session.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _manifest_digest(current)
    if digest not in {intent["before_sha256"], _manifest_digest(intent["manifest"])}:
        raise ValueError("Session changed during target confirmation; recovery requires inspection")
    journal = session / "events/events.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    data = journal.read_bytes() if journal.exists() else b""
    offset = intent["journal_size"]
    event = (json.dumps(intent["event"], ensure_ascii=False) + "\n").encode("utf-8")
    if (type(offset) is not int or offset < 0 or len(data) < offset
            or hashlib.sha256(data[:offset]).hexdigest() != intent["journal_sha256"]
            or not event.startswith(data[offset:])):
        raise ValueError("Event journal changed during target confirmation; recovery requires inspection")
    tail = data[offset:]
    if tail != event:
        if tail:
            # Only a verified prefix of THIS transaction may be replaced.
            with journal.open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
        with journal.open("ab") as stream:
            stream.write(event)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        with journal.open("r+b") as stream:
            os.fsync(stream.fileno())
    atomic_write_json(manifest_path, intent["manifest"])
    pending.unlink()


def _confirm_target(session: Path, thought_digit: int | None, *, active_owner: bool = False) -> bool:
    if thought_digit is not None and (type(thought_digit) is not int or not 1 <= thought_digit <= 9):
        raise ValueError("默想数字必须是 1 到 9 之间的整数。")
    with (nullcontext() if active_owner else _active_session_lock(session)), _confirmation_lock(session):
        _recover_confirmation(session)
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("experiment"), dict):
            raise ValueError("会话实验配置无效，无法确认默想数字。")
        before = _manifest_digest(manifest)
        confirmed = thought_digit is not None
        timestamp = utc_now()
        transaction_id = uuid4().hex
        label = {"status": "confirmed" if confirmed else "unconfirmed",
                 "source": "post_experiment_confirmation", "thought_digit": thought_digit,
                 "confirmed_utc": timestamp if confirmed else None,
                 "confirmation_id": transaction_id,
                 "derivation": "is_target = event.digit == target_label.thought_digit"}
        manifest["target_label"] = label
        manifest["experiment"]["target_label_status"] = "confirmed_post_experiment" if confirmed else "unconfirmed_post_experiment"
        manifest["experiment"]["thought_digit"] = thought_digit
        event = {"event": "thought_digit_confirmed" if confirmed else "thought_digit_confirmation_skipped",
                 "timestamp_utc": timestamp, "confirmation_id": transaction_id,
                 "payload": {"thought_digit": thought_digit} if confirmed else {}}
        journal = session / "events/events.jsonl"
        history = journal.read_bytes() if journal.exists() else b""
        if history and not history.endswith(b"\n"):
            raise ValueError("Existing event journal is incomplete; refusing to overwrite history")
        atomic_write_json(session / CONFIRMATION_PENDING, {
            "schema": "brainsync-target-confirmation/1", "before_sha256": before,
            "manifest": manifest, "event": event, "journal_size": len(history),
            "journal_sha256": hashlib.sha256(history).hexdigest(),
        })
        _recover_confirmation(session)
        return confirmed


def safe_name(value: str, fallback: str = "session") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value or fallback


def atomic_write_json(target: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON document so a crash cannot leave a half-written manifest."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        temp.replace(target)
    except Exception:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def session_relative_path(session_dir: str | Path, artifact_path: str | Path) -> str:
    """Return a portable artifact path that is constrained to one session."""

    root = Path(session_dir).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    try:
        return artifact.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"录制文件必须位于会话目录内: {artifact}") from exc


def load_model_input_session(session_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """Load ready recorded input with confirmed supervision for calibration."""

    payload, artifact = load_inference_input_session(session_dir)
    target_label = payload.get("target_label")
    if not isinstance(target_label, dict) or target_label.get("status") != "confirmed":
        raise ValueError("会话尚未在实验结束后确认默想数字，禁止启动校准任务。")
    thought_digit = target_label.get("thought_digit")
    if isinstance(thought_digit, bool) or not isinstance(thought_digit, int) or not 1 <= thought_digit <= 9:
        raise ValueError("会话默想数字确认无效，禁止启动校准任务。")
    return payload, artifact


def _declared_session_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"会话缺少文件路径: {field}。")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"文件路径必须相对会话目录保存: {field}。")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"文件路径不能跳出会话目录: {field}。") from exc
    if not resolved.is_file():
        raise ValueError(f"文件不存在: {field} ({value})。")
    return resolved


def load_inference_input_session(session_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """Check recorded BIDS readiness without reading or requiring target supervision."""

    root = Path(session_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("目标会话中未找到 session.json。")
    with _confirmation_lock(root):
        return _load_inference_input_session(root)


def _load_inference_input_session(root: Path) -> tuple[dict[str, Any], Path]:
    if (root / CONFIRMATION_PENDING).exists():
        raise ValueError("目标确认尚未完整提交，请先恢复会话确认。")
    session_file = root / "session.json"
    try:
        payload = json.loads(session_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("目标会话中未找到 session.json。") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("session.json 不是有效 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("session.json 顶层必须是对象。")
    if payload.get("schema") != SESSION_SCHEMA:
        raise ValueError(
            f"会话 schema 不受支持: {payload.get('schema')!r}。"
            "请使用 GTN Workstation 生成的 BIDS raw v4 会话。"
        )
    if payload.get("status") not in ANALYSIS_READY_SESSION_STATUSES:
        raise ValueError(f"会话状态不允许算法执行: {payload.get('status')!r}。")
    recording = payload.get("recording")
    if not isinstance(recording, dict):
        raise ValueError("会话缺少录制信息。")
    timeline = recording.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("会话缺少录制时间轴。")
    if timeline.get("status") != "finalized":
        raise ValueError("会话录制时间轴未完成或与 EEG marker 时间基准不一致。")
    if timeline.get("time_base") not in {RAW_TIME_BASE, SEGMENTED_TIME_BASE}:
        raise ValueError("会话录制时间轴未完成或与 EEG marker 时间基准不一致。")
    if timeline.get("rest_segments_retained") is not True:
        raise ValueError("BIDS raw 必须保留休息段并在预处理阶段排除。")
    if recording.get("stage") != BRAIN_SYNC_STORAGE_CONTRACT.raw_stage:
        raise ValueError("会话录制不是 BIDS raw 输入阶段。")
    if recording.get("preprocessing_status") != BRAIN_SYNC_STORAGE_CONTRACT.preprocessing_status:
        raise ValueError("会话预处理状态无效。")
    quality = payload.get("quality")
    continuity = quality.get("eeg_continuity") if isinstance(quality, dict) else None
    if not isinstance(continuity, dict) or continuity.get("passed") is not True:
        raise ValueError("EEG 连续性质量未通过，禁止启动算法任务。")

    artifact = _declared_session_file(root, recording.get("path"), "recording.path")
    bids = recording.get("bids")
    if not isinstance(bids, dict):
        raise ValueError("会话缺少 BIDS sidecar 清单。")
    required_bids_files = (
        "dataset_description", "recording", "eeg_json", "channels_tsv",
        "events_tsv", "electrodes_tsv", "coordsystem_json",
    )
    for field in required_bids_files:
        _declared_session_file(root, bids.get(field), f"recording.bids.{field}")
    if (root / Path(str(bids["recording"]))).resolve() != artifact:
        raise ValueError("recording.path 与 recording.bids.recording 不一致。")
    for owner_name, owner in (("recording", recording), ("recording.timeline", timeline)):
        for field in ("source_path", "output_path"):
            if field in owner:
                _declared_session_file(root, owner[field], f"{owner_name}.{field}")
    return payload, artifact


def find_latest_model_input_session(root: str | Path) -> Path | None:
    """Return the newest session that passes the BIDS raw input contract."""

    base = Path(root).expanduser().resolve()
    candidates = sorted(
        base.glob("*/session.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for session_file in candidates:
        try:
            load_model_input_session(session_file.parent)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        return session_file.parent
    return None


class SessionStore:
    """Writes a reproducible session manifest and append-only event stream."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.session_dir: Path | None = None
        self._active_root: Path | None = None
        self._events = None
        self.session_id = ""
        self.bids_session_label = ""
        self._manifest: dict[str, Any] | None = None
        self._active_lease = None

    @property
    def active(self) -> bool:
        return self.session_dir is not None

    def start(
        self,
        config: ExperimentConfig,
        device: dict[str, Any],
        model: dict[str, Any],
        montage: ChannelMontage | None = None,
        runtime: dict[str, Any] | None = None,
    ) -> Path:
        if self.active:
            raise RuntimeError("已有活动会话，必须先结束当前会话。")
        acquisition_sample_rate_hz = device.get(
            "sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ
        )
        if (
            isinstance(acquisition_sample_rate_hz, bool)
            or not isinstance(acquisition_sample_rate_hz, (int, float))
            or not math.isfinite(float(acquisition_sample_rate_hz))
            or acquisition_sample_rate_hz <= 0
        ):
            raise ValueError("设备采样率必须是正数。")
        montage = montage or ChannelMontage()
        montage.validate()
        session_prefix = safe_name(config.subject_id, "P001")
        save_root = Path(config.save_dir).expanduser()
        if not save_root.is_absolute():
            save_root = self.root.parent / save_root
        save_root = save_root.resolve()
        save_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for run_index in range(1, 1000):
            session_id = f"{session_prefix}_{stamp}_{run_index:02d}"
            candidate = (save_root / session_id).resolve()
            try:
                candidate.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            self.session_id = session_id
            self.bids_session_label = f"{stamp.replace('_', 'T')}{run_index:02d}"
            self.session_dir = candidate
            self._active_root = candidate.parent.resolve()
            break
        else:
            raise RuntimeError("同一秒内无法分配新的会话序号。")
        manifest = {
            "schema": SESSION_SCHEMA,
            "session_id": self.session_id,
            "bids_session_label": self.bids_session_label,
            "started_utc": utc_now(),
            "ended_utc": None,
            "status": "running",
            "experiment": config.to_dict(),
            "device": device,
            "model": model,
            "channels": list(montage.eeg),
            "channel_types": list(montage.channel_types),
            "eeg_channels": list(montage.eeg_indices),
            "eog_channels": list(montage.eog_indices),
            "montage": montage.to_dict(),
            "runtime": runtime or {},
            "target_label": {
                "status": "pending",
                "source": "post_experiment_confirmation",
                "thought_digit": None,
                "confirmed_utc": None,
                "derivation": "is_target = event.digit == target_label.thought_digit",
            },
            "algorithm_boundary": {
                "frontend_owns_resampling": False,
                "frontend_owns_model_training": False,
                "frontend_owns_ocular_artifact_rejection": False,
                "ocular_artifact_rejection": "centered_eog_regression",
                "eog_channel_indices": list(montage.eog_indices),
                "eeg_channel_indices": list(montage.eeg_indices),
                "eog_trace_policy": "retain_raw_for_quality_and_regression",
                "timing_quality_control": "algorithm_posthoc_required",
                "input_contract": "BIDS raw continuous EEG + events.tsv stimulus/rest rows",
                "recording_path": "recording.path is the session-relative BIDS EEG input",
                "acquisition_sample_rate_hz": acquisition_sample_rate_hz,
                "source_sample_rate_hz": "recording.timeline.source_sample_rate_hz",
                "algorithm_target_sample_rate_hz": "declared by the selected algorithm contract",
                "event_filter": "events.tsv trial_type=stimulus",
                "event_time_field": "events.tsv onset relative to continuous recording start",
                "rest_policy": "retain in raw; exclude intersecting epochs while generating cache",
                "channel_positions": "BIDS electrodes.tsv + coordsystem.json (metres)",
            },
        }
        try:
            lease = _active_session_lock(self.session_dir)
            lease.__enter__()
            self._active_lease = lease
            # NOTE: a "raw/" placeholder directory used to be created here but
            # was never written to; BIDS raw data lives under sub-*/ses-*/eeg/.
            (self.session_dir / "analysis").mkdir(exist_ok=True)
            (self.session_dir / "events").mkdir(exist_ok=True)
            self._events = (self.session_dir / "events" / "events.jsonl").open("a", encoding="utf-8")
            self._manifest = manifest
            self.write_json("session.json", manifest)
            self.event("session_start", {"session_id": self.session_id})
        except Exception:
            self._clear_active_context()
            raise
        return self.session_dir

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        if self.session_dir is None:
            return
        atomic_write_json(self.session_dir / name, payload)

    def event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        if self._events is None:
            return
        if self.session_dir is not None and (self.session_dir / CONFIRMATION_PENDING).exists():
            raise ValueError("目标确认需要恢复后才能追加事件")
        record = {"event": name, "timestamp_utc": utc_now(), "payload": payload if payload is not None else {}}
        self._events.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events.flush()

    def update_manifest(self, **updates: Any) -> None:
        if self.session_dir is None or self._manifest is None:
            return
        if (self.session_dir / CONFIRMATION_PENDING).exists():
            raise ValueError("目标确认需要恢复后才能更新会话")
        updated = {**self._manifest, **updates}
        self.write_json("session.json", updated)
        self._manifest = updated

    def confirm_thought_digit(self, thought_digit: int | None) -> bool:
        if self.session_dir is None or self._manifest is None:
            raise RuntimeError("当前没有可确认默想数字的活动会话。")
        return self.confirm_thought_digit_for_session(self.session_dir, thought_digit)

    def confirm_thought_digit_for_session(self, session_dir: str | Path, thought_digit: int | None) -> bool:
        """Commit active and late confirmation through the same durable interface."""
        session = Path(session_dir).expanduser().resolve()
        result = _confirm_target(session, thought_digit, active_owner=self.session_dir == session)
        if self.session_dir == session:
            self._manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        return result

    def model_task(
        self,
        config: ModelTaskConfig,
        status: str,
        extra: dict[str, Any] | None = None,
        session_dir: str | Path | None = None,
    ) -> None:
        target_dir = Path(session_dir).expanduser().resolve() if session_dir is not None else self.session_dir
        if target_dir is None:
            return
        task_dir = target_dir / "analysis"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / "model_tasks.jsonl"
        with task_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "timestamp_utc": utc_now(),
                "status": status,
                "config": config.to_dict(),
                "extra": extra or {},
            }, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())

    def discard(self) -> None:
        if self.session_dir is None:
            return
        directory = self.session_dir.resolve()
        if directory.parent != self._active_root or directory.name != self.session_id:
            raise ValueError("Discard is restricted to the active session directory.")
        session_id = self.session_id
        active_root = self._active_root
        self.event("session_discard_requested", {"status": "aborted"})
        self.finish("aborted")
        shutil.rmtree(directory)
        atomic_write_json(active_root / "discarded" / f"{session_id}.json", {
            "session_id": session_id, "status": "discarded", "discarded_utc": utc_now(),
        })

    def finish(self, status: str = "completed") -> None:
        if status not in SESSION_STATUSES or status == "running":
            raise ValueError(f"无效的会话结束状态: {status}")
        session_dir = self.session_dir
        try:
            if self._events is not None:
                self.event("session_end", {"status": status})
                self._events.flush()
                os.fsync(self._events.fileno())
            if session_dir is not None and self._manifest is not None:
                self._manifest.update({"status": status, "ended_utc": utc_now()})
                self.write_json("session.json", self._manifest)
        finally:
            self._clear_active_context()

    def _clear_active_context(self) -> None:
        try:
            if self._events is not None:
                self._events.close()
        finally:
            self._events = None
            self._manifest = None
            self.session_dir = None
            self._active_root = None
            self.session_id = ""
            self.bids_session_label = ""
            self._release_active_lease()

    def _release_active_lease(self) -> None:
        lease, self._active_lease = self._active_lease, None
        if lease is not None:
            lease.__exit__(None, None, None)

    @staticmethod
    def incomplete_sessions(root: str | Path) -> list[Path]:
        base = Path(root).expanduser().resolve()
        if not base.is_dir():
            return []
        results: list[Path] = []
        for session_file in base.glob("*/session.json"):
            try:
                payload = json.loads(session_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if ((isinstance(payload, dict) and payload.get("status") == "running")
                    or (session_file.parent / CONFIRMATION_PENDING).exists()):
                results.append(session_file.parent)
        return results

    @staticmethod
    def recover_incomplete(session_dir: str | Path, reason: str = "application_recovery") -> None:
        path = Path(session_dir).expanduser().resolve()
        with _active_session_lock(path), _confirmation_lock(path):
            _recover_confirmation(path)
            SessionStore._recover_abandoned_session(path, reason)

    @staticmethod
    def _recover_abandoned_session(path: Path, reason: str) -> None:
        session_file = path / "session.json"
        payload = json.loads(session_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("session.json 顶层必须是对象。")
        if payload.get("status") != "running":
            return
        events_file = path / "events" / "events.jsonl"
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with events_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({
                "event": "session_recovered",
                "timestamp_utc": utc_now(),
                "payload": {"status": "aborted_recovered", "reason": reason},
            }, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        payload.update({"status": "aborted_recovered", "ended_utc": utc_now()})
        atomic_write_json(session_file, payload)
