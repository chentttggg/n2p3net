"""One acquisition session: sample support, pause ledger and durable raw export."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .bids import (
    inspect_eeg_recording,
    make_bids_session_layout,
    write_bids_dataset_metadata,
    write_bids_eeg_sidecars,
    write_bids_events,
)
from .device_state import DeviceState
from .models import (
    BRAIN_SYNC_STORAGE_CONTRACT,
    DEFAULT_SAMPLE_RATE_HZ,
    MODEL_EPOCH_SUPPORT,
    ChannelMontage,
    ExperimentConfig,
    edf_birthdate,
    utc_now,
)
from .recording_clock import SEGMENTED_TIME_BASE, stored_time
from .runtime import make_channel_config
from .storage import SessionStore, session_relative_path
from .timeline import RAW_TIME_BASE, inspect_recording_source, normalize_recording

RECORDING_FILE_READY_TIMEOUT_SECONDS = 10.0
RECORDING_FILE_READY_POLL_SECONDS = 0.1
RECORDING_STOP_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class RecordingOutcome:
    session_dir: Path | None
    status: str


class RecordingSession:
    def __init__(self, device, store: SessionStore, *, mock: bool, log, quality):
        self.device, self.store, self.mock = device, store, mock
        self.log, self.quality = log, quality
        self._finish_task = None
        self._state = "idle"
        self._outcome = RecordingOutcome(None, "idle")
        self._reset()

    def _reset(self):
        self._started = None
        self._source_path = None
        self._info = {}
        self._config = None
        self._layout = None
        self._montage = None
        self._impedance = []
        self._sample_rate = float(DEFAULT_SAMPLE_RATE_HZ)
        self._rest_segments = []
        self._pauses = []
        self._annotation_tasks = set()
        self._last_stimulus_sample = None
        self._last_stimulus_segment_id = None
        self._integrity_error = None
        self.discard_requested = False

    @property
    def busy(self):
        return self._state in {"preparing", "recording", "finishing"}

    @property
    def finishing(self):
        return self._state == "finishing"

    @property
    def started_monotonic(self):
        return self._started

    def fault(self, message: str):
        self._integrity_error = message

    async def prepare(self, config: ExperimentConfig, montage: ChannelMontage, *, sample_rate: int,
                      gain: int, model_info=None, runtime_info=None, requested_target=None, impedance=()) -> Path:
        if self.busy or self.store.active:
            raise RuntimeError("已有采集会话正在执行或收尾")
        self._reset()
        self._finish_task = None
        self._state = "preparing"
        try:
            config.validate_protocol()
            birthdate = edf_birthdate(config.birthdate)
            await self.device.configure(sample_rate, gain)
            snapshot = self.device.snapshot
            self.log(f"实验采集参数已应用: {snapshot.sample_rate_hz} Hz / {snapshot.gain} / {snapshot.configuration_verification}")
            info = {name: getattr(snapshot, name) for name in (
                "transport", "firmware", "sample_rate_hz", "gain", "configuration_verification", "configuration_note")}
            path = self.store.start(config, info, model_info or {}, montage=montage, runtime=runtime_info or {})
            self._config, self._montage, self._impedance = config, montage, list(impedance)
            if requested_target is not None:
                self.store.update_manifest(calibration={"requested_target": requested_target})
            self._layout = make_bids_session_layout(path, config.subject_id, self.store.bids_session_label)
            write_bids_dataset_metadata(self._layout, config)
            self.store.update_manifest(quality=self.quality())
            raw_dir = path / "sourcedata/brainsync"
            channel_config = make_channel_config(self.device.sdk, montage) if self.device.sdk is not None else None
            recording_path = await self.device.start_recording(
                str(raw_dir), config.subject_id, channel_config,
                sex=config.sex if config.sex in {"M", "F"} else "X", birthdate=birthdate,
                participant_name=config.subject_id)
            self._started = self.device.recording_started_monotonic or time.monotonic()
            returned = Path(recording_path).expanduser()
            self._source_path = (returned if returned.is_absolute() else raw_dir / returned).resolve()
            self._sample_rate = float(snapshot.sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ)
            self._info = {
                "path": session_relative_path(path, self._source_path),
                "source_path": session_relative_path(path, self._source_path),
                "stage": "source_acquisition", "preprocessing_status": "not_started",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "timeline": {"status": "continuous_recording", "time_base": RAW_TIME_BASE,
                             "rest_segments_retained": True},
            }
            self.store.event("recording_start", self._info)
            self.store.update_manifest(recording=self._info)
            self._state = "recording"
            return path
        except BaseException as error:
            await self.finish("aborted" if isinstance(error, asyncio.CancelledError) else "error")
            raise

    def pause(self):
        if (not self.store.active or self.device.state is not DeviceState.RECORDING
                or self.device.recording_paused):
            return
        count = self.device.pause_recording()
        pause = {"pause_id": len(self._pauses) + 1, "boundary_sample": count,
                 "start_wall_seconds": time.monotonic() - self._started,
                 "end_wall_seconds": None, "started_utc": utc_now()}
        self._pauses.append(pause)
        self.store.event("experiment_pause", pause)
        self.store.update_manifest(acquisition_pauses=list(self._pauses))

    def resume(self):
        if not self.store.active or not self.device.recording_paused:
            return
        self.device.resume_recording()
        pause = self._pauses[-1]
        pause.update(end_wall_seconds=time.monotonic() - self._started, ended_utc=utc_now())
        self.store.event("experiment_resume", {**pause, "countdown_seconds": 3})
        self.store.update_manifest(acquisition_pauses=list(self._pauses))

    async def finish(self, status: str) -> RecordingOutcome:
        if self._finish_task is None:
            if not self.busy and not self.store.active:
                return self._outcome
            self._state = "finishing"
            self._finish_task = asyncio.create_task(self._finish(status))
        return await asyncio.shield(self._finish_task)

    async def _finish(self, status: str) -> RecordingOutcome:
        path = self.store.session_dir
        final = "recording_error" if self._integrity_error else status
        try:
            try:
                self._close_open_rest_segment()
            except Exception as error:
                self.fault(str(error))
                final = "recording_error"
                self.log(f"休息区间收尾失败: {error}")
            try:
                await self._flush_annotations()
                self._finalize_pause_annotations()
            except Exception as error:
                self.fault(str(error))
                final = "recording_error"
                self.log(f"标记收尾失败: {error}")
            if final == "completed" and self.device.state is DeviceState.RECORDING:
                try:
                    await self.wait_for_support("response", {})
                except Exception as error:
                    self.fault(str(error))
                    final = "recording_error"
                    self.log(f"EEG 尾段未确认: {error}")
            if self.device.state is DeviceState.RECORDING or self.device.recovery_may_have_recording:
                stopped = await self._stop_recording("停止录制")
                if not stopped and (self.device.state is DeviceState.RECORDING
                                    or self.device.recovery_may_have_recording):
                    self.fault("停止录制失败，采集资源仍需恢复")
                    final = "recording_error"
            else:
                self.device.emergency_save_recording()
            if self.store.active:
                if self.discard_requested:
                    self.store.discard()
                    final = "discarded"
                else:
                    await self._finalize_raw_recording()
            clock = self.device.recording_clock_record
            if not self.discard_requested and (self._integrity_error or (clock and clock.get("integrity_failed"))):
                final = "recording_error"
        except (Exception, asyncio.CancelledError) as error:
            final = "recording_error"
            self.log(f"采集收尾失败: {error}")
            # Export must still be attempted by the device's emergency path.
            self.device.emergency_save_recording()
        finally:
            if self.store.active:
                try:
                    self.store.event("recording_end", {"status": final})
                except Exception as error:
                    self.log(f"录制结束事件写入失败: {error}")
                    final = "recording_error"
                try:
                    self.store.finish(final)
                except Exception as error:
                    self.log(f"会话最终状态写入失败，需在分析页恢复: {error}")
                    final = "recording_error"
            self._state = "idle"
            self._outcome = RecordingOutcome(path, final)
            self._started = None
        return self._outcome

    async def _stop_recording(self, context: str) -> bool:
        """Stop acquisition with a bound so UI cleanup cannot wait forever."""

        try:
            await asyncio.wait_for(
                self.device.stop_recording(),
                timeout=RECORDING_STOP_TIMEOUT_SECONDS,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log(f"{context}失败: {exc}")
            # The stop failed (timeout or device error). _stop_recording_locked
            # normally finalizes before touching the device, but if that
            # path was interrupted, the recorder still holds the only copy
            # of the samples. Save it synchronously right now.
            try:
                rescued = self.device.emergency_save_recording()
            except Exception as save_exc:
                rescued = None
                self.log(f"{context}后紧急保存录制失败: {save_exc}")
            if rescued:
                self.log(f"{context}失败，已紧急保存 本地 EDF 录制: {rescued}")
            if self.store.active:
                try:
                    self.store.event("recording_stop_error", {"error": str(exc)})
                except Exception as event_exc:
                    self.log(f"录制错误事件写入失败: {event_exc}")
            return False


    def _close_open_rest_segment(self, end_monotonic: float | None = None) -> None:
        if self._started is None or not self.store.active:
            return
        open_segment = next((segment for segment in reversed(self._rest_segments)
                             if segment.get("end_seconds") is None), None)
        if open_segment is None:
            return
        end_monotonic = time.monotonic() if end_monotonic is None else end_monotonic
        self.marker({
            "kind": "rest_end",
            "block_id": open_segment["block_id"],
            "software_rest_end_monotonic": end_monotonic,
            "synthetic": True,
        })


    def marker(self, event: dict[str, Any]) -> None:
        if self._started is None or not self.store.active:
            return
        kind = str(event.get("kind", ""))
        monotonic_keys = {
            "onset": "software_onset_monotonic",
            "offset": "software_offset_monotonic",
            "rest_start": "software_rest_start_monotonic",
            "rest_end": "software_rest_end_monotonic",
            "invalidated": "software_invalidated_monotonic",
        }
        monotonic_key = monotonic_keys.get(kind)
        if monotonic_key is None:
            return
        monotonic_time = event.get(monotonic_key)
        if monotonic_time is None:
            return
        relative_seconds = max(0.0, float(monotonic_time) - self._started)
        position_mapper = getattr(self.device, "recording_source_position", None)
        source_segment_id = None
        if callable(position_mapper):
            position = position_mapper(float(monotonic_time))
            source_sample, source_segment_id = position if position is not None else (None, None)
        else:
            mapper = getattr(self.device, "recording_source_sample", None)
            source_sample = mapper(float(monotonic_time)) if callable(mapper) else None
        if kind == "rest_start":
            self._rest_segments.append({
                "block_id": event.get("block_id"),
                "start_seconds": relative_seconds, "end_seconds": None,
                "start_source_sample": source_sample,
            })
        elif kind == "rest_end":
            for segment in reversed(self._rest_segments):
                if segment.get("end_seconds") is None:
                    segment["end_seconds"] = max(relative_seconds, float(segment["start_seconds"]))
                    segment["end_source_sample"] = source_sample
                    break
        eeg_time_seconds = relative_seconds
        if kind in {"onset", "offset", "invalidated"}:
            description = (
                f"STIM_{kind.upper()}"
                f"|trial={event.get('trial_id', '--')}"
                f"|digit={event.get('digit', '--')}"
            )
        else:
            description = f"REST_{kind.removeprefix('rest_').upper()}|block={event.get('block_id', '--')}"
        marker = dict(event)
        marker.update({
            "eeg_time_seconds": eeg_time_seconds,
            "wall_time_seconds": relative_seconds,
            "eeg_time_base": RAW_TIME_BASE,
            "annotation": description,
        })
        if source_sample is not None:
            marker.update(source_sample=int(source_sample), clock_alignment="delivery_anchored")
            if source_segment_id is not None:
                marker["source_segment_id"] = source_segment_id
            if kind == "onset":
                self._last_stimulus_sample = int(source_sample)
                self._last_stimulus_segment_id = source_segment_id
        else:
            marker["clock_alignment"] = "legacy_software_origin_unverified"
        try:
            self.store.event("recording_marker", marker)
        except Exception as exc:
            self.log(f"刺激事件写入失败，正在停止实验: {exc}")
            self.fault(str(exc))
            raise
        annotation_time = (
            source_sample / self._sample_rate if source_sample is not None else
            stored_time(relative_seconds, getattr(self, "_pauses", []), self._sample_rate)
        )
        task = asyncio.create_task(self.device.add_annotation(annotation_time, description))
        self._annotation_tasks.add(task)
        task.add_done_callback(self._annotation_done)


    async def wait_for_support(self, stage: str, request: dict) -> None:
        if stage == "baseline":
            support = await self.device.wait_for_recorded_baseline(
                not_before_monotonic=request.get("not_before_monotonic"))
        elif stage == "response":
            sample = getattr(self, "_last_stimulus_sample", None)
            segment_id = getattr(self, "_last_stimulus_segment_id", None)
            if sample is None or segment_id is None:
                raise RuntimeError("Stimulus has no sample-clock anchor.")
            rate = self._sample_rate
            start = sample + int(np.floor(MODEL_EPOCH_SUPPORT.tmin_ms * rate / 1000.0))
            stop = sample + int(np.ceil(MODEL_EPOCH_SUPPORT.tmax_ms * rate / 1000.0))
            await self.device.wait_for_recorded_sample(stop, start_sample=start, segment_id=segment_id)
            support = {"segment_id": segment_id, "start_sample": start,
                       "stop_sample_exclusive": stop + 1, "sample_rate_hz": rate, "checkpointed": True}
        else:
            raise ValueError("Unknown EEG sample-support stage.")
        if self.store.active:
            self.store.event("epoch_support_confirmed", {"stage": stage, **support})


    def _annotation_done(self, task: asyncio.Task) -> None:
        self._annotation_tasks.discard(task)
        if task.cancelled():
            self.fault("EDF 事件标记写入被取消")
            return
        try:
            task.result()
        except Exception as exc:
            self.fault(f"EDF 事件标记写入失败: {exc}")
            self.log(f"写入 EDF 事件标记失败: {exc}")


    async def _flush_annotations(self) -> None:
        if self._annotation_tasks:
            await asyncio.gather(*tuple(self._annotation_tasks), return_exceptions=True)


    def _finalize_pause_annotations(self) -> None:
        if not self._pauses or not self.store.active:
            return
        markers = []
        with (self.store.session_dir / "events" / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("event") == "recording_marker":
                    markers.append(record["payload"])
        invalid = {marker["trial_id"] for marker in markers if marker.get("kind") == "invalidated" or marker.get("interrupted")}
        annotations = []
        for marker in markers:
            onset = (
                marker["source_sample"] / self._sample_rate
                if marker.get("source_sample") is not None else
                stored_time(marker["wall_time_seconds"], self._pauses, self._sample_rate)
            )
            description = marker["annotation"]
            if marker.get("trial_id") in invalid:
                # EDFlib limits annotation text to 40 bytes. Put validity first.
                kind = {"onset": "ON", "offset": "OFF", "invalidated": "END"}[marker["kind"]]
                description = f"INVALID_{kind}|{marker['trial_id']}|d={marker['digit']}"
            annotations.append((onset, 0.0, description))
        for pause in self._pauses:
            onset = pause["boundary_sample"] / self._sample_rate
            annotations.append((onset, 0.0, f"PAUSE|id={pause['pause_id']}"))
            if pause["end_wall_seconds"] is not None:
                annotations.append((onset, 0.0, f"RESUME_COUNTDOWN|id={pause['pause_id']}|seconds=3"))
        self.device.replace_recording_annotations(sorted(annotations, key=lambda row: row[0]))


    async def _wait_for_recording_source(self, source_path: Path):
        deadline = time.monotonic() + RECORDING_FILE_READY_TIMEOUT_SECONDS
        previous_signature: tuple[int, int] | None = None
        stable_observations = 0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                stat = source_path.stat()
                signature = (int(stat.st_size), int(stat.st_mtime_ns))
                stable_observations = (
                    stable_observations + 1 if signature == previous_signature else 1
                )
                previous_signature = signature
                if stat.st_size > 0 and stable_observations >= 3:
                    return await asyncio.to_thread(inspect_recording_source, source_path)
            except (OSError, ValueError) as exc:
                last_error = exc
                stable_observations = 0
            await asyncio.sleep(RECORDING_FILE_READY_POLL_SECONDS)
        detail = f": {last_error}" if last_error is not None else ""
        raise TimeoutError(
            f"停止录制后 {RECORDING_FILE_READY_TIMEOUT_SECONDS:.0f} 秒内 EDF 未稳定可读{detail}"
        )


    async def _finalize_raw_recording(self) -> None:
        if not self.store.active or self._source_path is None:
            return
        source_path = self._source_path
        if self._layout is None or self._config is None:
            raise RuntimeError("BIDS raw recording context is missing.")
        output_path = self._layout.raw_path(source_path.suffix)
        audit_events_path = self.store.session_dir / "events" / "events.jsonl"
        if self.mock and not source_path.is_file():
            self.fault(f"模拟 EEG 原始录制文件缺失: {source_path}")
            self.store.event("recording_bids_export_skipped", {
                "reason": "mock_source_missing",
                "error": f"mock source does not exist: {source_path}",
            })
            self.store.update_manifest(recording={
                **self._info,
                "path": session_relative_path(self.store.session_dir, source_path),
                "source_path": session_relative_path(self.store.session_dir, source_path),
                "stage": "source_missing",
                "preprocessing_status": "blocked",
                "timeline": {
                    "status": "source_missing",
                    "time_base": RAW_TIME_BASE,
                    "rest_segments_retained": True,
                },
            })
            return
        source_info = await self._wait_for_recording_source(source_path)
        if not np.isclose(
            source_info.sample_rate_hz,
            self._sample_rate,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"EDF 实际采样率为 {source_info.sample_rate_hz:g} Hz，"
                f"采集配置记录为 {self._sample_rate:g} Hz。"
            )
        result = await asyncio.to_thread(
            normalize_recording,
            source_path,
            output_path,
            start_datetime=self._recording_start_datetime(),
        )
        if result is not None:
            raw_info = await asyncio.to_thread(inspect_eeg_recording, output_path)
            bids_events_path = write_bids_events(
                self._layout,
                audit_events_path,
                raw_info.sample_rate_hz,
                raw_info.duration_seconds,
                pauses=self._pauses,
            )
            sdk_version = str(
                getattr(self.device.sdk, "__version__", "n/a")
                if self.device.sdk is not None
                else "n/a"
            )
            write_bids_eeg_sidecars(
                self._layout,
                raw_info,
                self._montage,
                self._config,
                sdk_version=sdk_version,
                impedance_values=list(self._impedance),
                acquisition_pauses=self._pauses,
            )
            source_relative = session_relative_path(self.store.session_dir, source_path)
            output_relative = session_relative_path(self.store.session_dir, output_path)
            rest_segments = []
            for segment in self._rest_segments:
                if segment.get("end_seconds") is None:
                    continue
                endpoints = []
                for endpoint in ("start", "end"):
                    sample = segment.get(f"{endpoint}_source_sample")
                    endpoints.append(sample / raw_info.sample_rate_hz if sample is not None else
                                     stored_time(segment[f"{endpoint}_seconds"], self._pauses,
                                                 raw_info.sample_rate_hz))
                rest_segments.append({"start_seconds": endpoints[0], "end_seconds": endpoints[1],
                                      "duration_seconds": max(0.0, endpoints[1]-endpoints[0])})
            clock_path = source_path.with_name(source_path.name + ".clock.json")
            clock_record = None
            if clock_path.is_file():
                clock_record = json.loads(clock_path.read_text(encoding="utf-8"))
                clock_record["metadata_path"] = session_relative_path(self.store.session_dir, clock_path)
                clock_record["journal_path"] = session_relative_path(
                    self.store.session_dir, source_path.with_name(source_path.name + ".clock.jsonl")
                )
            timeline = {
                "status": "finalized",
                "time_base": SEGMENTED_TIME_BASE if self._pauses else RAW_TIME_BASE,
                "rest_segments_retained": True,
                "rest_segments": rest_segments,
                "acquisition_pauses": self._pauses,
                "pause_samples_retained": False,
                "source_sample_rate_hz": raw_info.sample_rate_hz,
                "output_duration_seconds": raw_info.duration_seconds,
                "header_repaired": result.header_repaired,
                "source_path": source_relative,
                "output_path": output_relative,
            }
            bids = {
                "dataset_description": "dataset_description.json",
                "recording": output_relative,
                "eeg_json": session_relative_path(
                    self.store.session_dir, self._layout.sidecar_path("eeg")
                ),
                "channels_tsv": session_relative_path(
                    self.store.session_dir, self._layout.sidecar_path("channels", ".tsv")
                ),
                "events_tsv": session_relative_path(self.store.session_dir, bids_events_path),
                "electrodes_tsv": session_relative_path(
                    self.store.session_dir,
                    self._layout.coordinate_path("electrodes", ".tsv"),
                ),
                "coordsystem_json": session_relative_path(
                    self.store.session_dir,
                    self._layout.coordinate_path("coordsystem", ".json"),
                ),
            }
            quality = self.quality()
            self.store.event("recording_timeline_finalized", timeline)
            self.store.update_manifest(quality=quality)
            self.store.update_manifest(recording={
                **self._info,
                "path": output_relative,
                "source_path": source_relative,
                "source_sample_rate_hz": raw_info.sample_rate_hz,
                "stage": BRAIN_SYNC_STORAGE_CONTRACT.raw_stage,
                "preprocessing_status": BRAIN_SYNC_STORAGE_CONTRACT.preprocessing_status,
                "preprocessing_note": "rest intervals are retained and marked in BIDS events.tsv",
                "timeline": timeline,
                "bids": bids,
                "clock": clock_record,
            })
            self.log(
                f"BIDS raw 已保存: {Path(output_path).name}；"
                f"保留 {len(rest_segments)} 个休息区间，后续预处理时删除"
            )


    def _recording_start_datetime(self) -> datetime | None:
        value = self._info.get("started_utc")
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

