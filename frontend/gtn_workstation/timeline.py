from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

RAW_TIME_BASE = "continuous_recording"
@dataclass(frozen=True)
class NormalizeResult:
    source_path: str
    output_path: str
    source_duration_seconds: float
    output_duration_seconds: float
    header_repaired: bool = False
    datarecord_duration_seconds: float | None = None
    source_sample_rate_hz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_duration_seconds": self.source_duration_seconds,
            "output_duration_seconds": self.output_duration_seconds,
            "header_repaired": self.header_repaired,
            "datarecord_duration_seconds": self.datarecord_duration_seconds,
            "source_sample_rate_hz": self.source_sample_rate_hz,
            "time_base": RAW_TIME_BASE,
            "samples_removed": False,
        }


@dataclass(frozen=True)
class RecordingSourceInfo:
    duration_seconds: float
    sample_rate_hz: float
    signal_count: int
    header_repair_required: bool


def _copy_header(header: dict[str, Any], start_datetime: Any) -> dict[str, Any]:
    keys = (
        "technician",
        "recording_additional",
        "patientname",
        "patient_additional",
        "patientcode",
        "equipment",
        "admincode",
        "sex",
        "birthdate",
    )
    result = {key: header.get(key, "") or "" for key in keys}
    result["startdate"] = start_datetime
    return result


def _annotation_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _atomic_copy(source: Path, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass
class _RecordingData:
    source_duration_seconds: float
    signal_headers: list[dict[str, Any]]
    sample_frequencies: list[float]
    signal_counts: list[int]
    signals: list[Any]
    header: dict[str, Any]
    start_datetime: datetime
    file_type: int
    annotations: tuple[list[float], list[float], list[str]]
    datarecord_duration_seconds: float
    header_repaired: bool = False


def _decode_fixed_field(data: bytes, start: int, width: int) -> str:
    return data[start : start + width].decode("latin1", errors="replace").strip()


def _parse_number(value: str, field_name: str, *, integer: bool = False) -> int | float:
    try:
        return int(value) if integer else float(value)
    except ValueError as exc:
        raise ValueError(f"EDF field {field_name!r} is not numeric: {value!r}.") from exc


def _parse_edf_date_time(date_text: str, time_text: str) -> datetime:
    try:
        return datetime.strptime(f"{date_text} {time_text}", "%d.%m.%y %H.%M.%S")
    except ValueError as exc:
        raise ValueError(
            f"EDF start date/time is invalid: {date_text!r} {time_text!r}."
        ) from exc


def _parse_birthdate(value: str) -> date:
    for fmt in ("%d-%b-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(value.upper(), fmt).date()
        except ValueError:
            continue
    return date(1970, 1, 1)


def _parse_sdk_edf(source: Path) -> _RecordingData:
    """Decode the known SDK EDF variant whose header pyedflib rejects."""

    import numpy as np
    import pyedflib

    data = source.read_bytes()
    if len(data) < 256:
        raise ValueError(f"EDF header is truncated: {source}")
    header_bytes = int(_decode_fixed_field(data, 184, 8))
    records = int(_decode_fixed_field(data, 236, 8))
    datarecord_duration = float(_decode_fixed_field(data, 244, 8))
    n_signals = int(_decode_fixed_field(data, 252, 4))
    if header_bytes != 256 + n_signals * 256 or records <= 0 or datarecord_duration <= 0:
        raise ValueError("SDK EDF has an invalid fixed header geometry.")
    if n_signals < 2:
        raise ValueError("SDK EDF must contain EEG signals and an annotation signal.")

    field_specs = (
        ("label", 16),
        ("transducer", 80),
        ("dimension", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefilter", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    )
    fields: dict[str, list[str]] = {}
    offset = 256
    for name, width in field_specs:
        fields[name] = [
            _decode_fixed_field(data, offset + index * width, width)
            for index in range(n_signals)
        ]
        offset += n_signals * width
    if offset != header_bytes:
        raise ValueError("SDK EDF signal header geometry does not match header_bytes.")

    annotation_indices = [
        index
        for index, label in enumerate(fields["label"])
        if label.casefold() == "edf annotations"
    ]
    if len(annotation_indices) != 1:
        raise ValueError("SDK EDF must contain exactly one EDF Annotations signal.")
    annotation_index = annotation_indices[0]

    signal_headers: list[dict[str, Any]] = []
    signal_frequencies: list[float] = []
    samples_per_record: list[int] = []
    for index in range(n_signals):
        if index == annotation_index:
            continue
        count = int(_parse_number(fields["samples_per_record"][index], "samples_per_record", integer=True))
        frequency = count / datarecord_duration
        if count <= 0 or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("SDK EDF contains an invalid EEG sample frequency.")
        samples_per_record.append(count)
        signal_frequencies.append(frequency)
        signal_headers.append(
            {
                "label": fields["label"][index] or f"EEG CH{index + 1}",
                "transducer": fields["transducer"][index],
                "dimension": fields["dimension"][index] or "uV",
                "physical_min": float(_parse_number(fields["physical_min"][index], "physical_min")),
                "physical_max": float(_parse_number(fields["physical_max"][index], "physical_max")),
                "digital_min": int(_parse_number(fields["digital_min"][index], "digital_min", integer=True)),
                "digital_max": int(_parse_number(fields["digital_max"][index], "digital_max", integer=True)),
                "prefilter": fields["prefilter"][index],
                "sample_frequency": frequency,
            }
        )

    record_width = sum(int(_parse_number(value, "samples_per_record", integer=True)) for value in fields["samples_per_record"]) * 2
    expected_size = header_bytes + records * record_width
    if len(data) != expected_size:
        raise ValueError(
            f"SDK EDF data size mismatch: expected {expected_size} bytes, got {len(data)}."
        )

    signal_counts = [records * count for count in samples_per_record]
    signals = [np.empty(count, dtype=np.float64) for count in signal_counts]
    annotations: list[tuple[float, float, str]] = []
    source_offset = header_bytes
    signal_index = 0
    for _record_index in range(records):
        record_annotation = b""
        signal_index = 0
        for index in range(n_signals):
            count = int(_parse_number(fields["samples_per_record"][index], "samples_per_record", integer=True))
            byte_count = count * 2
            chunk = data[source_offset : source_offset + byte_count]
            source_offset += byte_count
            if index == annotation_index:
                record_annotation = chunk
                continue
            digital = np.frombuffer(chunk, dtype="<i2").astype(np.float64)
            header = signal_headers[signal_index]
            physical_min = float(header["physical_min"])
            physical_max = float(header["physical_max"])
            digital_min = float(header["digital_min"])
            digital_max = float(header["digital_max"])
            if physical_max <= physical_min or digital_max <= digital_min:
                raise ValueError("SDK EDF contains an invalid physical/digital scale.")
            scaled = physical_min + (digital - digital_min) * (physical_max - physical_min) / (digital_max - digital_min)
            start = _record_index * samples_per_record[signal_index]
            signals[signal_index][start : start + count] = scaled
            signal_index += 1
        for match in re.finditer(rb"\+([0-9]+(?:\.[0-9]*)?)\x15([0-9]+(?:\.[0-9]*)?)\x14([^\x14\x00]+)", record_annotation):
            description = match.group(3).decode("utf-8", errors="replace").strip()
            if description:
                annotations.append((float(match.group(1)), float(match.group(2)), description))

    patient_tokens = _decode_fixed_field(data, 8, 80).split()
    patient_code = patient_tokens[0] if patient_tokens else ""
    sex = patient_tokens[1] if len(patient_tokens) > 1 and patient_tokens[1] in {"M", "F", "X"} else "X"
    birthdate = _parse_birthdate(patient_tokens[2]) if len(patient_tokens) > 2 else date(1970, 1, 1)
    patient_name = patient_tokens[3] if len(patient_tokens) > 3 else "BrainSync"
    patient_additional = " ".join(patient_tokens[4:])
    start_datetime = _parse_edf_date_time(_decode_fixed_field(data, 168, 8), _decode_fixed_field(data, 176, 8))
    header = {
        "technician": "BrainSync",
        "recording_additional": "SDK EDF header normalized",
        "patientname": patient_name,
        "patient_additional": patient_additional,
        "patientcode": patient_code,
        "equipment": "BrainSync_8CH_EEG",
        "admincode": "",
        "sex": sex,
        "birthdate": birthdate,
    }
    annotation_arrays = (
        [item[0] for item in annotations],
        [item[1] for item in annotations],
        [item[2] for item in annotations],
    )
    return _RecordingData(
        source_duration_seconds=records * datarecord_duration,
        signal_headers=signal_headers,
        sample_frequencies=signal_frequencies,
        signal_counts=signal_counts,
        signals=signals,
        header=header,
        start_datetime=start_datetime,
        file_type=pyedflib.FILETYPE_EDFPLUS,
        annotations=annotation_arrays,
        datarecord_duration_seconds=datarecord_duration,
        header_repaired=True,
    )


def _read_valid_edf(source: Path) -> _RecordingData:
    import numpy as np
    import pyedflib

    reader = pyedflib.EdfReader(str(source))
    try:
        source_duration = float(reader.getFileDuration())
        signal_headers = [dict(header) for header in reader.getSignalHeaders()]
        sample_frequencies = [float(header["sample_frequency"]) for header in signal_headers]
        signal_counts = [int(count) for count in reader.getNSamples()]
        signals = [np.asarray(reader.readSignal(index)) for index in range(len(signal_headers))]
        header = reader.getHeader()
        start_datetime = reader.getStartdatetime()
        file_type = int(reader.filetype)
        try:
            annotations = reader.readAnnotations()
            annotation_arrays = (
                [float(value) for value in annotations[0]],
                [float(value) for value in annotations[1]],
                [_annotation_text(value) for value in annotations[2]],
            )
        except Exception:
            annotation_arrays = ([], [], [])
    finally:
        reader.close()
    return _RecordingData(
        source_duration_seconds=source_duration,
        signal_headers=signal_headers,
        sample_frequencies=sample_frequencies,
        signal_counts=signal_counts,
        signals=signals,
        header=header,
        start_datetime=start_datetime,
        file_type=file_type,
        annotations=annotation_arrays,
        datarecord_duration_seconds=1.0,
    )


def _read_recording(source: Path) -> _RecordingData:
    try:
        return _read_valid_edf(source)
    except (OSError, ValueError):
        return _parse_sdk_edf(source)


def _single_sample_rate(sample_frequencies: Iterable[float]) -> float | None:
    values = [float(value) for value in sample_frequencies]
    if not values or not all(math.isfinite(value) and value > 0.0 for value in values):
        return None
    return values[0] if all(math.isclose(value, values[0]) for value in values[1:]) else None


def inspect_recording_source(path: str | Path) -> RecordingSourceInfo:
    """Parse a finalized source recording, including the known SDK EDF variant."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"未找到原始 EEG 录制文件: {source}")
    recording = _read_recording(source)
    sample_rate = _single_sample_rate(recording.sample_frequencies)
    if sample_rate is None:
        raise ValueError("原始 EEG 录制没有统一采样率。")
    return RecordingSourceInfo(
        duration_seconds=float(recording.source_duration_seconds),
        sample_rate_hz=sample_rate,
        signal_count=len(recording.signal_headers),
        header_repair_required=recording.header_repaired,
    )


def _write_recording(
    output: Path,
    recording: _RecordingData,
    signals: list[Any],
    annotations: tuple[Iterable[float], Iterable[float], Iterable[Any]],
    *,
    datarecord_duration_seconds: float,
) -> None:
    import pyedflib

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp{output.suffix}")
    writer = pyedflib.EdfWriter(str(temporary), len(recording.signal_headers), file_type=recording.file_type)
    try:
        writer.setSignalHeaders(recording.signal_headers)
        writer.setHeader(_copy_header(recording.header, recording.start_datetime))
        writer.setDatarecordDuration(float(datarecord_duration_seconds))
        writer.writeSamples(signals)
        for onset, duration, description in zip(*annotations, strict=True):
            writer.writeAnnotation(float(onset), float(duration), _annotation_text(description))
    finally:
        writer.close()
    try:
        _validate_written_recording(temporary, expected_signal_count=len(recording.signal_headers))
        temporary.replace(output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_written_recording(path: Path, *, expected_signal_count: int) -> None:
    """Fail closed unless the normalized output is readable as EDF+/BDF+."""

    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        if reader.signals_in_file != expected_signal_count:
            raise ValueError(
                f"规范化 EDF 通道数为 {reader.signals_in_file}，预期为 {expected_signal_count}。"
            )
        if reader.getFileDuration() <= 0.0:
            raise ValueError("规范化 EDF 没有有效时长。")
        if any(float(value) <= 0.0 for value in reader.getSampleFrequencies()):
            raise ValueError("规范化 EDF 包含无效通道采样率。")
    finally:
        reader.close()


def normalize_recording(
    source_path: str | Path,
    output_path: str | Path,
    *,
    start_datetime: datetime | None = None,
) -> NormalizeResult:
    """Copy valid EDF/BDF unchanged, repairing only malformed SDK metadata."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("规范化输出文件不能覆盖原始录制文件。")
    if not source.is_file():
        raise FileNotFoundError(f"未找到原始 EEG 录制文件: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    recording = _read_recording(source)
    if start_datetime is not None:
        recording.start_datetime = start_datetime.replace(tzinfo=None)
    source_duration = float(recording.source_duration_seconds)
    if not recording.signal_headers or source_duration <= 0.0:
        raise ValueError("原始 EEG 录制文件为空。")

    if recording.header_repaired:
        _write_recording(
            output,
            recording,
            recording.signals,
            recording.annotations,
            datarecord_duration_seconds=recording.datarecord_duration_seconds,
        )
    else:
        _atomic_copy(source, output)
    return NormalizeResult(
        source_path=str(source),
        output_path=str(output),
        source_duration_seconds=source_duration,
        output_duration_seconds=source_duration,
        header_repaired=recording.header_repaired,
        datarecord_duration_seconds=(
            recording.datarecord_duration_seconds if recording.header_repaired else None
        ),
        source_sample_rate_hz=_single_sample_rate(recording.sample_frequencies),
    )
