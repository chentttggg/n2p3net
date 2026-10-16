from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

# MNE standard_1005 after fiducial registration to the head frame, in metres.
STANDARD_CHANNEL_POSITIONS_M = {
    "fz": (-0.001229276531, 0.093274451792, 0.102639295161),
    "cz": (-0.001374138752, 0.027617093176, 0.140199497342),
    "p3": (-0.055038236082, -0.044210303575, 0.099908977747),
    "pz": (-0.001709450386, -0.045212991536, 0.126672923565),
    "p4": (0.053636010736, -0.044334523380, 0.100516028702),
    "po7": (-0.056948624551, -0.065923243761, 0.047907438129),
    "po8": (0.053557317704, -0.066416941583, 0.047850977629),
    "oz": (-0.002060251543, -0.082782991230, 0.060736630111),
}
_BRAIN_SYNC_CHANNEL_NAMES = ("Fz", "Cz", "P3", "Pz", "P4", "PO8", "EOG", "Oz")
_BRAIN_SYNC_CHANNEL_TYPES = ("eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eog", "eeg")
HARDWARE_CHANNEL_INDEXING = "zero_based"


@dataclass(frozen=True)
class EEGDataContract:
    """Shared acquisition metadata consumed by the app and algorithm adapter."""

    sample_rate_hz: int
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    channel_positions_m: tuple[tuple[float, float, float] | None, ...]
    coordinate_frame: str = "head"
    units: str = "m"


@dataclass(frozen=True)
class BrainSyncStorageContract:
    """Canonical raw-data interchange contract shared by acquisition outputs."""

    session_schema: str = "brainsync-gtn-session/4"
    bids_version: str = "1.11.0"
    bids_dataset_type: str = "raw"
    bids_task_name: str = "gtn"
    raw_stage: str = "bids_raw"
    raw_time_base: str = "continuous_recording"
    preprocessing_status: str = "pending"
    stimulus_trial_type: str = "stimulus"
    rest_trial_type: str = "rest"


BRAIN_SYNC_DATA_CONTRACT = EEGDataContract(
    sample_rate_hz=250,
    channel_names=_BRAIN_SYNC_CHANNEL_NAMES,
    channel_types=_BRAIN_SYNC_CHANNEL_TYPES,
    channel_positions_m=tuple(
        None if kind == "eog" else STANDARD_CHANNEL_POSITIONS_M[label.lower()]
        for label, kind in zip(_BRAIN_SYNC_CHANNEL_NAMES, _BRAIN_SYNC_CHANNEL_TYPES)
    ),
)
BRAIN_SYNC_STORAGE_CONTRACT = BrainSyncStorageContract()
CHANNEL_NAMES = BRAIN_SYNC_DATA_CONTRACT.channel_names
CHANNEL_TYPES = BRAIN_SYNC_DATA_CONTRACT.channel_types
DEFAULT_SAMPLE_RATE_HZ = BRAIN_SYNC_DATA_CONTRACT.sample_rate_hz
DEFAULT_CHANNEL_POSITIONS_M = BRAIN_SYNC_DATA_CONTRACT.channel_positions_m
DIGIT_VOCAB = tuple(range(1, 10))
EDF_UNKNOWN_BIRTHDATE = "01-JAN-1970"
BRAIN_SYNC_REFERENCE = json.loads(
    (Path(__file__).resolve().parent.parent / "models" / "acquisition_reference.json").read_text(encoding="utf-8")
)
if BRAIN_SYNC_REFERENCE.get("schema") != "brainsync-acquisition-reference/1":
    raise ValueError("Unsupported BrainSync acquisition reference contract.")
DEFAULT_REF_LABEL = BRAIN_SYNC_REFERENCE["label"]
DEFAULT_GND_LABEL = "T8(T4)"


@dataclass(frozen=True)
class EpochSupport:
    tmin_ms: float
    tmax_ms: float
    model_sample_rate_hz: float
    n_times: int

    @classmethod
    def load(cls, path: Path) -> EpochSupport:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.pop("schema", None) != "n2p3-epoch-support/1":
            raise ValueError("Unsupported model epoch support contract.")
        result = cls(**record)
        if not all(math.isfinite(value) for value in (
            result.tmin_ms, result.tmax_ms, result.model_sample_rate_hz
        )) or not result.tmin_ms < 0 < result.tmax_ms or result.model_sample_rate_hz <= 0:
            raise ValueError("Invalid model epoch time geometry.")
        expected = math.floor(
            (result.tmax_ms-result.tmin_ms)*result.model_sample_rate_hz/1000 + 1e-9
        )
        if result.n_times != expected:
            raise ValueError("Model epoch sample count disagrees with its physical support.")
        return result

    def record(self) -> dict[str, Any]:
        return {"schema": "n2p3-epoch-support/1", **asdict(self)}

    def baseline_guard_seconds(self, source_rate: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
        return -self.tmin_ms / 1000 + 1 / source_rate

    def response_guard_seconds(self, source_rate: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
        return self.tmax_ms / 1000 + 1 / source_rate


MODEL_EPOCH_SUPPORT = EpochSupport.load(
    Path(__file__).resolve().parent.parent / "models" / "epoch_support.json"
)
# Dry electrodes can have substantially higher contact impedance than wet electrodes.
IMPEDANCE_PASS_THRESHOLD_KOHM = 200.0


@dataclass(frozen=True)
class ChannelMontage:
    """Physical electrode labels attached to the eight ADC channels."""

    eeg: tuple[str, ...] = CHANNEL_NAMES
    ref_label: str = DEFAULT_REF_LABEL
    gnd_label: str = DEFAULT_GND_LABEL
    active_mask: int = 0xFF
    stim: tuple[dict[str, Any], ...] = ()
    channel_positions_m: tuple[tuple[float, float, float] | None, ...] = DEFAULT_CHANNEL_POSITIONS_M
    channel_types: tuple[str, ...] = _BRAIN_SYNC_CHANNEL_TYPES
    eog_derivation: dict[str, Any] | None = None

    @property
    def reference_details(self) -> dict[str, str]:
        return dict(BRAIN_SYNC_REFERENCE) if self.ref_label == DEFAULT_REF_LABEL else {}

    @property
    def ocular_derivation(self) -> dict[str, Any]:
        if self.eog_derivation is not None:
            return self.eog_derivation
        return {"EOG": {"kind": "single_ended", "placement": "left_eye_corner",
                        "reference": self.ref_label}} if self.eog_labels == ("EOG",) else {}

    def validate(self) -> None:
        try:
            eeg_labels = tuple(self.eeg)
        except TypeError as exc:
            raise ValueError("必须为 8 个采集通道提供非空电极标签。") from exc
        if len(eeg_labels) != 8 or any(not isinstance(label, str) or not label.strip() for label in eeg_labels):
            raise ValueError("必须为 8 个采集通道提供非空电极标签。")
        labels = [label.strip() for label in eeg_labels]
        normalized_labels = [label.casefold() for label in labels]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError("采集通道标签不能重复。")
        channel_types = tuple(str(value).strip().casefold() for value in self.channel_types)
        if len(channel_types) != len(labels) or any(value not in {"eeg", "eog"} for value in channel_types):
            raise ValueError("channel_types 必须为 8 个 eeg/eog 类型。")
        if not any(value == "eeg" for value in channel_types):
            raise ValueError("至少必须保留一个 EEG 通道。")
        if self.eog_derivation is not None:
            eog_labels = {label for label, kind in zip(labels, channel_types) if kind == "eog"}
            if not isinstance(self.eog_derivation, dict) or not set(self.eog_derivation) <= eog_labels:
                raise ValueError("EOG derivation must name measured EOG channels.")
            for derivation in self.eog_derivation.values():
                if (not isinstance(derivation, dict) or derivation.get("kind") not in {"single_ended", "bipolar"}
                        or any(not isinstance(derivation.get(key), str) or not derivation[key].strip()
                               for key in ("placement", "reference"))):
                    raise ValueError("EOG derivation requires kind, placement and reference.")
        if not isinstance(self.ref_label, str) or not isinstance(self.gnd_label, str):
            raise ValueError("REF 和 GND 标签必须是文本。")
        ref = self.ref_label.strip()
        gnd = self.gnd_label.strip()
        if not ref or not gnd:
            raise ValueError("REF 和 GND 标签不能为空。")
        normalized_ref = ref.casefold()
        normalized_gnd = gnd.casefold()
        if normalized_ref == normalized_gnd or normalized_ref in normalized_labels or normalized_gnd in normalized_labels:
            raise ValueError("REF、GND 不能与彼此或 EEG 通道标签重复。")
        if isinstance(self.active_mask, bool) or not isinstance(self.active_mask, int):
            raise ValueError("active_mask 必须是整数。")
        if self.active_mask < 0 or self.active_mask > 0xFF:
            raise ValueError("active_mask 必须是 0 到 255 之间的整数。")
        if self.active_mask == 0:
            raise ValueError("至少必须启用一个 EEG 通道。")
        if not any(
            kind == "eeg" and self.active_mask & (1 << index)
            for index, kind in enumerate(channel_types)
        ):
            raise ValueError("At least one scalp EEG channel must be active.")

        positions = tuple(self.channel_positions_m)
        if len(positions) != len(eeg_labels):
            raise ValueError("channel_positions_m 必须与 EEG 通道标签一一对应。")
        for kind, position in zip(channel_types, positions):
            if kind == "eog" and position is None:
                continue
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise ValueError("每个 channel_positions_m 必须是三维坐标。")
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in position
            ):
                raise ValueError("channel_positions_m 必须是有限数值坐标。")
            if max(abs(float(value)) for value in position) > 0.5:
                raise ValueError("channel_positions_m 必须使用米单位的 head 坐标。")
            if sum(float(value) ** 2 for value in position) <= 0.0:
                raise ValueError("channel_positions_m 不能是零坐标。")

        for item in self.stim:
            if not isinstance(item, dict):
                raise ValueError("刺激电极配置必须是对象。")
            if not {"label", "slot", "channel", "polarity"}.issubset(item):
                raise ValueError("刺激电极配置必须包含 label、slot、channel 和 polarity。")
            if not str(item["label"]).strip() or not str(item["polarity"]).strip():
                raise ValueError("刺激电极的 label 和 polarity 不能为空。")
            try:
                int(item["slot"])
                int(item["channel"])
            except (TypeError, ValueError) as exc:
                raise ValueError("刺激电极的 slot 和 channel 必须是整数。") from exc

    @property
    def active_indices(self) -> tuple[int, ...]:
        self.validate()
        return tuple(index for index in range(8) if self.active_mask & (1 << index))

    @property
    def active_count(self) -> int:
        return len(self.active_indices)

    @property
    def eeg_indices(self) -> tuple[int, ...]:
        self.validate()
        return tuple(index for index, value in enumerate(self.channel_types) if value.casefold() == "eeg")

    @property
    def eog_indices(self) -> tuple[int, ...]:
        self.validate()
        return tuple(index for index, value in enumerate(self.channel_types) if value.casefold() == "eog")

    @property
    def eeg_labels(self) -> tuple[str, ...]:
        return tuple(self.eeg[index] for index in self.eeg_indices)

    @property
    def eog_labels(self) -> tuple[str, ...]:
        return tuple(self.eeg[index] for index in self.eog_indices)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "brainsync-channel-montage/3",
            "channel_indexing": HARDWARE_CHANNEL_INDEXING,
            "active_mask": int(self.active_mask),
            "labels": list(self.eeg),
            "channel_types": list(self.channel_types),
            "eog_derivation": self.ocular_derivation,
            "eeg": [
                {"label": label, "channel": index, "ch_type": self.channel_types[index]}
                for index, label in enumerate(self.eeg)
            ],
            "channel_positions_m": [None if position is None else list(position) for position in self.channel_positions_m],
            "coordinate_frame": BRAIN_SYNC_DATA_CONTRACT.coordinate_frame,
            "units": BRAIN_SYNC_DATA_CONTRACT.units,
            "ref_label": self.ref_label,
            "reference_details": self.reference_details,
            "gnd_label": self.gnd_label,
            "stim": [dict(item) for item in self.stim],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChannelMontage":
        if not isinstance(payload, dict):
            raise ValueError("通道配置顶层必须是对象。")
        data = payload.get("channel_config", payload)
        if not isinstance(data, dict):
            raise ValueError("通道配置必须是对象。")
        if data.get("channel_indexing", HARDWARE_CHANNEL_INDEXING) != HARDWARE_CHANNEL_INDEXING:
            raise ValueError("通道编号必须使用硬件 0-based index。")
        eeg_items = data.get("eeg") or []
        if not isinstance(eeg_items, (list, tuple)) or any(not isinstance(item, dict) for item in eeg_items):
            raise ValueError("EEG 通道配置必须是对象数组。")
        labels = data.get("labels") or [item.get("label", "") for item in eeg_items]
        if data.get("labels") and eeg_items and list(labels) != [item.get("label", "") for item in eeg_items]:
            raise ValueError("labels and acquisition assignments disagree.")
        if not isinstance(labels, (list, tuple)) or len(labels) != 8:
            raise ValueError("通道配置必须包含 8 个采集通道。")
        raw_channel_types = data.get("channel_types")
        if raw_channel_types is None:
            raw_channel_types = (
                [item.get("ch_type", item.get("type")) for item in eeg_items]
                if eeg_items
                else [None] * len(labels)
            )
        if not isinstance(raw_channel_types, (list, tuple)) or len(raw_channel_types) != len(labels):
            raise ValueError("通道配置必须包含与 labels 对应的 channel_types。")
        if any(value is None for value in raw_channel_types):
            raise ValueError("Explicit EEG/EOG channel types are required; slot numbers do not identify electrodes.")
        channel_types = tuple(str(value).strip().casefold() for value in raw_channel_types)
        if (
            data.get("coordinate_frame", BRAIN_SYNC_DATA_CONTRACT.coordinate_frame)
            != BRAIN_SYNC_DATA_CONTRACT.coordinate_frame
            or data.get("units", BRAIN_SYNC_DATA_CONTRACT.units) != BRAIN_SYNC_DATA_CONTRACT.units
        ):
            raise ValueError("通道坐标必须使用 head 坐标系和米单位。")
        raw_positions = data.get("channel_positions_m", data.get("positions_m"))
        if raw_positions is None:
            try:
                positions = tuple(
                    None if kind == "eog" else STANDARD_CHANNEL_POSITIONS_M[str(label).strip().lower()]
                    for label, kind in zip(labels, channel_types)
                )
            except KeyError as exc:
                raise ValueError(
                    "旧通道配置缺少 channel_positions_m，且无法从标准电极标签推导坐标。"
                ) from exc
        elif isinstance(raw_positions, (list, tuple)):
            positions = tuple(None if position is None else tuple(float(value) for value in position) for position in raw_positions)
        else:
            raise ValueError("channel_positions_m 必须是三维坐标数组。")
        ref = data.get("ref_label")
        gnd = data.get("gnd_label")
        if isinstance(data.get("ref"), dict):
            ref = data["ref"].get("label", ref)
        if isinstance(data.get("gnd"), dict):
            gnd = data["gnd"].get("label", gnd)
        raw_active_mask = data.get("active_mask", 0xFF)
        if isinstance(raw_active_mask, bool):
            raise ValueError("active_mask 必须是整数。")
        try:
            active_mask = int(raw_active_mask)
        except (TypeError, ValueError) as exc:
            raise ValueError("active_mask 必须是整数。") from exc
        raw_stim = data.get("stim", [])
        if not isinstance(raw_stim, (list, tuple)):
            raise ValueError("刺激电极配置必须是数组。")
        montage = cls(
            eeg=tuple(str(label).strip() for label in labels),
            ref_label=str(ref or DEFAULT_REF_LABEL).strip(),
            gnd_label=str(gnd or DEFAULT_GND_LABEL).strip(),
            active_mask=active_mask,
            stim=tuple(dict(item) for item in raw_stim if isinstance(item, dict)),
            channel_positions_m=positions,
            channel_types=channel_types,
            eog_derivation=data.get("eog_derivation", {}),
        )
        montage.validate()
        if data.get("reference_details", montage.reference_details) != montage.reference_details:
            raise ValueError("REF label and physical reference details disagree.")
        return montage


def default_channel_positions_m(labels: tuple[str, ...] | list[str]) -> tuple[tuple[float, float, float] | None, ...]:
    """Return standard head-frame metre coordinates in the supplied label order."""

    try:
        return tuple(
            None if str(label).strip().upper() == "EOG"
            else STANDARD_CHANNEL_POSITIONS_M[str(label).strip().lower()] for label in labels
        )
    except KeyError as exc:
        raise ValueError("标准通道标签没有可用的 channel_positions_m。") from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def edf_birthdate(value: str) -> str:
    """Normalize user input to the EDF header date format."""
    if not isinstance(value, str):
        raise ValueError("出生日期格式应为 YYYY-MM-DD。")
    text = value.strip()
    if not text:
        return EDF_UNKNOWN_BIRTHDATE
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%d-%b-%Y").upper()
        except ValueError:
            continue
    raise ValueError("出生日期格式应为 YYYY-MM-DD。")


@dataclass
class ExperimentConfig:
    subject_id: str = "P001"
    age: str = ""
    sex: str = "unknown"
    birthdate: str = ""
    blocks: int = 10
    digits_per_block: int = 45
    stimulus_ms: int = 300
    soa_ms: int = 1500
    rest_sec: int = 20
    random_seed: int = 20260824
    screen_index: int = 1
    save_dir: str = "data"
    model_profile: str = "未选择模型"
    notes: str = ""
    instruction_text: str = "请注视屏幕中央的数字。看到目标数字时保持专注，不要移动身体。"

    def validate_protocol(self) -> None:
        vocabulary_size = len(DIGIT_VOCAB)
        integer_fields = {
            "Block 数": self.blocks,
            "每 Block 数字数": self.digits_per_block,
            "数字呈现时间": self.stimulus_ms,
            "SOA": self.soa_ms,
            "Block 休息时间": self.rest_sec,
            "随机种子": self.random_seed,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields.values()):
            raise ValueError("实验次数与计时参数必须是整数。")
        if self.blocks <= 0:
            raise ValueError("Block 数必须大于 0。")
        if self.digits_per_block <= 0 or self.digits_per_block % vocabulary_size:
            raise ValueError(
                f"每 Block 数字数必须是 {vocabulary_size} 的正整数倍，"
                "以保证每个候选数字重复次数完全一致。"
            )
        if self.stimulus_ms <= 0:
            raise ValueError("数字呈现时间必须大于 0。")
        if self.soa_ms < self.stimulus_ms:
            raise ValueError("刺激起始间隔（SOA）不能小于数字呈现时间。")
        if self.rest_sec < 0:
            raise ValueError("Block 休息时间不能小于 0。")
        if self.random_seed < 0:
            raise ValueError("随机种子不能小于 0。")

    @property
    def repetitions_per_digit(self) -> int:
        self.validate_protocol()
        return self.digits_per_block // len(DIGIT_VOCAB)

    def to_dict(self) -> dict[str, Any]:
        self.validate_protocol()
        payload = asdict(self)
        payload["digit_vocab"] = list(DIGIT_VOCAB)
        payload["sequence_policy"] = "balanced_random_permutation_cycles"
        payload["model_epoch_support"] = MODEL_EPOCH_SUPPORT.record()
        payload["repetitions_per_digit"] = self.repetitions_per_digit
        payload["target_label_status"] = "pending_post_experiment_confirmation"
        payload["created_utc"] = utc_now()
        return payload


@dataclass
class ModelTaskConfig:
    task: str = "train"
    model_manifest: str = ""
    python_executable: str = ""
    entrypoint: str = ""
    arguments: str = ""
    output_dir: str = "models/runs"
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_utc"] = utc_now()
        return payload


@dataclass(frozen=True)
class ModelTaskOption:
    key: str
    label: str
    description: str


MODEL_TASK_OPTIONS = (
    ModelTaskOption("preprocess", "生成模型输入缓存", "将自动选择的 BIDS raw 会话转换为模型输入缓存"),
    ModelTaskOption("train", "训练模型", "使用自动选择的数据会话训练模型"),
    ModelTaskOption("infer", "使用模型推理", "使用已有模型处理自动选择的数据会话"),
    ModelTaskOption("evaluate", "评估模型效果", "对自动选择的数据会话生成模型评估结果"),
)
MODEL_TASK_KEYS = frozenset(option.key for option in MODEL_TASK_OPTIONS)


@dataclass(frozen=True)
class ModelParameterOption:
    key: str
    label: str
    arguments: str = ""
    description: str = ""
    tasks: tuple[str, ...] = ()


DEFAULT_MODEL_PARAMETER_OPTION = ModelParameterOption(
    "default",
    "使用模型默认参数",
    description="由模型入口自行决定训练或推理参数",
)


def parse_model_parameter_options(value: Any) -> tuple[ModelParameterOption, ...]:
    """Validate parameter menus declared by a model manifest."""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("模型 manifest 的 parameter_options 必须是数组")
    options: list[ModelParameterOption] = []
    keys: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("模型 manifest 的参数方案必须是对象")
        key = item.get("key")
        label = item.get("label")
        arguments = item.get("arguments", "")
        description = item.get("description", "")
        tasks = item.get("tasks", ())
        if not isinstance(key, str) or not key.strip():
            raise ValueError("模型 manifest 的参数方案缺少 key")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("模型 manifest 的参数方案缺少 label")
        if not isinstance(arguments, str) or not isinstance(description, str):
            raise ValueError("模型 manifest 的参数方案文本字段必须是字符串")
        if not isinstance(tasks, (list, tuple)) or any(
            not isinstance(task, str) or task not in MODEL_TASK_KEYS for task in tasks
        ):
            raise ValueError("模型 manifest 的参数方案 tasks 无效")
        normalized_key = key.strip()
        if normalized_key in keys:
            raise ValueError(f"模型 manifest 的参数方案 key 重复: {normalized_key!r}")
        keys.add(normalized_key)
        options.append(
            ModelParameterOption(
                key=normalized_key,
                label=label.strip(),
                arguments=arguments,
                description=description.strip(),
                tasks=tuple(tasks),
            )
        )
    return tuple(options)


@dataclass
class TrialRecord:
    session_id: str
    block_id: int
    trial_index: int
    digit: int
    eeg_time_seconds: float | None = None
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceSnapshot:
    transport: str = "serial"
    firmware: str = "--"
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    gain: str = "Gain24"
    configuration_verification: str = "not_configured"
    configuration_note: str = ""
    packets: int = 0
    lost_packets: int = 0
    last_error: str = ""
    last_eeg_monotonic: float | None = None
    stream_started_monotonic: float | None = None

    @property
    def loss_rate(self) -> float:
        total = self.packets + self.lost_packets
        return (self.lost_packets / total * 100.0) if total else 0.0
