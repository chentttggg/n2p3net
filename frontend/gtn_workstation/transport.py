"""Transport differences and validated operator preferences for one device adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .models import DEFAULT_SAMPLE_RATE_HZ

# Packet batching is an SDK delivery unit, independent of the EEG sample rate.
EEG_CALLBACK_BATCH_SIZE = 250


@dataclass(frozen=True)
class TransportSpec:
    key: str
    label: str
    address_label: str
    address_hint: str
    opener: str
    default_address: str | None
    connection_callback: bool = False
    settle_seconds: float = 0.0
    allow_readback_timeout: bool = False

TRANSPORTS = (
    TransportSpec("serial", "串口 USB", "串口（可选）", "留空：SDK 自动发现", "open_brainsync_serial",
                  None),
    TransportSpec("ble", "BLE", "设备名（可选）", "默认 BrainSync", "open_brainsync_ble",
                  "BrainSync", connection_callback=True,
                  settle_seconds=1.0, allow_readback_timeout=True),
)
GAINS = (1, 2, 4, 6, 8, 12, 24)


def validate_acquisition_rate(sample_rate: int) -> None:
    if sample_rate != DEFAULT_SAMPLE_RATE_HZ:
        raise ValueError(f"工作站采集固定为 {DEFAULT_SAMPLE_RATE_HZ} Hz。")


def transport_spec(key: str) -> TransportSpec:
    for spec in TRANSPORTS:
        if spec.key == key:
            return spec
    raise ValueError(f"不支持的连接方式: {key}")


@dataclass
class ConnectionProfile:
    address: str = ""
    gain: int = 24


@dataclass
class ConnectionPreferences:
    transport: str = "ble"
    profiles: dict[str, ConnectionProfile] = field(
        default_factory=lambda: {spec.key: ConnectionProfile() for spec in TRANSPORTS},
    )

    @classmethod
    def from_dict(cls, payload: dict) -> ConnectionPreferences:
        if not isinstance(payload, dict) or payload.get("schema") != "brainsync-connection/1":
            raise ValueError("连接配置格式或版本无效")
        selected = transport_spec(payload.get("transport")).key
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError("缺少连接配置")
        result = cls(transport=selected)
        for spec in TRANSPORTS:
            record = profiles.get(spec.key)
            if not isinstance(record, dict):
                raise ValueError(f"缺少 {spec.label} 连接配置")
            address, gain = record.get("address"), record.get("gain")
            if not isinstance(address, str) or type(gain) is not int:
                raise ValueError(f"{spec.label} 连接配置字段类型无效")
            if gain not in GAINS:
                raise ValueError(f"增益无效: {gain}")
            result.profiles[spec.key] = ConnectionProfile(address.strip(), gain)
        return result

    def to_dict(self) -> dict:
        return {"schema": "brainsync-connection/1", "transport": self.transport,
                "profiles": {key: asdict(profile) for key, profile in self.profiles.items()}}
