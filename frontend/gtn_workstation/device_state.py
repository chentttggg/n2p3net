from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeviceState(str, Enum):
    """User-visible lifecycle of the single BrainSync device session."""

    DISCONNECTED = "disconnected"
    READY = "ready"
    PREVIEWING = "previewing"
    RECORDING = "recording"
    IMPEDANCE = "impedance"
    RECOVERY_REQUIRED = "recovery_required"


class DeviceStateError(RuntimeError):
    """Raised when an operation is invalid for the current lifecycle state."""


_ALLOWED_TRANSITIONS: dict[DeviceState, frozenset[DeviceState]] = {
    DeviceState.DISCONNECTED: frozenset({DeviceState.READY, DeviceState.RECOVERY_REQUIRED}),
    DeviceState.READY: frozenset({
        DeviceState.DISCONNECTED,
        DeviceState.PREVIEWING,
        DeviceState.RECORDING,
        DeviceState.IMPEDANCE,
        DeviceState.RECOVERY_REQUIRED,
    }),
    DeviceState.PREVIEWING: frozenset({
        DeviceState.READY,
        DeviceState.DISCONNECTED,
        DeviceState.RECOVERY_REQUIRED,
    }),
    DeviceState.RECORDING: frozenset({
        DeviceState.READY,
        DeviceState.DISCONNECTED,
        DeviceState.RECOVERY_REQUIRED,
    }),
    DeviceState.IMPEDANCE: frozenset({
        DeviceState.READY,
        DeviceState.DISCONNECTED,
        DeviceState.RECOVERY_REQUIRED,
    }),
    DeviceState.RECOVERY_REQUIRED: frozenset({
        DeviceState.READY,
        DeviceState.DISCONNECTED,
    }),
}


@dataclass
class DeviceLifecycle:
    """Single source of truth for operations exposed to the workstation UI."""

    state: DeviceState = DeviceState.DISCONNECTED
    recovery_reason: str = ""

    @property
    def connected(self) -> bool:
        return self.state is not DeviceState.DISCONNECTED

    @property
    def requires_recovery(self) -> bool:
        return self.state is DeviceState.RECOVERY_REQUIRED

    def require(self, operation: str, *allowed: DeviceState) -> None:
        if self.state not in allowed:
            names = "、".join(state.value for state in allowed)
            raise DeviceStateError(
                f"设备当前处于 {self.state.value} 状态，不能执行 {operation}（允许状态: {names}）。"
            )

    def transition(self, target: DeviceState, recovery_reason: str = "") -> None:
        if target is self.state:
            if target is DeviceState.RECOVERY_REQUIRED and recovery_reason:
                self.recovery_reason = recovery_reason
            return
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise DeviceStateError(f"非法设备状态迁移: {self.state.value} -> {target.value}")
        self.state = target
        self.recovery_reason = recovery_reason if target is DeviceState.RECOVERY_REQUIRED else ""

    def force_disconnected(self) -> None:
        """Use only after the SDK handle has been released or deemed unusable."""

        self.state = DeviceState.DISCONNECTED
        self.recovery_reason = ""
