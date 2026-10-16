from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfilt_zi, tf2sos
except ImportError:  # pragma: no cover - exercised only in incomplete deployments
    butter = filtfilt = iirnotch = sosfilt = sosfilt_zi = tf2sos = None


DEFAULT_BANDPASS_LOW_HZ = 1.0
DEFAULT_BANDPASS_HIGH_HZ = 40.0
DEFAULT_NOTCH_HZ = 50.0
DEFAULT_NOTCH_Q = 30.0
DEFAULT_BANDPASS_ORDER = 4
DEFAULT_EOG_BANDPASS_LOW_HZ = 0.1
DEFAULT_EOG_BANDPASS_HIGH_HZ = 10.0


@dataclass(frozen=True)
class RealtimeFilterConfig:
    """Software filter settings used for the live preview only."""

    bandpass_enabled: bool = True
    bandpass_low_hz: float = DEFAULT_BANDPASS_LOW_HZ
    bandpass_high_hz: float = DEFAULT_BANDPASS_HIGH_HZ
    notch_enabled: bool = True
    notch_hz: float = DEFAULT_NOTCH_HZ
    notch_q: float = DEFAULT_NOTCH_Q

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "live_preview_only",
            "implementation": "scipy.signal",
            "bandpass": {
                "enabled": self.bandpass_enabled,
                "low_hz": self.bandpass_low_hz,
                "high_hz": self.bandpass_high_hz,
                "order": DEFAULT_BANDPASS_ORDER,
            },
            "eog_bandpass": {
                "enabled_with_bandpass": self.bandpass_enabled,
                "low_hz": DEFAULT_EOG_BANDPASS_LOW_HZ,
                "high_hz": DEFAULT_EOG_BANDPASS_HIGH_HZ,
                "order": DEFAULT_BANDPASS_ORDER,
            },
            "notch": {
                "enabled": self.notch_enabled,
                "frequency_hz": self.notch_hz,
                "q": self.notch_q,
            },
        }


class FilterConfigurationError(ValueError):
    """Raised when filter settings are incompatible with the sample rate."""


def validate_realtime_filter_config(sample_rate_hz: float, config: RealtimeFilterConfig) -> None:
    try:
        sample_rate = float(sample_rate_hz)
    except (TypeError, ValueError) as exc:
        raise FilterConfigurationError("采样率必须是正数。") from exc
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise FilterConfigurationError("采样率必须是正数。")
    nyquist = sample_rate / 2.0
    if config.bandpass_enabled:
        try:
            bandpass_low = float(config.bandpass_low_hz)
            bandpass_high = float(config.bandpass_high_hz)
        except (TypeError, ValueError) as exc:
            raise FilterConfigurationError("带通截止频率必须是有限数字。") from exc
        if not np.isfinite(bandpass_low) or not np.isfinite(bandpass_high):
            raise FilterConfigurationError("带通截止频率必须是有限数字。")
        if bandpass_low <= 0 or bandpass_low >= bandpass_high:
            raise FilterConfigurationError("带通下限必须小于上限，且大于 0 Hz。")
        if bandpass_high >= nyquist:
            raise FilterConfigurationError(f"带通上限必须小于 Nyquist 频率 {nyquist:g} Hz。")
        if DEFAULT_EOG_BANDPASS_HIGH_HZ >= nyquist:
            raise FilterConfigurationError(
                f"EOG 带通上限必须小于 Nyquist 频率 {nyquist:g} Hz。"
            )
    if config.notch_enabled:
        try:
            notch_hz = float(config.notch_hz)
            notch_q = float(config.notch_q)
        except (TypeError, ValueError) as exc:
            raise FilterConfigurationError("工频陷波参数必须是有限数字。") from exc
        if not np.isfinite(notch_hz) or not np.isfinite(notch_q):
            raise FilterConfigurationError("工频陷波参数必须是有限数字。")
        if notch_hz <= 0 or notch_hz >= nyquist:
            raise FilterConfigurationError(f"工频频率必须位于 0 到 Nyquist 频率 {nyquist:g} Hz 之间。")
        if notch_q <= 0:
            raise FilterConfigurationError("工频陷波 Q 值必须大于 0。")


class RealtimeEEGFilter:
    """Stateful preview filter that uses EEG and EOG-specific band limits."""

    def __init__(
        self,
        sample_rate_hz: float,
        n_channels: int = 8,
        channel_types: Sequence[str] | None = None,
    ):
        if n_channels <= 0:
            raise ValueError("滤波器通道数必须大于 0。")
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_channels = int(n_channels)
        self.channel_types = tuple(
            str(value).strip().casefold() for value in (channel_types or ("eeg",) * n_channels)
        )
        if len(self.channel_types) != self.n_channels or any(
            value not in {"eeg", "eog"} for value in self.channel_types
        ):
            raise ValueError("滤波器 channel_types 必须与通道数对应，且只能使用 eeg/eog。")
        self.config = RealtimeFilterConfig()
        self.available = all(dependency is not None for dependency in (butter, iirnotch, sosfilt, sosfilt_zi, tf2sos))
        self._sos = {}
        self._states = [None] * self.n_channels
        self.configure(sample_rate_hz, self.config)

    def configure(self, sample_rate_hz: float, config: RealtimeFilterConfig) -> None:
        validate_realtime_filter_config(sample_rate_hz, config)
        filters = {}
        if self.available:
            for kind in set(self.channel_types):
                sections = []
                if config.notch_enabled:
                    sections.append(tf2sos(*iirnotch(config.notch_hz, config.notch_q, fs=sample_rate_hz)))
                if config.bandpass_enabled:
                    band = ([DEFAULT_EOG_BANDPASS_LOW_HZ, DEFAULT_EOG_BANDPASS_HIGH_HZ] if kind == "eog"
                            else [config.bandpass_low_hz, config.bandpass_high_hz])
                    sections.append(butter(DEFAULT_BANDPASS_ORDER, band, btype="bandpass", fs=sample_rate_hz, output="sos"))
                filters[kind] = np.vstack(sections) if sections else None
        self.sample_rate_hz = float(sample_rate_hz)
        self.config = replace(config)
        self._sos = filters
        self.reset()

    def reset(self) -> None:
        self._states = [None] * self.n_channels

    def set_channel_types(self, channel_types: Sequence[str]) -> None:
        normalized = tuple(str(value).strip().casefold() for value in channel_types)
        if len(normalized) != self.n_channels or any(
            value not in {"eeg", "eog"} for value in normalized
        ):
            raise ValueError("滤波器 channel_types 必须与通道数对应，且只能使用 eeg/eog。")
        self.channel_types = normalized
        self.configure(self.sample_rate_hz, self.config)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Filter an ``(n_samples, n_channels)`` batch while retaining state."""

        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(
                f"EEG 滤波输入必须是 (n_samples, {self.n_channels})，实际为 {values.shape}。"
            )
        if values.shape[0] == 0 or not self.available:
            return values.copy()

        if not np.isfinite(values).all():
            raise ValueError("EEG preview input must contain only finite samples.")
        filtered = values.copy()
        states = list(self._states)
        for channel, kind in enumerate(self.channel_types):
            sos = self._sos.get(kind)
            if sos is None:
                continue
            initial = states[channel]
            if initial is None:
                initial = sosfilt_zi(sos) * values[0, channel]
            filtered[:, channel], states[channel] = sosfilt(sos, values[:, channel], zi=initial)
        if not np.isfinite(filtered).all():
            raise ValueError("EEG preview filter produced non-finite samples.")
        self._states = states
        return filtered


@dataclass(frozen=True)
class OfflinePreprocessingConfig:
    """Analysis filters and artifact policy for continuous recordings."""

    eeg_low_hz: float = 0.5
    eeg_high_hz: float = 40.0
    eeg_filter_order: int = 4
    remove_eog_artifacts: bool = True


@dataclass(frozen=True)
class EOGArtifactCorrection:
    """Outputs kept together so algorithms cannot lose the original EOG trace."""

    corrected_data: np.ndarray
    eeg_data: np.ndarray
    eog_data: np.ndarray
    eeg_indices: tuple[int, ...]
    eog_indices: tuple[int, ...]
    regression_coefficients: np.ndarray
    method: str


def _channel_indices(channel_types: Sequence[str], expected_channels: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    normalized = tuple(str(value).strip().casefold() for value in channel_types)
    if len(normalized) != expected_channels or any(value not in {"eeg", "eog"} for value in normalized):
        raise ValueError("channel_types 必须与数据列数对应，且只能使用 eeg/eog。")
    eeg_indices = tuple(index for index, value in enumerate(normalized) if value == "eeg")
    eog_indices = tuple(index for index, value in enumerate(normalized) if value == "eog")
    if not eeg_indices:
        raise ValueError("预处理至少需要一个 EEG 通道。")
    return eeg_indices, eog_indices


def remove_eog_artifacts(
    data: np.ndarray,
    *,
    eeg_indices: Sequence[int],
    eog_indices: Sequence[int],
) -> EOGArtifactCorrection:
    """Regress centered EOG activity from EEG while retaining raw EOG samples."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("EOG 校正输入必须是非空的二维样本矩阵。")
    eeg = tuple(int(index) for index in eeg_indices)
    eog = tuple(int(index) for index in eog_indices)
    if not eeg or any(index < 0 or index >= values.shape[1] for index in (*eeg, *eog)):
        raise ValueError("EEG/EOG 通道索引超出数据范围。")
    corrected = values.copy()
    coefficients = np.zeros((len(eeg), len(eog)), dtype=np.float64)
    if eog:
        predictors = values[:, eog] - np.mean(values[:, eog], axis=0, keepdims=True)
        if np.any(np.ptp(predictors, axis=0) > 0.0):
            coefficients, _, _, _ = np.linalg.lstsq(predictors, values[:, eeg], rcond=None)
            corrected[:, eeg] -= predictors @ coefficients
    return EOGArtifactCorrection(
        corrected_data=corrected,
        eeg_data=corrected[:, eeg].copy(),
        eog_data=values[:, eog].copy(),
        eeg_indices=eeg,
        eog_indices=eog,
        regression_coefficients=coefficients.T,
        method="centered_eog_regression" if eog else "no_eog_channel",
    )


def preprocess_eeg_data(
    data: np.ndarray,
    sample_rate_hz: float,
    channel_types: Sequence[str],
    config: OfflinePreprocessingConfig | None = None,
) -> EOGArtifactCorrection:
    """Filter EEG channels, leave EOG raw, then apply EOG regression correction."""

    settings = config or OfflinePreprocessingConfig()
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("预处理输入必须是非空且有限的二维样本矩阵。")
    try:
        sample_rate = float(sample_rate_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("预处理采样率必须是正数。") from exc
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("预处理采样率必须是正数。")
    eeg_indices, eog_indices = _channel_indices(channel_types, values.shape[1])
    prepared = values.copy()
    if filtfilt is not None:
        nyquist = sample_rate / 2.0
        if not 0.0 < settings.eeg_low_hz < settings.eeg_high_hz < nyquist:
            raise ValueError("EEG 预处理带通范围必须位于 Nyquist 频率内。")
        numerator, denominator = butter(
            settings.eeg_filter_order,
            [settings.eeg_low_hz, settings.eeg_high_hz],
            btype="bandpass",
            fs=sample_rate,
        )
        for index in eeg_indices:
            prepared[:, index] = filtfilt(numerator, denominator, values[:, index])
    if not settings.remove_eog_artifacts:
        eog_result = remove_eog_artifacts(
            prepared, eeg_indices=eeg_indices, eog_indices=()
        )
        return replace(eog_result, eog_data=values[:, eog_indices].copy(), method="filter_only")
    return remove_eog_artifacts(
        prepared, eeg_indices=eeg_indices, eog_indices=eog_indices
    )
