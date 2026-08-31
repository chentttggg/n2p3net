"""data/preprocess.py 的两级测试（冒烟 + 语义）。

冒烟测试：验证形状、dtype、不报错。
语义测试：用已知答案的合成样例验证「语义正确」（重采样/通道映射/高通去 DC/点数对齐）。
"""

import mne
import numpy as np
import pytest

from data.channel import STANDARD_CHANNELS
from data.contract import CAUSAL_IIR_INITIAL_STATE
from data.preprocess import (
    _canonical,
    apply_trial_baseline,
    filter_continuous,
    highpass,
    map_channels,
    preprocess,
    resample,
)


# --------------------------------------------------------------------------- #
# 合成数据工具
# --------------------------------------------------------------------------- #
def make_raw(sfreq=512.0, n_seconds=20.0, ch_names=None, amp=10e-6, dc=0.0, seed=0):
    """Construct synthetic finite EEG samples."""
    if ch_names is None:
        ch_names = list(STANDARD_CHANNELS)
    rng = np.random.default_rng(seed)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((len(ch_names), n_times)) * amp + dc
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def make_events(sfreq=512.0, n_seconds=20.0, first=2.0, step=1.0):
    """构造刺激 events（单一类型 id=1），保证每个 epoch 都落在数据范围内。"""
    times = np.arange(first, n_seconds - 1.0, step)  # 留 1s 余量给 tmax=0.8
    samples = np.round(times * sfreq).astype(int)
    return np.column_stack(
        [samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)]
    )


# --------------------------------------------------------------------------- #
# 单元测试：通道名规范化
# --------------------------------------------------------------------------- #
def test_canonical_channel_name():
    assert _canonical("Fz") == "FZ"
    assert _canonical(" pz ") == "PZ"
    assert _canonical("Pz-A2") == "PZ"
    assert _canonical("PO7") == "PO7"
    assert _canonical("Cz_ref") == "CZ"


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_preprocess_smoke_shape_dtype():
    raw = make_raw(sfreq=512.0)
    events = make_events(sfreq=512.0)
    res = preprocess(raw, events, channels=STANDARD_CHANNELS)

    assert res.data.ndim == 3
    assert res.data.shape[1] == 8
    assert res.data.shape[2] == 128
    assert res.data.dtype == np.float32
    assert res.channel_mask.dtype == bool
    assert res.channel_mask.all()  # 合成数据含全部 8 通道
    assert res.sfreq == 128.0
    assert res.tmin == -0.2
    assert res.data.shape[0] >= 1  # 至少切出一个 epoch
    assert isinstance(res.n_epochs, int) and isinstance(res.n_times, int)
def test_preprocess_forward_phase_marks_causal_and_zero_phase_does_not():
    raw = make_raw(sfreq=512.0)
    events = make_events(sfreq=512.0)

    offline = preprocess(raw, events, channels=STANDARD_CHANNELS)
    assert offline.online_causal is False

    causal = preprocess(
        raw,
        events,
        channels=STANDARD_CHANNELS,
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )
    assert causal.online_causal is True
    assert np.isfinite(causal.data).all()





def test_preprocess_prefers_embedded_digitization_over_default_template():
    raw = make_raw(sfreq=100.0, ch_names=["X1", "X2"])
    positions = {"X1": (-0.03, 0.02, 0.08), "X2": (0.03, -0.02, 0.09)}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head"))

    result = preprocess(
        raw,
        make_events(sfreq=100.0),
        sfreq=100.0,
        l_freq=None,
        n_times=100,
        reject_threshold=None,
    )

    assert result.coordinate_registration.source == "individual_digitization"
    assert result.coordinate_registration.method == "identity_head"
    assert np.allclose(result.channel_positions_m, np.asarray(list(positions.values())))


# --------------------------------------------------------------------------- #
# 语义测试
# --------------------------------------------------------------------------- #
def test_resample_scales_events():
    """D-resample-events：重采样后 events 的 sample 索引必须按比例缩放。"""
    raw = make_raw(sfreq=512.0, n_seconds=20.0)
    events = make_events(sfreq=512.0, n_seconds=20.0)

    raw2, ev2 = resample(raw, events, sfreq=256.0)

    assert raw2.info["sfreq"] == 256.0
    # 第一个 event 原在 2.0s*512=1024 sample，重采样后应为 2.0s*256=512
    assert ev2[0, 0] == 512
    # 全部 event 均按比例缩放
    expected = np.round(events[:, 0] * 256.0 / 512.0).astype(int)
    assert np.array_equal(ev2[:, 0], expected)


def test_preprocess_rejects_missing_explicit_channel():
    """显式布局缺失必须失败，禁止 NaN/零填充伪装成物理观测。"""
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]  # 缺 P3, P4, PO8
    raw = make_raw(sfreq=512.0, ch_names=present)
    events = make_events(sfreq=512.0)
    with pytest.raises(ValueError, match="does not pad or substitute"):
        preprocess(raw, events, channels=STANDARD_CHANNELS)


def test_preprocess_native_layout_keeps_only_observed_sensors():
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]
    raw = make_raw(sfreq=512.0, ch_names=present)
    events = make_events(sfreq=512.0)
    result = preprocess(raw, events, channels=None)
    assert result.channel_names == tuple(name.upper() for name in present)
    assert result.data.shape[1] == len(present)
    assert result.channel_mask.all()
    assert np.isfinite(result.data).all()


def test_preprocess_channel_reorder():
    """通道重排：乱序输入 → 输出严格按标准蒙太奇顺序（用常数指纹验证）。"""
    order = ["Pz", "Fz", "Oz", "Cz", "P4", "PO7", "P3", "PO8"]  # 乱序
    finger = {"Fz": 1, "Cz": 2, "P3": 3, "Pz": 4, "P4": 5, "PO7": 6, "PO8": 7, "Oz": 8}
    n_times = int(20.0 * 256)
    data = np.zeros((8, n_times))
    for i, ch in enumerate(order):
        data[i, :] = finger[ch]
    info = mne.create_info(order, 256.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    events = make_events(sfreq=256.0, n_seconds=20.0)

    # Skip high-pass; it would remove the constant channel fingerprint.
    res = preprocess(
        raw,
        events,
        l_freq=None,
        h_freq=None,
        baseline_mode="none",
        reject_threshold=None,
        channels=STANDARD_CHANNELS,
    )

    for i, std in enumerate(STANDARD_CHANNELS):
        assert np.allclose(res.data[:, i, :], finger[std], atol=1e-3), f"通道 {std} 顺序错误"


def test_highpass_removes_dc():
    """高通去漂移：纯 DC 100 μV 经 0.1Hz 高通后应被显著抑制（中间段均值 <5 μV）。"""
    n_times = int(30.0 * 512)
    data = np.ones((1, n_times)) * 100e-6
    info = mne.create_info(["Fz"], 512.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    raw = highpass(raw, l_freq=0.1)

    d = raw.get_data()[0]
    mid = d[n_times // 4 : 3 * n_times // 4]  # 避开滤波边界
    assert np.abs(mid.mean()) < 5e-6


def test_filter_continuous_applies_declared_lowpass():
    """Manifest h_freq must reach MNE rather than being stored as inert metadata."""
    sfreq = 256.0
    times = np.arange(int(8 * sfreq)) / sfreq
    data = np.sin(2 * np.pi * 5 * times) + np.sin(2 * np.pi * 60 * times)
    raw = mne.io.RawArray(
        data[None, :], mne.create_info(["Fz"], sfreq, ch_types="eeg"), verbose=False
    )

    filtered = filter_continuous(raw, l_freq=None, h_freq=20.0, copy=True)
    spectrum = np.abs(np.fft.rfft(filtered.get_data()[0]))
    frequencies = np.fft.rfftfreq(len(times), d=1.0 / sfreq)
    amp_5hz = spectrum[np.argmin(np.abs(frequencies - 5.0))]
    amp_60hz = spectrum[np.argmin(np.abs(frequencies - 60.0))]

    assert amp_60hz < 0.05 * amp_5hz


def test_forward_iir_steady_state_does_not_turn_dc_offset_into_startup_artifact():
    sfreq = 128.0
    data = np.full((1, int(sfreq * 12)), 3e-3, dtype=np.float64)
    raw = mne.io.RawArray(
        data, mne.create_info(["Cz"], sfreq, ch_types="eeg"), verbose=False
    )

    filtered = filter_continuous(
        raw,
        l_freq=0.1,
        h_freq=30.0,
        phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
        copy=True,
    ).get_data()[0]

    assert np.max(np.abs(filtered[: int(2 * sfreq)])) < 1e-9


def test_forward_iir_never_uses_a_future_impulse():
    sfreq = 128.0
    data = np.zeros((1, int(sfreq * 8)), dtype=np.float64)
    impulse = int(5 * sfreq)
    data[0, impulse] = 1.0
    raw = mne.io.RawArray(
        data, mne.create_info(["Cz"], sfreq, ch_types="eeg"), verbose=False
    )

    filtered = filter_continuous(
        raw,
        l_freq=0.1,
        h_freq=30.0,
        phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
        copy=True,
    ).get_data()[0]

    assert np.all(filtered[:impulse] == 0.0)


def test_continuous_filtering_and_epoch_crop_do_not_commute() -> None:
    """Counterexample: an impulse just outside the epoch affects a continuous filter."""

    sfreq = 128.0
    event_sample = 256
    epoch_start = event_sample - round(0.2 * sfreq)
    epoch_stop = event_sample + round(0.8 * sfreq)
    data = np.zeros((1, 512), dtype=float)
    data[0, epoch_start - 4] = 1.0
    raw = mne.io.RawArray(data, mne.create_info(["Fz"], sfreq, ch_types="eeg"), verbose=False)

    filtered = filter_continuous(raw, l_freq=2.0, h_freq=30.0, copy=True)
    filter_then_crop = filtered.get_data(start=epoch_start, stop=epoch_stop)
    crop_then_filter_input = raw.get_data(start=epoch_start, stop=epoch_stop)

    assert np.count_nonzero(crop_then_filter_input) == 0
    assert np.linalg.norm(filter_then_crop) > 1e-3


def test_preprocess_rejects_nonfinite_source_samples():
    raw = make_raw(sfreq=256.0)
    raw._data[0, 100] = np.nan
    events = make_events(sfreq=256.0)

    with pytest.raises(ValueError, match="never imputes"):
        preprocess(raw, events, l_freq=None, channels=STANDARD_CHANNELS)


def test_preprocess_rejects_retired_fixed_artifact_threshold():
    raw = make_raw(sfreq=256.0)
    events = make_events(sfreq=256.0)
    with pytest.raises(ValueError, match="Fixed absolute-voltage epoch rejection is retired"):
        preprocess(raw, events, reject_threshold=150e-6, channels=STANDARD_CHANNELS)


def test_preprocess_copy_semantics():
    """copy=True 时不得污染调用方的 raw。"""
    raw = make_raw(sfreq=512.0)
    orig_sfreq = raw.info["sfreq"]
    orig_channels = list(raw.ch_names)
    events = make_events(sfreq=512.0)

    preprocess(raw, events, copy=True, channels=STANDARD_CHANNELS)

    assert raw.info["sfreq"] == orig_sfreq
    assert list(raw.ch_names) == orig_channels


def test_preprocess_n_times_alignment():
    """The 128 Hz model cache uses a half-open one-second epoch."""
    raw = make_raw(sfreq=250.0)  # 已是目标采样率，不触发重采样
    events = make_events(sfreq=250.0)

    res = preprocess(raw, events, channels=STANDARD_CHANNELS)
    assert res.data.shape[2] == 128

    res2 = preprocess(raw, events, n_times=None, channels=STANDARD_CHANNELS)
    assert res2.data.shape[2] == 129  # (0.8-(-0.2))*128 + 1


def test_mean_only_baseline_executes_on_the_half_open_prestimulus_window() -> None:
    data = np.zeros((2, 2, 10), dtype=np.float32)
    data[0, 0] = np.arange(10, dtype=np.float32) + 10.0
    data[0, 1] = np.arange(10, dtype=np.float32) - 5.0
    data[1] = data[0] * 2.0

    transformed = apply_trial_baseline(
        data,
        sfreq=10.0,
        tmin=-0.2,
        baseline_mode="mean_only",
    )

    np.testing.assert_allclose(transformed[:, :, :2].mean(axis=2), 0.0, atol=1e-7)
    np.testing.assert_allclose(transformed[:, :, 2], data[:, :, 2] - data[:, :, :2].mean(axis=2))


def test_preprocess_raises_on_bad_events():
    raw = make_raw(sfreq=512.0)
    bad = np.ones((5, 2))  # 非 (n,3) 格式
    with pytest.raises(ValueError):
        preprocess(raw, bad)


def test_preprocess_rejects_fractional_event_samples() -> None:
    raw = make_raw(sfreq=100.0)
    events = np.array([[200.5, 0.0, 1.0]])
    with pytest.raises(ValueError, match="integer dtype"):
        preprocess(raw, events)


def test_map_channels_no_standard_raises():
    raw = make_raw(sfreq=512.0, ch_names=["X1", "Y2", "Z3"])  # 无任何标准通道
    with pytest.raises(ValueError, match="does not pad or substitute"):
        map_channels(raw, channels=STANDARD_CHANNELS)


def test_preprocess_event_indices():
    """event_indices align with original event rows when all rows are finite."""
    raw = make_raw(sfreq=512.0)
    events = make_events(sfreq=512.0)
    res = preprocess(raw, events, reject_threshold=None, channels=STANDARD_CHANNELS)

    assert res.event_indices.shape[0] == res.data.shape[0]
    assert np.array_equal(res.event_indices, np.arange(len(events)))  # 全部 event 均在边界内


def test_preprocess_event_indices_preserve_artifact_rows_for_fold_local_qc():
    """Ingress retains finite artifact rows; fold-local QC owns later handling."""
    sfreq = 256.0  # 与目标采样率一致，避免 resample 的 anti-aliasing 扩散瞬态
    n_seconds = 20.0
    rng = np.random.default_rng(0)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((8, n_times)) * 10e-6
    info = mne.create_info(list(STANDARD_CHANNELS), sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    events = make_events(sfreq, n_seconds)
    raw._data[:, events[1, 0] : events[1, 0] + 20] = 1.0  # 第 2 个 event 注入伪迹

    res = preprocess(
        raw,
        events,
        l_freq=None,
        channels=STANDARD_CHANNELS,
    )

    assert 1 in res.event_indices
    assert res.event_indices.shape[0] == res.data.shape[0]


def test_preprocess_preserves_boundary_and_ignored_event_reasons() -> None:
    raw = make_raw(sfreq=256.0, n_seconds=5.0)
    events = np.asarray(
        [
            [0, 0, 1],
            [2 * 256, 0, 1],
            [3 * 256, 0, 2],
        ],
        dtype=np.int64,
    )
    result = preprocess(
        raw,
        events,
        sfreq=256.0,
        l_freq=None,
        event_id=[1],
        reject_threshold=None,
        channels=STANDARD_CHANNELS,
    )

    assert result.event_statuses.tolist() == [
        "boundary_dropped",
        "available",
        "acquisition_rejected",
    ]
    assert "NO_DATA" in result.event_status_details[0]
    assert "IGNORED" in result.event_status_details[2]
    assert result.event_evidence_indices.tolist() == [-1, 0, -1]
