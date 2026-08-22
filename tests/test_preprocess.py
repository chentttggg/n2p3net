"""data/preprocess.py 的两级测试（冒烟 + 语义）。

冒烟测试：验证形状、dtype、不报错。
语义测试：用已知答案的合成样例验证「语义正确」（重采样/通道映射/高通去 DC/伪迹剔除/点数对齐）。
"""

import numpy as np
import pytest

import mne

from data.preprocess import (
    STANDARD_CHANNELS,
    _canonical,
    highpass,
    map_channels,
    preprocess,
    reject_epochs,
    resample,
)


# --------------------------------------------------------------------------- #
# 合成数据工具
# --------------------------------------------------------------------------- #
def make_raw(sfreq=512.0, n_seconds=20.0, ch_names=None, amp=10e-6, dc=0.0, seed=0):
    """构造合成 Raw。amp 默认 10 μV，确保不会被 150 μV 伪迹阈值误剔除。"""
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
    return np.column_stack([samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])


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
    res = preprocess(raw, events)

    assert res.data.ndim == 3
    assert res.data.shape[1] == 8
    assert res.data.shape[2] == 256
    assert res.data.dtype == np.float32
    assert res.channel_mask.dtype == bool
    assert res.channel_mask.all()          # 合成数据含全部 8 通道
    assert res.sfreq == 256.0
    assert res.tmin == -0.2
    assert res.data.shape[0] >= 1          # 至少切出一个 epoch
    assert isinstance(res.n_epochs, int) and isinstance(res.n_times, int)


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


def test_preprocess_missing_channel_nan_and_mask():
    """缺失通道：mask 标记 False，data 对应位置填 NaN，存在通道非 NaN。"""
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]  # 缺 P3, P4, PO8
    raw = make_raw(sfreq=512.0, ch_names=present)
    events = make_events(sfreq=512.0)

    res = preprocess(raw, events)

    # 标准顺序：Fz,Cz,P3,Pz,P4,PO7,PO8,Oz → present 掩码
    expected_mask = np.array([True, True, False, True, False, True, False, True])
    assert np.array_equal(res.channel_mask, expected_mask)
    assert res.n_present == 5

    # 缺失通道全 NaN
    for idx in (2, 4, 6):  # P3, P4, PO8
        assert np.isnan(res.data[:, idx, :]).all()
    # 存在通道非 NaN
    for idx in (0, 1, 3, 5, 7):
        assert not np.isnan(res.data[:, idx, :]).any()


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

    # 跳过高通与伪迹剔除（常数指纹会被高通滤掉/被阈值剔除）
    res = preprocess(raw, events, l_freq=None, reject_threshold=None)

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


def test_reject_epochs_drops_outlier():
    """伪迹剔除：含超阈值样本的 epoch 被剔除，其余保留。"""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 8, 256)) * 20e-6
    data[2, 3, 100] = 1.0  # 第 2 个 epoch 的第 3 通道注入 1V 伪迹

    clean, dropped = reject_epochs(data, threshold=150e-6)

    assert np.array_equal(dropped, np.array([2]))
    assert clean.shape[0] == 4


def test_reject_epochs_ignores_nan():
    """伪迹剔除：缺失通道的 NaN 不应触发剔除。"""
    data = np.full((3, 8, 256), 1e-6, dtype=float)
    data[:, 2, :] = np.nan  # 第 2 通道全 NaN（缺失通道）
    clean, dropped = reject_epochs(data, threshold=150e-6)
    assert len(dropped) == 0
    assert clean.shape[0] == 3


def test_preprocess_copy_semantics():
    """copy=True 时不得污染调用方的 raw。"""
    raw = make_raw(sfreq=512.0)
    orig_sfreq = raw.info["sfreq"]
    orig_channels = list(raw.ch_names)
    events = make_events(sfreq=512.0)

    preprocess(raw, events, copy=True)

    assert raw.info["sfreq"] == orig_sfreq
    assert list(raw.ch_names) == orig_channels


def test_preprocess_n_times_alignment():
    """D-n-times-align：默认输出 256 点；n_times=None 保留 MNE 自然点数（257）。"""
    raw = make_raw(sfreq=256.0)  # 已是目标采样率，不触发重采样
    events = make_events(sfreq=256.0)

    res = preprocess(raw, events)
    assert res.data.shape[2] == 256

    res2 = preprocess(raw, events, n_times=None)
    assert res2.data.shape[2] == 257  # (0.8-(-0.2))*256 + 1


def test_preprocess_raises_on_bad_events():
    raw = make_raw(sfreq=512.0)
    bad = np.ones((5, 2))  # 非 (n,3) 格式
    with pytest.raises(ValueError):
        preprocess(raw, bad)


def test_map_channels_no_standard_raises():
    raw = make_raw(sfreq=512.0, ch_names=["X1", "Y2", "Z3"])  # 无任何标准通道
    with pytest.raises(ValueError):
        map_channels(raw)


def test_preprocess_event_indices():
    """event_indices：无伪迹剔除时，最终 epoch 对应原始 events 行索引（标签对齐的依据）。"""
    raw = make_raw(sfreq=512.0)
    events = make_events(sfreq=512.0)
    res = preprocess(raw, events, reject_threshold=None)

    assert res.event_indices.shape[0] == res.data.shape[0]
    assert np.array_equal(res.event_indices, np.arange(len(events)))  # 全部 event 均在边界内


def test_preprocess_event_indices_with_reject():
    """event_indices：伪迹剔除后，被剔除 epoch 对应的 events 行索引被移除。"""
    sfreq = 256.0  # 与目标采样率一致，避免 resample 的 anti-aliasing 扩散瞬态
    n_seconds = 20.0
    rng = np.random.default_rng(0)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((8, n_times)) * 10e-6
    info = mne.create_info(list(STANDARD_CHANNELS), sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    events = make_events(sfreq, n_seconds)
    raw._data[:, events[1, 0] : events[1, 0] + 20] = 1.0  # 第 2 个 event 注入伪迹

    res = preprocess(raw, events, l_freq=None)  # 默认 reject 150 μV；跳过高通避免瞬态扩散干扰 event_indices 验证

    assert 1 not in res.event_indices  # 第 2 个 event 被剔除
    assert res.event_indices.shape[0] == res.data.shape[0]
