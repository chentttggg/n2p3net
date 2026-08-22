"""data/dataset.py 的两级测试（冒烟 + 语义）。

语义重点：标签对齐（伪迹剔除后标签不错位）、元数据嵌入、文件格式分派。
"""

import numpy as np
import pytest

import mne

from data.dataset import SubjectData, build_subject, load_dataset, read_raw

STD = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]


# --------------------------------------------------------------------------- #
# 合成数据工具（与 test_preprocess 相同，测试模块保持自包含）
# --------------------------------------------------------------------------- #
def make_raw(sfreq=512.0, n_seconds=20.0, ch_names=None, amp=10e-6, seed=0):
    if ch_names is None:
        ch_names = list(STD)
    rng = np.random.default_rng(seed)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((len(ch_names), n_times)) * amp
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def make_events(sfreq=512.0, n_seconds=20.0, first=2.0, step=1.0):
    times = np.arange(first, n_seconds - 1.0, step)  # 留 1s 余量给 tmax=0.8
    samples = np.round(times * sfreq).astype(int)
    return np.column_stack([samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)])


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_build_subject_shape():
    raw = make_raw()
    events = make_events()
    s = build_subject(raw, events, labels=np.arange(len(events)))

    assert isinstance(s, SubjectData)
    assert s.data.shape[1] == 8
    assert s.data.shape[2] == 256
    assert s.data.dtype == np.float32
    assert s.labels.dtype == np.int64
    assert s.E_chn.shape == (8, 48)   # 6*n_freqs = 48
    assert s.E_sub.shape == (19,)      # 2*n_freqs + 3 = 19
    assert s.sfreq == 256.0
    assert s.n_epochs == s.data.shape[0]


def test_read_raw_unsupported():
    with pytest.raises(ValueError):
        read_raw("foo.xyz")


# --------------------------------------------------------------------------- #
# 语义测试
# --------------------------------------------------------------------------- #
def test_build_subject_labels_alignment_no_reject():
    """无伪迹剔除时，标签与 event 顺序一一对应（所有 event 均在数据边界内）。"""
    raw = make_raw()
    events = make_events()
    labels = np.arange(len(events))
    s = build_subject(raw, events, labels=labels, reject_threshold=None)

    assert s.labels.shape[0] == s.data.shape[0] == len(events)
    assert np.array_equal(s.labels, labels)


def test_build_subject_labels_alignment_with_reject():
    """伪迹剔除后，被剔除 epoch 的标签被同步剔除，其余保持顺序（D-label-align）。"""
    sfreq = 256.0  # 与目标采样率一致，避免 resample 的 anti-aliasing 扩散瞬态到相邻 epoch
    n_seconds = 20.0
    rng = np.random.default_rng(0)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((8, n_times)) * 10e-6
    info = mne.create_info(STD, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    events = make_events(sfreq, n_seconds)
    labels = np.arange(len(events))
    # 在第 3 个 event（index 2）的锁时点附近注入 1V 伪迹 → 其 epoch 必被剔除
    raw._data[:, events[2, 0] : events[2, 0] + 20] = 1.0

    s = build_subject(raw, events, labels=labels, l_freq=None)  # 默认 reject 150 μV；跳过高通以隔离标签对齐逻辑

    assert 2 not in s.labels
    assert s.labels.shape[0] == s.data.shape[0]
    expected = np.array([i for i in range(len(events)) if i != 2])
    assert np.array_equal(s.labels, expected)


def test_build_subject_no_labels():
    raw = make_raw()
    events = make_events()
    s = build_subject(raw, events, labels=None)
    assert s.labels is None


def test_build_subject_labels_length_mismatch():
    raw = make_raw()
    events = make_events()
    with pytest.raises(ValueError):
        build_subject(raw, events, labels=[0, 1, 2])  # 长度 != events 行数


def test_build_subject_metadata_embedding():
    """元数据：age/sex 正确传递，且 E_sub 末 3 维为性别 one-hot。"""
    raw = make_raw()
    events = make_events()
    s = build_subject(raw, events, age=25.0, sex="F", reject_threshold=None)

    assert s.age == 25.0
    assert s.sex == "F"
    assert np.array_equal(s.E_sub[-3:], [0, 1, 0])  # F → [0,1,0]


def test_build_subject_channel_mask_propagates():
    """缺失通道：channel_mask 与 preprocess 结果一致，缺失通道 E_chn 置 0。"""
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]  # 缺 P3/P4/PO8
    raw = make_raw(ch_names=present)
    events = make_events()
    s = build_subject(raw, events, reject_threshold=None)

    expected_mask = np.array([True, True, False, True, False, True, False, True])
    assert np.array_equal(s.channel_mask, expected_mask)
    # 缺失通道的坐标嵌入为 0
    for idx in (2, 4, 6):
        assert np.all(s.E_chn[idx] == 0)


def test_build_subject_nonstandard_three_channel():
    """review v6 P1：standard 为 3 导子集时，通道身份嵌入长度须与 mask/数据一致。"""
    raw = make_raw(ch_names=["Fz", "Cz", "Pz"])
    events = make_events()
    s = build_subject(
        raw, events, labels=np.arange(len(events)), standard=("Fz", "Cz", "Pz"),
        reject_threshold=None,
    )
    assert s.data.shape[1] == 3
    assert s.channel_mask.shape == (3,)
    assert s.E_chn.shape == (3, 48)


def test_load_dataset(tmp_path):
    """端到端：保存临时 .fif → load_dataset 读回并组装。"""
    raw = make_raw()
    fpath = tmp_path / "test_raw.fif"
    raw.save(fpath, overwrite=True)

    events = make_events()
    records = [{
        "path": str(fpath),
        "events": events,
        "labels": np.arange(len(events)),
        "age": 30.0,
        "sex": "M",
        "subject_id": "s1",
    }]

    subjects = load_dataset(records, preprocess_kwargs={"reject_threshold": None})
    assert len(subjects) == 1
    s = subjects[0]
    assert s.subject_id == "s1"
    assert s.age == 30.0
    assert s.sex == "M"
    assert s.data.shape[1] == 8
    assert s.labels.shape[0] == s.data.shape[0]


def test_read_raw_fif(tmp_path):
    """read_raw 读 .fif 返回 Raw，且通道/采样率正确。"""
    raw = make_raw()
    fpath = tmp_path / "r_raw.fif"
    raw.save(fpath, overwrite=True)

    loaded = read_raw(fpath)
    assert loaded.info["sfreq"] == 512.0
    assert loaded.ch_names == STD