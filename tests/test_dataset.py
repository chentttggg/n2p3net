"""data/dataset.py 的两级测试（冒烟 + 语义）。

语义重点：标签对齐、元数据嵌入、文件格式分派。
"""

import mne
import numpy as np
import pytest

from data.dataset import (
    EEGRecord,
    SubjectData,
    build_subject,
    load_dataset,
    read_event_table,
    read_raw,
)

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
    return np.column_stack(
        [samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)]
    )


def test_event_archive_rejects_fractional_labels_and_events(tmp_path) -> None:
    path = tmp_path / "events.npz"
    np.savez(path, events=np.array([[10, 0, 1]], dtype=np.int64), labels=np.array([0.9]))
    with pytest.raises(ValueError, match="labels must have an integer dtype"):
        read_event_table(path, sfreq=100.0)

    np.savez(path, events=np.array([[10.5, 0.0, 1.0]]), labels=np.array([1]))
    with pytest.raises(ValueError, match="Event array.*integer dtype"):
        read_event_table(path, sfreq=100.0)


# --------------------------------------------------------------------------- #
# 冒烟测试
# --------------------------------------------------------------------------- #
def test_build_subject_shape():
    raw = make_raw()
    events = make_events()
    s = build_subject(raw, events, labels=np.arange(len(events)))

    assert isinstance(s, SubjectData)
    assert s.data.shape[1] == 8
    assert s.data.shape[2] == 350
    assert s.data.dtype == np.float32
    assert s.labels.dtype == np.int64
    assert s.E_chn.shape == (8, 48)  # 6*n_freqs = 48
    assert s.E_sub.shape == (19,)  # 2*n_freqs + 3 = 19
    assert s.sfreq == 250.0
    assert s.n_epochs == s.data.shape[0]


def test_read_raw_missing():
    with pytest.raises(FileNotFoundError):
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


def test_build_subject_labels_alignment_preserves_finite_artifact_rows():
    """Ingress preserves labels; fold-local QC later owns artifact handling."""
    sfreq = 256.0  # 与目标采样率一致，避免 resample 的 anti-aliasing 扩散瞬态到相邻 epoch
    n_seconds = 20.0
    rng = np.random.default_rng(0)
    n_times = int(n_seconds * sfreq)
    data = rng.standard_normal((8, n_times)) * 10e-6
    info = mne.create_info(STD, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    events = make_events(sfreq, n_seconds)
    labels = np.arange(len(events))
    # Inject a finite transient. Ingress must preserve its row and label.
    raw._data[:, events[2, 0] : events[2, 0] + 20] = 1.0

    s = build_subject(
        raw,
        events,
        labels=labels,
        l_freq=None,
        channels=STD,
    )

    assert s.labels.shape[0] == s.data.shape[0]
    assert np.array_equal(s.labels, labels)


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


def test_build_subject_rejects_incomplete_explicit_layout():
    """显式布局有缺失时必须失败，不能静默补零。"""
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]  # 缺 P3/P4/PO8
    raw = make_raw(ch_names=present)
    events = make_events()
    with pytest.raises(ValueError, match="does not pad or substitute"):
        build_subject(raw, events, reject_threshold=None, channels=STD)


def test_build_subject_uses_native_layout_without_padding():
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]
    raw = make_raw(ch_names=present)
    events = make_events()
    subject = build_subject(raw, events, reject_threshold=None, channels=None)
    assert subject.channel_names == tuple(name.upper() for name in present)
    assert subject.data.shape[1] == len(present)
    assert subject.channel_mask.all()


def test_build_subject_nonstandard_three_channel():
    """review v6 P1：standard 为 3 导子集时，通道身份嵌入长度须与 mask/数据一致。"""
    raw = make_raw(ch_names=["Fz", "Cz", "Pz"])
    events = make_events()
    s = build_subject(
        raw,
        events,
        labels=np.arange(len(events)),
        channels=("Fz", "Cz", "Pz"),
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
    records = [
        EEGRecord(
            path=fpath,
            events=events,
            labels=np.arange(len(events)),
            age=30.0,
            sex="M",
            subject_id="s1",
        )
    ]

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
