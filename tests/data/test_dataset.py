"""data/dataset.py 的两级测试（冒烟 + 语义）。

语义重点：标签对齐、元数据嵌入、文件格式分派。
"""

import numpy as np
import pytest

from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.dataset import (
    EEGRecord,
    SubjectData,
    build_subject,
    load_dataset,
    read_event_table,
    read_raw,
)
from data.epochs import preprocessing_spec_from_contract
from tests.support.signal import make_events, make_raw

STD = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]


def test_event_archive_rejects_fractional_labels_and_events(tmp_path) -> None:
    path = tmp_path / "events.npz"
    np.savez(path, events=np.array([[10, 0, 1]], dtype=np.int64), labels=np.array([0.9]))
    with pytest.raises(ValueError, match="labels must have an integer dtype"):
        read_event_table(path, sfreq=100.0)

    np.savez(path, events=np.array([[10.5, 0.0, 1.0]]), labels=np.array([1]))
    with pytest.raises(ValueError, match="Event array.*integer dtype"):
        read_event_table(path, sfreq=100.0)


@pytest.mark.parametrize("channels,requested_channels", [(STD, STD), (["Fz", "Cz", "Pz"], None)])
def test_build_subject_preserves_shape_labels_metadata_and_times(channels, requested_channels):
    raw = make_raw(ch_names=channels)
    events = make_events()
    labels = np.arange(len(events))
    subject = build_subject(raw, events, labels=labels, channels=requested_channels, age=25.0, sex="F")

    assert isinstance(subject, SubjectData)
    assert subject.data.shape == (len(events), len(channels), DEFAULT_P300_DATA_CONTRACT.n_times)
    assert subject.data.dtype == np.float32
    assert subject.labels.dtype == np.int64
    assert subject.channel_names == tuple(name.upper() for name in channels)
    assert subject.channel_mask.shape == (len(channels),)
    assert subject.channel_mask.all()
    assert subject.E_chn.shape == (len(channels), 48)
    assert subject.E_sub.shape == (19,)
    assert subject.sfreq == 128.0
    assert subject.n_epochs == len(events)
    assert subject.age == 25.0
    assert subject.sex == "F"
    np.testing.assert_array_equal(subject.E_sub[-3:], [0, 1, 0])
    np.testing.assert_array_equal(subject.labels, labels)
    np.testing.assert_allclose(
        subject.metadata["acquisition_time_s"], events[:, 0] / raw.info["sfreq"], atol=1e-7
    )


def test_subject_contract_execution_preserves_numeric_output_and_records_actual_support():
    raw = make_raw(ch_names=STD)
    events = make_events()
    labels = np.arange(len(events))
    spec = preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT)
    historical = build_subject(raw, events, labels, channels=STD)
    executed = build_subject(raw, events, labels, preprocessing=spec, channels=STD)
    np.testing.assert_array_equal(executed.data, historical.data)
    np.testing.assert_array_equal(executed.labels, historical.labels)
    np.testing.assert_array_equal(executed.channel_positions_m, historical.channel_positions_m)
    np.testing.assert_array_equal(executed.event_timeline.event_ids, historical.event_timeline.event_ids)
    np.testing.assert_array_equal(executed.event_timeline.evidence_indices, historical.event_timeline.evidence_indices)
    assert "preprocessing_execution" not in historical.metadata
    receipt = executed.metadata["preprocessing_execution"]
    assert receipt["declaration"] == spec.record()
    assert receipt["output"]["shape"] == list(executed.data.shape)
    assert receipt["timing_source"] == executed.event_timeline.timing_source
    assert np.all(executed.event_timeline.evidence_available_times_s == raw.n_times / raw.info["sfreq"])
    with pytest.raises(ValueError, match="override"):
        build_subject(raw, events, labels, preprocessing=spec, channels=STD, n_times=100)


def test_read_raw_missing():
    with pytest.raises(FileNotFoundError):
        read_raw("foo.xyz")


def test_build_subject_labels_alignment_preserves_finite_artifact_rows():
    """Ingress preserves labels; fold-local QC later owns artifact handling."""
    raw = make_raw(sfreq=256.0, ch_names=STD)
    events = make_events(sfreq=256.0)
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
    raw = make_raw(ch_names=STD)
    events = make_events()
    s = build_subject(raw, events, labels=None)
    assert s.labels is None


def test_build_subject_labels_length_mismatch():
    raw = make_raw(ch_names=STD)
    events = make_events()
    with pytest.raises(ValueError):
        build_subject(raw, events, labels=[0, 1, 2])  # 长度 != events 行数


def test_build_subject_rejects_incomplete_explicit_layout():
    """显式布局有缺失时必须失败，不能静默补零。"""
    present = ["Fz", "Cz", "Pz", "PO7", "Oz"]  # 缺 P3/P4/PO8
    raw = make_raw(ch_names=present)
    events = make_events()
    with pytest.raises(ValueError, match="does not pad or substitute"):
        build_subject(raw, events, reject_threshold=None, channels=STD)


def test_load_dataset(tmp_path):
    """端到端：保存临时 .fif → load_dataset 读回并组装。"""
    raw = make_raw(ch_names=STD)
    fpath = tmp_path / "test_raw.fif"
    raw.save(fpath, overwrite=True)
    loaded = read_raw(fpath)
    assert loaded.info["sfreq"] == 512.0
    assert loaded.ch_names == STD

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
    strict = load_dataset(records, preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT))
    np.testing.assert_array_equal(strict[0].data, s.data)
    assert strict[0].metadata["preprocessing_execution"]["output"]["shape"] == list(s.data.shape)
