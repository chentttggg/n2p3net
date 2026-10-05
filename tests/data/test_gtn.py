"""GTN 原始读取、cohort 预处理和完整九选事件账本。

语义（读真实 GTN 数据）：
    - raw 为 Fz/Cz/Pz 3 导、1000Hz（丢弃 EOG）；
    - events 为数字刺激（第三列 = 1–9，过滤 New Segment）；
    - thought_number 与 .txt 的 'the number thought' 一致；
    - metadata 含 sex/age/handedness。
"""

from __future__ import annotations

import numpy as np
import pytest

from data.gtn import GTNData, _extract_digit, _nix_subject_ids, _parse_age, read_gtn_experiment
from data.gtn_dataset import build_gtn_epoch_dataset, gtn_preprocessing_for_cohort
from experiments.prepare_gtn_dataset import _derive_n_times
from tests import PROJECT_ROOT
from tests.support.signal import make_raw

# 真实 GTN 数据（用户已下载到项目内 mne_data）
GTN_ROOT = PROJECT_ROOT / "mne_data/MNE-P3-data"
EXP_DIR = GTN_ROOT / "Experiment_341_P3_Numbers"
HAS_GTN = (EXP_DIR / "Experiment_341_P3_Numbers.nix").exists()


# ---------------- 单元测试（不依赖数据） ----------------


def test_extract_digit():
    assert _extract_digit("Stimulus/S  3") == 3
    assert _extract_digit("Stimulus/S  9") == 9
    assert _extract_digit("New Segment/") is None
    # 控制码过滤（2026-08-21 实测发现 13/15 非数字刺激，不得误判为数字）
    assert _extract_digit("Stimulus/S 13") is None
    assert _extract_digit("Stimulus/S 15") is None


def test_parse_age():
    assert _parse_age("10 years") == 10
    assert _parse_age("unknown") is None


def test_gtn_n_times_uses_the_shared_exclusive_right_edge_rule():
    assert _derive_n_times(256.0, -200.0, 800.0) == 256
    assert _derive_n_times(128.0, 0.0, 1000.0) == 128


@pytest.mark.parametrize("cohort,low_cutoff,causal", [
    ("offline", .1, False), ("causal", .1, True),
    ("gtn_paper_offline", .5, False), ("gtn_paper", .5, True),
])
def test_gtn_builder_executes_cohort_and_preserves_complete_digit_ledger(tmp_path, monkeypatch, cohort, low_cutoff, causal):
    experiment = tmp_path / "Experiment_001_P3_Numbers"
    experiment.mkdir()
    digits = np.tile(np.arange(1, 10), 2)
    raw = make_raw(sfreq=256, n_seconds=24, ch_names=("Fz", "Cz", "Pz"))
    events = np.column_stack(((np.arange(len(digits)) + 2) * 256, np.zeros(len(digits), int), digits))
    gtn = GTNData(raw=raw, events=events, thought_number=4, subject_id="participant-1",
                  metadata={"sfreq": 256, "source_reference": "nose"})
    monkeypatch.setattr("data.gtn_dataset.read_gtn_experiment", lambda _path: gtn)
    spec = gtn_preprocessing_for_cohort(cohort)
    dataset = build_gtn_epoch_dataset(tmp_path, preprocessing=spec)
    assert spec.l_freq == low_cutoff
    assert spec.filter_phase == ("forward" if causal else "zero")
    assert (spec.sfreq, spec.h_freq, spec.tmin_ms, spec.tmax_ms, spec.baseline_mode) == (
        128., 30., -200., 1200., "mean_only")
    assert (spec.filter_method, spec.filter_order, spec.notch_freqs_hz, spec.zero_phase_edge_policy) == (
        "iir", 4, (), "legacy")
    assert dataset.X.shape == (18, 3, 179)
    bundle = dataset.provenance["preprocessing_execution"]
    assert bundle["unverified_inputs"] == 0 and len(bundle["recordings"]) == 1
    receipt = bundle["recordings"][0]
    assert receipt["declaration"] == spec.record()
    assert receipt["output"]["shape"] == list(dataset.X.shape)
    timeline = dataset.event_timeline
    assert timeline.online_causal is causal
    assert timeline.supports_full_candidate_chain
    np.testing.assert_array_equal(timeline.onset_samples, events[:, 0])
    np.testing.assert_array_equal(timeline.repetition_indices, np.repeat([0, 1], 9))
    np.testing.assert_array_equal(dataset.y, digits == 4)
    assert set(dataset.metadata["candidate_id"]) == {str(value) for value in range(1, 10)}


# ---------------- 集成测试（读真实数据） ----------------


@pytest.mark.skipif(not HAS_GTN, reason="GTN 数据未下载")
def test_read_gtn_experiment():
    d = read_gtn_experiment(EXP_DIR)

    # raw：Fz/Cz/Pz 3 导、1000Hz
    assert d.raw.ch_names == ["Fz", "Cz", "Pz"], d.raw.ch_names
    assert d.raw.info["sfreq"] == 1000.0
    assert d.raw.get_data().shape[0] == 3

    # events：数字刺激（第三列 1–9），无 New Segment
    assert d.events.ndim == 2 and d.events.shape[1] == 3
    assert d.events.shape[0] > 0
    assert set(d.events[:, 2].astype(int)).issubset(set(range(1, 10)))

    # thought_number 与 .txt 一致（Experiment_341 的心选数字 = 1）
    assert d.thought_number == 1
    assert d.metadata["thought_number"] == 1

    # metadata
    assert d.metadata["sex"] == "female"
    assert d.metadata["age"] == 10
    assert d.metadata["handedness"] == "right"

    # 事件时间戳单调递增且在数据时长内
    samples = d.events[:, 0].astype(int)
    assert (np.diff(samples) >= 0).all(), "事件应按时间排序"
    assert samples.max() < d.raw.n_times
    digits, counts = np.unique(d.events[:, 2].astype(int), return_counts=True)
    # 9 个数字都应出现，且每个数字出现多次（GTN 各数字 7–19 次不等，>=5 是安全下界）
    assert len(digits) == 9
    assert (counts >= 5).all(), f"每个数字应出现多次，得到 {dict(zip(digits, counts, strict=True))}"


@pytest.mark.parametrize("experiment", ["Experiment_515_P3_Numbers", "Experiment_531_P3_Numbers"])
def test_gtn_multi_txt_matches_nix_internal_subject(experiment):
    """review v6 P0-3：Experiment_515/531 的 Data 下有多个 .txt，必须按 NIX 内部被试名匹配。"""
    exp_dir = GTN_ROOT / experiment
    if not exp_dir.is_dir():
        pytest.skip(f"{experiment} 数据未下载")
    internal = _nix_subject_ids(next(exp_dir.glob("*.nix")))
    assert len(internal) == 1
    d = read_gtn_experiment(exp_dir)
    assert d.subject_id == internal[0]


def test_gtn_missing_txt_raises_with_internal_subject():
    """review v6 P0-3：Experiment_611 的 .nix 正常，但缺 thought 元数据，应显式报缺失。"""
    exp_dir = GTN_ROOT / "Experiment_611_P3_Numbers"
    if not exp_dir.exists():
        pytest.skip("Experiment_611 不存在")
    with pytest.raises(FileNotFoundError, match="P3Numbers_20150107_f_14_002"):
        read_gtn_experiment(exp_dir)
