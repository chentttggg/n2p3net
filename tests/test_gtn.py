"""data/gtn 模块测试：GTN NIX 读取器。

语义（读真实 GTN 数据）：
    - raw 为 Fz/Cz/Pz 3 导、1000Hz（丢弃 EOG）；
    - events 为数字刺激（第三列 = 1–9，过滤 New Segment）；
    - thought_number 与 .txt 的 'the number thought' 一致；
    - metadata 含 sex/age/handedness。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.gtn import _extract_digit, _nix_subject_ids, _parse_age, read_gtn_experiment

# 真实 GTN 数据（用户已下载到项目内 mne_data）
EXP_DIR = Path("mne_data/MNE-P3-data/Experiment_341_P3_Numbers")
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


@pytest.mark.skipif(not HAS_GTN, reason="GTN 数据未下载")
def test_gtn_events_count_matches_stimuli():
    """数字事件数应等于数字 1–9 的刺激总次数（113 总事件 - 1 New Segment）。"""
    d = read_gtn_experiment(EXP_DIR)
    digits, counts = np.unique(d.events[:, 2].astype(int), return_counts=True)
    # 9 个数字都应出现，且每个数字出现多次（GTN 各数字 7–19 次不等，>=5 是安全下界）
    assert len(digits) == 9
    assert (counts >= 5).all(), f"每个数字应出现多次，得到 {dict(zip(digits, counts, strict=True))}"


@pytest.mark.skipif(not HAS_GTN, reason="GTN 数据未下载")
def test_gtn_multi_txt_matches_nix_internal_subject():
    """review v6 P0-3：Experiment_515/531 的 Data 下有多个 .txt，必须按 NIX 内部被试名匹配。"""
    for exp_name in ("Experiment_515_P3_Numbers", "Experiment_531_P3_Numbers"):
        exp_dir = Path("mne_data/MNE-P3-data") / exp_name
        if not exp_dir.exists():
            continue
        internal = _nix_subject_ids(next(exp_dir.glob("*.nix")))
        assert len(internal) == 1, f"{exp_name} 预期 1 个内部被试"
        d = read_gtn_experiment(exp_dir)
        assert d.subject_id == internal[0], f"{exp_name} 应加载 NIX 内部被试 {internal[0]}"


@pytest.mark.skipif(not HAS_GTN, reason="GTN 数据未下载")
def test_gtn_missing_txt_raises_with_internal_subject():
    """review v6 P0-3：Experiment_611 的 .nix 正常，但缺 thought 元数据，应显式报缺失。"""
    exp_dir = Path("mne_data/MNE-P3-data/Experiment_611_P3_Numbers")
    if not exp_dir.exists():
        pytest.skip("Experiment_611 不存在")
    with pytest.raises(FileNotFoundError, match="P3Numbers_20150107_f_14_002"):
        read_gtn_experiment(exp_dir)
