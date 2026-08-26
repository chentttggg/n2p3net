"""GTN（Guess the Number）NIX 数据读取器。

职责（Phase 1 数据接入）：
    把 GTN 猜数字数据集的单个被试从 .nix（HDF5/NIX 格式）读成 MNE Raw + 刺激事件 + 元数据。
    这是唯一「9 选 1 猜数字」的公开数据（Vařeka 77.2% 锚点来源），心选数字是决策层的 ground truth。

明确「不做」：
    - 不做通道映射到 8 导蒙太奇（GTN 只有 Fz/Cz/Pz 3 导，映射/缺失填充是 data/dataset 层职责，
      本模块只读原始 3 导）。
    - 不做 epoch 切分/滤波/重采样（那走 data/preprocess.py）。

三思决策记录（供后续会话追溯）：
    D-gtn-nix       GTN 在 EEGBase 上是 NIX 格式（HDF5 文件头 \\x89HDF），**非 BrainVision**。用 h5py
                    直接读（h5py 是 MNE 依赖已装），无需 nixio。NIX 结构：EEG 在
                    data/EEG Data/data_arrays/P3Numbers_XXX/data（(4, N) float64），刺激事件在
                    data/EEG Data/data_arrays/STIMULUS_XXX_positions/data（(n_events, 2)）。
    D-gtn-channels  NIX 的 4 通道 = Fz/Cz/Pz/EOG（dimensions/1/labels）。只取 Fz/Cz/Pz（3 导 EEG），
                    丢弃 EOG；按通道名过滤而非硬编码前 3 导（健壮）。
    D-gtn-events    事件 label 形如 'Stimulus/S  3'（数字前有空格）或 'New Segment/'（记录起点，非刺激）。
                    用正则提取数字，过滤 'New Segment/'；position[:,1] 是时间（秒），×sfreq → 采样点。
    D-gtn-thought   心选数字（9 选 1 ground truth）从配套 .txt 的 'the number thought' 字段提取；
                    .txt 在同 experiment 目录的 Data/ 子目录下。
    D-gtn-sfreq     采样率从 .nix metadata 读（Amplifier/properties/SampleRate=1000Hz），不硬编码。

契约（输入 → 输出）：
    read_gtn_experiment(exp_dir) → GTNData{raw(Fz/Cz/Pz 3 导), events(n,3)[sample,0,digit],
        thought_number, metadata, subject_id}。

依赖的决策：roadmap Phase 1、data/preprocess.py（下游消费 Raw+events）。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import mne
import numpy as np

# GTN 的 3 导 EEG 通道（丢弃 EOG）
_EEG_CHANNELS = ("Fz", "Cz", "Pz")


@dataclass
class GTNData:
    """GTN 单个被试的数据。"""

    raw: mne.io.Raw
    events: np.ndarray  # (n_events, 3) MNE 格式 [sample, 0, digit]
    thought_number: int
    metadata: dict
    subject_id: str


def _parse_txt(txt_path: Path) -> dict:
    """解析 GTN 元数据 .txt → dict（key: value，按冒号切分）。"""
    md: dict = {}
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            md[k.strip()] = v.strip()
    return md


def _parse_age(age_str: str) -> int | None:
    """'10 years' → 10；无法解析返回 None。"""
    m = re.search(r"(\d+)", age_str)
    return int(m.group(1)) if m else None


def _extract_digit(label: str) -> int | None:
    """'Stimulus/S  3' → 3；'New Segment/' 等非刺激 → None。

    实测发现（2026-08-21）：完整 labels 中还混有 'Stimulus/S 13'（×1）与 'Stimulus/S 15'（×12）
    两个非数字控制码（Presentation 的 port_code，非数字刺激）。旧正则 ``(\\d)`` 只捕获单个数字，
    会把二者误判为数字 1，污染 target/non-target 标签。故改用 ``(\\d+)`` 并显式过滤到 1–9。
    """
    m = re.search(r"Stimulus/S\s+(\d+)", label)
    if m is None:
        return None
    d = int(m.group(1))
    return d if 1 <= d <= 9 else None


def _to_str(x) -> str:
    """h5py 的 bytes/object 元素 → 干净字符串（b'Fz' → 'Fz'）。"""
    return x.decode() if isinstance(x, bytes) else str(x)


def read_gtn(nix_path: str | Path, txt_path: str | Path) -> GTNData:
    """读一个 GTN 被试的 .nix + .txt → GTNData。"""
    nix_path = Path(nix_path)
    txt_path = Path(txt_path)
    # 关键：.nix 文件名是 'Experiment_XXX_P3_Numbers'，但内部 data_array 名 = 被试名
    # 'P3Numbers_YYYYMMDD_f_AGE_XXX'（与 .txt 文件名一致）。故从 .txt 文件名提取 subject_id。
    subject_id = txt_path.stem  # 如 'P3Numbers_20150618_f_10_001'

    with h5py.File(nix_path, "r") as f:
        arr = "data/EEG Data/data_arrays"
        # 该 .nix 里的 data_array 名 = 被试名（P3Numbers_...）
        eeg_base = f"{arr}/{subject_id}"
        stim_base = f"{arr}/STIMULUS_{subject_id}"

        # 采样率（D-gtn-sfreq）
        sfreq = float(
            f[
                f"{eeg_base}/metadata/sections/HardwareSettings/sections/Amplifier/properties/SampleRate"
            ][0]
        )

        # 通道名（D-gtn-channels）：按名过滤出 Fz/Cz/Pz，丢弃 EOG
        all_ch = [_to_str(c) for c in f[f"{eeg_base}/dimensions/1/labels"][:]]
        data = f[f"{eeg_base}/data"][:]  # (4, N) float64
        picks = [i for i, c in enumerate(all_ch) if c in _EEG_CHANNELS]
        picked_ch = [all_ch[i] for i in picks]
        eeg = data[picks, :]  # (3, N)

        # 刺激事件（D-gtn-events）
        stim_pos = f[f"{stim_base}_positions/data"][:]  # (n_events, 2)
        stim_labels = [
            _to_str(label) for label in f[f"{stim_base}_positions/dimensions/1/labels"][:]
        ]
        times_sec = stim_pos[:, 1]  # 第二列是时间（秒）
        samples = np.rint(times_sec * sfreq).astype(np.int64)
        digits = [_extract_digit(label) for label in stim_labels]
        mask = [d is not None for d in digits]
        events = np.column_stack(
            [
                samples[mask],
                np.zeros(sum(mask), dtype=np.int64),
                np.array([d for d in digits if d is not None]),
            ]
        ).astype(np.int64)

        # 记录日期时间
        try:
            rec_date = _to_str(f[f"{eeg_base}/metadata/sections/Recording/properties/StartDate"][0])
            rec_time = _to_str(f[f"{eeg_base}/metadata/sections/Recording/properties/StartTime"][0])
        except Exception:  # noqa: BLE001
            rec_date, rec_time = "", ""

    # 构造 MNE Raw（Fz/Cz/Pz 3 导）
    info = mne.create_info(picked_ch, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose=False)

    # 元数据（.txt + .nix）
    md = _parse_txt(txt_path)
    thought_number = int(md["the number thought"])
    metadata = {
        "sex": md.get("sex", ""),
        "age": _parse_age(md.get("age", "")),
        "handedness": md.get("handedness", ""),
        "thought_number": thought_number,
        "record_date": rec_date,
        "record_time": rec_time,
        "n_channels": len(picked_ch),
        "sfreq": sfreq,
    }

    return GTNData(
        raw=raw,
        events=events,
        thought_number=thought_number,
        metadata=metadata,
        subject_id=subject_id,
    )


def _nix_subject_ids(nix_path: Path) -> list[str]:
    """读 NIX 内部实际记录的被试 data_array 名（如 P3Numbers_YYYYMMDD_f_AGE_XXX）。

    目录下的 .txt 可能有多份（实测 Experiment_515/531 各有两个），必须以 NIX 内部
    被试名为准做精确匹配，不能盲取第一个 .txt（review v6 P0-3）。
    """
    with h5py.File(nix_path, "r") as f:
        arr = "data/EEG Data/data_arrays"
        return [k for k in f[arr].keys() if k.startswith("P3Numbers")]


def read_gtn_experiment(exp_dir: str | Path) -> GTNData:
    """读一个 GTN experiment 目录（含根下 .nix + Data/*.txt）。

    experiment 目录结构（EEGBase 下载后）：
        Experiment_XXX_P3_Numbers/
        ├── Experiment_XXX_P3_Numbers.nix
        ├── Data/
        │   └── P3Numbers_YYYYMMDD_f_AGE_XXX.txt
        └── Scenario/numbers.zip

    匹配规则（review v6 P0-3）：以 NIX 内部 data_array 名（= 被试名）为唯一事实来源，
    在 Data/*.txt 中按 stem 精确匹配；多余的 .txt 忽略。若无匹配 txt（如 Experiment_611），
    报元数据缺失错误，由上层登记剔除。
    """
    exp_dir = Path(exp_dir)
    nix_files = list(exp_dir.glob("*.nix"))
    if not nix_files:
        raise FileNotFoundError(f"{exp_dir} 下未找到 .nix 文件。")
    if len(nix_files) > 1:
        warnings.warn(
            f"{exp_dir} 下有多个 .nix 文件，仅使用第一个 {nix_files[0].name}。",
            stacklevel=2,
        )
    nix_path = nix_files[0]

    internal_ids = _nix_subject_ids(nix_path)
    if not internal_ids:
        raise FileNotFoundError(
            f"{nix_path} 内未找到 P3Numbers data_array（不是预期 GTN NIX 结构）。"
        )

    txt_files = list((exp_dir / "Data").glob("*.txt"))
    txt_by_stem = {p.stem: p for p in txt_files}
    orphan_stems = [stem for stem in txt_by_stem if stem not in internal_ids]
    if orphan_stems:
        warnings.warn(
            f"{exp_dir / 'Data'} 存在未匹配 NIX 内部被试的 .txt（孤儿文件）："
            f"{sorted(orphan_stems)}；将按 NIX 内部 data_array 名精确匹配（review v6 P0-3）。",
            stacklevel=2,
        )
    for subject_id in internal_ids:
        if subject_id in txt_by_stem:
            return read_gtn(nix_path, txt_by_stem[subject_id])

    raise FileNotFoundError(
        f"{exp_dir / 'Data'} 下没有与 NIX 内部被试 {internal_ids} 匹配的 .txt 元数据文件"
        f"（the number thought 缺失，无法用于 9 选 1 评估）。现有 txt："
        f"{[p.name for p in txt_files] or '无'}。"
    )
