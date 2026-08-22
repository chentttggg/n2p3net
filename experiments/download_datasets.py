"""下载 MOABB P300 数据集（掉线重试 + 多线程并行下载）。

用法（在项目根目录，用项目 .venv）：
    # 先下最小最快的 BNCI2014_008（8 被试，电极完全匹配我们的 8 导）验证流程
    .venv/Scripts/python.exe experiments/download_datasets.py --dataset bnci008

    # 下 ERP CORE P3（40 成人，先下 5 个被试测试，再不带 --subjects 下全部）
    .venv/Scripts/python.exe experiments/download_datasets.py --dataset erpcore --subjects 5

    # 下 Brain Invaders（64 被试干电极），8 线程、每文件 5 次尝试
    .venv/Scripts/python.exe experiments/download_datasets.py --dataset bi2014a --subjects 10 --max-workers 8 --retries 5

关键决策（D-download / D-retry / D-parallel）：
- **掉线重试（非断点续传）**：每个文件的下载失败后按指数退避重试 `retries` 次；重试时用
  `force_update=True` 删除上次中断残留的半截文件后整文件重下（pooch 直接写目标文件、中断会留
  半截，故必须删掉重下，否则 MOABB 会把半截文件误判为「已下载」而跳过）。
- **多线程并行**：下载是网络 io 密集，用 ThreadPoolExecutor 并行下不同被试的原始文件；处理
  （重采样/滤波/切窗）是 CPU 密集，串行做。下载与处理分离：先并行 data_path 下载，再 get_data 处理。
- 统一 resample=256（对齐采样率）；8 导蒙太奇 channels 选 Fz/Cz/P3/Pz/P4/PO7/PO8/Oz。
- fmin=1/fmax=24 是 MOABB P300 默认带通；tmin=0/tmax=1s 锁时在刺激。与 N2P3-Net 的 0.1Hz 高通、
  -200~800ms 的差异留到接入脚本统一处理。
- BI2014a 的 16 导（干电极）缺 Fz/PO7/PO8，channels 不选（用全量 16 导）。

下载缓存：原始数据存 ~/mne_data/（MOABB 自动，已下载的文件跳过不重复下）；处理后 X/y 存
experiments/cache/。
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from moabb.datasets import BI2014a, BNCI2014_008, ErpCore2021_P3
from moabb.paradigms import P300

# 8 导蒙太奇（BNCI2014_008 / ERP CORE P3 完全包含；BI2014a 仅 5 导重叠，故用全量 16 导）
CH_8 = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

# BI2014a 16 导全量顺序（MOABB METADATA.sensors；channels=None 时 paradigm 保持该顺序）
BI2014A_CH16 = [
    "Fp1", "Fp2", "F5", "AFz", "F6", "T7", "Cz", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]

# name -> (数据集类, paradigm 选通道 or None=全量, 实际落盘通道顺序)
DATASETS = {
    "erpcore": (ErpCore2021_P3, CH_8, CH_8),   # 40 成人、30 导、1024Hz、2 类（视觉 oddball P3）
    "bnci008": (BNCI2014_008, CH_8, CH_8),     # 8 被试(ALS)、8 导(恰好=我们的蒙太奇)、256Hz
    "bi2014a": (BI2014a, None, BI2014A_CH16),  # 64 被试、16 干电极、512Hz、2 类（缺 Fz/PO7/PO8）
}


def _default_workers() -> int:
    """默认并行线程数 = min(os.cpu_count() - 1, 16)，至少 1（D-workers）。

    下载是网络 io 密集（线程主要在等网络），理论上可超核心数；但过高的并发会：①触发源站
    （Zenodo/OSF/TU Graz）对单 IP 的限流甚至封禁；②分薄本机带宽，总吞吐不线性增长。
    故默认取「核心数−1」并封顶 16（单 IP 下载的经验甜点区）。要更激进可用 --max-workers 覆盖。
    cpu_count 返回 None（极少数环境）时回退 4。
    """
    n = os.cpu_count() or 4
    return max(1, min(n - 1, 16))


def _init_mne_data_root() -> None:
    """预热 MNE 数据根路径，避免多线程并发写 MNE 配置（幂等，减少 race）。"""
    try:
        from mne import set_config

        set_config("MNE_DATA", str(Path.home() / "mne_data"))
    except Exception:
        pass


def _download_one(ds, subject: int, retries: int, backoff: float) -> tuple[int, bool, str | None]:
    """下载单个被试，带掉线重试（指数退避）。返回 (subject, ok, error)。"""
    last_err = None
    for attempt in range(retries):
        try:
            # attempt>0 时 force_update=True：删除上次中断残留的半截文件后整文件重下
            ds.data_path(subject=subject, force_update=(attempt > 0))
            return subject, True, None
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"  [retry] subject {subject} 第 {attempt + 1} 次失败：{e}；{wait:.0f}s 后重试")
                time.sleep(wait)
    return subject, False, str(last_err)


def _parallel_download(ds, subjects: list[int], max_workers: int, retries: int, backoff: float) -> None:
    """多线程并行下载原始数据（io 密集），全部成功后返回，否则抛错。"""
    print(f"并行下载 {len(subjects)} 名被试（{max_workers} 线程，每文件最多 {retries} 次尝试）...")
    results: list[tuple[int, bool, str | None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, ds, s, retries, backoff): s for s in subjects}
        for fut in as_completed(futures):
            results.append(fut.result())

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"下载完成：成功 {n_ok} / 失败 {n_fail}")
    for subject, ok, err in results:
        if not ok:
            print(f"  [fail] subject {subject}: {err}")
    if n_fail:
        raise RuntimeError(
            f"{n_fail} 名被试下载失败。已下载的会缓存，修正网络后重跑本命令即可（自动跳过已完成的）。"
        )


def download(
    name: str,
    subjects: list[int] | None,
    out_dir: str = "experiments/cache",
    max_workers: int | None = None,
    retries: int = 3,
    backoff: float = 2.0,
):
    """并行下载（带重试）+ 串行处理 + 保存。返回 (X, y, metadata)。"""
    cls, channels, saved_channels = DATASETS[name]
    ds = cls()
    subjs = subjects if subjects is not None else ds.subject_list
    if max_workers is None:
        max_workers = _default_workers()

    _init_mne_data_root()

    # 阶段 1：多线程并行下载（带掉线重试）
    _parallel_download(ds, subjs, max_workers, retries, backoff)

    # 阶段 2：串行处理（读缓存 + 重采样/滤波/选通道/切窗）
    print(f"[{name}] 处理中（重采样 256Hz + 选通道 + 1–24Hz 带通）...")
    paradigm = P300(resample=256, channels=channels)
    X, y, metadata = paradigm.get_data(dataset=ds, subjects=subjs)

    print(f"[{name}] 完成：X={X.shape}  dtype={X.dtype}")
    print(f"[{name}]        y={y.shape}  类别={np.unique(y)}")
    print(f"[{name}]        metadata={metadata.shape}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{name}.npz",
        X=X,
        y=y,
        channel_names=np.asarray(saved_channels, dtype=object),
    )
    metadata.to_csv(out / f"{name}_metadata.csv", index=False)
    print(f"[{name}] 已保存 X/y/channel_names → {out / f'{name}.npz'}")
    return X, y, metadata


def main():
    ap = argparse.ArgumentParser(description="下载 MOABB P300 数据集（掉线重试 + 多线程）")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS), help="数据集名")
    ap.add_argument("--subjects", type=int, default=None, help="下载前 N 名被试（不填=全部）")
    ap.add_argument("--out-dir", default="experiments/cache", help="处理后数据保存目录")
    ap.add_argument("--max-workers", type=int, default=_default_workers(), help="并行下载线程数（默认=机器核心数-1，封顶 16）")
    ap.add_argument("--retries", type=int, default=3, help="每个文件最大下载尝试次数")
    ap.add_argument("--backoff", type=float, default=2.0, help="重试退避基数（秒，指数增长）")
    args = ap.parse_args()

    subjs = list(range(1, args.subjects + 1)) if args.subjects is not None else None
    download(args.dataset, subjs, args.out_dir, args.max_workers, args.retries, args.backoff)


if __name__ == "__main__":
    main()
