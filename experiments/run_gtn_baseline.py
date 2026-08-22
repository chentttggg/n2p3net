"""GTN 猜数字基线实验入口（Phase 1）。

职责（roadmap Phase 1 + constitution P8）：
    把 GTN（唯一「9 选 1 猜数字」公开数据）读入 → 统一预处理 → LOSO 跨受试评估 → 9 选 1 命中率，
    供 SWLDA / xDAWN+RG / EEGNet / EEG-Inception / EEG Conformer / 免费地板 在完全同一协议下公平对比。

用法（项目根目录，用 .venv）：
    .venv/Scripts/python.exe experiments/run_gtn_baseline.py --model eegnet --subjects 20 --epochs 20
    .venv/Scripts/python.exe experiments/run_gtn_baseline.py --model swlda --subjects 20
    .venv/Scripts/python.exe experiments/run_gtn_baseline.py --model all --subjects 20

三思决策记录（2026-08-21，基于对 GTN 真实数据的实测核查，可能推翻 review v4 部分结论）：
    D-gtn-3ch       GTN 只有 Fz/Cz/Pz 3 导。deep 基线用 n_chans=3（真实通道数），**不零填充到 8 导**。
                    理由：GTN 本质是 3 导数据（非「8 导缺 5 导」），零填充 5 个全 0 通道是噪音浪费、
                    无谓增加 braindecode 空间卷积的参数量；复现 Vařeka 2016 的 3 导 MLP 77.2% 锚点
                    需要 3 导公平条件。classic/riemann 同理用 3 导子集（preprocess standard=3 导）。
    D-gtn-n-epochs  实测 247 名被试可读（Experiment_611 是缺 .txt thought 元数据，非 HDF5 损坏）；
                    原始数字事件 mean≈205（范围 58–372），**无一人达 review v4 声称的 500**。
                    默认 ±150μV 伪迹剔除后 3 名被试 0 试次（质量排除），另有 2 对目录共享
                    同一 NIX 内部 subject_id 被显式登记跳过，最终可评估 242 名；
                    每数字平均试次 K≈17（非 50）。77%±3 锚点须按该 K 校准（review v6 实测）。
    D-gtn-base-rate 实测 GTN **没有「0」刺激**（.sce 定义 tx0 但 NIX 未记录任何 port_code=10 事件；
                    数字 1–9 各约 5600 次，另有 13/15 两个控制码 ×13）。故 target 基率 = 1/9，
                    **pos_weight 应保持 ≈8**，review v4 的「1.4 pos_weight 8→9」结论错误，予以推翻。
    D-gtn-ctrl-code NIX 刺激 labels 混有 'Stimulus/S 13'（×1）与 'Stimulus/S 15'（×12）两个非数字
                    控制码，已由 data/gtn._extract_digit 过滤（1–9 之外返回 None），不进入标签。
    D-gtn-scale     LOSO = 留一被试，247 名被试 × 每 fold 重训，CPU 上 deep 模型成本高。脚本默认
                    支持 --subjects 限缩（小规模冒烟）与 --epochs 调低；全量留给 GPU/XPU 或后台。

契约（输入 → 输出）：
    GTN 目录（mne_data/MNE-P3-data/Experiment_*_P3_Numbers）→ 打印各模型 LOSO 命中率 / balanced acc / AUC。

依赖的决策：data/gtn（NIX 读取）、data/preprocess（3 导预处理）、baselines/deep|classic|riemann、
    baselines/evaluate（LOSO + 命中率）、constitution P8/D3。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Optional



# 让 experiments/ 下的脚本能 import src/ 下的包（review v4 2.2 的 PYTHONPATH 约定）
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))




# GTN 3 导（真实通道，非 8 导蒙太奇子集）
GTN_STANDARD = ("Fz", "Cz", "Pz")
GTN_ROOT = Path(__file__).resolve().parent.parent / "mne_data" / "MNE-P3-data"


@contextmanager
def _single_blas_thread():
    """数据加载线程内把 BLAS 限制为单线程，避免 N 线程 × 16 BLAS 线程互相抢占。"""
    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # noqa: BLE001
        yield
        return
    with threadpool_limits(limits=1):
        yield


def _load_one_gtn_subject(args):
    """线程 worker：读取 + 预处理一个 GTN 被试，返回 (exp_name, payload_or_error, ok)。

    payload = (subj_id, data, y, digits, thought_number)（数组均可跨线程共享）。
    """
    exp_dir, sfreq, l_freq, tmin, tmax = args
    exp_name = exp_dir.name
    try:
        from data.gtn import read_gtn_experiment
        from data.preprocess import preprocess

        with _single_blas_thread():
            g = read_gtn_experiment(exp_dir)
            result = preprocess(
                g.raw,
                g.events,
                standard=GTN_STANDARD,
                sfreq=sfreq,
                l_freq=l_freq,
                tmin=tmin,
                tmax=tmax,
            )
        digits = g.events[result.event_indices, 2].astype(int)
        y = (digits == g.thought_number).astype(int)
        return exp_name, (g.subject_id, result.data, y, digits, g.thought_number), True
    except Exception as e:  # noqa: BLE001 —— 单被试损坏不应拖垮全量
        return exp_name, f"{exp_name}: {type(e).__name__}: {e}", False

# deep 基线走 CUDA/XPU，fold 不并行（避免显存争用）；经典基线可用线程池并行 fold
DEEP_MODELS = {"eegnet", "inception", "conformer"}
CLASSIC_MODELS = {"template", "windowlr", "swlda", "xdawn"}


def _configure_threads(threads: int) -> int:
    """在 import numpy/MNE/sklearn 之前配置 BLAS/OpenMP 线程数。

    review v6 性能复核：本项目 NumPy 使用 scipy-openblas（OpenBLAS 0.3.34，
    NO_AFFINITY），不设置时经典基线只用到 1 个线程；设置 16 线程时 xDAWN
    30-fold 从 47.2s 降到 40.9s，向量化 OAS 后进一步降到 ~5.8s。默认取
    min(24, 逻辑核数) 并在 32 线程机器上留出余量；用户可用 --threads 覆盖。
    """
    logical = os.cpu_count() or 1
    n_threads = int(threads) if threads > 0 else min(24, logical)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(n_threads))
    return n_threads


def _gtn_cache_filename(sfreq, l_freq, tmin, tmax, n_subjects) -> str:
    """预处理的张量缓存文件名（参数变化即换 key，避免跨预处理口径串用）。"""
    return (
        f"gtn_3ch_sf{sfreq:.0f}_lf{l_freq:.1f}_"
        f"tm{tmin:.1f}_tx{tmax:.1f}_n{n_subjects}.npz"
    )


def _save_gtn_cache(path: Path, X, y, digits, subject_ids, true_digits, skipped) -> None:
    """把预处理后的 GTN 张量缓存为单个未压缩 .npz（D-preprocess-cache）。

    训练/评估阶段直接读缓存，完全跳过 MNE 读取、重采样、0.1Hz 高通与 epoch 切分。
    subject_ids / true_keys / skipped 是变长字符串，需 allow_pickle=True；该缓存只在
    本机本地生成，不外发、不接收不可信文件。
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    true_keys = np.array(list(true_digits.keys()), dtype=object)
    true_values = np.array(list(true_digits.values()), dtype=np.int64)
    np.savez(
        tmp,
        allow_pickle=True,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        digits=np.asarray(digits, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=object),
        true_keys=true_keys,
        true_values=true_values,
        skipped=np.asarray(skipped, dtype=object),
    )
    tmp.replace(path)


def _load_gtn_cache(path: Path):
    """读张量缓存，返回 (X, y, digits, subject_ids, true_digits, skipped)。"""
    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        X = z["X"]
        y = z["y"]
        digits = z["digits"]
        subject_ids = z["subject_ids"].astype(str)
        true_keys = z["true_keys"].astype(str).tolist()
        true_values = z["true_values"].astype(np.int64).tolist()
        skipped = [str(x) for x in z["skipped"].tolist()]
    true_digits = dict(zip(true_keys, true_values))
    if len(X) != len(y) or len(X) != len(digits) or len(X) != len(subject_ids):
        raise ValueError(f"缓存损坏：X/y/digits/subject_ids 长度不一致：{X.shape}/{len(y)}/{len(digits)}/{len(subject_ids)}")
    return X, y, digits, subject_ids, true_digits, skipped


def save_subject_scores(summary, model_name: str, path: Path) -> Path:
    """把 evaluate() 返回的逐被试 (predicted, true, group) 记录落盘为 JSON。

    供 experiments/run_paired_test.py 做配对置换检验：每个 group（GTN=被试）一个
    hit（0/1），比较两模型时按 group 对齐并取交集，避免任何“分母悄悄变化”。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for predicted, true, group in summary.subject_records:
        records.append(
            {
                "subject": str(group),
                "predicted": None if predicted is None else int(predicted),
                "true": int(true),
                "hit": int(predicted is not None and predicted == true),
            }
        )
    payload = {
        "model": model_name,
        "n_subjects": len(records),
        "hit_rate_mean": float(summary.hit_rate_mean),
        "hit_rate_std": float(summary.hit_rate_std),
        "balanced_acc_mean": float(summary.balanced_acc_mean),
        "auc_mean": None if summary.auc_mean != summary.auc_mean else float(summary.auc_mean),
        "records": records,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path



def load_gtn_subjects(
    max_subjects: int | None = None,
    *,
    sfreq: float = 256.0,
    l_freq: float = 0.1,
    tmin: float = -0.2,
    tmax: float = 0.8,
    load_jobs: int = 0,
    cache_path: Optional[str | Path] = None,
) -> tuple:
    """读 GTN 被试 → 汇总 (X, y, digits, subject_ids, true_digits, skipped)。

    X ∈ R^{N×3×T} float32（3 导，缺失通道无 NaN，因 3 导全部存在）；
    y ∈ {0,1}^N（target = 试次数字 == 心选数字，基率 1/9，D-gtn-base-rate）；
    digits ∈ {1..9}^N；subject_ids 为被试字符串 id；true_digits {subj_id: thought}。
    skipped 为读取失败/元数据缺失/全 epoch 剔除的被试（如 Experiment_611 缺 .txt），
    供调用方感知数据损失与可评估口径。
    """
    # 在 _configure_threads() 之后才 import，确保 BLAS 线程配置先生效
    import numpy as np

    # 张量缓存命中时直接返回，完全跳过 MNE 预处理（D-preprocess-cache）。
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                cached = _load_gtn_cache(cache_path)
                print(f"[data] 命中预处理缓存：{cache_path}（跳过 MNE 预处理）", flush=True)
                return cached
            except Exception as e:  # noqa: BLE001 —— 缓存损坏则回退重新预处理并覆盖
                print(f"[data] 缓存读取失败（{type(e).__name__}: {e}），回退重新预处理", flush=True)

    exps = sorted([d for d in GTN_ROOT.iterdir() if d.is_dir() and d.name.startswith("Experiment")])
    if max_subjects:
        exps = exps[:max_subjects]

    X_list, y_list, d_list, s_list = [], [], [], []
    true_digits: dict = {}
    true_digit_sources: dict[str, str] = {}
    skipped: list[str] = []

    # 线程池并行预处理（D-load-threads）：读取 NIX + MNE 滤波/切窗是 CPU/IO 密集，
    # 不同被试完全独立；实测 20 被试 4 线程 3.2s→1.0s。worker 内 BLAS 限单线程。
    load_jobs = int(load_jobs) if load_jobs and load_jobs > 0 else min(6, os.cpu_count() or 1)
    task_args = [(exp, sfreq, l_freq, tmin, tmax) for exp in exps]
    if load_jobs > 1 and len(exps) > 1:
        with ThreadPoolExecutor(max_workers=load_jobs) as executor:
            load_results = list(executor.map(_load_one_gtn_subject, task_args))
    else:
        load_results = [_load_one_gtn_subject(args) for args in task_args]

    for exp_name, payload, ok in load_results:
        if not ok:
            skipped.append(payload)
            continue
        subj_id, data, y, digits, thought_number = payload

        # 数据质量登记（review v6 P0-3）：伪迹剔除后 0 试次的被试不得静默缩小命中率分母
        if len(y) == 0:
            skipped.append(f"{subj_id} ({exp_name}): 伪迹剔除后 0 试次，质量排除")
            continue

        # 重复目录登记（audit P1-1）：部分 Experiment 目录共享同一 NIX 内部 subject_id。
        # 若静默合并会让 248−4=244 的叙事无法解释 242 的实际可评估数。
        if subj_id in true_digits:
            skipped.append(
                f"{subj_id} ({exp_name}): 与 {true_digit_sources[subj_id]} 重复的 NIX 被试，"
                f"已跳过以避免分母/标签覆盖"
            )
            continue

        X_list.append(data)  # (N, 3, T) float32，无 NaN
        y_list.append(y)
        d_list.append(digits)
        s_list.extend([subj_id] * len(y))
        true_digits[subj_id] = thought_number
        true_digit_sources[subj_id] = exp_name

    if not X_list:
        raise RuntimeError("未成功读取任何 GTN 被试。检查 mne_data/MNE-P3-data 目录。")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    digits = np.concatenate(d_list).astype(np.int64)
    subject_ids = np.array(s_list)

    if cache_path is not None:
        try:
            _save_gtn_cache(cache_path, X, y, digits, subject_ids, true_digits, skipped)
            print(f"[data] 已写入预处理缓存：{cache_path}", flush=True)
        except Exception as e:  # noqa: BLE001 —— 缓存写失败不影响训练
            print(f"[data] 缓存写入失败（{type(e).__name__}: {e}），继续训练", flush=True)

    return X, y, digits, subject_ids, true_digits, skipped


def make_model(name: str, n_chans: int, n_times: int, sfreq: float, epochs: int, device=None,
               batch_size: int = 512, pretrained_state_dict=None, load_mapping=None,
               freeze_prefixes=(), strict_load=False):
    """按名字构造基线模型（deep 显式传通道数/时间点；classic 传采样率/时间窗）。

    P9 辅助预训练参数仅对 deep 模型生效；见 transfer_policy 方式 A/C。
    """
    from baselines.classic import SWLDA, TemplateMatching, WindowLogisticRegression
    from baselines.riemann import XdawnRiemann

    name = name.lower()
    if name in ("eegnet", "inception", "conformer"):
        from baselines.deep import DeepBaseline, DeepConfig

        return DeepBaseline(
            name,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            config=DeepConfig(epochs=epochs, batch_size=batch_size),
            device=device,
            pretrained_state_dict=pretrained_state_dict,
            load_mapping=load_mapping,
            freeze_prefixes=tuple(freeze_prefixes),
            strict_load=strict_load,
        )
    if name == "swlda":
        return SWLDA(sfreq=sfreq, tmin=-0.2)
    if name == "windowlr":
        return WindowLogisticRegression(sfreq=sfreq, tmin=-0.2, window_ms=(250.0, 500.0))
    if name == "template":
        return TemplateMatching(sfreq=sfreq, tmin=-0.2, window_ms=(250.0, 500.0))
    if name == "xdawn":
        return XdawnRiemann()
    raise ValueError(f"未知模型 {name!r}。可选 eegnet/inception/conformer/swlda/windowlr/template/xdawn/all。")


def main():
    ap = argparse.ArgumentParser(description="GTN 猜数字基线 LOSO 评估")
    ap.add_argument("--model", default="eegnet",
                    help="eegnet/inception/conformer/swlda/windowlr/template/xdawn/all")
    ap.add_argument("--subjects", type=int, default=None, help="限缩前 N 名被试（小规模冒烟用）")
    ap.add_argument("--epochs", type=int, default=30, help="deep 基线训练 epoch 数")
    ap.add_argument("--sfreq", type=float, default=256.0)
    ap.add_argument("--l-freq", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=0,
                    help="BLAS/OpenMP 线程数（0=自动 min(24,逻辑核数)；经典基线性能项）")
    ap.add_argument("--fold-jobs", type=int, default=0,
                    help="经典基线的并行 fold 线程数（0=自动 6）")
    ap.add_argument("--deep-jobs", type=int, default=0,
                    help="deep 基线的并行 fold 线程数（0=自动 2；GPU 上小模型并发可提升利用率）")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="deep 基线训练 batch size（GPU 实测 64→512 约 4-9x 加速）")
    ap.add_argument("--load-jobs", type=int, default=0,
                    help="GTN 读入/预处理的并行线程数（0=自动 min(6,逻辑核数)）")
    ap.add_argument("--cache-dir", default="experiments/cache",
                    help="GTN 预处理张量缓存目录（命中后跳过 MNE 预处理）")
    ap.add_argument("--no-cache", action="store_true",
                    help="禁用预处理张量缓存，强制每次从 NIX 重新预处理")
    ap.add_argument("--save-scores-dir", default=None,
                    help="若给定，每个模型跑完后把逐被试 (predicted,true,group) 记录保存为 "
                         "<dir>/<model>.json，供 experiments/run_paired_test.py 做配对置换检验")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "xpu", "cpu"),
                    help="deep 基线训练设备：auto 按 CUDA→XPU→CPU 检测；dropout/卷积随模型所在设备执行")
    ap.add_argument("--pretrained-checkpoint", default=None,
                    help="P9 辅助预训练 checkpoint（experiments/run_aux_pretrain.py 输出）")
    ap.add_argument("--pretrained-mapping", default=None,
                    help="P9 load_mapping JSON 路径：{source_key: target_key_or_null}")
    ap.add_argument("--freeze-prefixes", default="",
                    help="P9 冻结层前缀，逗号分隔；如 final_layer")
    ap.add_argument("--strict-load", action="store_true",
                    help="P9 预训练权重缺失/形状不匹配时直接报错")
    args = ap.parse_args()

    # 必须先配置线程，再 import numpy/MNE/sklearn（OpenBLAS 启动时读取线程数）
    n_threads = _configure_threads(args.threads)
    logical = os.cpu_count() or 1
    classic_fold_jobs = args.fold_jobs if args.fold_jobs > 0 else min(6, logical)
    deep_fold_jobs = args.deep_jobs if args.deep_jobs > 0 else min(2, logical)
    # P9 预训练实验需要原始模型的 load_report；并行 fold 会 deepcopy 模型导致报告丢失。
    if args.pretrained_checkpoint and args.deep_jobs == 0:
        deep_fold_jobs = 1
    import mne
    import numpy as np
    import torch

    mne.set_log_level("WARNING")
    load_jobs = args.load_jobs if args.load_jobs > 0 else min(6, logical)

    # deep 基线显式设备选择：dropout/卷积/AMP 随模型一起落在 GPU/XPU/CPU。
    if args.device == "auto":
        device = None
    else:
        import torch

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda 但当前环境不可用 CUDA。")
        if device.type == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("--device xpu 但当前环境不可用 XPU。")

    # 预处理张量缓存：同一预处理口径只做一次 MNE 链路，之后直接读 .npz。
    cache_path: Path | None = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # v5（2026-08-22）：--subjects 限缩统一从 nall 缓存派生，避免旧 n<N> 子集缓存
        # 与全量 242 口径的被试/试次不一致而悄悄改变分母。
        cache_path = cache_dir / _gtn_cache_filename(args.sfreq, args.l_freq, -0.2, 0.8, "all")

    print(f"[perf] BLAS/OpenMP threads={n_threads}；load jobs={load_jobs}；"
          f"classic fold jobs={classic_fold_jobs}；deep fold jobs={deep_fold_jobs}；"
          f"deep batch={args.batch_size}；device={args.device}", flush=True)
          # removed duplicate perf line

    from baselines.evaluate import evaluate, loso_folds

    # review v6 P0-3：全量口径显式登记 248 目录 − 缺元数据/全剔除 = 可评估被试数
    all_exp_dirs = [d for d in GTN_ROOT.iterdir() if d.is_dir() and d.name.startswith("Experiment")]
    total_dirs = len(all_exp_dirs)

    print(f"[data] 读入 GTN 被试（subjects={args.subjects or '全部'}，sfreq={args.sfreq}，l_freq={args.l_freq}；"
          f"cache={cache_path or '禁用'}）...")
    X, y, digits, subject_ids, true_digits, skipped = load_gtn_subjects(
        args.subjects, sfreq=args.sfreq, l_freq=args.l_freq, load_jobs=load_jobs, cache_path=cache_path
    )
    if args.subjects:
        keep_subj = np.unique(subject_ids)[: args.subjects]
        keep = np.isin(subject_ids, keep_subj)
        X, y, digits, subject_ids = X[keep], y[keep], digits[keep], subject_ids[keep]
        true_digits = {k: v for k, v in true_digits.items() if k in set(keep_subj.tolist())}
    n_subj = len(np.unique(subject_ids))
    n_times = X.shape[2]
    # 每被试平均试次数 / 每数字平均试次数（K，命中率锚点校准用）
    trials_per_subj = X.shape[0] / n_subj
    print(f"[data] 完成：X={X.shape} y={y.shape} 被试数={n_subj}/{total_dirs} 目录"
          f"（可评估/总目录，review v6 P0-3）")
    print(f"[data] 试次/被试≈{trials_per_subj:.1f}（K≈{trials_per_subj / 9:.1f}/数字）")
          
    print(f"[data] target 基率={y.mean():.4f}（应为 1/9≈0.111，D-gtn-base-rate）")
    if skipped:
        print(f"[data] 跳过 {len(skipped)} 个目录/被试（含缺 .txt 元数据、全 epoch 剔除与重复 NIX 被试）："
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
        print(f"[data] 告警：请以 n_subj/{total_dirs} 作为可评估口径，"
              f"不要静默把被剔除被试从分母中消失。")

    # P9 辅助预训练：加载 checkpoint 与 load_mapping；仅 deep 模型使用。
    pretrained_state_dict = None
    load_mapping = None
    if args.pretrained_checkpoint:
        from baselines.deep import DeepBaseline

        pretrained_state_dict = DeepBaseline.load_state_dict_file(args.pretrained_checkpoint)
        print(f"[p9] 加载辅助预训练 checkpoint：{args.pretrained_checkpoint}")
        if args.pretrained_mapping:
            load_mapping = json.loads(Path(args.pretrained_mapping).read_text(encoding="utf-8"))
            print(f"[p9] load_mapping 键数：{len(load_mapping)}")
        if args.freeze_prefixes:
            print(f"[p9] 冻结前缀：{args.freeze_prefixes}")
        # 需要完整 load_report 时，deep fold 不并行（并行路径会 deepcopy 模型并丢失原始报告）。
        if args.deep_jobs == 0:
            deep_fold_jobs = 1


    folds = loso_folds(subject_ids)
    if args.model == "all":
        models = ["template", "windowlr", "swlda", "xdawn", "eegnet", "inception", "conformer"]
    else:
        models = [args.model]

    # P1-4 防线：预训练 checkpoint 在进入 LOSO 前做一次显式形状/映射检查，
    # 避免 strict_load=False 时通道不匹配的层被静默跳过、用户误以为预训练生效。
    if pretrained_state_dict is not None:
        freeze_prefixes = (
            tuple(x.strip() for x in args.freeze_prefixes.split(",") if x.strip())
            if args.freeze_prefixes
            else ()
        )
        for name in models:
            if name not in DEEP_MODELS:
                continue
            probe = DeepBaseline(
                name,
                n_chans=3,
                n_times=n_times,
                sfreq=args.sfreq,
                device=torch.device("cpu"),
                pretrained_state_dict=pretrained_state_dict,
                load_mapping=load_mapping,
                freeze_prefixes=freeze_prefixes,
                strict_load=args.strict_load,
            )
            probe.model_ = probe._make_model()
            probe._apply_pretrained_state_dict()
            n_loaded = sum(1 for e in probe.load_report if e["event"] == "loaded")
            n_mismatch = sum(1 for e in probe.load_report if e["event"] == "shape_mismatch")
            n_missing = sum(1 for e in probe.load_report if e["event"] == "missing_target")
            print(
                f"[p9] {name} 预训练形状检查：loaded={n_loaded} "
                f"shape_mismatch={n_mismatch} missing_target={n_missing}"
            )
            if (n_mismatch or n_missing) and not args.strict_load:
                print(
                    "[p9] 警告：存在形状不匹配/缺失目标层，这些层将保持随机初始化。"
                    "建议用 --channels Fz,Cz,Pz 重新预训练，或加 --strict-load 阻断。"
                )

    for name in models:
        fold_jobs = deep_fold_jobs if name in DEEP_MODELS else classic_fold_jobs
        print(f"\n[run] 模型={name}（LOSO，{len(folds)} folds, fold_jobs={fold_jobs}）...")
        model = make_model(
            name, n_chans=3, n_times=n_times, sfreq=args.sfreq, epochs=args.epochs, device=device,
            batch_size=args.batch_size,
              pretrained_state_dict=pretrained_state_dict if name in DEEP_MODELS else None,
              load_mapping=load_mapping if name in DEEP_MODELS else None,
              freeze_prefixes=(
                  tuple(x.strip() for x in args.freeze_prefixes.split(",") if x.strip())
                  if args.freeze_prefixes and name in DEEP_MODELS
                  else ()
              ),
              strict_load=args.strict_load if name in DEEP_MODELS else False,
        )
        summary = evaluate(model, X, y, digits, subject_ids, true_digits, folds, n_jobs=fold_jobs)
        print(f"[result] {name}: 命中率={summary.hit_rate_mean:.4f} (±{summary.hit_rate_std:.4f})  "
              f"balanced_acc={summary.balanced_acc_mean:.4f}  AUC={summary.auc_mean:.4f}  "
              f"(chance=0.111)")
        if args.save_scores_dir:
            scores_path = save_subject_scores(
                summary, name, Path(args.save_scores_dir) / f"{name}.json"
            )
            print(f"[scores] 已保存逐被试结果：{scores_path}")


if __name__ == "__main__":
    main()
