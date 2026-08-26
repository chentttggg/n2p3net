"""GTN 猜数字基线实验入口（Phase 1）。

职责（roadmap Phase 1 + constitution P8）：
    把 GTN（唯一「9 选 1 猜数字」公开数据）读入 → 统一预处理 → LOSO 跨受试评估 → 9 选 1 命中率，
    供 SWLDA / xDAWN+RG / EEGNet / EEG-Inception / EEG Conformer / 免费地板 在完全同一协议下公平对比。

用法（项目根目录，用 .venv）：
    .venv/Scripts/python.exe experiments/run_gtn_baseline.py --model eegnet --subjects 20 --epochs 20
    .venv/Scripts/python.exe experiments/run_gtn_baseline.py --model swlda --subjects 20
    .venv/Scripts/python.exe experiments/run_gtn_baseline.py --model all --subjects 20

三思决策记录（2026-08-21，基于对 GTN 真实数据的实测核查，可能推翻 review v4 部分结论）：
    D-gtn-3ch       GTN 只有 Fz/Cz/Pz 3 导，所有基线都按原生 3 导输入构造。
                    GTN 是独立的三导物理布局，不作为缺失的八导布局处理；这样才能公平复现
                    Vařeka 2016 的 3 导 MLP 77.2% 锚点。classic/riemann 同理使用原生 3 导输入。
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
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

# 让 experiments/ 下的脚本能 import src/ 下的包（review v4 2.2 的 PYTHONPATH 约定）
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# GTN 3 导（真实通道，非 8 导蒙太奇子集）
GTN_STANDARD = ("Fz", "Cz", "Pz")
# Shared default for all deep 3-channel GTN experiments.  An explicit
# ``--epochs`` remains available for ablations and locked reruns.
GTN_DEFAULT_DEEP_EPOCHS = 30
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

    payload = (subj_id, data, y, digits, thought_number, timeline)（数组均可跨线程共享）。
    """
    exp_dir, sfreq, l_freq, tmin, tmax, n_times = args
    exp_name = exp_dir.name
    try:
        import numpy as np

        from data.events import ScheduledEventTimeline
        from data.gtn import read_gtn_experiment
        from data.preprocess import preprocess

        with _single_blas_thread():
            g = read_gtn_experiment(exp_dir)
            result = preprocess(
                g.raw,
                g.events,
                channels=GTN_STANDARD,
                sfreq=sfreq,
                l_freq=l_freq,
                tmin=tmin,
                tmax=tmax,
                n_times=n_times,
            )
        digits = g.events[result.event_indices, 2].astype(int)
        y = (digits == g.thought_number).astype(int)
        timeline = ScheduledEventTimeline(
            event_ids=np.asarray([f"gtn:{exp_name}:{index}" for index in range(len(g.events))]),
            group_ids=np.repeat(g.subject_id, len(g.events)),
            subject_ids=np.repeat(g.subject_id, len(g.events)),
            stimulus_ids=np.asarray(g.events[:, 2], dtype=np.int64),
            onset_samples=result.event_samples,
            onset_times_s=result.event_times_s,
            evidence_available_times_s=result.evidence_available_times_s,
            evidence_indices=result.event_evidence_indices,
            statuses=result.event_statuses,
            status_details=result.event_status_details,
            dataset_ids=np.repeat("gtn", len(g.events)),
            session_ids=np.repeat("", len(g.events)),
            run_ids=np.repeat(exp_name, len(g.events)),
            selection_ids=np.repeat(g.subject_id, len(g.events)),
            complete=True,
            online_causal=result.online_causal,
            timing_source="gtn_nix_event_samples;epoch_right_edge;acausal_preprocessing",
        ).validate(n_epochs=len(result.data))
        return (
            exp_name,
            (g.subject_id, result.data, y, digits, g.thought_number, timeline),
            True,
        )
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
        f"gtn_events_v2_3ch_sf{sfreq:.0f}_lf{l_freq:.1f}_"
        f"tm{tmin:.1f}_tx{tmax:.1f}_n{n_subjects}.npz"
    )


def _save_gtn_cache(
    path: Path, X, y, digits, subject_ids, true_digits, skipped, event_timeline
) -> None:
    """把预处理后的 GTN 张量缓存为单个未压缩 .npz（D-preprocess-cache）。

    训练/评估阶段直接读缓存，完全跳过 MNE 读取、重采样、0.1Hz 高通与 epoch 切分。
    缓存使用固定 dtype 数组且禁止 pickle。完整事件账本保留被伪迹/边界剔除的刺激，
    模型张量仍只存可用 epoch。
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    from data.events import EVENT_TIMELINE_SCHEMA

    event_timeline.validate(n_epochs=len(X))
    true_keys = np.array(list(true_digits.keys()), dtype=str)
    true_values = np.array(list(true_digits.values()), dtype=np.int64)
    np.savez(
        tmp,
        schema=np.asarray("n2p3net_gtn_cache/2"),
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        digits=np.asarray(digits, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=str),
        true_keys=true_keys,
        true_values=true_values,
        skipped=np.asarray(skipped, dtype=str),
        event_schema=np.asarray(EVENT_TIMELINE_SCHEMA),
        event_ids=np.asarray(event_timeline.event_ids, dtype=str),
        event_group_ids=np.asarray(event_timeline.group_ids, dtype=str),
        event_subject_ids=np.asarray(event_timeline.subject_ids, dtype=str),
        event_stimulus_ids=np.asarray(event_timeline.stimulus_ids, dtype=np.int64),
        event_onset_samples=np.asarray(event_timeline.onset_samples, dtype=np.int64),
        event_onset_times_s=np.asarray(event_timeline.onset_times_s, dtype=np.float64),
        event_evidence_available_times_s=np.asarray(
            event_timeline.evidence_available_times_s, dtype=np.float64
        ),
        event_evidence_indices=np.asarray(event_timeline.evidence_indices, dtype=np.int64),
        event_statuses=np.asarray(event_timeline.statuses, dtype=str),
        event_status_details=np.asarray(event_timeline.status_details, dtype=str),
        event_dataset_ids=np.asarray(event_timeline.dataset_ids, dtype=str),
        event_session_ids=np.asarray(event_timeline.session_ids, dtype=str),
        event_run_ids=np.asarray(event_timeline.run_ids, dtype=str),
        event_selection_ids=np.asarray(event_timeline.selection_ids, dtype=str),
        event_complete=np.asarray(event_timeline.complete),
        event_online_causal=np.asarray(event_timeline.online_causal),
        event_timing_source=np.asarray(event_timeline.timing_source),
    )
    tmp.replace(path)


def _load_gtn_cache(path: Path):
    """Load schema-v2 tensors and the complete scheduled-event ledger."""
    import numpy as np

    from data.events import EVENT_TIMELINE_SCHEMA, ScheduledEventTimeline

    with np.load(path, allow_pickle=False) as z:
        schema = str(np.asarray(z["schema"]).item()) if "schema" in z.files else "legacy"
        if schema != "n2p3net_gtn_cache/2":
            raise ValueError(
                f"Unsupported GTN cache schema {schema!r}; rebuild to retain all scheduled events."
            )
        event_schema = str(np.asarray(z["event_schema"]).item())
        if event_schema != EVENT_TIMELINE_SCHEMA:
            raise ValueError(f"Unsupported event timeline schema {event_schema!r}.")
        X = z["X"]
        y = z["y"]
        digits = z["digits"]
        for field_name, values in (
            ("y", y),
            ("digits", digits),
            ("true_values", z["true_values"]),
            ("event_stimulus_ids", z["event_stimulus_ids"]),
            ("event_onset_samples", z["event_onset_samples"]),
            ("event_evidence_indices", z["event_evidence_indices"]),
        ):
            if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
                values.dtype, np.bool_
            ):
                raise ValueError(f"GTN cache {field_name} must have an integer dtype.")
        for field_name in ("event_complete", "event_online_causal"):
            if np.asarray(z[field_name]).dtype != np.dtype(bool):
                raise ValueError(f"GTN cache {field_name} must be a strict boolean.")
        subject_ids = z["subject_ids"].astype(str)
        true_keys = z["true_keys"].astype(str).tolist()
        true_values = z["true_values"].astype(np.int64).tolist()
        skipped = [str(x) for x in z["skipped"].tolist()]
        timeline = ScheduledEventTimeline(
            event_ids=np.asarray(z["event_ids"], dtype=str),
            group_ids=np.asarray(z["event_group_ids"], dtype=str),
            subject_ids=np.asarray(z["event_subject_ids"], dtype=str),
            stimulus_ids=np.asarray(z["event_stimulus_ids"], dtype=np.int64),
            onset_samples=np.asarray(z["event_onset_samples"], dtype=np.int64),
            onset_times_s=np.asarray(z["event_onset_times_s"], dtype=np.float64),
            evidence_available_times_s=np.asarray(
                z["event_evidence_available_times_s"], dtype=np.float64
            ),
            evidence_indices=np.asarray(z["event_evidence_indices"], dtype=np.int64),
            statuses=np.asarray(z["event_statuses"], dtype=str),
            status_details=np.asarray(z["event_status_details"], dtype=str),
            dataset_ids=np.asarray(z["event_dataset_ids"], dtype=str),
            session_ids=np.asarray(z["event_session_ids"], dtype=str),
            run_ids=np.asarray(z["event_run_ids"], dtype=str),
            selection_ids=np.asarray(z["event_selection_ids"], dtype=str),
            complete=bool(np.asarray(z["event_complete"]).item()),
            online_causal=bool(np.asarray(z["event_online_causal"]).item()),
            timing_source=str(np.asarray(z["event_timing_source"]).item()),
        )
    if len(true_keys) != len(set(true_keys)) or any(not key for key in true_keys):
        raise ValueError("GTN cache truth keys must be unique and non-empty.")
    true_digits = dict(zip(true_keys, true_values, strict=True))
    if len(X) != len(y) or len(X) != len(digits) or len(X) != len(subject_ids):
        raise ValueError(
            f"缓存损坏：X/y/digits/subject_ids 长度不一致：{X.shape}/{len(y)}/{len(digits)}/{len(subject_ids)}"
        )
    if not np.isfinite(X).all():
        raise ValueError("GTN cache X contains NaN/inf.")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("GTN cache y must be binary.")
    if any(not subject for subject in subject_ids):
        raise ValueError("GTN cache subject ids must be non-empty.")
    timeline.validate(n_epochs=len(X))
    event_evidence = np.asarray(timeline.evidence_indices, dtype=np.int64)
    available = event_evidence >= 0
    aligned_groups = np.empty(len(X), dtype=object)
    aligned_digits = np.empty(len(X), dtype=np.int64)
    aligned_groups[event_evidence[available]] = np.asarray(timeline.group_ids)[available]
    aligned_digits[event_evidence[available]] = np.asarray(timeline.stimulus_ids)[available]
    if not np.array_equal(aligned_groups.astype(str), subject_ids.astype(str)):
        raise ValueError("GTN cache event groups disagree with model-ready subject ids.")
    if not np.array_equal(aligned_digits, digits.astype(np.int64)):
        raise ValueError("GTN cache event stimuli disagree with model-ready digits.")
    if set(timeline.groups) != set(true_digits):
        raise ValueError("GTN cache timeline groups differ from the frozen truth universe.")
    expected_y = np.asarray(
        [int(digit == true_digits[group]) for digit, group in zip(digits, subject_ids, strict=True)],
        dtype=np.int64,
    )
    if not np.array_equal(y.astype(np.int64), expected_y):
        raise ValueError("GTN cache labels violate y == (stimulus_digit == frozen truth).")
    return X, y, digits, subject_ids, true_digits, skipped, timeline


def save_subject_scores(
    summary,
    model_name: str,
    path: Path,
    *,
    seed: int | None = None,
    evaluation_mode: str = "development",
    protocol_sha256: str | None = None,
    confirmatory_id: str | None = None,
    confirmatory_lock_sha256: str | None = None,
    source_sha256: str | None = None,
    runtime_sha256: str | None = None,
    external_assets_sha256: str | None = None,
) -> Path:
    """把 evaluate() 返回的逐被试 (predicted, true, group) 记录落盘为 JSON。

    供 experiments/run_paired_test.py 做配对置换检验：每个 group（GTN=被试）一个
    hit（0/1）。正式比较要求完整冻结全集逐项一致，禁止取交集改变分母。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def serialized_records(rows):
        output = []
        seen: set[str] = set()
        for predicted, true, group in rows:
            subject = str(group)
            if subject in seen:
                raise ValueError(f"Duplicate score record for evaluation unit {subject!r}.")
            seen.add(subject)
            output.append(
                {
                    "subject": subject,
                    "predicted": None if predicted is None else int(predicted),
                    "true": int(true),
                    "available": predicted is not None,
                    "hit": int(predicted is not None and predicted == true),
                }
            )
        return output

    records = serialized_records(summary.subject_records)
    primary_metric = getattr(summary, "primary_decision_metric", None)
    primary_records = []
    if primary_metric and primary_metric in getattr(summary, "decision_metrics", {}):
        metric = summary.decision_metrics[primary_metric]
        primary_records = serialized_records(metric.subject_records)
    expected_units = tuple(str(unit) for unit in summary.evaluation_units)
    if tuple(row["subject"] for row in records) != expected_units:
        raise ValueError("All-trial records do not exactly match the frozen evaluation universe.")
    if tuple(row["subject"] for row in primary_records) != expected_units:
        raise ValueError("Primary records do not exactly match the frozen evaluation universe.")
    descriptive_records_by_metric = {
        str(name): list(rows)
        for name, rows in getattr(summary, "descriptive_decision_records", {}).items()
    }
    for name, rows in descriptive_records_by_metric.items():
        if tuple(str(row["subject"]) for row in rows) != expected_units:
            raise ValueError(
                f"Descriptive records for {name!r} do not match the frozen evaluation universe."
            )
    descriptive_primary_records = descriptive_records_by_metric.get(primary_metric, [])
    payload = {
        "schema": "n2p3net_subject_scores/2",
        "model": model_name,
        "seed": seed,
        "evaluation_mode": evaluation_mode,
        "protocol_sha256": protocol_sha256,
        "confirmatory_id": confirmatory_id,
        "confirmatory_lock_sha256": confirmatory_lock_sha256,
        "source_sha256": source_sha256,
        "runtime_sha256": runtime_sha256,
        "external_assets_sha256": external_assets_sha256,
        "dataset_sha256": summary.dataset_sha256,
        "cohort_sha256": summary.cohort_sha256,
        "evaluation_units": list(expected_units),
        "fold_protocol": summary.fold_protocol,
        "created_utc": datetime.now(UTC).isoformat(),
        "n_subjects": len(expected_units),
        "n_covered": sum(row["available"] for row in records),
        "hit_rate_mean": float(summary.hit_rate_mean),
        "hit_rate_std": float(summary.hit_rate_std),
        "balanced_acc_mean": float(summary.balanced_acc_mean),
        "auc_mean": None if summary.auc_mean != summary.auc_mean else float(summary.auc_mean),
        "primary_decision_metric": primary_metric,
        "primary_hit_rate": (
            float(summary.decision_metrics[primary_metric].hit_rate)
            if primary_metric in getattr(summary, "decision_metrics", {})
            else float(summary.hit_rate_mean)
        ),
        "primary_n_subjects": len(expected_units),
        "primary_n_covered": sum(row["available"] for row in primary_records),
        "primary_coverage": (
            float(summary.decision_metrics[primary_metric].coverage)
            if primary_metric in getattr(summary, "decision_metrics", {})
            else 0.0
        ),
        "primary_conditional_hit_rate": (
            float(summary.decision_metrics[primary_metric].conditional_hit_rate)
            if primary_metric in getattr(summary, "decision_metrics", {})
            else None
        ),
        "primary_budget_semantics": (
            summary.decision_metrics[primary_metric].budget_semantics
            if primary_metric in getattr(summary, "decision_metrics", {})
            else None
        ),
        "primary_metric_gate": dict(getattr(summary, "primary_metric_gate", {}) or {}),
        "primary_records": primary_records,
        "descriptive_primary_records": descriptive_primary_records,
        "descriptive_records_by_metric": descriptive_records_by_metric,
        "repetition_efficiency": (
            asdict(summary.repetition_efficiency)
            if summary.repetition_efficiency is not None
            else None
        ),
        "records": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["score_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
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
    n_times: int | None = None,
    load_jobs: int = 0,
    cache_path: str | Path | None = None,
    cache_fail_closed: bool = False,
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
                if cache_fail_closed:
                    raise RuntimeError(f"Frozen GTN cache failed validation: {cache_path}.") from e
                print(f"[data] 缓存读取失败（{type(e).__name__}: {e}），回退重新预处理", flush=True)

    exps = sorted([d for d in GTN_ROOT.iterdir() if d.is_dir() and d.name.startswith("Experiment")])
    if max_subjects:
        exps = exps[:max_subjects]

    from data.events import concatenate_event_timelines

    X_list, y_list, d_list, s_list, timelines = [], [], [], [], []
    true_digits: dict = {}
    true_digit_sources: dict[str, str] = {}
    skipped: list[str] = []

    # 线程池并行预处理（D-load-threads）：读取 NIX + MNE 滤波/切窗是 CPU/IO 密集，
    # 不同被试完全独立；实测 20 被试 4 线程 3.2s→1.0s。worker 内 BLAS 限单线程。
    load_jobs = int(load_jobs) if load_jobs and load_jobs > 0 else min(6, os.cpu_count() or 1)
    if n_times is None:
        n_times = int(round((float(tmax) - float(tmin)) * float(sfreq)))
    if n_times < 2:
        raise ValueError("Derived GTN epoch length must contain at least two samples.")
    task_args = [(exp, sfreq, l_freq, tmin, tmax, n_times) for exp in exps]
    if load_jobs > 1 and len(exps) > 1:
        with ThreadPoolExecutor(max_workers=load_jobs) as executor:
            load_results = list(executor.map(_load_one_gtn_subject, task_args))
    else:
        load_results = [_load_one_gtn_subject(args) for args in task_args]

    for exp_name, payload, ok in load_results:
        if not ok:
            skipped.append(payload)
            continue
        subj_id, data, y, digits, thought_number, timeline = payload

        # 重复目录登记（audit P1-1）：部分 Experiment 目录共享同一 NIX 内部 subject_id。
        # 若静默合并会让 248−4=244 的叙事无法解释 242 的实际可评估数。
        if subj_id in true_digits:
            skipped.append(
                f"{subj_id} ({exp_name}): 与 {true_digit_sources[subj_id]} 重复的 NIX 被试，"
                f"已跳过以避免分母/标签覆盖"
            )
            continue

        true_digits[subj_id] = thought_number
        true_digit_sources[subj_id] = exp_name
        timelines.append(timeline)

        # Preserve the subject in the frozen ITT universe even when every epoch is unavailable.
        if len(y) == 0:
            skipped.append(f"{subj_id} ({exp_name}): all scheduled epochs unavailable; ITT miss")
            continue

        X_list.append(data)  # (N, 3, T) float32，无 NaN
        y_list.append(y)
        d_list.append(digits)
        s_list.extend([subj_id] * len(y))

    if not X_list:
        raise RuntimeError("未成功读取任何 GTN 被试。检查 mne_data/MNE-P3-data 目录。")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    digits = np.concatenate(d_list).astype(np.int64)
    subject_ids = np.array(s_list)
    event_timeline = concatenate_event_timelines(timelines)

    if cache_path is not None:
        try:
            _save_gtn_cache(
                cache_path, X, y, digits, subject_ids, true_digits, skipped, event_timeline
            )
            print(f"[data] 已写入预处理缓存：{cache_path}", flush=True)
        except Exception as e:  # noqa: BLE001 —— 缓存写失败不影响训练
            print(f"[data] 缓存写入失败（{type(e).__name__}: {e}），继续训练", flush=True)

    return X, y, digits, subject_ids, true_digits, skipped, event_timeline


def make_model(
    name: str,
    n_chans: int,
    n_times: int,
    sfreq: float,
    epochs: int,
    device=None,
    batch_size: int = 256,
    pretrained_state_dict=None,
    load_mapping=None,
    freeze_prefixes=(),
    strict_load=False,
    seed: int = 0,
):
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
            config=DeepConfig(epochs=epochs, batch_size=batch_size, seed=seed),
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
    raise ValueError(
        f"未知模型 {name!r}。可选 eegnet/inception/conformer/swlda/windowlr/template/xdawn/all。"
    )


def main():
    ap = argparse.ArgumentParser(description="GTN 猜数字基线 LOSO 评估", allow_abbrev=False)
    ap.add_argument(
        "--model",
        default="eegnet",
        help="eegnet/inception/conformer/swlda/windowlr/template/xdawn/all",
    )
    ap.add_argument("--subjects", type=int, default=None, help="限缩前 N 名被试（小规模冒烟用）")
    ap.add_argument(
        "--epochs",
        type=int,
        default=GTN_DEFAULT_DEEP_EPOCHS,
        help="deep 基线训练 epoch 数",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--evaluation-mode", choices=("development", "confirmatory"), default="development"
    )
    ap.add_argument("--protocol-sha256", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dataset-sha256", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--confirmatory-lock", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--source-sha256", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--runtime-sha256", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--external-assets-sha256", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--cohort-manifest",
        default=str(
            Path(__file__).resolve().parent / "protocols" / "gtn_confirmatory_cohort_v1.json"
        ),
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--primary-decision",
        default="exact_llr@3",
        choices=tuple(
            [
                f"{semantics}_{aggregation}@{k}"
                for semantics in ("exact", "prefix_minK")
                for aggregation in ("sum", "mean", "llr")
                for k in ("1", "3", "5", "10", "15")
            ]
            + [f"all_{aggregation}" for aggregation in ("sum", "mean", "llr")]
            + [
                f"flash_{aggregation}@{n}"
                for aggregation in ("sum", "mean", "llr")
                for n in ("9", "27", "45", "90", "135")
            ]
        ),
    )
    ap.add_argument("--fixed-error-rate", type=float, default=0.05)
    ap.add_argument("--primary-min-coverage", type=float, default=0.90)
    ap.add_argument("--efficiency-min-coverage", type=float, default=0.90)
    ap.add_argument(
        "--repetition-duration-s",
        type=float,
        default=None,
        help="Measured seconds for one complete 1-9 repetition; required for bits/min ITR.",
    )
    ap.add_argument("--sfreq", type=float, default=256.0)
    ap.add_argument("--l-freq", type=float, default=0.1)
    ap.add_argument(
        "--epoch-tmax",
        type=float,
        default=0.8,
        help="exclusive epoch right edge in seconds; Neural-RIDE uses 1.2",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="BLAS/OpenMP 线程数（0=自动 min(24,逻辑核数)；经典基线性能项）",
    )
    ap.add_argument(
        "--fold-jobs", type=int, default=0, help="经典基线的并行 fold 线程数（0=自动 6）"
    )
    ap.add_argument(
        "--deep-jobs",
        type=int,
        default=0,
        help="deep 基线的并行 fold 线程数（0=自动 2；GPU 上小模型并发可提升利用率）",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="deep 基线训练 batch size（默认 256；可按显存手动调整）",
    )
    ap.add_argument(
        "--load-jobs",
        type=int,
        default=0,
        help="GTN 读入/预处理的并行线程数（0=自动 min(6,逻辑核数)）",
    )
    ap.add_argument(
        "--cache-dir",
        default="experiments/cache",
        help="GTN 预处理张量缓存目录（命中后跳过 MNE 预处理）",
    )
    ap.add_argument(
        "--no-cache", action="store_true", help="禁用预处理张量缓存，强制每次从 NIX 重新预处理"
    )
    ap.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help="build/validate the requested preprocessing cache and exit before LOSO",
    )
    ap.add_argument(
        "--save-scores-dir",
        default=None,
        help="若给定，每个模型跑完后把逐被试 (predicted,true,group) 记录保存为 "
        "<dir>/<model>.json，供 experiments/run_paired_test.py 做配对置换检验",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "xpu", "cpu"),
        help="deep 基线训练设备：auto 按 CUDA→XPU→CPU 检测；dropout/卷积随模型所在设备执行",
    )
    ap.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="P9 辅助预训练 checkpoint（experiments/run_eeg_pretrain.py 输出）",
    )
    ap.add_argument(
        "--pretrained-mapping",
        default=None,
        help="P9 load_mapping JSON 路径：{source_key: target_key_or_null}",
    )
    ap.add_argument("--freeze-prefixes", default="", help="P9 冻结层前缀，逗号分隔；如 final_layer")
    ap.add_argument(
        "--strict-load", action="store_true", help="P9 预训练权重缺失/形状不匹配时直接报错"
    )
    args = ap.parse_args()
    if not 0.0 < args.primary_min_coverage <= 1.0:
        ap.error("--primary-min-coverage must be in (0,1]")
    if not 0.0 < args.efficiency_min_coverage <= 1.0:
        ap.error("--efficiency-min-coverage must be in (0,1]")

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
        if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("--device xpu 但当前环境不可用 XPU。")
    if args.evaluation_mode == "confirmatory" and args.model in DEEP_MODELS:
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Confirmatory deep-baseline evaluation requires explicit --device cuda; "
                "CPU/XPU fallback is forbidden."
            )
        if deep_fold_jobs != 1:
            raise RuntimeError(
                "Confirmatory deep-baseline evaluation requires --deep-jobs 1 because "
                "concurrent folds share the process-wide CUDA RNG."
            )

    # 预处理张量缓存：同一预处理口径只做一次 MNE 链路，之后直接读 .npz。
    cache_path: Path | None = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # v5（2026-08-22）：--subjects 限缩统一从 nall 缓存派生，避免旧 n<N> 子集缓存
        # 与全量 242 口径的被试/试次不一致而悄悄改变分母。
        cache_path = cache_dir / _gtn_cache_filename(
            args.sfreq, args.l_freq, -0.2, args.epoch_tmax, "all"
        )

    print(
        f"[perf] BLAS/OpenMP threads={n_threads}；load jobs={load_jobs}；"
        f"classic fold jobs={classic_fold_jobs}；deep fold jobs={deep_fold_jobs}；"
        f"deep batch={args.batch_size}；device={args.device}",
        flush=True,
    )
    # removed duplicate perf line

    from baselines.evaluate import evaluate, loso_folds

    # review v6 P0-3：全量口径显式登记 248 目录 − 缺元数据/全剔除 = 可评估被试数
    all_exp_dirs = [d for d in GTN_ROOT.iterdir() if d.is_dir() and d.name.startswith("Experiment")]
    total_dirs = len(all_exp_dirs)

    print(
        f"[data] 读入 GTN 被试（subjects={args.subjects or '全部'}，sfreq={args.sfreq}，l_freq={args.l_freq}；"
        f"cache={cache_path or '禁用'}）..."
    )
    epoch_n_times = int(round((args.epoch_tmax + 0.2) * args.sfreq))
    X, y, digits, subject_ids, true_digits, skipped, event_timeline = load_gtn_subjects(
        args.subjects,
        sfreq=args.sfreq,
        l_freq=args.l_freq,
        tmin=-0.2,
        tmax=args.epoch_tmax,
        n_times=epoch_n_times,
        load_jobs=load_jobs,
        cache_path=cache_path,
        cache_fail_closed=args.evaluation_mode == "confirmatory",
    )
    if args.subjects:
        keep_subj = np.unique(subject_ids)[: args.subjects]
        keep = np.isin(subject_ids, keep_subj)
        X, y, digits, subject_ids = X[keep], y[keep], digits[keep], subject_ids[keep]
        true_digits = {k: v for k, v in true_digits.items() if k in set(keep_subj.tolist())}
        event_timeline = event_timeline.subset_groups(set(keep_subj.astype(str).tolist()))
    n_subj = len(np.unique(subject_ids))
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest() if cache_path else None
    if args.dataset_sha256 is not None and cache_sha256 != args.dataset_sha256:
        raise RuntimeError(
            f"Frozen dataset hash mismatch: expected {args.dataset_sha256}, got {cache_sha256}."
        )
    if args.evaluation_mode == "confirmatory":
        if args.subjects is not None or args.model == "all" or cache_path is None:
            raise RuntimeError(
                "Confirmatory evaluation requires one model and the complete versioned cache."
            )
        if not args.protocol_sha256 or not args.dataset_sha256:
            raise RuntimeError("Confirmatory runner requires frozen protocol and dataset hashes.")
        from baselines.experiment_protocol import (
            claim_confirmatory_seed,
            confirmatory_units_from_manifest,
            external_assets_sha256,
            runtime_environment_sha256,
            source_tree_sha256,
            validate_confirmatory_lock,
            validate_eligibility_manifest,
        )

        eligibility_manifest = validate_eligibility_manifest(
            args.cohort_manifest,
            dataset="gtn",
            truth_by_unit=true_digits,
        )
        evaluation_units = confirmatory_units_from_manifest(
            eligibility_manifest,
            true_digits,
        )
        if (
            not args.source_sha256
            or source_tree_sha256(Path(__file__).resolve().parent.parent) != args.source_sha256
            or not args.runtime_sha256
            or runtime_environment_sha256() != args.runtime_sha256
        ):
            raise RuntimeError("Confirmatory source or dependency runtime identity mismatch.")
        if external_assets_sha256(
            {
                "frozen_erp_prior": None,
                "pretrained_checkpoint": args.pretrained_checkpoint,
                "pretrained_mapping": args.pretrained_mapping,
            }
        ) != args.external_assets_sha256:
            raise RuntimeError("Confirmatory external asset identity mismatch.")
        if not args.confirmatory_lock:
            raise RuntimeError("Confirmatory runner requires a one-use lock manifest.")
        lock_payload, confirmatory_lock_sha256 = validate_confirmatory_lock(
            args.confirmatory_lock,
            dataset_sha256=cache_sha256,
            protocol_sha256=args.protocol_sha256,
            primary_metric=args.primary_decision,
            seed=args.seed,
            runner="baseline",
            model=args.model,
        )
        claim_confirmatory_seed(
            args.confirmatory_lock,
            seed=args.seed,
            run_identity={"runner": "baseline", "model": args.model},
        )
        confirmatory_id = str(lock_payload["confirmatory_id"])
    else:
        confirmatory_id = None
        confirmatory_lock_sha256 = None
        evaluation_units = tuple(sorted(true_digits))
    n_times = X.shape[2]
    # 每被试平均试次数 / 每数字平均试次数（K，命中率锚点校准用）
    trials_per_subj = X.shape[0] / n_subj
    print(
        f"[data] 完成：X={X.shape} y={y.shape} 被试数={n_subj}/{total_dirs} 目录"
        f"（可评估/总目录，review v6 P0-3）"
    )
    print(f"[data] 试次/被试≈{trials_per_subj:.1f}（K≈{trials_per_subj / 9:.1f}/数字）")
    if args.prepare_cache_only:
        print(
            f"[data] cache ready: {cache_path or 'disabled'}; "
            f"epoch=[-0.2,{args.epoch_tmax})s, n_times={n_times}",
            flush=True,
        )
        return

    print(f"[data] target 基率={y.mean():.4f}（应为 1/9≈0.111，D-gtn-base-rate）")
    if skipped:
        print(
            f"[data] 跳过 {len(skipped)} 个目录/被试（含缺 .txt 元数据、全 epoch 剔除与重复 NIX 被试）："
            f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}"
        )
        print(
            f"[data] 告警：请以 n_subj/{total_dirs} 作为可评估口径，"
            f"不要静默把被剔除被试从分母中消失。"
        )

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
    if args.evaluation_mode == "confirmatory":
        allowed = set(evaluation_units)
        folds = [
            (train_mask, test_mask)
            for train_mask, test_mask in folds
            if str(subject_ids[test_mask][0]) in allowed
        ]
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
            name,
            n_chans=3,
            n_times=n_times,
            sfreq=args.sfreq,
            epochs=args.epochs,
            device=device,
            batch_size=args.batch_size,
            pretrained_state_dict=pretrained_state_dict if name in DEEP_MODELS else None,
            load_mapping=load_mapping if name in DEEP_MODELS else None,
            freeze_prefixes=(
                tuple(x.strip() for x in args.freeze_prefixes.split(",") if x.strip())
                if args.freeze_prefixes and name in DEEP_MODELS
                else ()
            ),
            strict_load=args.strict_load if name in DEEP_MODELS else False,
            seed=args.seed,
        )
        summary = evaluate(
            model,
            X,
            y,
            digits,
            subject_ids,
            true_digits,
            folds,
            n_jobs=fold_jobs,
            primary_decision_metric=args.primary_decision,
            fixed_error_rate=args.fixed_error_rate,
            primary_min_coverage=args.primary_min_coverage,
            efficiency_min_coverage=args.efficiency_min_coverage,
            repetition_duration_s=args.repetition_duration_s,
            flash_budgets=(9, 27, 45, 90, 135),
            event_timeline=event_timeline,
            evaluation_units=evaluation_units,
            fold_protocol=(
                "partial_loso" if args.evaluation_mode == "confirmatory" else "loso"
            ),
            dataset_sha256=cache_sha256,
        )
        primary = summary.decision_metrics[args.primary_decision]
        print(
            f"[result] {name}: primary {args.primary_decision}={primary.hit_rate:.4f} "
            f"coverage={primary.n_covered}/{primary.n_total}; "
            f"all-trial sum={summary.hit_rate_mean:.4f} (±{summary.hit_rate_std:.4f})  "
            f"balanced_acc={summary.balanced_acc_mean:.4f}  AUC={summary.auc_mean:.4f}  "
            f"(uniform nominal chance=0.111; empirical priors recorded)"
        )
        primary_gate = summary.primary_metric_gate
        print(
            f"[primary metric gate] passed={primary_gate.get('passed')} "
            f"claim_eligible={primary_gate.get('claim_eligible')} "
            f"coverage={primary_gate.get('n_covered')}/{primary_gate.get('n_total')} "
            f"failed_checks={primary_gate.get('failed_checks', [])}",
            flush=True,
        )
        efficiency = summary.repetition_efficiency
        print(
            f"[efficiency] {efficiency.aggregation}/{efficiency.budget_semantics} "
            f"error<={efficiency.target_error_rate:.3f}: "
            f"K={efficiency.repetitions_to_target_error}; "
            f"minimum_coverage={efficiency.minimum_coverage:.3f}",
            flush=True,
        )
        if args.save_scores_dir:
            if args.evaluation_mode == "confirmatory" and (
                source_tree_sha256(Path(__file__).resolve().parent.parent) != args.source_sha256
                or runtime_environment_sha256() != args.runtime_sha256
                or external_assets_sha256(
                    {
                        "frozen_erp_prior": None,
                        "pretrained_checkpoint": args.pretrained_checkpoint,
                        "pretrained_mapping": args.pretrained_mapping,
                    }
                )
                != args.external_assets_sha256
            ):
                raise RuntimeError("Source, runtime, or external assets changed during evaluation.")
            scores_path = save_subject_scores(
                summary,
                name,
                Path(args.save_scores_dir) / f"{name}.json",
                seed=args.seed,
                evaluation_mode=args.evaluation_mode,
                protocol_sha256=args.protocol_sha256,
                confirmatory_id=confirmatory_id,
                confirmatory_lock_sha256=confirmatory_lock_sha256,
                source_sha256=args.source_sha256,
                runtime_sha256=args.runtime_sha256,
                external_assets_sha256=args.external_assets_sha256,
            )
            print(f"[scores] 已保存逐被试结果：{scores_path}")


if __name__ == "__main__":
    main()
