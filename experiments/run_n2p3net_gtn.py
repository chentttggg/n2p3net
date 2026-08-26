"""Run the current Neural-RIDE protocol on native GTN Fz/Cz/Pz epochs.

The entry point consumes only the versioned [-200,+1200) GTN cache. It never
pads GTN to an eight-channel layout or substitutes electrodes. Cross-dataset
training uses the separately audited ``run_multidataset_transfer.py`` entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Keep the parent CUDA-free until Linux fork workers initialize their own
# contexts. PyTorch's NVML-based availability check does not poison fork.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
if __name__ == "__main__" and sys.platform == "win32" and not sys.flags.utf8_mode:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run([sys.executable, *sys.argv], env=env, check=False)
    raise SystemExit(completed.returncode)
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.evaluate import (  # noqa: E402
    _fold_threadpool_limits,
    evaluate,
    loso_folds,
)
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from data.channel import build_channel_identity  # noqa: E402
from experiments.run_gtn_baseline import (  # noqa: E402
    GTN_DEFAULT_DEEP_EPOCHS,
    GTN_STANDARD,
    _gtn_cache_filename,
    _load_gtn_cache,
    save_subject_scores,
)
from models.component_window import (  # noqa: E402
    GTN_CHILD_DTAU_BOUNDS,
    GTN_CHILD_SIGMA_BOUNDS,
    GTN_CHILD_TAU0_BOUNDS,
    GTN_CHILD_TAU0_MS,
)
from models.erp_calibration import FoldERPCalibrator  # noqa: E402
from models.heads import Z2_AUX_POOLS  # noqa: E402
from models.n2p3net import N2P3Net  # noqa: E402
from models.time_axis import EpochTimeAxis  # noqa: E402
from train.batch import DEFAULT_TRAINING_BATCH, TrainingBatchConfig  # noqa: E402
from train.device import get_device  # noqa: E402
from train.recipe import (  # noqa: E402
    GTN_DIGIT_TASK,
    NEURAL_RIDE_V12,
    NEURAL_RIDE_V12_STRICT_PAST_RESEARCH,
)
from train.trainer import COMPILE_MODES, LR_SCHEDULES  # noqa: E402


def _build_gtn_x_and_identity(
    X3: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Build the one physically valid GTN layout: native Fz/Cz/Pz."""
    X = np.asarray(X3, dtype=np.float32)
    mask = torch.ones(3, dtype=torch.bool)
    identity = build_channel_identity(
        ch_names=GTN_STANDARD,
        channel_mask=tuple(mask.tolist()),
        allow_missing_positions=False,
    )
    E_chn = torch.from_numpy(identity.embedding)
    return X, E_chn, mask


def _postprocess_cpu_threads() -> int:
    """Return the parent-only finalization CPU budget.

    Fold workers divide the host budget through ``FOLD_CPU_THREADS``. Once they
    have exited, the parent may reclaim the full OpenMP budget for the required
    final-fold refit and artifact inference.
    """

    raw = os.environ.get("POSTPROCESS_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
    try:
        threads = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"POSTPROCESS_CPU_THREADS must be a positive integer, got {raw!r}"
        ) from exc
    if threads < 1:
        raise ValueError(f"POSTPROCESS_CPU_THREADS must be positive, got {threads}")
    return threads


def _configure_parent_cpu_scheduler(cpu_threads: int) -> dict[str, int]:
    """Initialize the parent process CPU schedulers before training starts."""

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    current_interop = torch.get_num_interop_threads()
    if current_interop != cpu_threads:
        try:
            torch.set_num_interop_threads(cpu_threads)
        except RuntimeError as exc:
            raise RuntimeError(
                "Parent CPU scheduler must be configured before PyTorch parallel work; "
                f"requested {cpu_threads} inter-op threads, current {current_interop}."
            ) from exc
    return {
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
    }


def _outer_prequential_claim_gate(per_fold) -> dict[str, object]:
    """Summarize locked outer-fold evidence without changing the fitted model."""

    auc_deltas: list[float] = []
    brier_deltas: list[float] = []
    active: list[bool] = []
    density_passed: list[bool] = []
    complete = True
    for fold in per_fold:
        audit = getattr(fold, "audit", {}) or {}
        branches = audit.get("branches", {})
        final = branches.get("final", {})
        pcw = branches.get("pcw", {})
        gate = audit.get("prequential_gate", {})
        required = (
            final.get("auc"),
            pcw.get("auc"),
            final.get("brier"),
            pcw.get("brier"),
        )
        if any(value is None or not np.isfinite(float(value)) for value in required):
            complete = False
            continue
        auc_deltas.append(float(final["auc"]) - float(pcw["auc"]))
        brier_deltas.append(float(final["brier"]) - float(pcw["brier"]))
        active.append(float(audit.get("prequential_coefficient", 0.0)) > 0.0)
        density_passed.append(bool(gate.get("passed", False)))

    n_folds = len(per_fold)
    active_fraction = float(np.mean(active)) if active else 0.0
    auc_win_fraction = float(np.mean(np.asarray(auc_deltas) > 0.0)) if auc_deltas else 0.0
    brier_win_fraction = float(np.mean(np.asarray(brier_deltas) < 0.0)) if brier_deltas else 0.0
    checks = {
        "complete_outer_fold_audits": complete and len(auc_deltas) == n_folds,
        "at_least_five_outer_folds": n_folds >= 5,
        "density_gate_passed_every_fold": bool(density_passed) and all(density_passed),
        "fusion_active_in_strict_majority": active_fraction > 0.5,
        "mean_auc_improved": bool(auc_deltas) and float(np.mean(auc_deltas)) > 0.0,
        "auc_improved_in_strict_majority": auc_win_fraction > 0.5,
        "mean_brier_improved": bool(brier_deltas) and float(np.mean(brier_deltas)) < 0.0,
        "brier_improved_in_strict_majority": brier_win_fraction > 0.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "n_outer_folds": n_folds,
        "active_fusion_fraction": active_fraction,
        "auc_delta_by_fold": auc_deltas,
        "mean_auc_delta": float(np.mean(auc_deltas)) if auc_deltas else None,
        "auc_win_fraction": auc_win_fraction,
        "brier_delta_by_fold": brier_deltas,
        "mean_brier_delta": float(np.mean(brier_deltas)) if brier_deltas else None,
        "brier_win_fraction": brier_win_fraction,
        "scope": "locked_outer_test_claim_only",
    }


def _recipe_for_innovation_weight(weight: float):
    return NEURAL_RIDE_V12_STRICT_PAST_RESEARCH if weight > 0.0 else NEURAL_RIDE_V12


def _recipe_for_research_route(innovation_weight: float, z2_aux_mode: str):
    """Select the named recipe for orthogonal research branches.

    ``z2_aux_mode`` is one of ``off``, ``add`` or ``replace``. Production
    keeps both research branches off (E5). Enabling an auxiliary branch
    always yields an explicitly named research recipe, never the canonical
    ``neural_ride_v12_pcw_fail_closed`` name.
    """

    from dataclasses import replace


    if z2_aux_mode not in ("off", "add", "replace"):
        raise ValueError(f"z2_aux_mode must be off/add/replace, got {z2_aux_mode!r}.")
    base = _recipe_for_innovation_weight(innovation_weight)
    if z2_aux_mode == "off":
        return base
    prefix = "neural_ride_v12_strict_past_z2_aux" if innovation_weight > 0.0 else "neural_ride_v12_z2_aux"
    return replace(
        base,
        name=f"{prefix}_{z2_aux_mode}_research",
        use_z2_aux_head=True,
        z2_aux_head_mode=z2_aux_mode,
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return get_device()
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("--device xpu was requested, but XPU is unavailable.")
    return device


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="GTN → N2P3Net 闭环", allow_abbrev=False)
    ap.add_argument("--subjects", type=int, default=None, help="限缩前 N 名被试")
    ap.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="run only the first N deterministic LOSO folds without shrinking the train pool",
    )
    ap.add_argument(
        "--fold-offset",
        type=int,
        default=0,
        help="skip this many deterministic LOSO folds before applying --max-folds",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=GTN_DEFAULT_DEEP_EPOCHS,
        help="epoch 上限；须覆盖 variance warmup/ramp 和至少一个 joint epoch",
    )
    ap.add_argument(
        "--epoch-trajectory-audit",
        action="store_true",
        help=(
            "development only: save every raw epoch checkpoint and score each on the outer "
            "test fold; diagnostic results are forbidden for checkpoint selection"
        ),
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=NEURAL_RIDE_V12.early_stop_patience,
        help="验证损失连续 N epoch 不改善即停（GLM 协议）",
    )
    ap.add_argument(
        "--val-subject-frac",
        type=float,
        default=0.08,
        help="每 fold 训练被试中留作验证的比例（GLM 协议）",
    )
    ap.add_argument(
        "--audit-subjects",
        type=int,
        default=4,
        help="untouched inner subjects reserved for ERP and prequential structure gates",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAINING_BATCH.physical_batch_size,
        help="physical micro-batch size held by one forward/backward pass",
    )
    ap.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help=(
            "optimizer batch represented by gradient accumulation; defaults to "
            f"{DEFAULT_TRAINING_BATCH.effective_batch_size}"
        ),
    )
    ap.add_argument(
        "--accum-steps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default="eager",
        help="PyTorch model compilation mode; eager is the reproducible fallback",
    )
    ap.add_argument(
        "--fold-jobs",
        type=int,
        default=8,
        help="并行 LOSO fold 数（开发模式默认 8；确认性模式必须为 1）",
    )
    ap.add_argument(
        "--fold-backend",
        choices=("auto", "process", "thread"),
        default="auto",
        help="fold 并发后端；Linux 自动使用独立多进程，thread 仅作兼容回退",
    )
    ap.add_argument("--lr", type=float, default=NEURAL_RIDE_V12.lr)
    ap.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default=NEURAL_RIDE_V12.lr_schedule,
        help="optimizer-step learning-rate schedule",
    )
    ap.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=NEURAL_RIDE_V12.lr_warmup_fraction,
        help="fraction of planned optimizer steps used for linear warmup",
    )
    ap.add_argument(
        "--min-lr-ratio",
        type=float,
        default=NEURAL_RIDE_V12.min_lr_ratio,
        help="cosine schedule floor as a fraction of base LR",
    )
    ap.add_argument("--weight-decay", type=float, default=NEURAL_RIDE_V12.weight_decay)
    ap.add_argument("--lambda2", type=float, default=NEURAL_RIDE_V12.lambda2)
    ap.add_argument("--lambda3", type=float, default=NEURAL_RIDE_V12.lambda3)
    ap.add_argument(
        "--lambda-pcw",
        type=float,
        default=NEURAL_RIDE_V12.lambda_pcw,
        help="Neural-RIDE PCW-only auxiliary BCE weight",
    )
    ap.add_argument(
        "--lambda-digit",
        type=float,
        default=NEURAL_RIDE_V12.lambda_digit,
        help="fixed-K nine-digit set cross-entropy weight",
    )
    ap.add_argument(
        "--lambda-conditional-nll",
        type=float,
        default=NEURAL_RIDE_V12.lambda_conditional_nll,
        help="causal clean/artifact mixture conditional NLL weight",
    )
    ap.add_argument(
        "--repetition-refit-epochs",
        type=int,
        default=NEURAL_RIDE_V12.repetition_refit_epochs,
        help="density-head-only epochs after inner-validation temperature calibration",
    )
    ap.add_argument(
        "--repetition-v12",
        action=argparse.BooleanOptionalAction,
        default=NEURAL_RIDE_V12.repetition_v12,
        help="enable/disable the v12 additive-LLR repetition backbone + fidelity estimator（生产默认 True；legacy 仅历史对照）",
    )
    ap.add_argument(
        "--repetition-state-residual",
        action="store_true",
        default=NEURAL_RIDE_V12.repetition_state_residual,
        help="v12 state residual（默认 gain=0，须经 audit gate 才可非零）",
    )
    ap.add_argument(
        "--repetition-state-residual-l2-weight",
        type=float,
        default=NEURAL_RIDE_V12.repetition_state_residual_l2_weight,
        help="v12 state residual delta^2 shrink weight（blueprint 3.2）",
    )
    ap.add_argument(
        "--measurement-windows",
        action=argparse.BooleanOptionalAction,
        default=NEURAL_RIDE_V12.use_measurement_windows,
        help="enable/disable object L: fold-local latency posterior + gated detached PCW consumer",
    )
    ap.add_argument(
        "--measurement-anchor-ms",
        type=float,
        default=NEURAL_RIDE_V12.measurement_anchor_ms,
    )
    ap.add_argument(
        "--measurement-grid-radius-ms",
        type=float,
        default=NEURAL_RIDE_V12.measurement_grid_radius_ms,
    )
    ap.add_argument(
        "--measurement-grid-step-ms",
        type=float,
        default=NEURAL_RIDE_V12.measurement_grid_step_ms,
    )
    ap.add_argument(
        "--measurement-window-width-ms",
        type=float,
        default=NEURAL_RIDE_V12.measurement_window_width_ms,
    )
    ap.add_argument(
        "--measurement-refit-epochs",
        type=int,
        default=NEURAL_RIDE_V12.measurement_refit_epochs,
    )
    ap.add_argument(
        "--digit-evidence-ks",
        default=",".join(str(k) for k in NEURAL_RIDE_V12.digit_evidence_ks),
        help="online acquisition checkpoints for the nested repetition objective",
    )
    ap.add_argument(
        "--digit-evidence-weights",
        default=",".join(str(weight) for weight in NEURAL_RIDE_V12.digit_evidence_weights),
        help="comma-separated weights aligned with --digit-evidence-ks",
    )
    ap.add_argument("--lambda-amp", type=float, default=NEURAL_RIDE_V12.lambda_amp)
    ap.add_argument(
        "--lambda-recon",
        type=float,
        default=NEURAL_RIDE_V12.lambda_recon,
        help="gate-aligned ERP class-contrast reconstruction weight",
    )
    ap.add_argument(
        "--lambda-innovation",
        type=float,
        default=NEURAL_RIDE_V12.lambda_innovation,
        help="strict-past prequential observation NLL weight",
    )
    ap.add_argument(
        "--innovation-ar-order",
        type=int,
        default=NEURAL_RIDE_V12.innovation_ar_order,
        help="fold-local ridge VAR order used as the causal likelihood baseline",
    )
    ap.add_argument(
        "--lambda-morphology-l0",
        type=float,
        default=NEURAL_RIDE_V12.lambda_morphology_l0,
        help="Hard-Concrete optional morphology atom sparsity weight",
    )
    ap.add_argument(
        "--variance-warmup-epochs",
        type=int,
        default=None,
        help="mean-only epochs before faithful variance training; defaults to the active recipe",
    )
    ap.add_argument(
        "--variance-ramp-epochs",
        type=int,
        default=None,
        help="linear ramp duration; defaults to the active recipe",
    )
    ap.add_argument(
        "--recon-bootstrap-samples",
        type=int,
        default=NEURAL_RIDE_V12.recon_bootstrap_samples,
        help="class-stratified fold-local ERP target bootstrap replicates",
    )
    ap.add_argument(
        "--recon-split-half-repeats",
        type=int,
        default=NEURAL_RIDE_V12.recon_split_half_repeats,
        help="fold-local averaged-ERP reliability audit replicates",
    )
    ap.add_argument(
        "--lambda-jit", type=float, default=0.0, help="自监督 jitter 一致性默认关闭（方案 B）"
    )
    ap.add_argument("--jit-prob", type=float, default=0.0)
    ap.add_argument("--jit-max-ms", type=float, default=40.0)
    ap.add_argument(
        "--encoder-depth",
        type=int,
        default=NEURAL_RIDE_V12.encoder_depth,
        help="Stage 2 TCN depth；TCN dilation 由该 dep 自动生成，正式默认 4",
    )
    ap.add_argument(
        "--encoder-bn-momentum",
        type=float,
        default=NEURAL_RIDE_V12.encoder_bn_momentum,
        help="TCN BatchNorm EMA momentum calibrated for the physical batch",
    )
    ap.add_argument(
        "--disable-bn-recalibration",
        action="store_true",
        help="skip the final optimization-fold BatchNorm running-stat pass",
    )
    ap.add_argument(
        "--encoder-type", default=NEURAL_RIDE_V12.encoder_type, choices=("tcn", "conformer")
    )
    ap.add_argument(
        "--encoder-norm",
        default=NEURAL_RIDE_V12.encoder_norm,
        choices=("ln", "bn"),
        help="TCN block 归一化（GLM 消融轴）。默认 bn：三组实测（12/60 被试，"
        "含/不含再参考）BN 一致优于 LN +0.5~0.9pt AUC；ln=旧默认回退",
    )
    ap.add_argument(
        "--tcn-pointwise-execution",
        choices=("conv1d", "linear"),
        default=NEURAL_RIDE_V12.tcn_pointwise_execution,
        help="TCN 1x1 mixing API; linear folds B*T and preserves checkpoint layout",
    )
    ap.add_argument(
        "--tokenizer-init",
        default=NEURAL_RIDE_V12.tokenizer_init,
        choices=("random", "bandpass"),
        help="GLM v3：时间卷积初始化。默认 bandpass（2026-08-23 定案）：Gabor 带通，"
        "诊断证据=随机 init 的 FIR 从未学出 ERP 形状（~60Hz 白噪不动），修复后 "
        "60 被 AUC +1.55pt、242 被 hit 与 EEGNet 完全打平（203/242 vs 203/242，"
        "McNemar p=1.0）；random=kaiming 旧默认回退。核长分层分配频带，"
        "k=129 占据 P3b δ-θ 带 [1.5,7]Hz",
    )
    ap.add_argument(
        "--disable-tokenizer-fusion",
        action="store_true",
        help="use the legacy temporal Conv1d plus spatial einsum path for scheduling A/B",
    )
    ap.add_argument(
        "--tokenizer-post-norm",
        default=NEURAL_RIDE_V12.tokenizer_post_norm,
        choices=("none", "bn"),
        help="GLM v3：每尺度时间卷积后 BatchNorm1d（EEG-Inception/ATCNet 标准 "
        "结构；修 4× 尺度幅值失衡 + 提供非线性位点，防多尺度线性塌缩）",
    )
    ap.add_argument(
        "--tokenizer-post-act",
        default=NEURAL_RIDE_V12.tokenizer_post_act,
        choices=("none", "elu", "gelu"),
        help="GLM v3：时间卷积后激活（ELU 保负电位，EEG 文献论点）。默认 none=旧行为",
    )
    ap.add_argument(
        "--model-size",
        default="default",
        choices=("default", "mini", "mini_a"),
        help="capacity preset；精确参数量按实际启用模块计算并写入 run record",
    )
    ap.add_argument(
        "--rereference",
        dest="use_rereference",
        action=argparse.BooleanOptionalAction,
        default=NEURAL_RIDE_V12.use_rereference,
        help="GLM v2：门控参考层（gate init=0 → 恒等起步，训练自证开合），"
        "默认开启。60 被实测：开启 hit .9000（全系最高）vs 关闭 .8833，"
        "AUC 0.7536 vs 0.7575（噪声带内等价）；60 被数据量下网络自学到"
        " gate≈[0.20,0.13,0.15]（轻度再参考去共模）。Phase 3 跨数据集"
        "时按域学参考变换（鼻参考 GTN ↔ 平均参考 ERP CORE ↔ A1 耳参考"
        "自采 8 导）。--no-rereference 关闭。旧版「强制 CAR」已废弃"
        "（<32 导无效，Junghöfer 2001/Luck 2014；GTN 实测销毁 P3b 4.36×）",
    )
    ap.add_argument(
        "--p3b-sigma-hi",
        type=float,
        default=GTN_CHILD_SIGMA_BOUNDS[2][1],
        help=(
            "P3b σ 上界 ms（GTN 儿童宽 P3b 专用默认 150；成人数据建议显式传 "
            "PCW_CANONICAL_SIGMA_BOUNDS 的 P3b 上界 80）"
        ),
    )
    ap.add_argument(
        "--dtau-readout",
        default="attention_softargmax",
        choices=("attention_softargmax", "global_pool", "maxmean", "attention"),
        help="Δτ 读出路径；正式 recipe 固定 attention_softargmax，其余仅作 claim-gate 研究对照",
    )
    ap.add_argument(
        "--p3b-tau0-ms",
        type=float,
        default=GTN_CHILD_TAU0_MS[2],
        help="P3b 先验中心（GTN 儿童数据实测峰值 460–490ms，成人仍可用 350）",
    )
    ap.add_argument(
        "--p3b-tau0-lo",
        type=float,
        default=GTN_CHILD_TAU0_BOUNDS[2][0],
        help="P3b τ0 生理界下界（ms）",
    )
    ap.add_argument(
        "--p3b-tau0-hi",
        type=float,
        default=GTN_CHILD_TAU0_BOUNDS[2][1],
        help="P3b τ0 生理界上界（ms）",
    )
    ap.add_argument(
        "--z2-aux-head",
        choices=("off", "add", "replace"),
        default="off",
        help=(
            "research-only full-Z2 auxiliary trial head（E5 claim-gate 对照）："
            "off=生产默认 PCW-only；add=logit_pcw+head_z2(Z2)；"
            "replace=logit_target=head_z2(Z2)，PCW 仍作为 side readout 由 lambda_pcw 训练"
        ),
    )
    ap.add_argument(
        "--z2-aux-pool",
        choices=Z2_AUX_POOLS,
        default="attention",
        help="full-Z2 auxiliary head 的时间池化方式（仅 --z2-aux-head != off 时生效）",
    )
    ap.add_argument(
        "--erp-calibration",
        default="fixed",
        choices=("fixed", "fold"),
        help="fixed=development 内置或独立冻结 prior；fold=每外层折仅用内部训练被试重新校准",
    )
    ap.add_argument(
        "--frozen-erp-prior",
        default=None,
        help="独立开发集冻结先验 JSON；须声明 calibration_scope=independent_development",
    )
    ap.add_argument("--head-dropout", type=float, default=NEURAL_RIDE_V12.head_dropout)
    ap.add_argument("--encoder-dropout", type=float, default=NEURAL_RIDE_V12.encoder_dropout)
    ap.add_argument(
        "--no-spatial-max-norm",
        action="store_true",
        help="关闭 tokenizer 空间权重 max-norm=1（回退）",
    )
    ap.add_argument("--augment", action="store_true")
    ap.add_argument(
        "--benchmark", action="store_true", help="只训第一个 fold 并报告时间/显存，不跑全量评估"
    )
    ap.add_argument("--cache-dir", default="experiments/cache")
    ap.add_argument(
        "--epoch-tmax-ms",
        type=float,
        default=1200.0,
        help="exclusive physiological context right edge; GTN default +1200 ms",
    )
    ap.add_argument(
        "--save-scores-dir", default=None, help="逐被试 scores JSON 目录；缺省使用 run-dir"
    )
    ap.add_argument("--run-dir", default="experiments/runs")
    ap.add_argument("--run-name", default=None, help="实验名；缺省 n2p3net_gtn_<UTC时间戳>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "xpu", "cpu"),
        default="auto",
        help="训练设备；锁定多种子运行显式要求 CUDA，开发 smoke 可使用 CPU。",
    )
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
        default=str(_ROOT / "experiments" / "protocols" / "gtn_confirmatory_cohort_v1.json"),
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--primary-decision",
        default="prefix_minK_chain_llr@3",
        choices=tuple(
            [
                f"{semantics}_{aggregation}@{k}"
                for semantics in ("exact", "prefix_minK")
                for aggregation in ("sum", "mean", "llr")
                for k in ("1", "3", "5", "10", "15")
            ]
            + [f"prefix_minK_chain_llr@{k}" for k in ("1", "3", "5", "10", "15")]
            + [f"all_{aggregation}" for aggregation in ("sum", "mean", "llr", "chain_llr")]
            + [
                f"flash_{aggregation}@{n}"
                for aggregation in ("sum", "mean", "llr")
                for n in ("9", "27", "45", "90", "135")
            ]
        ),
        help="预注册的主要 GTN 被试级指标；K=3 覆盖率至少 0.90，K=15 仅作次要指标",
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
    args = ap.parse_args()
    try:
        batch_config = TrainingBatchConfig.from_cli(
            physical_batch_size=args.batch_size,
            effective_batch_size=args.effective_batch_size,
            accum_steps=args.accum_steps,
            default=DEFAULT_TRAINING_BATCH,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if args.fold_jobs < 1:
        ap.error("--fold-jobs must be positive")
    if args.compile_mode != "eager" and args.fold_jobs != 1:
        ap.error("compiled runs require --fold-jobs 1 to avoid compiler/cache contention")
    postprocess_cpu_threads = _postprocess_cpu_threads()
    parent_cpu_scheduler = _configure_parent_cpu_scheduler(postprocess_cpu_threads)
    print(
        "[cpu scheduler] parent "
        f"intraop={parent_cpu_scheduler['intraop_threads']} "
        f"interop={parent_cpu_scheduler['interop_threads']}; "
        f"fold_workers={args.fold_jobs}x{os.environ.get('FOLD_CPU_THREADS', '2')}",
        flush=True,
    )
    if not 0.0 < args.primary_min_coverage <= 1.0:
        ap.error("--primary-min-coverage must be in (0,1]")
    if not 0.0 < args.efficiency_min_coverage <= 1.0:
        ap.error("--efficiency-min-coverage must be in (0,1]")
    if args.evaluation_mode == "confirmatory" and args.epoch_trajectory_audit:
        ap.error("--epoch-trajectory-audit is development-only and forbidden in confirmatory mode")
    if args.frozen_erp_prior:
        frozen_prior_path = Path(args.frozen_erp_prior).expanduser().resolve()
        if not frozen_prior_path.is_file():
            ap.error(f"--frozen-erp-prior 文件不存在：{frozen_prior_path}")
    else:
        frozen_prior_path = None
    device = _resolve_device(args.device)
    fold_backend = args.fold_backend
    if fold_backend == "auto":
        fold_backend = "process" if os.name == "posix" else "thread"
    if (
        args.evaluation_mode == "confirmatory"
        and args.erp_calibration == "fixed"
        and args.frozen_erp_prior is None
    ):
        raise RuntimeError(
            "Confirmatory fixed ERP calibration requires --frozen-erp-prior from an "
            "independent development dataset. Use --erp-calibration fold for train-side calibration."
        )
    if args.evaluation_mode == "confirmatory" and device.type != "cuda":
        raise RuntimeError(
            "Confirmatory N2P3-Net evaluation requires CUDA; CPU fallback is forbidden."
        )
    if args.evaluation_mode == "confirmatory" and args.fold_jobs != 1:
        raise RuntimeError(
            "Confirmatory N2P3-Net evaluation requires --fold-jobs 1 because concurrent "
            "folds share the process-wide CUDA RNG."
        )
    digit_evidence_ks = tuple(int(v) for v in args.digit_evidence_ks.split(",") if v.strip())
    digit_evidence_weights = tuple(
        float(v) for v in args.digit_evidence_weights.split(",") if v.strip()
    )
    if (
        not digit_evidence_ks
        or tuple(sorted(set(digit_evidence_ks))) != digit_evidence_ks
        or len(digit_evidence_ks) != len(digit_evidence_weights)
    ):
        ap.error("digit evidence Ks must be ordered/unique and match their weights")
    if max(digit_evidence_ks) not in (5, 15):
        ap.error(
            "digit_evidence_ks must cover the v12 main development horizon K=5 "
            "(default 1,3,5) or the locked K=15 protocol (1,3,5,10,15)"
        )
    if "chain_llr" in args.primary_decision and (
        args.lambda_conditional_nll <= 0.0 or args.repetition_refit_epochs <= 0
    ):
        ap.error(
            "a chain_llr primary requires --lambda-conditional-nll > 0 and "
            "--repetition-refit-epochs > 0; use sum/mean/llr for a trial-only ablation"
        )
    active_recipe = _recipe_for_research_route(args.lambda_innovation, args.z2_aux_head)
    if args.variance_warmup_epochs is None:
        args.variance_warmup_epochs = active_recipe.variance_warmup_epochs
    if args.variance_ramp_epochs is None:
        args.variance_ramp_epochs = active_recipe.variance_ramp_epochs
    # v5（2026-08-22）：--subjects 限缩也从全量缓存派生（旧的 n<N> 子集缓存由旧口径生成，
    # 与全量 242 口径的被试/试次不一致，会悄悄改变分母；统一用 nall 后再 np.unique 截取）。
    cache_dir = Path(args.cache_dir)
    epoch_tmax_s = args.epoch_tmax_ms / 1000.0
    cache_path = cache_dir / _gtn_cache_filename(256.0, 0.1, -0.2, epoch_tmax_s, "all")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"未找到 GTN 全量缓存 {cache_path}；请先运行 "
            "experiments/run_gtn_baseline.py --epoch-tmax "
            f"{epoch_tmax_s:g} --prepare-cache-only 生成缓存。"
        )
    X3, y, digits, subject_ids, true_digits, skipped, event_timeline = _load_gtn_cache(cache_path)
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if args.dataset_sha256 is not None and cache_sha256 != args.dataset_sha256:
        raise RuntimeError(
            f"Frozen dataset hash mismatch: expected {args.dataset_sha256}, got {cache_sha256}."
        )
    if args.evaluation_mode == "confirmatory":
        if args.subjects is not None or args.fold_offset or args.max_folds is not None:
            raise RuntimeError("Confirmatory evaluation requires the complete locked LOSO cohort.")
        if args.benchmark or not args.protocol_sha256 or not args.dataset_sha256:
            raise RuntimeError("Confirmatory runner requires frozen hashes and forbids benchmark.")
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
            or source_tree_sha256(_ROOT) != args.source_sha256
            or not args.runtime_sha256
            or runtime_environment_sha256() != args.runtime_sha256
        ):
            raise RuntimeError("Confirmatory source or dependency runtime identity mismatch.")
        if external_assets_sha256(
            {
                "frozen_erp_prior": args.frozen_erp_prior,
                "pretrained_checkpoint": None,
                "pretrained_mapping": None,
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
            runner="n2p3net",
            model="n2p3net",
        )
        claim_confirmatory_seed(
            args.confirmatory_lock,
            seed=args.seed,
            run_identity={"runner": "n2p3net", "model": "n2p3net"},
        )
        confirmatory_id = str(lock_payload["confirmatory_id"])
    else:
        confirmatory_id = None
        confirmatory_lock_sha256 = None
        evaluation_units = tuple(sorted(true_digits))

    if args.subjects:
        keep_subj = np.unique(subject_ids)[: args.subjects]
        keep = np.isin(subject_ids, keep_subj)
        X3, y, digits, subject_ids = X3[keep], y[keep], digits[keep], subject_ids[keep]
        true_digits = {k: v for k, v in true_digits.items() if k in set(keep_subj.tolist())}
        event_timeline = event_timeline.subset_groups(set(keep_subj.astype(str).tolist()))

    X_model, E_chn, channel_mask = _build_gtn_x_and_identity(X3)
    # GTN uses the native Fz/Cz/Pz layout for every trial: a dense all-true
    # per-trial channel mask is the explicit contract required by the
    # capability-based evaluation adapter.
    trial_channel_mask = np.ones(X_model.shape[:2], dtype=bool)
    print(f"[n2p3net] X={X_model.shape} y={y.shape} subjects={len(np.unique(subject_ids))}")

    run_name = args.run_name or datetime.now(UTC).strftime("n2p3net_gtn_%Y%m%d_%H%M%SZ")
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    existing_progress: list[dict[str, object]] = []
    if progress_path.is_file():
        for raw_line in progress_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                existing_progress.append(row)
    existing_fold_ids = [
        int(row["fold"])
        for row in existing_progress
        if row.get("type") == "fold" and isinstance(row.get("fold"), int)
    ]
    existing_fold_count = len(set(existing_fold_ids))
    display_fold_offset = max(existing_fold_ids, default=-1) + 1
    batch_index = sum(1 for row in existing_progress if row.get("type") == "manifest")
    previous_record: dict[str, object] | None = None
    record_path = run_dir / "record.json"
    if record_path.is_file():
        try:
            loaded_record = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(loaded_record, dict):
                previous_record = loaded_record
        except (OSError, json.JSONDecodeError):
            previous_record = None
    epoch_progress_dir = (run_dir / "epochs").resolve()
    epoch_progress_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = Path(args.save_scores_dir) if args.save_scores_dir else run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "schema": "n2p3net_gtn_run/2",
        "run_name": run_name,
        "started_utc": datetime.now(UTC).isoformat(),
        "mode": "benchmark" if args.benchmark else "evaluate",
        "args": vars(args),
        "data": {
            "cache_path": str(cache_path),
            "cache_sha256": cache_sha256,
            "X3_shape": list(X3.shape),
            "model_input_shape": list(X_model.shape),
            "n_trials": int(X3.shape[0]),
            "n_subjects": int(len(np.unique(subject_ids))),
            "target_rate": float(y.mean()),
            "trials_per_subject": float(X3.shape[0] / max(len(np.unique(subject_ids)), 1)),
            "skipped": [str(s) for s in skipped],
        },
    }

    model_channels = tuple(GTN_STANDARD)
    model_kwargs = active_recipe.model_kwargs(
        n_channels=3,
        channel_names=model_channels,
        tmin_ms=-200.0,
        tmax_ms=args.epoch_tmax_ms,
        sfreq=256.0,
        n_time=X_model.shape[2],
        baseline_mode="trial",
        tau0_ms=(GTN_CHILD_TAU0_MS[0], GTN_CHILD_TAU0_MS[1], args.p3b_tau0_ms),
        tau0_bounds=(
            GTN_CHILD_TAU0_BOUNDS[0],
            GTN_CHILD_TAU0_BOUNDS[1],
            (args.p3b_tau0_lo, args.p3b_tau0_hi),
        ),
        sigma_bounds=(
            GTN_CHILD_SIGMA_BOUNDS[0],
            GTN_CHILD_SIGMA_BOUNDS[1],
            (GTN_CHILD_SIGMA_BOUNDS[2][0], args.p3b_sigma_hi),
        ),
        dtau_bounds=GTN_CHILD_DTAU_BOUNDS,
        overrides={
            "spatial_max_norm": None if args.no_spatial_max_norm else 1.0,
            "encoder_dropout": args.encoder_dropout,
            "head_dropout": args.head_dropout,
            "component_decoder": (args.lambda_recon > 0.0 or args.lambda_morphology_l0 > 0.0),
            "use_innovation_likelihood": args.lambda_innovation > 0.0,
            "dtau_readout": args.dtau_readout,
            "encoder_depth": args.encoder_depth,
            "encoder_type": args.encoder_type,
            "encoder_norm": args.encoder_norm,
            "encoder_bn_momentum": args.encoder_bn_momentum,
            "tokenizer_init": args.tokenizer_init,
            "tokenizer_post_norm": args.tokenizer_post_norm,
            "tokenizer_post_act": args.tokenizer_post_act,
            "tcn_pointwise_execution": args.tcn_pointwise_execution,
            "use_rereference": args.use_rereference,
            "repetition_v12": args.repetition_v12,
            "repetition_state_residual": args.repetition_state_residual,
            "use_measurement_windows": args.measurement_windows,
            "measurement_anchor_ms": args.measurement_anchor_ms,
            "measurement_grid_radius_ms": args.measurement_grid_radius_ms,
            "measurement_grid_step_ms": args.measurement_grid_step_ms,
            "measurement_window_width_ms": args.measurement_window_width_ms,
            "measurement_refit_epochs": args.measurement_refit_epochs,
            "use_z2_aux_head": args.z2_aux_head != "off",
            "z2_aux_head_mode": args.z2_aux_head if args.z2_aux_head != "off" else "add",
            "z2_aux_pool": args.z2_aux_pool,
        },
    )
    erp_prior_record = {
        "mode": args.erp_calibration,
        "source": (
            "independent_development_json"
            if frozen_prior_path is not None
            else ("builtin_development" if args.erp_calibration == "fixed" else "outer_train_subjects")
        ),
        "calibration_scope": (
            "independent_development"
            if frozen_prior_path is not None
            else ("development" if args.erp_calibration == "fixed" else "outer_train_inner_validation")
        ),
    }
    if frozen_prior_path is not None:
        calib = json.loads(frozen_prior_path.read_text(encoding="utf-8"))
        if calib.get("calibration_scope") != "independent_development":
            ap.error("--frozen-erp-prior 必须声明 calibration_scope=independent_development。")
        if args.erp_calibration == "fold":
            ap.error("--erp-calibration fold 与 --frozen-erp-prior 互斥。")
        required_prior_keys = ("tau0_ms", "tau0_bounds", "sigma_bounds")
        missing_prior_keys = [key for key in required_prior_keys if key not in calib]
        if missing_prior_keys:
            ap.error(f"--frozen-erp-prior 缺少字段：{missing_prior_keys}")
        if not isinstance(calib.get("dataset"), dict):
            ap.error("--frozen-erp-prior 必须包含独立开发数据集的 dataset record。")
        model_kwargs.update(
            tau0_ms=tuple(float(v) for v in calib["tau0_ms"]),
            tau0_bounds=tuple(tuple(float(x) for x in b) for b in calib["tau0_bounds"]),
            sigma_bounds=tuple(tuple(float(x) for x in b) for b in calib["sigma_bounds"]),
            dtau_bounds=tuple(
                tuple(float(x) for x in b)
                for b in calib.get("dtau_bounds", GTN_CHILD_DTAU_BOUNDS)
            ),
        )
        erp_prior_record.update(
            {
                "path": str(frozen_prior_path),
                "sha256": hashlib.sha256(frozen_prior_path.read_bytes()).hexdigest(),
                "source_dataset": calib["dataset"].get("name"),
            }
        )
        print(
            f"[frozen-erp-prior] {frozen_prior_path}: "
            f"tau0_ms={[round(v) for v in calib['tau0_ms']]}",
            flush=True,
        )
    erp_prior_record["resolved"] = {
        "tau0_ms": model_kwargs["tau0_ms"],
        "tau0_bounds": model_kwargs["tau0_bounds"],
        "sigma_bounds": model_kwargs["sigma_bounds"],
        "dtau_bounds": model_kwargs["dtau_bounds"],
    }
    record["erp_prior"] = erp_prior_record
    if args.disable_tokenizer_fusion:
        model_kwargs["tokenizer_temporal_spatial_fusion"] = False
    # Capacity ablations scale both the PCW and independent likelihood paths.
    if args.model_size == "mini":
        model_kwargs.update(
            d_model=32,
            filters_per_scale=4,
            temporal_kernels=(33, 65),
            encoder_depth=1,
            innovation_d_model=16,
        )
    elif args.model_size == "mini_a":
        model_kwargs.update(
            d_model=16,
            filters_per_scale=2,
            temporal_kernels=(65,),
            encoder_depth=0,
            innovation_d_model=8,
        )
    model_parameter_count = N2P3Net(**model_kwargs).num_parameters()
    print(f"[model] parameters={model_parameter_count:,}", flush=True)
    trainer_config = active_recipe.trainer_config(
        GTN_DIGIT_TASK,
        epochs=args.epochs,
        seed=args.seed,
        batch_config=batch_config,
        overrides={
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_fraction": args.lr_warmup_fraction,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "lambda2": args.lambda2,
            "lambda3": args.lambda3,
            "lambda_pcw": args.lambda_pcw,
            "lambda_digit": args.lambda_digit,
            "lambda_conditional_nll": args.lambda_conditional_nll,
            "repetition_refit_epochs": args.repetition_refit_epochs,
            "repetition_v12": args.repetition_v12,
            "repetition_state_residual_l2_weight": args.repetition_state_residual_l2_weight,
            "digit_evidence_ks": digit_evidence_ks,
            "digit_evidence_weights": digit_evidence_weights,
            "lambda_amp": args.lambda_amp,
            "lambda_recon": args.lambda_recon,
            "lambda_innovation": args.lambda_innovation,
            "innovation_ar_order": args.innovation_ar_order,
            "lambda_morphology_l0": args.lambda_morphology_l0,
            "variance_warmup_epochs": args.variance_warmup_epochs,
            "variance_ramp_epochs": args.variance_ramp_epochs,
            "recon_bootstrap_samples": args.recon_bootstrap_samples,
            "recon_split_half_repeats": args.recon_split_half_repeats,
            "lambda_jit": args.lambda_jit,
            "compile_mode": args.compile_mode,
            "jit_prob": args.jit_prob,
            "jit_max_ms": args.jit_max_ms,
            "augment": args.augment,
            "early_stop_patience": args.early_stop_patience,
            "epoch_trajectory_audit": args.epoch_trajectory_audit,
            "recalibrate_batch_norm": not args.disable_bn_recalibration,
        },
    )
    trainer_kwargs = asdict(trainer_config)
    print(
        f"[perf] GTN physical_batch={batch_config.physical_batch_size} "
        f"accum_steps={batch_config.accumulation_steps} "
        f"effective_batch={batch_config.effective_batch_size} fold_jobs={args.fold_jobs} "
        f"fold_backend={fold_backend} "
        f"compile={args.compile_mode} "
        f"amp={'bf16' if device.type in ('cuda', 'xpu') else 'off'} "
        f"tf32={'on' if device.type == 'cuda' else 'off'}",
        flush=True,
    )
    val_subject_frac = args.val_subject_frac
    erp_calibrator = (
        FoldERPCalibrator(
            EpochTimeAxis(-200.0, args.epoch_tmax_ms, 256.0, X_model.shape[2]),
            tuple(GTN_STANDARD),
            sigma_bounds=(
                GTN_CHILD_SIGMA_BOUNDS[0],
                GTN_CHILD_SIGMA_BOUNDS[1],
                (GTN_CHILD_SIGMA_BOUNDS[2][0], args.p3b_sigma_hi),
            ),
        )
        if args.erp_calibration == "fold"
        else None
    )

    record["model_kwargs"] = model_kwargs
    record["model_parameter_count"] = model_parameter_count
    record["batch_config"] = batch_config.record()
    record["trainer_kwargs"] = trainer_kwargs
    record["recipe"] = active_recipe.record(GTN_DIGIT_TASK, trainer_config)
    record["environment"] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        "device": str(device),
        "performance": {
            "physical_batch_size": batch_config.physical_batch_size,
            "effective_batch_size": batch_config.effective_batch_size,
            "accumulation_steps": batch_config.accumulation_steps,
            "fold_jobs": int(args.fold_jobs),
            "fold_backend": fold_backend,
            "amp_dtype": "bfloat16" if device.type in ("cuda", "xpu") else None,
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
                if hasattr(torch, "get_float32_matmul_precision")
                else None
            ),
            "tf32_matmul": bool(
                getattr(getattr(torch.backends, "cuda", None), "matmul", None)
                and torch.backends.cuda.matmul.allow_tf32
            )
            if device.type == "cuda"
            else False,
            "fused_adamw": device.type == "cuda",
            "compile_mode": args.compile_mode,
            "tokenizer_temporal_spatial_fusion_requested": not args.disable_tokenizer_fusion,
            "tcn_pointwise_execution": args.tcn_pointwise_execution,
        },
    }
    adapter = N2P3NetBaseline(
        model_kwargs=model_kwargs,
        trainer_kwargs=trainer_kwargs,
        E_chn=E_chn,
        channel_mask=channel_mask,
        val_subject_frac=val_subject_frac,
        audit_subjects=args.audit_subjects,
        erp_calibrator=erp_calibrator,
        device=device,
    )
    adapter.configure_epoch_progress(epoch_progress_dir)

    folds = loso_folds(subject_ids)
    if args.evaluation_mode == "confirmatory":
        allowed = set(evaluation_units)
        folds = [
            (train_mask, test_mask)
            for train_mask, test_mask in folds
            if str(subject_ids[test_mask][0]) in allowed
        ]
    if args.fold_offset < 0:
        raise ValueError("--fold-offset must be non-negative.")
    folds = folds[args.fold_offset :]
    if args.max_folds is not None:
        if args.max_folds < 1:
            raise ValueError("--max-folds must be positive.")
        folds = folds[: args.max_folds]
    wall_t0 = time.perf_counter()

    # Append batches when a queue reuses one logical run name.
    progress_f = progress_path.open("a" if existing_progress else "w", encoding="utf-8")
    completed_folds = 0

    def _write_progress(fold_idx: int, fold_result, records) -> None:
        nonlocal completed_folds
        completed_folds += 1
        # records: list[(predicted, true, subject)]（见 evaluate._evaluate_one_fold）
        hits = [1 if p == t else 0 for p, t, _ in records] if records else []
        train_losses = list(getattr(fold_result, "train_losses", ()) or ())
        val_losses = list(getattr(fold_result, "val_losses", ()) or ())
        val_innovation_nlls = list(
            getattr(fold_result, "val_innovation_nlls", ()) or ()
        )
        task_val_aucs = list(getattr(fold_result, "task_val_aucs", ()) or ())
        line = {
            "type": "fold",
            "fold": display_fold_offset + fold_idx,
            "batch_fold": fold_idx,
            "batch_index": batch_index,
            "n_folds_done": existing_fold_count + completed_folds,
            "subject": str(records[0][2]) if records else None,
            "hit": hits[0] if hits else None,
            "fold_bacc": float(fold_result.balanced_acc),
            "fold_auc": float(fold_result.auc),
            "fit_sec": fold_result.fit_sec,
            "fit_peak_memory_mb": fold_result.fit_peak_memory_mb,
            "epochs_ran": fold_result.epochs_ran,
            "train_losses": [round(float(v), 4) for v in train_losses][-12:],
            "val_losses": [round(float(v), 4) for v in val_losses][-12:],
            "val_innovation_nlls": [
                round(float(v), 6) for v in val_innovation_nlls
            ][-12:],
            "task_val_aucs": [
                None if v is None else round(float(v), 6) for v in task_val_aucs
            ][-12:],
            "final_task_val_auc": fold_result.final_task_val_auc,
            "best_task_epoch": fold_result.best_task_epoch,
            "best_density_epoch": fold_result.best_density_epoch,
            "best_task_val_loss": fold_result.best_task_val_loss,
            "best_density_nll": fold_result.best_density_nll,
            "task_patience_exhausted": fold_result.task_patience_exhausted,
            "epoch_trajectory_audit": fold_result.epoch_trajectory_audit,
            "neural_ride_audit": fold_result.audit,
            "ts": datetime.now(UTC).isoformat(),
        }
        progress_f.write(json.dumps(line, ensure_ascii=False) + "\n")
        progress_f.flush()

    progress_f.write(
        json.dumps(
            {
                "type": "manifest",
                "run_name": run_name,
                "total_folds": display_fold_offset + len(folds),
                "batch_total_folds": len(folds),
                "batch_index": batch_index,
                "batch_fold_offset": display_fold_offset,
                "folds_completed_before": existing_fold_count,
                "n_trials": int(len(X_model)),
                "model_kwargs": {k: str(v) for k, v in model_kwargs.items()},
                "model_parameter_count": model_parameter_count,
                "trainer_kwargs": trainer_kwargs,
                "epoch_progress": "epochs/fold_<fold>.jsonl",
                "started_utc": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    progress_f.flush()

    if args.benchmark:
        train_mask, test_mask = folds[0]
        Xtr, ytr = X_model[train_mask], y[train_mask]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Benchmark bypasses evaluate(), so apply the same per-fold CPU budget
        # that _run_fold_core / serial evaluate now enforce.
        with _fold_threadpool_limits():
            if getattr(adapter, "fit_accepts_trial_context", False):
                adapter.fit(
                    Xtr,
                    ytr,
                    subject_ids=subject_ids[train_mask],
                    digits=digits[train_mask],
                )
            elif getattr(adapter, "fit_accepts_subject_ids", False):
                adapter.fit(Xtr, ytr, subject_ids=subject_ids[train_mask])
            else:
                adapter.fit(Xtr, ytr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"[benchmark] fit {elapsed:.2f}s for {len(Xtr)} trials × {args.epochs} epochs")
        print(f"[benchmark] seconds/epoch ≈ {elapsed / max(args.epochs, 1):.3f}")
        peak_mb = None
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"[benchmark] GPU peak memory ≈ {peak_mb:.1f} MB")
        record["benchmark"] = {
            "n_train_trials": int(len(Xtr)),
            "epochs": args.epochs,
            "wall_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(args.epochs, 1),
            "gpu_peak_memory_mb": peak_mb,
            "fit_durations_sec": adapter.fit_durations,
            "fit_peak_memory_mb": adapter.fit_peak_memory_mb,
            "training_history": adapter.last_history,
        }
        record["finished_utc"] = datetime.now(UTC).isoformat()
        record_path = run_dir / "record.json"
        record_tmp_path = run_dir / f".record.json.{os.getpid()}.tmp"
        record_tmp_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        record_tmp_path.replace(record_path)
        print(f"[record] {record_path}")
        progress_f.close()
        return

    summary = evaluate(
        adapter,
        X_model,
        y,
        digits,
        subject_ids,
        true_digits,
        folds,
        n_jobs=args.fold_jobs,
        fold_id_offset=display_fold_offset,
        primary_decision_metric=args.primary_decision,
        fixed_error_rate=args.fixed_error_rate,
        primary_min_coverage=args.primary_min_coverage,
        efficiency_min_coverage=args.efficiency_min_coverage,
        repetition_duration_s=args.repetition_duration_s,
        flash_budgets=(9, 27, 45, 90, 135),
        event_timeline=event_timeline,
        trial_channel_mask=trial_channel_mask,
        evaluation_units=(
            tuple(
                sorted(
                    {
                        str(subject)
                        for _, test_mask in folds
                        for subject in np.unique(subject_ids[test_mask])
                    }
                )
            )
            if args.fold_offset or args.max_folds is not None
            else evaluation_units
        ),
        fold_protocol=(
            "partial_loso"
            if args.fold_offset or args.max_folds is not None or args.evaluation_mode == "confirmatory"
            else "loso"
        ),
        dataset_sha256=cache_sha256,
        on_fold_end=_write_progress,
        parallel_backend=fold_backend,
    )
    progress_f.write(
        json.dumps(
            {
                "type": "done",
                "batch_index": batch_index,
                "total_folds": display_fold_offset + len(folds),
                "primary_metric_gate": summary.primary_metric_gate,
                "ts": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    progress_f.close()

    # Process workers return compact final-fold artifacts with FoldResult, so
    # post-processing never needs to fit the final fold again in the parent.
    last_fold = summary.per_fold[-1]
    print(
        f"[finalize] cpu_threads={postprocess_cpu_threads} "
        "for post-processing; using final-fold worker artifacts",
        flush=True,
    )
    wall_seconds = time.perf_counter() - wall_t0
    primary_metric = summary.decision_metrics[args.primary_decision]
    print(
        f"[result] N2P3Net: primary {args.primary_decision}="
        f"{primary_metric.hit_rate:.4f} coverage={primary_metric.n_covered}/"
        f"{primary_metric.n_total}; all-trial sum={summary.hit_rate_mean:.4f} "
        f"(±{summary.hit_rate_std:.4f}) bacc={summary.balanced_acc_mean:.4f} "
        f"AUC={summary.auc_mean:.4f} "
        f"(uniform nominal chance=0.111; empirical priors recorded)"
    )
    primary_gate = summary.primary_metric_gate
    print(
        "[primary metric gate] "
        f"passed={primary_gate['passed']} "
        f"claim_eligible={primary_gate['claim_eligible']} "
        f"coverage={primary_gate['n_covered']}/"
        f"{primary_gate['n_total']} "
        f"minimum={primary_gate['minimum_coverage']:.3f} "
        f"failed_checks={primary_gate.get('failed_checks', [])}; "
        "results remain descriptive when the gate fails",
        flush=True,
    )
    efficiency = summary.repetition_efficiency
    print(
        f"[efficiency] {efficiency.aggregation}/{efficiency.budget_semantics} error<="
        f"{efficiency.target_error_rate:.3f}: "
        f"K={efficiency.repetitions_to_target_error}; "
        f"minimum_coverage={efficiency.minimum_coverage:.3f}",
        flush=True,
    )

    if args.evaluation_mode == "confirmatory" and (
        source_tree_sha256(_ROOT) != args.source_sha256
        or runtime_environment_sha256() != args.runtime_sha256
        or external_assets_sha256(
            {
                "frozen_erp_prior": args.frozen_erp_prior,
                "pretrained_checkpoint": None,
                "pretrained_mapping": None,
            }
        )
        != args.external_assets_sha256
    ):
        raise RuntimeError("Source, runtime, or external assets changed during evaluation.")
    scores_path = save_subject_scores(
        summary,
        "n2p3net",
        scores_dir / "n2p3net.json",
        seed=args.seed,
        evaluation_mode=args.evaluation_mode,
        protocol_sha256=args.protocol_sha256,
        confirmatory_id=confirmatory_id,
        confirmatory_lock_sha256=confirmatory_lock_sha256,
        source_sha256=args.source_sha256,
        runtime_sha256=args.runtime_sha256,
        external_assets_sha256=args.external_assets_sha256,
    )
    print(f"[scores] {scores_path}")
    prequential_claim_gate = _outer_prequential_claim_gate(summary.per_fold)
    print(
        "[prequential claim] "
        f"passed={prequential_claim_gate['passed']} "
        f"active_fraction={prequential_claim_gate['active_fusion_fraction']:.3f} "
        f"mean_auc_delta={prequential_claim_gate['mean_auc_delta']}",
        flush=True,
    )

    record["results"] = {
        "hit_rate_mean": float(summary.hit_rate_mean),
        "hit_rate_std": float(summary.hit_rate_std),
        "balanced_acc_mean": float(summary.balanced_acc_mean),
        "auc_mean": None if summary.auc_mean != summary.auc_mean else float(summary.auc_mean),
        "transductive_balanced_acc_mean": summary.transductive_balanced_acc_mean,
        "primary_decision_metric": args.primary_decision,
        "primary_metric_gate": summary.primary_metric_gate,
        "descriptive_decision_records": summary.descriptive_decision_records,
        "decision_metrics": {
            name: {
                "hit_rate": metric.hit_rate,
                "evidence_budget": metric.evidence_budget,
                "aggregation": metric.aggregation,
                "n_covered": metric.n_covered,
                "n_total": metric.n_total,
                "coverage": metric.coverage,
            }
            for name, metric in summary.decision_metrics.items()
        },
        "repetition_efficiency": asdict(summary.repetition_efficiency),
        "confound_baselines": {
            name: {"hit_rate": baseline.hit_rate, "n_subjects": baseline.n_subjects}
            for name, baseline in summary.confound_baselines.items()
        },
        "per_fold": [
            {
                "fold": display_fold_offset + batch_fold_idx,
                "batch_fold": batch_fold_idx,
                "batch_index": batch_index,
                "hit_rate": f.hit_rate,
                "balanced_acc": f.balanced_acc,
                "auc": None if f.auc != f.auc else f.auc,
                "n_subjects": f.n_subjects,
                "n_test_trials": f.n_test_trials,
                "threshold": f.threshold,
                "threshold_source": f.threshold_source,
                "transductive_balanced_acc": f.transductive_balanced_acc,
                "epochs_ran": f.epochs_ran,
                "best_task_epoch": f.best_task_epoch,
                "best_density_epoch": f.best_density_epoch,
                "best_task_val_loss": f.best_task_val_loss,
                "best_density_nll": f.best_density_nll,
                "task_patience_exhausted": f.task_patience_exhausted,
                "val_innovation_nlls": f.val_innovation_nlls,
                "task_val_aucs": f.task_val_aucs,
                "val_objective_losses": f.val_objective_losses,
                "epoch_trajectory_audit": f.epoch_trajectory_audit,
                "neural_ride_audit": f.audit,
            }
            for batch_fold_idx, f in enumerate(summary.per_fold)
        ],
        "scores_path": str(scores_path),
        "prequential_claim_gate": prequential_claim_gate,
    }
    record["timing"] = {
        "total_wall_seconds": wall_seconds,
        "postprocess_cpu_threads": postprocess_cpu_threads,
        "parent_cpu_scheduler": parent_cpu_scheduler,
        "fit_durations_sec": [
            fold.fit_sec for fold in summary.per_fold if fold.fit_sec is not None
        ],
        "fit_peak_memory_mb": [
            fold.fit_peak_memory_mb
            for fold in summary.per_fold
            if fold.fit_peak_memory_mb is not None
        ],
        "parent_final_fold_refit_sec": None,
        "component_artifact_source": "fold_worker_summary",
    }
    previous_results = (previous_record or {}).get("results", {})
    previous_per_fold = (
        previous_results.get("per_fold", [])
        if isinstance(previous_results, dict)
        else []
    )
    current_per_fold = record["results"]["per_fold"]
    if isinstance(previous_per_fold, list) and previous_per_fold:
        combined_per_fold = [*previous_per_fold, *current_per_fold]
        record["results"]["per_fold"] = combined_per_fold
        for field_name in (
            "hit_rate",
            "balanced_acc",
            "transductive_balanced_acc",
            "auc",
        ):
            values = np.asarray(
                [row.get(field_name) for row in combined_per_fold], dtype=float
            )
            finite = values[np.isfinite(values)]
            mean_name = {
                "hit_rate": "hit_rate_mean",
                "balanced_acc": "balanced_acc_mean",
                "transductive_balanced_acc": "transductive_balanced_acc_mean",
                "auc": "auc_mean",
            }[field_name]
            record["results"][mean_name] = (
                float(finite.mean()) if len(finite) else None
            )
            if field_name == "hit_rate":
                record["results"]["hit_rate_std"] = (
                    float(finite.std()) if len(finite) else None
                )
        previous_timing = (
            (previous_record or {}).get("timing", {})
            if isinstance(previous_record, dict)
            else {}
        )
        if isinstance(previous_timing, dict):
            for field_name in ("fit_durations_sec", "fit_peak_memory_mb"):
                prior_values = previous_timing.get(field_name, [])
                current_values = record["timing"].get(field_name, [])
                if isinstance(prior_values, list) and isinstance(current_values, list):
                    record["timing"][field_name] = [*prior_values, *current_values]
        record["batch_index"] = batch_index
        record["batches_accumulated"] = batch_index + 1
    record["training_history_last_fold"] = last_fold.training_history
    # GLM：记录早停协议是否生效（最后 fold 的验证被试数与 val 曲线长度）。
    record["protocol"] = {
        "val_subject_frac": val_subject_frac,
        "early_stop_patience": args.early_stop_patience,
        "last_fold_val_subjects": last_fold.val_subjects,
        "last_fold_audit_subjects": last_fold.audit_subjects,
        "last_fold_n_val_epochs": len(last_fold.training_history.get("val_losses", []))
        if last_fold.training_history
        else None,
        "erp_calibration": args.erp_calibration,
        "last_fold_erp_calibration": last_fold.erp_calibration,
        "epoch_trajectory_audit": {
            "enabled": bool(args.epoch_trajectory_audit),
            "scope": "development_only",
            "checkpoint_selection_allowed": False,
            "checkpoint_pattern": "epochs/checkpoints/fold_<fold>/epoch_<epoch>.pt",
            "trajectory_pattern": "epochs/checkpoints/fold_<fold>/trajectory.json",
            "chain_metrics_included": False,
        },
    }

    record["components_last_fold"] = last_fold.component_summary
    record["finished_utc"] = datetime.now(UTC).isoformat()

    record_path = run_dir / "record.json"
    # The queue uses record.json as its completion barrier. Publish only after
    # the complete cumulative record has been written, so the next batch can
    # never observe a truncated file and discard the previous batch.
    record_tmp_path = run_dir / f".record.json.{os.getpid()}.tmp"
    record_tmp_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record_tmp_path.replace(record_path)
    print(f"[record] {record_path}")


if __name__ == "__main__":
    main()
