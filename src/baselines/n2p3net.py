"""N2P3Net 的 Baseline 适配器：把 Trainer 训练出的模型接到 evaluate() 协议。

GTN-N2P3Net 闭环使用同一 evaluate()，因此 hit rate / bacc / AUC / scores 的
口径与 7 个基线完全一致。每个 fit 都重新构造 N2P3Net，避免 LOSO fold 间权重泄漏。

GLM（2026-08-22）：被试级验证早停协议。evaluate() 会把训练 fold 的 subject_ids
传给本适配器（fit_accepts_subject_ids = True）；fit 按被试分组切出验证集，
Trainer 在 val loss 不再改善时早停并恢复最佳权重。动机（失败诊断 §2.4）：
固定 10ep 是「欠拟合赌博」——一折逐 epoch 曲线显示 held-out 指标在 epoch 10–11
见顶后崩塌（30ep bacc 0.68→0.55），无验证协议的固定 epoch 要么停在峰值前、
要么冲进过拟合区。被试级（而非试次级随机）切分保证同被试试次不跨 train/val，
验证损失才能诚实反映跨被试泛化。
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from baselines.calibration import (
    WeightedLogitTemperatureCalibration,
    fit_weighted_logit_temperature,
)
from baselines.classic import Baseline
from baselines.fusion_v12 import (
    curve_improvement,
    e_process_diagnostics,
    fit_nested_fusion,
    replay_chain_stopping_from_contributions,
    risk_coverage_curve,
    select_v12_fusion,
    trajectory_contributions,
    two_hypothesis_conformal_flags,
)
from baselines.validation import (
    SubjectValidationSplit,
    subject_disjoint_audit_split,
    subject_disjoint_validation_split,
)
from measurement.latency_measurement import (
    LatencyMeasurement,
    LatencyPosterior,
    detached_expected_window,
)
from models.innovation import CausalInnovationOutput
from models.n2p3net import N2P3Net
from models.reliability_v12 import (
    CleanProbabilityEstimator,
    convert_prior_odds,
    evaluate_clean_probability_gate,
    evaluate_fidelity_gate,
)
from models.repetition import extract_quality_features
from models.repetition_v12 import AdditiveRepetitionEvidence, state_residual_gate_decision
from train.contracts import TrialContext
from train.device import empty_cache, get_device
from train.preloaded import GTNSetDataLoader, PreloadedDataLoader
from train.prequential import (
    prequential_log_likelihood_ratio,
    prequential_score_per_trial,
)
from train.prequential_audit import audit_prequential_model
from train.trainer import Trainer, TrainerConfig


def _as_float32_epochs(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X)
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("X must have a floating dtype.")
    return values.astype(np.float32, copy=False)

# Pre-registered thresholds for the synthetic-corruption reliability audit.
# They are claim-eligibility thresholds only: a failed gate records
# ``checks``/``failed_checks`` and keeps the chain result descriptive; it must
# never delete the experiment summary.
REPETITION_RELIABILITY_THRESHOLDS: dict[str, float] = {
    "artifact_auc_min": 0.80,
    "artifact_brier_max": 0.15,
    "artifact_nll_max": 0.50,
    "artifact_ece10_max": 0.20,
    "clean_mean_reliability_min": 0.70,
    "corrupt_mean_reliability_max": 0.30,
    "mean_reliability_gap_min": 0.40,
    "paired_clean_wins_min": 0.80,
    "target_leakage_auc_deviation_max": 0.10,
    "target_nonlinear_leakage_auc_deviation_max": 0.10,
    "target_distribution_ks_max": 0.20,
    "subject_artifact_auc_min": 0.65,
    "subject_reliability_gap_min": 0.25,
    "subject_target_auc_deviation_max": 0.25,
    "subject_target_nonlinear_auc_deviation_max": 0.25,
    "subject_target_ks_max": 0.40,
}


@dataclass
class N2P3NetBaselineConfig:
    """N2P3Net 进入 evaluate 协议的训练配置。"""

    model_kwargs: dict = field(default_factory=dict)
    trainer_kwargs: dict = field(default_factory=dict)


def _make_epoch_checkpoint_sink(
    *, enabled: bool, fold_id: int | None = None, directory: str | Path | None = None
) -> tuple[
    Callable[[dict[str, object], dict[str, torch.Tensor]], None] | None,
    list[Path | dict[str, object]],
]:
    """Persist raw end-of-epoch states for the opt-in development audit."""

    references: list[Path | dict[str, object]] = []
    if not enabled:
        return None, references
    if directory is None:
        directory = os.environ.get("N2P3NET_EPOCH_PROGRESS_DIR")
    fold_raw = str(fold_id) if fold_id is not None else os.environ.get("N2P3NET_FOLD_ID")
    if directory and fold_raw is not None:
        try:
            fold = int(fold_raw)
        except ValueError:
            fold = None
        if fold is not None:
            checkpoint_dir = Path(directory) / "checkpoints" / f"fold_{fold}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            def persist(
                event: dict[str, object], state_dict: dict[str, torch.Tensor]
            ) -> None:
                epoch = int(event["epoch"])
                target = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                temporary = checkpoint_dir / f"epoch_{epoch:03d}.pt.tmp"
                torch.save(
                    {
                        "schema": "n2p3net_epoch_trajectory_checkpoint/1",
                        "development_only": True,
                        "event": dict(event),
                        "state_dict": state_dict,
                    },
                    temporary,
                )
                os.replace(temporary, target)
                references.append(target)

            return persist, references

    # Library/tests may not have a run directory. Keep the opt-in behavior
    # usable there while production runners always take the persistent path.
    def retain(event: dict[str, object], state_dict: dict[str, torch.Tensor]) -> None:
        references.append(
            {
                "schema": "n2p3net_epoch_trajectory_checkpoint/1",
                "development_only": True,
                "event": dict(event),
                "state_dict": state_dict,
            }
        )

    return retain, references


def _binary_ece(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(probabilities), 1)
    ece = 0.0
    for index in range(n_bins):
        upper_inclusive = index == n_bins - 1
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if upper_inclusive
            else probabilities < edges[index + 1]
        )
        if np.any(mask):
            ece += (
                np.count_nonzero(mask)
                / total
                * abs(probabilities[mask].mean() - labels[mask].mean())
            )
    return float(ece)


def _subject_balanced_weights(subject_ids: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """Return trial weights giving every validation subject equal total mass."""

    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1 or len(subject_ids) == 0:
        raise ValueError("subject_ids must be a non-empty one-dimensional array.")
    _, inverse, counts = np.unique(subject_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    weights /= weights.sum()
    return torch.from_numpy(weights).to(device=device)


def _weighted_llr_scale(
    values: np.ndarray,
    subject_ids: np.ndarray,
    *,
    device: torch.device,
) -> float:
    scores = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device=device)
    weights = _subject_balanced_weights(subject_ids, device=device)
    root_mean_square = (weights * scores.square()).sum().sqrt().clamp_min(1e-6)
    return float(root_mean_square.cpu())



def _subject_balanced_bce(
    logits: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    *,
    device: torch.device,
) -> float:
    values = torch.from_numpy(np.asarray(logits, dtype=np.float32)).to(device=device)
    target = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device=device)
    weights = _subject_balanced_weights(subject_ids, device=device)
    per_trial = F.binary_cross_entropy_with_logits(values, target, reduction="none")
    return float((weights * per_trial).sum().detach().cpu())


def _two_sample_ks(first: np.ndarray, second: np.ndarray) -> float:
    """Exact empirical one-dimensional KS distance without a SciPy dependency."""

    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    support = np.sort(np.unique(np.concatenate((first, second))))
    if not len(first) or not len(second):
        return float("nan")
    first_cdf = np.searchsorted(first, support, side="right") / len(first)
    second_cdf = np.searchsorted(second, support, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _reliability_gate_metrics(
    clean_rho: np.ndarray,
    corrupt_rho: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, object]:
    """Audit synthetic-corruption calibration and target-label independence."""

    clean_rho = np.asarray(clean_rho, dtype=np.float64).reshape(-1)
    corrupt_rho = np.asarray(corrupt_rho, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    subject_ids = np.asarray(subject_ids).reshape(-1)
    n = len(clean_rho)
    if corrupt_rho.ndim != 2 or corrupt_rho.shape[1] != n:
        raise ValueError("corrupt_rho must be (n_corruptions, n_clean_trials).")
    if labels.shape != (n,) or subject_ids.shape != (n,):
        raise ValueError("labels and subject_ids must align with clean_rho.")
    if n < 8 or np.unique(labels).size != 2:
        checks: dict[str, bool] = {
            "minimum_validation_trials": n >= 8,
            "both_target_classes_present": np.unique(labels).size == 2,
        }
        return {
            "passed": False,
            "scope": "held_out_synthetic_corruption_only",
            "n_samples": n,
            "failure": "needs_at_least_eight_trials_and_both_target_classes",
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "thresholds": dict(REPETITION_RELIABILITY_THRESHOLDS),
        }

    corrupt_flat = corrupt_rho.reshape(-1)
    probabilities = np.concatenate((clean_rho, corrupt_flat)).clip(1e-6, 1.0 - 1e-6)
    artifact_labels = np.concatenate(
        (np.ones(n, dtype=np.int64), np.zeros(len(corrupt_flat), dtype=np.int64))
    )
    artifact_auc = float(roc_auc_score(artifact_labels, probabilities))
    artifact_brier = float(np.mean(np.square(probabilities - artifact_labels)))
    artifact_nll = float(
        -np.mean(
            artifact_labels * np.log(probabilities)
            + (1 - artifact_labels) * np.log1p(-probabilities)
        )
    )
    artifact_ece = _binary_ece(probabilities, artifact_labels)
    clean_mean = float(clean_rho.mean())
    corrupt_mean = float(corrupt_flat.mean())
    gap = clean_mean - corrupt_mean
    paired_clean = np.broadcast_to(clean_rho, corrupt_rho.shape)
    paired_win_rate = float(
        np.mean((paired_clean > corrupt_rho) + 0.5 * (paired_clean == corrupt_rho))
    )

    target = clean_rho[labels == 1]
    nontarget = clean_rho[labels == 0]
    target_auc = float(roc_auc_score(labels, clean_rho))
    centered_score = np.abs(clean_rho - np.median(clean_rho))
    target_deviation_auc = float(roc_auc_score(labels, centered_score))
    target_ks = _two_sample_ks(target, nontarget)

    subject_metrics: list[dict[str, float | str]] = []
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        subject_labels = labels[mask]
        if np.unique(subject_labels).size != 2:
            continue
        subject_clean = clean_rho[mask]
        subject_corrupt = corrupt_rho[:, mask]
        subject_artifact_labels = np.concatenate(
            (
                np.ones(mask.sum(), dtype=np.int64),
                np.zeros(subject_corrupt.size, dtype=np.int64),
            )
        )
        subject_probabilities = np.concatenate((subject_clean, subject_corrupt.reshape(-1)))
        subject_centered = np.abs(subject_clean - np.median(subject_clean))
        subject_metrics.append(
            {
                "subject": str(subject),
                "n_trials": int(mask.sum()),
                "n_target_trials": int(np.count_nonzero(subject_labels == 1)),
                "n_nontarget_trials": int(np.count_nonzero(subject_labels == 0)),
                "artifact_auc": float(
                    roc_auc_score(subject_artifact_labels, subject_probabilities)
                ),
                "reliability_gap": float(subject_clean.mean() - subject_corrupt.mean()),
                "target_auc_deviation": abs(
                    float(roc_auc_score(subject_labels, subject_clean)) - 0.5
                ),
                "target_nonlinear_auc_deviation": abs(
                    float(roc_auc_score(subject_labels, subject_centered)) - 0.5
                ),
                "target_ks": _two_sample_ks(
                    subject_clean[subject_labels == 1],
                    subject_clean[subject_labels == 0],
                ),
            }
        )
    if not subject_metrics:
        return {
            "passed": False,
            "scope": "held_out_synthetic_corruption_only",
            "n_samples": n,
            "failure": "no_validation_subject_contains_both_target_classes",
            "checks": {"every_validation_subject_has_both_classes": False},
            "failed_checks": ["every_validation_subject_has_both_classes"],
            "thresholds": dict(REPETITION_RELIABILITY_THRESHOLDS),
        }

    subject_artifact_auc_min = min(item["artifact_auc"] for item in subject_metrics)
    subject_gap_min = min(item["reliability_gap"] for item in subject_metrics)
    subject_target_auc_deviation_max = max(item["target_auc_deviation"] for item in subject_metrics)
    subject_target_nonlinear_auc_deviation_max = max(
        item["target_nonlinear_auc_deviation"] for item in subject_metrics
    )
    subject_target_ks_max = max(item["target_ks"] for item in subject_metrics)
    t = REPETITION_RELIABILITY_THRESHOLDS
    checks = {
        "artifact_auc": artifact_auc >= t["artifact_auc_min"],
        "artifact_brier": artifact_brier <= t["artifact_brier_max"],
        "artifact_nll": artifact_nll <= t["artifact_nll_max"],
        "artifact_ece10": artifact_ece <= t["artifact_ece10_max"],
        "clean_mean_reliability": clean_mean >= t["clean_mean_reliability_min"],
        "corrupt_mean_reliability": corrupt_mean <= t["corrupt_mean_reliability_max"],
        "mean_reliability_gap": gap >= t["mean_reliability_gap_min"],
        "paired_clean_wins": paired_win_rate >= t["paired_clean_wins_min"],
        "target_leakage_auc": (
            abs(target_auc - 0.5) <= t["target_leakage_auc_deviation_max"]
        ),
        "target_nonlinear_leakage_auc": (
            abs(target_deviation_auc - 0.5) <= t["target_nonlinear_leakage_auc_deviation_max"]
        ),
        "target_distribution_ks": target_ks <= t["target_distribution_ks_max"],
        "subject_artifact_auc_min": subject_artifact_auc_min >= t["subject_artifact_auc_min"],
        "subject_reliability_gap_min": subject_gap_min >= t["subject_reliability_gap_min"],
        "subject_target_auc_deviation_max": (
            subject_target_auc_deviation_max <= t["subject_target_auc_deviation_max"]
        ),
        "subject_target_nonlinear_auc_deviation_max": (
            subject_target_nonlinear_auc_deviation_max
            <= t["subject_target_nonlinear_auc_deviation_max"]
        ),
        "subject_target_ks_max": subject_target_ks_max <= t["subject_target_ks_max"],
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "scope": "held_out_synthetic_corruption_only",
        "n_samples": n,
        "n_corruptions_per_sample": int(corrupt_rho.shape[0]),
        "artifact_auc": artifact_auc,
        "artifact_brier": artifact_brier,
        "artifact_nll": artifact_nll,
        "artifact_ece10": artifact_ece,
        "clean_mean_reliability": clean_mean,
        "corrupt_mean_reliability": corrupt_mean,
        "mean_reliability_gap": gap,
        "paired_clean_wins": paired_win_rate,
        "target_leakage_auc": target_auc,
        "target_nonlinear_leakage_auc": target_deviation_auc,
        "target_distribution_ks": target_ks,
        "subject_artifact_auc_min": subject_artifact_auc_min,
        "subject_reliability_gap_min": subject_gap_min,
        "subject_target_auc_deviation_max": subject_target_auc_deviation_max,
        "subject_target_nonlinear_auc_deviation_max": (subject_target_nonlinear_auc_deviation_max),
        "subject_target_ks_max": subject_target_ks_max,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "thresholds": dict(t),
        "n_validation_subjects": int(np.unique(subject_ids).size),
        "n_complete_subjects": len(subject_metrics),
        "subject_metrics": subject_metrics,
    }


def _seed_model_initialization(seed: int) -> None:
    """Reset accelerator and host RNGs before constructing every fold model."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(int(seed))


class N2P3NetBaseline(Baseline):
    """fit(X,y[,subject_ids]) / predict_logit(X) 适配器。

    Parameters
    ----------
    model_kwargs : dict
        透传给 N2P3Net 的构造参数。
    trainer_kwargs : dict
        透传给 TrainerConfig 的字段；epochs/batch_size/early_stop_patience 等在此设置。
    E_chn / channel_mask / device :
        与 Trainer 契约一致；通道轴必须是数据集的原生物理布局，缺失通道只能通过显式 mask 表示。
    val_subject_frac : float | None
        被试级验证集比例（GLM 协议，默认 0.08）；None 关闭验证早停（旧固定 epoch 行为）。
    val_subjects_min / val_subjects_max : int
        验证被试数的下/上限（小池保护：太少验证信号不稳；太多挤占训练数据）。
    """

    # evaluate() 检测此属性决定是否传 subject_ids（GLM：被试级验证早停）。
    fit_accepts_subject_ids = True
    fit_accepts_group_ids = True
    fit_accepts_trial_context = True
    fit_accepts_acquisition_indices = True
    fit_accepts_trial_channel_mask = True
    predict_accepts_acquisition_indices = True
    predict_accepts_trial_channel_mask = True
    auxiliary_predict_accepts_trial_channel_mask = True

    def __init__(
        self,
        model_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        E_chn: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
        val_subject_frac: float | None = 0.08,
        val_subjects_min: int = 2,
        val_subjects_max: int = 12,
        audit_subjects: int = 4,
        erp_calibrator: Callable[[np.ndarray, np.ndarray, np.ndarray], dict] | None = None,
    ):
        self.model_kwargs = dict(model_kwargs or {})
        self.trainer_kwargs = dict(trainer_kwargs or {})
        self.device = device if device is not None else get_device()
        # Keep prototype inputs on CPU so Linux fork workers can inherit a
        # CUDA-free adapter and initialize their own CUDA context safely.
        self.E_chn = E_chn.detach().cpu() if E_chn is not None else None
        if channel_mask is not None and channel_mask.dtype != torch.bool:
            raise ValueError("channel_mask must have boolean dtype.")
        self.channel_mask = channel_mask.detach().cpu() if channel_mask is not None else None
        if self.channel_mask is not None:
            expected_channels = int(self.model_kwargs.get("n_channels", len(self.channel_mask)))
            if self.channel_mask.shape != (expected_channels,) or not bool(
                self.channel_mask.any()
            ):
                raise ValueError("channel_mask must match n_channels and retain one channel.")
        self.val_subject_frac = val_subject_frac
        self.val_subjects_min = int(val_subjects_min)
        self.val_subjects_max = int(val_subjects_max)
        self.audit_subjects = int(audit_subjects)
        self.erp_calibrator = erp_calibrator
        self._fitted = False
        self._runtime_E_chn: torch.Tensor | None = None
        self._runtime_channel_mask: torch.Tensor | None = None
        # 实验记录：每个 fold 的训练耗时 / 显存峰值 / 最后一个 fold 的 history。
        self.fit_durations: list[float] = []
        self.fit_peak_memory_mb: list[float] = []
        self.last_history: dict | None = None
        # GLM：每 fold 实际用到的训练/验证被试数（记录协议是否生效）。
        self.last_val_subjects: int | None = None
        self.last_audit_subjects: int | None = None
        self.last_erp_calibration: dict | None = None
        self.calibration_logits_: np.ndarray | None = None
        self.calibration_labels_: np.ndarray | None = None
        self.calibration_source_: str | None = None
        self.calibration_branch_logits_: dict[str, np.ndarray] | None = None
        self.repetition_temperature_calibration_: WeightedLogitTemperatureCalibration | None = None
        self.repetition_fitted_ = False
        self.repetition_ready_ = False
        self.repetition_reliability_audit_: dict[str, object] | None = None
        # Object L: fold-local measurement posterior and gated PCW consumer.
        self.measurement_estimator_: LatencyMeasurement | None = None
        self.measurement_posterior_: LatencyPosterior | None = None
        self.measurement_gate_: dict[str, object] | None = None
        self.measurement_gain_: float = 0.0
        self._measurement_cache_key_: str | None = None
        self._measurement_cache_posterior_: LatencyPosterior | None = None
        self._measurement_cache_window_: torch.Tensor | None = None
        self._measurement_posterior_n_: int = -1
        self._measurement_channel_mask_: np.ndarray | None = None
        # Object Q optional hard-label activation path.
        self.clean_probability_pool_: dict[str, object] | None = None
        self.last_pos_weight: float | None = None
        self.last_train_prior: float | None = None
        self.prequential_audit_: dict[str, object] | None = None
        self.generative_profile_ = None
        self.prequential_variant_: str | None = None
        self.prequential_scale_: float = 1.0
        self.prequential_coefficient_: float = 0.0
        self.prequential_base_slope_: float = 1.0
        self.prequential_base_intercept_: float = 0.0
        self.epoch_trajectory_checkpoints_: list[Path | dict[str, object]] = []
        self._evaluation_fold_id: int | None = None

    @property
    def _model_E_chn(self) -> torch.Tensor | None:  # noqa: N802
        return self._runtime_E_chn if self._runtime_E_chn is not None else self.E_chn

    @property
    def _model_channel_mask(self) -> torch.Tensor | None:
        return (
            self._runtime_channel_mask
            if self._runtime_channel_mask is not None
            else self.channel_mask
        )

    def _validated_trial_channel_mask(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        """Validate an optional per-trial mask against the fixed model layout."""

        X = _as_float32_epochs(X)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
        static = (
            self.channel_mask.detach().cpu().numpy().astype(bool)
            if self.channel_mask is not None
            else None
        )
        if trial_channel_mask is None:
            if static is not None and bool((X[:, ~static, :] != 0.0).any()):
                raise ValueError("X must be zero where channel_mask is false.")
            return None
        mask = np.asarray(trial_channel_mask)
        if mask.dtype != np.dtype(bool):
            raise ValueError("trial_channel_mask must have boolean dtype.")
        if mask.shape != X.shape[:2]:
            raise ValueError("trial_channel_mask must have shape (N,C) matching X.")
        if not bool(mask.any(axis=1).all()):
            raise ValueError("Every trial must retain at least one observed channel.")
        if static is not None and bool((mask & ~static[None]).any()):
            raise ValueError("trial_channel_mask cannot enable a permanently absent channel.")
        if bool((X[~mask] != 0.0).any()):
            raise ValueError("X must be zero where trial_channel_mask is false.")
        return mask

    def _release_fold_runtime(self) -> None:
        """Release the previous fold before constructing the next CUDA model."""

        stale_model = getattr(self, "model_", None)
        self.model_ = None
        self._fitted = False
        self._runtime_E_chn = None
        self._runtime_channel_mask = None
        self.generative_profile_ = None
        self.prequential_audit_ = None
        self.prequential_variant_ = None
        self.prequential_base_slope_ = 1.0
        self.prequential_base_intercept_ = 0.0
        self.repetition_temperature_calibration_ = None
        self.repetition_fitted_ = False
        self.repetition_ready_ = False
        self.repetition_reliability_audit_ = None
        self.measurement_estimator_ = None
        self.measurement_posterior_ = None
        self._measurement_channel_mask_ = None
        self.measurement_gate_ = None
        self.measurement_gain_ = 0.0
        self._measurement_cache_key_ = None
        self._measurement_cache_posterior_ = None
        self._measurement_cache_window_ = None
        self._measurement_posterior_n_ = -1
        self.clean_probability_pool_ = None
        self.calibration_logits_ = None
        self.calibration_labels_ = None
        self.calibration_source_ = None
        self.calibration_branch_logits_ = None
        self.epoch_trajectory_checkpoints_ = []
        self.last_history = None
        if stale_model is None:
            return
        del stale_model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            empty_cache(self.device)
            torch.cuda.ipc_collect()
            allocated_mb = torch.cuda.memory_allocated() / 1e6
            reserved_mb = torch.cuda.memory_reserved() / 1e6
            print(
                f"[fold cleanup] released previous model; "
                f"cuda allocated={allocated_mb:.1f} MB reserved={reserved_mb:.1f} MB",
                flush=True,
            )
        else:
            empty_cache(self.device)

    def _resolve_fold_class_statistics(self, config: TrainerConfig, labels: np.ndarray) -> None:
        labels = np.asarray(labels).reshape(-1)
        positives = int(np.count_nonzero(labels == 1))
        negatives = int(np.count_nonzero(labels == 0))
        if positives == 0 or negatives == 0:
            raise ValueError("Fold training requires both target classes.")
        if config.auto_pos_weight:
            config.pos_weight = negatives / positives
        self.last_pos_weight = float(config.pos_weight)
        self.last_train_prior = positives / (positives + negatives)

    def _subject_validation_split(
        self,
        subject_ids: np.ndarray,
        digits: np.ndarray | None = None,
    ) -> SubjectValidationSplit:
        split = subject_disjoint_validation_split(
            subject_ids,
            fraction=self.val_subject_frac,
            min_subjects=self.val_subjects_min,
            max_subjects=self.val_subjects_max,
            seed=int(self.trainer_kwargs.get("seed", 0)),
        )
        evidence_ks = tuple(self.trainer_kwargs.get("digit_evidence_ks", (1, 3, 5, 10, 15)))
        if (
            max(
                float(self.trainer_kwargs.get("lambda_digit", 0.0)),
                float(self.trainer_kwargs.get("lambda_conditional_nll", 0.0)),
            )
            <= 0.0
            or digits is None
            or split.n_validation_subjects == 0
        ):
            return split
        kmin = min(int(k) for k in evidence_ks)
        subject_ids = np.asarray(subject_ids)
        digits = np.asarray(digits)
        eligible = []
        for subject in np.unique(subject_ids):
            counts = [
                np.count_nonzero((subject_ids == subject) & (digits == digit))
                for digit in range(1, 10)
            ]
            if min(counts) >= kmin:
                eligible.append(subject)
        if len(eligible) < split.n_validation_subjects:
            raise ValueError(
                "Set-supervised validation cannot satisfy minimum-K coverage: "
                f"need {split.n_validation_subjects} subjects, found {len(eligible)} at K={kmin}."
            )
        rng = np.random.default_rng(int(self.trainer_kwargs.get("seed", 0)))
        selected = tuple(
            rng.choice(
                np.asarray(eligible), size=split.n_validation_subjects, replace=False
            ).tolist()
        )
        validation_mask = np.isin(subject_ids, selected)
        return SubjectValidationSplit(~validation_mask, validation_mask, selected)

    def _split_val_subjects(
        self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """按被试分组切验证集；返回 (X_train, y_train, X_val, y_val, n_val_subjects)。

        验证被试由 cfg.seed 决定（确定性）；被试数 = clamp(round(frac·N), min, max)。
        训练被试数 < 4 时不足以分组，返回整集训练（n_val_subjects=0）。
        """
        split = self._subject_validation_split(np.asarray(subject_ids))
        return (
            X[split.train_mask],
            y[split.train_mask],
            X[split.validation_mask],
            y[split.validation_mask],
            split.n_validation_subjects,
        )

    def _fit_common(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        train_subject_ids: np.ndarray | None = None,
        train_group_ids: np.ndarray | None = None,
        train_digits: np.ndarray | None = None,
        train_acquisition_indices: np.ndarray | None = None,
        train_trial_channel_mask: np.ndarray | None = None,
        val_subject_ids: np.ndarray | None = None,
        val_group_ids: np.ndarray | None = None,
        val_digits: np.ndarray | None = None,
        val_acquisition_indices: np.ndarray | None = None,
        val_trial_channel_mask: np.ndarray | None = None,
        reconstruction_X: np.ndarray | None = None,
        reconstruction_y: np.ndarray | None = None,
        reconstruction_trial_channel_mask: np.ndarray | None = None,
        audit_X: np.ndarray | None = None,
        audit_y: np.ndarray | None = None,
        audit_subject_ids: np.ndarray | None = None,
        audit_group_ids: np.ndarray | None = None,
        audit_digits: np.ndarray | None = None,
        audit_acquisition_indices: np.ndarray | None = None,
        audit_trial_channel_mask: np.ndarray | None = None,
    ) -> N2P3NetBaseline:
        """构造 model/trainer/loader 并跑 fit（fit 主路径与域对齐适配器共用）。

        X_val/y_val 给定时启用验证早停（Trainer 恢复 val loss 最佳权重）。
        """
        self._release_fold_runtime()
        model_kwargs = dict(self.model_kwargs)
        cfg = TrainerConfig(**self.trainer_kwargs)
        self.last_erp_calibration = None
        self._runtime_E_chn = None
        self._runtime_channel_mask = None
        if self.erp_calibrator is not None:
            if train_subject_ids is None:
                raise ValueError("fold-local ERP calibration requires training subject_ids.")
            if getattr(self.erp_calibrator, "accepts_trial_channel_mask", False):
                calibration = self.erp_calibrator(
                    X_train,
                    y_train,
                    train_subject_ids,
                    trial_channel_mask=train_trial_channel_mask,
                )
            else:
                calibration = self.erp_calibrator(X_train, y_train, train_subject_ids)
            for key in ("tau0_ms", "tau0_bounds", "sigma_bounds"):
                model_kwargs[key] = calibration[key]
            self.last_erp_calibration = calibration
        _seed_model_initialization(cfg.seed)
        model = N2P3Net(**model_kwargs)
        self._resolve_fold_class_statistics(cfg, y_train)
        trainer = Trainer(
            model,
            cfg,
            E_chn=self.E_chn,
            channel_mask=self.channel_mask,
            device=self.device,
            fold_id=self._evaluation_fold_id,
        )

        set_objective_active = cfg.lambda_digit > 0.0 or cfg.lambda_conditional_nll > 0.0
        if set_objective_active and (train_digits is None or train_group_ids is None):
            raise ValueError("Set objectives require training digits and selection group IDs.")

        # Upload each fold's trial arrays once.  PreloadedDataLoader and
        # GTNSetDataLoader both retain the tensors, so passing these same CUDA
        # tensors prevents the trial and set paths from allocating duplicate
        # copies of X/y in device memory.
        train_X_device = torch.from_numpy(X_train).to(self.device)
        train_y_device = torch.from_numpy(y_train).to(self.device, dtype=torch.float32)
        loader = PreloadedDataLoader(
            train_X_device,
            train_y_device,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed,
            channel_mask=(
                torch.from_numpy(np.asarray(train_trial_channel_mask, dtype=bool)).to(self.device)
                if train_trial_channel_mask is not None
                else None
            ),
            device=self.device,
        )
        set_loader = None
        if set_objective_active:
            _, train_groups = np.unique(train_group_ids, return_inverse=True)
            set_loader = GTNSetDataLoader(
                train_X_device,
                train_y_device,
                torch.from_numpy(np.asarray(train_digits, dtype=np.int64)),
                torch.from_numpy(train_groups.astype(np.int64)),
                acquisition_indices=(
                    torch.from_numpy(np.asarray(train_acquisition_indices, dtype=np.int64))
                    if train_acquisition_indices is not None
                    else None
                ),
                channel_mask=(
                    torch.from_numpy(np.asarray(train_trial_channel_mask, dtype=bool)).to(
                        self.device
                    )
                    if train_trial_channel_mask is not None
                    else None
                ),
                evidence_ks=cfg.digit_evidence_ks,
                batch_size=cfg.batch_size,
                shuffle=True,
                seed=cfg.seed,
                main_domain=cfg.main_domain,
                device=self.device,
                validate_finite=not loader.finite_validated,
            )
            set_loader.finite_validated = loader.finite_validated
        val_loader = None
        val_set_loader = None
        if X_val is not None and len(X_val) > 0:
            val_X_device = torch.from_numpy(X_val).to(self.device)
            val_y_device = torch.from_numpy(y_val).to(self.device, dtype=torch.float32)
            val_loader = PreloadedDataLoader(
                val_X_device,
                val_y_device,
                batch_size=cfg.batch_size,
                shuffle=False,
                seed=cfg.seed,
                channel_mask=(
                    torch.from_numpy(np.asarray(val_trial_channel_mask, dtype=bool)).to(
                        self.device
                    )
                    if val_trial_channel_mask is not None
                    else None
                ),
                device=self.device,
            )
            if set_objective_active:
                if val_digits is None or val_group_ids is None:
                    raise ValueError(
                        "Set-supervised validation requires digits and selection group IDs."
                    )
                _, val_groups = np.unique(val_group_ids, return_inverse=True)
                val_set_loader = GTNSetDataLoader(
                    val_X_device,
                    val_y_device,
                    torch.from_numpy(np.asarray(val_digits, dtype=np.int64)),
                    torch.from_numpy(val_groups.astype(np.int64)),
                    acquisition_indices=(
                        torch.from_numpy(np.asarray(val_acquisition_indices, dtype=np.int64))
                        if val_acquisition_indices is not None
                        else None
                    ),
                    channel_mask=(
                        torch.from_numpy(np.asarray(val_trial_channel_mask, dtype=bool)).to(
                            self.device
                        )
                        if val_trial_channel_mask is not None
                        else None
                    ),
                    evidence_ks=cfg.digit_evidence_ks,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    seed=cfg.seed,
                    main_domain=cfg.main_domain,
                    device=self.device,
                )
                val_set_loader.finite_validated = val_loader.finite_validated
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if reconstruction_X is None:
            reconstruction_X = X_train
            reconstruction_y = y_train
        if reconstruction_y is None or len(reconstruction_X) != len(reconstruction_y):
            raise ValueError("Outer-fold reconstruction X/y must have equal lengths.")
        reconstruction_context = TrialContext(
            X=torch.from_numpy(np.asarray(reconstruction_X, dtype=np.float32)),
            y=torch.from_numpy(np.asarray(reconstruction_y, dtype=np.float32)),
            channel_mask=(
                torch.from_numpy(np.asarray(reconstruction_trial_channel_mask, dtype=bool))
                if reconstruction_trial_channel_mask is not None
                else None
            ),
        )
        if audit_X is not None and len(audit_X) > 0:
            if audit_y is None or len(audit_X) != len(audit_y):
                raise ValueError("Audit X/y must have equal lengths.")
        epoch_progress_callback = self.epoch_progress_callback()
        epoch_checkpoint_sink, epoch_checkpoint_references = _make_epoch_checkpoint_sink(
            enabled=cfg.epoch_trajectory_audit,
            fold_id=self._evaluation_fold_id,
            directory=self._epoch_progress_dir,
        )
        history = trainer.fit(
            loader,
            val_loader=val_loader,
            train_set_loader=set_loader,
            val_set_loader=val_set_loader,
            reconstruction_context=reconstruction_context,
            on_epoch_end=epoch_progress_callback,
            on_epoch_checkpoint=epoch_checkpoint_sink,
        )
        self.epoch_trajectory_checkpoints_ = epoch_checkpoint_references
        if set_loader is not None:
            history["gtn_set_loader"] = {
                "evidence_ks": list(set_loader.evidence_ks),
                "evidence_kmax": set_loader.evidence_k,
                "eligible_groups": set_loader.n_groups_eligible,
                "total_groups": set_loader.n_groups_total,
                "coverage": set_loader.set_coverage,
                "coverage_by_k": set_loader.coverage_by_k,
                "sets_per_epoch": set_loader.n_sets_per_epoch,
            }
        self.fit_durations.append(time.perf_counter() - t0)
        if self.device.type == "cuda":
            self.fit_peak_memory_mb.append(torch.cuda.max_memory_allocated() / 1e6)
        else:
            self.fit_peak_memory_mb.append(float("nan"))
        self.last_history = history

        self.model_ = model
        self._fitted = True
        self._runtime_E_chn = trainer.E_chn
        self._runtime_channel_mask = trainer.channel_mask
        self.generative_profile_ = trainer.generative_profile
        self.prequential_audit_ = None
        self.prequential_variant_ = None
        self.prequential_coefficient_ = 0.0
        self.prequential_base_slope_ = 1.0
        self.prequential_base_intercept_ = 0.0
        # Object L: fit the fold-local latency posterior and gate its detached
        # PCW consumer before any repetition calibration consumes logits.
        measurement_gate = self._fit_measurement_branch(
            X_train,
            y_train,
            X_val,
            y_val,
            train_subject_ids,
            val_subject_ids,
            train_trial_channel_mask=train_trial_channel_mask,
            val_trial_channel_mask=val_trial_channel_mask,
        )
        history["measurement_gate"] = measurement_gate
        if audit_X is not None and len(audit_X) > 0 and trainer.generative_profile is not None:
            if audit_subject_ids is None:
                raise ValueError("Prequential audit requires subject IDs.")
            report = self._audit_prequential(
                audit_X,
                np.asarray(audit_y),
                np.asarray(audit_subject_ids),
                audit_trial_channel_mask,
            )
            self.prequential_audit_ = report
            history["prequential_audit"] = report
            if bool(report["passed"]):
                self.prequential_variant_ = str(report["selected_variant"])
                if X_val is not None and len(X_val) > 0:
                    history["prequential_fusion"] = self._select_prequential_fusion(
                        X_val,
                        y_val,
                        val_subject_ids,
                        val_trial_channel_mask,
                        eligible_variants=tuple(report["eligible_variants"]),
                        density_selected_variant=str(report["selected_variant"]),
                    )
        self.repetition_temperature_calibration_ = None
        self.repetition_fitted_ = False
        self.repetition_ready_ = False
        self.repetition_reliability_audit_ = None
        if cfg.lambda_conditional_nll > 0.0:
            if X_val is None or len(X_val) == 0:
                raise ValueError(
                    "Conditional repetition evidence requires a subject-disjoint inner "
                    "validation fold for temperature calibration."
                )
            validation_logits = self._predict_model_logit(
                X_val, trial_channel_mask=val_trial_channel_mask
            )
            temperature_calibration = fit_weighted_logit_temperature(
                validation_logits,
                y_val,
                pos_weight=float(self.last_pos_weight),
                train_prior=float(self.last_train_prior),
                source="subject_disjoint_validation",
            )
            self.repetition_temperature_calibration_ = temperature_calibration
            model.repetition_evidence.set_evidence_calibration(
                pos_weight=temperature_calibration.pos_weight,
                train_prior=temperature_calibration.train_prior,
                temperature=temperature_calibration.temperature,
            )
            refit_history = trainer.refit_repetition_evidence(set_loader)
            history["repetition_temperature_calibration"] = {
                "pos_weight": temperature_calibration.pos_weight,
                "train_prior": temperature_calibration.train_prior,
                "offset": temperature_calibration.offset,
                "temperature": temperature_calibration.temperature,
                "validation_nll": temperature_calibration.validation_nll,
                "source": temperature_calibration.source,
                "n_samples": temperature_calibration.n_samples,
            }
            history["repetition_refit_losses"] = refit_history
            self.repetition_fitted_ = bool(refit_history)
            if (
                isinstance(model.repetition_evidence, AdditiveRepetitionEvidence)
                and model.repetition_evidence.state_residual is not None
            ):
                if audit_X is None or audit_y is None or audit_subject_ids is None:
                    raise ValueError(
                        "Repetition-v12 state-residual gate requires untouched audit trials."
                    )
                state_gate = self._gate_repetition_state_residual(
                    np.asarray(audit_X),
                    np.asarray(audit_y),
                    np.asarray(audit_subject_ids),
                    audit_digits,
                    audit_acquisition_indices,
                )
                history["repetition_state_residual_gate"] = state_gate
                if not bool(state_gate.get("passed", False)):
                    model.repetition_evidence.set_state_residual_gain(0.0)
            if epoch_progress_callback is not None and refit_history:
                main_epochs = len(history.get("train_losses", []))
                refit_limit = main_epochs + len(refit_history)
                for offset, loss in enumerate(refit_history, start=1):
                    epoch_progress_callback(
                        {
                            "epoch": main_epochs + offset,
                            "epoch_limit": refit_limit,
                            "train_loss": float(loss),
                            "task_val_loss": None,
                            "task_val_auc": None,
                            "objective_val_loss": None,
                            "val_innovation_nll": None,
                            "phase": "repetition_refit",
                            "selection_active": False,
                            "patience_left": None,
                            "early_stop_patience": int(cfg.early_stop_patience),
                            "best_epoch": history.get("best_epoch"),
                            "best_task_epoch": history.get("best_task_epoch"),
                            "best_density_epoch": history.get("best_density_epoch"),
                            "best_val_loss": history.get("best_task_val_loss"),
                            "best_task_val_loss": history.get("best_task_val_loss"),
                            "best_density_nll": history.get("best_density_nll"),
                            "task_patience_exhausted": history.get(
                                "task_patience_exhausted", False
                            ),
                            "will_early_stop": False,
                        }
                    )
            if isinstance(model.repetition_evidence, AdditiveRepetitionEvidence):
                reliability_audit = self._audit_reliability_v12(
                    X_val, y_val, val_subject_ids, val_trial_channel_mask
                )
                self.repetition_reliability_audit_ = reliability_audit
                history["repetition_reliability_audit"] = reliability_audit
                history["repetition_clean_probability_audit"] = reliability_audit[
                    "clean_probability"
                ]
                self.repetition_ready_ = self.repetition_fitted_ and bool(
                    reliability_audit["fidelity"]["passed"]
                )
            else:
                reliability_audit = self._audit_repetition_reliability(
                    X_val, y_val, val_subject_ids, val_trial_channel_mask
                )
                self.repetition_reliability_audit_ = reliability_audit
                history["repetition_reliability_audit"] = reliability_audit
                self.repetition_ready_ = self.repetition_fitted_ and bool(reliability_audit["passed"])
            if val_digits is not None and val_acquisition_indices is not None:
                history["validation_stopping_replay"] = self._replay_validation_stopping(
                    X_val,
                    y_val,
                    val_digits,
                    val_subject_ids,
                    val_acquisition_indices,
                    trial_channel_mask=val_trial_channel_mask,
                )
            if self.prequential_variant_ is not None:
                history["innovation_anomaly_audit"] = self._audit_innovation_anomaly(
                    X_val,
                    val_subject_ids,
                    self.prequential_variant_,
                    trial_channel_mask=val_trial_channel_mask,
                )
                history["innovation_e_process_audit"] = self._audit_prequential_e_process(
                    X_val,
                    y_val,
                    val_trial_channel_mask,
                )
        self.calibration_logits_ = None
        self.calibration_labels_ = None
        self.calibration_source_ = None
        self.calibration_branch_logits_ = None
        if X_val is not None and len(X_val) > 0:
            self.calibration_logits_ = self._predict_model_logit(
                X_val, trial_channel_mask=val_trial_channel_mask
            )
            self.calibration_labels_ = np.asarray(y_val, dtype=np.int64).copy()
            self.calibration_source_ = "subject_disjoint_validation"
            self.calibration_branch_logits_ = {
                key: np.asarray(value)
                for key, value in self.predict_branches(
                    X_val, trial_channel_mask=val_trial_channel_mask
                ).items()
                if key
                not in {
                    "prequential_coefficient",
                    "measurement_coefficient",
                    "prequential_base_slope",
                    "prequential_base_intercept",
                }
            }
        return self

    def _collect_prequential(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, CausalInnovationOutput, torch.Tensor]:
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        tensor = torch.from_numpy(X).to(self.device)
        mask_tensor = (
            torch.from_numpy(trial_channel_mask).to(self.device)
            if trial_channel_mask is not None
            else None
        )
        observations: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        diagonals: list[torch.Tensor] = []
        factors: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(tensor), 256):
                batch = tensor[start : start + 256]
                runtime_channel_mask = (
                    mask_tensor[start : start + 256]
                    if mask_tensor is not None
                    else self._model_channel_mask
                )
                output = self.model_(
                    batch,
                    E_chn=self._model_E_chn,
                    channel_mask=runtime_channel_mask,
                    domain_id=self._main_domain_ids(len(batch)),
                    likelihood_channel_mask=self.generative_profile_.channel_mask,
                    likelihood_class_means=self.generative_profile_.class_means,
                )
                if output.likelihood is None:
                    raise RuntimeError("Prequential audit requires model likelihood outputs.")
                observations.append(output.likelihood.likelihood_observation.float())
                masks.append(output.likelihood.observation_mask)
                innovation = output.likelihood.causal_innovation
                means.append(innovation.history_correction.float())
                diagonals.append(innovation.log_variance_scale.float())
                factors.append(innovation.factor_scale.float())
        return (
            torch.cat(observations),
            CausalInnovationOutput(
                history_correction=torch.cat(means),
                log_variance_scale=torch.cat(diagonals),
                factor_scale=torch.cat(factors),
            ),
            torch.cat(masks),
        )

    def _prequential_llr(
        self,
        X: np.ndarray,
        variant: str,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.generative_profile_ is None:
            raise RuntimeError("Prequential likelihood requires a fitted GenerativeProfile.")
        observation, innovation, observation_mask = self._collect_prequential(
            X, trial_channel_mask
        )
        with torch.inference_mode():
            llr = prequential_log_likelihood_ratio(
                observation,
                innovation,
                self.generative_profile_.to(self.device),
                variant=variant,
                observation_mask=observation_mask,
            )
        return llr.cpu().numpy().astype(np.float64)

    def _prequential_nll_pair(
        self,
        X: np.ndarray,
        variant: str,
        trial_channel_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(negative_nll, positive_nll)`` per trial."""
        if self.generative_profile_ is None:
            raise RuntimeError("Prequential likelihood requires a fitted GenerativeProfile.")
        observation, innovation, observation_mask = self._collect_prequential(
            X, trial_channel_mask
        )
        profile = self.generative_profile_.to(self.device)
        with torch.inference_mode():
            negative = prequential_score_per_trial(
                observation,
                innovation,
                profile,
                hypothesis=0,
                variant=variant,
                observation_mask=observation_mask,
            )
            positive = prequential_score_per_trial(
                observation,
                innovation,
                profile,
                hypothesis=1,
                variant=variant,
                observation_mask=observation_mask,
            )
        return (
            negative.nll_sum.cpu().numpy().astype(np.float64),
            positive.nll_sum.cpu().numpy().astype(np.float64),
        )

    def _audit_innovation_anomaly(
        self,
        X: np.ndarray,
        subject_ids: np.ndarray,
        variant: str,
        *,
        alpha: float = 0.05,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Two-hypothesis conformal anomaly flags on subject-disjoint validation."""
        nll0, nll1 = self._prequential_nll_pair(X, variant, trial_channel_mask)
        subject_ids = np.asarray(subject_ids)
        flags0 = np.zeros(len(X), dtype=bool)
        flags1 = np.zeros(len(X), dtype=bool)
        for subject in np.unique(subject_ids):
            held_out = subject_ids == subject
            fitting = ~held_out
            flags0[held_out] = two_hypothesis_conformal_flags(
                nll0[fitting], nll0[held_out], alpha=alpha
            )
            flags1[held_out] = two_hypothesis_conformal_flags(
                nll1[fitting], nll1[held_out], alpha=alpha
            )
        both_reject = flags0 & flags1
        return {
            "alpha": alpha,
            "out_of_model_fraction": float(np.mean(both_reject)),
            "n_subjects": int(len(np.unique(subject_ids))),
            "n_trials": int(len(X)),
        }

    def _audit_prequential_e_process(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Null-hypothesis e-process diagnostic from prequential LLR."""
        if self.prequential_variant_ is None:
            return {"available": False}
        llr = (
            self._prequential_llr(X, self.prequential_variant_)
            if trial_channel_mask is None
            else self._prequential_llr(X, self.prequential_variant_, trial_channel_mask)
        )
        null = np.asarray(y).reshape(-1) == 0
        if not null.any():
            return {"available": False, "failure": "no_null_trials"}
        report = e_process_diagnostics(llr[null])
        report["available"] = True
        return report

    def _audit_prequential(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        observation, innovation, observation_mask = self._collect_prequential(
            X, trial_channel_mask
        )
        _, groups = np.unique(subject_ids, return_inverse=True)
        report = audit_prequential_model(
            observation,
            innovation,
            self.generative_profile_.to(self.device),
            torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device),
            torch.from_numpy(groups.astype(np.int64)).to(self.device),
            observation_mask=observation_mask,
        )
        return report.to_dict()

    def _fit_prequential_fusion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray | None,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Cross-fit the likelihood fusion on held-out validation subjects.

        The density-family audit and this coefficient use disjoint subjects. Within
        validation, leave-one-subject-out predictions decide whether fusion is useful;
        the final coefficient is fit on all validation subjects only after that gate.
        """

        if self.prequential_variant_ is None:
            raise RuntimeError("Cannot calibrate a prequential branch before audit selection.")
        if subject_ids is None:
            raise ValueError("Prequential fusion requires validation subject IDs.")
        y = np.asarray(y, dtype=np.float32)
        subject_ids = np.asarray(subject_ids)
        if y.shape != (len(X),) or subject_ids.shape != (len(X),):
            raise ValueError("Fusion X, y and subject_ids must contain the same trials.")
        llr = (
            self._prequential_llr(X, self.prequential_variant_)
            if trial_channel_mask is None
            else self._prequential_llr(X, self.prequential_variant_, trial_channel_mask)
        )
        active_variant = self.prequential_variant_
        self.prequential_variant_ = None
        base = (
            self._predict_model_logit(X)
            if trial_channel_mask is None
            else self._predict_model_logit(X, trial_channel_mask=trial_channel_mask)
        ).astype(np.float32)
        self.prequential_variant_ = active_variant
        self.prequential_scale_ = _weighted_llr_scale(
            llr,
            subject_ids,
            device=self.device,
        )
        scaled = (llr / self.prequential_scale_).astype(np.float32)
        nested = fit_nested_fusion(
            base,
            scaled,
            y,
            subject_ids,
            n_bootstrap=400,
            seed=0,
            expected_sign="positive",
        )
        coefficient = float(nested["coefficient"])
        fused_nll = float(nested["fused_nll"])
        self.prequential_coefficient_ = coefficient
        self.prequential_base_slope_ = float(nested.get("base_slope", 1.0))
        self.prequential_base_intercept_ = float(nested.get("base_intercept", 0.0))
        return {
            "variant": active_variant,
            "coefficient": coefficient,
            "base_slope": self.prequential_base_slope_,
            "base_intercept": self.prequential_base_intercept_,
            "passed": bool(nested["passed"]),
            "fused_nll": fused_nll,
            "llr_scale": self.prequential_scale_,
            "llr_temperature": coefficient / self.prequential_scale_,
            "validation_base_nll": float(nested["base_nll"]),
            "validation_fused_nll": fused_nll,
            "crossfit_passed": bool(nested["passed"]),
            "crossfit_base_nll": float(nested["base_nll"]),
            "crossfit_fused_nll": fused_nll,
            "crossfit_subject_win_fraction": float(nested["strict_majority"]),
            "crossfit_coefficients": nested["loo_coefficients"],
            "c_ci": nested["c_ci"],
            "nll_improvement": nested["nll_improvement"],
            "auc_base": nested["auc_base"],
            "auc_fused": nested["auc_fused"],
            "auc_non_inferior": nested["auc_non_inferior"],
            "failure": nested.get("failure"),
            "source": "nested_cluster_bootstrap",
            "n_samples": int(len(X)),
            "n_subjects": int(nested["n_subjects"]),
        }

    def _select_prequential_fusion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray | None,
        trial_channel_mask: np.ndarray | None = None,
        *,
        eligible_variants: tuple[str, ...],
        density_selected_variant: str,
    ) -> dict[str, object]:
        """Select complementary evidence only among audit-eligible densities."""

        if not eligible_variants or density_selected_variant not in eligible_variants:
            raise ValueError("Fusion candidates must contain the audit density winner.")
        candidate_reports: dict[str, dict[str, object]] = {}
        for variant in eligible_variants:
            self.prequential_variant_ = variant
            candidate_reports[variant] = self._fit_prequential_fusion(
                X,
                y,
                subject_ids,
                trial_channel_mask,
            )
        selected = select_v12_fusion(
            candidate_reports,
            density_selected_variant=density_selected_variant,
        )
        self.prequential_variant_ = str(selected["variant"])
        self.prequential_scale_ = float(selected["llr_scale"])
        self.prequential_coefficient_ = float(selected["coefficient"])
        selected["llr_temperature"] = self.prequential_coefficient_ / self.prequential_scale_
        return selected

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
        digits: np.ndarray | None = None,
        acquisition_indices: np.ndarray | None = None,
        trial_channel_mask: np.ndarray | None = None,
    ) -> N2P3NetBaseline:
        X = _as_float32_epochs(X)
        y_raw = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
        if y_raw.shape != (len(X),):
            raise ValueError("y must be one-dimensional and aligned with X.")
        if not np.issubdtype(y_raw.dtype, np.integer) or set(
            np.unique(y_raw).tolist()
        ) != {0, 1}:
            raise ValueError("N2P3 binary training requires integer labels {0,1}.")
        y = y_raw.astype(np.int64, copy=False)
        expected_channels = self.model_kwargs.get("n_channels")
        expected_times = self.model_kwargs.get("n_time")
        if expected_channels is not None and X.shape[1] != int(expected_channels):
            raise ValueError(
                f"X has {X.shape[1]} channels; adapter requires {int(expected_channels)}."
            )
        if expected_times is not None and X.shape[2] != int(expected_times):
            raise ValueError(
                f"X has {X.shape[2]} samples; adapter requires {int(expected_times)}."
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        if subject_ids is None:
            self.last_val_subjects = None
            self.last_audit_subjects = None
            return self._fit_common(
                X,
                y,
                train_trial_channel_mask=trial_channel_mask,
                reconstruction_trial_channel_mask=trial_channel_mask,
            )
        subject_ids = np.asarray(subject_ids)
        if subject_ids.shape != (len(X),):
            raise ValueError("subject_ids must contain one value per trial.")
        group_ids = subject_ids if group_ids is None else np.asarray(group_ids)
        if digits is not None:
            digits = np.asarray(digits)
            if digits.shape != (len(X),) or not np.issubdtype(digits.dtype, np.integer):
                raise ValueError("digits must be a one-dimensional integer array aligned with X.")
            digits = digits.astype(np.int64, copy=False)
        if acquisition_indices is None:
            acquisition_indices = np.arange(len(X), dtype=np.int64)
        else:
            acquisition_indices = np.asarray(acquisition_indices)
            if not np.issubdtype(acquisition_indices.dtype, np.integer):
                raise ValueError("acquisition_indices must have an integer dtype.")
            acquisition_indices = acquisition_indices.astype(np.int64, copy=False)
        if acquisition_indices.shape != (len(X),):
            raise ValueError("acquisition_indices must contain one value per trial.")
        if group_ids.shape != (len(X),):
            raise ValueError("group_ids must contain one selection group per trial.")
        split = self._subject_validation_split(subject_ids, digits=digits)
        self.last_val_subjects = split.n_validation_subjects
        complete_audit_subjects = [
            subject
            for subject in np.unique(subject_ids[split.train_mask])
            if np.unique(y[(subject_ids == subject) & split.train_mask]).size == 2
        ]
        audit_split = subject_disjoint_audit_split(
            subject_ids,
            eligible_mask=split.train_mask,
            candidate_mask=np.isin(subject_ids, complete_audit_subjects),
            n_subjects=self.audit_subjects,
            seed=int(self.trainer_kwargs.get("seed", 0)),
        )
        self.last_audit_subjects = audit_split.n_audit_subjects
        optimization_mask = audit_split.optimization_mask
        return self._fit_common(
            X[optimization_mask],
            y[optimization_mask],
            X[split.validation_mask],
            y[split.validation_mask],
            train_subject_ids=subject_ids[optimization_mask],
            train_group_ids=group_ids[optimization_mask],
            train_digits=digits[optimization_mask] if digits is not None else None,
            train_acquisition_indices=acquisition_indices[optimization_mask],
            train_trial_channel_mask=(
                trial_channel_mask[optimization_mask]
                if trial_channel_mask is not None
                else None
            ),
            val_subject_ids=subject_ids[split.validation_mask],
            val_group_ids=group_ids[split.validation_mask],
            val_digits=digits[split.validation_mask] if digits is not None else None,
            val_acquisition_indices=acquisition_indices[split.validation_mask],
            val_trial_channel_mask=(
                trial_channel_mask[split.validation_mask]
                if trial_channel_mask is not None
                else None
            ),
            # The ERP profile is supervised (it uses target/non-target labels),
            # so inner-validation subjects must not influence the training loss
            # or the fold-local likelihood profile.
            reconstruction_X=X[optimization_mask],
            reconstruction_y=y[optimization_mask],
            reconstruction_trial_channel_mask=(
                trial_channel_mask[optimization_mask]
                if trial_channel_mask is not None
                else None
            ),
            audit_X=X[audit_split.audit_mask],
            audit_y=y[audit_split.audit_mask],
            audit_subject_ids=subject_ids[audit_split.audit_mask],
            audit_group_ids=group_ids[audit_split.audit_mask],
            audit_digits=digits[audit_split.audit_mask] if digits is not None else None,
            audit_acquisition_indices=acquisition_indices[audit_split.audit_mask],
            audit_trial_channel_mask=(
                trial_channel_mask[audit_split.audit_mask]
                if trial_channel_mask is not None
                else None
            ),
        )

    def predict_logit(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_logit。")
        X = _as_float32_epochs(X)
        if X.ndim != 3:
            raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
        expected_channels = self.model_kwargs.get("n_channels")
        expected_times = self.model_kwargs.get("n_time")
        if expected_channels is not None and X.shape[1] != int(expected_channels):
            raise ValueError(
                f"X has {X.shape[1]} channels; adapter requires {int(expected_channels)}."
            )
        if expected_times is not None and X.shape[2] != int(expected_times):
            raise ValueError(
                f"X has {X.shape[2]} samples; adapter requires {int(expected_times)}."
            )
        if len(X) == 0:
            raise ValueError("predict_logit requires at least one trial.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)

        return self._predict_model_logit(X, trial_channel_mask=trial_channel_mask)

    def predict_epoch_trajectory_logits(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        """Return raw-checkpoint logits without changing the selected final model."""

        if not self._fitted:
            raise RuntimeError("请先 fit 再评估 epoch trajectory。")
        if not bool(self.trainer_kwargs.get("epoch_trajectory_audit", False)):
            return []
        if not self.epoch_trajectory_checkpoints_:
            raise RuntimeError("Epoch trajectory audit was enabled but no checkpoints were saved.")
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        final_state = {
            key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()
        }
        rows: list[dict[str, object]] = []
        try:
            for reference in self.epoch_trajectory_checkpoints_:
                if isinstance(reference, Path):
                    payload = torch.load(reference, map_location="cpu", weights_only=True)
                    checkpoint = str(reference)
                else:
                    payload = reference
                    checkpoint = "in_memory"
                if payload.get("schema") != "n2p3net_epoch_trajectory_checkpoint/1":
                    raise ValueError("Unsupported epoch trajectory checkpoint schema.")
                self.model_.load_state_dict(payload["state_dict"], strict=True)
                logits = (
                    self._predict_model_logit(X)
                    if trial_channel_mask is None
                    else self._predict_model_logit(
                        X, trial_channel_mask=trial_channel_mask
                    )
                )
                rows.append(
                    {
                        **dict(payload["event"]),
                        "checkpoint": checkpoint,
                        "logits": logits,
                    }
                )
        finally:
            self.model_.load_state_dict(final_state, strict=True)
        return rows

    # ------------------------------------------------------------------
    # Object L: fold-local latency measurement and gated PCW consumption.
    # ------------------------------------------------------------------
    def _measurement_enabled(self) -> bool:
        return bool(self.model_kwargs.get("use_measurement_windows", False))

    def _measurement_estimator(self) -> LatencyMeasurement:
        if self.measurement_estimator_ is not None:
            return self.measurement_estimator_
        model = self.model_
        anchor = float(self.model_kwargs.get("measurement_anchor_ms", 460.0))
        tmin = float(model.tmin_ms) if hasattr(model, "tmin_ms") else float(self.model_kwargs.get("tmin_ms", -200.0))
        tmax = float(model.tmax_ms) if hasattr(model, "tmax_ms") else float(self.model_kwargs.get("tmax_ms", 1200.0))
        sfreq = float(model.sfreq) if hasattr(model, "sfreq") else float(self.model_kwargs.get("sfreq", 256.0))
        n_time = int(model.n_time) if hasattr(model, "n_time") else int(self.model_kwargs.get("n_time", 358))
        time_ms = tmin + np.arange(n_time, dtype=np.float64) * (tmax - tmin) / n_time
        self.measurement_estimator_ = LatencyMeasurement(
            anchor_tau0_ms=anchor,
            sfreq=sfreq,
            time_ms=time_ms,
            grid_radius_ms=float(self.model_kwargs.get("measurement_grid_radius_ms", 60.0)),
            grid_step_ms=float(self.model_kwargs.get("measurement_grid_step_ms", 0.5)),
        )
        return self.measurement_estimator_

    def predict_measurement(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> LatencyPosterior | None:
        """Return the fold-fitted latency posterior for ``X``.

        Evaluation callers use this instead of reading
        ``measurement_posterior_``, which may be pinned to a larger training
        or validation array by the per-fold cache.
        """
        if not self._measurement_enabled() or self.model_.measurement_head is None:
            return None
        _ = self._measurement_windows(X, trial_channel_mask=trial_channel_mask)
        key = self._measurement_fingerprint(X, trial_channel_mask)
        if key == self._measurement_cache_key_:
            return self._measurement_cache_posterior_
        return None

    @staticmethod
    def _measurement_fingerprint(
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> str:
        values = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        digest = hashlib.blake2b(digest_size=16)
        digest.update(values.tobytes())
        if trial_channel_mask is not None:
            digest.update(np.ascontiguousarray(trial_channel_mask, dtype=np.bool_).tobytes())
        return digest.hexdigest()

    def _measurement_windows(
        self,
        X: np.ndarray,
        *,
        trial_channel_mask: np.ndarray | None = None,
    ) -> torch.Tensor | None:
        if not self._measurement_enabled() or self.model_.measurement_head is None:
            return None
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        estimator = self.measurement_estimator_
        if estimator is None or not estimator.fitted:
            return None
        measurement_mask = self._measurement_channel_mask_
        if measurement_mask is None:
            measurement_mask = np.ones(X.shape[1], dtype=bool)
        if trial_channel_mask is not None and not bool(
            trial_channel_mask[:, measurement_mask].all()
        ):
            # The covariance lives in one fixed sensor space. A runtime trial
            # missing one of those sensors cannot be scored as if its zero fill
            # were a physical observation; disable only the gated L branch.
            return None
        key = self._measurement_fingerprint(X, trial_channel_mask)
        if (
            key == self._measurement_cache_key_
            and self._measurement_cache_window_ is not None
        ):
            return self._measurement_cache_window_
        posterior = estimator.predict(np.asarray(X[:, measurement_mask], dtype=np.float64))
        width_ms = float(self.model_kwargs.get("measurement_window_width_ms", 50.0))
        window = torch.from_numpy(
            detached_expected_window(
                posterior, time_ms=estimator.time_ms, width_ms=width_ms
            ).astype(np.float32)
        )
        self._measurement_cache_key_ = key
        self._measurement_cache_posterior_ = posterior
        self._measurement_cache_window_ = window
        # Keep the *fold-level* posterior stable: per-subject replay slices
        # must not overwrite a larger posterior with a smaller one.
        if int(len(X)) >= self._measurement_posterior_n_:
            self.measurement_posterior_ = posterior
            self._measurement_posterior_n_ = int(len(X))
        return window

    def _predict_measurement_evidence(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the detached measurement contribution with gain=1."""
        if self.model_.measurement_head is None:
            return np.zeros(len(X), dtype=np.float64)
        windows = self._measurement_windows(X, trial_channel_mask=trial_channel_mask)
        if windows is None:
            return np.zeros(len(X), dtype=np.float64)
        original_gain = float(self.model_.measurement_gain.detach().cpu())
        self.model_.measurement_gain.fill_(1.0)
        try:
            return self._predict_model_logit(
                X,
                include_prequential=False,
                measurement_only=True,
                trial_channel_mask=trial_channel_mask,
            )
        finally:
            self.model_.measurement_gain.fill_(original_gain)

    def _fit_measurement_branch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        train_subject_ids: np.ndarray | None,
        val_subject_ids: np.ndarray | None,
        *,
        train_trial_channel_mask: np.ndarray | None = None,
        val_trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Fit the detached latency posterior on optimization subjects, refit the
        PCW measurement consumer, then gate it with nested M0/M1 on validation.

        The measurement window is NumPy/detached by construction; this method
        never lets classification gradients enter LatencyMeasurement.
        """
        self.model_.measurement_gain.fill_(0.0)
        self.measurement_estimator_ = None
        self.measurement_posterior_ = None
        self.measurement_gain_ = 0.0
        self._measurement_cache_key_ = None
        self._measurement_cache_posterior_ = None
        self._measurement_cache_window_ = None
        self._measurement_posterior_n_ = -1
        self._measurement_channel_mask_ = None
        self.clean_probability_pool_ = None
        self.measurement_gate_ = None
        if not self._measurement_enabled() or self.model_.measurement_head is None:
            return {"available": False, "failure": "measurement_windows_disabled"}
        if (
            X_val is None
            or y_val is None
            or val_subject_ids is None
            or train_subject_ids is None
            or len(X_val) == 0
        ):
            return {
                "available": True,
                "passed": False,
                "failure": "measurement_branch_requires_optimization_and_validation_subjects",
            }
        static_mask = (
            self.channel_mask.detach().cpu().numpy().astype(bool)
            if self.channel_mask is not None
            else np.ones(X_train.shape[1], dtype=bool)
        )
        train_masks = (
            np.broadcast_to(static_mask, X_train.shape[:2])
            if train_trial_channel_mask is None
            else np.asarray(train_trial_channel_mask, dtype=bool) & static_mask[None]
        )
        val_masks = (
            np.broadcast_to(static_mask, X_val.shape[:2])
            if val_trial_channel_mask is None
            else np.asarray(val_trial_channel_mask, dtype=bool) & static_mask[None]
        )
        measurement_mask = train_masks[0]
        if not bool((train_masks == measurement_mask[None]).all()) or not bool(
            (val_masks == measurement_mask[None]).all()
        ):
            return {
                "available": True,
                "passed": False,
                "failure": "measurement_branch_requires_homogeneous_channel_masks",
            }
        self._measurement_channel_mask_ = measurement_mask.copy()
        try:
            self._measurement_estimator().fit(
                np.asarray(X_train[:, measurement_mask], dtype=np.float64),
                np.asarray(y_train, dtype=np.int64),
                np.asarray(train_subject_ids),
            )
        except Exception as exc:  # fail closed, never let L block the fold
            return {"available": True, "passed": False, "failure": f"measurement_fit_failed:{type(exc).__name__}"}

        train_windows = self._measurement_windows(
            X_train, trial_channel_mask=train_trial_channel_mask
        )
        val_windows = self._measurement_windows(X_val, trial_channel_mask=val_trial_channel_mask)
        if train_windows is None or val_windows is None:
            return {"available": True, "passed": False, "failure": "measurement_windows_unavailable"}

        model = self.model_
        requires_grad = {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.measurement_head.parameters():
            parameter.requires_grad_(True)

        epochs = max(1, int(self.model_kwargs.get("measurement_refit_epochs", 5)))
        pos_weight = float(self.last_pos_weight or 1.0)
        optimizer = torch.optim.Adam(model.measurement_head.parameters(), lr=1e-3)

        model.measurement_gain.fill_(1.0)
        refit_losses: list[float] = []
        try:
            model.train()
            Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(self.device)
            yt = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1).to(self.device)
            wt = train_windows.to(self.device)
            for _ in range(epochs):
                total = torch.zeros((), device=self.device, dtype=torch.float32)
                count = 0
                for start in range(0, len(Xt), 256):
                    xb = Xt[start : start + 256]
                    wb = wt[start : start + 256]
                    yb = yt[start : start + 256]
                    out = model(
                        xb,
                        E_chn=self._model_E_chn,
                        channel_mask=(
                            torch.from_numpy(train_trial_channel_mask[start : start + 256]).to(
                                self.device
                            )
                            if train_trial_channel_mask is not None
                            else self._model_channel_mask
                        ),
                        domain_id=self._main_domain_ids(len(xb)),
                        return_attention=False,
                        return_likelihood=False,
                        measurement_window=wb,
                    )
                    if out.heads is None:
                        raise RuntimeError("Measurement refit requires model heads.")
                    loss = F.binary_cross_entropy_with_logits(
                        out.heads.logit_target, yb, pos_weight=torch.tensor(pos_weight, device=self.device)
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += loss.detach().float()
                    count += 1
                if count == 0:
                    raise RuntimeError("Measurement refit received no batches.")
                refit_losses.append(float((total / count).cpu()))
        except Exception as exc:
            self.measurement_gain_ = 0.0
            model.measurement_gain.fill_(0.0)
            return {
                "available": True,
                "passed": False,
                "coefficient": 0.0,
                "failure": f"measurement_refit_failed:{type(exc).__name__}",
                "refit_losses": refit_losses,
            }
        finally:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(requires_grad[name])
            model.measurement_gain.fill_(0.0)
            model.eval()

        try:
            base = self._predict_model_logit(
                np.asarray(X_val, dtype=np.float32),
                trial_channel_mask=val_trial_channel_mask,
            )
            evidence = self._predict_measurement_evidence(
                np.asarray(X_val, dtype=np.float32),
                trial_channel_mask=val_trial_channel_mask,
            )
            nested = fit_nested_fusion(
                base,
                evidence,
                np.asarray(y_val, dtype=np.float64),
                np.asarray(val_subject_ids),
                n_bootstrap=400,
                seed=0,
                expected_sign="positive",
            )
        except Exception as exc:
            nested = {
                "passed": False,
                "coefficient": 0.0,
                "failure": f"measurement_nested_fusion_failed:{type(exc).__name__}",
                "base_nll": float("nan"),
                "fused_nll": float("nan"),
                "nll_improvement": 0.0,
                "strict_majority": False,
                "c_ci": [float("nan"), float("nan")],
                "n_subjects": 0,
            }

        coefficient = float(nested.get("coefficient", 0.0)) if nested.get("passed") else 0.0
        self.measurement_gain_ = coefficient
        model.measurement_gain.fill_(coefficient)
        gate = {
            "available": True,
            "passed": bool(nested.get("passed", False)),
            "coefficient": coefficient,
            "refit_losses": refit_losses,
            "nested": nested,
            "n_train_trials": int(len(X_train)),
            "n_validation_trials": int(len(X_val)),
            "posterior_entropy_mean": float(self.measurement_posterior_.entropy.mean())
            if self.measurement_posterior_ is not None
            else float("nan"),
        }
        self.measurement_gate_ = gate
        return gate

    def _measurement_forward_extra(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Return window/posterior kwargs for model forwards on ``X``."""
        if not self._measurement_enabled() or self.model_.measurement_head is None:
            return {"measurement_window": None, "measurement_posterior": None}
        return {
            "measurement_window": self._measurement_windows(
                X, trial_channel_mask=trial_channel_mask
            ),
            "measurement_posterior": self.measurement_posterior_,
        }

    def _predict_model_logit(
        self,
        X: np.ndarray,
        *,
        include_prequential: bool = True,
        measurement_only: bool = False,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward already-preprocessed arrays without adapter-level transforms.

        The detached latency-posterior window is consumed here when the L
        object is enabled and its nested gate has set ``measurement_gain``.
        ``measurement_only=True`` returns the measurement contribution with
        the caller-supplied gain, used by the nested M0/M1 measurement gate.
        """
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        Xt = torch.from_numpy(X).to(self.device)
        mask_tensor = None
        if trial_channel_mask is not None:
            mask_tensor = torch.from_numpy(trial_channel_mask).to(self.device)
            if mask_tensor.shape != (len(X), X.shape[1]):
                raise ValueError("trial_channel_mask must have shape (N,C) matching X.")
        windows = self._measurement_windows(X, trial_channel_mask=trial_channel_mask)
        if windows is not None:
            windows = windows.to(self.device)
        chunk_size = 256
        self.model_.eval()
        out: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, Xt.shape[0], chunk_size):
                xb = Xt[i : i + chunk_size]
                wb = windows[i : i + chunk_size] if windows is not None else None
                domain_id = self._main_domain_ids(len(xb))
                runtime_channel_mask = (
                    mask_tensor[i : i + chunk_size]
                    if mask_tensor is not None
                    else self._model_channel_mask
                )
                output = self.model_(
                    xb,
                    E_chn=self._model_E_chn,
                    channel_mask=runtime_channel_mask,
                    domain_id=domain_id,
                    return_likelihood=self.prequential_variant_ is not None,
                    likelihood_class_means=(
                        self.generative_profile_.class_means
                        if self.generative_profile_ is not None
                        else None
                    ),
                    likelihood_channel_mask=(
                        self.generative_profile_.channel_mask
                        if self.generative_profile_ is not None
                        else None
                    ),
                    measurement_window=wb,
                    measurement_posterior=self.measurement_posterior_,
                )
                if output.heads is None:
                    raise RuntimeError("模型 forward 未返回 heads；请勿设置 return_heads=False。")
                if measurement_only:
                    value = (
                        output.measurement_contribution
                        if output.measurement_contribution is not None
                        else torch.zeros_like(output.heads.logit_target)
                    )
                    out.append(value.float().squeeze(-1).cpu())
                    continue
                logit = output.heads.logit_target.float().squeeze(-1)
                if (
                    include_prequential
                    and self.prequential_variant_ is not None
                    and float(self.prequential_coefficient_) != 0.0
                    and self.generative_profile_ is not None
                ):
                    if output.likelihood is None:
                        raise RuntimeError("Prequential fusion requires likelihood outputs.")
                    llr = prequential_log_likelihood_ratio(
                        output.likelihood.likelihood_observation,
                        output.likelihood.causal_innovation,
                        self.generative_profile_.to(self.device),
                        variant=self.prequential_variant_,
                        observation_mask=output.likelihood.observation_mask,
                    )
                    scaled_llr = llr / self.prequential_scale_
                    logit = (
                        float(self.prequential_base_intercept_)
                        + float(self.prequential_base_slope_) * logit
                        + self.prequential_coefficient_ * scaled_llr
                    )
                out.append(logit[:, None].cpu())
        return torch.cat(out).squeeze(-1).numpy().astype(np.float64)

    def _main_domain_ids(self, batch_size: int) -> torch.Tensor | None:
        if getattr(self.model_, "n_domains", None) is None:
            return None
        return torch.zeros(batch_size, device=self.device, dtype=torch.long)

    def _predict_repetition_inputs(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence_model = self.model_.repetition_evidence
        if evidence_model is None:
            raise RuntimeError("The fitted model has no repetition evidence head.")
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        Xt = torch.from_numpy(X).to(self.device)
        mask_tensor = None
        if trial_channel_mask is not None:
            mask_tensor = torch.from_numpy(trial_channel_mask).to(self.device)
        windows = self._measurement_windows(X, trial_channel_mask=trial_channel_mask)
        if windows is not None:
            windows = windows.to(self.device)
        logits_out: list[torch.Tensor] = []
        evidence_out: list[torch.Tensor] = []
        quality_out: list[torch.Tensor] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, Xt.shape[0], 256):
                xb = Xt[start : start + 256]
                wb = windows[start : start + 256] if windows is not None else None
                runtime_channel_mask = (
                    mask_tensor[start : start + 256]
                    if mask_tensor is not None
                    else self._model_channel_mask
                )
                output = self.model_(
                    xb,
                    E_chn=self._model_E_chn,
                    channel_mask=runtime_channel_mask,
                    domain_id=self._main_domain_ids(len(xb)),
                    return_attention=False,
                    return_likelihood=False,
                    measurement_window=wb,
                    measurement_posterior=self.measurement_posterior_,
                )
                if output.heads is None:
                    raise RuntimeError("Repetition inference requires final trial logits.")
                quality = extract_quality_features(
                    xb,
                    sfreq=float(self.model_.sfreq),
                    baseline_n=int(self.model_.baseline_n),
                    reference_slice=getattr(self.model_, "trial_reference_slice", None),
                    channel_mask=runtime_channel_mask,
                )
                logits = output.heads.logit_target.reshape(-1).float()
                logits_out.append(logits)
                evidence_out.append(evidence_model.correct_evidence(logits))
                quality_out.append(quality.float())
        return torch.cat(logits_out), torch.cat(evidence_out), torch.cat(quality_out)

    def _audit_repetition_reliability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray | None = None,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Gate chain inference using all subject-disjoint validation trials.

        The audit measures calibration against four predefined synthetic
        corruptions. It deliberately does not claim calibration to unlabelled
        real-world artifacts.
        """

        n = len(X)
        if subject_ids is None:
            subject_ids = np.zeros(n, dtype=np.int64)
        subject_ids = np.asarray(subject_ids)
        if subject_ids.shape != (n,):
            raise ValueError("reliability-audit subject_ids must align with X.")
        if n < 8:
            return {
                "passed": False,
                "scope": "held_out_synthetic_corruption_only",
                "n_samples": n,
                "failure": "needs_at_least_eight_validation_trials",
                "checks": {"minimum_validation_trials": False},
                "failed_checks": ["minimum_validation_trials"],
                "thresholds": dict(REPETITION_RELIABILITY_THRESHOLDS),
            }
        clean = np.asarray(X, dtype=np.float32).copy()
        def preserve_observed(values: np.ndarray) -> np.ndarray:
            if trial_channel_mask is not None:
                values = values.copy()
                values[~trial_channel_mask] = 0.0
            return values

        scale = clean.std(axis=2, keepdims=True).clip(min=1e-3)
        time = np.linspace(-1.0, 1.0, clean.shape[2], dtype=np.float32)
        alternating = np.where(np.arange(clean.shape[2]) % 2, 1.0, -1.0).astype(np.float32)
        high_frequency = preserve_observed(clean + 3.0 * scale * alternating[None, None, :])
        drift = preserve_observed(clean + 4.0 * scale * time[None, None, :])
        flatline = clean.copy()
        flatline[:, 0] = 0.0
        spike = clean.copy()
        spike[:, :, clean.shape[2] // 2] += 12.0 * scale[:, :, 0]
        spike = preserve_observed(spike)
        corruptions = (high_frequency, drift, flatline, spike)
        _, _, clean_quality = self._predict_repetition_inputs(
            clean, trial_channel_mask=trial_channel_mask
        )
        corrupt_quality = [
            self._predict_repetition_inputs(
                corruption, trial_channel_mask=trial_channel_mask
            )[2]
            for corruption in corruptions
        ]
        evidence_model = self.model_.repetition_evidence
        with torch.inference_mode():
            clean_rho = evidence_model.reliability(clean_quality).float().cpu().numpy()
            corrupt_rho = np.stack(
                [
                    evidence_model.reliability(quality).float().cpu().numpy()
                    for quality in corrupt_quality
                ]
            )
        return _reliability_gate_metrics(
            clean_rho,
            corrupt_rho,
            np.asarray(y, dtype=np.int64),
            subject_ids,
        )

    def set_clean_probability_pool(
        self,
        X: np.ndarray,
        clean_labels: np.ndarray,
        subject_ids: np.ndarray,
        *,
        calibration_prior: float,
        deployment_prior: float,
        stimulus_digits: np.ndarray | None = None,
        true_digits: np.ndarray | None = None,
        acquisition_indices: np.ndarray | None = None,
    ) -> None:
        """Register a hard clean/artifact pool for the Q clean_probability path.

        ``clean_labels`` is 1 for clean and 0 for artifact. The digit-chain
        gate needs ``stimulus_digits``, ``true_digits`` and acquisition order;
        without them the probability gate fails closed with a named failure
        instead of silently skipping the pre-registered chain requirement.
        """
        X = _as_float32_epochs(X)
        clean_labels = np.asarray(clean_labels, dtype=np.int64)
        subject_ids = np.asarray(subject_ids)
        if X.ndim != 3 or len(X) != len(clean_labels) or len(X) != len(subject_ids):
            raise ValueError("clean probability pool X/labels/subject_ids must align.")
        if not ((clean_labels == 0) | (clean_labels == 1)).all():
            raise ValueError("clean_labels must be binary.")
        if np.unique(clean_labels).size != 2:
            raise ValueError("clean probability pool needs clean and artifact rows.")
        if not 0.0 < calibration_prior < 1.0 or not 0.0 < deployment_prior < 1.0:
            raise ValueError("priors must lie strictly inside (0,1).")
        self.clean_probability_pool_ = {
            "X": X,
            "clean_labels": clean_labels,
            "subject_ids": subject_ids,
            "calibration_prior": float(calibration_prior),
            "deployment_prior": float(deployment_prior),
            "stimulus_digits": (
                None if stimulus_digits is None else np.asarray(stimulus_digits, dtype=np.int64)
            ),
            "true_digits": (
                None if true_digits is None else np.asarray(true_digits, dtype=np.int64)
            ),
            "acquisition_indices": (
                None
                if acquisition_indices is None
                else np.asarray(acquisition_indices, dtype=np.int64)
            ),
        }

    def _evaluate_clean_probability_pool(self) -> dict[str, object]:
        """Fit/calibrate/evaluate the hard-label clean probability path."""
        pool = self.clean_probability_pool_
        if pool is None:
            return {
                "available": False,
                "passed": False,
                "failure": "no_hard_clean_artifact_pool_registered",
            }
        X = pool["X"]
        labels = pool["clean_labels"]
        subject_ids = pool["subject_ids"]
        subjects = np.unique(subject_ids)
        if len(subjects) < 3:
            return {
                "available": True,
                "passed": False,
                "failure": "clean_probability_requires_three_subject_splits",
            }
        fit_subjects = subjects[: len(subjects) // 3]
        cal_subjects = subjects[len(subjects) // 3 : 2 * len(subjects) // 3]
        test_subjects = subjects[2 * len(subjects) // 3 :]
        fit_mask = np.isin(subject_ids, fit_subjects)
        cal_mask = np.isin(subject_ids, cal_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        if not (fit_mask.any() and cal_mask.any() and test_mask.any()):
            return {
                "available": True,
                "passed": False,
                "failure": "clean_probability_split_has_empty_part",
            }
        _, _, quality = self._predict_repetition_inputs(X)
        quality = quality.cpu().numpy().astype(np.float64)
        estimator = CleanProbabilityEstimator().fit(quality[fit_mask], labels[fit_mask])
        estimator.fit_calibrator(
            quality[cal_mask],
            labels[cal_mask],
            calibration_prior=float(pool["calibration_prior"]),
        )
        calibrated = estimator.predict_calibrated(quality[test_mask])
        converted = convert_prior_odds(
            calibrated,
            calibration_prior=float(pool["calibration_prior"]),
            deployment_prior=float(pool["deployment_prior"]),
        )

        chain_scores = None
        true_candidates = None
        digits = pool.get("stimulus_digits")
        true_digits = pool.get("true_digits")
        acquisition = pool.get("acquisition_indices")
        evidence_model = self.model_.repetition_evidence
        if (
            digits is not None
            and true_digits is not None
            and evidence_model is not None
        ):
            if acquisition is None:
                acquisition = np.arange(len(X), dtype=np.int64)
            digit_vocab = tuple(range(1, 10))
            trajectories: list[np.ndarray] = []
            true_idx: list[int] = []
            for subject in test_subjects:
                rows = np.flatnonzero(subject_ids == subject)
                rows = rows[np.argsort(acquisition[rows], kind="stable")]
                if not rows.size:
                    continue
                _, evidence, subject_quality = self._predict_repetition_inputs(X[rows])
                rho = convert_prior_odds(
                    estimator.predict_calibrated(
                        self._predict_repetition_inputs(X[rows])[2].cpu().numpy().astype(np.float64)
                    ),
                    calibration_prior=float(pool["calibration_prior"]),
                    deployment_prior=float(pool["deployment_prior"]),
                )
                scores = evidence_model.candidate_chain_scores_with_clean_probability(
                    evidence,
                    subject_quality,
                    torch.as_tensor(digits[rows], device=evidence.device, dtype=torch.long),
                    torch.as_tensor(rho, device=evidence.device, dtype=evidence.dtype),
                    digit_vocab=digit_vocab,
                )
                trajectories.append(scores.detach().cpu().numpy())
                true_idx.append(digit_vocab.index(int(true_digits[rows[0]])))
            if trajectories:
                chain_scores = np.stack(trajectories)
                true_candidates = np.asarray(true_idx)

        report = evaluate_clean_probability_gate(
            converted,
            labels[test_mask],
            chain_scores=chain_scores,
            true_candidates=true_candidates,
            reference_chain_scores=None,
            chain_required=True,
        )
        report["available"] = True
        report["prior_shift_applied"] = True
        report["calibration_prior"] = pool["calibration_prior"]
        report["deployment_prior"] = pool["deployment_prior"]
        report["n_pool_rows"] = int(len(X))
        report["n_test_rows"] = int(test_mask.sum())
        return report

    def _audit_reliability_v12(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray | None,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, object]:
        """v12 Q-object audit: fidelity is a rank gate, clean_probability is opt-in.

        The four synthetic corruption mechanisms are unseen by the fidelity
        estimator's training objective and therefore form the held-out
        corruption-type gate. Clean probability is never emitted without hard
        real artifact labels.
        """
        if subject_ids is None:
            subject_ids = np.zeros(len(X), dtype=np.int64)
        evidence_model = self.model_.repetition_evidence
        if not isinstance(evidence_model, AdditiveRepetitionEvidence):
            return {"fidelity": {"passed": False}, "clean_probability": {"available": False}}

        clean = np.asarray(X, dtype=np.float32).copy()
        def preserve_observed(values: np.ndarray) -> np.ndarray:
            if trial_channel_mask is not None:
                values = values.copy()
                values[~trial_channel_mask] = 0.0
            return values

        scale = clean.std(axis=2, keepdims=True).clip(min=1e-3)
        time = np.linspace(-1.0, 1.0, clean.shape[2], dtype=np.float32)
        alternating = np.where(np.arange(clean.shape[2]) % 2, 1.0, -1.0).astype(np.float32)
        corruptions = {
            "high_frequency": preserve_observed(
                clean + 3.0 * scale * alternating[None, None, :]
            ),
            "drift": preserve_observed(clean + 4.0 * scale * time[None, None, :]),
            "flatline": clean.copy(),
            "spike": clean.copy(),
        }
        corruptions["flatline"][:, 0] = 0.0
        corruptions["spike"][:, :, clean.shape[2] // 2] += 12.0 * scale[:, :, 0]
        corruptions["spike"] = preserve_observed(corruptions["spike"])

        _, _, clean_quality = self._predict_repetition_inputs(
            clean, trial_channel_mask=trial_channel_mask
        )
        corrupt_quality = {
            name: self._predict_repetition_inputs(
                corruption, trial_channel_mask=trial_channel_mask
            )[2]
            for name, corruption in corruptions.items()
        }
        n_clean = len(clean_quality)
        quality_parts = [clean_quality]
        label_parts = [np.ones(n_clean, dtype=np.int64)]
        type_parts = [np.full(n_clean, "clean", dtype=object)]
        subject_parts = [np.asarray(subject_ids)]
        for name, quality in corrupt_quality.items():
            quality_parts.append(quality)
            label_parts.append(np.zeros(len(quality), dtype=np.int64))
            type_parts.append(np.full(len(quality), name, dtype=object))
            subject_parts.append(np.asarray(subject_ids))

        fidelity_report = evaluate_fidelity_gate(
            evidence_model.backbone.fidelity_estimator,
            torch.cat(quality_parts, dim=0),
            np.concatenate(label_parts),
            np.concatenate(type_parts),
            np.concatenate(subject_parts),
            unseen_types=set(corruptions),
        )
        fidelity_dict = dict(fidelity_report.__dict__)
        with torch.inference_mode():
            clean_fidelity = (
                evidence_model.backbone.fidelity_estimator(clean_quality)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        target_labels = np.asarray(y, dtype=np.int64).reshape(-1)
        target_auc = float("nan")
        target_ks = float("nan")
        if np.unique(target_labels).size == 2:
            target_auc = float(roc_auc_score(target_labels, clean_fidelity))
            target = clean_fidelity[target_labels == 1]
            nontarget = clean_fidelity[target_labels == 0]
            target_ks = _two_sample_ks(target, nontarget)
        target_independence_passed = bool(
            np.isfinite(target_auc)
            and abs(target_auc - 0.5) <= 0.15
            and np.isfinite(target_ks)
            and target_ks <= 0.25
        )
        fidelity_dict["target_auc"] = target_auc
        fidelity_dict["target_ks"] = target_ks
        fidelity_dict["target_independence_passed"] = target_independence_passed
        fidelity_dict["passed"] = bool(fidelity_report.passed and target_independence_passed)
        clean_probability = self._evaluate_clean_probability_pool()
        return {
            "fidelity": fidelity_dict,
            "clean_probability": clean_probability,
        }



    def _replay_validation_stopping(
        self,
        X: np.ndarray,
        y: np.ndarray,
        digits: np.ndarray,
        subject_ids: np.ndarray,
        acquisition_indices: np.ndarray | None,
        *,
        trial_channel_mask: np.ndarray | None = None,
        thresholds: tuple[float, ...] = (0.8, 0.9, 0.95),
        clean_reject_fraction: float = 0.05,
    ) -> dict[str, object]:
        """Replay a fixed posterior-crossing rule on complete validation sequences.

        The replay now includes the Q clean-reject gate: the worst
        ``clean_reject_fraction`` of validation flashes by fidelity are masked
        out of the posterior. Risk-coverage curves are reported for both the
        gated and ungated rules; all metrics are descriptive only.
        """
        evidence_model = self.model_.repetition_evidence
        if evidence_model is None or digits is None:
            return {}
        if acquisition_indices is None:
            acquisition_indices = np.arange(len(X), dtype=np.int64)
        trajectories_list: list[np.ndarray] = []
        fidelity_list: list[np.ndarray] = []
        true_candidates: list[int] = []
        digit_vocab = tuple(range(1, 10))
        for subject in np.unique(np.asarray(subject_ids)):
            rows = np.flatnonzero(np.asarray(subject_ids) == subject)
            rows = rows[np.argsort(np.asarray(acquisition_indices)[rows], kind="stable")]
            subject_y = np.asarray(y)[rows]
            true_rows = np.flatnonzero(subject_y == 1)
            if len(true_rows) == 0:
                continue
            true_digit = int(np.asarray(digits)[rows[true_rows[0]]])
            if true_digit not in digit_vocab:
                continue
            row_trial_channel_mask = (
                trial_channel_mask[rows] if trial_channel_mask is not None else None
            )
            _, evidence, quality = self._predict_repetition_inputs(
                np.asarray(X)[rows], trial_channel_mask=row_trial_channel_mask
            )
            with torch.inference_mode():
                fidelity = evidence_model.reliability(quality).float().cpu().numpy()
            trajectory, _ = evidence_model.candidate_log_score_trajectory(
                evidence,
                quality,
                torch.as_tensor(np.asarray(digits)[rows], device=evidence.device, dtype=torch.long),
                digit_vocab=digit_vocab,
            )
            trajectories_list.append(trajectory.detach().cpu().numpy())
            fidelity_list.append(fidelity)
            true_candidates.append(digit_vocab.index(true_digit))
        if not trajectories_list:
            return {"n_sequences": 0}
        sequence_lengths = np.asarray(
            [trajectory.shape[1] for trajectory in trajectories_list], dtype=np.int64
        )
        max_length = int(sequence_lengths.max())
        padded = np.zeros(
            (len(trajectories_list), trajectories_list[0].shape[0], max_length),
            dtype=np.float64,
        )
        fidelity_padded = np.zeros((len(trajectories_list), max_length), dtype=np.float64)
        for index, (trajectory, fidelity) in enumerate(zip(trajectories_list, fidelity_list, strict=True)):
            padded[index, :, : trajectory.shape[1]] = trajectory
            fidelity_padded[index, : len(fidelity)] = fidelity

        active_mask = np.arange(max_length)[None, :] < sequence_lengths[:, None]
        contributions = trajectory_contributions(padded)
        contributions[np.broadcast_to(~active_mask[:, None, :], contributions.shape)] = 0.0
        true = np.asarray(true_candidates)

        fidelity_flat = fidelity_padded[active_mask]
        reject_threshold = float(np.quantile(fidelity_flat, clean_reject_fraction))
        valid_mask = active_mask & (fidelity_padded >= reject_threshold)

        def threshold_report(mask: np.ndarray) -> dict[str, object]:
            report: dict[str, object] = {}
            for threshold in thresholds:
                report[f"{threshold:.2f}"] = replay_chain_stopping_from_contributions(
                    contributions,
                    true,
                    valid_mask=mask,
                    threshold=threshold,
                    sequence_lengths=sequence_lengths,
                )
            return report

        gated_report = threshold_report(valid_mask)
        ungated_report = threshold_report(active_mask)
        curve_thresholds = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
        gated_curve = risk_coverage_curve(
            contributions,
            true,
            valid_mask=valid_mask,
            sequence_lengths=sequence_lengths,
            thresholds=curve_thresholds,
        )
        ungated_curve = risk_coverage_curve(
            contributions,
            true,
            valid_mask=active_mask,
            sequence_lengths=sequence_lengths,
            thresholds=curve_thresholds,
        )
        improvement = curve_improvement(gated_curve, ungated_curve)
        return {
            "clean_reject_threshold": reject_threshold,
            "clean_reject_fraction": float(np.mean(~valid_mask[active_mask])),
            "thresholds_gated": gated_report,
            "thresholds_ungated": ungated_report,
            "risk_coverage_gated": gated_curve,
            "risk_coverage_ungated": ungated_curve,
            "curve_improvement": improvement,
            "n_sequences": len(trajectories_list),
        }

    def _gate_repetition_state_residual(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        digits: np.ndarray,
        acquisition_indices: np.ndarray | None,
        *,
        n_bootstrap: int = 400,
    ) -> dict[str, object]:
        """Fail-closed held-out log-score gate for the v12 state residual.

        The gate compares, on untouched audit subjects only, the observed-path
        prequential log score of the additive backbone against the same score
        with the refitted state residual enabled. It never uses validation or
        outer test subjects.
        """
        evidence_model = self.model_.repetition_evidence
        if not isinstance(evidence_model, AdditiveRepetitionEvidence):
            return {"passed": False, "failure": "legacy_repetition_model"}
        if evidence_model.state_residual is None:
            return {"passed": False, "failure": "state_residual_not_instantiated"}
        if not len(X) or not len(y) or not len(subject_ids):
            return {"passed": False, "failure": "no_audit_trials"}
        if digits is None or len(digits) != len(X):
            return {"passed": False, "failure": "audit_digits_missing"}
        if acquisition_indices is None:
            acquisition_indices = np.arange(len(X), dtype=np.int64)

        trained_gain = evidence_model.state_residual_gain_value()
        subject_deltas: list[float] = []
        try:
            for subject in np.unique(np.asarray(subject_ids)):
                rows = np.flatnonzero(np.asarray(subject_ids) == subject)
                rows = rows[np.argsort(np.asarray(acquisition_indices)[rows], kind="stable")]
                _, evidence, quality = self._predict_repetition_inputs(X[rows])
                labels = torch.as_tensor(
                    np.asarray(y, dtype=np.float32)[rows], device=evidence.device
                )
                with torch.inference_mode():
                    evidence_model.set_state_residual_gain(0.0)
                    backbone = evidence_model.forward_sequence(
                        evidence, quality, labels
                    ).observed_log_prob.sum()
                    evidence_model.set_state_residual_gain(trained_gain)
                    residual = evidence_model.forward_sequence(
                        evidence, quality, labels
                    ).observed_log_prob.sum()
                subject_deltas.append(float((residual - backbone).detach().cpu()))
        finally:
            evidence_model.set_state_residual_gain(trained_gain)

        decision = state_residual_gate_decision(
            np.asarray(subject_deltas), n_bootstrap=n_bootstrap, seed=0
        )
        if not bool(decision["passed"]):
            evidence_model.set_state_residual_gain(0.0)
        decision["state_residual_enabled"] = bool(decision["passed"])
        decision["trained_gain"] = trained_gain
        return decision


    def predict_repetition_candidates(
        self,
        X: np.ndarray,
        digits: np.ndarray,
        subject_ids: np.ndarray,
        *,
        digit_vocab: Sequence[int] = tuple(range(1, 10)),
        evidence_budgets: Sequence[int | None] = (1, 3, 5, 10, 15, None),
        acquisition_indices: np.ndarray | None = None,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, dict[str, object]]:
        """Return candidate-path decisions from all target/non-target flashes."""

        if not self._fitted:
            raise RuntimeError("请先 fit 再做 repetition accumulation。")
        if self.model_.repetition_evidence is None or not self.repetition_fitted_:
            return {}
        X = _as_float32_epochs(X)
        digits = np.asarray(digits)
        subject_ids = np.asarray(subject_ids)
        if digits.shape != (len(X),) or not np.issubdtype(digits.dtype, np.integer):
            raise ValueError("digits must be a one-dimensional integer array aligned with X.")
        if subject_ids.shape != (len(X),):
            raise ValueError("X/digits/subject_ids lengths must match.")
        if acquisition_indices is None:
            acquisition_indices = np.arange(len(X), dtype=np.int64)
        else:
            acquisition_indices = np.asarray(acquisition_indices)
            if not np.issubdtype(acquisition_indices.dtype, np.integer):
                raise ValueError("acquisition_indices must have an integer dtype.")
        if acquisition_indices.shape != (len(X),):
            raise ValueError("acquisition_indices must contain one value per trial.")
        _, evidence, quality = self._predict_repetition_inputs(
            X, trial_channel_mask=trial_channel_mask
        )
        vocab_values = tuple(digit_vocab)
        if (
            not vocab_values
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in vocab_values)
            or len(set(vocab_values)) != len(vocab_values)
        ):
            raise ValueError("digit_vocab must contain unique integer candidates.")
        vocab = np.asarray(vocab_values, dtype=np.int64)
        if any(
            budget is not None
            and (
                isinstance(budget, bool)
                or not isinstance(budget, (int, np.integer))
                or budget < 1
            )
            for budget in evidence_budgets
        ):
            raise ValueError("evidence_budgets must contain positive integers or None.")
        results: dict[str, dict[str, object]] = {}
        for budget in evidence_budgets:
            budget_name = "all" if budget is None else str(int(budget))
            predictions: list[int] = []
            covered_subjects: list[object] = []
            candidate_scores: list[np.ndarray] = []
            mean_reliability: list[float] = []
            for subject in np.unique(subject_ids):
                subject_rows = np.flatnonzero(subject_ids == subject)
                subject_rows = subject_rows[
                    np.argsort(acquisition_indices[subject_rows], kind="stable")
                ]
                if len(np.unique(acquisition_indices[subject_rows])) != len(subject_rows):
                    raise ValueError("acquisition_indices must be unique within each subject.")
                complete = True
                for digit in vocab:
                    rows = subject_rows[digits[subject_rows] == digit]
                    required = 1 if budget is None else int(budget)
                    if len(rows) < required:
                        complete = False
                        break
                if not complete:
                    continue
                if budget is None:
                    rows = subject_rows
                else:
                    checkpoints = [
                        np.flatnonzero(digits[subject_rows] == digit)[int(budget) - 1]
                        for digit in vocab
                    ]
                    rows = subject_rows[: int(max(checkpoints)) + 1]
                row_tensor = torch.as_tensor(rows, device=self.device, dtype=torch.long)
                with torch.inference_mode():
                    scores, reliability = self.model_.repetition_evidence.candidate_log_scores(
                        evidence[row_tensor],
                        quality[row_tensor],
                        torch.as_tensor(digits[rows], device=self.device, dtype=torch.long),
                        digit_vocab=tuple(vocab.tolist()),
                    )
                predictions.append(int(vocab[int(scores.argmax())]))
                covered_subjects.append(subject)
                candidate_scores.append(scores.float().cpu().numpy())
                mean_reliability.append(float(reliability.float().mean().cpu()))
            metric_name = (
                "all_chain_llr" if budget is None else f"prefix_minK_chain_llr@{budget_name}"
            )
            results[metric_name] = {
                "predicted": np.asarray(predictions, dtype=object),
                "subject_ids": np.asarray(covered_subjects, dtype=object),
                "scores": (
                    np.stack(candidate_scores)
                    if candidate_scores
                    else np.empty((0, len(vocab)), dtype=float)
                ),
                "mean_reliability": np.asarray(mean_reliability, dtype=float),
                "claim_eligible": bool(self.repetition_ready_),
                "repetition_ready": bool(self.repetition_ready_),
            }
        return results

    def predict_full(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ):
        """返回 (logits, tau, sigma)，用于 Phase 2 成分记录。

        只对最后一个 fit 的模型生效；供实验记录使用，不作为 evaluate 主路径。
        """
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_full。")
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        Xt = torch.from_numpy(X).to(self.device)
        mask_tensor = (
            torch.from_numpy(trial_channel_mask).to(self.device)
            if trial_channel_mask is not None
            else None
        )
        windows = self._measurement_windows(X, trial_channel_mask=trial_channel_mask)
        if windows is not None:
            windows = windows.to(self.device)
        logits_list, tau_list, sigma_list = [], [], []
        self.model_.eval()
        with torch.inference_mode():
            for i in range(0, Xt.shape[0], 256):
                xb = Xt[i : i + 256]
                wb = windows[i : i + 256] if windows is not None else None
                runtime_channel_mask = (
                    mask_tensor[i : i + 256]
                    if mask_tensor is not None
                    else self._model_channel_mask
                )
                output = self.model_(
                    xb,
                    E_chn=self._model_E_chn,
                    channel_mask=runtime_channel_mask,
                    domain_id=self._main_domain_ids(len(xb)),
                    return_likelihood=False,
                    measurement_window=wb,
                    measurement_posterior=self.measurement_posterior_,
                )
                if output.heads is None:
                    raise RuntimeError("模型 forward 未返回 heads。")
                logits_list.append(output.heads.logit_target.float().cpu())
                tau_list.append(output.tau.float().cpu())
                sigma_list.append(output.sigma.float().cpu())
        logits = torch.cat(logits_list).squeeze(-1).numpy().astype(np.float64)
        tau = torch.cat(tau_list).numpy()
        sigma = sigma_list[0].numpy() if sigma_list else np.zeros((3, 2))
        return logits, tau, sigma

    def predict_branches(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return PCW, Z2-aux, gated measurement and prequential evidence.

        ``pcw`` is the raw PCW logit without measurement/aux branches.
        ``aux`` is the research-only full-Z2 head logit (zeros when disabled).
        ``final`` reproduces the deployed ``logit_target`` path, including the
        named z2-aux-head add/replace semantics, measurement contribution and
        prequential contribution.
        """
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_branches。")
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        Xt = torch.from_numpy(X).to(self.device)
        mask_tensor = None
        if trial_channel_mask is not None:
            mask_tensor = torch.from_numpy(trial_channel_mask).to(self.device)
        windows = self._measurement_windows(X, trial_channel_mask=trial_channel_mask)
        if windows is not None:
            windows = windows.to(self.device)
        pcw_parts: list[torch.Tensor] = []
        aux_parts: list[torch.Tensor] = []
        measurement_parts: list[torch.Tensor] = []
        prequential_parts: list[torch.Tensor] = []
        self.model_.eval()
        with torch.inference_mode():
            for i in range(0, Xt.shape[0], 256):
                xb = Xt[i : i + 256]
                wb = windows[i : i + 256] if windows is not None else None
                runtime_channel_mask = (
                    mask_tensor[i : i + 256]
                    if mask_tensor is not None
                    else self._model_channel_mask
                )
                output = self.model_(
                    xb,
                    E_chn=self._model_E_chn,
                    channel_mask=runtime_channel_mask,
                    domain_id=self._main_domain_ids(len(xb)),
                    return_likelihood=self.prequential_variant_ is not None,
                    likelihood_class_means=(
                        self.generative_profile_.class_means
                        if self.generative_profile_ is not None
                        else None
                    ),
                    likelihood_channel_mask=(
                        self.generative_profile_.channel_mask
                        if self.generative_profile_ is not None
                        else None
                    ),
                    measurement_window=wb,
                    measurement_posterior=self.measurement_posterior_,
                )
                if output.heads is None:
                    raise RuntimeError("Model does not expose Neural-RIDE branch logits.")
                if output.measurement_contribution is None:
                    measurement = torch.zeros_like(output.heads.logit_pcw)
                else:
                    measurement = output.measurement_contribution.float()
                pcw_parts.append((output.heads.logit_pcw.float() - measurement).cpu())
                aux_parts.append(
                    output.heads.logit_aux.float().cpu()
                    if output.heads.logit_aux is not None
                    else torch.zeros_like(output.heads.logit_pcw.float()).cpu()
                )
                measurement_parts.append(measurement.cpu())
                if self.prequential_variant_ is not None and self.generative_profile_ is not None:
                    if output.likelihood is None:
                        raise RuntimeError("Prequential branch audit requires likelihood output.")
                    prequential_parts.append(
                        prequential_log_likelihood_ratio(
                            output.likelihood.likelihood_observation,
                            output.likelihood.causal_innovation,
                            self.generative_profile_.to(self.device),
                            variant=self.prequential_variant_,
                            observation_mask=output.likelihood.observation_mask,
                        ).cpu()
                    )
        pcw = torch.cat(pcw_parts).squeeze(-1).numpy().astype(np.float64)
        aux = torch.cat(aux_parts).squeeze(-1).numpy().astype(np.float64)
        measurement = torch.cat(measurement_parts).squeeze(-1).numpy().astype(np.float64)
        prequential_llr = (
            torch.cat(prequential_parts).numpy().astype(np.float64)
            if prequential_parts
            else np.zeros_like(pcw)
        )
        prequential_contribution = self.prequential_coefficient_ * (
            prequential_llr / self.prequential_scale_
        )
        aux_mode = getattr(self.model_, "z2_aux_head_mode", "off")
        if aux_mode == "replace":
            target_core = aux + measurement
        elif aux_mode == "add":
            target_core = pcw + measurement + aux
        else:
            target_core = pcw + measurement
        calibrated_base = (
            float(self.prequential_base_intercept_)
            + float(self.prequential_base_slope_) * target_core
        )
        return {
            "final": calibrated_base + prequential_contribution,
            "pcw": pcw,
            "aux": aux,
            "z2_aux_head_mode": aux_mode,
            "measurement_contribution": measurement,
            "measurement_coefficient": float(self.measurement_gain_),
            "prequential_llr": prequential_llr,
            "prequential_contribution": prequential_contribution,
            "prequential_coefficient": float(self.prequential_coefficient_),
            "prequential_base_slope": float(self.prequential_base_slope_),
            "prequential_base_intercept": float(self.prequential_base_intercept_),
        }

    def predict_interpretability(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return localization/decomposition arrays and fixed-noise stability pairs."""

        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_interpretability。")
        X = _as_float32_epochs(X)
        trial_channel_mask = self._validated_trial_channel_mask(X, trial_channel_mask)
        effective_mask = trial_channel_mask
        if effective_mask is None and self.channel_mask is not None:
            effective_mask = np.broadcast_to(
                self.channel_mask.detach().cpu().numpy().astype(bool), X.shape[:2]
            )
        rng = np.random.default_rng(0)
        scale = X.std(axis=2, keepdims=True).clip(min=1e-6)
        perturbed = X + (0.01 * scale * rng.standard_normal(X.shape)).astype(np.float32)
        if effective_mask is not None:
            perturbed[~effective_mask] = 0.0

        def collect(values: np.ndarray) -> dict[str, np.ndarray]:
            tensor = torch.from_numpy(values).to(self.device)
            mask_tensor = (
                torch.from_numpy(np.asarray(trial_channel_mask)).to(self.device)
                if trial_channel_mask is not None
                else None
            )
            tau_parts: list[torch.Tensor] = []
            variance_parts: list[torch.Tensor] = []
            erp_ratio_parts: list[torch.Tensor] = []
            include_erp = self.model_.component_decoder is not None
            with torch.inference_mode():
                for start in range(0, len(tensor), 256):
                    runtime_channel_mask = (
                        mask_tensor[start : start + 256]
                        if mask_tensor is not None
                        else self._model_channel_mask
                    )
                    output = self.model_(
                        tensor[start : start + 256],
                        E_chn=self._model_E_chn,
                        channel_mask=runtime_channel_mask,
                        domain_id=self._main_domain_ids(len(tensor[start : start + 256])),
                        return_likelihood=include_erp,
                    )
                    tau_parts.append(output.tau.float().cpu())
                    if output.erp is not None and output.likelihood is not None:
                        variance_parts.append(output.erp.amplitude_variance.float().cpu())
                        erp_energy = output.erp.reconstruction.float().square().mean((1, 2))
                        remainder = (
                            output.likelihood.likelihood_observation
                            - output.erp.reconstruction.float()
                        )
                        remainder_energy = remainder.square().mean((1, 2))
                        erp_ratio_parts.append(
                            (erp_energy / (erp_energy + remainder_energy + 1e-8)).cpu()
                        )
            result = {"tau": torch.cat(tau_parts).numpy()}
            if variance_parts:
                result["amplitude_variance"] = torch.cat(variance_parts).numpy()
                result["erp_energy_ratio"] = torch.cat(erp_ratio_parts).numpy()
            return result

        clean = collect(X)
        noisy = collect(perturbed)
        clean["tau_perturbed"] = noisy["tau"]
        clean["tau_bounds"] = np.stack(
            [
                self.model_.component_window.tau0_lo.detach().cpu().numpy()
                + self.model_.component_window.dtau_lo.detach().cpu().numpy(),
                self.model_.component_window.tau0_hi.detach().cpu().numpy()
                + self.model_.component_window.dtau_hi.detach().cpu().numpy(),
            ],
            axis=1,
        )
        return clean
