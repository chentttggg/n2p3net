"""模块 #12：总损失（Total Loss）。

职责（blueprint §8）：
    L = L_target + λ_pcw·L_PCW + λ_digit·L_digit + λ2·L_early + λ3·L_tau
        + λ_amp·L_amp + λ_recon·L_recon + λ_jit·L_jit + λ4·L_MMD
        - L_target / L_PCW：统一 final 与 PCW-only 的 BCE
        - L_digit：GTN fixed-K 九数字集合交叉熵
        - L_recon：fold-local 分频归一化的复数频谱成分重建误差
        - L_early ：Head-B 早期证据 BCE（同标签，同样 pos_weight≈8）
        - L_tau   ：潜伏期正则 Σ(τ_c−τ0_c)²（τ 不偏离先验中心太远，逐试次偏移 Δτ 的软约束）
        - L_amp   ：Head-D 幅值损失（有 A+X 时回归 P3b 窗内 Pz 物理幅值，否则 L2 自一致性正则，
                    保证 constitution P7 的幅值头被真实消费）
        - L_MMD   ：跨域特征级 RBF-MMD（Phase 3 可选，默认 λ4=0 不启用）

明确「不做」：
    - σ 正则：σ 由 sigmoid 软参数化天然有界 [20,80]ms，蓝图 §8 明确无需额外正则。
    - 参考抖动/时间扭曲等数据增强：train/augment.py。

三思决策记录（供后续会话追溯）：
    D-pos-weight     L_target 与 L_early 均用 pos_weight≈8：两者共享 target/non-target 标签、同受
                     1/9 不平衡，不加 pos_weight 会学「全判非目标」。蓝图 Head-A 明确写 pos_weight，
                     Head-B 因「同标签」继承同一 pos_weight。
    D-tau-scale      **尺度修正**：旧蓝图写 λ3≈1e-2 + L_tau=Σ(τ−τ0)²，但 τ 是 ms 单位、Δτ∈±30ms，
                     使 Δτ² ~225–900，λ3·L_tau ≈ 2–9 会**远超 L_target(≈0.7)**、反客为主（蓝图未细算
                     尺度，同 Conformer 参数账问题）。修正：L_tau = mean((τ−τ0)²)/tau_scale²，默认
                     tau_scale=50ms（蓝图 §5 的 tanh 半幅），使初始 λ3·L_tau≈1e-3 ≪ L_target。此修正
                     不改变 L_tau 的方向（仍是「Δτ 别太大」），只把它校准到「小正则」的量级。
    D-tau0-not-supervised  L_tau 用 τ−τ0（不 detach），τ0 的梯度经「τ 中的 +τ0」与「−τ0」精确抵消为 0，
                     ​故 L_tau 只正则 Δτ（逐试次偏移），**不监督 τ0**（τ0 仅被分类损失经 τ→A→H 监督）。
                     这是蓝图「τ0 数据驱动（被分类监督找群体平均）」的语义。
    D-mmd-phase3     L_MMD 是 Phase 3 跨域组件，默认 λ4=0；接口预留 z_features/domain_ids，Phase 2 零开销。
    D-device         pos_weight 显式对齐 logits 的 device/dtype（AMP bf16 兼容）。

契约（输入 → 输出）：
    output(N2P3NetOutput) + tau0(3,) + y(B,1) → Losses{total, target, early, tau, amp, mmd}（均标量）。

依赖的决策：blueprint §5/§8、constitution P7/D7、heads.D-logit-out、component_window.D-tau-param。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.innovation import CausalInnovationOutput
from models.n2p3net import N2P3NetOutput
from models.repetition import RepetitionEvidenceModel
from train.contracts import GenerativeProfile, ReconstructionProfile, SetMetadata
from train.prequential import nested_prequential_training_loss


@dataclass
class Losses:
    """总损失及各分项（均 0 维标量 Tensor，可 backward）。"""

    total: torch.Tensor
    target: torch.Tensor
    early: torch.Tensor
    tau: torch.Tensor
    mmd: torch.Tensor
    amp: torch.Tensor | None = None
    jit: torch.Tensor | None = None
    pcw: torch.Tensor | None = None
    digit: torch.Tensor | None = None
    recon: torch.Tensor | None = None
    recon_erp: torch.Tensor | None = None
    recon_erp_waveform: torch.Tensor | None = None
    recon_erp_projection: torch.Tensor | None = None
    recon_nll: torch.Tensor | None = None
    conditional_nll: torch.Tensor | None = None
    innovation_nll: torch.Tensor | None = None
    morphology_l0: torch.Tensor | None = None
    orth: torch.Tensor | None = None
    domain: torch.Tensor | None = None
    private: torch.Tensor | None = None


def _bce_with_pos_weight(logits: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE with logits + pos_weight（对齐 device/dtype，D-pos-weight/D-device）。"""
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    if y.dim() == 1:
        y = y.view(-1, 1)  # (N,) → (N,1) 防御（review P3：data 层出 (N,) 标签）
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def tau_regularization(tau: torch.Tensor, tau0: torch.Tensor, tau_scale_ms: float) -> torch.Tensor:
    """L_tau = mean((τ−τ0)²) / tau_scale²（尺度修正 D-tau-scale；τ0 不被监督 D-tau0-not-supervised）。"""
    return ((tau - tau0[None, :]) ** 2).mean() / (tau_scale_ms**2)


def shared_private_orthogonality(
    shared: torch.Tensor,
    private: torch.Tensor,
) -> torch.Tensor:
    """Normalized ``||H_s^T H_p||_F^2`` after removing batch means."""

    if shared.dim() != 2 or private.dim() != 2 or shared.shape[0] != private.shape[0]:
        raise ValueError("shared/private codes must be aligned two-dimensional tensors.")
    if shared.shape[0] < 2:
        return shared.sum() * 0.0
    shared_centered = shared.float() - shared.float().mean(dim=0, keepdim=True)
    private_centered = private.float() - private.float().mean(dim=0, keepdim=True)
    cross = shared_centered.T @ private_centered / float(shared.shape[0] - 1)
    return cross.square().mean()


def active_domain_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_domain_indices: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    """Cross-entropy over the training vocabulary only.

    Held-out classifier columns are removed before log-softmax, so they neither
    alter the denominator nor receive a backward gradient.
    """

    if logits.dim() != 2:
        raise ValueError("Domain logits must be a (B,D) tensor.")
    labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    if labels.numel() != logits.shape[0]:
        raise ValueError("Transfer domain labels must align with model outputs.")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("Transfer domain labels contain an index outside the vocabulary.")
    if active_domain_indices is None:
        active = torch.arange(logits.shape[1], device=logits.device, dtype=torch.long)
    else:
        active = torch.as_tensor(
            active_domain_indices,
            device=logits.device,
            dtype=torch.long,
        ).reshape(-1)
    if active.numel() == 0 or active.unique().numel() != active.numel():
        raise ValueError("active_domain_indices must be non-empty and unique.")
    if bool(((active < 0) | (active >= logits.shape[1])).any()):
        raise ValueError(
            "active_domain_indices contains a classifier column outside the vocabulary."
        )
    label_map = torch.full((logits.shape[1],), -1, device=logits.device, dtype=torch.long)
    label_map[active] = torch.arange(active.numel(), device=logits.device)
    mapped = label_map[labels]
    if bool((mapped < 0).any()):
        unknown = labels[mapped < 0].unique().detach().cpu().tolist()
        raise ValueError(f"Transfer labels contain held-out domain indices {unknown}.")
    return F.cross_entropy(logits.index_select(1, active).float(), mapped)


def gtn_set_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    stimulus_digits: torch.Tensor,
    set_group_ids: torch.Tensor,
    *,
    digit_vocab: Sequence[int] = tuple(range(1, 10)),
    evidence_k: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Nine-way GTN loss over complete subject-by-digit evidence sets.

    ``set_group_ids < 0`` marks samples that do not belong to a GTN set, such
    as auxiliary-domain trials. Incomplete or non-fixed-K groups are excluded
    rather than silently changing the decision objective.
    """
    flat_logits = logits.reshape(-1)
    flat_y = y.reshape(-1).to(device=flat_logits.device, dtype=flat_logits.dtype)
    digits = stimulus_digits.reshape(-1).to(device=flat_logits.device, dtype=torch.long)
    groups = set_group_ids.reshape(-1).to(device=flat_logits.device, dtype=torch.long)
    if not (len(flat_logits) == len(flat_y) == len(digits) == len(groups)):
        raise ValueError("logits/y/stimulus_digits/set_group_ids lengths must match.")

    vocab = torch.as_tensor(digit_vocab, device=flat_logits.device, dtype=torch.long)
    matches = digits[:, None] == vocab[None, :]
    valid = (groups >= 0) & matches.any(dim=1)
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0, 0

    logits_v = flat_logits[valid]
    y_v = flat_y[valid]
    groups_v = groups[valid]
    digit_cols = matches[valid].to(torch.long).argmax(dim=1)
    unique_groups, group_cols = torch.unique(groups_v, sorted=True, return_inverse=True)
    n_groups = int(unique_groups.numel())
    n_digits = int(vocab.numel())
    bucket = group_cols * n_digits + digit_cols

    scores = torch.zeros(n_groups * n_digits, device=flat_logits.device, dtype=flat_logits.dtype)
    counts = torch.zeros(n_groups * n_digits, device=flat_logits.device, dtype=torch.long)
    positive = torch.zeros_like(scores)
    scores.scatter_add_(0, bucket, logits_v)
    counts.scatter_add_(0, bucket, torch.ones_like(bucket))
    positive.scatter_add_(0, bucket, y_v)
    scores = scores.view(n_groups, n_digits)
    counts = counts.view(n_groups, n_digits)
    positive = positive.view(n_groups, n_digits)

    eligible = (counts > 0).all(dim=1)
    if evidence_k is not None:
        eligible &= (counts == int(evidence_k)).all(dim=1)
    # A valid oddball set has target trials under exactly one stimulus digit.
    eligible &= (positive > 0).sum(dim=1) == 1
    if not bool(eligible.any()):
        return flat_logits.sum() * 0.0, 0
    targets = positive[eligible].argmax(dim=1)
    return F.cross_entropy(scores[eligible], targets), int(eligible.sum().item())


def gtn_multi_k_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    metadata: SetMetadata,
    *,
    evidence_ks: Sequence[int] = (1, 3, 5, 10, 15),
    evidence_weights: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.45),
    digit_vocab: Sequence[int] = tuple(range(1, 10)),
) -> tuple[torch.Tensor, dict[int, int]]:
    """Digit CE at real online checkpoints with ragged per-K coverage."""

    batch_size = logits.reshape(-1).numel()
    metadata.validate(batch_size)
    ks = tuple(int(k) for k in evidence_ks)
    if not ks or any(k < 1 for k in ks) or tuple(sorted(set(ks))) != ks:
        raise ValueError("evidence_ks must be unique positive integers in ascending order.")
    if len(evidence_weights) != len(ks) or any(float(w) < 0.0 for w in evidence_weights):
        raise ValueError("evidence_weights must be non-negative and match evidence_ks.")
    weight = torch.as_tensor(evidence_weights, device=logits.device, dtype=logits.dtype)
    if not bool(weight.sum() > 0):
        raise ValueError("At least one multi-K evidence weight must be positive.")
    weight = weight / weight.sum()

    if metadata.sequence_ranks is None:
        raise ValueError("Online multi-K supervision requires acquisition-order sequence_ranks.")

    flat_logits = logits.reshape(-1)
    flat_y = y.reshape(-1)
    vocab = torch.as_tensor(digit_vocab, device=flat_logits.device, dtype=torch.long)
    total = flat_logits.sum() * 0.0
    active_weight = weight.sum() * 0.0
    coverage: dict[int, int] = {}
    for idx, k in enumerate(ks):
        prefix = torch.zeros_like(metadata.group_ids, dtype=torch.bool)
        groups = torch.unique(metadata.group_ids[metadata.group_ids >= 0], sorted=True)
        for group in groups:
            rows = torch.nonzero(metadata.group_ids == group, as_tuple=False).flatten()
            rows = rows[torch.argsort(metadata.sequence_ranks[rows], stable=True)]
            group_digits = metadata.stimulus_digits[rows]
            occurrence_positions = [
                torch.nonzero(group_digits == digit, as_tuple=False).flatten() for digit in vocab
            ]
            if any(positions.numel() < k for positions in occurrence_positions):
                continue
            checkpoint = int(
                torch.stack([positions[k - 1] for positions in occurrence_positions]).max()
            )
            prefix[rows[: checkpoint + 1]] = True
        loss_k, n_groups = gtn_set_cross_entropy(
            flat_logits[prefix],
            flat_y[prefix],
            metadata.stimulus_digits[prefix],
            metadata.group_ids[prefix],
            digit_vocab=digit_vocab,
        )
        coverage[k] = n_groups
        if n_groups:
            total = total + weight[idx] * loss_k
            active_weight = active_weight + weight[idx]
    if not metadata.prevalidated and not bool(active_weight > 0.0):
        raise ValueError("L_digit found no complete online checkpoint in this batch.")
    return total / active_weight, coverage


def repetition_multi_k_objective(
    weighted_logits: torch.Tensor,
    quality_features: torch.Tensor,
    y: torch.Tensor,
    metadata: SetMetadata,
    evidence_model: RepetitionEvidenceModel,
    *,
    evidence_ks: Sequence[int] = (1, 3, 5, 10, 15),
    evidence_weights: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.45),
    digit_vocab: Sequence[int] = tuple(range(1, 10)),
    reliability_aux_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """LEGACY_V11 nested candidate-path CE plus observed conditional NLL.

    Every candidate consumes every target and non-target flash in acquisition
    order.  Scores are sums of conditional log densities, not an arbitrary set
    readout and not a quality-weighted trial logit. The historical reliability
    anchor below is intentionally retained only for ``NEURAL_RIDE_V11_LEGACY``;
    v12 uses ``fidelity_margin_rank_loss`` in ``repetition_v12_objective.py``.
    """

    logits = weighted_logits.reshape(-1)
    labels = y.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if quality_features.shape != (logits.numel(), evidence_model.n_quality_features):
        raise ValueError("quality_features must align with weighted_logits.")
    if not metadata.prevalidated:
        metadata.validate(logits.numel())
    if metadata.sequence_ranks is None:
        raise ValueError(
            "Conditional repetition modeling requires acquisition-order sequence_ranks."
        )

    ks = tuple(int(k) for k in evidence_ks)
    if not ks or any(k < 1 for k in ks) or tuple(sorted(set(ks))) != ks:
        raise ValueError("evidence_ks must be unique positive integers in ascending order.")
    if len(evidence_weights) != len(ks) or any(float(w) < 0.0 for w in evidence_weights):
        raise ValueError("evidence_weights must be non-negative and match evidence_ks.")
    if reliability_aux_weight < 0.0:
        raise ValueError("reliability_aux_weight must be non-negative.")
    weights = torch.as_tensor(evidence_weights, device=logits.device, dtype=logits.dtype)
    if not bool(weights.sum() > 0.0):
        raise ValueError("At least one multi-K evidence weight must be positive.")
    weights = weights / weights.sum()
    kmax = metadata.prevalidated_kmax if metadata.prevalidated else None
    all_ks_covered = kmax is not None and max(ks) <= int(kmax)

    vocab = torch.as_tensor(digit_vocab, device=logits.device, dtype=torch.long)
    evidence = evidence_model.correct_evidence(logits)
    main_groups = torch.unique(metadata.group_ids[metadata.group_ids >= 0], sorted=True)
    if main_groups.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, zero, {k: 0 for k in ks}
    if metadata.prevalidated:
        main_groups = main_groups.tolist()

    group_rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for group in main_groups:
        rows = torch.nonzero(metadata.group_ids == group, as_tuple=False).flatten()
        order = torch.argsort(metadata.sequence_ranks[rows], stable=True)
        rows = rows[order]
        group_digits = metadata.stimulus_digits[rows]
        group_labels = labels[rows]
        positive_digits = torch.unique(group_digits[group_labels > 0.5])
        if not metadata.prevalidated and (positive_digits.numel() != 1 or not bool((positive_digits[0] == vocab).any())):
            continue
        group_rows.append(rows)
        targets.append(torch.nonzero(vocab == positive_digits[0], as_tuple=False).flatten()[0])

    if not group_rows:
        raise ValueError("Conditional NLL found no valid observed GTN sequence.")
    lengths = torch.as_tensor(
        [rows.numel() for rows in group_rows], device=logits.device, dtype=torch.long
    )
    padded_evidence = torch.nn.utils.rnn.pad_sequence(
        [evidence[rows] for rows in group_rows], batch_first=True
    )
    padded_quality = torch.nn.utils.rnn.pad_sequence(
        [quality_features[rows] for rows in group_rows], batch_first=True
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        [labels[rows] for rows in group_rows], batch_first=True
    )
    padded_digits = torch.nn.utils.rnn.pad_sequence(
        [metadata.stimulus_digits[rows] for rows in group_rows], batch_first=True
    )
    n_groups, max_length = padded_evidence.shape
    n_candidates = int(vocab.numel())
    candidate_labels = (padded_digits[:, None] == vocab[None, :, None]).to(logits.dtype)
    all_evidence = torch.cat(
        (padded_evidence, padded_evidence[:, None].expand(-1, n_candidates, -1).flatten(0, 1))
    )
    all_quality = torch.cat(
        (
            padded_quality,
            padded_quality[:, None]
            .expand(-1, n_candidates, -1, -1)
            .flatten(0, 1),
        )
    )
    all_labels = torch.cat((padded_labels, candidate_labels.flatten(0, 1)))
    all_lengths = torch.cat((lengths, lengths[:, None].expand(-1, n_candidates).reshape(-1)))
    sequence = evidence_model.forward_batched_sequences(
        all_evidence, all_quality, all_labels, all_lengths
    )
    active = torch.arange(max_length, device=logits.device)[None] < lengths[:, None]
    conditional_nll = -sequence.observed_log_prob[:n_groups][active].mean()
    if reliability_aux_weight > 0.0:
        main_quality = quality_features[metadata.group_ids >= 0]
        conditional_nll = conditional_nll + float(
            reliability_aux_weight
        ) * evidence_model.reliability_identification_loss(main_quality)
    candidate_log_prob = sequence.observed_log_prob[n_groups:].reshape(
        n_groups, n_candidates, max_length
    )
    trajectory = candidate_log_prob.cumsum(dim=-1) - math.log(n_candidates)
    occurrence_counts = candidate_labels.long().cumsum(dim=-1)
    target_tensor = torch.stack(targets).to(device=logits.device, dtype=torch.long)
    nested_ce = logits.sum() * 0.0
    active_weight = weights.sum() * 0.0
    coverage: dict[int, int] = {}
    for index, k in enumerate(ks):
        has_k = occurrence_counts[:, :, -1].ge(k).all(dim=1)
        if not all_ks_covered and not bool(has_k.any()):
            coverage[k] = 0
            continue
        first_k = occurrence_counts.ge(k).long().argmax(dim=-1)
        checkpoint = first_k.max(dim=1).values
        rows = torch.nonzero(has_k, as_tuple=False).flatten()
        scores = trajectory[rows, :, checkpoint[rows]]
        nested_ce = nested_ce + weights[index] * F.cross_entropy(scores, target_tensor[rows])
        active_weight = active_weight + weights[index]
        coverage[k] = int(rows.numel())
    if not metadata.prevalidated and not bool(active_weight > 0.0):
        raise ValueError("Conditional repetition objective found no requested online checkpoint.")
    return nested_ce / active_weight, conditional_nll, coverage


def _resolve_reconstruction_bands(
    *,
    n_time: int,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]],
    prior_weights: Sequence[float],
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    """Clip requested bands to Nyquist and retain only bands with FFT bins."""

    if len(bands_hz) != len(prior_weights):
        raise ValueError("bands_hz and prior_weights lengths must match.")
    freqs = torch.fft.rfftfreq(n_time, d=1.0 / float(sfreq))
    nyquist = float(sfreq) / 2.0
    active_bands: list[tuple[float, float]] = []
    active_weights: list[float] = []
    for idx, ((low, high), prior) in enumerate(zip(bands_hz, prior_weights, strict=True)):
        low = max(0.0, float(low))
        high = min(float(high), nyquist)
        if not low < high:
            continue
        upper = freqs <= high if idx == len(bands_hz) - 1 else freqs < high
        mask = (freqs >= low) & upper & (freqs > 0.0)
        if bool(mask.any()):
            active_bands.append((low, high))
            active_weights.append(float(prior))
    if not active_bands:
        raise ValueError(
            f"No reconstruction band contains a positive FFT bin for T={n_time}, sfreq={sfreq}."
        )
    if sum(active_weights) <= 0.0:
        raise ValueError("Active reconstruction bands require positive prior mass.")
    return tuple(active_bands), tuple(active_weights)


@torch.no_grad()
def estimate_reconstruction_profile(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    sfreq: float,
    tmin_ms: float,
    bands_hz: Sequence[tuple[float, float]],
    prior_weights: Sequence[float],
    channel_mask: torch.Tensor | None = None,
    erp_interval_ms: tuple[float, float] = (150.0, 900.0),
    edge_smoothing_ms: float = 20.0,
    bootstrap_samples: int = 64,
    split_half_repeats: int = 16,
    bootstrap_seed: int = 0,
) -> ReconstructionProfile:
    """Estimate fold-local ERP targets, target uncertainty and reliability.

    Bootstrap resampling is stratified by target class. Split-half statistics
    are diagnostics of the averaged contrast and do not supervise single-trial
    latency.
    """
    if X.dim() != 3:
        raise ValueError(f"X must be (N,C,T), got {tuple(X.shape)}.")
    labels = y.reshape(-1).to(device=X.device)
    if len(labels) != X.shape[0]:
        raise ValueError("X/y lengths must match.")
    pos = labels > 0.5
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        raise ValueError("Reconstruction profile requires both target classes.")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least two.")
    if split_half_repeats < 0:
        raise ValueError("split_half_repeats must be non-negative.")
    if channel_mask is None:
        channel_mask = torch.ones(X.shape[1], dtype=torch.bool, device=X.device)
    else:
        channel_mask = channel_mask.to(device=X.device, dtype=torch.bool)
    if channel_mask.shape != (X.shape[1],) or not bool(channel_mask.any()):
        raise ValueError("channel_mask must identify at least one of X's channels.")

    active_bands, active_priors = _resolve_reconstruction_bands(
        n_time=X.shape[-1],
        sfreq=sfreq,
        bands_hz=bands_hz,
        prior_weights=prior_weights,
    )
    valid_x = X[:, channel_mask].float()
    spectrum = torch.fft.rfft(valid_x, dim=-1)
    freqs = torch.fft.rfftfreq(X.shape[-1], d=1.0 / float(sfreq)).to(X.device)

    scales: list[torch.Tensor] = []
    snrs: list[torch.Tensor] = []
    for band_idx, (low, high) in enumerate(active_bands):
        upper = freqs <= high if band_idx == len(active_bands) - 1 else freqs < high
        mask = (freqs >= low) & upper & (freqs > 0.0)
        if not bool(mask.any()):
            raise ValueError(f"Frequency band {(low, high)} contains no FFT bins.")
        xb = spectrum[..., mask]
        power = xb.abs().square()
        scales.append(power.mean().sqrt().clamp_min(1e-6))
        mean_pos = xb[pos].mean(dim=0)
        mean_neg = xb[neg].mean(dim=0)
        signal = (mean_pos - mean_neg).abs().square().mean()
        var_pos = (xb[pos] - mean_pos).abs().square().mean()
        var_neg = (xb[neg] - mean_neg).abs().square().mean()
        snrs.append(signal / (0.5 * (var_pos + var_neg) + 1e-8))

    scales_t = torch.stack(scales).float()
    snr_t = torch.stack(snrs).float().clamp_min(1e-8)
    prior = torch.as_tensor(active_priors, device=X.device, dtype=torch.float32)
    prior = prior / prior.sum().clamp_min(1e-8)
    reference = snr_t.median().clamp_min(1e-8)
    reliability_factor = (snr_t / reference).sqrt().clamp(0.5, 2.0)
    weights = prior * reliability_factor
    weights = weights / weights.sum().clamp_min(1e-8)
    times_ms = tmin_ms + torch.arange(X.shape[-1], device=X.device, dtype=torch.float32) * (
        1000.0 / float(sfreq)
    )
    start_ms, end_ms = (float(v) for v in erp_interval_ms)
    if not start_ms < end_ms:
        raise ValueError("erp_interval_ms must have positive width.")
    smooth = max(float(edge_smoothing_ms), 1e-3)
    time_mask = torch.sigmoid((times_ms - start_ms) / smooth) * torch.sigmoid(
        (end_ms - times_ms) / smooth
    )
    contrast = X[pos].float().mean(dim=0) - X[neg].float().mean(dim=0)
    profile_mask = channel_mask.to(dtype=contrast.dtype)[:, None] * time_mask[None]
    contrast = contrast * profile_mask

    # Generate resampling indices on CPU so the profile path is portable to
    # backends that do not expose a device-specific torch.Generator (for
    # example some XPU builds). Only the small index tensors are transferred.
    generator = torch.Generator(device="cpu").manual_seed(int(bootstrap_seed))

    def bootstrap_means(samples: torch.Tensor) -> torch.Tensor:
        flat = samples.float().flatten(start_dim=1)
        count = flat.shape[0]
        chunks: list[torch.Tensor] = []
        for start in range(0, bootstrap_samples, 16):
            size = min(16, bootstrap_samples - start)
            draws = torch.randint(
                count,
                (size, count),
                generator=generator,
                device="cpu",
            ).to(X.device)
            weights = torch.zeros(size, count, device=X.device, dtype=torch.float32)
            weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float32))
            chunks.append((weights @ flat / float(count)).view(size, *samples.shape[1:]))
        return torch.cat(chunks, dim=0)

    bootstrap_contrasts = bootstrap_means(X[pos]) - bootstrap_means(X[neg])
    target_variance = bootstrap_contrasts.var(dim=0, unbiased=True) * profile_mask.square()

    split_correlations: list[torch.Tensor] = []
    split_nrmse: list[torch.Tensor] = []
    positive = X[pos].float()
    negative = X[neg].float()
    completed_splits = 0
    if min(positive.shape[0], negative.shape[0]) >= 2:
        for _ in range(split_half_repeats):
            pos_order = torch.randperm(positive.shape[0], generator=generator, device="cpu").to(
                X.device
            )
            neg_order = torch.randperm(negative.shape[0], generator=generator, device="cpu").to(
                X.device
            )
            pos_half = positive.shape[0] // 2
            neg_half = negative.shape[0] // 2
            left = (
                positive[pos_order[:pos_half]].mean(dim=0)
                - negative[neg_order[:neg_half]].mean(dim=0)
            ) * profile_mask
            right = (
                positive[pos_order[-pos_half:]].mean(dim=0)
                - negative[neg_order[-neg_half:]].mean(dim=0)
            ) * profile_mask
            selected = profile_mask > 1e-3
            left_values = left[selected]
            right_values = right[selected]
            left_centered = left_values - left_values.mean()
            right_centered = right_values - right_values.mean()
            denominator = (
                (left_centered.square().sum() * right_centered.square().sum())
                .sqrt()
                .clamp_min(1e-8)
            )
            split_correlations.append((left_centered * right_centered).sum() / denominator)
            reference_energy = (
                (0.5 * (left_values.square().mean() + right_values.square().mean()))
                .sqrt()
                .clamp_min(1e-8)
            )
            split_nrmse.append(
                (left_values - right_values).square().mean().sqrt() / reference_energy
            )
            completed_splits += 1
    correlation = (
        float(torch.stack(split_correlations).median().clamp(-1.0, 1.0))
        if split_correlations
        else 0.0
    )
    nrmse = float(torch.stack(split_nrmse).median()) if split_nrmse else 0.0
    profile = ReconstructionProfile(
        bands_hz=active_bands,
        band_scales=scales_t,
        band_weights=weights,
        evoked_snr=snr_t,
        evoked_contrast=contrast,
        evoked_target_variance=target_variance,
        target_rate=labels.float().mean(),
        channel_mask=channel_mask,
        time_mask=time_mask,
        sfreq=float(sfreq),
        n_time=int(X.shape[-1]),
        source_n_trials=int(X.shape[0]),
        bootstrap_samples=int(bootstrap_samples),
        split_half_repeats=completed_splits,
        split_half_correlation=correlation,
        split_half_nrmse=nrmse,
    )
    profile.validate(n_channels=X.shape[1])
    return profile


def phase_preserving_band_reconstruction_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]],
    band_scales: torch.Tensor,
    band_weights: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
    time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Band-normalized complex-spectrum MSE; phase and latency are retained."""
    if target.shape != reconstruction.shape or target.dim() != 3:
        raise ValueError(
            "target/reconstruction must share shape (B,C,T), got "
            f"{tuple(target.shape)} and {tuple(reconstruction.shape)}."
        )
    if len(bands_hz) != band_scales.numel() or len(bands_hz) != band_weights.numel():
        raise ValueError("Band definitions, scales and weights must have equal length.")
    target_work = target.detach().float()
    recon_work = reconstruction.float()
    if channel_mask is not None:
        channel_mask = channel_mask.to(device=target.device, dtype=torch.bool)
        if channel_mask.shape != (target.shape[1],) or not bool(channel_mask.any()):
            raise ValueError("channel_mask must select at least one target channel.")
        target_work = target_work[:, channel_mask]
        recon_work = recon_work[:, channel_mask]
    if time_mask is not None:
        time_mask = time_mask.to(device=target.device, dtype=torch.float32)
        if time_mask.shape != (target.shape[-1],):
            raise ValueError("time_mask must be (T,).")
        target_work = target_work * time_mask[None, None]
        recon_work = recon_work * time_mask[None, None]
    target_fft = torch.fft.rfft(target_work, dim=-1)
    recon_fft = torch.fft.rfft(recon_work, dim=-1)
    freqs = torch.fft.rfftfreq(target.shape[-1], d=1.0 / float(sfreq)).to(target.device)
    scales = band_scales.to(device=target.device, dtype=torch.float32).clamp_min(1e-6)
    weights = band_weights.to(device=target.device, dtype=torch.float32)
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    for band_idx, (low, high) in enumerate(bands_hz):
        upper = freqs <= high if band_idx == len(bands_hz) - 1 else freqs < high
        mask = (freqs >= low) & upper
        if not bool(mask.any()):
            raise ValueError(f"Frequency band {(low, high)} contains no FFT bins.")
        error = (target_fft[..., mask] - recon_fft[..., mask]).abs().square().mean()
        total = total + weights[band_idx] * error / scales[band_idx].square()
    return total


def normalized_waveform_reconstruction_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    channel_mask: torch.Tensor | None = None,
    time_mask: torch.Tensor | None = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Energy-normalized Huber loss for robust, identifiable ERP morphology."""

    if target.shape != reconstruction.shape or target.dim() != 3:
        raise ValueError("target/reconstruction must share shape (B,C,T).")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive.")
    target_work = target.detach().float()
    reconstruction_work = reconstruction.float()
    if channel_mask is not None:
        channels = channel_mask.to(device=target.device, dtype=torch.bool)
        if channels.shape != (target.shape[1],) or not bool(channels.any()):
            raise ValueError("channel_mask must select at least one target channel.")
        target_work = target_work[:, channels]
        reconstruction_work = reconstruction_work[:, channels]
    weights = torch.ones(target_work.shape[-1], device=target.device, dtype=torch.float32)
    if time_mask is not None:
        weights = time_mask.to(device=target.device, dtype=torch.float32)
        if weights.shape != (target.shape[-1],):
            raise ValueError("time_mask must be (T,).")
    weights = weights[None, None]
    pointwise = F.huber_loss(
        reconstruction_work,
        target_work,
        reduction="none",
        delta=float(huber_delta),
    )
    error_energy = (pointwise * weights).sum()
    target_energy = (target_work.square() * weights).sum().clamp_min(1e-6)
    return error_energy / target_energy


def complex_mean_difference_loss(
    background: torch.Tensor,
    y: torch.Tensor,
    *,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]],
    band_scales: torch.Tensor,
    band_weights: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
    reference_contrast: torch.Tensor | None = None,
    time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize phase-locked class means while leaving class power unconstrained."""

    if background.dim() != 3:
        raise ValueError("background must be (B,C,T).")
    labels = y.reshape(-1).to(device=background.device)
    if labels.numel() != background.shape[0]:
        raise ValueError("background/y lengths must match.")
    positive = labels > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        return background.sum() * 0.0
    work = background.float()
    if channel_mask is not None:
        channels = channel_mask.to(device=background.device, dtype=torch.bool)
        if channels.shape != (background.shape[1],) or not bool(channels.any()):
            raise ValueError("channel_mask must select at least one background channel.")
        work = work[:, channels]
    if time_mask is not None:
        time_weight = time_mask.to(device=background.device, dtype=torch.float32)
        if time_weight.shape != (background.shape[-1],):
            raise ValueError("time_mask must be (T,).")
        work = work * time_weight[None, None]
    spectrum = torch.fft.rfft(work, dim=-1)
    frequencies = torch.fft.rfftfreq(background.shape[-1], d=1.0 / float(sfreq)).to(
        background.device
    )
    weights = band_weights.to(background.device, dtype=torch.float32)
    total = torch.zeros((), device=background.device, dtype=torch.float32)
    reference_fft = None
    if reference_contrast is not None:
        reference = reference_contrast.to(device=background.device, dtype=torch.float32)
        if reference.shape != background.shape[1:]:
            raise ValueError("reference_contrast must be (C,T).")
        if channel_mask is not None:
            reference = reference[channels]
        if time_mask is not None:
            reference = reference * time_weight[None]
        reference_fft = torch.fft.rfft(reference, dim=-1)
    scales = band_scales.to(background.device, dtype=torch.float32).clamp_min(1e-6)
    for band_index, (low, high) in enumerate(bands_hz):
        upper = frequencies <= high if band_index == len(bands_hz) - 1 else frequencies < high
        frequency_mask = (frequencies >= low) & upper
        if not bool(frequency_mask.any()):
            raise ValueError(f"Frequency band {(low, high)} contains no FFT bins.")
        class_gap = spectrum[positive][..., frequency_mask].mean(dim=0) - spectrum[negative][
            ..., frequency_mask
        ].mean(dim=0)
        denominator = (
            reference_fft[..., frequency_mask].abs().square().mean().clamp_min(1e-8)
            if reference_fft is not None
            else scales[band_index].square()
        )
        total = total + weights[band_index] * class_gap.abs().square().mean() / denominator
    return total


def _class_contrast(
    signal: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor | None:
    """Return a class-balanced mean contrast, or ``None`` for one-class batches."""

    labels = y.reshape(-1).to(device=signal.device)
    if signal.dim() != 3 or signal.shape[0] != labels.numel():
        raise ValueError("signal must be (B,C,T) and align with y.")
    positive = labels > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        return None
    return signal[positive].float().mean(dim=0) - signal[negative].float().mean(dim=0)


def normalized_contrast_waveform_loss(
    target: torch.Tensor,
    estimate: torch.Tensor,
    *,
    channel_mask: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Squared waveform NRMSE for offline ERP reconstruction audit.

    Normalization uses the evoked template itself, so a zero estimate has loss
    exactly one regardless of the much larger single-trial EEG power.
    """

    if target.shape != estimate.shape or target.dim() != 2:
        raise ValueError("target and estimate must share shape (C,T).")
    channels = channel_mask.to(device=target.device, dtype=torch.bool)
    time_weight = time_mask.to(device=target.device, dtype=torch.float32)
    if channels.shape != (target.shape[0],) or time_weight.shape != (target.shape[1],):
        raise ValueError("Invalid channel/time mask for ERP contrast.")
    weights = time_weight[None]
    target_work = target.detach().float()[channels]
    estimate_work = estimate.float()[channels]
    numerator = ((estimate_work - target_work).square() * weights).sum()
    denominator = (target_work.square() * weights).sum().clamp_min(1e-8)
    return numerator / denominator


def phase_preserving_contrast_loss(
    target: torch.Tensor,
    estimate: torch.Tensor,
    profile: ReconstructionProfile,
) -> torch.Tensor:
    """Squared complex-spectrum NRMSE with template-energy normalization."""

    if target.shape != estimate.shape or target.dim() != 2:
        raise ValueError("target and estimate must share shape (C,T).")
    channels = profile.channel_mask.to(device=target.device, dtype=torch.bool)
    time_mask = profile.time_mask.to(device=target.device, dtype=torch.float32)
    target_fft = torch.fft.rfft(target.detach().float()[channels] * time_mask[None], dim=-1)
    error_fft = torch.fft.rfft(
        (estimate.float()[channels] - target.detach().float()[channels]) * time_mask[None],
        dim=-1,
    )
    frequencies = torch.fft.rfftfreq(profile.n_time, d=1.0 / float(profile.sfreq)).to(target.device)
    weights = profile.band_weights.to(device=target.device, dtype=torch.float32)
    numerator = torch.zeros((), device=target.device, dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    for band_index, (low, high) in enumerate(profile.bands_hz):
        upper = (
            frequencies <= high if band_index == len(profile.bands_hz) - 1 else frequencies < high
        )
        mask = (frequencies >= low) & upper
        if not bool(mask.any()):
            raise ValueError(f"Frequency band {(low, high)} contains no FFT bins.")
        numerator = numerator + weights[band_index] * error_fft[..., mask].abs().square().mean()
        denominator = (
            denominator + weights[band_index] * target_fft[..., mask].abs().square().mean()
        )
    return numerator / denominator.clamp_min(1e-8)


def template_projection_loss(
    target: torch.Tensor,
    estimate: torch.Tensor,
    *,
    channel_mask: torch.Tensor,
    time_mask: torch.Tensor,
) -> torch.Tensor:
    """Squared residual gain in the zero-lag ERP-template direction."""

    if target.shape != estimate.shape or target.dim() != 2:
        raise ValueError("target and estimate must share shape (C,T).")
    channels = channel_mask.to(device=target.device, dtype=torch.bool)
    weights = time_mask.to(device=target.device, dtype=torch.float32)[None]
    target_work = target.detach().float()[channels]
    residual = target_work - estimate.float()[channels]
    denominator = (target_work.square() * weights).sum().clamp_min(1e-8)
    gain = (target_work * residual * weights).sum() / denominator
    return gain.square()


def heteroscedastic_contrast_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    channel_mask: torch.Tensor,
    time_mask: torch.Tensor,
    target_variance: torch.Tensor,
) -> torch.Tensor:
    """Faithful NLL for the observed class-mean contrast, not pseudo trials."""

    if target.shape != mean.shape or target.shape != variance.shape:
        raise ValueError("target, mean and variance must share shape (C,T).")
    if target_variance.shape != target.shape:
        raise ValueError("target_variance must match the class contrast.")
    channels = channel_mask.to(device=target.device, dtype=torch.bool)
    weights = time_mask.to(device=target.device, dtype=torch.float32)[None]
    if channels.shape != (target.shape[0],) or weights.shape[-1] != target.shape[1]:
        raise ValueError("Invalid channel/time mask for contrast NLL.")
    target_work = target.detach().float()[channels]
    error2 = (target_work - mean.detach().float()[channels]).square()
    total_variance = (
        variance.float()[channels] + target_variance.detach().float()[channels]
    ).clamp_min(1e-6)
    # Dimensionless scaling prevents the log term from depending on arbitrary
    # sensor units while preserving the proper error/variance relationship.
    scale2 = (
        (target_work.square() * weights).sum() / weights.expand_as(target_work).sum()
    ).clamp_min(1e-6)
    normalized_variance = total_variance / scale2
    nll = 0.5 * (error2 / total_variance + normalized_variance.log())
    return (nll * weights).sum() / weights.expand_as(nll).sum().clamp_min(1e-6)


def neural_ride_reconstruction_loss(
    output: N2P3NetOutput,
    y: torch.Tensor,
    profile: ReconstructionProfile,
    *,
    sample_mask: torch.Tensor | None = None,
    waveform_weight: float = 1.0,
    projection_weight: float = 1.0,
    nll_weight: float = 0.1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fit the offline ERP decoder to the fold-local class contrast."""

    if output.erp is None:
        raise ValueError("Neural-RIDE reconstruction requires output.erp.")
    profile.validate(n_channels=output.erp.reconstruction.shape[1])
    if output.erp.reconstruction.shape[-1] != profile.n_time:
        raise ValueError("ReconstructionProfile time axis does not match model output.")
    if any(
        weight < 0.0
        for weight in (
            waveform_weight,
            projection_weight,
            nll_weight,
        )
    ):
        raise ValueError("Reconstruction sub-loss weights must be non-negative.")

    y_selected = y if sample_mask is None else y[sample_mask]
    erp_mean = output.erp.reconstruction
    erp_variance = output.erp.waveform_variance
    if sample_mask is not None:
        erp_mean = erp_mean[sample_mask]
        erp_variance = erp_variance[sample_mask]
    if erp_mean.shape[0] == 0:
        zero = output.erp.reconstruction.sum() * 0.0
        return zero, zero, zero, zero, zero

    labels = y_selected.reshape(-1).to(device=erp_mean.device)
    positive = labels > 0.5
    negative = ~positive
    erp_contrast = _class_contrast(erp_mean, labels)
    if erp_contrast is None:
        erp_spectrum = erp_mean.sum() * 0.0
        erp_waveform = erp_spectrum
        erp_projection = erp_spectrum
        erp_nll = erp_spectrum
    else:
        template = profile.evoked_contrast.to(device=erp_mean.device, dtype=torch.float32)
        erp_spectrum = phase_preserving_contrast_loss(template, erp_contrast, profile)
        erp_waveform = normalized_contrast_waveform_loss(
            template,
            erp_contrast,
            channel_mask=profile.channel_mask,
            time_mask=profile.time_mask,
        )
        erp_projection = template_projection_loss(
            template,
            erp_contrast,
            channel_mask=profile.channel_mask,
            time_mask=profile.time_mask,
        )
        contrast_variance = erp_variance[positive].float().sum(dim=0) / float(
            positive.sum().square()
        ) + erp_variance[negative].float().sum(dim=0) / float(negative.sum().square())
        erp_nll = heteroscedastic_contrast_nll(
            template,
            erp_contrast,
            contrast_variance,
            channel_mask=profile.channel_mask,
            time_mask=profile.time_mask,
            target_variance=profile.evoked_target_variance,
        )
    total = (
        erp_spectrum
        + waveform_weight * erp_waveform
        + projection_weight * erp_projection
        + nll_weight * erp_nll
    )
    return (
        total,
        erp_spectrum,
        erp_waveform,
        erp_projection,
        erp_nll,
    )


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, bandwidth: float | None = None) -> torch.Tensor:
    """无偏 RBF-MMD²（两分布样本 x/y，D-mmd-phase3，Phase 3 用）。

    bandwidth=None 时用 median heuristic（样本并集的成对距离中位数）；固定 1.0 在 D=64
    下会 exp(-d²/2)→0、梯度全零（review v6 P1，实测 rbf_mmd2(x,x+1)≈-1e-16）。
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"rbf_mmd2 输入须为 (N,D) 二维张量，得到 x={x.shape}、y={y.shape}。")
    # torch.pdist 不支持 bf16；MMD 显式提升到 fp32 计算，梯度回传不受影响。
    x = x.float()
    y = y.float()
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    if bandwidth is None:
        z = torch.cat([x, y], dim=0)
        if z.shape[0] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        bandwidth = float(torch.pdist(z).median().clamp_min(1e-6).item())
        if bandwidth <= 0.0:
            bandwidth = 1.0

    def gaussian_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a[:, None, :] - b[None, :, :]  # (n, m, D)
        return torch.exp(-(diff**2).sum(-1) / (2.0 * bandwidth**2))

    kxx = gaussian_kernel(x, x)
    kyy = gaussian_kernel(y, y)
    kxy = gaussian_kernel(x, y)
    n, m = x.shape[0], y.shape[0]
    xx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    yy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    xy = kxy.sum() / (n * m)
    return xx + yy - 2.0 * xy


def compute_losses(
    output: N2P3NetOutput,
    tau0: torch.Tensor,
    y: torch.Tensor,
    *,
    lambda2: float = 0.3,
    lambda3: float = 0.0,
    lambda_pcw: float = 0.0,
    lambda_digit: float = 0.0,
    set_metadata: SetMetadata | None = None,
    digit_evidence_ks: Sequence[int] = (1, 3, 5, 10, 15),
    digit_evidence_weights: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.45),
    lambda_amp: float = 0.0,
    lambda_recon: float = 0.0,
    reconstruction_profile: ReconstructionProfile | None = None,
    recon_waveform_weight: float = 1.0,
    recon_projection_weight: float = 1.0,
    recon_nll_weight: float = 0.1,
    lambda_innovation: float = 0.0,
    generative_profile: GenerativeProfile | None = None,
    generative_profile_validated: bool = False,
    innovation_covariance_weight: float = 1.0,
    lambda_morphology_l0: float = 0.0,
    lambda_jit: float = 0.0,
    tau_shift: torch.Tensor | None = None,
    shift_ms: torch.Tensor | None = None,
    pos_weight: float = 8.0,
    tau_scale_ms: float = 50.0,
    z_features: torch.Tensor | None = None,
    domain_ids: torch.Tensor | None = None,
    active_domain_indices: Sequence[int] | torch.Tensor | None = None,
    main_domain: int = 0,
    aux_domain: int = 1,
    lambda4: float = 0.0,
    mmd_bandwidth: float | None = None,
    lambda_orth: float = 0.0,
    lambda_adv: float = 0.0,
    lambda_private: float = 0.0,
    reconstruct_all_domains: bool = False,
    X: torch.Tensor | None = None,
    pz_channel: int = 3,
) -> Losses:
    """Compute the Neural-RIDE trial, branch, set, component and domain losses.

    Parameters
    ----------
    output : N2P3NetOutput
        模型前向输出（含 heads 的 logit_target/logit_early、tau、attention）。
    tau0 : torch.Tensor
        (3,) 先验中心（component_window.tau0_bounded）。
    y : torch.Tensor
        (B,1) target 标签（1=target, 0=non-target）。
    lambda2 / lambda3 : float
        Head-B / L_tau 权重。正式 v11 PCW recipe 为 0；仅研究性识别性实验可显式启用。
    lambda_amp : float
        Head-D 幅值损失权重（constitution P7；默认 0 保持旧接口，Trainer 默认开启）。
      lambda_jit / tau_shift / shift_ms : float / Tensor / Tensor
          Phase 2 自监督 jitter 一致性：tau_shift 是已知偏移 X_shift 的 τ，shift_ms 为
          (B,) 物理毫秒偏移；L_jit = mean((tau_shift - tau - shift_ms)²)/tau_scale²。
          无标签，仅锚定 τ 的物理 ms 尺度（E3 兼容）。
    pos_weight : float
        BCE 正样本权重（默认 8，target 占 1/9）。
    tau_scale_ms : float
        L_tau 的归一化尺度（默认 50ms，D-tau-scale）。
    z_features / domain_ids : Tensor | None
        Phase 3 跨域 MMD 的特征（(B,D)）与域标签（(B,)）；None 或 λ4=0 时跳过。
    lambda4 : float
        L_MMD 权重（Phase 3 启用，默认 0）。
      main_domain / aux_domain : int
          P9 域标签：main_domain（默认 0）= GTN；aux_domain（默认 1）= 辅助 P300 域。
          domain_ids 存在时，L_target/L_early/L_amp 只对 main_domain 样本计算；
          L_MMD 只在 main_domain 与 aux_domain 之间计算。辅助域标签严禁进入主监督。
    X : torch.Tensor | None
        (B,C,T) 原始/增强后输入；与 output.attention 一起提供时，L_amp 使用
        P3b 软窗对 Pz 信号的物理幅值回归（blueprint Head-D）。None 时退化为
        Head-D 幅值的 L2 自一致性正则，保证幅值头一定被损失消费（review v6 P1）。
    pz_channel : int
        Pz 在 X 通道轴上的索引（默认 3，对应标准 8 导蒙太奇）。
    """
    # P9 硬隔离：domain_ids 存在时，主监督只对 GTN（main_domain）样本计算。
    if output.heads is None:
        raise ValueError("compute_losses requires model heads.")
    loss_device = output.heads.logit_target.device
    if domain_ids is not None:
        domain_ids = domain_ids.to(device=loss_device)
        if domain_ids.numel() != output.heads.logit_target.shape[0]:
            raise ValueError(
                f"domain_ids 长度须等于 batch_size，得到 {domain_ids.numel()} "
                f"vs {output.heads.logit_target.shape[0]}。"
            )
        main_mask = (domain_ids == main_domain).view(-1)
    else:
        main_mask = None

    if main_mask is not None and not bool(main_mask.any()):
        # 当前 batch 全是辅助域：主监督与 τ 自监督均为零，只保留可选的 L_MMD。
        L_target = torch.zeros(
            (), device=output.heads.logit_target.device, dtype=output.heads.logit_target.dtype
        )
        L_early = torch.zeros_like(L_target)
        L_amp = torch.zeros_like(L_target)
        L_pcw = torch.zeros_like(L_target)
    else:
        logit_target = (
            output.heads.logit_target[main_mask]
            if main_mask is not None
            else output.heads.logit_target
        )
        logit_early = (
            output.heads.logit_early[main_mask]
            if main_mask is not None
            else output.heads.logit_early
        )
        y_main = y[main_mask] if main_mask is not None else y
        L_target = _bce_with_pos_weight(logit_target, y_main, pos_weight)
        L_early = _bce_with_pos_weight(logit_early, y_main, pos_weight)
        pcw_logits = (
            output.heads.logit_pcw[main_mask] if main_mask is not None else output.heads.logit_pcw
        )
        L_pcw = _bce_with_pos_weight(pcw_logits, y_main, pos_weight)

        if lambda_amp > 0.0:
            if X is not None and output.attention is not None:
                # Head-D 物理幅值（P9：只对 GTN 样本计算）。
                pz = pz_channel if pz_channel < X.shape[1] else X.shape[1] - 1
                a_p3b = output.attention[:, 2, :]
                a_p3b = a_p3b[main_mask] if main_mask is not None else a_p3b
                x_main = X[main_mask] if main_mask is not None else X
                amp_main = (
                    output.heads.amplitude[main_mask]
                    if main_mask is not None
                    else output.heads.amplitude
                )
                target_amp = (a_p3b * x_main[:, pz, :]).sum(dim=1)
                L_amp = F.mse_loss(amp_main.squeeze(-1), target_amp)
            else:
                amp = (
                    output.heads.amplitude[main_mask]
                    if main_mask is not None
                    else output.heads.amplitude
                )
                L_amp = amp.pow(2).mean()
        else:
            L_amp = torch.zeros_like(L_target)

    if lambda_digit > 0.0:
        if set_metadata is None:
            raise ValueError("lambda_digit>0 requires explicit SetMetadata.")
        L_digit, _ = gtn_multi_k_cross_entropy(
            output.heads.logit_target,
            y,
            set_metadata,
            evidence_ks=digit_evidence_ks,
            evidence_weights=digit_evidence_weights,
        )
    else:
        L_digit = torch.zeros_like(L_target)

    if lambda_recon > 0.0:
        if reconstruction_profile is None:
            raise ValueError("lambda_recon>0 requires an explicit ReconstructionProfile.")
        (
            L_recon,
            L_recon_erp,
            L_recon_erp_waveform,
            L_recon_erp_projection,
            L_recon_nll,
        ) = neural_ride_reconstruction_loss(
            output,
            y,
            reconstruction_profile,
            sample_mask=None if reconstruct_all_domains else main_mask,
            waveform_weight=recon_waveform_weight,
            projection_weight=recon_projection_weight,
            nll_weight=recon_nll_weight,
        )
    else:
        L_recon = torch.zeros_like(L_target)
        L_recon_erp = torch.zeros_like(L_target)
        L_recon_erp_waveform = torch.zeros_like(L_target)
        L_recon_erp_projection = torch.zeros_like(L_target)
        L_recon_nll = torch.zeros_like(L_target)

    if lambda_innovation < 0.0:
        raise ValueError("lambda_innovation must be non-negative.")
    if lambda_innovation > 0.0:
        if output.likelihood is None or generative_profile is None:
            raise ValueError(
                "lambda_innovation>0 requires a likelihood output and GenerativeProfile."
            )
        observation = output.likelihood.likelihood_observation
        innovation = output.likelihood.causal_innovation
        labels_for_innovation = y
        if main_mask is not None:
            observation = observation[main_mask]
            labels_for_innovation = y[main_mask]
            innovation = CausalInnovationOutput(
                history_correction=innovation.history_correction[main_mask],
                log_variance_scale=innovation.log_variance_scale[main_mask],
                factor_scale=innovation.factor_scale[main_mask],
                hypothesis_labels=(
                    None
                    if innovation.hypothesis_labels is None
                    else innovation.hypothesis_labels[main_mask]
                ),
            )
        if observation.shape[0] == 0:
            L_innovation_nll = L_target * 0.0
        else:
            L_innovation_nll = nested_prequential_training_loss(
                observation,
                innovation,
                generative_profile,
                labels_for_innovation,
                covariance_weight=innovation_covariance_weight,
                observation_mask=(
                    output.likelihood.observation_mask[main_mask]
                    if main_mask is not None
                    else output.likelihood.observation_mask
                ),
                validate_profile=not generative_profile_validated,
                mask_is_homogeneous=output.likelihood.mask_is_homogeneous,
            )
    else:
        L_innovation_nll = torch.zeros_like(L_target)

    if lambda_morphology_l0 < 0.0:
        raise ValueError("lambda_morphology_l0 must be non-negative.")
    if lambda_morphology_l0 > 0.0:
        if output.erp is None:
            raise ValueError("Morphology L0 regularization requires an ERP decoder output.")
        L_morphology_l0 = output.erp.expected_l0.mean()
    else:
        L_morphology_l0 = torch.zeros_like(L_target)

    # P9：domain_ids 存在时 L_tau 只作用于主域；辅助域只进 L_MMD。
    if main_mask is not None and not main_mask.any():
        L_tau = torch.zeros((), device=loss_device, dtype=output.tau.dtype)
    else:
        tau_main = output.tau[main_mask] if main_mask is not None else output.tau
        L_tau = tau_regularization(tau_main, tau0, tau_scale_ms)

    if lambda_jit > 0.0 and tau_shift is not None and shift_ms is not None:
        shift_ms = shift_ms.to(device=loss_device, dtype=output.tau.dtype)
        tau_shift_main = tau_shift[main_mask] if main_mask is not None else tau_shift
        tau_main = output.tau[main_mask] if main_mask is not None else output.tau
        shift_main = shift_ms[main_mask] if main_mask is not None else shift_ms
        if tau_main.shape[0] == 0:
            L_jit = torch.zeros((), device=loss_device, dtype=output.tau.dtype)
        else:
            L_jit = ((tau_shift_main - tau_main - shift_main[:, None]) ** 2).mean() / (
                tau_scale_ms**2
            )
    else:
        L_jit = torch.zeros_like(L_tau)

    if lambda4 > 0.0 and z_features is not None and domain_ids is not None:
        # N2P3NetOutput.features 是 (B,T,D)；MMD 契约是 (B,D)。这里按时间维平均池化，
        # 等价于用整个 epoch 的编码均值做域对齐；2D 输入则直接使用。
        if z_features.dim() == 3:
            z_pooled = z_features.mean(dim=1)
        elif z_features.dim() == 2:
            z_pooled = z_features
        else:
            raise ValueError(f"z_features 须为 (B,D) 或 (B,T,D)，得到 {z_features.shape}。")
        d0 = z_pooled[domain_ids == main_domain]
        d1 = z_pooled[domain_ids == aux_domain]
        L_mmd = rbf_mmd2(d0, d1, bandwidth=mmd_bandwidth)
    else:
        L_mmd = torch.zeros_like(L_target)

    transfer_weights = (lambda_orth, lambda_adv, lambda_private)
    if any(weight < 0.0 for weight in transfer_weights):
        raise ValueError("Shared/private transfer loss weights must be non-negative.")
    if any(weight > 0.0 for weight in transfer_weights):
        if domain_ids is None:
            raise ValueError("Shared/private transfer losses require domain_ids.")
        if (
            output.shared_features is None
            or output.private_features is None
            or output.domain_logits is None
            or output.dataset_logits is None
        ):
            raise ValueError(
                "Shared/private transfer losses require a model configured with shared_private=True."
            )
        labels = domain_ids.reshape(-1).to(device=loss_device, dtype=torch.long)
        L_orth = shared_private_orthogonality(
            output.shared_features,
            output.private_features,
        )
        L_domain = active_domain_cross_entropy(
            output.domain_logits,
            labels,
            active_domain_indices,
        )
        L_private = active_domain_cross_entropy(
            output.dataset_logits,
            labels,
            active_domain_indices,
        )
    else:
        L_orth = torch.zeros_like(L_target)
        L_domain = torch.zeros_like(L_target)
        L_private = torch.zeros_like(L_target)

    total = (
        L_target
        + lambda_pcw * L_pcw
        + lambda_digit * L_digit
        + lambda2 * L_early
        + lambda3 * L_tau
        + lambda_amp * L_amp
        + lambda_recon * L_recon
        + lambda_innovation * L_innovation_nll
        + lambda_morphology_l0 * L_morphology_l0
        + lambda_jit * L_jit
        + lambda4 * L_mmd
        + lambda_orth * L_orth
        + lambda_adv * L_domain
        + lambda_private * L_private
    )
    return Losses(
        total=total,
        target=L_target,
        early=L_early,
        tau=L_tau,
        amp=L_amp,
        mmd=L_mmd,
        jit=L_jit,
        pcw=L_pcw,
        digit=L_digit,
        recon=L_recon,
        recon_erp=L_recon_erp,
        recon_erp_waveform=L_recon_erp_waveform,
        recon_erp_projection=L_recon_erp_projection,
        recon_nll=L_recon_nll,
        innovation_nll=L_innovation_nll,
        morphology_l0=L_morphology_l0,
        orth=L_orth,
        domain=L_domain,
        private=L_private,
    )
