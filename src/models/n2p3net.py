"""Neural-RIDE model with PCW classification and strict-past density outputs.

The discriminative logit is PCW-constrained. The parameter-disjoint innovation
graph predicts both class-conditional densities from fixed-coordinate history;
it is fused only after an independent, subject-disjoint prequential audit.
The offline ERP decoder is retained as an explicit research opt-in for
reconstruction and interpretation and is never subtracted into a
classification bypass. Both research graphs default off at the model boundary;
the named research recipes must enable them deliberately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from models.component_decoder import ERPComponentDecoder, ERPComponentOutput
from models.component_window import ComponentWindow
from models.encoder import DEFAULT_ENCODER_DEPTH, Stage2Encoder
from models.erp_uncertainty import ERPTrialAggregate, inverse_variance_aggregate
from models.heads import HeadsOutput, MultiTaskHeads, Z2AuxiliaryHead
from models.innovation import (
    CausalInnovationDecoder,
    CausalInnovationOutput,
    CausalObservationEncoder,
)
from models.reference import WeightedRereference
from models.repetition import RepetitionEvidenceModel
from models.repetition_v12 import AdditiveRepetitionEvidence
from models.time_axis import EpochTimeAxis
from models.tokenizer import ERPTokenizer, TokenizerOutput
from models.transfer import LowRankDatasetAdapter, SharedPrivateEncoder


@dataclass
class StrictPastLikelihoodOutput:
    """Fixed-coordinate observation and its class-conditional past moments."""

    likelihood_observation: torch.Tensor
    observation_mask: torch.Tensor
    causal_innovation: CausalInnovationOutput
    mask_is_homogeneous: bool = False


@dataclass
class N2P3NetOutput:
    """N2P3-Net 完整前向输出。

    Attributes
    ----------
    heads : HeadsOutput
        多任务头输出（logit_target / logit_early / amplitude，及 p_target/p_early property）。
    tau : torch.Tensor
        (B,3) ``pcw_tau``：分类 routing 窗口中心（ms）。它不是生理潜伏期；
        生理潜伏期只由 ``measurement.LatencyMeasurement`` 提供
        ``measured_tau_posterior``。
    sigma : torch.Tensor
        (3,2) 成分窗宽（ms；[:,0]=σ_up、[:,1]=σ_down）。
    H : torch.Tensor
        (B,3,D) 成分表示（可解释性）。
    attention : torch.Tensor | None
        (B,3,T) 软窗分布（return_attention=True 时；可解释性 / Head-D 精确幅值用）。
    """

    heads: HeadsOutput | None
    tau: torch.Tensor
    sigma: torch.Tensor
    H: torch.Tensor
    attention: torch.Tensor | None
    features: torch.Tensor | None = None
    erp: ERPComponentOutput | None = None
    likelihood: StrictPastLikelihoodOutput | None = None
    canonical_covariance: torch.Tensor | None = None
    shared_features: torch.Tensor | None = None
    private_features: torch.Tensor | None = None
    domain_logits: torch.Tensor | None = None
    dataset_logits: torch.Tensor | None = None
    measurement_features: torch.Tensor | None = None
    measurement_contribution: torch.Tensor | None = None
    measured_tau_posterior: object | None = None

    @property
    def pcw_tau(self) -> torch.Tensor:
        """Classification routing window center; never a physiological latency."""
        return self.tau


@dataclass
class ERPDeploymentOutput:
    """Monte-Carlo predictive moments with explicit ERP latency semantics."""

    mean: torch.Tensor
    aleatoric_variance: torch.Tensor
    epistemic_variance: torch.Tensor
    total_variance: torch.Tensor
    anchor_latency_ms: torch.Tensor
    component_peak_latency_ms: torch.Tensor
    waveform_peak_latency_ms: torch.Tensor

    def aggregate_trials(
        self,
        *,
        trial_mask: torch.Tensor | None = None,
        variance_floor: float = 1e-6,
    ) -> ERPTrialAggregate:
        """Combine trial estimates using calibrated total predictive precision."""

        return inverse_variance_aggregate(
            self.mean,
            self.total_variance,
            dim=0,
            trial_mask=trial_mask,
            variance_floor=variance_floor,
        )


class N2P3Net(nn.Module):
    """Map sensor epochs to PCW, ERP and strict-past likelihood outputs.

    Parameters
    ----------
    n_channels : int
      channel_names : Sequence[str] | None
          通道名（非 8 导蒙太奇必填；GTN 原生 3 导传 ("Fz","Cz","Pz")）。
      head_dropout / encoder_dropout : float
          瘦头 dropout 与 encoder dropout（默认 0.25，EEGNet 借鉴）。
      spatial_max_norm : float | None
          tokenizer 空间权重 max-norm（默认 1.0；None 恢复旧行为）。
        通道数 C（默认 8）。
    d_model : int
        隐藏维 D（默认 64）。
    use_rereference : bool
        是否启用加权再参考（Stage 0.1，默认 True）。
    baseline_n : int | None
        基线段标准化用的基线点数。None 时由 tmin_ms/sfreq 推导（默认 −200ms@256Hz → 51 点，
        review v6 P1 消除硬编码与 T/tmin 的耦合）。
    tau0_bounds : Sequence[tuple[float, float]] | None
        每成分 τ0 的生理界（ms），None 使用 ComponentWindow 默认（review v6 P1）。
      tau0_ms : Sequence[float] | None
          每成分 τ0 先验中心（ms），None 使用 ComponentWindow 默认（成人先验 220/300/350）。
          GTN（儿童）由 runner 显式传 220/300/460（Phase 2 失败诊断：真实 P3b 峰值 ~460–490ms）。
    temporal_kernels : Sequence[int]
        tokenizer 多尺度时间卷积核长。
    filters_per_scale : int
        tokenizer 每尺度滤波器数。
    encoder_depth : int
        Stage 2 序列编码层数（消融轴，默认 4，配合 TCN dilation 1/4/16/32）。
    encoder_type : str
        "tcn"（默认，2026-08-20 决策）或 "conformer"（备选，消融对照）。
    tcn_dilations : Sequence[int] | None
        TCN 膨胀系数覆盖；None 时由 encoder_depth 自动生成 canonical prefix。
    d_chn_in / d_sub_in : int
        E_chn / E_sub 输入维（与 data/channel、data/metadata 对齐）。
      dtau_readout : str
          Δτ 读出路径（component_window.DTAU_READOUTS，正式默认 attention_softargmax）。
      encoder_norm : str
          TCN block 归一化（GLM 消融轴）："bn"（recipe 默认，BatchNorm1d，训练折统计/
          推理冻结）或 "ln"（构造器旧默认回退）。
      tokenizer_init : str
          时间卷积初始化（GLM v3）："random"（kaiming，默认旧行为）或 "bandpass"
          （Gabor 带通；诊断证据：随机 init 的滤波器从未学出 ERP 形状）。
      tokenizer_post_norm / tokenizer_post_act : str
          时间卷积后每尺度归一化/激活（GLM v3）："none"（默认旧行为）/"bn"；激活
          "none"/"elu"（默认）/"gelu"。
      sigma_bounds : Sequence[tuple[float, float]] | None
          PCW 每成分窗宽 σ 的 [lo, hi]（ms），None 用 ComponentWindow 默认
          （成人先验 N2 [20,50]、P3a/P3b [20,80]）。GTN 儿童的 P3b 宽达 300–650ms
          （ERP 实测 350–650ms 窗差值仍 11–14μV），runner 应传 P3b 上界 150。
    """

    def __init__(
        self,
        n_channels: int = 8,
        channel_names: Sequence[str] | None = None,
        d_model: int = 64,
        use_rereference: bool = True,
        baseline_n: int | None = None,
        baseline_mode: str = "trial",
        trial_reference_window_ms: tuple[float, float] | None = None,
        trial_reference_center: str = "mean",
        trial_reference_scale: str = "none",
        tmin_ms: float = -200.0,
        tmax_ms: float = 800.0,
        sfreq: float = 256.0,
        n_time: int = 256,
        temporal_kernels: Sequence[int] = (13, 33, 65, 129),
        filters_per_scale: int = 16,
        encoder_depth: int = DEFAULT_ENCODER_DEPTH,
        encoder_type: str = "tcn",
        encoder_dropout: float = 0.25,
        encoder_causal: bool = False,
        tcn_dilations: Sequence[int] | None = None,
        tcn_pointwise_execution: str = "linear",
        encoder_bn_momentum: float = 0.1,
        d_chn_in: int = 48,
        d_sub_in: int = 19,
        n_domains: int | None = None,
        tau0_bounds: Sequence[tuple[float, float]] | None = None,
        tau0_ms: Sequence[float] | None = None,
        dtau_readout: str = "attention_softargmax",
        dtau_bounds: Sequence[tuple[float, float]] | None = None,
        sigma_bounds: Sequence[tuple[float, float]] | None = None,
        encoder_norm: str = "ln",
        tokenizer_init: str = "random",
        tokenizer_post_norm: str = "none",
        tokenizer_post_act: str = "elu",
        tokenizer_temporal_spatial_fusion: bool = True,
        head_dropout: float = 0.25,
        use_z2_aux_head: bool = False,
        z2_aux_head_mode: str = "add",
        z2_aux_pool: str = "attention",
        z2_aux_dropout: float = 0.25,
        spatial_max_norm: float | None = 1.0,
        component_decoder: bool = False,
        use_innovation_likelihood: bool = False,
        innovation_d_model: int = 28,
        innovation_kernel_size: int = 9,
        innovation_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        innovation_dropout: float = 0.1,
        innovation_covariance_rank: int = 2,
        use_repetition_evidence: bool = False,
        repetition_hidden_size: int = 24,
        repetition_v12: bool = False,
        repetition_state_residual: bool = False,
        use_measurement_windows: bool = False,
        measurement_anchor_ms: float = 460.0,
        measurement_grid_radius_ms: float = 60.0,
        measurement_grid_step_ms: float = 0.5,
        measurement_window_width_ms: float = 50.0,
        measurement_refit_epochs: int = 5,
        channel_positions_m: Sequence[Sequence[float]] | None = None,
        canonical_channel_names: Sequence[str] | None = None,
        canonical_positions_m: Sequence[Sequence[float]] | None = None,
        canonical_noise_variance: float | Sequence[float] = 0.05,
        canonical_length_scale: float = 0.055,
        canonical_residual_attention: bool = True,
        canonical_residual_limit: float = 0.10,
        dataset_adapter_rank: int = 0,
        shared_private: bool = False,
        private_dim: int | None = None,
        grl_scale: float = 1.0,
        task_head_shared_only: bool = False,
    ):
        super().__init__()
        self.validate_input_finite = True
        time_axis = EpochTimeAxis(
            tmin_ms=float(tmin_ms),
            tmax_ms=float(tmax_ms),
            sfreq=float(sfreq),
            n_time=int(n_time),
        )
        if innovation_d_model < 1:
            raise ValueError("innovation_d_model must be positive.")
        if dataset_adapter_rank < 0 or dataset_adapter_rank > d_model:
            raise ValueError("dataset_adapter_rank must lie in [0,d_model].")
        if (dataset_adapter_rank > 0 or shared_private) and (n_domains is None or n_domains < 2):
            raise ValueError("Dataset adapters and shared/private transfer require n_domains>=2.")
        # GLM v3.1（2026-08-24 BNCI 排查）：标准化策略按「有无真基线段」选择。
        # trial（默认，GTN）：(X−μ_b)/σ_b，μ/σ 取前 baseline_n 点——要求 epoch 含
        #   pre-stimulus 基线段（GTN −200~0ms 共 51 点，σ 稳定）。
        # mean_only：只减 μ_b 不除 σ——基线点数过少（如 4 点）时 σ 估计是噪声放大器
        #   （BNCI 单 fold 实测：除 σ 0.7365 → 不除 0.8177，+8.1pt）。
        # trial_reference：使用显式物理时间窗的逐试次参考；可用于没有刺激前段的
        # stimulus-locked 数据，默认只中心化，不把短参考窗的噪声当作尺度。
        # none：恒等——配合适配器层 fold 级 z-score（EEGNet 式预处理；
        #   BNCI 实测最优 0.8346，与 EEGNet 持平）。
        if baseline_mode not in ("trial", "mean_only", "none", "trial_reference"):
            raise ValueError(
                "baseline_mode 须为 trial/mean_only/none/trial_reference，"
                f"得到 {baseline_mode!r}。"
            )
        if baseline_mode == "trial" and tmin_ms >= 0.0:
            raise ValueError(
                "baseline_mode='trial' requires a true pre-stimulus interval (tmin_ms < 0)."
            )
        if trial_reference_center not in ("mean", "median"):
            raise ValueError("trial_reference_center 须为 mean 或 median。")
        if trial_reference_scale not in ("none", "std", "mad"):
            raise ValueError("trial_reference_scale 须为 none、std 或 mad。")

        self.trial_reference_slice: tuple[int, int] | None = None
        if baseline_mode == "trial_reference":
            if trial_reference_window_ms is None or len(trial_reference_window_ms) != 2:
                raise ValueError(
                    "baseline_mode='trial_reference' requires "
                    "trial_reference_window_ms=(start_ms,end_ms)."
                )
            start_ms, end_ms = (float(trial_reference_window_ms[0]), float(trial_reference_window_ms[1]))
            if not math.isfinite(start_ms) or not math.isfinite(end_ms) or start_ms >= end_ms:
                raise ValueError("trial_reference_window_ms must be a finite increasing interval.")
            if start_ms < tmin_ms or end_ms > tmax_ms:
                raise ValueError(
                    "trial_reference_window_ms must lie inside the physical epoch time axis."
                )
            # The right edge is exclusive, matching EpochTimeAxis and the cache contract.
            start_index = int(math.ceil((start_ms - tmin_ms) * sfreq / 1000.0 - 1e-9))
            end_index = int(math.ceil((end_ms - tmin_ms) * sfreq / 1000.0 - 1e-9))
            if end_index - start_index < 2:
                raise ValueError("trial_reference window must contain at least two samples.")
            self.trial_reference_slice = (start_index, end_index)
            baseline_n = end_index - start_index
        elif baseline_n is None:
            if tmin_ms < 0.0:
                baseline_n = max(1, int(round(-tmin_ms / 1000.0 * sfreq)))
            elif baseline_mode == "none":
                baseline_n = 0
            else:
                raise ValueError(
                    "baseline_n must be explicit when no pre-stimulus interval exists."
                )
        if not 0 <= int(baseline_n) <= int(n_time):
            raise ValueError(f"baseline_n must be in [0,n_time], got {baseline_n}/{n_time}.")
        if baseline_mode == "trial" and int(baseline_n) < 2:
            raise ValueError("baseline_mode='trial' needs at least two baseline samples.")
        self.baseline_n = int(baseline_n)
        self.baseline_mode = baseline_mode
        self.trial_reference_center = trial_reference_center
        self.trial_reference_scale = trial_reference_scale
        self.time_axis = time_axis
        self.measurement_anchor_ms = float(measurement_anchor_ms)
        self.measurement_grid_radius_ms = float(measurement_grid_radius_ms)
        self.measurement_grid_step_ms = float(measurement_grid_step_ms)
        self.measurement_window_width_ms = float(measurement_window_width_ms)
        self.measurement_refit_epochs = int(measurement_refit_epochs)
        self.n_channels = int(n_channels)
        self.n_domains = int(n_domains) if n_domains is not None else None
        self.task_head_shared_only = bool(task_head_shared_only)
        self.tmin_ms = float(tmin_ms)
        self.tmax_ms = float(tmax_ms)
        # Internal temporal modules retain their historical attribute names;
        # values remain milliseconds and the public constructor is explicit.
        self.tmin = self.tmin_ms
        self.tmax = self.tmax_ms
        self.sfreq = float(sfreq)
        self.n_time = int(n_time)
        # Stage 0.1 加权再参考（可选；GLM v2：门控参考层 + 按域条件化，见 reference.py）
        self.reference = (
            WeightedRereference(n_channels, n_domains=n_domains) if use_rereference else None
        )

        # Stage 1 时空 token 化（v5.1：原生通道名 + spatial max-norm；GLM v3：带通 init + BN/ELU）
        self.tokenizer = ERPTokenizer(
            n_channels=n_channels,
            channel_names=channel_names,
            spatial_max_norm=spatial_max_norm,
            d_model=d_model,
            temporal_kernels=temporal_kernels,
            filters_per_scale=filters_per_scale,
            d_chn_in=d_chn_in,
            d_sub_in=d_sub_in,
            tmin=self.tmin_ms,
            tmax=self.tmax_ms,
            n_time=self.n_time,
            sfreq=self.sfreq,
            init=tokenizer_init,
            post_norm=tokenizer_post_norm,
            post_act=tokenizer_post_act,
            temporal_spatial_fusion=tokenizer_temporal_spatial_fusion,
            channel_positions_m=channel_positions_m,
            canonical_channel_names=canonical_channel_names,
            canonical_positions_m=canonical_positions_m,
            canonical_noise_variance=canonical_noise_variance,
            canonical_length_scale=canonical_length_scale,
            canonical_residual_attention=canonical_residual_attention,
            canonical_residual_limit=canonical_residual_limit,
        )
        self.channel_names = tuple(self.tokenizer.channel_names)
        self.canonical_channel_names = tuple(self.tokenizer.spatial_channel_names)

        # Stage 2 序列编码（域条件仿射暴露，P1：n_domains 传给 encoder）+ 参数化成分窗
        self.encoder = Stage2Encoder(
            d_model=d_model,
            depth=encoder_depth,
            encoder_type=encoder_type,
            dropout=encoder_dropout,
            n_domains=n_domains,
            tcn_dilations=tcn_dilations,
            tcn_causal=encoder_causal,
            norm=encoder_norm,
            tcn_pointwise_execution=tcn_pointwise_execution,
            bn_momentum=encoder_bn_momentum,
        )
        self.dataset_adapter = (
            LowRankDatasetAdapter(d_model, int(n_domains), rank=dataset_adapter_rank)
            if dataset_adapter_rank > 0
            else None
        )
        self.shared_private_encoder = (
            SharedPrivateEncoder(
                d_model,
                int(n_domains),
                private_dim=private_dim,
                grl_scale=grl_scale,
            )
            if shared_private
            else None
        )
        cw_kwargs: dict = {
            "d_model": d_model,
            "dtau_readout": dtau_readout,
            "tmin": self.tmin_ms,
            "tmax": self.tmax_ms,
        }
        if tau0_bounds is not None:
            cw_kwargs["tau0_bounds"] = tau0_bounds
        if tau0_ms is not None:
            cw_kwargs["tau0_ms"] = tau0_ms
        if dtau_bounds is not None:
            cw_kwargs["dtau_bounds"] = dtau_bounds
        if sigma_bounds is not None:
            cw_kwargs["sigma_bounds"] = sigma_bounds
        self.component_window = ComponentWindow(**cw_kwargs)
        self.component_decoder = (
            ERPComponentDecoder(
                d_model=d_model,
                n_channels=n_channels,
                tmin_ms=self.tmin_ms,
                sfreq=self.sfreq,
                n_time=self.n_time,
            )
            if component_decoder
            else None
        )

        # The likelihood graph is parameter-disjoint and research-opt-in.
        self.innovation_encoder = (
            CausalObservationEncoder(
                n_channels=n_channels,
                d_model=innovation_d_model,
                kernel_size=innovation_kernel_size,
                dilations=innovation_dilations,
                dropout=innovation_dropout,
                normalize_io=False,
            )
            if use_innovation_likelihood
            else None
        )
        self.innovation_decoder = (
            CausalInnovationDecoder(
                innovation_d_model,
                n_channels,
                covariance_rank=innovation_covariance_rank,
            )
            if use_innovation_likelihood
            else None
        )
        if use_z2_aux_head and z2_aux_head_mode not in ("add", "replace"):
            raise ValueError("z2_aux_head_mode must be 'add' or 'replace' when enabled.")
        # Stage 3 PCW-constrained heads.
        self.heads = MultiTaskHeads(
            d_model=d_model,
            dropout=head_dropout,
        )
        # Full-Z2 auxiliary classifier is research-only and disabled in the
        # production recipe. When a named research recipe enables it, the
        # forward pass combines or replaces the PCW logit accordingly while
        # keeping logit_pcw and the interpretable readouts intact.
        self.z2_aux_head = (
            Z2AuxiliaryHead(
                d_model=d_model,
                pool=z2_aux_pool,
                dropout=z2_aux_dropout,
            )
            if use_z2_aux_head
            else None
        )
        self.z2_aux_head_mode = z2_aux_head_mode if use_z2_aux_head else "off"
        self.measurement_head = (
            nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, 1, bias=False))
            if use_measurement_windows
            else None
        )
        # Fail-closed measurement consumption: the detached posterior-window
        # contribution is multiplied by a zero-initialized gain. Only the
        # outer fold pipeline may raise it after the nested M0/M1 gate passes.
        if use_measurement_windows:
            self.register_buffer("measurement_gain", torch.zeros(()))
        else:
            self.measurement_gain = torch.zeros(())
        self.repetition_evidence = (
            AdditiveRepetitionEvidence(
                hidden_size=repetition_hidden_size,
                state_residual=repetition_state_residual,
            )
            if use_repetition_evidence and repetition_v12
            else RepetitionEvidenceModel(hidden_size=repetition_hidden_size)
            if use_repetition_evidence
            else None
        )

    def preprocess_input(
        self,
        X: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the exact Stage-0 transform used by ``forward``."""
        if channel_mask is not None:
            if channel_mask.dtype != torch.bool:
                raise ValueError("channel_mask must have boolean dtype for Stage 0.")
            mask = channel_mask.to(device=X.device, dtype=X.dtype)
            if mask.shape == (X.shape[1],):
                mask = mask[None].expand(X.shape[0], -1)
            if mask.shape != X.shape[:2]:
                raise ValueError("channel_mask must be (C,) or (B,C) for Stage 0.")
            finite = torch.isfinite(X) | ~mask.to(dtype=torch.bool)[:, :, None]
        else:
            finite = torch.isfinite(X)
        # The eager check gives callers a precise data-contract error. During
        # torch.compile tracing it would force Tensor.item() and split the graph;
        # fold data is already validated before entering the compiled model.
        if self.validate_input_finite and not torch.compiler.is_compiling() and not bool(finite.all()):
            raise ValueError(
                "Observed Stage-0 samples must be finite; mark missing channels explicitly."
            )
        if channel_mask is not None:
            # IEEE NaN * 0 is still NaN.  Missing channels are semantic
            # absence, so replace their storage values instead of multiplying.
            X = torch.where(mask.to(dtype=torch.bool)[:, :, None], X, torch.zeros_like(X))
        X0 = (
            self.reference(X, channel_mask, domain_id=domain_id)
            if self.reference is not None
            else X
        )
        return self._baseline_standardize(X0)

    def preprocess_likelihood_observation(
        self,
        X: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map epochs to the fixed coordinate system scored by the likelihood.

        Learned rereferencing is intentionally excluded. Without the transform
        Jacobian, allowing likelihood gradients to change sensor coordinates
        would make NLL values incomparable and create a scale shortcut.
        """

        if X.dim() != 3:
            raise ValueError("Likelihood input must be (B,C,T).")
        if channel_mask is None:
            finite = torch.isfinite(X)
        else:
            if channel_mask.dtype != torch.bool:
                raise ValueError("channel_mask must have boolean dtype for likelihood input.")
            finite_mask = channel_mask.to(device=X.device, dtype=torch.bool)
            if finite_mask.shape == (X.shape[1],):
                finite_mask = finite_mask[None].expand(X.shape[0], -1)
            if finite_mask.shape != X.shape[:2]:
                raise ValueError("channel_mask must be (C,) or (B,C) for likelihood input.")
            finite = torch.isfinite(X) | ~finite_mask[:, :, None]
        if self.validate_input_finite and not torch.compiler.is_compiling() and not bool(finite.all()):
            raise ValueError(
                "Observed likelihood samples must be finite; mark missing channels explicitly."
            )
        observation = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if channel_mask is not None:
            mask = channel_mask.to(device=X.device, dtype=X.dtype)
            if mask.shape == (X.shape[1],):
                mask = mask[None].expand(X.shape[0], -1)
            if mask.shape != X.shape[:2]:
                raise ValueError("channel_mask must be (C,) or (B,C) for likelihood input.")
            observation = observation * mask[:, :, None]
        return self._baseline_standardize(observation)

    def _baseline_standardize(self, X: torch.Tensor) -> torch.Tensor:
        """基线段标准化（D-baseline + GLM v3.1 baseline_mode）。

        trial：(X − μ_b)/σ_b（μ/σ 取前 baseline_n 点逐通道；须有真 pre-stimulus 基线段）。
        mean_only：(X − μ_b)（σ 点数不足时防噪声放大）。
        trial_reference：按显式物理时间窗逐试次中心化，可选 std 或 robust MAD 尺度。
        none：恒等（配合适配器层 fold 级 z-score）。
        """
        if getattr(self, "baseline_mode", "trial") == "none":
            return X
        if self.baseline_mode == "trial_reference":
            if self.trial_reference_slice is None:
                raise RuntimeError("trial_reference mode was not initialized with a time window.")
            start, stop = self.trial_reference_slice
            b = X[:, :, start:stop]
        else:
            b = X[:, :, : self.baseline_n]  # (B, C, n_b)
        mu = b.mean(dim=2, keepdim=True)  # (B, C, 1)
        if self.baseline_mode == "trial_reference":
            if self.trial_reference_center == "median":
                mu = b.median(dim=2, keepdim=True).values
            centered = X - mu
            if self.trial_reference_scale == "none":
                return centered
            if self.trial_reference_scale == "mad":
                scale = (
                    (b - mu).abs().median(dim=2, keepdim=True).values * 1.4826
                )
            else:
                scale = b.std(dim=2, keepdim=True)
            return centered / scale.clamp(min=1e-6)
        if getattr(self, "baseline_mode", "trial") == "mean_only":
            return X - mu
        std = b.std(dim=2, keepdim=True).clamp(min=1e-6)  # (B, C, 1) 防除零（D-std-clamp）
        return (X - mu) / std

    def forward(
        self,
        X: torch.Tensor,
        E_chn: torch.Tensor | None = None,
        E_sub: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
        return_attention: bool = False,
        return_heads: bool = True,
        return_likelihood: bool = True,
        likelihood_input: torch.Tensor | None = None,
        likelihood_channel_mask: torch.Tensor | None = None,
        likelihood_class_means: torch.Tensor | None = None,
        likelihood_labels: torch.Tensor | None = None,
        measurement_window: torch.Tensor | None = None,
        measurement_posterior: object | None = None,
    ) -> N2P3NetOutput:
        """X (B,C,T) → N2P3NetOutput。

        channel_mask : (C,) bool 可选，缺失通道掩码（review v3 P0，传 reference 重归一化）。
        domain_id : (B,) long 可选，域条件仿射的域标签（Phase 3）。
        return_heads : bool
            L_jit 的 shifted forward 只需 τ，传 False 跳过 Head-A/B/D 以节省算力。
        """
        if X.dim() != 3:
            raise ValueError(f"X must be (B,C,T), got {tuple(X.shape)}.")
        if X.shape[1] != self.n_channels:
            raise ValueError(
                f"X has {X.shape[1]} channels; model contract requires {self.n_channels}."
            )
        if X.shape[2] != self.n_time:
            raise ValueError(
                f"X has {X.shape[2]} time samples; physical time-axis contract requires "
                f"n_time={self.n_time}. Resample/crop explicitly instead of changing the "
                "effective sampling interval inside the model."
            )
        if likelihood_input is not None and likelihood_input.shape != X.shape:
            raise ValueError("likelihood_input must match X shape exactly.")
        if likelihood_channel_mask is None:
            likelihood_channel_mask = channel_mask
        if likelihood_class_means is not None and likelihood_class_means.shape != (
            2,
            self.n_channels,
            self.n_time,
        ):
            raise ValueError("likelihood_class_means must be (2,C,T).")
        if likelihood_labels is not None:
            likelihood_labels = likelihood_labels.reshape(-1).to(device=X.device)
            if likelihood_labels.numel() != X.shape[0]:
                raise ValueError("likelihood_labels must have one value per trial.")
            if not bool(((likelihood_labels == 0) | (likelihood_labels == 1)).all()):
                raise ValueError("likelihood_labels must contain only binary values 0 and 1.")
            likelihood_labels = likelihood_labels.long()
        if measurement_window is not None and self.measurement_head is None:
            raise ValueError("measurement_window requires use_measurement_windows=True.")
        if measurement_window is not None and measurement_window.shape != (X.shape[0], self.n_time):
            raise ValueError("measurement_window must be (B,T).")
        # Stage 0：门控加权再参考（mask 重归一化 + 按域条件化）+ 基线段标准化
        X0 = self.preprocess_input(X, channel_mask=channel_mask, domain_id=domain_id)

        # Stage 1：token 化
        tokenized = self.tokenizer(
            X0,
            E_chn,
            E_sub,
            channel_mask=channel_mask,
            return_details=True,
        )
        if not isinstance(tokenized, TokenizerOutput):
            raise RuntimeError("Tokenizer detail contract was not honored.")
        Z = tokenized.tokens

        # Stage 2：序列编码（域条件仿射）+ 参数化成分窗
        Z2 = self.encoder(Z, domain_id=domain_id)
        if self.dataset_adapter is not None:
            Z2 = self.dataset_adapter(Z2, domain_id)
        shared_features = Z2.mean(dim=1)
        private_features = None
        domain_logits = None
        dataset_logits = None
        if self.shared_private_encoder is not None:
            split = self.shared_private_encoder(Z2)
            Z2 = split.shared_sequence
            shared_features = split.shared
            private_features = split.private
            domain_logits = split.domain_logits
            dataset_logits = split.dataset_logits
        need_erp = return_heads and self.component_decoder is not None
        if return_attention or need_erp:
            H, tau, sigma, A_internal = self.component_window(Z2, return_attention=True)
        else:
            H, tau, sigma = self.component_window(Z2)
            A_internal = None

        erp_output = None
        if need_erp:
            if self.component_decoder is None or A_internal is None:
                raise RuntimeError("ERP decomposition construction contract is incomplete.")
            erp_output = self.component_decoder(H, tau, sigma)
        likelihood_output = None
        if return_heads and return_likelihood and self.innovation_encoder is not None:
            if self.innovation_decoder is None:
                raise RuntimeError("Innovation likelihood decoder is not enabled.")
            likelihood_observation = self.preprocess_likelihood_observation(
                X if likelihood_input is None else likelihood_input,
                channel_mask=likelihood_channel_mask,
            ).detach()
            class_means = (
                torch.zeros(
                    2,
                    self.n_channels,
                    self.n_time,
                    device=likelihood_observation.device,
                    dtype=likelihood_observation.dtype,
                )
                if likelihood_class_means is None
                else likelihood_class_means.to(
                    device=likelihood_observation.device,
                    dtype=likelihood_observation.dtype,
                )
            )
            if likelihood_channel_mask is None:
                likelihood_mask = torch.ones(
                    X.shape[0],
                    self.n_channels,
                    device=likelihood_observation.device,
                    dtype=torch.bool,
                )
            else:
                likelihood_mask = likelihood_channel_mask.to(
                    device=likelihood_observation.device,
                    dtype=torch.bool,
                )
                if likelihood_mask.shape == (self.n_channels,):
                    likelihood_mask = likelihood_mask[None].expand(X.shape[0], -1)
                if likelihood_mask.shape != X.shape[:2]:
                    raise ValueError("likelihood_channel_mask must be (C,) or (B,C).")
            if likelihood_labels is None:
                hypothesis_residual = likelihood_observation[:, None] - class_means[None]
            else:
                hypothesis_residual = (
                    likelihood_observation - class_means[likelihood_labels]
                )[:, None]
            hypothesis_residual = hypothesis_residual * likelihood_mask[:, None, :, None].to(
                likelihood_observation.dtype
            )
            flattened_hypotheses = hypothesis_residual.flatten(0, 1)
            innovation_features = self.innovation_encoder(flattened_hypotheses).reshape(
                X.shape[0], hypothesis_residual.shape[1], self.n_time, -1
            )
            causal_innovation = self.innovation_decoder(
                innovation_features,
                channel_mask=likelihood_mask,
                hypothesis_labels=likelihood_labels,
            )
            likelihood_output = StrictPastLikelihoodOutput(
                likelihood_observation=likelihood_observation,
                observation_mask=likelihood_mask,
                causal_innovation=causal_innovation,
                mask_is_homogeneous=(
                    likelihood_channel_mask is None or likelihood_channel_mask.dim() == 1
                ),
            )

        heads_out = self.heads(H) if return_heads else None
        if heads_out is not None and self.z2_aux_head is not None:
            logit_aux = self.z2_aux_head(Z2)
            if self.z2_aux_head_mode == "replace":
                logit_target = logit_aux
            else:
                logit_target = heads_out.logit_pcw + logit_aux
            heads_out = HeadsOutput(
                logit_target=logit_target,
                logit_pcw=heads_out.logit_pcw,
                logit_early=heads_out.logit_early,
                amplitude=heads_out.amplitude,
                logit_aux=logit_aux,
            )
        measurement_features = None
        measurement_contribution = None
        if measurement_window is not None:
            window = measurement_window.to(device=Z2.device, dtype=Z2.dtype).detach()
            measurement_features = torch.einsum("bt,btd->bd", window, Z2)
            if heads_out is not None and self.measurement_head is not None:
                contribution = self.measurement_gain.to(
                    device=Z2.device, dtype=Z2.dtype
                ) * self.measurement_head(measurement_features)
                measurement_contribution = contribution.detach()
                heads_out = HeadsOutput(
                    logit_target=heads_out.logit_target + contribution,
                    logit_pcw=heads_out.logit_pcw + contribution,
                    logit_early=heads_out.logit_early,
                    amplitude=heads_out.amplitude,
                    logit_aux=heads_out.logit_aux,
                )

        return N2P3NetOutput(
            heads=heads_out,
            tau=tau,
            sigma=sigma,
            H=H,
            attention=A_internal if return_attention else None,
            features=Z2,
            erp=erp_output,
            likelihood=likelihood_output,
            canonical_covariance=tokenized.canonical_covariance,
            shared_features=shared_features,
            private_features=private_features,
            domain_logits=domain_logits,
            dataset_logits=dataset_logits,
            measurement_features=measurement_features,
            measurement_contribution=measurement_contribution,
            measured_tau_posterior=measurement_posterior,
        )

    def num_parameters(self) -> int:
        """可学习参数量（参数账，D-budget）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict_erp_uncertainty(
        self,
        X: torch.Tensor,
        E_chn: torch.Tensor | None = None,
        E_sub: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
        *,
        mc_samples: int = 16,
    ) -> ERPDeploymentOutput:
        """Combine decoder aleatoric moments with MC-dropout epistemic variance."""

        if mc_samples < 2:
            raise ValueError("mc_samples must be at least two for epistemic variance.")
        was_training = self.training
        self.eval()
        dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
        for module in self.modules():
            if isinstance(module, dropout_types):
                module.train()

        reconstructions: list[torch.Tensor] = []
        aleatoric: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        component_peaks: list[torch.Tensor] = []
        try:
            for _ in range(mc_samples):
                output = self(
                    X,
                    E_chn,
                    E_sub,
                    channel_mask=channel_mask,
                    domain_id=domain_id,
                    return_attention=False,
                    return_heads=True,
                )
                if output.erp is None:
                    raise RuntimeError(
                        "ERP uncertainty prediction requires component_decoder=True."
                    )
                erp = output.erp
                reconstructions.append(erp.reconstruction)
                aleatoric.append(erp.waveform_variance)
                anchors.append(erp.anchor_latency_ms)
                component_peaks.append(erp.component_peak_latency_ms)
        finally:
            self.train(was_training)

        samples = torch.stack(reconstructions, dim=0)
        mean = samples.mean(dim=0)
        epistemic_variance = samples.var(dim=0, unbiased=False)
        aleatoric_variance = torch.stack(aleatoric, dim=0).mean(dim=0)
        times = self.component_decoder.times_ms.to(device=mean.device, dtype=mean.dtype)
        waveform_peak_latency_ms = times[mean.argmax(dim=-1)]
        return ERPDeploymentOutput(
            mean=mean,
            aleatoric_variance=aleatoric_variance,
            epistemic_variance=epistemic_variance,
            total_variance=aleatoric_variance + epistemic_variance,
            anchor_latency_ms=torch.stack(anchors, dim=0).mean(dim=0),
            component_peak_latency_ms=torch.stack(component_peaks, dim=0).mean(dim=0),
            waveform_peak_latency_ms=waveform_peak_latency_ms,
        )
