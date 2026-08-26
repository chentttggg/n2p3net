"""Canonical Neural-RIDE architecture and training recipes.

Experiment entry points may describe a dataset and request explicit ablations,
but they must not duplicate or silently redefine the Neural-RIDE defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from models.component_window import (
    PCW_CANONICAL_DTAU_BOUNDS,
    PCW_CANONICAL_SIGMA_BOUNDS,
    PCW_CANONICAL_TAU0_BOUNDS,
    PCW_CANONICAL_TAU0_MS,
)
from models.encoder import DEFAULT_ENCODER_DEPTH
from train.batch import TrainingBatchConfig
from train.trainer import TrainerConfig


@dataclass(frozen=True)
class TaskCapabilities:
    """Supervision that is physically present in a dataset."""

    name: str
    has_digit_sets: bool
    pos_weight: float

    def __post_init__(self) -> None:
        if self.pos_weight <= 0.0:
            raise ValueError("TaskCapabilities.pos_weight must be positive.")


GTN_DIGIT_TASK = TaskCapabilities("gtn_digit_selection", has_digit_sets=True, pos_weight=8.0)
BINARY_ODDBALL_TASK = TaskCapabilities("binary_oddball", has_digit_sets=False, pos_weight=5.0)


@dataclass(frozen=True)
class NeuralRideRecipe:
    """Single source of truth for one Neural-RIDE protocol version."""

    name: str = "neural_ride_v12_pcw_fail_closed"

    # Architecture defaults shared by every dataset.
    d_model: int = 64
    # dep is the sole Stage 2 capacity setting; canonical TCN dilation is derived downstream.
    encoder_depth: int = DEFAULT_ENCODER_DEPTH
    encoder_type: str = "tcn"
    encoder_norm: str = "bn"
    # Preserve the sample-scale EMA horizon of momentum=0.1 at batch 1024.
    encoder_bn_momentum: float = 1.0 - 0.9 ** 0.25
    encoder_dropout: float = 0.25
    # At physical batch 256, retaining B,D,T across all TCN blocks removes
    # pointwise layout copies and wins the two-order steady-state benchmark.
    tcn_pointwise_execution: str = "conv1d"
    tokenizer_init: str = "bandpass"
    tokenizer_post_norm: str = "none"
    tokenizer_post_act: str = "none"
    tokenizer_temporal_spatial_fusion: bool = True
    use_rereference: bool = True
    head_dropout: float = 0.25
    spatial_max_norm: float = 1.0
    component_decoder: bool = False
    use_innovation_likelihood: bool = False
    innovation_d_model: int = 28
    innovation_kernel_size: int = 9
    innovation_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    innovation_dropout: float = 0.1
    innovation_covariance_rank: int = 2
    use_repetition_evidence: bool = True
    repetition_hidden_size: int = 24
    # Direct construction follows the active production contract. Legacy-v11
    # keeps its old semantics through an explicit override below.
    repetition_v12: bool = True
    repetition_state_residual: bool = False
    repetition_state_residual_l2_weight: float = 0.0
    repetition_v12_evidence_ks: tuple[int, ...] = (1, 3, 5)
    repetition_v12_evidence_weights: tuple[float, ...] = (0.34, 0.33, 0.33)
    # Object L is opt-in and gated: the model only instantiates its detached
    # measurement-window consumer when the fold pipeline requests it.
    use_measurement_windows: bool = False
    measurement_anchor_ms: float = 460.0
    measurement_grid_radius_ms: float = 60.0
    measurement_grid_step_ms: float = 0.5
    measurement_window_width_ms: float = 50.0
    measurement_refit_epochs: int = 5
    canonical_channel_names: tuple[str, ...] | None = None
    canonical_noise_variance: float = 0.05
    canonical_length_scale: float = 0.055
    canonical_residual_attention: bool = True
    canonical_residual_limit: float = 0.10
    dataset_adapter_rank: int = 0
    shared_private: bool = False
    private_dim: int | None = None
    task_head_shared_only: bool = False
    tau0_ms: tuple[float, ...] = PCW_CANONICAL_TAU0_MS
    tau0_bounds: tuple[tuple[float, float], ...] = PCW_CANONICAL_TAU0_BOUNDS
    sigma_bounds: tuple[tuple[float, float], ...] = PCW_CANONICAL_SIGMA_BOUNDS
    dtau_bounds: tuple[tuple[float, float], ...] = PCW_CANONICAL_DTAU_BOUNDS
    # Full-Z2 auxiliary trial head. Research-only E5 claim-gate contrast;
    # the production recipe keeps it disabled and PCW-only.
    use_z2_aux_head: bool = False
    z2_aux_head_mode: str = "add"
    z2_aux_pool: str = "attention"
    z2_aux_dropout: float = 0.25

    # Training defaults shared by every task unless capability-gated below.
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    lr_warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.10
    erp_decoder_lr_multiplier: float = 5.0
    weight_decay: float = 2.5e-5
    lambda2: float = 0.3
    lambda3: float = 0.0
    lambda_pcw: float = 0.3
    lambda_digit: float = 0.2
    lambda_conditional_nll: float = 0.10
    repetition_reliability_aux_weight: float = 1.0
    repetition_reliability_lr_multiplier: float = 10.0
    repetition_refit_epochs: int = 5
    auto_pos_weight: bool = True
    digit_evidence_ks: tuple[int, ...] = (1, 3, 5, 10, 15)
    digit_evidence_weights: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.45)
    lambda_amp: float = 0.0
    # Direct class-contrast loss is normalized by ERP-template energy. A zero
    # decoder therefore contributes 2.0 (waveform + spectrum), so unit weight
    # keeps the offline ERP interpretation objective numerically visible.
    lambda_recon: float = 0.0
    recon_waveform_weight: float = 1.0
    recon_projection_weight: float = 1.0
    recon_nll_weight: float = 0.1
    # The likelihood graph is parameter-disjoint and consumes detached fixed-
    # coordinate observations. Unit weight is its natural proper-score scale;
    # shrinking it would only shrink that graph's effective learning rate.
    lambda_innovation: float = 0.0
    innovation_score_interval_ms: tuple[float, float] = (0.0, 800.0)
    innovation_ar_order: int = 32
    recon_bootstrap_samples: int = 128
    recon_split_half_repeats: int = 32
    lambda_morphology_l0: float = 0.0
    variance_warmup_epochs: int = 5
    variance_ramp_epochs: int = 10
    lambda_jit: float = 0.0
    jit_prob: float = 0.0
    jit_max_ms: float = 40.0
    augment: bool = False
    early_stop_patience: int = 6
    track_pcw_gradients: bool = True
    recalibrate_batch_norm: bool = True
    lambda_orth: float = 0.0
    lambda_adv: float = 0.0
    lambda_private: float = 0.0
    reconstruct_all_domains: bool = False

    def model_kwargs(
        self,
        *,
        n_channels: int,
        channel_names: tuple[str, ...],
        tmin_ms: float,
        tmax_ms: float,
        sfreq: float,
        n_time: int,
        baseline_mode: str,
        trial_reference_window_ms: tuple[float, float] | None = None,
        trial_reference_center: str = "mean",
        trial_reference_scale: str = "none",
        channel_positions_m: tuple[tuple[float, float, float], ...] | None = None,
        tau0_ms: tuple[float, ...] | None = None,
        tau0_bounds: tuple[tuple[float, float], ...] | None = None,
        sigma_bounds: tuple[tuple[float, float], ...] | None = None,
        dtau_bounds: tuple[tuple[float, float], ...] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build model kwargs from a physical dataset contract plus explicit ablations."""

        kwargs: dict[str, Any] = {
            "n_channels": int(n_channels),
            "channel_names": tuple(channel_names),
            "d_model": self.d_model,
            "encoder_depth": self.encoder_depth,
            "encoder_type": self.encoder_type,
            "encoder_norm": self.encoder_norm,
            "encoder_bn_momentum": self.encoder_bn_momentum,
            "encoder_dropout": self.encoder_dropout,
            "tcn_pointwise_execution": self.tcn_pointwise_execution,
            "tokenizer_init": self.tokenizer_init,
            "tokenizer_post_norm": self.tokenizer_post_norm,
            "tokenizer_post_act": self.tokenizer_post_act,
            "tokenizer_temporal_spatial_fusion": self.tokenizer_temporal_spatial_fusion,
            "use_rereference": self.use_rereference,
            "head_dropout": self.head_dropout,
            "use_z2_aux_head": self.use_z2_aux_head,
            "z2_aux_head_mode": self.z2_aux_head_mode,
            "z2_aux_pool": self.z2_aux_pool,
            "z2_aux_dropout": self.z2_aux_dropout,
            "spatial_max_norm": self.spatial_max_norm,
            "component_decoder": self.component_decoder,
            "use_innovation_likelihood": self.use_innovation_likelihood,
            "innovation_d_model": self.innovation_d_model,
            "innovation_kernel_size": self.innovation_kernel_size,
            "innovation_dilations": self.innovation_dilations,
            "innovation_dropout": self.innovation_dropout,
            "innovation_covariance_rank": self.innovation_covariance_rank,
            "use_repetition_evidence": self.use_repetition_evidence,
            "repetition_hidden_size": self.repetition_hidden_size,
            "repetition_v12": self.repetition_v12,
            "repetition_state_residual": self.repetition_state_residual,
            "use_measurement_windows": self.use_measurement_windows,
            "measurement_anchor_ms": self.measurement_anchor_ms,
            "measurement_grid_radius_ms": self.measurement_grid_radius_ms,
            "measurement_grid_step_ms": self.measurement_grid_step_ms,
            "measurement_window_width_ms": self.measurement_window_width_ms,
            "measurement_refit_epochs": self.measurement_refit_epochs,
            "channel_positions_m": channel_positions_m,
            "canonical_channel_names": self.canonical_channel_names,
            "canonical_noise_variance": self.canonical_noise_variance,
            "canonical_length_scale": self.canonical_length_scale,
            "canonical_residual_attention": self.canonical_residual_attention,
            "canonical_residual_limit": self.canonical_residual_limit,
            "dataset_adapter_rank": self.dataset_adapter_rank,
            "shared_private": self.shared_private,
            "private_dim": self.private_dim,
            "task_head_shared_only": self.task_head_shared_only,
            "dtau_readout": "attention_softargmax",
            "tau0_ms": tuple(self.tau0_ms if tau0_ms is None else tau0_ms),
            "tau0_bounds": tuple(
                tuple(v) for v in (self.tau0_bounds if tau0_bounds is None else tau0_bounds)
            ),
            "sigma_bounds": tuple(
                tuple(v) for v in (self.sigma_bounds if sigma_bounds is None else sigma_bounds)
            ),
            "dtau_bounds": tuple(
                tuple(v) for v in (self.dtau_bounds if dtau_bounds is None else dtau_bounds)
            ),
            "tmin_ms": float(tmin_ms),
            "tmax_ms": float(tmax_ms),
            "sfreq": float(sfreq),
            "n_time": int(n_time),
            "baseline_mode": baseline_mode,
            "trial_reference_window_ms": trial_reference_window_ms,
            "trial_reference_center": trial_reference_center,
            "trial_reference_scale": trial_reference_scale,
        }
        if overrides:
            unknown = set(overrides) - set(kwargs)
            if unknown:
                raise ValueError(f"Unknown Neural-RIDE model overrides: {sorted(unknown)}")
            kwargs.update(overrides)
        return kwargs

    def trainer_config(
        self,
        task: TaskCapabilities,
        *,
        epochs: int,
        batch_size: int | None = None,
        seed: int,
        batch_config: TrainingBatchConfig | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> TrainerConfig:
        """Build a capability-aware TrainerConfig without entry-point defaults."""

        if batch_config is not None:
            if batch_size is not None:
                raise ValueError("pass batch_config or batch_size, not both.")
            batch_size = batch_config.physical_batch_size
        if batch_size is None:
            raise ValueError("batch_size or batch_config is required.")

        config = TrainerConfig(
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=self.lr,
            lr_schedule=self.lr_schedule,
            lr_warmup_fraction=self.lr_warmup_fraction,
            min_lr_ratio=self.min_lr_ratio,
            erp_decoder_lr_multiplier=self.erp_decoder_lr_multiplier,
            weight_decay=self.weight_decay,
            lambda2=self.lambda2,
            lambda3=self.lambda3,
            lambda_pcw=self.lambda_pcw,
            lambda_digit=self.lambda_digit if task.has_digit_sets else 0.0,
            lambda_conditional_nll=(self.lambda_conditional_nll if task.has_digit_sets else 0.0),
            repetition_reliability_aux_weight=self.repetition_reliability_aux_weight,
            repetition_reliability_lr_multiplier=self.repetition_reliability_lr_multiplier,
            repetition_refit_epochs=self.repetition_refit_epochs,
            repetition_v12=self.repetition_v12,
            repetition_state_residual_l2_weight=self.repetition_state_residual_l2_weight,
            repetition_v12_evidence_ks=self.repetition_v12_evidence_ks,
            repetition_v12_evidence_weights=self.repetition_v12_evidence_weights,
            auto_pos_weight=self.auto_pos_weight,
            digit_evidence_ks=self.digit_evidence_ks,
            digit_evidence_weights=self.digit_evidence_weights,
            lambda_amp=self.lambda_amp,
            lambda_recon=self.lambda_recon,
            recon_waveform_weight=self.recon_waveform_weight,
            recon_projection_weight=self.recon_projection_weight,
            recon_nll_weight=self.recon_nll_weight,
            lambda_innovation=self.lambda_innovation,
            innovation_score_interval_ms=self.innovation_score_interval_ms,
            innovation_ar_order=self.innovation_ar_order,
            recon_bootstrap_samples=self.recon_bootstrap_samples,
            recon_split_half_repeats=self.recon_split_half_repeats,
            lambda_morphology_l0=self.lambda_morphology_l0,
            variance_warmup_epochs=self.variance_warmup_epochs,
            variance_ramp_epochs=self.variance_ramp_epochs,
            lambda_jit=self.lambda_jit,
            jit_prob=self.jit_prob,
            jit_max_ms=self.jit_max_ms,
            augment=self.augment,
            early_stop_patience=self.early_stop_patience,
            track_pcw_gradients=self.track_pcw_gradients,
            recalibrate_batch_norm=self.recalibrate_batch_norm,
            lambda_orth=self.lambda_orth,
            lambda_adv=self.lambda_adv,
            lambda_private=self.lambda_private,
            reconstruct_all_domains=self.reconstruct_all_domains,
            pos_weight=task.pos_weight,
            seed=int(seed),
        )
        effective_overrides = dict(overrides or {})
        if batch_config is not None:
            configured_steps = batch_config.accumulation_steps
            requested_steps = effective_overrides.get("accum_steps", configured_steps)
            if int(requested_steps) != configured_steps:
                raise ValueError("batch_config conflicts with overrides['accum_steps'].")
            effective_overrides["accum_steps"] = configured_steps
        if effective_overrides:
            valid = set(asdict(config))
            unknown = set(effective_overrides) - valid
            if unknown:
                raise ValueError(f"Unknown Neural-RIDE trainer overrides: {sorted(unknown)}")
            config = replace(config, **effective_overrides)
        if not task.has_digit_sets and (
            config.lambda_digit != 0.0 or config.lambda_conditional_nll != 0.0
        ):
            raise ValueError(
                f"Task {task.name!r} has no digit-set labels; set losses must be zero."
            )
        if config.lambda_pcw <= 0.0:
            raise ValueError(
                "The canonical Neural-RIDE recipe requires lambda_pcw>0; "
                "use a separately named ablation recipe to remove PCW supervision."
            )
        return config

    def record(self, task: TaskCapabilities, config: TrainerConfig) -> dict[str, Any]:
        return {
            "recipe": self.name,
            "task": asdict(task),
            "trainer_config": asdict(config),
        }


NEURAL_RIDE_V11_LEGACY = NeuralRideRecipe(
    name="neural_ride_v11_legacy_repetition",
    repetition_v12=False,
)
NEURAL_RIDE_V11 = NEURAL_RIDE_V11_LEGACY
NEURAL_RIDE_V11_STRICT_PAST_RESEARCH = replace(
    NEURAL_RIDE_V11,
    name="neural_ride_v11_strict_past_research",
    use_innovation_likelihood=True,
    lambda_innovation=1.0,
    variance_warmup_epochs=3,
    variance_ramp_epochs=5,
)
NEURAL_RIDE_V11_TRANSFER = replace(
    NEURAL_RIDE_V11,
    name="neural_ride_v11_canonical_transfer",
    canonical_channel_names=("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"),
    dataset_adapter_rank=8,
    shared_private=True,
    private_dim=32,
    task_head_shared_only=True,
    lambda_orth=0.01,
    lambda_adv=0.10,
    lambda_private=0.10,
    reconstruct_all_domains=False,
)

# Active v12 production recipe: additive LLR repetition backbone + fidelity
# rank objective, K=1/3/5 as the main development estimand, and the L object
# available as an explicitly gated measurement-window consumer.
NEURAL_RIDE_V12 = replace(
    NEURAL_RIDE_V11,
    name="neural_ride_v12_pcw_fail_closed",
    repetition_v12=True,
    repetition_state_residual_l2_weight=1e-4,
    repetition_v12_evidence_ks=(1, 3, 5),
    repetition_v12_evidence_weights=(0.34, 0.33, 0.33),
    digit_evidence_ks=(1, 3, 5),
    digit_evidence_weights=(0.34, 0.33, 0.33),
)
NEURAL_RIDE_V12_STRICT_PAST_RESEARCH = replace(
    NEURAL_RIDE_V12,
    name="neural_ride_v12_strict_past_research",
    use_innovation_likelihood=True,
    lambda_innovation=1.0,
    variance_warmup_epochs=3,
    variance_ramp_epochs=5,
)

# Named research-only E5 claim-gate contrasts. These recipes enable the
# unconstrained full-Z2 auxiliary trial head; they are never selected by the
# production default and must not be promoted without the pre-registered
# nested M0/M1 cluster-bootstrap gate. ``replace`` keeps PCW as a side
# readout trained by lambda_pcw while the main trial logit comes from Z2.
NEURAL_RIDE_V12_Z2_AUX_RESEARCH = replace(
    NEURAL_RIDE_V12,
    name="neural_ride_v12_z2_aux_research",
    use_z2_aux_head=True,
    z2_aux_head_mode="add",
    z2_aux_pool="attention",
)
NEURAL_RIDE_V12_Z2_AUX_REPLACE_RESEARCH = replace(
    NEURAL_RIDE_V12_Z2_AUX_RESEARCH,
    name="neural_ride_v12_z2_aux_replace_research",
    z2_aux_head_mode="replace",
)
