"""Typed data contracts for Neural-RIDE training and reconstruction.

``prevalidated`` marks contexts/metadata produced by the project loaders whose
complete tensors were validated once at construction. Hot training loops keep
shape checks but skip repeated device->host value checks for these objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SetMetadata:
    """Metadata for complete GTN evidence sets in one batch.

    ``group_ids`` are local to a batch. ``repetition_ranks`` are zero-based
    ranks within each group/digit cell, making K={3,5,10,15} nested prefixes
    of the same Kmax set. Values below zero mark non-GTN auxiliary samples.

    ``prevalidated_kmax`` is a batch-level coverage guarantee: the largest K
    such that every main GTN group in this batch has at least K repetitions
    of every candidate digit. Objectives may skip per-K emptiness checks only
    up to this value; for larger K they must filter partial-coverage groups.
    """

    stimulus_digits: torch.Tensor
    group_ids: torch.Tensor
    repetition_ranks: torch.Tensor
    sequence_ranks: torch.Tensor | None = None
    prevalidated: bool = False
    prevalidated_kmax: int | None = None

    def validate(self, batch_size: int) -> None:
        tensors = (self.stimulus_digits, self.group_ids, self.repetition_ranks)
        if any(t.dim() != 1 or t.numel() != batch_size for t in tensors):
            shapes = tuple(tuple(t.shape) for t in tensors)
            raise ValueError(
                "SetMetadata tensors must be one-dimensional and match the batch; "
                f"got {shapes} for batch_size={batch_size}."
            )
        if self.prevalidated:
            return
        main = self.group_ids >= 0
        if bool(main.any()):
            if bool((self.stimulus_digits[main] < 1).any()):
                raise ValueError("GTN set stimulus digits must be positive.")
            if bool((self.repetition_ranks[main] < 0).any()):
                raise ValueError("GTN set repetition ranks must be non-negative.")
        if self.sequence_ranks is not None:
            if self.sequence_ranks.dim() != 1 or self.sequence_ranks.numel() != batch_size:
                raise ValueError(
                    "SetMetadata.sequence_ranks must be one-dimensional and match the batch."
                )
            if bool(main.any()) and bool((self.sequence_ranks[main] < 0).any()):
                raise ValueError("GTN set sequence ranks must be non-negative.")
            for group in torch.unique(self.group_ids[main]):
                ranks = self.sequence_ranks[main & (self.group_ids == group)]
                if ranks.unique().numel() != ranks.numel():
                    raise ValueError("GTN sequence ranks must be unique within each group.")

    def to(self, device: torch.device, *, non_blocking: bool = False) -> SetMetadata:
        return SetMetadata(
            stimulus_digits=self.stimulus_digits.to(
                device, dtype=torch.long, non_blocking=non_blocking
            ),
            group_ids=self.group_ids.to(device, dtype=torch.long, non_blocking=non_blocking),
            repetition_ranks=self.repetition_ranks.to(
                device, dtype=torch.long, non_blocking=non_blocking
            ),
            sequence_ranks=(
                self.sequence_ranks.to(device, dtype=torch.long, non_blocking=non_blocking)
                if self.sequence_ranks is not None
                else None
            ),
            prevalidated=self.prevalidated,
            prevalidated_kmax=self.prevalidated_kmax,
        )


@dataclass(frozen=True)
class TrialContext:
    """One trial batch plus optional domain and GTN set metadata."""

    X: torch.Tensor
    y: torch.Tensor
    domain_id: torch.Tensor | None = None
    set_metadata: SetMetadata | None = None
    channel_mask: torch.Tensor | None = None
    prevalidated: bool = False

    def validate(self) -> None:
        if self.X.dim() != 3:
            raise ValueError(f"TrialContext.X must be (B,C,T), got {tuple(self.X.shape)}.")
        batch_size = self.X.shape[0]
        if self.y.reshape(-1).numel() != batch_size:
            raise ValueError("TrialContext X/y lengths must match.")
        if self.domain_id is not None and self.domain_id.reshape(-1).numel() != batch_size:
            raise ValueError("TrialContext X/domain_id lengths must match.")
        if self.channel_mask is not None:
            if self.channel_mask.dtype != torch.bool:
                raise ValueError("TrialContext channel_mask must have boolean dtype.")
            if self.channel_mask.shape not in {
                (self.X.shape[1],),
                (batch_size, self.X.shape[1]),
            }:
                raise ValueError("TrialContext channel_mask must be (C,) or (B,C).")
            if not self.prevalidated:
                mask = self.channel_mask
                if mask.dim() == 1:
                    mask = mask[None].expand(batch_size, -1)
                if not bool(mask.any(dim=1).all()):
                    raise ValueError("Every TrialContext row must retain an observed channel.")
                if bool((self.X[~mask] != 0.0).any()):
                    raise ValueError("TrialContext masked channels must be exactly zero.")
        if self.set_metadata is not None:
            self.set_metadata.validate(batch_size)

    def to(self, device: torch.device, *, non_blocking: bool = False) -> TrialContext:
        context = TrialContext(
            X=self.X.to(device, non_blocking=non_blocking),
            y=self.y.to(device, dtype=torch.float32, non_blocking=non_blocking),
            domain_id=(
                self.domain_id.to(device, dtype=torch.long, non_blocking=non_blocking)
                if self.domain_id is not None
                else None
            ),
            set_metadata=(
                self.set_metadata.to(device, non_blocking=non_blocking)
                if self.set_metadata is not None
                else None
            ),
            channel_mask=(
                self.channel_mask.to(device, dtype=torch.bool, non_blocking=non_blocking)
                if self.channel_mask is not None
                else None
            ),
            prevalidated=self.prevalidated,
        )
        context.validate()
        return context


@dataclass(frozen=True)
class ReconstructionProfile:
    """Optimization-training statistics defining the offline ERP target."""

    bands_hz: tuple[tuple[float, float], ...]
    band_scales: torch.Tensor
    band_weights: torch.Tensor
    evoked_snr: torch.Tensor
    evoked_contrast: torch.Tensor
    evoked_target_variance: torch.Tensor
    target_rate: torch.Tensor
    channel_mask: torch.Tensor
    time_mask: torch.Tensor
    sfreq: float
    n_time: int
    source_n_trials: int
    bootstrap_samples: int
    split_half_repeats: int
    split_half_correlation: float
    split_half_nrmse: float
    scope: str = "optimization_train_only"

    def validate(self, *, n_channels: int | None = None) -> None:
        n_bands = len(self.bands_hz)
        tensors = (self.band_scales, self.band_weights, self.evoked_snr)
        if any(t.dim() != 1 or t.numel() != n_bands for t in tensors):
            raise ValueError("ReconstructionProfile band tensors must match bands_hz.")
        if self.evoked_contrast.dim() != 2:
            raise ValueError("evoked_contrast must be (C,T).")
        channels, n_time = self.evoked_contrast.shape
        if self.evoked_target_variance.shape != self.evoked_contrast.shape:
            raise ValueError("evoked_target_variance must match evoked_contrast.")
        if not bool(torch.isfinite(self.evoked_target_variance).all()):
            raise ValueError("evoked_target_variance must be finite.")
        if bool((self.evoked_target_variance < 0.0).any()):
            raise ValueError("evoked_target_variance must be non-negative.")
        if n_channels is not None and channels != n_channels:
            raise ValueError(
                f"ReconstructionProfile has {channels} channels; expected {n_channels}."
            )
        if n_time != self.n_time or self.time_mask.shape != (self.n_time,):
            raise ValueError("ReconstructionProfile time dimensions are inconsistent.")
        if self.channel_mask.shape != (channels,):
            raise ValueError("ReconstructionProfile channel_mask must be (C,).")
        if not bool(self.channel_mask.any()):
            raise ValueError("ReconstructionProfile requires at least one valid channel.")
        if not 0.0 < float(self.target_rate) < 1.0:
            raise ValueError("ReconstructionProfile requires both target classes.")
        if self.source_n_trials < 2:
            raise ValueError("ReconstructionProfile requires at least two source trials.")
        if self.bootstrap_samples < 2 or self.split_half_repeats < 0:
            raise ValueError("Invalid bootstrap/split-half replicate counts.")
        if not -1.0 <= self.split_half_correlation <= 1.0:
            raise ValueError("split_half_correlation must be in [-1,1].")
        if self.split_half_nrmse < 0.0:
            raise ValueError("split_half_nrmse must be non-negative.")
        if not torch.isclose(
            self.band_weights.sum().float(),
            torch.ones((), device=self.band_weights.device),
            atol=1e-5,
        ):
            raise ValueError("ReconstructionProfile band_weights must sum to one.")

    def to(self, device: torch.device) -> ReconstructionProfile:
        return ReconstructionProfile(
            bands_hz=self.bands_hz,
            band_scales=self.band_scales.to(device),
            band_weights=self.band_weights.to(device),
            evoked_snr=self.evoked_snr.to(device),
            evoked_contrast=self.evoked_contrast.to(device),
            evoked_target_variance=self.evoked_target_variance.to(device),
            target_rate=self.target_rate.to(device),
            channel_mask=self.channel_mask.to(device),
            time_mask=self.time_mask.to(device),
            sfreq=self.sfreq,
            n_time=self.n_time,
            source_n_trials=self.source_n_trials,
            bootstrap_samples=self.bootstrap_samples,
            split_half_repeats=self.split_half_repeats,
            split_half_correlation=self.split_half_correlation,
            split_half_nrmse=self.split_half_nrmse,
            scope=self.scope,
        )


@dataclass(frozen=True)
class GenerativeProfile:
    """Fixed-coordinate class templates for prequential observation density.

    The profile is estimated on optimization-training subjects only. It is
    deliberately separate from :class:`ReconstructionProfile`: the latter
    lives in the offline ERP decoder's learned sensor coordinates, whereas
    these tensors live in a fixed baseline-standardized observation space.
    """

    class_means: torch.Tensor
    class_channel_variances: torch.Tensor
    ar_coefficients: torch.Tensor
    ar_channel_variances: torch.Tensor
    target_rate: torch.Tensor
    channel_mask: torch.Tensor
    score_time_mask: torch.Tensor
    sfreq: float
    tmin_ms: float
    n_time: int
    source_n_trials: int
    class_low_rank_diagonal: torch.Tensor | None = None
    class_low_rank_factor: torch.Tensor | None = None
    ar_low_rank_diagonal: torch.Tensor | None = None
    ar_low_rank_factor: torch.Tensor | None = None
    scope: str = "optimization_train_only"

    def validate(self, *, n_channels: int | None = None) -> None:
        if self.class_means.dim() != 3 or self.class_means.shape[0] != 2:
            raise ValueError("class_means must be (2,C,T).")
        _, channels, n_time = self.class_means.shape
        if n_channels is not None and channels != n_channels:
            raise ValueError(f"GenerativeProfile has {channels} channels; expected {n_channels}.")
        if n_time != self.n_time or self.score_time_mask.shape != (self.n_time,):
            raise ValueError("GenerativeProfile time dimensions are inconsistent.")
        if self.class_channel_variances.shape != (2, channels):
            raise ValueError("class_channel_variances must be (2,C).")
        if (
            self.ar_coefficients.dim() != 3
            or self.ar_coefficients.shape[1:] != (channels, channels)
            or self.ar_coefficients.shape[0] < 1
        ):
            raise ValueError("ar_coefficients must be (order,C,C).")
        if self.ar_channel_variances.shape != (2, channels):
            raise ValueError("ar_channel_variances must be (2,C).")
        if not bool(torch.isfinite(self.class_means).all()):
            raise ValueError("class_means must be finite.")
        if not bool(torch.isfinite(self.class_channel_variances).all()):
            raise ValueError("class_channel_variances must be finite.")
        if bool((self.class_channel_variances <= 0.0).any()):
            raise ValueError("class_channel_variances must be positive.")
        if not bool(torch.isfinite(self.ar_coefficients).all()):
            raise ValueError("ar_coefficients must be finite.")
        if not bool(torch.isfinite(self.ar_channel_variances).all()) or bool(
            (self.ar_channel_variances <= 0.0).any()
        ):
            raise ValueError("ar_channel_variances must be finite and positive.")
        low_rank_parts = (
            self.class_low_rank_diagonal,
            self.class_low_rank_factor,
            self.ar_low_rank_diagonal,
            self.ar_low_rank_factor,
        )
        if any(part is not None for part in low_rank_parts):
            if any(part is None for part in low_rank_parts):
                raise ValueError("GenerativeProfile low-rank parameters must be all present.")
            assert self.class_low_rank_diagonal is not None
            assert self.class_low_rank_factor is not None
            assert self.ar_low_rank_diagonal is not None
            assert self.ar_low_rank_factor is not None
            if self.class_low_rank_diagonal.shape != (2, channels) or (
                self.ar_low_rank_diagonal.shape != (2, channels)
            ):
                raise ValueError("low-rank diagonals must be (2,C).")
            if (
                self.class_low_rank_factor.dim() != 3
                or self.class_low_rank_factor.shape[:2] != (2, channels)
                or self.ar_low_rank_factor.shape != self.class_low_rank_factor.shape
            ):
                raise ValueError("low-rank factors must share shape (2,C,rank).")
            if self.class_low_rank_factor.shape[-1] not in (1, 2):
                raise ValueError("GenerativeProfile covariance rank must be 1 or 2.")
            if not all(bool(torch.isfinite(part).all()) for part in low_rank_parts):
                raise ValueError("GenerativeProfile low-rank parameters must be finite.")
            if bool((self.class_low_rank_diagonal <= 0.0).any()) or bool(
                (self.ar_low_rank_diagonal <= 0.0).any()
            ):
                raise ValueError("GenerativeProfile low-rank diagonals must be positive.")
        if self.channel_mask.shape != (channels,) or not bool(self.channel_mask.any()):
            raise ValueError("channel_mask must select at least one channel.")
        if not bool(torch.isfinite(self.score_time_mask).all()) or bool(
            (self.score_time_mask < 0.0).any()
        ):
            raise ValueError("score_time_mask must be finite and non-negative.")
        if not bool(((self.score_time_mask == 0.0) | (self.score_time_mask == 1.0)).all()):
            raise ValueError("Additive likelihood scoring requires a binary score_time_mask.")
        if not bool((self.score_time_mask > 0.0).any()):
            raise ValueError("score_time_mask must select at least one time sample.")
        sample_times_ms = self.tmin_ms + torch.arange(
            self.n_time,
            device=self.score_time_mask.device,
            dtype=torch.float32,
        ) * (1000.0 / float(self.sfreq))
        if bool(((self.score_time_mask > 0.0) & (sample_times_ms < 0.0)).any()):
            raise ValueError(
                "Strict-past scoring cannot include pre-stimulus samples because the fixed "
                "baseline transform uses the complete pre-stimulus interval."
            )
        if not 0.0 < float(self.target_rate) < 1.0:
            raise ValueError("GenerativeProfile requires both target classes.")
        if self.source_n_trials < 2:
            raise ValueError("GenerativeProfile requires at least two source trials.")
        if self.scope != "optimization_train_only":
            raise ValueError("GenerativeProfile scope must be optimization_train_only.")

    def to(self, device: torch.device) -> GenerativeProfile:
        return GenerativeProfile(
            class_means=self.class_means.to(device),
            class_channel_variances=self.class_channel_variances.to(device),
            ar_coefficients=self.ar_coefficients.to(device),
            ar_channel_variances=self.ar_channel_variances.to(device),
            target_rate=self.target_rate.to(device),
            channel_mask=self.channel_mask.to(device),
            score_time_mask=self.score_time_mask.to(device),
            sfreq=self.sfreq,
            tmin_ms=self.tmin_ms,
            n_time=self.n_time,
            source_n_trials=self.source_n_trials,
            class_low_rank_diagonal=(
                self.class_low_rank_diagonal.to(device)
                if self.class_low_rank_diagonal is not None
                else None
            ),
            class_low_rank_factor=(
                self.class_low_rank_factor.to(device)
                if self.class_low_rank_factor is not None
                else None
            ),
            ar_low_rank_diagonal=(
                self.ar_low_rank_diagonal.to(device)
                if self.ar_low_rank_diagonal is not None
                else None
            ),
            ar_low_rank_factor=(
                self.ar_low_rank_factor.to(device) if self.ar_low_rank_factor is not None else None
            ),
            scope=self.scope,
        )
