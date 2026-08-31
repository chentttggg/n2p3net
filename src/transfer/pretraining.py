"""Masked reconstruction pretraining task around an existing compact trunk.

The task owns a ``N2P3Net`` trunk and a discardable ``WaveDecoderHead``. The
trunk is used only through ``forward_features``; its configured readout is not
part of pretraining and is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from models.n2p3net import N2P3Net
from transfer.heads import SubjectProbeHead, WaveDecoderHead
from transfer.losses import ReconstructionLossConfig, estimate_band_weights, reconstruction_loss
from transfer.masking import MaskingConfig, apply_time_mask, make_temporal_mask


@dataclass
class PretrainingConfig:
    mask: MaskingConfig = field(default_factory=MaskingConfig)
    loss: ReconstructionLossConfig = field(default_factory=ReconstructionLossConfig)
    seed: int = 0
    band_weight_estimation_samples: int = 4096
    subject_probe_subjects: int = 0

    def validate(self, sfreq: float, n_times: int) -> None:
        self.mask.validate(n_times)
        self.loss.validate(sfreq)
        if self.band_weight_estimation_samples < 1:
            raise ValueError("band_weight_estimation_samples must be positive.")
        if self.subject_probe_subjects < 0:
            raise ValueError("subject_probe_subjects must be non-negative.")


class PretrainingTask(nn.Module):
    """Evaluate the pretraining loss without owning an optimizer or data loader."""

    def __init__(self, trunk: N2P3Net, config: PretrainingConfig) -> None:
        super().__init__()
        if trunk.n_times is None:
            raise ValueError("pretraining requires a trunk with a fixed n_times.")
        config.validate(trunk.sfreq, trunk.n_times)
        self.config = config
        self.trunk = trunk
        self.decoder = WaveDecoderHead(
            trunk_channels=len(trunk.mst_branches) * trunk.mst_features_per_scale,
            output_channels=trunk.n_channels,
            st_pool_size=trunk.st_pool_size,
        )
        self._band_weights: torch.Tensor | None = None
        self.probe: SubjectProbeHead | None = None
        if config.subject_probe_subjects > 1:
            self.probe = SubjectProbeHead(
                len(trunk.mst_branches)
                * trunk.mst_features_per_scale
                * trunk._pooled_time_samples(trunk.n_times, trunk.st_pool_size),
                config.subject_probe_subjects,
            )

    @torch.no_grad()
    def update_band_weights(self, x: torch.Tensor) -> None:
        """Estimate and freeze weights from one fixed source-training subset."""

        self._band_weights = estimate_band_weights(
            x,
            sfreq=self.trunk.sfreq,
            bands_hz=self.config.loss.bands_hz,
        )

    def loss_components(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = make_temporal_mask(x.shape[-1], config=self.config.mask, generator=generator)
        masked, keep = apply_time_mask(x, mask.to(device=x.device))
        features = self.trunk.forward_features(masked)
        x_hat = self.decoder(features, keep)
        components = reconstruction_loss(
            x,
            x_hat,
            sfreq=self.trunk.sfreq,
            config=self.config.loss,
            weights=self._band_weights,
            sample_mask=mask.to(device=x.device)[None, None, :],
        )
        if self.probe is not None and subject_ids is not None:
            if subject_ids.shape != (x.shape[0],):
                raise ValueError("subject probe requires subject_ids aligned with x.")
            flattened = features.detach().flatten(start_dim=1)
            probe_logits = self.probe(flattened)
            components["subject_probe"] = nn.functional.cross_entropy(
                probe_logits,
                subject_ids.to(device=x.device),
            )
            components["subject_probe_correct"] = probe_logits.argmax(dim=1).eq(
                subject_ids.to(device=x.device)
            ).sum()
        return components

    def subject_probe_components(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the audit probe on unmasked stop-gradient trunk features."""

        if self.probe is None:
            raise RuntimeError("subject probe is disabled.")
        if subject_ids.shape != (x.shape[0],):
            raise ValueError("subject_ids must align with the probe batch.")
        with torch.no_grad():
            features = self.trunk.forward_features(x).flatten(start_dim=1)
        logits = self.probe(features.detach())
        targets = subject_ids.to(device=x.device)
        return {
            "loss": nn.functional.cross_entropy(logits, targets),
            "correct": logits.argmax(dim=1).eq(targets).sum(),
        }

    def discard_decoder(self) -> None:
        """Remove pretraining-only parameters after pretraining is complete."""

        self.decoder = None  # type: ignore[assignment]
        self.probe = None

    @property
    def band_weights(self) -> torch.Tensor | None:
        return self._band_weights
