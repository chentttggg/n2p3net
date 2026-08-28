"""Performance-first training adapter for :class:`models.n2p3net.N2P3Net`."""

from __future__ import annotations

import math

import torch

from baselines.deep import DeepBaseline, DeepConfig
from data.contract import DEFAULT_P300_DATA_CONTRACT
from models.n2p3net import POOLING_MODES, N2P3Net
from train.runtime import GpuPerformanceScheduler


class N2P3NetBaseline(DeepBaseline):
    """Train the project's compact neural model with the common baseline contract."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        config: DeepConfig | None = None,
        device: torch.device | None = None,
        runtime: GpuPerformanceScheduler | None = None,
        *,
        channel_mask=None,
        tmin_s: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms / 1000.0,
        pooling_mode: str = "ms_flatten",
        interaction_rank: int = 8,
        mlp_hidden_features: int = 16,
    ) -> None:
        if not math.isfinite(tmin_s):
            raise ValueError("tmin_s must be finite.")
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {sorted(POOLING_MODES)}.")
        # DeepBaseline owns the validated fold-local training and calibration path.
        super().__init__(
            "eegnet",
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            config=config,
            device=device,
            runtime=runtime,
            channel_mask=channel_mask,
        )
        self.model_name = "n2p3net"
        self.tmin_s = float(tmin_s)
        self.pooling_mode = pooling_mode
        self.interaction_rank = int(interaction_rank)
        self.mlp_hidden_features = int(mlp_hidden_features)

    def _make_model(self) -> N2P3Net:
        return N2P3Net(
            n_channels=self.n_chans,
            n_times=self.n_times,
            sfreq=self.sfreq,
            tmin_s=self.tmin_s,
            pooling_mode=self.pooling_mode,
            interaction_rank=self.interaction_rank,
            mlp_hidden_features=self.mlp_hidden_features,
            dropout=0.25,
        )

    def architecture_record(self) -> dict[str, object]:
        """Return the physical pooling contract before a fold constructs the model."""

        return N2P3Net.default_architecture_record(
            pooling_mode=self.pooling_mode,
            tmin_s=self.tmin_s,
            sfreq=self.sfreq,
            n_times=self.n_times,
            interaction_rank=self.interaction_rank,
            mlp_hidden_features=self.mlp_hidden_features,
        )
