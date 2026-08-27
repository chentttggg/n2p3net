"""Performance-first training adapter for :class:`models.n2p3net.N2P3Net`."""

from __future__ import annotations

import torch

from baselines.deep import DeepBaseline, DeepConfig
from models.n2p3net import N2P3Net
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
    ) -> None:
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

    def _make_model(self) -> N2P3Net:
        return N2P3Net(n_channels=self.n_chans, dropout=0.25)
