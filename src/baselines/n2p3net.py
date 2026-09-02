"""Performance-first training adapter for :class:`models.n2p3net.N2P3Net`."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from baselines.deep import DeepBaseline, DeepConfig
from data.contract import DEFAULT_P300_DATA_CONTRACT
from models.adapters import FeatureAdapterConfig
from models.n2p3net import (
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    POOLING_MODES,
    N2P3ArchitectureConfig,
    N2P3Net,
)
from train.runtime import GpuPerformanceScheduler


class N2P3NetBaseline(DeepBaseline):
    """Train the project's compact neural model with the common baseline contract."""

    # The N2P3 trunk can route batches through per-domain residual adapters
    # and exposes its adapted features for the alignment objective.
    supports_domain_adapters = True
    supports_feature_alignment = True

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
        pooling_mode: str = DEFAULT_N2P3_POOLING_MODE,
        architecture: N2P3ArchitectureConfig = DEFAULT_N2P3_ARCHITECTURE,
        feature_adapter: FeatureAdapterConfig | None = None,
        adapter_domains: Sequence[str] | None = None,
    ) -> None:
        if not math.isfinite(tmin_s):
            raise ValueError("tmin_s must be finite.")
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {sorted(POOLING_MODES)}.")
        if not isinstance(architecture, N2P3ArchitectureConfig):
            raise TypeError("architecture must be an N2P3ArchitectureConfig.")
        if feature_adapter is not None and not isinstance(
            feature_adapter, FeatureAdapterConfig
        ):
            raise TypeError("feature_adapter must be a FeatureAdapterConfig.")
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
        self.architecture = architecture
        self.feature_adapter = feature_adapter
        self.adapter_domains = (
            None if adapter_domains is None else tuple(str(domain) for domain in adapter_domains)
        )

    def _make_model(self) -> N2P3Net:
        return N2P3Net(
            n_channels=self.n_chans,
            n_times=self.n_times,
            sfreq=self.sfreq,
            tmin_s=self.tmin_s,
            pooling_mode=self.pooling_mode,
            feature_adapter=self.feature_adapter,
            adapter_domains=self.adapter_domains,
            **self.architecture.model_kwargs(),
        )

    def architecture_record(self) -> dict[str, object]:
        """Return the physical pooling contract before a fold constructs the model."""

        return N2P3Net.default_architecture_record(
            pooling_mode=self.pooling_mode,
            tmin_s=self.tmin_s,
            sfreq=self.sfreq,
            n_times=self.n_times,
            architecture=self.architecture,
        )
