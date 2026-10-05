"""Performance-first training adapter for :class:`models.n2p3net.N2P3Net`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from baselines.deep import DeepBaseline, DeepConfig
from data.contract import DEFAULT_P300_DATA_CONTRACT
from models.adapters import (
    FeatureDomainTransfer,
    FeatureTransferConfig,
    feature_transfer_architecture_record,
)
from models.adapters import source_domain_axis_record as build_source_domain_axis_record
from models.n2p3net import (
    DEFAULT_N2P3_ARCHITECTURE,
    DEFAULT_N2P3_POOLING_MODE,
    POOLING_MODES,
    N2P3ArchitectureConfig,
    N2P3Net,
)
from models.splitbn import SplitBNConfig, resolve_splitbn_config
from research.contracts import semantic_sha256
from train.runtime import GpuPerformanceScheduler


class N2P3NetBaseline(DeepBaseline):
    """Train the project's compact neural model with the common baseline contract."""

    # The wrapper exposes structured invariant/private views for source
    # auxiliary losses and owns the target-only residual adapter.
    supports_feature_transfer = True
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
        feature_transfer: FeatureTransferConfig | None = None,
        source_domain_vocabulary: Sequence[str] | None = None,
        splitbn_domain_count: int | None = None,
        splitbn_target_buffer_source: str | None = None,
        splitbn_source_weights: Mapping[str, float] | None = None,
        pretrained_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if not math.isfinite(tmin_s):
            raise ValueError("tmin_s must be finite.")
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {sorted(POOLING_MODES)}.")
        if not isinstance(architecture, N2P3ArchitectureConfig):
            raise TypeError("architecture must be an N2P3ArchitectureConfig.")
        if feature_transfer is not None and not isinstance(
            feature_transfer, FeatureTransferConfig
        ):
            raise TypeError("feature_transfer must be a FeatureTransferConfig.")
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
            pretrained_state_dict=pretrained_state_dict,
        )
        self.model_name = "n2p3net"
        self.tmin_s = float(tmin_s)
        self.pooling_mode = pooling_mode
        self.architecture = architecture
        self.feature_transfer = feature_transfer
        indexed_counts = tuple(
            count
            for count in (
                splitbn_domain_count,
                None if feature_transfer is None else feature_transfer.source_domain_count,
                None if feature_transfer is None else feature_transfer.private_domain_count,
            )
            if count is not None and count > 0
        )
        if source_domain_vocabulary is None:
            if indexed_counts or splitbn_source_weights is not None:
                raise ValueError(
                    "indexed domain mechanisms require source_domain_vocabulary."
                )
            self.source_domain_vocabulary: tuple[str, ...] | None = None
            self._source_domain_axis_record: dict[str, object] | None = None
            positional_weights = None
        else:
            axis_record = build_source_domain_axis_record(
                tuple(source_domain_vocabulary), splitbn_source_weights
            )
            self.source_domain_vocabulary = tuple(axis_record["vocabulary"])
            self._source_domain_axis_record = axis_record
            if any(count != len(self.source_domain_vocabulary) for count in indexed_counts):
                raise ValueError(
                    "every indexed domain count must match source_domain_vocabulary."
                )
            weights_by_domain = axis_record["source_weights_by_domain"]
            positional_weights = (
                None
                if weights_by_domain is None
                else tuple(
                    weights_by_domain[domain]
                    for domain in self.source_domain_vocabulary
                )
            )
        self.splitbn_config: SplitBNConfig | None = resolve_splitbn_config(
            domain_count=splitbn_domain_count,
            target_buffer_source=splitbn_target_buffer_source,
            source_weights=positional_weights,
        )

    def source_domain_axis_record(self) -> dict[str, object] | None:
        """Return the shared index-to-domain authority for checkpoint payloads."""

        if self._source_domain_axis_record is None:
            return None
        return {
            **self._source_domain_axis_record,
            "vocabulary": list(self._source_domain_axis_record["vocabulary"]),
            "domain_to_index": dict(
                self._source_domain_axis_record["domain_to_index"]
            ),
            "source_weights_by_domain": (
                None
                if self._source_domain_axis_record["source_weights_by_domain"] is None
                else dict(self._source_domain_axis_record["source_weights_by_domain"])
            ),
        }

    def _make_model(self) -> N2P3Net | FeatureDomainTransfer:
        core = N2P3Net(
            n_channels=self.n_chans,
            n_times=self.n_times,
            sfreq=self.sfreq,
            tmin_s=self.tmin_s,
            pooling_mode=self.pooling_mode,
            **self.architecture.model_kwargs(),
            **(
                {}
                if self.splitbn_config is None
                else self.splitbn_config.model_kwargs()
            ),
        )
        if self.feature_transfer is None:
            return core
        return FeatureDomainTransfer(core, config=self.feature_transfer)

    def architecture_record(self) -> dict[str, object]:
        """Return the physical pooling contract before a fold constructs the model."""

        record = N2P3Net.default_architecture_record(
            pooling_mode=self.pooling_mode,
            tmin_s=self.tmin_s,
            sfreq=self.sfreq,
            n_times=self.n_times,
            architecture=self.architecture,
            splitbn=self.splitbn_config,
        )
        if self.feature_transfer is not None:
            record["feature_transfer"] = feature_transfer_architecture_record(
                self.feature_transfer,
                feature_channels=len(self.architecture.mst_kernel_sizes)
                * self.architecture.mst_features_per_scale,
            )
        axis_record = self.source_domain_axis_record()
        if axis_record is not None and (
            self.splitbn_config is not None
            or (
                self.feature_transfer is not None
                and (
                    self.feature_transfer.source_domain_count > 0
                    or self.feature_transfer.private_domain_count > 0
                )
            )
        ):
            record["source_domain_axis_digest"] = semantic_sha256(axis_record)
        return record
