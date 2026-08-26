"""End-to-end P300 Probe--Erase--Closure audit runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import P300ModelAdapter
from .closure import ClosureConfig, ClosureResult, P300ClosureAuditor
from .erase import EraseConfig, EraseResult, P300EraseAuditor
from .features import P300FeatureConfig, P300FeatureLexicon
from .metrics import MetricSpec
from .probe import LayerProbeAuditor, ProbeConfig, ProbeResult
from .types import AuditInputError, P300AuditData, P300Split


@dataclass(frozen=True)
class P300AuditReport:
    """Serializable output of one model/task/split audit."""

    model_name: str
    feature_names: tuple[str, ...]
    feature_families: tuple[str, ...]
    probes: tuple[ProbeResult, ...]
    erasures: tuple[EraseResult, ...]
    closures: tuple[ClosureResult, ...]
    primary_erase_metric: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def encoded_features(self) -> tuple[str, ...]:
        return tuple(result.feature for result in self.probes if result.selection_encoded)

    @property
    def causal_features(self) -> tuple[str, ...]:
        return tuple(result.feature for result in self.erasures if result.representation_causal)

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_count": len(self.feature_names),
            "encoded_count": len(self.encoded_features),
            "representation_causal_count": len(self.causal_features),
            "closure_metrics": {result.metric: result.closure_ratio for result in self.closures},
            "primary_erase_metric": self.primary_erase_metric,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "feature_families": list(self.feature_families),
            "probes": [result.to_dict() for result in self.probes],
            "erasures": [result.to_dict() for result in self.erasures],
            "closures": [result.to_dict() for result in self.closures],
            "primary_erase_metric": self.primary_erase_metric,
            "metadata": self.metadata,
            "summary": self.summary(),
        }

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


class P300PECAuditor:
    """Orchestrate the complete P300 audit without modifying the model."""

    def __init__(
        self,
        *,
        feature_config: P300FeatureConfig | None = None,
        probe_config: ProbeConfig | None = None,
        erase_config: EraseConfig | None = None,
        closure_config: ClosureConfig | None = None,
    ) -> None:
        self.lexicon = P300FeatureLexicon(feature_config)
        self.probe_auditor = LayerProbeAuditor(probe_config)
        self.erase_auditor = P300EraseAuditor(erase_config)
        self.closure_auditor = P300ClosureAuditor(closure_config)

    def run(
        self,
        adapter: P300ModelAdapter,
        data: P300AuditData,
        split: P300Split,
        *,
        model_name: str = "unnamed_p300_model",
        layers: Sequence[str] | None = None,
        erase_metric: MetricSpec | None = None,
        closure_metrics: Sequence[MetricSpec] | None = None,
        batch_size: int = 256,
        metadata: dict[str, Any] | None = None,
    ) -> P300AuditReport:
        erase_metric = erase_metric or MetricSpec("binary_auc")
        if not str(model_name).strip():
            raise AuditInputError("model_name must be non-empty.")
        split.validate_against(data)
        features = self.lexicon.extract(data)
        requested_layers = tuple(adapter.layer_names if layers is None else layers)
        if not requested_layers:
            raise AuditInputError("At least one representation layer must be selected.")
        activations = adapter.collect_activations(
            data.X, layers=requested_layers, batch_size=batch_size
        )
        probes = self.probe_auditor.run(activations, features, split, data)
        erasures = self.erase_auditor.run(
            adapter,
            data,
            split,
            features,
            activations,
            probes,
            metric=erase_metric,
            batch_size=batch_size,
        )
        causal = tuple(result.feature for result in erasures if result.representation_causal)
        selected_closure_metrics = (
            tuple(closure_metrics)
            if closure_metrics is not None
            else (
                MetricSpec("binary_auc"),
                *((MetricSpec("digit_hit"),) if data.has_digit_labels else ()),
            )
        )
        closures: tuple[ClosureResult, ...] = ()
        if causal:
            model_test_score = adapter.predict_scores(data.X, batch_size=batch_size)[split.test]
            closures = self.closure_auditor.run(
                data,
                split,
                features,
                causal,
                model_test_score,
                metrics=selected_closure_metrics,
            )
        report_metadata = {
            "n_trials": data.n_trials,
            "n_channels": data.n_channels,
            "n_times": data.n_times,
            "channel_names": list(data.channel_names),
            "layers": list(requested_layers),
            "split_sizes": {
                "train": int(split.train.size),
                "validation": int(split.validation.size),
                "test": int(split.test.size),
            },
            "digit_metric_available": data.has_digit_labels,
            **(metadata or {}),
        }
        return P300AuditReport(
            model_name=str(model_name),
            feature_names=features.names,
            feature_families=features.families,
            probes=probes,
            erasures=erasures,
            closures=closures,
            primary_erase_metric=erase_metric.name,
            metadata=report_metadata,
        )
