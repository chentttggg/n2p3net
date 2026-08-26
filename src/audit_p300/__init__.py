"""P300-specific Probe--Erase--Closure auditing toolkit.

This package is intentionally additive.  It does not modify or import the
existing N2P3-Net model implementations; integration is performed by an
explicit :class:`P300ModelAdapter`.
"""

from .adapters import (
    ArrayP300Adapter,
    LayerIntervention,
    P300ModelAdapter,
    TorchLayerBinding,
    TorchP300Adapter,
)
from .closure import ClosureConfig, ClosureResult, P300ClosureAuditor
from .erase import CrossCovarianceEraser, EraseConfig, EraseResult, P300EraseAuditor
from .features import P300FeatureConfig, P300FeatureLexicon, P300Windows
from .metrics import BootstrapSummary, MetricSpec, benjamini_hochberg, digit_hit
from .probe import LayerProbeAuditor, ProbeConfig, ProbeResult
from .runner import P300AuditReport, P300PECAuditor
from .types import AuditInputError, FeatureTable, P300AuditData, P300Split

__all__ = [
    "ArrayP300Adapter",
    "AuditInputError",
    "BootstrapSummary",
    "ClosureConfig",
    "ClosureResult",
    "CrossCovarianceEraser",
    "EraseConfig",
    "EraseResult",
    "FeatureTable",
    "LayerIntervention",
    "LayerProbeAuditor",
    "MetricSpec",
    "P300AuditData",
    "P300AuditReport",
    "P300ClosureAuditor",
    "P300EraseAuditor",
    "P300FeatureConfig",
    "P300FeatureLexicon",
    "P300ModelAdapter",
    "P300PECAuditor",
    "P300Split",
    "P300Windows",
    "ProbeConfig",
    "ProbeResult",
    "TorchLayerBinding",
    "TorchP300Adapter",
    "benjamini_hochberg",
    "digit_hit",
]
