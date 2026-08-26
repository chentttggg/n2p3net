from __future__ import annotations

import numpy as np

from models.research_gates import evaluate_morphology_recovery


def test_morphology_gate_requires_accuracy_and_calibrated_intervals() -> None:
    true = np.arange(100, dtype=float) + 300.0
    predicted = true + 10.0
    width = np.full(100, 80.0)
    lower = true - 30.0
    upper = true + 30.0
    # Deliberately miss exactly ten rows for 90% empirical coverage.
    lower[:10] = true[:10] + 1.0
    report = evaluate_morphology_recovery(true, predicted, width, width * 1.10, lower, upper)
    assert report.passed
    assert report.latency_mae_ms == 10.0
    assert report.interval_coverage == 0.90


def test_morphology_gate_rejects_overwide_intervals() -> None:
    true = np.arange(20, dtype=float) + 300.0
    report = evaluate_morphology_recovery(
        true,
        true,
        np.full(20, 80.0),
        np.full(20, 80.0),
        true - 100.0,
        true + 100.0,
    )
    assert not report.passed
    assert not report.checks["interval_coverage"]
