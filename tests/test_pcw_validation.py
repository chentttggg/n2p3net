from __future__ import annotations

import numpy as np

from models.pcw_validation import (
    ablation_evidence,
    evaluate_pcw_claim_gate,
    gradient_health,
    latency_recovery,
    split_half_stability,
)


def test_pcw_claim_gate_passes_only_complete_evidence() -> None:
    true = np.linspace(280.0, 500.0, 40)
    recovery = latency_recovery(true, true + 5.0 * np.sin(np.arange(40)))
    gradient = gradient_health(
        np.full(20, 2e-3),
        np.full(20, 1.0),
        np.array([220.0, 300.0, 420.0]),
        np.array([220.0, 300.0, 430.0]),
    )
    split = split_half_stability(true, true + 4.0 * np.cos(np.arange(40)))
    ablation = ablation_evidence(0.78, 0.76, 0.75)
    gate = evaluate_pcw_claim_gate(recovery, gradient, split, ablation)
    assert gate.passed
    assert gate.allowed_claim == "fold-calibrated discriminative routing window"


def test_pcw_claim_gate_downgrades_current_failure_pattern() -> None:
    true = np.linspace(280.0, 500.0, 40)
    predicted = np.full_like(true, 460.0)
    recovery = latency_recovery(true, predicted)
    gradient = gradient_health(
        np.full(20, 1e-5),
        np.full(20, 1.0),
        np.array([220.0, 300.0, 460.0]),
        np.array([220.0, 300.0, 460.03]),
    )
    split = split_half_stability(true, true[::-1])
    ablation = ablation_evidence(0.754, 0.753, 0.744)
    gate = evaluate_pcw_claim_gate(recovery, gradient, split, ablation)
    assert not gate.passed
    assert gate.allowed_claim == "fold-calibrated routing window"
    assert "synthetic_latency_recovery" in gate.failures
    assert "gradient_and_drift" in gate.failures
    assert "real_split_half_stability" in gate.failures
    assert "fixed_window_mean_pool_ablation" in gate.failures
