from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kernel_all_evidence_eval_20260831",
    ROOT
    / "doc"
    / "evidence"
    / "gtn_20260831"
    / "kernel_v4"
    / "kernel_all_evidence_eval_20260831.py",
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _dataset() -> SimpleNamespace:
    X = np.asarray(
        [
            [[1e-6] * 8, [2e-6] * 8],
            [[2e-6] * 8, [3e-6] * 8],
            [[3e-6] * 8, [4e-6] * 8],
            [[4e-6] * 8, [5e-6] * 8],
            [[5e-6] * 8, [6e-6] * 8],
            [[6e-6] * 8, [7e-6] * 8],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        name="toy",
        X=X,
        y=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        subject_ids=np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"]),
        n_channels=2,
        n_times=8,
        channel_mask=np.asarray([True, True]),
        trial_channel_mask=None,
        preprocessing=SimpleNamespace(tmin_ms=-200.0, sfreq=128.0),
    )


def _valid_payload(dataset: SimpleNamespace) -> dict[str, object]:
    source_rows = np.asarray([True, True, True, True, False, False])
    mask = HARNESS._effective_trial_mask(dataset)[source_rows]
    mean, std = HARNESS._masked_input_stats(dataset.X[source_rows], mask)
    prior = float(dataset.y[source_rows].mean())
    return {
        "trunk_state_dict": {"weight": 1},
        "input_mean": mean.squeeze().tolist(),
        "input_std": std.squeeze().tolist(),
        "input_preprocessing": {},
        "input_channel_names": ["C1", "C2"],
        "input_source_reference": "toy-ref",
        "source_cache_sha256": HARNESS.CACHE_SHA256,
        "classifier_trained": True,
        "training_subject_keys": ["toy\0s1", "toy\0s2"],
        "training_cache_subject_keys": [
            f"{HARNESS.CACHE_SHA256}\0s1",
            f"{HARNESS.CACHE_SHA256}\0s2",
        ],
        "training_subjects": ["s1", "s2"],
        "source_subjects": ["s1", "s2", "s3"],
        "source_dataset_name": "toy",
        "holdout_subjects": ["s3"],
        "source_full_refit": True,
        "source_refit_epochs": 4,
        "source_calibration": {
            "pos_weight": 8.0,
            "train_prior": prior,
            "temperature": 1.0,
            "source": "source_full_refit_weighted_ce_analytic",
        },
        "training_pos_weight": 8.0,
        "training_prior": prior,
        "n_source_epochs_used": 4,
        "qc_ptp_uv": 100.0,
        "qc_dropped_source_epochs": 0,
        "source_label_counts_before_qc": [2, 2],
        "qc_dropped_source_epochs_by_label": [0, 0],
        "source_label_retention_by_label": [1.0, 1.0],
        "best_epoch": 3,
        "config": {
            "pooling_mode": "full_unfold",
            "temporal_kernel_size": 35,
            "epochs": 100,
            "batch_size": 512,
            "seed": 20260828,
            "training": "N2P3NetBaseline supervised (LOSO-identical path)",
        },
        "architecture": HARNESS.expected_architecture_record(dataset, 35),
    }


def _contrast(delta: float, ci: tuple[float, float], p: float, seeds: tuple[float, ...]):
    return {
        "balanced_all_operational_delta": delta,
        "balanced_all_paired_subject_bootstrap_ci95": list(ci),
        "holm_adjusted_p": p,
        "balanced_all_operational_delta_by_seed": {
            str(seed): value for seed, value in zip(HARNESS.SEEDS, seeds, strict=True)
        },
    }


def test_checkpoint_gate_requires_exact_training_complement() -> None:
    dataset = _dataset()
    payload = _valid_payload(dataset)
    HARNESS.validate_checkpoint_payload_contract(
        payload,
        dataset,
        target_subjects=["s3"],
        cache_sha256=HARNESS.CACHE_SHA256,
        kernel=35,
        seed=20260828,
    )
    payload["training_subjects"] = ["s1"]
    with pytest.raises(ValueError, match="exact holdout complement"):
        HARNESS.validate_checkpoint_payload_contract(
            payload,
            dataset,
            target_subjects=["s3"],
            cache_sha256=HARNESS.CACHE_SHA256,
            kernel=35,
            seed=20260828,
        )


def test_checkpoint_gate_rejects_hidden_architecture_change() -> None:
    dataset = _dataset()
    architecture = HARNESS.expected_architecture_record(dataset, 35)
    architecture["dropout"] = 0.5
    with pytest.raises(ValueError, match="differs outside"):
        HARNESS.validate_architecture_record(architecture, dataset=dataset, kernel=35)


def test_winner_must_beat_both_other_kernels() -> None:
    metrics = {
        "33": {"balanced_all_operational_hit_seed_mean": 0.55},
        "35": {"balanced_all_operational_hit_seed_mean": 0.60},
        "65": {"balanced_all_operational_hit_seed_mean": 0.54},
    }
    contrasts = {
        "K35-K33": _contrast(0.05, (0.02, 0.08), 0.01, (0.04, 0.05, 0.06)),
        "K35-K65": _contrast(0.06, (-0.01, 0.12), 0.20, (0.05, 0.06, 0.07)),
        "K33-K65": _contrast(0.01, (-0.02, 0.04), 0.50, (0.0, 0.01, 0.02)),
    }
    selection = HARNESS.build_winner_selection(metrics, contrasts, [])
    assert selection["winner_qualified_by_frozen_rule"] is False
    assert selection["all_opponent_checks"]["65"]["qualified"] is False


def test_winner_fails_on_practical_seed_reversal() -> None:
    metrics = {
        "33": {"balanced_all_operational_hit_seed_mean": 0.55},
        "35": {"balanced_all_operational_hit_seed_mean": 0.60},
        "65": {"balanced_all_operational_hit_seed_mean": 0.54},
    }
    contrasts = {
        "K35-K33": _contrast(0.05, (0.02, 0.08), 0.01, (0.04, 0.05, 0.06)),
        "K35-K65": _contrast(0.06, (0.02, 0.10), 0.01, (0.10, 0.10, -0.02)),
        "K33-K65": _contrast(0.01, (-0.02, 0.04), 0.50, (0.0, 0.01, 0.02)),
    }
    selection = HARNESS.build_winner_selection(metrics, contrasts, [])
    assert selection["winner_qualified_by_frozen_rule"] is False
    assert (
        selection["all_opponent_checks"]["65"]["requirements"]["seed_direction_stable"]
        is False
    )


def test_r2_subject_remains_eligible_for_v3_all_evidence_endpoint() -> None:
    cost = {
        "available_trials": 18,
        "scheduled_stimuli_through_decision": 20,
        "elapsed_seconds": 2.0,
        "decision_evidence_available_time_s": 2.0,
    }
    record = {
        "r_balanced_all": 2,
        "balanced_all": {"hit": True, "cost": cost},
        "raw_all": {"hit": True, "cost": cost},
        "binary_auc": 0.75,
        "hit_by_r": {"1": False, "2": True},
        "cost_by_r": {"1": cost, "2": cost},
    }
    summary = HARNESS.summarize_records([record])
    assert summary["eligible"] == 1
    assert summary["balanced_all_operational_hit"] == 1.0


def test_cost_counts_stimuli_until_evidence_is_ready() -> None:
    onsets = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    available = onsets + 1.2
    cost = HARNESS.evidence_cost(
        [0],
        np.arange(len(onsets)),
        0.0,
        scheduled_onsets=onsets,
        scheduled_available=available,
    )
    assert cost["scheduled_stimuli_through_decision"] == 7
    assert cost["elapsed_seconds"] == pytest.approx(1.2)
