from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from baselines.calibration import LogitCalibration
from baselines.multidataset_acceptance import (
    UnlabeledCalibrationReport,
    adapt_unlabeled_adapter,
    binary_acceptance_metrics,
    random_channel_masks,
    repetition_acceptance_metrics,
    unlabeled_calibration_split,
)
from data.channel import STANDARD_CHANNELS
from models.n2p3net import N2P3Net


def _calibration() -> LogitCalibration:
    return LogitCalibration(
        threshold=0.0,
        threshold_balanced_acc=1.0,
        llr_slope=1.0,
        llr_intercept=0.0,
        source="source_validation",
        n_samples=20,
    )


def test_binary_acceptance_metrics_use_source_calibration() -> None:
    metrics = binary_acceptance_metrics(
        np.array([-4.0, -2.0, 2.0, 4.0]),
        np.array([0, 0, 1, 1]),
        _calibration(),
        calibration_prior=0.5,
    )

    assert metrics.balanced_accuracy == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.nll < 0.1
    assert metrics.ece < 0.1


def test_acceptance_contracts_reject_lossy_shapes_and_masks() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_acceptance_metrics(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            _calibration(),
            calibration_prior=0.5,
        )
    with pytest.raises(ValueError, match="integer dtype"):
        binary_acceptance_metrics(
            np.array([-1.0, 1.0]),
            np.array([0.0, 1.0]),
            _calibration(),
            calibration_prior=0.5,
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        random_channel_masks(
            np.ones((2, 3), dtype=np.int64),
            drop_fraction=0.5,
            repeats=1,
            seed=0,
        )


def test_unlabeled_window_uses_acquisition_time_not_epoch_duration() -> None:
    subjects = np.array(["a", "a", "a", "b", "b", "b"])
    metadata = pd.DataFrame({"acquisition_time_s": [100, 120, 140, 5, 34, 36]})

    calibration, evaluation = unlabeled_calibration_split(
        subjects,
        metadata,
        duration_s=30.0,
        sfreq=256.0,
    )

    assert calibration.tolist() == [True, True, False, True, True, False]
    assert np.array_equal(evaluation, ~calibration)


def test_unlabeled_window_accumulates_time_across_recording_resets() -> None:
    subjects = np.array(["a"] * 6)
    metadata = pd.DataFrame(
        {
            "acquisition_time_s": [0, 10, 20, 0, 10, 20],
            "session": [1, 1, 1, 2, 2, 2],
        }
    )

    calibration, _ = unlabeled_calibration_split(
        subjects,
        metadata,
        duration_s=35.0,
        sfreq=256.0,
    )

    assert calibration.tolist() == [True, True, True, True, False, False]


def test_unlabeled_window_fails_when_any_subject_has_no_evaluation_suffix() -> None:
    subjects = np.array(["a", "a", "b", "b"])
    metadata = pd.DataFrame({"acquisition_time_s": [0.0, 40.0, 0.0, 10.0]})
    with pytest.raises(ValueError, match="subjects.*b"):
        unlabeled_calibration_split(subjects, metadata, duration_s=30.0, sfreq=256.0)


def test_random_channel_masks_never_reenable_or_remove_every_channel() -> None:
    base = np.array([[True, True, False], [True, True, True]])
    masks = random_channel_masks(base, drop_fraction=0.75, repeats=3, seed=9)

    assert all(mask.any(axis=1).all() for mask in masks)
    assert all(not np.any(mask & ~base) for mask in masks)
    assert np.array_equal(
        masks[0], random_channel_masks(base, drop_fraction=0.75, repeats=1, seed=9)[0]
    )


def test_unlabeled_adapter_api_and_update_do_not_accept_labels() -> None:
    assert "y" not in inspect.signature(adapt_unlabeled_adapter).parameters
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=64,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=64.0,
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        component_decoder=False,
        n_domains=2,
        canonical_channel_names=STANDARD_CHANNELS,
        dataset_adapter_rank=4,
        shared_private=True,
        private_dim=8,
    )
    X = torch.randn(4, 3, 64)
    before = model.dataset_adapter.up.detach().clone()

    report = adapt_unlabeled_adapter(
        model,
        X,
        torch.ones(4, 3, dtype=torch.bool),
        domain_id=1,
        expected_target_prior=0.2,
        source_calibration=_calibration(),
        steps=2,
        batch_size=2,
        seed=3,
    )

    assert report.label_access is False
    assert torch.equal(model.dataset_adapter.up[0], before[0])
    assert not torch.equal(model.dataset_adapter.up[1], before[1])


def test_repetition_acceptance_reports_llr_fixed_k() -> None:
    groups = np.repeat(["a", "b"], 6)
    digits = np.tile(np.repeat([1, 2, 3], 2), 2)
    true = {"a": 2, "b": 3}
    labels = np.asarray([digit == true[group] for group, digit in zip(groups, digits, strict=True)])
    logits = np.where(labels, 3.0, -1.0)
    metadata = pd.DataFrame({"stimulus_digit": digits})

    result = repetition_acceptance_metrics(
        logits,
        labels,
        metadata,
        groups,
        _calibration(),
        evidence_ks=(2,),
    )

    assert result["points"]["2"]["accuracy"] == 1.0
    assert result["points"]["2"]["coverage"] == 1.0
    assert result["repetitions_to_target_error"] == 2


def test_repetition_acceptance_rejects_k_times_duration_itr() -> None:
    metadata = pd.DataFrame({"stimulus_digit": [1, 2]})
    with pytest.raises(ValueError, match=r"K \* repetition_duration_s"):
        repetition_acceptance_metrics(
            np.asarray([1.0, -1.0]),
            np.asarray([1, 0]),
            metadata,
            np.asarray(["a", "a"]),
            _calibration(),
            repetition_duration_s=1.0,
        )


def test_subjectwise_tta_restarts_from_zero_shot_for_every_subject(monkeypatch) -> None:
    import experiments.run_multidataset_transfer as runner

    class FakeMultiMontage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter_value = torch.nn.Parameter(torch.tensor(0.0))
            self.domain_index = {"target": 1}

        def branch(self, domain: str):
            assert domain == "target"
            return self

    model = FakeMultiMontage()
    seen: list[tuple[float, float]] = []

    def fake_adapt(branch, X, channel_mask, **kwargs):
        del channel_mask, kwargs
        seen.append((float(branch.adapter_value.detach()), float(X.mean())))
        with torch.no_grad():
            branch.adapter_value.add_(1.0)
        return UnlabeledCalibrationReport(
            n_samples=len(X),
            steps=1,
            final_loss=0.0,
            consistency_loss=0.0,
            prior_loss=0.0,
            entropy_loss=0.0,
            label_access=False,
            trainable_parameters=("adapter_value",),
        )

    def fake_predict(model_, target, domain, rows, *, batch_size, **kwargs):
        del target, domain, batch_size, kwargs
        assert float(model_.adapter_value.detach()) == 1.0
        return np.linspace(-1.0, 1.0, int(rows.sum()))

    monkeypatch.setattr(runner, "adapt_unlabeled_adapter", fake_adapt)
    monkeypatch.setattr(runner, "_predict_logits", fake_predict)
    target = SimpleNamespace(
        subject_ids=np.array(["a"] * 4 + ["b"] * 4),
        metadata=pd.DataFrame({"acquisition_time_s": [0.0, 10.0, 31.0, 40.0] * 2}),
        preprocessing=SimpleNamespace(sfreq=100.0),
        X=np.concatenate(
            [np.ones((4, 2, 3), dtype=np.float32), np.full((4, 2, 3), 2.0, dtype=np.float32)]
        ),
        y=np.array([0, 1, 0, 1] * 2),
    )
    result = runner._run_subjectwise_unlabeled_calibration(
        model,
        target,
        "target",
        duration_s=30.0,
        base_masks=np.ones((8, 2), dtype=bool),
        calibration=_calibration(),
        prior=0.5,
        steps=1,
        batch_size=2,
        lr=1e-4,
        seed=4,
    )

    assert seen == [(0.0, 1.0), (0.0, 2.0)]
    assert float(model.adapter_value.detach()) == 0.0
    assert result["adaptation_unit"] == "subject"
    assert result["state_shared_across_subjects"] is False
    assert set(result["subject_metrics"]) == {"a", "b"}
