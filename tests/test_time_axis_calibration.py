from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.n2p3net import N2P3NetBaseline
from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import observed_only_timeline
from models.erp_calibration import calibrate_erp_fold
from models.n2p3net import N2P3Net
from models.time_axis import EpochTimeAxis
from train.factory import build_binary_neural_ride_adapter


def _adapter_dataset(
    channels: tuple[str, ...], preprocessing: PreprocessingSpec | None = None
) -> EpochDataset:
    identity = build_channel_identity(channels, allow_missing_positions=False)
    profile = preprocessing or PreprocessingSpec()
    n_epochs = 8
    subjects = np.repeat(np.array(["s1", "s2"]), n_epochs // 2)
    return EpochDataset(
        name=f"synthetic_{len(channels)}ch",
        X=np.zeros((n_epochs, len(channels), profile.n_times), dtype=np.float32),
        y=np.tile(np.array([0, 1], dtype=np.int64), n_epochs // 2),
        subject_ids=subjects,
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(len(channels), dtype=bool),
        preprocessing=profile,
        event_timeline=observed_only_timeline(
            dataset_id="synthetic",
            subject_ids=subjects,
            stimulus_ids=np.tile(np.array([0, 1]), n_epochs // 2),
        ),
    )


def test_time_axis_rejects_second_values_at_millisecond_api() -> None:
    with pytest.raises(ValueError, match="milliseconds, not seconds"):
        N2P3Net(tmin_ms=-0.2, tmax_ms=0.8, sfreq=256.0, n_time=256)


def test_stimulus_locked_axis_without_baseline_is_valid() -> None:
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        tmin_ms=0.0,
        tmax_ms=1000.0,
        sfreq=256.0,
        n_time=256,
        baseline_mode="none",
    )
    assert model.time_axis.duration_ms == pytest.approx(1000.0)
    assert model.baseline_n == 0


def test_trial_baseline_requires_prestimulus_samples() -> None:
    with pytest.raises(ValueError, match="pre-stimulus"):
        N2P3Net(tmin_ms=0.0, tmax_ms=1000.0, baseline_mode="trial")


def test_forward_rejects_silently_shortened_epoch() -> None:
    model = N2P3Net(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    with pytest.raises(ValueError, match="physical time-axis contract"):
        model(torch.randn(2, 3, 128))


def test_generic_adapter_uses_versioned_neural_ride_axis_and_recipe() -> None:
    dataset = _adapter_dataset(("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"))
    adapter = build_binary_neural_ride_adapter(
        dataset,
        epochs=30,
        batch_size=8,
        seed=1,
        fold_erp_calibration=False,
    )
    assert adapter.model_kwargs["tmin_ms"] == -200.0
    assert adapter.model_kwargs["tmax_ms"] == 1200.0
    assert adapter.model_kwargs["n_time"] == 358
    assert adapter.trainer_kwargs["lambda_pcw"] == pytest.approx(0.3)
    assert adapter.trainer_kwargs["lambda_recon"] == 0.0
    assert adapter.model_kwargs["component_decoder"] is False
    assert adapter.trainer_kwargs["lambda_digit"] == 0.0
    assert adapter.trainer_kwargs["innovation_ar_order"] == 32
    EpochTimeAxis(
        adapter.model_kwargs["tmin_ms"],
        adapter.model_kwargs["tmax_ms"],
        adapter.model_kwargs["sfreq"],
        adapter.model_kwargs["n_time"],
    )


def test_generic_adapter_carries_trial_reference_profile() -> None:
    profile = PreprocessingSpec(
        name="stimulus_locked_trial_reference",
        sfreq=256.0,
        l_freq=None,
        tmin_ms=0.0,
        tmax_ms=1000.0,
        n_times=256,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(0.0, 50.0),
        trial_reference_center="mean",
        trial_reference_scale="none",
        reject_threshold_v=None,
    )
    adapter = build_binary_neural_ride_adapter(
        _adapter_dataset(
            ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"),
            profile,
        ),
        epochs=1,
        batch_size=8,
        seed=1,
        fold_erp_calibration=False,
    )
    assert adapter.model_kwargs["baseline_mode"] == "trial_reference"
    assert adapter.model_kwargs["trial_reference_window_ms"] == (0.0, 50.0)
    assert adapter.model_kwargs["trial_reference_center"] == "mean"
    assert adapter.model_kwargs["trial_reference_scale"] == "none"
    model = N2P3Net(**adapter.model_kwargs)
    assert model.trial_reference_slice == (0, 13)


def test_generic_adapter_accepts_native_16_channels_with_same_recipe() -> None:
    channels = (
        "Fp1",
        "Fp2",
        "F5",
        "AFz",
        "F6",
        "T7",
        "Cz",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "Oz",
        "O2",
    )
    adapter = build_binary_neural_ride_adapter(
        _adapter_dataset(channels),
        epochs=30,
        batch_size=8,
        seed=1,
        fold_erp_calibration=False,
    )
    assert adapter.model_kwargs["n_channels"] == 16
    assert adapter.model_kwargs["channel_names"].index("PZ") == 10
    reference = build_binary_neural_ride_adapter(
        _adapter_dataset(("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")),
        epochs=30,
        batch_size=8,
        seed=1,
        fold_erp_calibration=False,
    )
    for key in (
        "lambda_pcw",
        "lambda_recon",
        "lambda_digit",
        "innovation_ar_order",
    ):
        assert adapter.trainer_kwargs[key] == reference.trainer_kwargs[key]


def test_generic_adapter_accepts_permanently_absent_layout_channels() -> None:
    dataset = _adapter_dataset(("Fz", "Cz", "Pz"))
    dataset.channel_mask[1] = False
    dataset.X[:, 1] = 0.0

    adapter = build_binary_neural_ride_adapter(
        dataset,
        epochs=1,
        batch_size=8,
        fold_erp_calibration=False,
    )

    assert adapter.channel_mask.tolist() == [True, False, True]
    assert torch.all(adapter.E_chn[1] == 0.0)


def _make_fold_erp() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    axis = EpochTimeAxis(-200.0, 800.0, 256.0, 256)
    t = axis.samples_ms()
    Xs, ys, subjects = [], [], []
    for subject in range(6):
        for label in (0, 1):
            for _ in range(8):
                x = rng.normal(0.0, 0.2, (3, 256))
                if label:
                    x += 4.0 * np.exp(-0.5 * ((t - 480.0) / 45.0) ** 2)
                Xs.append(x.astype(np.float32))
                ys.append(label)
                subjects.append(subject)
    return np.stack(Xs), np.asarray(ys), np.asarray(subjects)


def test_fold_calibration_recovers_training_erp_peak() -> None:
    X, y, subjects = _make_fold_erp()
    result = calibrate_erp_fold(
        X,
        y,
        subjects,
        time_axis=EpochTimeAxis(-200.0, 800.0, 256.0, 256),
        channel_names=("Fz", "Cz", "Pz"),
    )
    assert result["calibration_scope"] == "outer_train_inner_subtrain"
    assert result["tau0_ms"][2] == pytest.approx(480.0, abs=8.0)
    assert result["n_subjects"] == 6
    assert result["prior_source"] == "fold_calibration"
    assert result["sigma_contract"] == ((20.0, 50.0), (20.0, 80.0), (20.0, 80.0))
    for (lo, hi) in result["sigma_bounds"]:
        assert 20.0 <= lo < hi <= 80.0


def test_fold_calibration_ignores_masked_zero_fill_channels() -> None:
    X, y, subjects = _make_fold_erp()
    X[:, 1:] = 0.0
    mask = np.zeros(X.shape[:2], dtype=bool)
    mask[:, 0] = True

    result = calibrate_erp_fold(
        X,
        y,
        subjects,
        time_axis=EpochTimeAxis(-200.0, 800.0, 256.0, 256),
        channel_names=("Fz", "Cz", "Pz"),
        trial_channel_mask=mask,
    )

    assert result["tau0_ms"][2] == pytest.approx(480.0, abs=8.0)


def test_fold_calibration_repairs_overlapping_component_upper_bounds() -> None:
    from models.erp_calibration import _ordered_tau_bounds

    repaired = _ordered_tau_bounds(
        ((216.875, 376.875), (210.0, 390.0), (433.8, 613.0)),
        (296.875, 300.0, 523.4375),
        tmin_ms=0.0,
        tmax_ms=1000.0,
    )

    assert repaired[1][1] > repaired[0][1] + 30.0
    assert repaired[2][1] + 150.0 > repaired[1][1] + 1.0


def test_adapter_calibrator_never_sees_inner_validation_subjects() -> None:
    X, y, subjects = _make_fold_erp()
    seen: list[np.ndarray] = []

    def calibrator(X_train, y_train, train_subject_ids):
        seen.append(np.unique(train_subject_ids))
        return {
            "tau0_ms": (220.0, 300.0, 460.0),
            "tau0_bounds": ((180.0, 280.0), (250.0, 380.0), (350.0, 600.0)),
            "sigma_bounds": ((20.0, 50.0), (20.0, 80.0), (40.0, 150.0)),
        }

    adapter = N2P3NetBaseline(
        model_kwargs={
            "n_channels": 3,
            "channel_names": ("Fz", "Cz", "Pz"),
            "encoder_depth": 0,
        },
        trainer_kwargs={"epochs": 1, "batch_size": 32, "augment": False, "seed": 3},
        device=torch.device("cpu"),
        val_subject_frac=0.34,
        val_subjects_min=2,
        val_subjects_max=2,
        erp_calibrator=calibrator,
    )
    split = adapter._subject_validation_split(subjects)
    adapter.fit(X, y, subject_ids=subjects)
    assert len(seen) == 1
    assert set(seen[0]).isdisjoint(split.validation_subjects)
    assert len(seen[0]) == 4
