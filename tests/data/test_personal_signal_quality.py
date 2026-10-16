"""Distribution changes cannot become unsupported personal artifact labels."""

import copy

import numpy as np
import pytest

from data.personal_quality import PersonalEpochQualityPolicy, personal_epoch_quality_from_record


def _calibration():
    time = np.arange(179) / 128 - .2
    rng = np.random.default_rng(321)
    X = rng.normal(size=(36, 6, len(time))) * 1e-6
    X += np.sin(2 * np.pi * 8 * time)[None, None] * np.linspace(5, 20, 6)[None, :, None] * 1e-6
    return time, X, np.repeat("p1", len(X)), np.repeat(["s1", "s2", "s3"], 12)


def test_personal_quality_other_channels_and_masks_do_not_change_fixed_channel_admission():
    _, X, subjects, groups = _calibration()
    model = PersonalEpochQualityPolicy().fit(X, subjects, fit_groups=groups)
    changed = X.copy()
    changed[:, [0, 1, 3, 4, 5]] *= 100
    mask = np.ones(X.shape[:2], dtype=bool)
    mask[:, [0, 1]] = False
    original, scaled, missing = model.transform(X), model.transform(changed), model.transform(X, mask)
    for result in (scaled, missing):
        np.testing.assert_array_equal(result.trial_channel_mask[:, 2], original.trial_channel_mask[:, 2])
        for name in result.diagnostic_masks:
            np.testing.assert_array_equal(result.diagnostic_masks[name][:, 2], original.diagnostic_masks[name][:, 2])


@pytest.mark.parametrize("amplitude", [1e-12, 1e-6, 80e-6, 1e-3])
def test_dynamic_low_amplitude_and_large_smooth_erp_are_diagnostics_not_artifact_proof(amplitude):
    time, X, subjects, groups = _calibration()
    model = PersonalEpochQualityPolicy().fit(X, subjects, fit_groups=groups)
    for width in (.04, .12, .25):
        signals = X[:4].copy()
        signals[:, 2] = amplitude * np.exp(-.5 * ((time - .35) / width) ** 2)
        signals[:, 5] = amplitude * np.sin(2 * np.pi * 12 * time)
        result = model.transform(signals)
        assert result.trial_channel_mask.all() and not result.drop_epoch_mask.any()
        np.testing.assert_array_equal(result.X, signals)


def test_constant_signal_and_upstream_missing_channels_remain_unusable():
    _, X, subjects, groups = _calibration()
    model = PersonalEpochQualityPolicy().fit(X, subjects, fit_groups=groups)
    values = X[:3].copy()
    values[0, :4] = 0
    values[1, 2] = 12e-6
    observed = np.ones(values.shape[:2], dtype=bool)
    observed[2, :2] = False
    result = model.transform(values, observed)
    assert result.drop_epoch_mask.tolist() == [True, False, False]
    assert not result.trial_channel_mask[0, :4].any()
    assert not result.trial_channel_mask[1, 2]
    assert not result.trial_channel_mask[2, :2].any()
    assert not result.global_epoch_scale_mask.any()
    assert result.summary()["hard_rejection_basis"] == "exact_constant_epoch_signal"


def test_personal_quality_diagnostics_are_frozen_and_roundtrip_semantics_are_strict():
    _, X, subjects, groups = _calibration()
    model = PersonalEpochQualityPolicy().fit(X, subjects, fit_groups=groups)
    record = model.record()
    restored = personal_epoch_quality_from_record(record)
    result = restored.transform(X * 100)
    assert result.diagnostic_masks["amplitude_above_calibration"].any()
    assert result.trial_channel_mask.all()
    assert restored.record() == record
    for key, value in (("method", "fold_local_ptp_cv"), ("hard_rejection_basis", "amplitude"),
                       ("n_channels", 5), ("schema", "n2p3_personal_epoch_quality/99")):
        changed = copy.deepcopy(record)
        changed[key] = value
        with pytest.raises(ValueError):
            personal_epoch_quality_from_record(changed)


def test_degenerate_calibration_has_explicit_missing_diagnostic_support():
    X = np.zeros((8, 6, 179), dtype=np.float32)
    model = PersonalEpochQualityPolicy().fit(X, np.repeat("p1", len(X)))
    assert not model.diagnostic_group_counts.any()
    restored = personal_epoch_quality_from_record(model.record())
    assert restored.transform(X).all_channels_bad.all()
    with pytest.raises(ValueError, match="finite"):
        restored.transform(np.full_like(X, np.nan))
