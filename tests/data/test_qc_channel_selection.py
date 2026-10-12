"""Montage QC is recomputed from actual preprocessed samples, never ratios."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy
from data.domain import adapt_native_reference_selection
from data.epochs import (
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    save_epoch_dataset,
    select_epoch_channels,
)
from data.qc_features import QC_CHANNEL_SELECTION_PROGRAM, compute_epoch_qc_features
from tests.support.datasets import reference_dataset

CHANNELS = ("Fz", "Cz", "Pz", "Oz")


def dataset(amplitudes):
    amplitudes = np.atleast_2d(np.asarray(amplitudes, dtype=np.float32))
    values = amplitudes[:, :, None] * np.asarray([0., .2, .7, 1.], dtype=np.float32)
    result = reference_dataset(values, reference="nose", channels=CHANNELS)
    result.qc_features = compute_epoch_qc_features(values, channel_mask=result.channel_mask)
    return result


def assert_actual_qc(selected):
    direct = compute_epoch_qc_features(selected.X, channel_mask=selected.channel_mask,
        trial_channel_mask=selected.trial_channel_mask)
    for field in ("relative_ptp", "channel_std_v", "epoch_scale_v", "observed_mask"):
        np.testing.assert_array_equal(getattr(selected.qc_features, field), getattr(direct, field))


@pytest.mark.parametrize("scale", [1e-6, 1e-19, 0.])
def test_selected_montage_matches_actual_compute_at_normal_tiny_and_zero_scales(scale):
    source = dataset(np.array([100., 100., 10., 10.]) * scale)
    selected = select_epoch_channels(source, ("Pz", "Oz"))
    assert_actual_qc(selected)
    np.testing.assert_array_equal(selected.qc_features.epoch_scale_v, np.array([10*scale], dtype=np.float32))
    assert selected.provenance["channel_selection_qc_program"] == QC_CHANNEL_SELECTION_PROGRAM
    assert selected.lineage.entities[-1].parameters["qc_feature_program"] == QC_CHANNEL_SELECTION_PROGRAM


def test_epsilon_floor_inverse_is_wrong_even_before_summary_rounding():
    source = dataset([0., 0., 2e-19, 4e-19])
    selected = select_epoch_channels(source, ("Pz",))
    assert_actual_qc(selected)
    bad_inverse = (source.qc_features.relative_ptp[:, 2].astype(np.float64)
                   * source.qc_features.epoch_scale_v)
    assert selected.qc_features.epoch_scale_v[0] == np.float32(2e-19)
    assert bad_inverse[0] < selected.qc_features.epoch_scale_v[0] / 1000


def test_partial_static_and_trial_masks_and_repeated_selection_preserve_actual_population():
    source = dataset(np.tile([100e-6, 100e-6, 10e-6, 20e-6], (3, 1)))
    source.channel_mask[1] = False
    observed = np.tile(source.channel_mask, (3, 1))
    observed[0, 0] = False
    observed[1, 3] = False
    source.trial_channel_mask = observed
    source.X[~observed] = 0
    source.qc_features = compute_epoch_qc_features(source.X, channel_mask=source.channel_mask,
                                                   trial_channel_mask=observed)
    source.validate()
    first = select_epoch_channels(source, ("Oz", "Pz", "Cz"))
    repeated = select_epoch_channels(first, ("Pz", "Oz"))
    direct = select_epoch_channels(source, ("Pz", "Oz"))
    assert_actual_qc(first)
    assert_actual_qc(repeated)
    for field in ("relative_ptp", "channel_std_v", "epoch_scale_v", "observed_mask"):
        np.testing.assert_array_equal(getattr(repeated.qc_features, field), getattr(direct.qc_features, field))
    with pytest.raises(ValueError, match="retain one observed channel"):
        select_epoch_channels(source, ("Oz",))


def test_same_montage_and_reordering_use_actual_samples_without_mutating_parent_qc():
    source = dataset([100e-6, 100e-6, 10e-6, 20e-6])
    original = {name: np.asarray(getattr(source.qc_features, name)).copy()
                for name in ("relative_ptp", "channel_std_v", "epoch_scale_v", "observed_mask")}
    shared = select_epoch_channels(source, source.channel_names, copy=False)
    reordered = select_epoch_channels(source, ("Oz", "Pz", "Cz", "Fz"))
    assert np.shares_memory(shared.X, source.X)
    assert not shared.X.flags.writeable and source.X.flags.writeable
    assert_actual_qc(shared)
    assert_actual_qc(reordered)
    for name, values in original.items():
        np.testing.assert_array_equal(getattr(source.qc_features, name), values)


def test_missing_cached_features_remain_lazy_until_the_real_cache_writer(tmp_path):
    source = dataset([100e-6, 100e-6, 10e-6, 10e-6])
    source.qc_features = None
    selected = select_epoch_channels(source, ("Pz", "Oz"))
    assert selected.qc_features is None
    assert selected.lineage.entities[-1].parameters["qc_feature_program"] is None
    restored = load_epoch_dataset(save_epoch_dataset(tmp_path / "lazy-selected.npz", selected))
    assert_actual_qc(restored)


def test_feature_api_requires_actual_selected_samples_and_preserves_typed_empty_payload():
    source = dataset([100e-6, 100e-6, 10e-6, 10e-6])
    with pytest.raises(TypeError, match="selected_X"):
        source.qc_features.select_channels([2, 3])
    with pytest.raises(ValueError, match="align"):
        source.qc_features.select_channels([2, 3], selected_X=source.X)
    empty = source.qc_features.subset(np.array([], dtype=np.int64))
    selected = empty.select_channels([2, 3], selected_X=np.empty((0, 2, 4), dtype=np.float32))
    selected.validate(n_epochs=0, n_channels=2)
    assert selected.relative_ptp.shape == (0, 2)
    with pytest.raises(ValueError, match="at least one epoch"):
        select_epoch_channels(replace(source, X=source.X[:0]), ("Pz", "Oz"))


def test_actual_channel_owner_save_load_has_new_qc_and_derived_cache_identity(tmp_path):
    source = dataset([100e-6, 100e-6, 10e-6, 10e-6])
    parent_path = save_epoch_dataset(tmp_path / "parent.npz", source)
    parent_bytes = parent_path.read_bytes()
    loaded = load_epoch_dataset(parent_path, validation="attested")
    original_attestation = loaded_epoch_cache_attestation(loaded)
    selected = select_epoch_channels(loaded, ("Pz", "Oz"), copy=False)
    selected_path = save_epoch_dataset(tmp_path / "new-selected.npz", selected)
    restored = load_epoch_dataset(selected_path, validation="attested")
    assert_actual_qc(restored)
    child_attestation = loaded_epoch_cache_attestation(restored)
    assert child_attestation["sha256"] != original_attestation["sha256"]
    assert child_attestation["decoded_dataset_sha256"] != original_attestation["decoded_dataset_sha256"]
    assert restored.lineage.entity_digest != loaded.lineage.entity_digest
    assert restored.lineage.entities[-1].parameters["qc_feature_program"] == QC_CHANNEL_SELECTION_PROGRAM
    assert parent_path.read_bytes() == parent_bytes
    # The independent domain selection route already uses the same compute owner.
    native = adapt_native_reference_selection(source, ("Pz", "Oz"))
    assert_actual_qc(native)
    np.testing.assert_array_equal(native.qc_features.epoch_scale_v, restored.qc_features.epoch_scale_v)


def test_saved_selected_qc_gate_ignores_artifact_on_removed_electrodes(tmp_path):
    amplitude = np.tile([100e-6, 100e-6, 10e-6, 10e-6], (24, 1))
    amplitude[:, 2:] *= np.linspace(.9, 1.1, 24)[:, None]
    source = dataset(amplitude)
    source.X[-1, :2] *= 50
    source.qc_features = compute_epoch_qc_features(source.X, channel_mask=source.channel_mask)
    selected = load_epoch_dataset(save_epoch_dataset(tmp_path / "selected.npz",
        select_epoch_channels(source, ("Pz", "Oz"))))
    subjects = np.repeat(["s0", "s1", "s2", "s3"], 6)
    policy = FoldLocalArtifactPolicy(candidate_quantiles=(.9,), max_bad_channel_fraction=.5,
                                    cv_splits=2, min_clean_epochs=2, global_scale_mad_z=4.)
    fresh = compute_epoch_qc_features(selected.X, channel_mask=selected.channel_mask)
    cached_model = policy.fit(selected.X, subjects, qc_features=selected.qc_features)
    direct_model = policy.fit(selected.X, subjects, qc_features=fresh)
    np.testing.assert_array_equal(cached_model.ptp_thresholds, direct_model.ptp_thresholds)
    assert cached_model.global_scale_log_center == direct_model.global_scale_log_center
    cached = cached_model.transform(selected.X, qc_features=selected.qc_features)
    direct = direct_model.transform(selected.X, qc_features=fresh)
    np.testing.assert_array_equal(cached.global_epoch_scale_mask, direct.global_epoch_scale_mask)
    assert not cached.global_epoch_scale_mask[-1]
