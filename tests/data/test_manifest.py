from __future__ import annotations

import mne
import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy, parse_candidate_quantiles
from data.epochs import PreprocessingSpec
from data.manifest import build_manifest_dataset, load_manifest, resolve_manifest_channels
from experiments.prepare_eeg_dataset import _derive_exclusive_n_times
from experiments.run_eeg_loso import _safe_auto_run_component, _validate_run_name
from tests.support.io import read_json, write_json

POSITIONS = {
    "X1": (-0.04, 0.03, 0.08),
    "X2": (0.00, 0.01, 0.10),
    "X3": (0.04, -0.01, 0.08),
    "X4": (0.00, -0.05, 0.07),
}


def _write_raw(path, channels: tuple[str, ...], seed: int) -> None:
    rng = np.random.default_rng(seed)
    info = mne.create_info(channels, sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(rng.normal(0.0, 5e-6, (len(channels), 800)), info, verbose=False)
    montage = mne.channels.make_dig_montage(
        ch_pos={channel: POSITIONS[channel] for channel in channels},
        coord_frame="head",
    )
    raw.set_montage(montage)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            duration=[0.0] * 6,
            description=["NonTarget", "Target"] * 3,
        )
    )
    raw.save(path, overwrite=True, verbose=False)


def _write_manifest(tmp_path, *, layout_policy: str = "intersection"):
    _write_raw(tmp_path / "s1_raw.fif", ("X1", "X2", "X3"), seed=1)
    _write_raw(tmp_path / "s2_raw.fif", ("X2", "X3", "X4"), seed=2)
    manifest_path = tmp_path / "dataset.json"
    write_json(
        manifest_path,
        {
            "schema": "n2p3net_raw_manifest/2",
            "name": "heterogeneous_cap",
            "montage": "embedded",
            "layout_policy": layout_policy,
            "label_map": {"NonTarget": 0, "Target": 1},
            "preprocessing": PreprocessingSpec(
                name="test_profile", sfreq=100.0, l_freq=None, h_freq=None,
                tmin_ms=-200.0, tmax_ms=800.0, n_times=100, baseline_mode="none",
                reject_threshold_v=None,
            ).record(),
            "records": [
                {
                    "path": "s1_raw.fif",
                    "subject_id": "s1",
                    "session": "ignored",
                    "reference": "linked-mastoids",
                },
                {
                    "path": "s2_raw.fif",
                    "subject_id": "s2",
                    "reference": "linked-mastoids",
                },
            ],
        },
    )
    return manifest_path


def test_manifest_intersection_builds_fixed_physical_layout(tmp_path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path))
    assert resolve_manifest_channels(manifest) == ("X2", "X3")
    dataset = build_manifest_dataset(manifest)
    assert dataset.channel_names == ("X2", "X3")
    assert dataset.X.shape == (12, 2, 100)
    assert dataset.channel_mask.all()
    assert np.isfinite(dataset.X).all()
    assert set(dataset.subject_ids) == {"s1", "s2"}
    assert np.array_equal(dataset.y, np.tile([0, 1], 6))
    assert dataset.metadata.loc[dataset.metadata["subject"] == "s1", "session"].eq("ignored").all()
    assert dataset.provenance["layout_policy"] == "intersection"
    assert dataset.provenance["montage"] == "embedded"
    registrations = dataset.provenance["coordinate_registration"]["per_subject"]
    assert set(registrations) == {"s1", "s2"}
    assert all(record["source"] == "individual_digitization" for record in registrations.values())
    assert all(record["output_frame"] == "head" for record in registrations.values())
    assert all(record["units"] == "m" for record in registrations.values())
    receipts = dataset.provenance["preprocessing_execution"]
    assert receipts["schema"] == "n2p3_preprocessing_execution_bundle/1"
    assert receipts["unverified_inputs"] == 0
    assert all(item["declaration"] == manifest.preprocessing.record() for item in receipts["recordings"])


def test_manifest_explicit_preprocessing_cannot_silently_fill_missing_fields(tmp_path):
    path = _write_manifest(tmp_path)
    payload = read_json(path)
    payload["preprocessing"] = {}
    write_json(path, payload)
    with pytest.raises(ValueError, match="complete|missing"):
        load_manifest(path)


def test_manifest_executes_declared_mean_baseline(tmp_path) -> None:
    path = _write_manifest(tmp_path)
    payload = read_json(path)
    payload["preprocessing"]["baseline_mode"] = "mean_only"
    write_json(path, payload)

    dataset = build_manifest_dataset(load_manifest(path))

    np.testing.assert_allclose(dataset.X[:, :, :20].mean(axis=2), 0.0, atol=1e-10)


def test_manifest_forward_filter_uses_declared_initial_state(tmp_path) -> None:
    path = _write_manifest(tmp_path)
    payload = read_json(path)
    payload["preprocessing"].update(
        l_freq=0.1,
        h_freq=30.0,
        filter_phase="forward",
        causal_iir_initial_state="steady_state_first_sample",
        baseline_mode="mean_only",
    )
    write_json(path, payload)
    dataset = build_manifest_dataset(load_manifest(path))
    assert dataset.X.shape == (12, 2, 100)
    assert np.isfinite(dataset.X).all()
    assert dataset.preprocessing.causal_iir_initial_state == "steady_state_first_sample"
    np.testing.assert_allclose(dataset.X[:, :, :20].mean(axis=2), 0.0, atol=1e-10)


def test_manifest_explicit_channel_positions_are_used(tmp_path) -> None:
    path = _write_manifest(tmp_path)
    payload = read_json(path)
    payload["records"] = [payload["records"][0]]
    payload["channels"] = ["X1", "X2", "X3"]
    payload["montage"] = None
    payload["channel_positions_m"] = {name: list(POSITIONS[name]) for name in ("X1", "X2", "X3")}
    write_json(path, payload)

    dataset = build_manifest_dataset(load_manifest(path))

    assert np.allclose(
        dataset.channel_positions_m, np.asarray([POSITIONS[name] for name in ("X1", "X2", "X3")])
    )
    assert all(
        registration["source"] == "manifest"
        for registration in dataset.provenance["coordinate_registration"]["per_subject"].values()
    )


def test_manifest_strict_rejects_different_channel_sets(tmp_path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, layout_policy="strict"))
    with pytest.raises(ValueError, match="Strict layout policy"):
        resolve_manifest_channels(manifest)


def test_manifest_union_builds_zero_filled_trial_masks(tmp_path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, layout_policy="union"))
    assert resolve_manifest_channels(manifest) == ("X1", "X2", "X3", "X4")

    dataset = build_manifest_dataset(manifest)

    assert dataset.X.shape == (12, 4, 100)
    assert dataset.trial_channel_mask is not None
    first = dataset.subject_ids == "s1"
    second = dataset.subject_ids == "s2"
    assert dataset.trial_channel_mask[first].all(axis=0).tolist() == [True, True, True, False]
    assert dataset.trial_channel_mask[second].all(axis=0).tolist() == [False, True, True, True]
    assert np.count_nonzero(dataset.X[first, 3]) == 0
    assert np.count_nonzero(dataset.X[second, 0]) == 0


def test_manifest_rejects_fractional_label_mapping(tmp_path) -> None:
    path = _write_manifest(tmp_path)
    payload = read_json(path)
    payload["label_map"] = {"NonTarget": 0.9, "Target": 1.0}
    write_json(path, payload)

    with pytest.raises(ValueError, match="label_map values must be integers"):
        build_manifest_dataset(load_manifest(path))


def test_manifest_rejects_unknown_fields(tmp_path) -> None:
    path = _write_manifest(tmp_path)
    payload = read_json(path)
    payload["montgae"] = "standard_1005"
    write_json(path, payload)
    with pytest.raises(ValueError, match="unknown fields"):
        load_manifest(path)


def test_runner_run_names_are_relative_and_path_safe() -> None:
    assert _validate_run_name("bnci/dep4_fold2") == "bnci/dep4_fold2"
    assert _safe_auto_run_component("BNCI 2014/008") == "BNCI_2014_008"
    for bad in ("../escape", "/absolute", "C:\\absolute", "a//b", ".", "a/../b"):
        with pytest.raises(ValueError, match="safe relative"):
            _validate_run_name(bad)


def test_preprocessing_derives_exclusive_endpoint_width() -> None:
    assert _derive_exclusive_n_times(sfreq=256.0, tmin_ms=-200.0, tmax_ms=800.0) == 256
    assert _derive_exclusive_n_times(sfreq=128.0, tmin_ms=0.0, tmax_ms=1000.0) == 128


def test_artifact_quantile_parser_preserves_the_declared_order() -> None:
    assert parse_candidate_quantiles("0.99, 0.995, 0.999") == (0.99, 0.995, 0.999)


def test_artifact_policy_accepts_the_gt_n_low_variance_floor() -> None:
    FoldLocalArtifactPolicy(
        candidate_quantiles=parse_candidate_quantiles("0.99,0.995,0.999"), flat_quantile=0.0
    ).validate()
