from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.ea_artifact import (
    EAFeatureArtifact,
    EAFitPlan,
    assert_parent_replay,
    authorization_from_dataset,
    fit_ea_feature_artifact,
    load_ea_feature_artifact,
    save_ea_feature_artifact,
)
from data.epochs import load_epoch_dataset, save_epoch_dataset
from data.euclidean_alignment import (
    EuclideanAlignmentConfig,
    fit_euclidean_alignment,
    transform_euclidean_alignment,
)
from tests.support.ea import make_ea_dataset


def _parent(tmp_path, *, domains=None):
    dataset = make_ea_dataset("ea-artifact", ("s01", "s02", "s03", "s04"), domains=domains)
    path = tmp_path / "parent.npz"
    save_epoch_dataset(path, dataset)
    return load_epoch_dataset(path, require_labels=True, validation="attested")


def _artifact(tmp_path, *, rows=(0, 1), scope="source_rows", domains=None):
    parent = _parent(tmp_path, domains=domains)
    auth = authorization_from_dataset(parent, fit_scope=scope, fit_rows=rows)
    plan = EAFitPlan("experiment/split1", auth, "native_preserve", EuclideanAlignmentConfig())
    artifact = fit_ea_feature_artifact(parent, expected_plan=plan)
    return parent, plan, artifact


def test_ea_artifact_round_trips_and_replays_parent_identity(tmp_path):
    parent, plan, artifact = _artifact(tmp_path)
    output = save_ea_feature_artifact(tmp_path / "features.npz", artifact,
                                      parent=parent, expected_plan=plan)
    restored = load_ea_feature_artifact(output, parent=parent, expected_plan=plan)
    np.testing.assert_array_equal(restored.X, artifact.X)
    np.testing.assert_array_equal(restored.transform(parent), artifact.X)
    assert restored.component_names[0] == "ea_component_000"
    assert "channel_positions_m" not in restored.semantic_record()
    assert_parent_replay(restored, parent=parent, expected_plan=plan)


def test_ea_replay_rejects_state_fitted_outside_authorized_rows(tmp_path):
    parent, plan, artifact = _artifact(tmp_path)
    # Numerically sound EA state with correct schema/scope, wrong fit observations.
    state = fit_euclidean_alignment(parent.X[[2, 3]] * 7, fit_scope="source_rows")
    invalid = replace(artifact, alignment={**artifact.alignment, "state": state.record()},
                      X=transform_euclidean_alignment(parent.X, state))
    with pytest.raises(ValueError, match="fit replay"):
        save_ea_feature_artifact(tmp_path / "forged.npz", invalid,
                                 parent=parent, expected_plan=plan)
    assert not (tmp_path / "forged.npz").exists()


def test_ea_replay_rejects_arbitrary_output_despite_valid_state(tmp_path):
    parent, plan, artifact = _artifact(tmp_path)
    invalid = replace(artifact, X=artifact.X * 2)
    with pytest.raises(ValueError, match="output.*transform replay"):
        save_ea_feature_artifact(tmp_path / "forged.npz", invalid,
                                 parent=parent, expected_plan=plan)


def test_ea_load_requires_independent_expected_authorization(tmp_path):
    parent, plan, artifact = _artifact(tmp_path)
    output = save_ea_feature_artifact(tmp_path / "features.npz", artifact,
                                      parent=parent, expected_plan=plan)
    other = authorization_from_dataset(parent, fit_scope="source_rows", fit_rows=[2, 3])
    with pytest.raises(ValueError, match="caller's expected plan"):
        load_ea_feature_artifact(output, parent=parent, expected_plan=replace(plan, authorization=other))


def test_ea_rejects_in_memory_parent_mutation_even_with_valid_old_attestation(tmp_path):
    parent, plan, artifact = _artifact(tmp_path)
    parent.X[0] *= 7
    with pytest.raises(ValueError, match="content changed"):
        assert_parent_replay(artifact, parent=parent, expected_plan=plan)


@pytest.mark.parametrize("scope", ["target_calibration", "unlabelled_buffer"])
def test_ea_fit_only_uses_declared_rows_and_predicts_without_refit(tmp_path, scope):
    parent, plan, artifact = _artifact(tmp_path, scope=scope)
    # A source-trained transform must whiten the fit rows, not the scored rows.
    covariance = np.einsum("nct,nut->cu", artifact.X[:2], artifact.X[:2]) / (2 * parent.n_times)
    np.testing.assert_allclose(covariance, np.eye(parent.n_channels), atol=2e-7)
    changed = replace(parent, X=parent.X.copy(), verified_cache_attestation=None)
    changed.X[2:] *= 20
    path = save_epoch_dataset(tmp_path / "new.npz", changed)
    future = load_epoch_dataset(path, validation="attested")
    predicted = artifact.transform(future)
    np.testing.assert_allclose(predicted[2:], 20 * artifact.X[2:], rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(predicted[:2], artifact.X[:2])


def test_ea_per_domain_fit_requires_every_output_domain(tmp_path):
    parent = _parent(tmp_path, domains=("A", "A", "B", "B"))
    auth = authorization_from_dataset(parent, fit_scope="per_domain", fit_rows=[0, 1])
    with pytest.raises(ValueError, match="missing domain"):
        fit_ea_feature_artifact(parent, expected_plan=EAFitPlan(
            "experiment/split1", auth, "native_preserve", EuclideanAlignmentConfig()))


def test_ea_per_domain_routes_each_row_to_its_own_state(tmp_path):
    parent, plan, artifact = _artifact(tmp_path, rows=(3, 1, 2, 0), scope="per_domain",
                                      domains=("A", "A", "B", "B"))
    output = save_ea_feature_artifact(tmp_path / "domains.npz", artifact,
                                      parent=parent, expected_plan=plan)
    restored = load_ea_feature_artifact(output, parent=parent, expected_plan=plan)
    for subset in (restored.X[:2], restored.X[2:]):
        covariance = np.einsum("nct,nut->cu", subset, subset) / (2 * parent.n_times)
        np.testing.assert_allclose(covariance, np.eye(parent.n_channels), atol=2e-7)
    with pytest.raises(ValueError, match="domain routing"):
        assert_parent_replay(replace(restored, row_domain_ids=("B", "B", "A", "A")),
                             parent=parent, expected_plan=plan)


def test_ea_features_cannot_predict_after_state_edit(tmp_path):
    parent, _, artifact = _artifact(tmp_path)
    artifact.alignment["state"]["buffer_digest"] = "0" * 64
    with pytest.raises(ValueError, match="unchanged"):
        artifact.transform(parent)


def test_ea_artifact_rejects_mask_and_scope_mismatch(tmp_path):
    _, _, artifact = _artifact(tmp_path)
    with pytest.raises(ValueError, match="component masks"):
        replace(artifact, component_mask=np.zeros_like(artifact.component_mask))
    with pytest.raises(ValueError, match="scope/domain/dimension"):
        replace(artifact, fit_plan=replace(artifact.fit_plan,
                authorization=replace(artifact.authorization, fit_scope="unlabelled_buffer")))


@pytest.mark.parametrize("dynamic", [False, True])
def test_ea_authorization_rejects_missing_channels(tmp_path, dynamic):
    dataset = make_ea_dataset("missing", ("s01", "s02", "s03"))
    if dynamic:
        dataset.X[1, 1] = 0
        dataset.trial_channel_mask = np.ones((dataset.n_epochs, dataset.n_channels), dtype=bool)
        dataset.trial_channel_mask[1, 1] = False
    else:
        dataset.X[:, 0] = 0
        dataset.channel_mask[0] = False
    path = save_epoch_dataset(tmp_path / "missing.npz", dataset)
    parent = load_epoch_dataset(path, validation="attested")
    with pytest.raises(ValueError, match="parent channel"):
        authorization_from_dataset(parent, fit_scope="source_rows", fit_rows=np.arange(parent.n_epochs))


@pytest.mark.parametrize("rows", [[0.1, 1.1], [True, False], [0, 0], [-1, 0], [0, 4]])
def test_ea_authorization_rejects_invalid_indices(tmp_path, rows):
    parent = _parent(tmp_path)
    with pytest.raises(ValueError, match="fit_rows"):
        authorization_from_dataset(parent, fit_scope="source_rows", fit_rows=rows)


def test_ea_unverified_constructor_cannot_supply_predictions(tmp_path):
    parent, _, artifact = _artifact(tmp_path)
    unverified = EAFeatureArtifact(**{name: getattr(artifact, name) for name, f in
                                     artifact.__dataclass_fields__.items() if f.init})
    with pytest.raises(ValueError, match="parent-verified"):
        unverified.transform(parent)


def _resign_archive(path, artifact, **array_updates):
    """Simulate internally consistent but scientifically false producer metadata."""
    import hashlib
    import json
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload.update(array_updates)
    np.savez(path, **payload)
    path.with_suffix(".record.json").write_text(json.dumps(artifact.record(
        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), byte_size=path.stat().st_size)),
        encoding="utf-8")


def test_ea_loader_rejects_resigned_but_uncomputed_output(tmp_path):
    parent, plan, artifact = _artifact(tmp_path)
    path = save_ea_feature_artifact(tmp_path / "features.npz", artifact,
                                   parent=parent, expected_plan=plan)
    invalid = replace(artifact, X=artifact.X + 1)
    _resign_archive(path, invalid, X=invalid.X)
    with pytest.raises(ValueError, match="transform replay"):
        load_ea_feature_artifact(path, parent=parent, expected_plan=plan)


@pytest.mark.parametrize("change", ["buffer", "count", "fit_rows"])
def test_ea_loader_rejects_resigned_false_fit_record(tmp_path, change):
    import json

    from data.euclidean_alignment import EuclideanAlignmentState
    parent, plan, artifact = _artifact(tmp_path)
    state = EuclideanAlignmentState.from_record(artifact.alignment["state"])
    if change == "buffer":
        altered = replace(state, buffer_digest="0" * 64)
    elif change == "count":
        altered = replace(state, sample_count=3)
    else:
        altered = fit_euclidean_alignment(parent.X[[2, 3]], fit_scope=plan.authorization.fit_scope)
    invalid = replace(artifact, alignment={**artifact.alignment, "state": altered.record()},
                      X=transform_euclidean_alignment(parent.X, altered))
    path = save_ea_feature_artifact(tmp_path / "features.npz", artifact,
                                   parent=parent, expected_plan=plan)
    _resign_archive(path, invalid, X=invalid.X,
                   alignment_json=np.asarray(json.dumps(invalid.alignment, sort_keys=True)))
    with pytest.raises(ValueError, match="fit replay"):
        load_ea_feature_artifact(path, parent=parent, expected_plan=plan)


def test_ea_load_decodes_hashed_snapshot_if_path_changes_before_decode(tmp_path, monkeypatch):
    parent, plan, artifact = _artifact(tmp_path)
    path = save_ea_feature_artifact(tmp_path / "features.npz", artifact,
                                   parent=parent, expected_plan=plan)
    original_load = np.load
    def swap_at_decode(source, **kwargs):
        path.write_bytes(b"concurrently replaced invalid archive")
        return original_load(source, **kwargs)
    monkeypatch.setattr(np, "load", swap_at_decode)
    restored = load_ea_feature_artifact(path, parent=parent, expected_plan=plan)
    np.testing.assert_array_equal(restored.X, artifact.X)


def test_ea_fit_minimum_applies_to_each_domain(tmp_path):
    parent = _parent(tmp_path, domains=("A", "A", "B", "B"))
    auth = authorization_from_dataset(parent, fit_scope="per_domain", fit_rows=[0, 1, 2])
    with pytest.raises(ValueError, match="shape"):
        fit_ea_feature_artifact(parent, expected_plan=EAFitPlan(
            "experiment/split1", auth, "native_preserve", EuclideanAlignmentConfig()))


def test_ea_prediction_rejects_new_unfitted_domain(tmp_path):
    parent, _, artifact = _artifact(tmp_path, rows=(0, 1, 2, 3), scope="per_domain",
                                     domains=("A", "A", "B", "B"))
    dataset = make_ea_dataset("future", ("s01", "s02", "s03", "s04"), domains=("C", "C", "C", "C"))
    path = save_epoch_dataset(tmp_path / "future.npz", dataset)
    future = load_epoch_dataset(path, validation="attested")
    with pytest.raises(ValueError, match="lacks an input domain"):
        artifact.transform(future)


def test_ea_load_binds_numeric_operator_to_callers_plan(tmp_path):
    parent, plan, _ = _artifact(tmp_path)
    alternate_plan = replace(plan, config=replace(plan.config, shrinkage=0.6))
    other = fit_ea_feature_artifact(parent, expected_plan=alternate_plan)
    path = save_ea_feature_artifact(tmp_path / "different_operator.npz", other,
                                   parent=parent, expected_plan=alternate_plan)
    with pytest.raises(ValueError, match="caller's expected plan"):
        load_ea_feature_artifact(path, parent=parent, expected_plan=plan)
    loaded = load_ea_feature_artifact(path, parent=parent, expected_plan=alternate_plan)
    np.testing.assert_array_equal(loaded.transform(parent), other.X)


def test_ea_fit_plan_record_round_trips_exact_recipe_and_authorization(tmp_path):
    import json
    parent, plan, _ = _artifact(tmp_path)
    restored = EAFitPlan.from_record(json.loads(json.dumps(plan.record())))
    assert restored.record() == plan.record()
    np.testing.assert_array_equal(fit_ea_feature_artifact(parent, expected_plan=restored).X,
                                  fit_ea_feature_artifact(parent, expected_plan=plan).X)


def test_ea_artifact_owns_arrays_without_locking_or_aliasing_callers(tmp_path):
    parent, _, artifact = _artifact(tmp_path)
    mask = np.ones_like(artifact.component_mask)
    values = artifact.X.copy()
    constructed = replace(artifact, X=values, y=parent.y, subject_ids=parent.subject_ids, component_mask=mask)
    assert values.flags.writeable and parent.y.flags.writeable and parent.subject_ids.flags.writeable
    assert mask.flags.writeable
    original = constructed.X.copy()
    values *= 7
    parent.y[0] = 1 - parent.y[0]
    parent.subject_ids[0] = "changed"
    mask[0, 0] = False
    np.testing.assert_array_equal(constructed.X, original)
    assert constructed.y[0] != parent.y[0]
    assert constructed.subject_ids[0] != parent.subject_ids[0]
    assert constructed.component_mask.all()


def test_ea_unlabelled_fit_save_load_and_prediction_are_valid(tmp_path):
    parent = _parent(tmp_path)
    unlabelled = replace(parent, y=None, verified_cache_attestation=None)
    path = save_epoch_dataset(tmp_path / "unlabelled.npz", unlabelled)
    unlabelled = load_epoch_dataset(path, validation="attested")
    auth = authorization_from_dataset(unlabelled, fit_scope="unlabelled_buffer", fit_rows=[0, 1])
    plan = EAFitPlan("experiment/unlabelled", auth, "native_preserve", EuclideanAlignmentConfig())
    artifact = fit_ea_feature_artifact(unlabelled, expected_plan=plan)
    output = save_ea_feature_artifact(tmp_path / "unlabelled-ea.npz", artifact,
                                      parent=unlabelled, expected_plan=plan)
    restored = load_ea_feature_artifact(output, parent=unlabelled, expected_plan=plan)
    assert restored.y is None
    np.testing.assert_array_equal(restored.transform(unlabelled), restored.X)
