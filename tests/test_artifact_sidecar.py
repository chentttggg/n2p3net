from __future__ import annotations

import numpy as np

from data import artifact_sidecar
from data.artifact import FoldLocalArtifactModel, FoldLocalArtifactPolicy


def _folds() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.array([True, True, False, False]),
            np.array([False, False, True, True]),
        )
    ]


def _model(policy: FoldLocalArtifactPolicy) -> FoldLocalArtifactModel:
    return FoldLocalArtifactModel(
        policy=policy,
        ptp_thresholds=np.array([1.0, 2.0]),
        flat_std_thresholds=np.array([0.01, 0.02]),
        selected_quantiles=np.array([0.95, 0.975]),
        selected_bad_channel_fraction=0.25,
        global_scale_log_center=-10.0,
        global_scale_log_robust_std=0.5,
        fit_n_epochs=20,
        fit_groups=("1", "2"),
    )


def test_fold_qc_fingerprint_excludes_sidecar_schema_and_code_metadata(monkeypatch) -> None:
    policy = FoldLocalArtifactPolicy()
    kwargs = {"cache_sha256": "a" * 64, "folds": _folds(), "policy": policy}
    before = artifact_sidecar.fold_artifact_fingerprint(**kwargs)

    monkeypatch.setattr(artifact_sidecar, "SIDECAR_SCHEMA", "different/schema")
    after = artifact_sidecar.fold_artifact_fingerprint(**kwargs)

    assert before == after


def test_fold_qc_fingerprint_changes_with_policy_or_fold_identity() -> None:
    base = artifact_sidecar.fold_artifact_fingerprint(
        cache_sha256="b" * 64,
        folds=_folds(),
        policy=FoldLocalArtifactPolicy(),
    )
    changed_policy = artifact_sidecar.fold_artifact_fingerprint(
        cache_sha256="b" * 64,
        folds=_folds(),
        policy=FoldLocalArtifactPolicy(global_scale_mad_z=5.0),
    )
    changed_folds = artifact_sidecar.fold_artifact_fingerprint(
        cache_sha256="b" * 64,
        folds=[(_folds()[0][1], _folds()[0][0])],
        policy=FoldLocalArtifactPolicy(),
    )

    assert len({base, changed_policy, changed_folds}) == 3


def test_fold_qc_sidecar_round_trip_and_schema_miss(tmp_path) -> None:
    policy = FoldLocalArtifactPolicy()
    fingerprint = artifact_sidecar.fold_artifact_fingerprint(
        cache_sha256="c" * 64,
        folds=_folds(),
        policy=policy,
    )
    path = tmp_path / "fold-qc.json"
    artifact_sidecar.save_fold_artifact_sidecar(
        path,
        fingerprint=fingerprint,
        cache_sha256="c" * 64,
        policy=policy,
        models={0: _model(policy)},
    )

    loaded = artifact_sidecar.load_fold_artifact_sidecar(
        path,
        expected_fingerprint=fingerprint,
        expected_fold_count=1,
    )

    assert loaded is not None
    np.testing.assert_allclose(loaded[0].ptp_thresholds, [1.0, 2.0])
    assert loaded[0].fit_groups == ("1", "2")

    payload = path.read_text(encoding="utf-8").replace(
        artifact_sidecar.SIDECAR_SCHEMA, "unsupported/schema"
    )
    path.write_text(payload, encoding="utf-8")
    assert (
        artifact_sidecar.load_fold_artifact_sidecar(
            path,
            expected_fingerprint=fingerprint,
            expected_fold_count=1,
        )
        is None
    )
