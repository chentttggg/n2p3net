import copy
from dataclasses import replace

import numpy as np
import pytest

from baselines.temporal_transfer import TemporalBasis
from tests.support.datasets import binary_dataset


def test_source_temporal_basis_is_training_only_and_channel_count_independent():
    source = binary_dataset()
    rows = np.arange(20)
    basis = TemporalBasis.fit_source([(source, rows)], rank=2)
    changed = copy.deepcopy(source)
    changed.X[20:] *= 91
    changed.y[20:] = 1 - changed.y[20:]
    again = TemporalBasis.fit_source([(changed, rows)], rank=2)
    np.testing.assert_array_equal(basis.matrix, again.matrix)
    restored = TemporalBasis.from_record(basis.record())
    restored.verify_source([(source, rows)])
    np.testing.assert_array_equal(restored.project(source.X[:, :1]), basis.project(source.X[:, :1]))
    assert basis.project(source.X[:, :1]).shape == (source.n_epochs, 1, 2)
    with pytest.raises(ValueError, match="participant"):
        basis.validate_target(source)
    bad = basis.record()
    bad["basis"][0][0] += .1
    with pytest.raises(ValueError, match="digest"):
        TemporalBasis.from_record(bad)
    changed.X[0, 0, 0] *= 7
    with pytest.raises(ValueError, match="replay"):
        restored.verify_source([(changed, rows)])


def test_fixed_temporal_basis_has_no_source_claim_and_requires_exact_time_contract():
    data = binary_dataset()
    basis = TemporalBasis.fixed_cosine(data.preprocessing, rank=4)
    assert basis.record()["fit_role"] == "fixed_analytic"
    basis.validate_target(data)
    np.testing.assert_allclose(basis.matrix.T @ basis.matrix, np.eye(4), atol=1e-14)
    altered = replace(data, preprocessing=replace(data.preprocessing, l_freq=2.))
    with pytest.raises(ValueError, match="preprocessing"):
        basis.validate_target(altered)
    from research.contracts import semantic_sha256
    forged = basis.record()
    forged["basis"] = np.linalg.qr(np.random.default_rng(22).normal(size=(data.n_times, 4)))[0].tolist()
    forged["digest"] = semantic_sha256({k: v for k, v in forged.items() if k != "digest"})
    with pytest.raises(ValueError, match="cosine"):
        TemporalBasis.from_record(forged)


def test_source_basis_cannot_hide_same_study_under_new_subject_or_fabricate_study_ledger():
    from data.identity import DatasetIdentityTable
    from research.contracts import semantic_sha256

    source = binary_dataset()
    basis = TemporalBasis.fit_source([(source, np.arange(20))], rank=2)
    target = copy.deepcopy(source)
    target.subject_ids[:] = "p2"
    target.identity_table = DatasetIdentityTable.from_source_rows(["p2"], [basis.record()["source_studies"][0]])
    with pytest.raises(ValueError, match="complete target study"):
        basis.validate_target(target)
    fake = basis.record()
    fake["source_studies"] = ["fictional_safe_source"]
    fake["digest"] = semantic_sha256({k: v for k, v in fake.items() if k != "digest"})
    with pytest.raises(ValueError, match="study ledger"):
        TemporalBasis.from_record(fake)


def test_source_basis_uses_local_observed_channels_and_rejects_unidentified_rank():
    source = binary_dataset()
    source.channel_mask[:] = [True, False, False]
    source.X[:, 1:] = 0
    fitted = TemporalBasis.fit_source([(source, np.arange(20))], rank=1)
    assert len(fitted.record()["fit_receipts"][0]["observed_channels"]) == 1
    with pytest.raises(ValueError, match="identifiable"):
        TemporalBasis.fit_source([(source, np.arange(20))], rank=2)
    with pytest.raises(ValueError, match="twice"):
        TemporalBasis.fit_source([(source, np.arange(20)), (source, np.arange(20, 40))], rank=1)


def test_source_temporal_subspace_is_invariant_to_orthogonal_spatial_rotation_and_gain():
    source = binary_dataset()
    original = TemporalBasis.fit_source([(source, np.arange(20))], rank=2)
    rotated = copy.deepcopy(source)
    Q, _ = np.linalg.qr(np.random.default_rng(99).normal(size=(3, 3)))
    rotated.X = (Q @ source.X * 1000).astype(source.X.dtype)
    alternative = TemporalBasis.fit_source([(rotated, np.arange(20))], rank=2)
    # Compare projectors: arbitrary signs inside an eigenspace are not evidence
    # of a different temporal hypothesis.
    np.testing.assert_allclose(original.matrix @ original.matrix.T,
                               alternative.matrix @ alternative.matrix.T, atol=2e-8, rtol=1e-5)
