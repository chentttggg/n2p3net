from __future__ import annotations

import numpy as np
import pytest

from baselines.structured_erp import StructuredERP


def test_toeplitz_lda_learns_polarity_and_roundtrips():
    rng = np.random.default_rng(12)
    y = np.tile([0, 1], 20)
    X = rng.normal(size=(40, 3, 24)) * 1e-6
    X[:, 1] += y[:, None] * np.sin(np.linspace(0, np.pi, 24)) * 3e-6
    fitted = StructuredERP("toeplitz_lda", bin_width=3, baseline_mode="none").fit(X, y)
    scores = fitted.predict_logit(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()
    record, state = fitted.export_state()
    restored = StructuredERP.from_state(record, state)
    np.testing.assert_array_equal(restored.predict_logit(X), scores)
    assert np.linalg.eigvalsh(fitted.covariance_).min() > 0


def test_standard_toeplitz_requires_no_baseline():
    with pytest.raises(ValueError, match="baseline"):
        StructuredERP("toeplitz_lda", baseline_mode="trial")


def test_toeplitz_matches_frozen_official_reference_scores():
    # Official jsosulski/toeplitzlda numerical classes, source hashes and the
    # independent execution harness are recorded in toeplitz_reference/parity.json.
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 3, 8))
    y = np.arange(40) % 2
    X[y == 1] += .3
    scores = StructuredERP("toeplitz_lda").fit(X, y).predict_logit(X)
    np.testing.assert_allclose(scores[:5], [-1.1305659857169839, 2.938014724141602,
        .796695391769547, .8212817516322777, -2.3569781327600623], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("family", ["xdawn_erp", "toeplitz_lda", "pooled_lda"])
def test_families_preserve_scale_and_reject_nonfinite_or_wrong_geometry(family):
    rng = np.random.default_rng(35)
    X = rng.normal(size=(30, 3, 24))
    y = np.arange(30) % 2
    X[y == 1, 1] += np.sin(np.linspace(0, np.pi, 24))
    fit = StructuredERP(family, bin_width=2).fit(X, y)
    scaled = StructuredERP(family, bin_width=2).fit(X * 1e-6, y)
    np.testing.assert_allclose(fit.predict_logit(X), scaled.predict_logit(X * 1e-6), atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="geometry"):
        fit.predict_logit(X[:, :, :-1])
    bad = X.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit.predict_logit(bad)
    with pytest.raises(ValueError, match="binary"):
        StructuredERP(family).fit(X, np.zeros(len(X)))
    record, state = fit.export_state()
    state["coefficient"][0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        StructuredERP.from_state(record, state)


@pytest.mark.parametrize("family", ["xdawn_erp", "toeplitz_lda", "pooled_lda"])
def test_singular_spatial_observations_and_fewer_trials_than_features(family):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(8, 1, 24))
    X = np.repeat(X, 4, axis=1)
    y = np.arange(8) % 2
    fitted = StructuredERP(family, bin_width=2).fit(X, y)
    assert np.isfinite(fitted.predict_logit(X)).all()
    record, state = fitted.export_state()
    np.testing.assert_array_equal(StructuredERP.from_state(record, state).predict_logit(X), fitted.predict_logit(X))
    if family == "xdawn_erp":
        assert record["observed_rank"] == record["effective_nfilter"] == 1
        assert state["filters"].shape == (2, 4)


def test_toeplitz_is_equivariant_under_channel_permutation():
    rng = np.random.default_rng(88)
    X = rng.normal(size=(14, 3, 21))
    X *= np.array([1., 10., .3])[None, :, None]
    y = np.arange(14) % 2
    original = StructuredERP("toeplitz_lda", bin_width=3).fit(X, y)
    permuted = StructuredERP("toeplitz_lda", bin_width=3).fit(X[:, [2, 0, 1]], y)
    np.testing.assert_allclose(original.predict_logit(X), permuted.predict_logit(X[:, [2, 0, 1]]), atol=1e-10)
