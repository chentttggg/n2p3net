from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from transfer.binary import copy_shrinkage_lda_to_binary_head, fit_shrinkage_feature_model
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def test_lda_head_folding_preserves_scaler_and_log_odds():
    rng = np.random.default_rng(18)
    train = rng.normal(size=(40, 7)) * np.arange(1, 8) + 17
    labels = (np.arange(40) % 5 == 0).astype(int)
    fitted = fit_shrinkage_feature_model(train, labels)
    head = torch.nn.Linear(7, 2, dtype=torch.float64)
    copy_shrinkage_lda_to_binary_head(fitted, head)
    probe = rng.normal(size=(13, 7)) * np.arange(1, 8) - 2
    with torch.inference_mode():
        logits = head(torch.from_numpy(probe))
    np.testing.assert_allclose(
        (logits[:, 1] - logits[:, 0]).numpy(), fitted.decision_function(probe), atol=1e-10
    )


@pytest.mark.parametrize(
    "label_offset,extra_features,coefficient_scale,match",
    [
        (1, 0, 1.0, "classes"),
        (0, 1, 1.0, "feature count"),
        (0, 0, np.nan, "nonfinite classifier parameters"),
    ],
    ids=["classes", "shape", "nonfinite"],
)
def test_invalid_lda_head_is_rejected_before_mutation(
    label_offset, extra_features, coefficient_scale, match
):
    values = np.random.default_rng(19).normal(size=(30, 4))
    labels = np.arange(len(values)) % 2 + label_offset
    fitted = fit_shrinkage_feature_model(values, labels)
    head = torch.nn.Linear(values.shape[1] + extra_features, 2)
    fitted.steps[1][1].coef_[0, 0] *= coefficient_scale
    original = copy.deepcopy(head.state_dict())
    with pytest.raises(ValueError, match=match):
        copy_shrinkage_lda_to_binary_head(fitted, head)
    for name, value in head.state_dict().items():
        assert torch.equal(value, original[name])


@pytest.mark.parametrize("pooling", ["full_unfold", "global_average"])
def test_fitted_features_share_head_inputs_and_never_update_test_bn(pooling):
    rng = np.random.default_rng(20)
    X = rng.normal(size=(32, 3, 179)).astype(np.float32) * 1e-5
    y = (np.arange(32) % 4 == 0).astype(int)
    adapter = SubjectAdapter(
        N2P3Net(3, n_times=179, pooling_mode=pooling),
        config=SubjectAdapterConfig(
            head_kind="full_fine", epochs=1, batch_size=16, seed=20,
            input_statistics="target_prefix", freeze_batchnorm_stats=False,
            val_group_fraction=None, refit_full_prefix=False,
        ),
    )
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict_features(X)
    adapter.fit(X, y)
    probe = X[:5].copy()
    before = copy.deepcopy(adapter.trunk.state_dict())
    features = adapter.predict_features(probe)
    with torch.inference_mode():
        logits = adapter.trunk.classifier(torch.from_numpy(features))
    np.testing.assert_allclose(
        (logits[:, 1] - logits[:, 0]).numpy(), adapter.predict_logit(probe), atol=2e-6
    )
    changed = np.concatenate([probe, np.ones_like(probe[:1]) * 100])
    np.testing.assert_allclose(adapter.predict_features(changed)[:5], features, atol=2e-6)
    for name, value in adapter.trunk.state_dict().items():
        assert torch.equal(value, before[name])
