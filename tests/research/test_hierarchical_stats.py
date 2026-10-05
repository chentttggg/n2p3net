from __future__ import annotations

import numpy as np
import pytest

from stats.hierarchical import paired_interval


@pytest.mark.parametrize(
    "differences,scope,unit,mean,seed",
    [
        ([0.1, -0.1, 0.2, 0.0], "conditional_frozen_models", "participant", 0.05, 7),
        ([0.03, -0.01, 0.02], "training_procedure", "independent_training_replicate", 1 / 75, 11),
    ],
)
def test_interval_preserves_the_declared_independent_sampling_unit(
    differences, scope, unit, mean, seed,
):
    result = paired_interval(
        np.asarray(differences), inference_scope=scope, iterations=1000, seed=seed
    )
    assert result.sampling_unit == unit
    assert result.n_units == len(differences)
    assert result.mean_difference == pytest.approx(mean)


@pytest.mark.parametrize(
    "differences,scope,message",
    [
        ([0.1], "training_procedure", "at least two"),
        ([0.1, 0.2], "folds", "unsupported"),
    ],
)
def test_interval_rejects_fold_or_single_replicate_shortcuts(differences, scope, message):
    with pytest.raises(ValueError, match=message):
        paired_interval(np.asarray(differences), inference_scope=scope, iterations=1000, seed=3)
