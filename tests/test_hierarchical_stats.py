from __future__ import annotations

import numpy as np
import pytest

from stats.hierarchical import paired_interval


def test_conditional_interval_declares_participant_as_sampling_unit() -> None:
    result = paired_interval(
        np.asarray([0.1, -0.1, 0.2, 0.0]),
        inference_scope="conditional_frozen_models",
        iterations=1000,
        seed=7,
    )
    assert result.sampling_unit == "participant"
    assert result.n_units == 4
    assert result.mean_difference == pytest.approx(0.05)


def test_training_interval_declares_retraining_replicate_not_subject() -> None:
    result = paired_interval(
        np.asarray([0.03, -0.01, 0.02]),
        inference_scope="training_procedure",
        iterations=1000,
        seed=11,
    )
    assert result.sampling_unit == "independent_training_replicate"
    assert result.n_units == 3


def test_interval_rejects_fold_or_single_replicate_shortcuts() -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_interval(
            np.asarray([0.1]),
            inference_scope="training_procedure",
            iterations=1000,
            seed=3,
        )
    with pytest.raises(ValueError, match="unsupported"):
        paired_interval(
            np.asarray([0.1, 0.2]),
            inference_scope="folds",  # type: ignore[arg-type]
            iterations=1000,
            seed=3,
        )
