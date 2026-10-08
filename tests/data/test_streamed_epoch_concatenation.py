"""Shared final source rows preserve every original concatenation identity."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.domain import namespace_epoch_dataset
from data.epochs import concatenate_epoch_datasets
from tests.support.datasets import reference_dataset


def _inputs():
    first = namespace_epoch_dataset(reference_dataset(np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 1e-7,
                                                       reference="nose"), "A")
    second = namespace_epoch_dataset(reference_dataset(np.arange(36, dtype=np.float32).reshape(3, 3, 4) * -1e-7,
                                                        reference="nose"), "B")
    return first, second


def test_streamed_source_concatenation_is_exact_and_does_not_copy_final_eeg():
    originals = _inputs()
    expected = concatenate_epoch_datasets(originals, name="source")
    shared = np.empty_like(expected.X)
    sources = []
    start = 0
    for original in originals:
        view = shared[start:start + original.n_epochs]
        np.copyto(view, original.X)
        view.flags.writeable = False
        sources.append(replace(original, X=view))
        start += original.n_epochs
    shared.flags.writeable = False
    actual = concatenate_epoch_datasets(sources, name="source", shared_X=shared)
    assert actual.X is shared and not actual.X.flags.writeable
    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(actual.subject_ids, expected.subject_ids)
    assert actual.event_timeline.fingerprint() == expected.event_timeline.fingerprint()
    assert actual.identity_table.payload() == expected.identity_table.payload()
    assert actual.lineage.payload() == expected.lineage.payload()
    assert actual.record() == expected.record()


def test_streamed_source_concat_cannot_relabel_an_unrelated_shared_buffer():
    originals = _inputs()
    unrelated = np.zeros((5, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="exact ordered rows"):
        concatenate_epoch_datasets(originals, name="source", shared_X=unrelated)


def test_streamed_source_concat_rejects_reversed_or_reordered_memory_views():
    originals = _inputs()
    shared = np.zeros((5, 3, 4), dtype=np.float32)
    shifted = [replace(originals[0], X=shared[3:5]), replace(originals[1], X=shared[:3])]
    with pytest.raises(ValueError, match="exact ordered rows"):
        concatenate_epoch_datasets(shifted, name="source", shared_X=shared)
