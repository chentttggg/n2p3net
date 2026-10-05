from __future__ import annotations

import numpy as np
import pytest

from transfer.binary import matched_input_digests


def test_matched_input_digest_binds_features_labels_mask_and_preprocessing():
    X = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    mask = np.ones((4, 2), dtype=bool)
    spec = {"name": "contract", "sfreq": 128.0, "filter_phase": "zero"}
    baseline = matched_input_digests(X, y, np.asarray([1, 3]), mask, spec)
    assert len(baseline["input_sha256"]) == len(baseline["test_labels_sha256"]) == 64
    changed = X.copy()
    changed[1, 0, 0] += 1
    assert matched_input_digests(changed, y, [1, 3], mask, spec) != baseline
    changed_mask = mask.copy()
    changed_mask[3, 1] = False
    assert matched_input_digests(X, y, [1, 3], changed_mask, spec) != baseline
    changed_spec = {**spec, "filter_phase": "forward"}
    assert matched_input_digests(X, y, [1, 3], mask, changed_spec) != baseline


def test_matched_input_digest_rejects_unaligned_mask_and_empty_scope():
    X = np.zeros((2, 1, 4), dtype=np.float32)
    y = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="nonempty"):
        matched_input_digests(X, y, [], None, {})
    with pytest.raises(ValueError, match="aligned"):
        matched_input_digests(X, y, [0], np.ones((1, 1), dtype=bool), {})
