"""Large source QC keeps the original per-epoch numeric statistics."""
from __future__ import annotations

import numpy as np
import pytest

from data.qc_features import compute_epoch_qc_features


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["slice", "fortran", "reverse"])
def test_source_qc_matches_full_float64_reductions_with_missing_channels(dtype, layout):
    # Non-contiguous, multiple blocks and a partial final block exercise the
    # actual source-memory path without requiring a full public source pool.
    data = np.random.default_rng(80217).normal(0., 3e-6, (8193, 6, 181)).astype(dtype)[:, :, :179]
    if layout == "fortran":
        data = np.asfortranarray(data)
    elif layout == "reverse":
        data = data[::-1, :, ::-1]
    data[3] = 0
    observed = np.ones(data.shape[:2], dtype=bool)
    observed[::3, 1] = False
    observed[1::7, 4] = False
    original = data.astype(np.float64, copy=False)
    ptp = np.ptp(original, axis=2)
    scale = np.nanmedian(np.where(observed, ptp, np.nan), axis=1)
    relative = np.where(observed, ptp / np.maximum(scale[:, None], np.finfo(np.float64).eps), 0.).astype(np.float32)
    deviation = np.where(observed, np.std(original, axis=2), 0.).astype(np.float32)
    actual = compute_epoch_qc_features(data, channel_mask=np.ones(6, dtype=bool), trial_channel_mask=observed)
    np.testing.assert_array_equal(actual.relative_ptp, relative)
    np.testing.assert_array_equal(actual.channel_std_v, deviation)
    np.testing.assert_array_equal(actual.epoch_scale_v, scale.astype(np.float32))
    np.testing.assert_array_equal(actual.observed_mask, observed)


def test_source_qc_rejects_nonfinite_in_late_block():
    values = np.zeros((5001, 6, 179), dtype=np.float32)
    values[-1, 2, 18] = np.nan
    with pytest.raises(ValueError, match="finite floating-point"):
        compute_epoch_qc_features(values, channel_mask=np.ones(6, dtype=bool))


@pytest.mark.parametrize("shape", [(1, 0, 179), (1, 6, 0)])
def test_source_qc_rejects_empty_physical_axes(shape):
    with pytest.raises(ValueError, match="non-empty"):
        compute_epoch_qc_features(np.zeros(shape, dtype=np.float32), channel_mask=np.ones(shape[1], dtype=bool))
