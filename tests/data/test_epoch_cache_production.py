"""Fresh source writers hand off their real bytes without pretending to reload."""
from __future__ import annotations

from dataclasses import replace
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest

from data.epochs import (
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    produce_epoch_cache,
)
from tests.support.datasets import reference_dataset


def _dataset():
    return reference_dataset(np.random.default_rng(817).normal(0., 1e-6, (12, 3, 4)).astype(np.float32),
                             reference="nose")


def test_fresh_written_source_state_and_attested_resume_bind_same_actual_cache(tmp_path):
    dataset = _dataset()
    production = produce_epoch_cache(tmp_path / "source.npz", dataset)
    identity = production.verify()
    assert identity["schema"] == "n2p3_verified_epoch_cache_production/1"
    assert identity["origin"] == "validated_cache_writer_same_process"
    with pytest.raises(ValueError, match="loaded with validation"):
        loaded_epoch_cache_attestation(dataset)
    resumed = load_epoch_dataset(production.path, validation="attested")
    actual = loaded_epoch_cache_attestation(resumed)
    assert identity["sha256"] == actual["sha256"]
    assert identity["decoded_dataset_sha256"] == actual["decoded_dataset_sha256"]
    np.testing.assert_array_equal(resumed.X, dataset.X)
    np.testing.assert_array_equal(resumed.y, dataset.y)
    np.testing.assert_array_equal(resumed.subject_ids, dataset.subject_ids)
    with ZipFile(production.path) as archive:
        assert archive.getinfo("X.npy").compress_type == ZIP_STORED


def test_written_source_handoff_rejects_post_write_signal_change(tmp_path):
    production = produce_epoch_cache(tmp_path / "source.npz", _dataset())
    production.dataset.X[2, 1, 0] += np.float32(1e-6)
    with pytest.raises(ValueError, match="source dataset changed"):
        production.verify()


def test_written_source_handoff_rejects_modified_persisted_cache(tmp_path):
    production = produce_epoch_cache(tmp_path / "source.npz", _dataset())
    with production.path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source cache changed"):
        production.verify()


def test_source_writer_does_not_attest_unserialized_float64_state(tmp_path):
    original = _dataset()
    dataset = replace(original, X=original.X.astype(np.float64))
    with pytest.raises(ValueError, match="canonical float32"):
        produce_epoch_cache(tmp_path / "source.npz", dataset)
    assert not (tmp_path / "source.npz").exists()


@pytest.mark.parametrize("field,dtype", [("y", np.uint8), ("channel_positions_m", np.float64)])
def test_source_production_rejects_uncanonical_serialized_metadata(tmp_path, field, dtype):
    original = _dataset()
    changed = replace(original, **{field: np.asarray(getattr(original, field)).astype(dtype)})
    with pytest.raises(ValueError, match="canonical.*" + field):
        produce_epoch_cache(tmp_path / "source.npz", changed)
    assert not (tmp_path / "source.npz").exists()


def test_source_production_does_not_change_fortran_layout_of_generic_arrays(tmp_path):
    original = _dataset()
    dataset = replace(original, X=np.asfortranarray(original.X))
    with pytest.raises(ValueError, match="canonical contiguous"):
        produce_epoch_cache(tmp_path / "source.npz", dataset)
    assert not (tmp_path / "source.npz").exists()
