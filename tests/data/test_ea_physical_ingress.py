"""Legacy EA cache bytes remain inspectable, but cannot masquerade as electrodes."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from data.epochs import (
    load_epoch_dataset,
    load_historical_epoch_dataset,
    save_epoch_dataset,
)
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.checkpoint import PreparedTransferCheckpoint, load_n2p3_trunk_checkpoint


def _legacy_cache(tmp_path):
    dataset = binary_dataset()
    checkpoint = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    path = save_epoch_dataset(tmp_path / "legacy.npz", dataset)
    # Reproduce the old on-disk format directly; active saving must now reject it.
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files if key != "decoded_dataset_sha256"}
    provenance = json.loads(payload["provenance_json"].item())
    provenance["data_alignment"] = {"schema": "n2p3_euclidean_alignment_fit/1", "scope": "source_rows"}
    payload["provenance_json"] = np.asarray(json.dumps(provenance))
    np.savez(path, **payload)
    record_path = path.with_suffix(".record.json")
    record = json.loads(record_path.read_text())
    record["provenance"] = provenance
    record["cache_attestation"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["cache_attestation"]["byte_size"] = path.stat().st_size
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return path, checkpoint


@pytest.mark.parametrize("validation", ["full", "attested"])
def test_legacy_ea_cache_only_loads_through_explicit_historical_path(tmp_path, validation):
    path, checkpoint = _legacy_cache(tmp_path)
    with pytest.raises(ValueError, match="not physical"):
        load_epoch_dataset(path, validation=validation)
    historical = load_historical_epoch_dataset(path, validation=validation)
    assert historical.provenance["data_alignment"]["scope"] == "source_rows"
    assert np.isfinite(historical.X).all()
    with pytest.raises(ValueError, match="Historical audit"):
        historical.validate()
    with pytest.raises(ValueError, match="Historical audit"):
        PreparedTransferCheckpoint(checkpoint, historical)
    with pytest.raises(ValueError, match="Historical audit"):
        save_epoch_dataset(tmp_path / "laundered.npz", historical)


def test_checkpoint_rejects_in_memory_ea_pseudovolts(tmp_path):
    dataset = binary_dataset()
    checkpoint = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    # No historical loader is involved: the common checkpoint gate must protect itself.
    dataset.provenance["data_alignment"] = {"schema": "legacy"}
    with pytest.raises(ValueError, match="not physical"):
        load_n2p3_trunk_checkpoint(checkpoint, dataset)
