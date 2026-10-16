from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call, sentinel

import pytest

from data.lineage import DataLineage
from experiments import adapt_eeg_domain, preflight_artifact_qc

VERIFIED_LOAD = {
    "schema": "n2p3_verified_epoch_cache_load/1",
    "cache_attestation_schema": "n2p3net_epoch_cache_attestation/1",
    "full_contract_validated": True,
    "sha256": "1" * 64,
    "byte_size": 123,
    "decoded_dataset_sha256": "2" * 64,
}


@dataclass
class _LoadedDataset:
    provenance: dict[str, Any]
    lineage: DataLineage
    verified_cache_attestation: dict[str, object] | None


def test_domain_adapter_binds_the_exact_loaded_parent_into_lineage(tmp_path: Path) -> None:
    source = _LoadedDataset(
        provenance={"source": "fixture"},
        lineage=DataLineage.source(parameters={"source": "fixture"}),
        verified_cache_attestation=dict(VERIFIED_LOAD),
    )
    source_path = tmp_path / "source.npz"

    bound = adapt_eeg_domain._bind_loaded_parent_cache(source, source_path)

    assert bound.verified_cache_attestation is None
    assert bound.provenance["direct_parent_cache"] == {
        "schema": adapt_eeg_domain.DIRECT_PARENT_CACHE_SCHEMA,
        "role": "direct_parent_cache",
        "verified_load": VERIFIED_LOAD,
        "path": str(source_path),
    }
    entity = bound.lineage.entities[-1]
    assert entity.operation == "bind_loaded_epoch_cache"
    assert entity.parameters == {
        "schema": adapt_eeg_domain.DIRECT_PARENT_CACHE_SCHEMA,
        "role": "direct_parent_cache",
        "verified_load": VERIFIED_LOAD,
    }
    assert "path" not in entity.parameters


@pytest.mark.parametrize("full_contract_check,expected_calls", [
    (False, [call.load("cache.npz", require_labels=True, validation="attested"),
             call.attest(sentinel.stable)]),
    (True, [call.load("cache.npz", require_labels=True, validation="full"),
            call.write("cache.npz", sentinel.scanned, already_validated=True),
            call.load("cache.npz", require_labels=True, validation="attested"),
            call.attest(sentinel.stable)]),
])
def test_artifact_preflight_always_returns_an_attested_stable_load(
    monkeypatch, full_contract_check, expected_calls,
) -> None:
    datasets = {"full": sentinel.scanned, "attested": sentinel.stable}
    calls = Mock()
    calls.load.side_effect = lambda path, *, require_labels, validation: datasets[validation]
    calls.attest.return_value = VERIFIED_LOAD
    monkeypatch.setattr(preflight_artifact_qc, "load_epoch_dataset", calls.load)
    monkeypatch.setattr(preflight_artifact_qc, "write_epoch_dataset_record", calls.write)
    monkeypatch.setattr(preflight_artifact_qc, "loaded_epoch_cache_attestation", calls.attest)

    dataset, attestation = preflight_artifact_qc._load_verified_preflight_dataset(
        "cache.npz", full_contract_check=full_contract_check,
    )

    assert dataset is sentinel.stable
    assert attestation == VERIFIED_LOAD
    assert calls.mock_calls == expected_calls
