"""Small identity, study-axis and cache-writer boundaries; no EEG or fitting."""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from data import artifact_sidecar
from data.artifact import FoldLocalArtifactPolicy
from data.exclusions import assert_identity_participants_allowed, assert_origin_keys_allowed
from data.identity import (
    DatasetIdentityTable,
    ParticipantIdentityRecord,
    origin_subject_key,
    source_dataset_ids_from_identity,
)
from data.study import STUDY_MEMBERSHIP_SCHEMA, checkpoint_study_membership, study_ids_for_identity
from research.contracts import TrainingRunContract, semantic_sha256


def _table(key):
    return DatasetIdentityTable((ParticipantIdentityRecord("p", (key,), (), "source_verified"),))


def _checkpoint(domain):
    table = _table(origin_subject_key("capture-A", "p"))
    contract = TrainingRunContract(
        source_cache_sha256="1" * 64,
        source_identity_digest=table.digest(),
        source_snapshot_sha256="2" * 64,
        architecture={"fixture": "no_model"},
        preprocessing={"fixture": "no_eeg"},
        optimizer={"fixture": "no_fit"},
        validation={"fixture": "no_fit"},
        objective={"source_risk_contract": {"domain_mass": {domain: 1.0}}},
        seed=0,
        training_participant_keys=table.authority_keys(),
        holdout_participant_keys=(),
    )
    return table, {
        "training_contract": contract.record(),
        "training_contract_digest": contract.digest(),
        "training_identity_ledger": table.payload(),
        "training_identity_ledger_digest": table.digest(),
    }


def _proof(study):
    content = {
        "schema": STUDY_MEMBERSHIP_SCHEMA,
        "stream_to_study": {"capture-A": study},
        "evidence": [{"basis": "synthetic_exact_producer_boundary", "study_id": study}],
    }
    return {**content, "digest": semantic_sha256(content)}


@pytest.mark.parametrize("domain", ["capture-A", "independent-training-domain", "Study"])
def test_exact_proof_resolves_study_independently_of_risk_domain_label(domain):
    table, payload = _checkpoint(domain)
    old = copy.deepcopy(payload)
    membership = checkpoint_study_membership(payload, proofs=(_proof("Study"),))
    assert study_ids_for_identity(table, membership) == ("Study",)
    assert payload == old


def test_study_proof_gates_remain_closed():
    _, payload = _checkpoint("independent-training-domain")
    with pytest.raises(ValueError, match="unresolved.*stream"):
        checkpoint_study_membership(payload)
    with pytest.raises(ValueError, match="conflicting.*proof"):
        checkpoint_study_membership(payload, proofs=(_proof("A"), _proof("B")))
    forged = _proof("A")
    forged["stream_to_study"]["capture-A"] = "B"
    with pytest.raises(ValueError, match="digest"):
        checkpoint_study_membership(payload, proofs=(forged,))


def test_exact_stream_fallback_preserves_historical_membership():
    table, payload = _checkpoint("capture-A")
    membership = checkpoint_study_membership(payload)
    assert study_ids_for_identity(table, membership) == ("capture-A",)
    assert membership["evidence"][0]["basis"] == "exact_source_id_in_bound_training_domain_mass"


def test_unproved_exact_stream_fallback_keeps_bound_domain_roster_gate():
    _, payload = _checkpoint("capture-A")
    risk = payload["training_contract"]["objective"]["source_risk_contract"]
    risk["domain_mass"] = {"capture-A": 0.5, "unproved-extra-domain": 0.5}
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    with pytest.raises(ValueError, match="bound source risk domains"):
        checkpoint_study_membership(payload)


@pytest.mark.parametrize("origin", [
    "n2p3-origin://BI2014a/participant/62/extra",
    "n2p3-origin://BI2014a/participant/",
    "n2p3-origin://BI2014a/other/62",
    "n2p3-origin://BI2014a/participant/62?ignored=1",
    "n2p3-origin://BI2014a/participant/62#ignored",
    "n2p3-origin://BI2014a/participant/%36%32",
])
def test_noncanonical_source_origins_fail_at_source_and_exclusion_consumers(origin):
    with pytest.raises(ValueError, match="canonical origin"):
        source_dataset_ids_from_identity(_table(origin))
    with pytest.raises(ValueError, match="canonical origin"):
        assert_origin_keys_allowed((origin,))
    with pytest.raises(ValueError, match="canonical origin"):
        assert_identity_participants_allowed(_table(origin))


def test_canonical_producer_roundtrip_and_author_exclusions():
    assert_origin_keys_allowed((origin_subject_key("BI2014a", "61"),))
    assert_identity_participants_allowed(_table(origin_subject_key("BI2014a", "61")))
    with pytest.raises(ValueError, match="excluded"):
        assert_identity_participants_allowed(_table(origin_subject_key("BI2014a", "62")))
    source, person = "Study / 中文?", "p/1?#%"
    key = origin_subject_key(source, person)
    assert source_dataset_ids_from_identity(_table(key)) == (source,)
    assert_origin_keys_allowed((key,))


@pytest.mark.parametrize("key", [
    "opaque-authority", "source-a:s", "registry:participant-17",
    "other://BI2014a/participant/62", "external://[opaque-authority",
])
def test_opaque_authority_keys_are_not_source_library_addresses(key):
    table = _table(key)
    assert table.records[0].authority_key() == key
    assert_origin_keys_allowed((key,))
    assert_identity_participants_allowed(table)
    with pytest.raises(ValueError):
        source_dataset_ids_from_identity(table)


def _save(path, writer):
    return artifact_sidecar.save_fold_artifact_sidecar(
        path,
        fingerprint="a" * 64,
        cache_sha256="b" * 64,
        policy=FoldLocalArtifactPolicy(),
        models={0: SimpleNamespace(record=lambda: {"writer": writer})},
    )


def test_concurrent_sidecar_writers_publish_complete_payloads_without_temp_collision(tmp_path, monkeypatch):
    destination = tmp_path / "fold.json"
    both_written = threading.Barrier(2)
    first_published = threading.Event()
    entry_lock = threading.Lock()
    entries = 0
    original_replace = Path.replace

    def synchronized_replace(source, target):
        nonlocal entries
        if Path(target) == destination:
            with entry_lock:
                entries += 1
                ordinal = entries
            both_written.wait(timeout=10)
            if ordinal == 2:
                assert first_published.wait(timeout=10)
            else:
                try:
                    return original_replace(source, target)
                finally:
                    first_published.set()
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_save, destination, writer) for writer in ("left", "right")]
        assert [future.result(timeout=15) for future in futures] == [destination, destination]
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["models"] in ({"0": {"writer": "left"}}, {"0": {"writer": "right"}})
    assert payload["fingerprint"] == "a" * 64
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fold.json"]


def test_failed_sidecar_publish_cleans_only_its_temp_and_preserves_existing_cache(tmp_path, monkeypatch):
    destination = tmp_path / "fold.json"
    destination.write_text("previous cache", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        _save(destination, "unpublished")
    assert destination.read_text(encoding="utf-8") == "previous cache"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fold.json"]
