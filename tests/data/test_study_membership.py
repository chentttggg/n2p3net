"""Study grouping follows explicit ingress evidence, never identity-name prefixes."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from data.bi2014a_candidate import BI2014A_CANDIDATE_TASK_CONTRACT
from data.domain import (
    annotate_source_domain,
    namespace_epoch_dataset,
    project_binary_evidence_source_view,
)
from data.epochs import concatenate_epoch_datasets, save_epoch_dataset
from data.study import (
    checkpoint_study_membership,
    dataset_study_membership,
    load_study_membership_proof,
    study_ids_for_identity,
)
from tests.support.checkpoints import checkpoint_identity_fields
from tests.support.datasets import reference_dataset
from tests.support.io import read_json, write_json


def _stream(stream, person, study):
    data = reference_dataset(np.zeros((4, 3, 128), np.float32), reference="A1")
    data = replace(
        data,
        name=study,
        subject_ids=np.repeat(person, 4),
        preprocessing=replace(data.preprocessing, sfreq=128, n_times=128),
        event_timeline=replace(
            data.event_timeline, dataset_ids=np.repeat(stream, 4), subject_ids=np.repeat(person, 4)
        ),
        metadata=data.metadata.assign(subject=person),
        provenance={
            **data.provenance,
            "candidate_task_contract": replace(
                BI2014A_CANDIDATE_TASK_CONTRACT, dataset_id=study
            ).record(),
        },
    )
    return project_binary_evidence_source_view(data)


def test_lineage_groups_exact_streams_under_explicit_study_without_prefix_guessing():
    left = namespace_epoch_dataset(_stream("unrelated-stream-A", "a", "Study"), "left")
    right = namespace_epoch_dataset(_stream("unrelated-stream-B", "b", "Study"), "right")
    merged = concatenate_epoch_datasets([left, right], name="Merged")
    record = dataset_study_membership(merged)
    assert record["stream_to_study"] == {
        "unrelated-stream-A": "Study",
        "unrelated-stream-B": "Study",
    }
    assert study_ids_for_identity(merged.identity_table, record) == ("Study",)
    alias = annotate_source_domain(merged, "unrelated-domain-name")
    assert dataset_study_membership(alias)["stream_to_study"] == record["stream_to_study"]


def test_native_proof_reads_embedded_metadata_and_preserves_legacy_checkpoint_identity(tmp_path):
    data = _stream("arbitrary-capture", "a", "Study")
    cache = save_epoch_dataset(tmp_path / "native.npz", data)
    proof = load_study_membership_proof(cache)
    payload = checkpoint_identity_fields(data.identity_table, data)
    old = copy.deepcopy(payload)
    membership = checkpoint_study_membership(payload, proofs=(proof,))
    assert study_ids_for_identity(data.identity_table, membership) == ("Study",)
    assert membership["proofs"][0]["cache_sha256"] == proof["proof_source"]["cache_sha256"]
    assert payload == old
    with pytest.raises(ValueError, match="unresolved.*stream"):
        checkpoint_study_membership(payload)


def test_conflicting_explicit_study_assignments_are_rejected():
    left = namespace_epoch_dataset(_stream("same-stream", "a", "StudyA"), "left")
    right = namespace_epoch_dataset(_stream("same-stream", "b", "StudyB"), "right")
    merged = concatenate_epoch_datasets([left, right], name="Merged")
    with pytest.raises(ValueError, match="conflicting.*study"):
        dataset_study_membership(merged)


def test_whole_study_gate_catches_other_person_with_unrelated_stream_name():
    from transfer.checkpoint import assert_checkpoint_library_exclusion

    source = _stream("capture-X", "a", "Study")
    target = _stream("capture-Y", "b", "Study")
    payload = checkpoint_identity_fields(source.identity_table, source)
    source_membership = checkpoint_study_membership(
        payload, proofs=(dataset_study_membership(source),)
    )
    with pytest.raises(ValueError, match="excluded target libraries"):
        assert_checkpoint_library_exclusion(
            payload,
            target.identity_table,
            source_membership=source_membership,
            target_membership=dataset_study_membership(target),
        )


def test_membership_proof_rejects_forged_sidecar_candidate_contract(tmp_path):

    data = _stream("capture-X", "a", "Study")
    cache = save_epoch_dataset(tmp_path / "native.npz", data)
    record_path = cache.with_suffix(".record.json")
    record = read_json(record_path)
    record["provenance"]["dataset_class"] = "OtherStudy"
    write_json(record_path, record)
    with pytest.raises(ValueError, match="sidecar metadata differs"):
        load_study_membership_proof(cache)


def test_checkpoint_identity_gate_rejects_unbound_new_study_membership():
    from transfer.checkpoint import checkpoint_training_identity_ledger

    data = _stream("capture-X", "a", "Study")
    payload = checkpoint_identity_fields(data.identity_table, data)
    payload["source_study_membership"] = dataset_study_membership(data)
    with pytest.raises(ValueError, match="study membership.*training contract"):
        checkpoint_training_identity_ledger(payload)
