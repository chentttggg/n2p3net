"""Standalone source closure includes sealed transport-only joint packages."""
from __future__ import annotations

import hashlib
import json

import pytest

from data.study import dataset_study_membership
from research.contracts import semantic_sha256
from research.execution import atomic_write_json
from research.frozen_metadata import plain_metadata
from research.source_metadata_store import (
    SourceMetadataStore,
    admit_source_dependencies,
    collect_source_dependencies,
    export_source_package_closure,
    source_dependency_descriptor,
)
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.checkpoint import ScopedSourceCheckpointPreparation


def _joint_transport(tmp_path):
    data = binary_dataset()
    source = write_checkpoint(tmp_path / "original.pt", data, source_snapshot_sha256="a" * 64)
    content = source.read_bytes()
    prepared = ScopedSourceCheckpointPreparation().prepare(content, data, "p1")
    prepared.assert_library_exclusion(target_membership=dataset_study_membership(data))
    reference = plain_metadata(prepared.source_storage_context("p1")[1])
    store = SourceMetadataStore(tmp_path / "shared-source")
    store.publish(content)
    package = tmp_path / "prefix-only"
    descriptor = source_dependency_descriptor(package, store, (reference,))
    record = {"schema": "n2p3_joint_flat_prefix_inner_replay/1", "source_metadata": descriptor,
        "original_source_files": {"NN": {"schema": "n2p3_joint_source_checkpoint_reference/1",
            "family": "trial", "role": "architecture_only", "reference": reference}}}
    atomic_write_json(package / "prefix_inner_replay.json", {"record": record,
        "record_sha256": semantic_sha256(record)})
    return package, descriptor, reference, content


def test_joint_transport_only_standalone_export_recovers_original_source_without_original_store(tmp_path, monkeypatch):
    package, descriptor, reference, content = _joint_transport(tmp_path)
    original_seal = (package / "prefix_inner_replay.json").read_bytes()
    destination = tmp_path / "standalone"
    restored = export_source_package_closure(package, destination)
    assert (restored / "prefix_inner_replay.json").read_bytes() == original_seal
    copies = tuple(destination.rglob(reference["checkpoint_sha256"] + ".pt"))
    assert len(copies) == 1 and copies[0].read_bytes() == content
    original_read = SourceMetadataStore.read

    def isolated_read(self, current):
        assert self.directory.is_relative_to(destination.resolve()), "export still depends on the original store"
        return original_read(self, current)

    monkeypatch.setattr(SourceMetadataStore, "read", isolated_read)
    admission = admit_source_dependencies(descriptor, restored)
    assert admission.content(reference) == content
    assert hashlib.sha256(admission.content(reference)).hexdigest() == reference["checkpoint_sha256"]


def test_joint_prefix_collection_is_explicit_and_requires_original_prefix_seal(tmp_path):
    package, descriptor, _, _ = _joint_transport(tmp_path)
    assert collect_source_dependencies(package, (package,)) is None
    assert collect_source_dependencies(package, (package,), include_joint_prefixes=True) == descriptor
    path = package / "prefix_inner_replay.json"
    payload = json.loads(path.read_bytes())
    payload["record"]["source_metadata"]["references"][0]["checkpoint_bytes"] += 1
    path.write_text(json.dumps(payload), encoding="utf8")
    assert collect_source_dependencies(package, (package,)) is None
    with pytest.raises(ValueError, match="sealed"):
        collect_source_dependencies(package, (package,), include_joint_prefixes=True)


def test_joint_prefix_union_rejects_another_store_without_self_admitting_prefix_descriptor(tmp_path):
    package, descriptor, reference, _ = _joint_transport(tmp_path)
    numerical = {"schema": "n2p3_fitted_candidate_ensemble/4", "source_metadata": descriptor}
    atomic_write_json(package / "ensemble.json", {"record": numerical, "record_sha256": semantic_sha256(numerical)})
    assert collect_source_dependencies(package, (package,)) == descriptor
    path = package / "prefix_inner_replay.json"
    payload = json.loads(path.read_bytes())
    foreign = source_dependency_descriptor(package, SourceMetadataStore(tmp_path / "foreign-store"), (reference,))
    payload["record"]["source_metadata"] = foreign
    payload["record_sha256"] = semantic_sha256(payload["record"])
    path.write_text(json.dumps(payload), encoding="utf8")
    assert collect_source_dependencies(package, (package,)) == descriptor
    with pytest.raises(ValueError, match="one actual complete source store"):
        collect_source_dependencies(package, (package,), include_joint_prefixes=True)
