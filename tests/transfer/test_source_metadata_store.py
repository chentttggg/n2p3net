"""Complete source storage, independent ingress and target state isolation."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.study import dataset_study_membership
from research.contracts import semantic_sha256
from research.frozen_metadata import plain_metadata
from research.source_metadata_store import (
    SourceMetadataStore,
    _SourceMetadataAdmission,
    compact_source_semantics,
    expand_source_semantics,
)
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.calibration_state import FrozenTrialPredictor, _numeric_state_digest
from transfer.checkpoint import ScopedSourceCheckpointPreparation, load_checkpoint_payload


def _source(tmp_path):
    data = binary_dataset()
    path = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="a" * 64)
    payload = load_checkpoint_payload(path)
    payload["training_contract"]["optimizer"]["config"] = {"pos_weight": 8.}
    payload["training_contract"]["validation"]["original_complete_rows"] = list(range(30000))
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    preparation = ScopedSourceCheckpointPreparation()
    prepared = preparation.prepare(content, data, "p1")
    prepared.assert_library_exclusion(target_membership=dataset_study_membership(data))
    return data, content, preparation, prepared


def test_complete_shared_source_and_isolated_standalone_replay(tmp_path):
    data, content, owner, prepared = _source(tmp_path)
    predictor = FrozenTrialPredictor.from_source(prepared, "p1")
    store = SourceMetadataStore(tmp_path / "shared")
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    predictor.save(first, metadata_store=store)
    predictor.with_fit_binding({"full_receipts": ["unchanged"]}).save(second, metadata_store=store)
    assert len(list(store.directory.glob("*.pt"))) == 1
    assert next(store.directory.glob("*.pt")).read_bytes() == content
    assert first.stat().st_size < len(content) / 2
    restored = FrozenTrialPredictor.load(first, metadata_store=store)
    assert restored.digest() == predictor.digest()
    assert plain_metadata(restored.record) == plain_metadata(predictor.record)
    np.testing.assert_array_equal(restored.probability_scores(data.X)["posterior_logits"],
                                  predictor.probability_scores(data.X)["posterior_logits"])
    legacy = tmp_path / "legacy.pt"
    torch.save({"schema": "n2p3_frozen_trial_predictor/2", "record": plain_metadata(predictor.record),
                "state": predictor._state}, legacy)
    assert FrozenTrialPredictor.load(legacy).digest() == predictor.digest()
    standalone = tmp_path / "standalone.pt"
    restored.save(standalone)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    shutil.copyfile(standalone, isolated / "model.pt")
    assert store.directory.is_relative_to(tmp_path.resolve())
    shutil.rmtree(store.directory)
    independent = FrozenTrialPredictor.load(isolated / "model.pt")
    assert independent.digest() == predictor.digest()
    np.testing.assert_array_equal(independent.predict_logit(data.X), predictor.predict_logit(data.X))
    assert owner.record()["source_admissions"] == 1


def test_fresh_ingress_rejects_changed_blob_and_original_source_numeric_state(tmp_path):
    _, content, _, prepared = _source(tmp_path)
    predictor = FrozenTrialPredictor.from_source(prepared, "p1")
    store = SourceMetadataStore(tmp_path / "shared")
    path = tmp_path / "model.pt"
    predictor.save(path, metadata_store=store)
    FrozenTrialPredictor.load(path, metadata_store=store)
    blob = next(store.directory.glob("*.pt"))
    blob.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="bytes or size"):
        FrozenTrialPredictor.load(path, metadata_store=store)
    blob.write_bytes(content)
    payload = torch.load(path, weights_only=True)
    payload["state"]["trunk.classifier.weight"].add_(1)
    payload["record"]["class_calibration"]["fit_contract"]["frozen_state_digest"] = _numeric_state_digest(payload["state"])
    record = expand_source_semantics({"probability_semantics": payload["record"]["class_calibration"],
        "source_contract_reference": payload["record"]["class_calibration"]["fit_contract"]["training_contract"]},
        _SourceMetadataAdmission(store=store))
    payload["expanded_record_digest"] = semantic_sha256({**payload["record"],
        "class_calibration": record["probability_semantics"]})
    torch.save(payload, path)
    with pytest.raises(ValueError, match="numerical state differs"):
        FrozenTrialPredictor.load(path, metadata_store=store)


def test_report_pool_share_complete_authority_and_fresh_target_binding(tmp_path):
    data, content, owner, prepared = _source(tmp_path)
    predictor = FrozenTrialPredictor.from_source(prepared, "p1")
    store = SourceMetadataStore(tmp_path / "shared")
    store.publish(content)
    cell = {"probability_semantics": predictor.record["class_calibration"],
            "source_contract_reference": predictor.source_contract_reference}
    report = {"cells": [cell], "probability_pools": [{"expert_origins": [cell]}]}
    stored = compact_source_semantics(report)
    encoded = json.dumps(stored).encode()
    assert len(encoded) < len(json.dumps(plain_metadata(report)).encode()) / 5
    ingress = _SourceMetadataAdmission(store=store)
    restored = expand_source_semantics(json.loads(encoded), ingress)
    assert semantic_sha256(restored) == semantic_sha256(report)
    assert ingress.source_admissions == 1
    assert ingress.target_bindings == 2
    changed_storage = copy.deepcopy(stored)
    changed_storage["cells"][0]["probability_semantics"] = plain_metadata(predictor.record["class_calibration"])
    with pytest.raises(ValueError, match="lost its explicit storage dependency"):
        expand_source_semantics(changed_storage, _SourceMetadataAdmission(store=store))
    current = copy.deepcopy(data)
    current.identity_table = data.identity_table.relabel_local_subjects({"p1": "fresh-alias"})
    current.subject_ids = np.full(data.n_epochs, "fresh-alias")
    fresh = owner.prepare(content, current, "fresh-alias")
    fresh.assert_library_exclusion(target_membership=dataset_study_membership(current))
    a, b = prepared.trunk_for_target("p1"), fresh.trunk_for_target("fresh-alias")
    with torch.no_grad():
        for value in a.state_dict().values():
            value.add_(1)
    for name, value in b.state_dict().items():
        torch.testing.assert_close(value, fresh.trunk_for_target("fresh-alias").state_dict()[name], rtol=0, atol=0)
    prepared.input_statistics[0][:] = 999
    assert not np.array_equal(prepared.input_statistics[0], fresh.input_statistics[0])
    changed = replace(current, channel_names=list(reversed(current.channel_names)))
    with pytest.raises(ValueError, match="input domain"):
        owner.prepare(content, changed, "fresh-alias")
    assert owner.record()["source_admissions"] == 1


def test_source_reference_cannot_relabel_original_participant_or_drop_dependency(tmp_path):
    _, content, _, prepared = _source(tmp_path)
    predictor = FrozenTrialPredictor.from_source(prepared, "p1")
    store = SourceMetadataStore(tmp_path / "shared")
    store.publish(content)
    reference = plain_metadata(predictor.source_contract_reference)
    target = reference["target_binding"]
    source_identity = plain_metadata(prepared.payload["training_identity_ledger"])
    target["identity"] = source_identity
    target["subject"] = source_identity["records"][0]["local_subject_id"]
    target["study_membership"] = plain_metadata(prepared.payload["source_study_membership"])
    with pytest.raises(ValueError, match="exclude|includes target"):
        _SourceMetadataAdmission(store=store).resolve(reference)
    path = tmp_path / "model.pt"
    predictor.save(path, metadata_store=store)
    with pytest.raises(ValueError, match="requires its complete metadata store"):
        FrozenTrialPredictor.load(path)
    assert hashlib.sha256(next(store.directory.glob("*.pt")).read_bytes()).hexdigest() == predictor.source_contract_reference["checkpoint_sha256"]


def test_same_transaction_sibling_source_reference_requires_original_byte_size(tmp_path):
    _, content, _, prepared = _source(tmp_path)
    predictor = FrozenTrialPredictor.from_source(prepared, "p1")
    store = SourceMetadataStore(tmp_path / "shared")
    store.publish(content)
    original = plain_metadata(predictor.source_contract_reference)
    ingress = _SourceMetadataAdmission(store=store)
    ingress.resolve(original)
    changed = copy.deepcopy(original)
    changed["checkpoint_bytes"] += 1
    with pytest.raises(ValueError, match="bytes or size"):
        ingress.resolve(changed)
    assert ingress.source_admissions == 1


def test_public_report_source_schema_requires_reference_and_keeps_legacy_full_values(tmp_path):
    from experiments.run_erp_comparator import load_binary_report

    _, content, _, prepared = _source(tmp_path)
    predictor = FrozenTrialPredictor.from_source(prepared, "p1")
    store = SourceMetadataStore(tmp_path / "shared")
    store.publish(content)
    legacy = {"schema": "n2p3_erp_comparator_result/2", "cells": [{
        "probability_semantics": plain_metadata(predictor.record["class_calibration"])}]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_binary_report(path) == legacy
    changed = {**legacy, "schema": "n2p3_erp_comparator_result/3"}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit storage dependency"):
        load_binary_report(path, metadata_store=store)
    legal = {"schema": changed["schema"], "cells": [{
        "probability_semantics": predictor.record["class_calibration"],
        "source_contract_reference": predictor.source_contract_reference}]}
    path.write_text(json.dumps(compact_source_semantics(legal)), encoding="utf-8")
    assert semantic_sha256(load_binary_report(path, metadata_store=store)) == semantic_sha256(legal)


def test_serial_ingress_two_current_participants_share_only_source_prototype(tmp_path):
    from data.identity import DatasetIdentityTable
    from transfer.checkpoint import SourceCheckpointPreparation

    data, content, _, _ = _source(tmp_path)
    second = copy.deepcopy(data)
    second.subject_ids = np.full(data.n_epochs, "second")
    second.identity_table = DatasetIdentityTable.from_source_rows(second.subject_ids.tolist(), ["synthetic_p300"] * data.n_epochs)
    owner = SourceCheckpointPreparation()
    first = owner.prepare(content, data, "p1")
    fresh = owner.prepare(content, second, "second")
    first.assert_library_exclusion(target_membership=dataset_study_membership(data))
    fresh.assert_library_exclusion(target_membership=dataset_study_membership(second))
    assert owner.source_admissions == 1 and owner.target_bindings == 2
    a, b = first.trunk_for_target("p1"), fresh.trunk_for_target("second")
    with torch.no_grad():
        for value in a.state_dict().values():
            value.add_(1)
    first.input_statistics[0][:] = 999
    for name, value in b.state_dict().items():
        torch.testing.assert_close(value, owner.prepare(content, second, "second").trunk_for_target("second").state_dict()[name], rtol=0, atol=0)
    assert not np.array_equal(first.input_statistics[0], fresh.input_statistics[0])
    leaked = copy.deepcopy(second)
    source_identity = plain_metadata(first.payload["training_identity_ledger"])
    leaked.identity_table = DatasetIdentityTable.from_payload(source_identity)
    leaked.subject_ids = np.full(data.n_epochs, source_identity["records"][0]["local_subject_id"])
    with pytest.raises(ValueError, match="includes target"):
        owner.prepare(content, leaked, leaked.subject_ids[0])
    assert owner.source_admissions == 1


def test_binary_actual_serial_save_replay_parent_and_pool_share_source(tmp_path):
    from data.epochs import load_epoch_dataset
    from data.source_records import prepare_source_record_prefixes
    from experiments import run_erp_comparator as runner
    from research.cloud_execution import CloudResult
    from tests.support.raw_prefix import write_real_record

    item, spec, _ = write_real_record(tmp_path)
    prefix = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "prepared",
        preprocessing=spec, budgets=(0, 8), minimum_test_trials=2, prepare_inner=False)
    collection = tmp_path / "prefix.json"
    collection.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": prefix["preprocessing"], "preparation_method": prefix["preparation_method"],
        "records": [prefix]}), encoding="utf-8")
    data = load_epoch_dataset(prefix["calibration"]["8"]["cache"], validation="attested")
    checkpoint = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="b" * 64)
    payload = load_checkpoint_payload(checkpoint)
    payload["training_contract"]["optimizer"]["config"] = {"pos_weight": 8.}
    payload["training_contract"]["validation"]["original_complete_rows"] = list(range(30000))
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    torch.save(payload, checkpoint)
    catalog = [{"outer_target_study": "Huebner2017", "source_seed": 0, "path": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "procedure": {"kind": "fixed_recipe", "selection": "not_applicable"}}]
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["1"], "budgets": [0, 8],
        "arms": [{"name": family, "family": family, "prefix_collection": str(collection),
                  "epochs": 1, "batch_size": 32} for family in ("full_fine", "scratch", "waveform_lda")],
        "allowed_input_axes": [], "evidence_scope": "software_correctness", "seed_pairs": [[0, 19]],
        "source_seed_catalog": catalog,
        "contrasts": {"full_minus_scratch": {"full_fine": 1, "scratch": -1}},
        "probability_pools": [{"name": "source_only", "expert_ids": ["full_fine"], "weights": [1.]}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    resources = {"target_native_threads_per_worker": 1, "target_worker_memory_bytes": 2**28}
    # Build actual tasks; the serial adapter performs the real fitting and saved replay.
    from research.source_execution import resolve_source_resources

    resources = resolve_source_resources({"source_worker_memory_gib": .25, "target_worker_memory_gib": .25,
        "cpu_budget": 2, "target_workers": 1}, 1, "cpu")
    shared_directory = tmp_path / "shared-programme-sources"
    programme = runner.build_binary_cloud_tasks(path, tmp_path / "run", resources,
                                               metadata_directory=shared_directory)
    task = programme.tasks[0]
    returned = runner.comparison_participant_adapter(task.arguments)
    outcome = CloudResult(task.task_id, "completed", value=returned)
    programme.on_finished(outcome)
    report = programme.finalize({task.task_id: outcome})
    store = SourceMetadataStore(shared_directory)
    restored = runner.load_binary_report(programme.output / "result.json", metadata_store=store)
    assert restored["requested_cells"] == 6
    assert restored["physical_target_fits"] == 3
    assert len(list(store.directory.glob("*.pt"))) == 1
    assert restored["requested_budget_coverage"] == report["requested_budget_coverage"]
    assert semantic_sha256(restored) == semantic_sha256(report)
    assert all(cell["saved_replay_exact"] for cell in restored["cells"] if cell["status"] == "completed")
    stored = json.loads((programme.output / "result.json").read_bytes())
    zero = next(cell for cell in stored["cells"] if cell["family"] == "full_fine" and cell["K"] == 0)
    assert zero["probability_semantics"]["fit_contract"]["training_contract"]["schema"].endswith("reference/1")
    assert "original_complete_rows" not in (programme.output / "result.json").read_text()
    for cell in restored["cells"]:
        if cell["K"] > 0:
            payload = torch.load(programme.output / cell["model_artifact"]["path"], weights_only=True)
            assert payload["schema"] == "n2p3_frozen_trial_predictor/2"
            assert "source_metadata" not in payload
