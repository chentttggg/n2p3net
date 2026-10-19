"""Persistence and replay tests for B03's inner nested producers."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from research.contracts import semantic_sha256
from research.nested_selection_replay import (
    InnerReplayRecorder,
    persist_inner_replay,
    replay_inner_replay,
)
from tests.research.test_calibration_selection import candidates, groups


@pytest.fixture
def persisted_case(tmp_path):
    from research.calibration_selection import fit_nested_calibration_selection

    calibration = groups(3)
    recorder = InnerReplayRecorder()
    result = fit_nested_calibration_selection(calibration, candidates(), baseline_name="fixed",
        inner_replay_recorder=recorder)
    directory = persist_inner_replay(result, calibration, candidates(), tmp_path / "inner",
        recorder=recorder)
    return calibration, result, directory


def test_inner_replay_persists_full_numerical_packages_and_decision(persisted_case):
    calibration, result, directory = persisted_case
    payload = json.loads((directory / "inner_replay.json").read_bytes())
    record = payload["record"]
    assert payload["record_sha256"] == semantic_sha256(record)
    assert record["schema"] == "n2p3_inner_model_replay/1"
    assert record["inner_models_persisted"] is True
    assert record["outer_scores_used_for_final_selection"] is False
    assert semantic_sha256(record["selection_decision"]) == semantic_sha256(result.record["final_selection"]["decision"])
    assert semantic_sha256(record["requested_denominator"]) == semantic_sha256(result.record["outer_requested"])
    assert len(record["group_inputs"]) == len(calibration)
    assert record["producers"]
    for producer in record["producers"]:
        assert producer["training_scope"]["group_records"]
        assert producer["input_identities"]
        assert "source_checkpoint_roles" in producer
        if producer["status"] == "completed":
            package = directory / producer["package"]
            assert (package / "ensemble.json").is_file()
            assert producer["package_sha256"]
            ensemble = json.loads((package / "ensemble.json").read_bytes())
            assert ensemble["record"]["experts"]
            if ensemble["record"].get("OOF_probability") is not None:
                assert ensemble["record"]["OOF_probability"]["rows"]
        else:
            assert producer["failure"]


def test_inner_persistence_without_same_run_capture_is_rejected(tmp_path):
    from research.calibration_selection import fit_nested_calibration_selection

    calibration = groups(3)
    result = fit_nested_calibration_selection(calibration, candidates(), baseline_name="fixed")
    with pytest.raises(ValueError, match="recorder|original fit run"):
        persist_inner_replay(result, calibration, candidates(), tmp_path / "uncaptured")


def test_inner_replay_roundtrip_rechecks_producers_and_boundaries(persisted_case):
    calibration, _, directory = persisted_case
    replay = replay_inner_replay(directory, calibration=calibration, candidates=candidates())
    assert replay["status"] == "replayed"
    assert replay["boundary_violations"] == []
    assert replay["producer_count"] > 0
    assert replay["exact_score_checks"] > 0


def test_inner_replay_rejects_label_mutation(persisted_case):
    calibration, _, directory = persisted_case
    changed = replace(calibration[0], labels=~calibration[0].labels, target_candidate="b")
    with pytest.raises(ValueError, match="label|input|identity|scope"):
        replay_inner_replay(directory, calibration=(changed, *calibration[1:]), candidates=candidates())


def test_inner_replay_rejects_sample_mutation_even_if_raw_digest_is_stale(persisted_case):
    calibration, _, directory = persisted_case
    view = replace(calibration[0].view, samples=np.asarray(calibration[0].view.samples) * 1.001)
    changed = replace(calibration[0], view=view)
    with pytest.raises(ValueError, match="sample|input|producer|digest|scope"):
        replay_inner_replay(directory, calibration=(changed, *calibration[1:]), candidates=candidates())


def test_inner_replay_rejects_scope_mutation(persisted_case):
    calibration, _, directory = persisted_case
    view = replace(calibration[0].view, raw_scope=(100, 651))
    changed = replace(calibration[0], view=view)
    with pytest.raises(ValueError, match="scope|input|identity"):
        replay_inner_replay(directory, calibration=(changed, *calibration[1:]), candidates=candidates())


def test_inner_replay_rejects_manifest_mutation(persisted_case):
    _, _, directory = persisted_case
    manifest = directory / "inner_replay.json"
    original = manifest.read_bytes()
    payload = json.loads(original)
    payload["record"]["producers"][0]["fit_key"] = "f" * 64
    payload["record_sha256"] = semantic_sha256(payload["record"])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|digest|producer"):
        replay_inner_replay(directory)
    manifest.write_bytes(original)


def test_inner_replay_records_actual_grouped_temperature_oof_rows(tmp_path):
    from research.calibration_selection import fit_nested_calibration_selection

    calibration = groups(4)
    recorder = InnerReplayRecorder()
    result = fit_nested_calibration_selection(calibration, candidates(), baseline_name="fixed",
        inner_replay_recorder=recorder)
    directory = persist_inner_replay(result, calibration, candidates(), tmp_path / "oof",
        recorder=recorder)
    payload = json.loads((directory / "inner_replay.json").read_bytes())["record"]
    grouped = [producer for producer in payload["producers"] if producer["oof_rows"]]
    assert grouped
    assert all(row["training_record_ids"] for producer in grouped for row in producer["oof_rows"])
    assert all(row["evidence"] for producer in grouped for row in producer["oof_rows"])
    replay = replay_inner_replay(directory, calibration=calibration, candidates=candidates())
    assert replay["exact_score_checks"] > 0
