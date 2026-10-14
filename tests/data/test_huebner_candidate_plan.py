"""Original public builder and metadata manifest ownership; no model loading."""
import hashlib
import json
from pathlib import Path

from data.huebner2017_preparation import (
    huebner_candidate_plan,
    write_huebner_original_manifest_declarations,
)
from transfer.ensemble_fit import CalibrationExpertProgram


def test_quartet_builder_declares_exact_waveform_control_and_keeps_full_selector_roster():
    source = {"outer_target_study": "Huebner2017", "source_seed": 20261014,
              "path": "original.pt", "sha256": "a"*64, "procedure": {"kind": "selection_refit"}}
    plan = huebner_candidate_plan([{"dataset_id": "Huebner2017", "path": "source.json", "sha256": "b"*64}],
        source, budgets=(0, 200, 500), nn_epochs=1, evidence_scope="software_correctness")
    experts = {record["expert_id"]: CalibrationExpertProgram.from_record(record) for record in plan["experts"]}
    assert experts["waveform"].recipe.initialization == "waveform_lda"
    assert experts["waveform"].profile.digest == experts["NN"].profile.digest
    assert experts["NN"].recipe.epochs == experts["scratch"].recipe.epochs == 1
    assert len(plan["mixtures"]) == 5
    assert [candidate["name"] for candidate in plan["prefix_selection"]["candidates"]] == [
        "source_nn_fixed", "scratch_nn_fixed", "source_nn_xdawn_fixed", "scratch_nn_xdawn_fixed"]


def test_manifest_bytes_remain_bound_to_original_rows_across_later_file_change(tmp_path, monkeypatch):
    envelope = {"dataset_id": "Huebner2017", "source_format": "original_vmrk_plus_NEMAR_BDF", "source_subject_id": "1",
        "record_sources": [{"subject": "1", "block": 2, "run": 0, "bdf": None, "marker": {"sha256": "a"*64}}]}
    participant = tmp_path/"subject01.json"
    participant.write_text(json.dumps(envelope), encoding="utf-8")
    manifest = {"schema": "n2p3_huebner_participant_envelope_manifest/1", "records": [
        {"path": str(participant), "sha256": hashlib.sha256(participant.read_bytes()).hexdigest()}]}
    source = tmp_path/"manifest.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original_read = Path.read_bytes
    def read_bytes(path):
        payload = original_read(path)
        if path == participant:
            source.write_text(json.dumps({**manifest, "changed_after_admission": True}), encoding="utf-8")
        return payload
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    roster = write_huebner_original_manifest_declarations(source, tmp_path/"output")
    assert roster["original_metadata_manifest"]["sha256"] == original_sha
    assert hashlib.sha256(original_read(source)).hexdigest() != original_sha
