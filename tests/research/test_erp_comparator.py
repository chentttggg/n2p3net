"""The executable E2 chain on independently filtered real BDF containers."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.brainsync import load_brainsync_sessions
from data.epochs import save_epoch_dataset
from data.source_records import prepare_source_record_prefixes
from experiments.fit_source_temporal_basis import fit_plan
from experiments.run_erp_comparator import run_comparison
from tests.support.brainsync import write_bids_session
from tests.support.raw_prefix import write_real_record


def test_raw_prefix_erp_bridge_runs_all_requested_arms_and_reloads(tmp_path):
    item, spec, _ = write_real_record(tmp_path)
    collections = {}
    for mode in ("mean_only", "none"):
        pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / mode,
            preprocessing=replace(spec, baseline_mode=mode), budgets=(0, 8), minimum_test_trials=2)
        path = tmp_path / (mode + ".json")
        path.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
            "preprocessing": pair["preprocessing"], "preparation_method": pair["preparation_method"],
            "records": [pair]}), encoding="utf-8")
        collections[mode] = str(path)
    arms = [{"name": family + "_" + mode, "prefix_collection": collections[mode],
             "family": family, "bin_width": 16} for family, mode in
            [("xdawn_erp", "mean_only"), ("pooled_lda", "mean_only"), ("pooled_lda", "none"), ("toeplitz_lda", "none"),
             ("waveform_lda", "mean_only"), ("waveform_lda", "none")]]
    source_session = write_bids_session(tmp_path / "source", session_id="source", target=3,
        started_utc="2026-09-05T08:00:00Z", blocks=(tuple(range(1, 10)) * 3,))
    source = load_brainsync_sessions((source_session,), preprocessing=replace(spec, baseline_mode="none"))
    cache = save_epoch_dataset(tmp_path / "source.npz", source)
    source_plan = tmp_path / "source-plan.json"
    source_plan.write_text(json.dumps({"schema": "n2p3_source_temporal_plan/1", "rank": 2,
        "excluded_target_studies": ["Huebner2017"], "sources": [{"cache": str(cache), "rows": "all"}]}), encoding="utf-8")
    basis_path = tmp_path / "source-basis.json"
    fit_plan(source_plan, basis_path)
    for kind in ("fixed_cosine", "source_supervised"):
        arms.append({"name": kind, "prefix_collection": collections["none"], "family": "temporal_lda",
            "bin_width": 1, "temporal_basis_kind": kind, "temporal_rank": 2,
            **({"source_temporal_basis": str(basis_path)} if kind == "source_supervised" else {})})
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["1"], "budgets": [8],
            "arms": arms, "allowed_input_axes": ["baseline"], "evidence_scope": "software_correctness"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    report = run_comparison(path, tmp_path / "result")
    assert report["requested_cells"] == 8
    assert all(cell["status"] == "completed" for cell in report["cells"]), report["cells"]
    assert all(cell["saved_replay_exact"] for cell in report["cells"])
    assert [cell["arm"] for cell in report["cells"] if cell["source_pretraining_used"]] == ["source_supervised"]
    assert len({cell["comparison_identity"]["score"]["evidence_sha256"] for cell in report["cells"]}) == 1
    plan["allowed_input_axes"] = []
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="materialized input"):
        run_comparison(path, tmp_path / "undeclared")
    assert not (tmp_path / "undeclared").exists()


def test_erp_runner_retains_unavailable_requested_budget_and_executes_legal_anchor(tmp_path, monkeypatch):
    import experiments.run_erp_comparator as runner
    from data.epochs import load_epoch_dataset
    from data.identity import DatasetIdentityTable
    from research.contracts import semantic_sha256
    from tests.support.checkpoints import write_checkpoint

    item, spec, _ = write_real_record(tmp_path)
    pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "prepared",
        preprocessing=spec, budgets=(0, 8, 200), minimum_test_trials=2)
    collection = tmp_path / "prefix.json"
    collection.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": pair["preprocessing"], "preparation_method": pair["preparation_method"],
        "records": [pair]}), encoding="utf-8")
    calibration = load_epoch_dataset(pair["calibration"]["8"]["cache"], validation="attested")
    checkpoint = write_checkpoint(tmp_path / "source.pt", calibration, source_snapshot_sha256="e"*64)
    arms = [{"name": family, "prefix_collection": str(collection), "family": family,
             **({"checkpoint": str(checkpoint), "epochs": 1, "batch_size": 32} if family in {"full_fine", "scratch"} else {})}
            for family in ("waveform_lda", "full_fine", "scratch")]
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["1"], "budgets": [8, 200],
        "arms": arms, "allowed_input_axes": [], "evidence_scope": "software_correctness",
        "target_seeds": [5], "contrasts": {"full_minus_scratch": {"full_fine": 1, "scratch": -1}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loader = runner.load_target_prefixes
    streamed = []
    def checked_loader(*args, **kwargs):
        assert kwargs["subjects"] == ["1"]
        streamed.append(True)
        return loader(*args, **kwargs)
    monkeypatch.setattr(runner, "load_target_prefixes", checked_loader)
    report = run_comparison(path, tmp_path / "result")
    assert len(streamed) == 6  # Three arms, one participant, preflight + execution.
    assert len(report["cells"]) == 6
    assert all(c["status"] == "completed" for c in report["cells"] if c["K"] == 8), report["cells"]
    assert all(c["status"] == "protocol_unavailable" for c in report["cells"] if c["K"] == 200)
    assert {c["source_seed"] for c in report["cells"]} == {0}
    assert {c["target_seed"] for c in report["cells"]} == {5}
    assert report["statistics"]["evidence_scope"] == "software_correctness"
    assert (tmp_path / "result" / "statistics.json").is_file()
    assert report["statistics"]["requested_budget_coverage"][1]["protocol_unavailable_cells"] == 3
    completed = {c["arm"]: c for c in report["cells"] if c["K"] == 8}
    assert completed["full_fine"]["source_pretraining_used"]
    assert not completed["scratch"]["source_pretraining_used"]
    assert completed["full_fine"]["checkpoint_origin"]["source_snapshot_sha256"] == "e"*64

    # Same-architecture scratch cannot inherit pretrained tensors or statistics.
    original = torch.load(checkpoint, weights_only=True)
    altered = torch.load(checkpoint, weights_only=True)
    for value in altered["trunk_state_dict"].values():
        if value.dtype.is_floating_point:
            value.add_(1.)
    altered["input_mean"] = [99.] * calibration.n_channels
    torch.save(altered, checkpoint)
    plan["arms"] = [arms[-1]]
    plan["budgets"] = [8]
    plan["contrasts"] = {}
    path.write_text(json.dumps(plan), encoding="utf-8")
    repeated = run_comparison(path, tmp_path / "scratch-repeated")
    assert repeated["cells"][0]["status"] == "completed"
    with np.load(tmp_path / "result" / "00004.npz") as before, np.load(tmp_path / "scratch-repeated" / "00000.npz") as after:
        np.testing.assert_array_equal(before["logits"], after["logits"])

    # A different person from the held-out library still violates study holdout.
    table = DatasetIdentityTable.from_source_rows(["other"], ["Huebner2017"])
    original["training_identity_ledger"] = table.payload()
    original["training_identity_ledger_digest"] = table.digest()
    original["training_contract"]["source_identity_digest"] = table.digest()
    original["training_contract"]["training_participant_keys"] = list(table.authority_keys())
    original["training_contract_digest"] = semantic_sha256(original["training_contract"])
    torch.save(original, checkpoint)
    with pytest.raises(ValueError, match="complete target study"):
        run_comparison(path, tmp_path / "leaking-study")
    assert not (tmp_path / "leaking-study").exists()


def test_comparator_requires_explicit_evidence_scope_and_frozen_public_execution(tmp_path):
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["1"], "budgets": [8],
            "arms": [{"name": "lda", "family": "waveform_lda"}], "allowed_input_axes": []}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="evidence_scope"):
        run_comparison(path, tmp_path / "missing")
    plan["evidence_scope"] = "public_development"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="executed source_snapshot"):
        run_comparison(path, tmp_path / "unfrozen")


@pytest.mark.parametrize("violation", ["missing_scratch", "missing_waveform", "budget", "architecture", "input", "bn_policy", "contrast_bypass"])
def test_nn_transfer_refuses_unmatched_controls_before_fitting(tmp_path, violation, monkeypatch):
    import experiments.run_erp_comparator as runner
    from data.epochs import load_epoch_dataset
    from models.n2p3net import N2P3Net
    from research.contracts import semantic_sha256
    from tests.support.checkpoints import write_checkpoint

    item, spec, _ = write_real_record(tmp_path)
    collection_paths = {}
    datasets = {}
    for mode in ("mean_only", "none"):
        pair = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / mode,
            preprocessing=replace(spec, baseline_mode=mode), budgets=(0, 8), minimum_test_trials=2)
        collection = tmp_path / f"{mode}.json"
        collection.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
            "preprocessing": pair["preprocessing"], "preparation_method": pair["preparation_method"],
            "records": [pair]}))
        collection_paths[mode] = str(collection)
        datasets[mode] = load_epoch_dataset(pair["calibration"]["8"]["cache"], validation="attested")
    checkpoint = write_checkpoint(tmp_path / "source.pt", datasets["mean_only"], source_snapshot_sha256="e"*64)
    full = {"name": "full", "family": "full_fine", "prefix_collection": collection_paths["mean_only"],
            "checkpoint": str(checkpoint), "epochs": 1, "batch_size": 32}
    scratch = {**full, "name": "scratch", "family": "scratch"}
    lda = {"name": "waveform", "family": "waveform_lda", "prefix_collection": collection_paths["mean_only"]}
    if violation == "budget":
        scratch.update(epochs=2, batch_size=4, lr=.02)
    if violation == "bn_policy":
        scratch.update(bn_buffers="source")
    if violation == "architecture":
        payload = torch.load(checkpoint, weights_only=True)
        data = datasets["mean_only"]
        trunk = N2P3Net(data.n_channels, n_times=data.n_times, sfreq=data.preprocessing.sfreq,
            tmin_s=data.preprocessing.tmin_ms/1000, pooling_mode="full_unfold", temporal_kernel_size=17)
        payload["trunk_state_dict"] = trunk.state_dict()
        payload["architecture"] = trunk.architecture_record()
        payload["training_contract"]["architecture"] = trunk.architecture_record()
        payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
        other = tmp_path / "different-architecture.pt"
        torch.save(payload, other)
        scratch["checkpoint"] = str(other)
    if violation == "input":
        other = write_checkpoint(tmp_path / "different-input.pt", datasets["none"], source_snapshot_sha256="e"*64)
        scratch.update(checkpoint=str(other), prefix_collection=collection_paths["none"])
    arms = [full] + ([] if violation == "missing_scratch" else [scratch]) + ([] if violation == "missing_waveform" else [lda])
    plan = {"schema": "n2p3_erp_comparator_plan/1", "subjects": ["1"], "budgets": [8],
        "arms": arms, "allowed_input_axes": ["baseline"], "evidence_scope": "software_correctness"}
    if violation == "contrast_bypass":
        plan["arms"].append({**scratch, "name": "unmatched_scratch", "epochs": 2})
        plan["contrasts"] = {"invalid_transfer_claim": {"full": 1, "unmatched_scratch": -1}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    def forbidden_fit(*args, **kwargs):
        pytest.fail("unmatched comparison reached model fitting")
    monkeypatch.setattr(runner, "fit_calibration_candidate", forbidden_fit)
    with pytest.raises(ValueError, match="matched.*control"):
        run_comparison(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
