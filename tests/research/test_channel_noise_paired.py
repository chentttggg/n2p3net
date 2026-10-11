"""Actual producer and CLI tests for frozen-response paired noise controls."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.linalg import solve_triangular

from data.candidate_task import MembershipDecision
from research.channel_noise_mechanism import FrozenERPResponse, NoiseGeometry
from research.contracts import DecisionPlanContract, DecisionPlanEntry, ParticipantSelectionFailure
from research.execution import UnidentifiableFitError
from tests.support.bnci import bnci_mat_fields
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel, _event_design
from transfer.ensemble_views import (
    CalibrationSelection,
    PredictionDecisionView,
    prepare_bnci_calibration_groups,
)
from transfer.noise_model import array_digest


@pytest.fixture
def actual_case(tmp_path):
    rng = np.random.default_rng(20260918)
    basis = np.array([[1., 0.], [.5, 1.], [0., .25]])
    mean = np.array([[.4, -.2], [.15, .1], [.5, .3], [-.1, .2]])
    groups = []
    for index in range(4):
        decision = MembershipDecision.direct(["a", "b", "a", "b"], ("a", "b"),
            event_ids=tuple(f"group{index}/event{j}" for j in range(4)),
            repetition_indices=np.array([0, 0, 1, 1]), repetitions=2)
        onsets = np.array([8, 24, 48, 64])
        labels = np.array([True, False, True, False])
        design = _event_design(onsets, labels, 96, basis)
        view = PredictionDecisionView(f"record{index}", "synthetic/p", f"decision{index}",
            design @ mean + .2 * rng.normal(size=(96, 2)), onsets, decision, "a" * 64,
            (index * 128, index * 128 + 96), 128., ("Cz", "Pz"), ("EEG", "EEG"),
            "declared native", "software.npz", "synthetic", "p", index * 128 + 96)
        groups.append(CalibrationSelection(view, labels, "a"))
    records = [CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels)
               for group in groups[:3]]
    model = ContinuousDecisionModel.fit(records, sfreq=128., channels=("Cz", "Pz"),
        candidate_ids=("a", "b"), operator_identity="native_unfiltered_volts/1", basis=basis,
        fit_scope="target_calibration", ridge=1., noise_order=8, conditioning_samples=32)
    path = tmp_path / "actual_model.npz"
    model.save(path)
    geometry = NoiseGeometry(("Cz", "Pz"), 128., "V", "declared native", "native_unfiltered_volts/1", 2)
    return groups, model, path, geometry


def records_of(groups):
    return [{"record_id": g.view.record_id, "participant_key": g.view.participant_key,
             "recording_id": g.view.source_sha256, "read_support": list(g.view.raw_scope),
             "fit_scope": "target_calibration", "source_sha256": g.view.source_sha256,
             "input_sha256": array_digest(g.view.samples, g.view.onsets, g.labels),
             "n_samples": len(g.view.samples), "original_view": g.view.record()} for g in groups]


def builder_module():
    name = "experiments.build_channel_noise_audit_manifest"
    assert importlib.util.find_spec(name) is not None, "paired physical input builder is missing"
    return __import__(name, fromlist=["build_paired_noise_input"])


def test_modern_response_uses_saved_r_and_original_physical_covariance(actual_case):
    groups, model, path, geometry = actual_case
    convert = getattr(FrozenERPResponse, "from_continuous_model", None)
    assert callable(convert), "modern frozen producer conversion is missing"
    response = convert(ContinuousDecisionModel.load(path), geometry=geometry,
        source_model_sha256="b" * 64, response_id="fixed", training_records=records_of(groups[:3]))
    R = model.posterior_state["precision_root"]
    factor = np.kron(solve_triangular(R, np.eye(4)), np.linalg.cholesky(model.innovation_covariance))
    np.testing.assert_array_equal(response.mean, model.kernel_mean)
    np.testing.assert_array_equal(response.coefficient_factor, factor)
    wrong = np.kron(solve_triangular(R.T, np.eye(4), lower=True), np.linalg.cholesky(model.innovation_covariance))
    assert np.linalg.norm(factor @ factor.T - wrong @ wrong.T) > .001


def test_modern_response_checks_complete_actual_fit_records(actual_case):
    groups, model, _, geometry = actual_case
    convert = getattr(FrozenERPResponse, "from_continuous_model", None)
    assert callable(convert), "modern frozen producer conversion is missing"
    for change in ("missing_record", "input", "support"):
        records = records_of(groups[:3])
        if change == "missing_record":
            records.pop()
        elif change == "input":
            records[0]["input_sha256"] = "c" * 64
        else:
            records[0]["read_support"] = [1, 97]
        with pytest.raises(ValueError):
            convert(model, geometry=geometry, source_model_sha256="b" * 64,
                    response_id="fixed", training_records=records)


def make_manifest(actual_case, tmp_path):
    groups, _, model_path, _ = actual_case
    builder = builder_module()
    return builder.build_paired_noise_input(
        [{"model": model_path, "training": groups[:3], "held": [groups[3].view]}],
        truth={"synthetic/p": {"decision3": "a"}}, output=tmp_path / "input",
        evidence_scope="software_correctness", seed=20260918)


def test_actual_paired_cli_binds_physical_views_and_common_metrics(actual_case, tmp_path):
    path = make_manifest(actual_case, tmp_path)
    script = Path(__file__).resolve().parents[2] / "experiments/audit_channel_noise_mechanism.py"
    output = tmp_path / "run"
    process = subprocess.run([sys.executable, str(script), "--input-manifest", str(path), "--output", str(output)],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((output / "report.json").read_bytes())
    assert result["status"] == "completed" and len(result["cells"]) == 2
    assert result["winner_program"] == "CandidateEvidence_normalized_log_probability/1"
    assert result["metrics"]["shared8"]["probability_coverage"] == 1
    assert result["metrics"]["private8"]["probability_coverage"] == 1
    assert result["contrasts"]["private8_minus_shared8"]["paired_probability_count"] == 1
    assert all(cell["evidence"]["expected_event_counts"] == [2, 2] for cell in result["cells"])
    assert all(cell["saved_noise_score_replay_exact"] for cell in result["cells"])


def test_expected_private_failure_keeps_requested_probability_ledger(actual_case, tmp_path, monkeypatch):
    from experiments.audit_channel_noise_mechanism import audit
    from research.channel_noise_mechanism import FittedChannelNoise

    path = make_manifest(actual_case, tmp_path)
    original = FittedChannelNoise.score

    def score(self, *args, **kwargs):
        if self.family == "private":
            raise UnidentifiableFitError("declared private score failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FittedChannelNoise, "score", score)
    result = audit(path, tmp_path / "failure")
    assert len(result["cells"]) == 2 and result["metrics"]["private8"]["probability_coverage"] == 0
    contrast = result["contrasts"]["private8_minus_shared8"]
    assert len(contrast["missing_probability_ledger"]) == 1
    assert not contrast["development_gate_passed"] and contrast["lost_probability_count"] == 1


def test_unresolved_requested_participant_is_not_dropped(actual_case, tmp_path):
    groups, _, model_path, _ = actual_case
    path = builder_module().build_paired_noise_input(
        [{"model": model_path, "training": groups[:3], "held": [groups[3].view]}],
        truth={"synthetic/p": {"decision3": "a"}, "synthetic/missing": {}},
        selection_failures=(ParticipantSelectionFailure("synthetic/missing", "decision_selection", "source unavailable"),),
        output=tmp_path / "input", evidence_scope="software_correctness")
    from experiments.audit_channel_noise_mechanism import audit
    result = audit(path, tmp_path / "unresolved")
    assert result["metrics"]["shared8"]["requested_participant_count"] == 2
    assert not result["contrasts"]["private8_minus_shared8"]["development_gate_passed"]
    assert "paired_nll_unavailable" in result["contrasts"]["private8_minus_shared8"]["gate_reasons"]


def test_new_paired_winner_preserves_near_tie_probability_but_names_changed_program():
    from experiments.audit_channel_noise_mechanism import paired_candidate_evidence
    from models.candidate_evidence import CandidateExpertIdentity
    from research.channel_noise_mechanism import FittedChannelNoise

    geometry = NoiseGeometry(("Cz",), 128., "V", "native", "native_unfiltered_volts/1", 1)
    noise = FittedChannelNoise.from_parameters(np.empty((1, 0)), np.ones((1, 1)),
        geometry=geometry, conditioning_samples=2, family="shared", reference_id="analytic_white")
    response = FrozenERPResponse(np.array([[1.], [0.]]), np.empty((2, 0)), geometry,
        "a" * 64, "known_mean", {"fit_scope": "analytic_reference"})
    pulse = np.array([0., 0., 1., -1., -1., 1.])
    designs = np.zeros((2, 6, 2))
    designs[1, :, 0] = pulse
    score = noise.score((.5 * pulse + 1.25e-11 * pulse)[:, None], designs, response,
        candidate_ids=("a", "b"), geometry=geometry, record_id="held", recording_id="raw", sample_start=0)
    identity = CandidateExpertIdentity("test", "a" * 64, "b" * 64, None, "c" * 64, "conditional_log_probability", False)
    evidence = paired_candidate_evidence(score, candidate_vocabulary=("a", "b"), participant_key="p",
        decision_id="d", physical_support_digest="d" * 64, expert_identities=(identity,),
        evidence_level=2, expected_event_counts=(2, 2), observed_event_counts=(2, 2))
    assert score["winner"] is None and evidence.predicted == "b"
    np.testing.assert_array_equal(evidence.log_probability, score["log_probability"])


@pytest.fixture(scope="module")
def public_owner_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual_public_owner")
    participants, truth, sources = [], {}, []
    owner = {"authorized_labels": 400, "group_count": 3, "response_seconds": 1 / 256,
             "pre_roll_seconds": 0.}
    for index in range(2):
        fields = bnci_mat_fields()
        fields["X"] = np.random.default_rng(124 + index).normal(0., 4., fields["X"].shape).astype(np.float32)
        path = root / f"A0{index + 1}.mat"
        savemat(path, {"data": fields})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = prepare_bnci_calibration_groups(path, source_sha256=digest, **owner).groups
        model = fit_public_response(groups[:2])
        model_path = root / f"model{index}.npz"
        model.save(model_path)
        person = groups[0].view.participant_key
        participants.append({"model": model_path, "training": groups[:2], "held": [groups[2].view]})
        truth[person] = {groups[2].view.decision_id: groups[2].target_candidate}
        sources.append({"source": {"path": str(path), "sha256": digest}, "owner": owner})
    return participants, truth, sources


def fit_public_response(groups):
    return ContinuousDecisionModel.fit(
        [CalibrationRecord(g.view.record_id, g.view.samples, g.view.onsets, g.labels) for g in groups],
        sfreq=256., channels=groups[0].view.channel_names,
        candidate_ids=groups[0].view.decision.candidate_vocabulary, operator_identity="native_unfiltered_volts/1",
        basis=np.ones((1, 1)), fit_scope="target_calibration", ridge=1., noise_order=8, conditioning_samples=32)


def no_noise_fit(monkeypatch):
    from research.channel_noise_mechanism import FittedChannelNoise
    calls = []

    def fit(cls, *args, **kwargs):
        calls.append(True)
        raise UnidentifiableFitError("software probe stops after physical/producer preflight")

    monkeypatch.setattr(FittedChannelNoise, "fit", classmethod(fit))
    return calls


def test_public_builder_cannot_drop_original_source_participant(public_owner_case, tmp_path):
    participants, truth, sources = public_owner_case
    first = participants[0]["training"][0].view.participant_key
    with pytest.raises(ValueError, match="cohort|participant"):
        builder_module().build_paired_noise_input(participants[:1], truth={first: truth[first]},
            source_records=sources, output=tmp_path / "omitted", evidence_scope="public_calibration_development")


def test_public_audit_rejects_jointly_resealed_source_roster_omission(public_owner_case, tmp_path, monkeypatch):
    from experiments.audit_channel_noise_mechanism import audit, paired_input_digests
    participants, truth, sources = public_owner_case
    path = builder_module().build_paired_noise_input(participants, truth=truth, source_records=sources,
        output=tmp_path / "input", evidence_scope="public_calibration_development")
    payload = json.loads(path.read_bytes())
    payload["participants"] = payload["participants"][:1]
    first = participants[0]["training"][0].view.participant_key
    views = [g.view for g in participants[0]["training"]] + participants[0]["held"]
    cache, identity = paired_input_digests(views)
    payload["decision_plan"] = DecisionPlanContract(cache, identity, (first,),
        (DecisionPlanEntry(first, views[-1].decision_id, truth[first][views[-1].decision_id]),)).record()
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls = no_noise_fit(monkeypatch)
    with pytest.raises(ValueError, match="cohort|participant"):
        audit(path, tmp_path / "omitted")
    assert not calls


def test_wrong_legal_c_column_and_real_refitted_model_do_not_authorize_public_labels(public_owner_case, tmp_path, monkeypatch):
    participants, truth, sources = public_owner_case
    wrong = []
    for group in participants[0]["training"]:
        vocabulary = group.view.decision.candidate_vocabulary
        candidate = next(value for value in vocabulary if value != group.target_candidate)
        labels = group.view.decision.membership[:, vocabulary.index(candidate)]
        wrong.append(replace(group, labels=labels, target_candidate=candidate))
    wrong_path = tmp_path / "real_wrong_label_model.npz"
    fit_public_response(wrong).save(wrong_path)
    rows = [{**participants[0], "model": wrong_path, "training": wrong}, participants[1]]
    calls = no_noise_fit(monkeypatch)
    with pytest.raises(ValueError, match="label|target|source"):
        path = builder_module().build_paired_noise_input(rows, truth=truth, source_records=sources,
            output=tmp_path / "wrong", evidence_scope="public_calibration_development")
        from experiments.audit_channel_noise_mechanism import audit
        audit(path, tmp_path / "wrong-result")
    assert not calls


def test_public_audit_checks_actual_c_labels_after_joint_model_and_input_reseal(public_owner_case, tmp_path, monkeypatch):
    from experiments.audit_channel_noise_mechanism import audit
    participants, truth, sources = public_owner_case
    path = builder_module().build_paired_noise_input(participants, truth=truth, source_records=sources,
        output=tmp_path / "honest", evidence_scope="public_calibration_development")
    payload = json.loads(path.read_bytes())
    wrong = []
    for group in participants[0]["training"]:
        vocabulary = group.view.decision.candidate_vocabulary
        candidate = next(value for value in vocabulary if value != group.target_candidate)
        wrong.append(replace(group, labels=group.view.decision.membership[:, vocabulary.index(candidate)],
                             target_candidate=candidate))
    wrong_model_path = tmp_path / "wrong_actual_fit.npz"
    fit_public_response(wrong).save(wrong_model_path)
    descriptor = payload["participants"][0]["response"]["source_model"]
    descriptor.update(path=str(wrong_model_path), sha256=hashlib.sha256(wrong_model_path.read_bytes()).hexdigest())
    for row, group in zip(payload["participants"][0]["training"], wrong, strict=True):
        arrays_path = path.parent / row["arrays"]["path"]
        with np.load(arrays_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["labels"] = group.labels
        arrays["design"] = _event_design(group.view.onsets, group.labels, len(group.view.samples), np.ones((1, 1)))
        np.savez_compressed(arrays_path, **arrays)
        row["arrays"]["sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        row["target_candidate"] = group.target_candidate
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls = no_noise_fit(monkeypatch)
    with pytest.raises(ValueError, match="C labels|source authorization"):
        audit(path, tmp_path / "wrong-result")
    assert not calls


def test_honest_public_source_cohort_can_partition_a_selection_failure(public_owner_case, tmp_path, monkeypatch):
    from experiments.audit_channel_noise_mechanism import audit
    participants, truth, sources = public_owner_case
    first = participants[0]["training"][0].view.participant_key
    failed = participants[1]["training"][0].view.participant_key
    path = builder_module().build_paired_noise_input(participants[:1], truth={first: truth[first], failed: {}},
        source_records=sources, selection_failures=(ParticipantSelectionFailure(failed, "decision_selection", "model unavailable"),),
        output=tmp_path / "partition", evidence_scope="public_calibration_development")
    calls = no_noise_fit(monkeypatch)
    result = audit(path, tmp_path / "partition-result")
    assert len(calls) == 2 and result["metrics"]["shared8"]["requested_participant_count"] == 2
    assert result["metrics"]["shared8"]["probability_coverage"] == 0


@pytest.mark.parametrize("change", ["labels", "onsets", "physical_support", "requests", "missing_R"])
def test_coherent_paired_manifest_tampering_is_rejected(actual_case, tmp_path, change):
    path = make_manifest(actual_case, tmp_path)
    payload = json.loads(path.read_bytes())
    if change == "requests":
        payload["decision_plan"]["entries"] = []
    elif change == "missing_R":
        descriptor = payload["participants"][0]["response"]["source_model"]
        model_path = Path(descriptor["path"])
        with np.load(model_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files if key != "posterior_precision_root"}
        np.savez(model_path, **arrays)
        import hashlib
        descriptor["sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    else:
        participant = payload["participants"][0]
        row = participant["training"][0] if change == "labels" else participant["held"][0]
        if change == "labels":
            arrays_path = path.parent / row["arrays"]["path"]
            with np.load(arrays_path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            arrays["labels"] = ~arrays["labels"]
            np.savez(arrays_path, **arrays)
            import hashlib
            row["arrays"]["sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        else:
            receipt = row["view"]
            receipt["onsets"][0] += 1
            if change == "physical_support":
                receipt["raw_scope"] = [receipt["raw_scope"][0] + 1, receipt["raw_scope"][1] + 1]
                receipt["available_sample"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    from experiments.audit_channel_noise_mechanism import audit
    with pytest.raises(ValueError):
        audit(path, tmp_path / "bad_run")
