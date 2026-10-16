"""Genuine original-prefix fit, captured scoring and complete posterior replay."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from research.contracts import semantic_sha256
from tests.research.test_joint_flat_history_score_cache import small_history_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    fit_calibration_prefix_programs,
    validate_beta_uniform_oof,
)
from transfer.ensemble_views import RecordCalibrationPrefixCompiler
from transfer.joint_flat_model import JointFlatModel
from transfer.pool_calibration import FORWARD_BETA_STRATEGY


@pytest.fixture(scope="module")
def original_beta_calls(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual-forward-beta")
    with cpu_thread_budget(1):
        base, _, _, original, _ = small_history_case(directory)
        compiler = RecordCalibrationPrefixCompiler(original.adapter, budgets=(0, 2), profiles=(),
            authorization_preprocessing=replace(original.authorization_preprocessing,
                tmin_ms=0., tmax_ms=31.25, baseline_mode="none"),
            response_seconds=.03125, pre_roll_seconds=.0078125)
        spec = CalibrationExpertProgram("joint", "joint_flat", joint_flat=base.programme)
        program = EnsembleProgram.equal_weights("small-beta", (spec.expert_id,))
        fits, histories, results = {}, {}, {}
        for budget in (0, 2):
            prefix = compiler.compile(budget)
            programs = fit_calibration_prefix_programs(prefix, (spec,), program,
                strategies=("fixed", FORWARD_BETA_STRATEGY), source_checkpoints={spec.expert_id: base.source},
                beta_uniform_prior={"a": 1, "b": 1}, prefix_compiler=compiler,
                _prefix_fit_cache=fits, _joint_history_captures=histories)
            assert isinstance(programs[FORWARD_BETA_STRATEGY], FittedEnsemble)
            assert programs["fixed"]._experts is programs[FORWARD_BETA_STRATEGY]._experts
            results[budget] = programs
    return directory, compiler, spec, program, results, fits, histories


@pytest.mark.parametrize("budget", (0, 2))
def test_actual_prefix_prior_and_nonempty_posterior_save_load_reuse_base_capture(
        tmp_path, monkeypatch, original_beta_calls, budget):
    _, compiler, _, _, results, _, _ = original_beta_calls
    prefix, programs = compiler.compile(budget), results[budget]
    fitted = programs[FORWARD_BETA_STRATEGY]
    assert fitted.beta_calibration["n_decisions"] == budget // 2
    assert fitted.temperature_record["strategy"] == "fixed"
    assert semantic_sha256(fitted.beta_oof_record["calibration_groups"]) == semantic_sha256(
        [g.record() for g in prefix.complete_groups])
    raw_digest = programs["fixed"]._experts[0].predictor.digest()
    assert fitted._experts[0].predictor.digest() == raw_digest
    view = next(v for v in compiler.independent_views().decisions if v.decision_id.endswith("selection:1"))
    captured = {}

    def remember(name, _permission, capture):
        captured[name] = capture

    with cpu_thread_budget(1):
        first = FittedEnsemble.score_programs(programs, view, history_context=(compiler, prefix),
                                             history_recorder=remember)
    assert captured

    def forbidden(*_args, **_kwargs):
        raise AssertionError("saved probability replay refit or sampled")

    monkeypatch.setattr(JointFlatModel, "fit_prefix", forbidden)
    from transfer import joint_flat_model
    monkeypatch.setattr(joint_flat_model, "_run_prefix", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    package = FittedEnsemble.save_programs(tmp_path / "programmes", programs,
                                           participant_key=fitted.participant_key)
    restored = FittedEnsemble.load_programs(package)
    assert restored[FORWARD_BETA_STRATEGY].beta_calibration == fitted.beta_calibration
    assert restored[FORWARD_BETA_STRATEGY]._experts[0].predictor.digest() == raw_digest
    standalone = FittedEnsemble.load(fitted.save(tmp_path / "single-beta"))
    assert standalone.beta_calibration == fitted.beta_calibration
    assert standalone._experts[0].predictor.digest() == raw_digest
    with cpu_thread_budget(1):
        replayed = FittedEnsemble.score_programs(restored, view, history_context=(compiler, prefix),
                                                captured_histories=captured)
    assert {name: row["joint"].record() for name, row in replayed.items()} == {
        name: row["joint"].record() for name, row in first.items()}
    before = first["fixed"]["joint"].expert_identities[0]
    after = first[FORWARD_BETA_STRATEGY]["joint"].expert_identities[0]
    assert before.model_digest == after.model_digest == raw_digest
    assert before.numerical_input_digest == after.numerical_input_digest
    assert before.fit_digest != after.fit_digest
    assert after.probability_calibrated


def test_complete_original_rows_and_labels_reconstruct_the_full_posterior(original_beta_calls):
    _, _, _, program, results, _, _ = original_beta_calls
    fitted = results[2][FORWARD_BETA_STRATEGY]
    record = json.loads(json.dumps(fitted.beta_oof_record))
    assert semantic_sha256(validate_beta_uniform_oof(record, program, {"a": 1, "b": 1})) == semantic_sha256(fitted.beta_calibration)
    for mutation in ("label", "raw", "missing", "future"):
        changed = deepcopy(record)
        if mutation == "label":
            changed["calibration_groups"][0]["target_candidate"] = "q1"
        elif mutation == "raw":
            changed["rows"][0]["expert_evidence"]["joint"]["log_probability"] = np.log([.9, .1]).tolist()
        elif mutation == "missing":
            changed["rows"].clear()
        else:
            changed["rows"][0]["training_record_ids"] = [changed["rows"][0]["held_record_id"]]
        with pytest.raises(ValueError, match="Beta|beta"):
            validate_beta_uniform_oof(changed, program)
    assert semantic_sha256(record) == semantic_sha256(fitted.beta_oof_record)


def test_resealed_label_authority_cannot_rebind_the_original_base_model(original_beta_calls):
    from transfer.ensemble_fit import _beta_uniform_oof_state

    _, _, _, program, results, _, _ = original_beta_calls
    original = results[2][FORWARD_BETA_STRATEGY]
    changed = json.loads(json.dumps(original.beta_oof_record))
    changed["calibration_groups"][0]["target_candidate"] = "q1"
    changed["fit_authority"]["temperature_group_records"] = deepcopy(changed["calibration_groups"])
    changed["fit_authority_sha256"] = semantic_sha256(changed["fit_authority"])
    changed["beta_calibration"] = _beta_uniform_oof_state(changed, program, {"a": 1, "b": 1})
    # The attacker can reseal every new scalar/JSON relation. The original
    # numerical model retains the original complete-C fit digest.
    validate_beta_uniform_oof(changed, program)
    with pytest.raises(ValueError, match="original numerical model's C fit authority"):
        FittedEnsemble(original.scorer, original._experts, original.fit_receipts,
            original.temperature_record, target_contract=original._target_contract,
            beta_calibration=changed["beta_calibration"], beta_oof_record=changed)


def test_new_strategy_requires_explicit_prior_before_any_fit(original_beta_calls, monkeypatch):
    _, compiler, spec, program, _, _, _ = original_beta_calls
    monkeypatch.setattr(JointFlatModel, "fit_prefix", lambda *args, **kwargs: pytest.fail("fit before prior admission"))
    with pytest.raises(ValueError, match="explicit prior"):
        fit_calibration_prefix_programs(compiler.compile(0), (spec,), program,
            strategies=(FORWARD_BETA_STRATEGY,), prefix_compiler=compiler)


def test_failed_nonempty_forward_roster_stays_typed_unavailable(original_beta_calls, monkeypatch):
    from research.execution import ExpectedSubjectError
    from transfer.ensemble_fit import _FittedExpert

    _, compiler, spec, program, results, fits, histories = original_beta_calls
    source = results[2]["fixed"]._experts[0].predictor.source
    monkeypatch.setattr(_FittedExpert, "score", lambda self, view, **kwargs:
        self.unavailable(view, "original software forecast unavailable", "not_available"))
    programs = fit_calibration_prefix_programs(compiler.compile(2), (spec,), program,
        strategies=("fixed", FORWARD_BETA_STRATEGY), source_checkpoints={spec.expert_id: source},
        beta_uniform_prior={"a": 1, "b": 1}, prefix_compiler=compiler,
        _prefix_fit_cache=fits, _joint_history_captures=histories)
    assert isinstance(programs[FORWARD_BETA_STRATEGY], ExpectedSubjectError)
    failure = programs[FORWARD_BETA_STRATEGY].record(subject=compiler.target["participant_key"])
    assert "target:selection:0" in failure["detail"]
    assert "unavailable" in failure["detail"]
    assert programs["fixed"].beta_calibration is None


def test_original_issuer_excludes_adjacent_preroll_overlap_before_declaring_held_roster(original_beta_calls):
    from data.candidate_input import OriginalCandidateSelection
    from data.candidate_task import MembershipDecision

    _, original, _, _, _, _, _ = original_beta_calls
    source = original.adapter.recordings[0]
    decision = MembershipDecision.direct(("q0", "q1"), ("q0", "q1"),
        event_ids=("target:2:0", "target:2:1"), repetition_indices=np.zeros(2, dtype=np.int64), repetitions=1)
    onsets = np.array([100, 140, 300, 340, 450, 490])
    labels = np.array([True, False, True, False, True, False])
    # Declare a new software input. Never mutate the original fixture ledger.
    digest = hashlib.sha256(source.source_sha256.encode() + onsets.tobytes() + labels.tobytes()).hexdigest()
    source = replace(source, source_sha256=digest, native_onsets=onsets, authorized_binary_labels=labels,
        selections=(*source.selections, OriginalCandidateSelection("target:selection:2", (4, 5), decision, "q0")))
    adapter = replace(original.adapter, recordings=(source,))
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 4), profiles=(),
        authorization_preprocessing=original.authorization_preprocessing,
        response_seconds=.03125, pre_roll_seconds=.78125)
    prefix = compiler.compile(4)
    assert [group.view.record_id for group in prefix.complete_groups] == ["target:selection:0"]
    excluded = compiler._current_view(source, 1)
    assert excluded.raw_scope[0] == 100
    assert excluded.samples is None
    assert excluded.unavailable_reason == "previous_unknown_decision_overlaps_finite_response_support"
    # The actual input owner refuses this view before the forward learner's
    # full held roster is formed. A synthetic fallback must not invent an OOF.
    from transfer.ensemble_views import CalibrationSelection
    with pytest.raises(ValueError):
        compiler.before_group(prefix, CalibrationSelection(excluded, labels[2:4], "q0"))
