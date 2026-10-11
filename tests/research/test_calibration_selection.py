"""Real synthetic fitter counterexamples for complete group-nested selection."""

import json
from dataclasses import replace

import numpy as np
import pytest

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.calibration_selection import (
    CalibrationProgramCandidate,
    FittedCalibrationSelection,
    _failure_evidence,
    _metrics,
    _selection,
    _SubsetFits,
    fit_nested_calibration_selection,
    select_calibration_candidate,
)
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, SubjectFailureCode
from tests.support.checkpoints import write_checkpoint
from tests.transfer.test_ensemble_fit import programs, selections
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import FrozenSourceCheckpoint, fit_calibration_ensemble
from transfer.ensemble_views import build_calibration_epoch_dataset
from transfer.kernel_prior import KernelPriorSpec


@pytest.fixture(autouse=True)
def native_one():
    with cpu_thread_budget(1):
        yield


def groups(count=5):
    result = selections()
    for index in range(len(result), count):
        original = result[index % len(result)]
        view = replace(original.view, record_id=f"nested{index}", decision_id=f"nested_selection{index}",
            samples=np.asarray(original.view.samples)+np.random.default_rng(782+index).normal(0, .2e-6, original.view.samples.shape),
            raw_scope=(1000+600*index, 1551+600*index),
            available_sample=1551+600*index)
        result.append(replace(original, view=view))
    return tuple(result[:count])


def candidates():
    spec = replace(programs()[0], noise_order=0, n_basis=1)
    programme = EnsembleProgram.equal_weights("nested-raw", (spec.expert_id,))
    return (CalibrationProgramCandidate("fixed", (spec,), programme),
            CalibrationProgramCandidate("temperature", (spec,), programme, "grouped_temperature"))


@pytest.fixture(scope="module")
def nested_case():
    import transfer.ensemble_fit as owner

    original = owner._fit_expert
    actual_fit_calls = []

    def actual_fit(*args, **kwargs):
        actual_fit_calls.append(kwargs)
        return original(*args, **kwargs)

    with cpu_thread_budget(1), pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_fit_expert", actual_fit)
        calibration = groups()
        result = fit_nested_calibration_selection(calibration, candidates(), baseline_name="fixed")
    assert len(actual_fit_calls) == result.record["actual_fixed_fit_states"]
    return calibration, result


def test_real_nested_temperature_excludes_both_selection_and_outer_held(nested_case):
    calibration, result = nested_case
    assert len(result.record["outer_rows"]) == len(calibration) == 5
    assert result.record["outer_scores_used_for_final_selection"] is False
    assert result.record["exact_subset_cache_hits"] > 0
    assert result.record["actual_fixed_fit_states"] > 0
    for outer in result.record["outer_rows"]:
        v = outer["held_record_id"]
        assert v not in {group["view"]["record_id"] for group in outer["refit_scope"]["group_records"]}
        for row in outer["selection"]["candidate_selection_rows"]["temperature"]:
            h = row["held_record_id"]
            assert row["status"] == "scored"
            source = row["producer"]
            assert v not in {receipt["record_id"] for receipt in source["training_receipts"]}
            assert h not in {receipt["record_id"] for receipt in source["training_receipts"]}
            for temperature_oof in source["temperature_calibration"]["OOF_groups"]:
                assert v not in temperature_oof["training_record_ids"]
                assert h not in temperature_oof["training_record_ids"]
                assert temperature_oof["held_record_id"] not in {v, h}


def test_outer_truth_and_inner_truth_cannot_change_excluded_programs(nested_case):
    calibration, result = nested_case
    altered = replace(calibration[0], labels=~calibration[0].labels, target_candidate="b")
    # Only the train side of v=0 is evaluated; neither version passes its label.
    fits = _SubsetFits({}, "cpu")
    before, trace = _selection(calibration[1:], candidates(), "fixed", fits)
    after, second = _selection((altered, *calibration[1:])[1:], candidates(), "fixed", fits)
    assert before == after and trace == second
    assert semantic_sha256(trace) == semantic_sha256(result.record["outer_rows"][0]["selection"])
    # Now change a genuine selection-held label h within C\v and execute both
    # selectors. Other training rows may change; h's entire producer must not.
    training = calibration[1:]
    changed_h = replace(training[0], labels=~training[0].labels, target_candidate="b")
    _, altered_selection = _selection((changed_h, *training[1:]), candidates(), "fixed", fits)
    for name in ("fixed", "temperature"):
        original_h = trace["candidate_selection_rows"][name][0]
        altered_h = altered_selection["candidate_selection_rows"][name][0]
        assert semantic_sha256(original_h) == semantic_sha256(altered_h)
    # h=0's complete program is fitted on the other groups; its own truth changes
    # only the evaluator, while that inner program and temperature stay fixed.
    model, key = fits.fit(calibration[1:], candidates()[1])
    changed_model, changed_key = fits.fit((altered, *calibration[1:])[1:], candidates()[1])
    assert model is changed_model and key == changed_key
    assert model.score(calibration[0].view).record() == changed_model.score(altered.view).record()


def test_same_ids_different_actual_labels_and_samples_do_not_alias_cache():
    calibration = groups(3)
    fits = _SubsetFits({}, "cpu")
    first, key = fits.fit(calibration, candidates()[0])
    label_change = (replace(calibration[0], labels=~calibration[0].labels, target_candidate="b"), *calibration[1:])
    second, changed = fits.fit(label_change, candidates()[0])
    assert first is not second and key != changed
    sample_change = (replace(calibration[0], view=replace(calibration[0].view,
        samples=np.asarray(calibration[0].view.samples)*1.001)), *calibration[1:])
    third, changed_again = fits.fit(sample_change, candidates()[0])
    assert third is not first and changed_again not in {key, changed}


def test_grouped_insufficient_groups_keeps_outer_and_fixed_baseline():
    calibration = groups(3)
    result = fit_nested_calibration_selection(calibration, candidates(), baseline_name="fixed")
    assert len(result.record["outer_rows"]) == 3
    for outer in result.record["outer_rows"]:
        failed = outer["selection"]["candidate_selection_rows"]["temperature"]
        assert len(failed) == 2 and all(row["status"] == "unavailable" for row in failed)
        assert outer["selected_candidate"]["name"] == "fixed"
    assert result.record["selected_candidate"]["name"] == "fixed"


def finite_evidence(group, probability, *, missing=False):
    identity = CandidateExpertIdentity("fixture", group.view.raw_input_digest, "a"*64, None, "b"*64, "finite_fixture", False)
    counts = tuple(map(int, group.view.decision.membership.sum(axis=0)))
    return CandidateEvidence(group.view.decision.candidate_vocabulary, None if missing else np.log(probability),
        group.view.participant_key, group.view.decision_id, group.view.physical_support_digest,
        (identity,), group.view.decision.repetitions, counts, counts,
        failure_kind="fit_failure" if missing else None, failure_reason="finite_expected_failure" if missing else None)


def test_dropping_failed_probabilities_cannot_select_success_support_winner():
    calibration = groups(3)
    baseline = _metrics(calibration, [finite_evidence(group, [.6, .4]) for group in calibration])
    candidate = _metrics(calibration, [finite_evidence(group, [.999, .001]) for group in calibration[:2]])
    selected, trace = select_calibration_candidate({"baseline": baseline, "candidate": candidate},
        baseline_name="baseline", tie_order=("candidate", "baseline"))
    assert selected == "baseline"
    assert trace["comparisons"]["candidate"]["complete_requested_probabilities"] is False
    assert len(candidate.cells) == 3 and candidate.probability_coverage == 2/3
    tied, _ = select_calibration_candidate({"a": baseline, "b": baseline}, baseline_name="b", tie_order=("a", "b"))
    assert tied == "a"


def test_expected_fit_failure_and_group_unavailability_keep_distinct_requested_kinds():
    calibration = groups(3)
    candidate = candidates()[0]
    numerical = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="nested_fit", detail="failure fixture")
    insufficient = ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
        stage="nested_temperature", detail="group fixture")
    evidence = [_failure_evidence(calibration[0], candidate, "c"*64, numerical),
        _failure_evidence(calibration[1], candidate, "d"*64, insufficient), finite_evidence(calibration[2], [.8, .2])]
    metrics = _metrics(calibration, evidence)
    assert len(metrics.cells) == 3 and metrics.probability_coverage == 1/3
    assert metrics.cells[0].failure_kind == "fit_failure"
    assert metrics.cells[1].failure_kind == "not_available"
    assert metrics.participants[0].fit_failure_count == 1
    assert metrics.participants[0].not_available_count == 1


def test_final_original_package_and_selection_receipt_replay(nested_case, tmp_path):
    calibration, result = nested_case
    path = result.save(tmp_path/"selected")
    expected = semantic_sha256(result.record)
    restored = FittedCalibrationSelection.load(path, expected_selection_sha256=expected)
    assert restored.record == result.record
    future = groups(6)[5].view
    assert restored.score(future).record() == result.score(future).record()
    original_receipt = (path/"calibration_selection.json").read_bytes()
    payload = json.loads((path/"calibration_selection.json").read_bytes())
    old_strategy = payload["record"]["selected_candidate"]["strategy"]
    payload["record"]["selected_candidate"]["strategy"] = "fixed" if old_strategy == "grouped_temperature" else "grouped_temperature"
    payload["record_sha256"] = semantic_sha256(payload["record"])
    (path/"calibration_selection.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt|programme"):
        FittedCalibrationSelection.load(path, expected_selection_sha256=expected)
    (path/"calibration_selection.json").write_bytes(original_receipt)


def test_expected_final_failure_keeps_complete_nested_requested_receipt(nested_case, tmp_path):
    _, result = nested_case
    failure = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="final_refit", detail="explicit lifecycle fixture")
    record = {**result.record, "final_failure": failure.record(subject=result.record["participant_key"]),
              "final_producer_sha256": None}
    failed = FittedCalibrationSelection(failure, record)
    directory = failed.save(tmp_path/"failed_final")
    restored = FittedCalibrationSelection.load(directory, expected_selection_sha256=semantic_sha256(record))
    assert len(restored.record["outer_rows"]) == len(result.record["outer_rows"])
    assert restored.record["outer_requested"] == result.record["outer_requested"]
    assert set(path.name for path in directory.iterdir()) == {"calibration_selection.json"}
    with pytest.raises(ExpectedSubjectError, match="explicit lifecycle fixture"):
        restored.score(groups(6)[5].view)
    with pytest.raises(AttributeError):
        restored.final = result.final


def test_real_source_checkpoint_and_matched_scratch_roles_in_selection(tmp_path):
    calibration = groups(3)
    scratch = programs()[-1]
    source = replace(scratch, expert_id="source", recipe=replace(scratch.recipe, name="source", initialization="source"))
    data = build_calibration_epoch_dataset(calibration, source.profile)
    checkpoint = write_checkpoint(tmp_path/"source.pt", data, source_snapshot_sha256="c"*64)
    frozen = FrozenSourceCheckpoint.from_path(checkpoint)
    programme = EnsembleProgram.equal_weights("paired-source-scratch", (source.expert_id, scratch.expert_id))
    candidate = CalibrationProgramCandidate("paired", (source, scratch), programme)
    result = fit_nested_calibration_selection(calibration, (candidate,), baseline_name="paired",
        source_checkpoints={source.expert_id: frozen, scratch.expert_id: frozen})
    assert result.record["source_checkpoint_sha256"] == {source.expert_id: frozen.digest, scratch.expert_id: frozen.digest}
    for outer in result.record["outer_rows"]:
        identities = {row["program"]["expert_id"]: row for row in outer["producer"]["experts"]}
        assert identities[source.expert_id]["source_digest"] == frozen.digest
        assert identities[scratch.expert_id]["source_digest"] is None
    directory = result.save(tmp_path/"source_scratch")
    restored = FittedCalibrationSelection.load(directory, expected_selection_sha256=semantic_sha256(result.record))
    assert restored.record == result.record


def test_two_groups_are_unavailable_not_an_implicit_fallback():
    with pytest.raises(ExpectedSubjectError, match="three authorized complete groups"):
        fit_nested_calibration_selection(groups(2), candidates(), baseline_name="fixed")


def test_normal_multiple_basis_conditional_g_prior_nested_caller(tmp_path):
    calibration = groups(5)
    spec = replace(programs()[0], noise_order=0, n_basis=2, conditioning_samples=32,
        ridge=None, kernel_prior=KernelPriorSpec.calibration_g(9.))
    programme = EnsembleProgram.equal_weights("conditional-g-nested", (spec.expert_id,))
    declared = (CalibrationProgramCandidate("fixed-g", (spec,), programme),
                CalibrationProgramCandidate("temperature-g", (spec,), programme, "grouped_temperature"))
    result = fit_nested_calibration_selection(calibration, declared, baseline_name="fixed-g")
    for outer in result.record["outer_rows"]:
        assert outer["status"] == "scored"
        for row in outer["selection"]["candidate_selection_rows"]["temperature-g"]:
            assert row["status"] == "scored"
            program = row["producer"]["experts"][0]["program"]
            assert program["conditioning_samples"] == 32 and program["n_basis"] == 2
            assert program["kernel_prior"] == spec.kernel_prior.record()
            assert row["producer"]["temperature_calibration"]["strategy"] == "grouped_temperature"
    directory = result.save(tmp_path/"conditional_g")
    restored = FittedCalibrationSelection.load(directory, expected_selection_sha256=semantic_sha256(result.record))
    assert restored.record == result.record


def test_original_ridge_qr_red_and_complete_nested_caller_now_green(tmp_path):
    base = selections()
    copied = replace(base[0], view=replace(base[0].view, record_id="nested4", decision_id="nested_selection4",
        samples=np.asarray(base[0].view.samples)*1.01, raw_scope=(3400, 3951), available_sample=3951))
    saved = (base[3], copied)
    spec = replace(programs()[0], noise_order=0)
    programme = EnsembleProgram.equal_weights("original-ridge-nested", (spec.expert_id,))
    direct = fit_calibration_ensemble(saved, (spec,), programme)
    assert direct.fit_available
    # Preserve the original correlated-waveform five-group fixture. The exact
    # formerly failing pair is included in legitimate nested temperature fits.
    calibration = (*base, copied)
    declared = (CalibrationProgramCandidate("fixed-ridge", (spec,), programme),
                CalibrationProgramCandidate("temperature-ridge", (spec,), programme, "grouped_temperature"))
    result = fit_nested_calibration_selection(calibration, declared, baseline_name="fixed-ridge")
    assert len(result.record["outer_rows"]) == 5
    assert all(row["status"] == "scored" for row in result.record["outer_rows"])
    assert all(row["status"] == "scored" for outer in result.record["outer_rows"]
        for row in outer["selection"]["candidate_selection_rows"]["temperature-ridge"])
    directory = result.save(tmp_path/"ridge_nested")
    restored = FittedCalibrationSelection.load(directory, expected_selection_sha256=semantic_sha256(result.record))
    assert restored.record == result.record
