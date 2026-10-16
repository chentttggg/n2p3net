"""A real small history capture distinguishes private reuse from admission."""
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def small_history_case(tmp_path):
    from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
    from data.candidate_input import (
        CandidateDatasetAdapter,
        OriginalCandidateRecording,
        OriginalCandidateSelection,
    )
    from data.candidate_task import MembershipDecision
    from data.contract import DEFAULT_P300_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from data.identity import origin_subject_key
    from research.joint_flat_authority import build_source_parameters
    from tests.research.test_joint_flat_response import compiler_fixture
    from transfer.ensemble_views import RecordCalibrationPrefixCompiler
    from transfer.joint_flat_model import JointFlatModel
    from transfer.joint_flat_programme import (
        FrozenJointFlatSource,
        JointFlatProgram,
        JointFlatSamplingProgram,
    )

    def compiler(subject, seed, onsets, budgets):
        participant = origin_subject_key("software_history", subject)
        values = np.random.default_rng(seed).normal(0, 5e-6, (500, 8))
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        groups = []
        for ordinal in range(2):
            decision = MembershipDecision.direct(("q0", "q1"), ("q0", "q1"),
                event_ids=(f"{subject}:{ordinal}:0", f"{subject}:{ordinal}:1"),
                repetition_indices=np.zeros(2, dtype=np.int64), repetitions=1)
            groups.append(OriginalCandidateSelection(f"{subject}:selection:{ordinal}",
                (2*ordinal, 2*ordinal+1), decision, "q0"))
        record = OriginalCandidateRecording(subject, digest, subject + ".raw", participant, subject, "0", "0",
            256., BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE, len(values), np.array(onsets, dtype=np.int64),
            np.array([True, False, True, False]), tuple(groups), {"explicit_software_fixture": True},
            lambda start, stop: values[start:stop])
        adapter = CandidateDatasetAdapter("software_history", participant, subject, "explicit_software_fixture",
            (record,), (), {}, "actual_software_record", "a" * 64)
        return RecordCalibrationPrefixCompiler(adapter, budgets=budgets, profiles=(),
            authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
            response_seconds=.03125, pre_roll_seconds=.0078125)

    source = compiler_fixture("A01.mat", 840)
    target = compiler("target", 841, (100, 140, 300, 340), (0,))
    source_path = tmp_path / "small_original_source.npz"
    parameters, _ = build_source_parameters(((source, source.compile(120)),),
        target_participant_key=target.target["participant_key"], degrees_of_freedom=4., relative_noise_shape=1.,
        basis=np.ones((4, 1)), ridge=1., noise_order=0, conditioning_samples=2, source_model_path=source_path)
    frozen = FrozenJointFlatSource.from_parameters(parameters, original_model_content=source_path.read_bytes())
    sampler = JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 100., 0., 0., 100.,
        leaf_rows=16, qr_cache_bytes=65536)
    programme = JointFlatProgram(sampler, 2, "whole_observed_current_q")
    prefix = target.compile(0)
    model = JointFlatModel.fit_prefix(frozen, programme, compiler=target, prefix=prefix)
    view = next(view for view in target.independent_views().decisions if view.decision_id == "target:selection:0")
    stop = int(view.onsets[0] + view.raw_scope[0] + programme.conditioning_samples)
    cap = target.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
    return model, view, cap, target, prefix


@pytest.fixture(scope="module")
def actual_history(tmp_path_factory):
    from train.runtime import cpu_thread_budget

    directory = tmp_path_factory.mktemp("actual-small-history")
    with cpu_thread_budget(1):
        model, view, cap, compiler, prefix = small_history_case(directory)
        original = model.score_history(view, cap)
    assert not model._verified_history_components._entries
    assert model._verified_history_components.used_bytes == 0
    return model, view, cap, compiler, prefix, original.capture, directory


def _component_counter(monkeypatch):
    from transfer import joint_flat_model as owner

    original, calls = owner.collapsed_tail_components, []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("captured history replay sampled again")

    monkeypatch.setattr(owner, "collapsed_tail_components", counted)
    monkeypatch.setattr(owner, "_run_prefix", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    return calls


def test_true_capture_first_replay_computes_then_reuses_without_serializing_or_aliasing(actual_history, monkeypatch):
    from train.runtime import cpu_thread_budget
    from transfer.joint_flat_model import JointFlatModel

    base, view, cap, _, _, capture, directory = actual_history
    model = replace(base)
    assert model == base and model._verified_history_components is not base._verified_history_components
    calls, digest = _component_counter(monkeypatch), model.digest()
    before, after = directory / "before-cache.npz", directory / "after-cache.npz"
    model.save(before)
    with cpu_thread_budget(1):
        start = time.perf_counter()
        first = model.score_history(view, cap, captured=capture)
        first_seconds = time.perf_counter() - start
        assert len(calls) == 1 and model._verified_history_components.used_bytes > 0
        second = model.score_history(view, cap, captured=capture)
        assert len(calls) == 1 and model._verified_history_components.hits == 1
        assert first.capture.digest == second.capture.digest == capture.digest
        assert first.capture.binding['input']['numeric_producer'] == capture.binding['input']['numeric_producer']
        first.capture.logweights.shape = (first.capture.logweights.size,)
        first.summary['scoring_law'] = 'caller mutation'
        start = time.perf_counter()
        third = model.score_history(view, cap, captured=capture)
        hit_seconds = time.perf_counter() - start
        assert len(calls) == 1 and third.capture.digest == capture.digest
        assert third.summary['scoring_law'] == model.programme.scoring_law
        np.testing.assert_array_equal(third.log_probability, second.log_probability)
    entry = next(iter(model._verified_history_components._entries.values()))
    assert not np.shares_memory(entry[0], third.capture.logweights)
    with pytest.raises(ValueError):
        entry[0].setflags(write=True)
    assert model.digest() == digest
    model.save(after)
    with np.load(before, allow_pickle=False) as old, np.load(after, allow_pickle=False) as new:
        assert old.files == new.files
        for name in old.files:
            np.testing.assert_array_equal(old[name], new[name])
    reloaded = JointFlatModel.load(after)
    assert reloaded.digest() == digest and not reloaded._verified_history_components._entries
    assert not replace(model)._verified_history_components._entries
    (directory / "reuse-timing.json").write_text(json.dumps({
        'first_verified_replay_seconds': first_seconds, 'cache_hit_replay_seconds': hit_seconds,
        'component_calls_for_three_captured_replays': len(calls),
        'cached_payload_bytes': model._verified_history_components.used_bytes,
        'original_capture_digest': capture.digest, 'returned_capture_digest': third.capture.digest}, indent=2))


def test_changed_authority_input_or_capture_never_uses_verified_components(actual_history, monkeypatch):
    from train.runtime import cpu_thread_budget

    base, view, cap, compiler, prefix, capture, _ = actual_history
    model, calls = replace(base), _component_counter(monkeypatch)
    with cpu_thread_budget(1):
        model.score_history(view, cap, captured=capture)
        samples = view.samples.copy()
        samples[0, 0] = np.nextafter(samples[0, 0], np.inf)
        with pytest.raises(ValueError, match="record capability"):
            model.score_history(replace(view, samples=samples), cap, captured=capture)
        assert len(calls) == 1 and model._verified_history_components.hits == 0
        other_cap = compiler.record_read_scope(view, prefix=prefix,
            conditioning_stop=cap.binding['conditioning_sample_support'][1] + 1)
        with pytest.raises(ValueError, match="another actual history"):
            model.score_history(view, other_cap, captured=capture)
        assert len(calls) == 1
        wrong = capture.logdensity.copy()
        wrong[0, 0, 0] = np.nextafter(wrong[0, 0, 0], np.inf)
        with pytest.raises(ValueError, match="posterior components changed"):
            model.score_history(view, cap, captured=replace(capture, logdensity=wrong))
        assert len(calls) == 2 and len(model._verified_history_components._entries) == 1
        assert model.score_history(view, cap, captured=capture).capture.digest == capture.digest
        assert len(calls) == 2


def test_consumer_source_and_effective_native_context_change_require_recomputation(actual_history, monkeypatch):
    from train.runtime import cpu_thread_budget
    from transfer import joint_flat_model as owner

    base, view, cap, _, _, capture, directory = actual_history
    model, calls = replace(base), _component_counter(monkeypatch)
    actual_consumer = owner._history_consumer_context()
    assert actual_consumer['current_modules']['modules_sha256']['transfer/joint_flat_model.py'] == hashlib.sha256(
        Path(owner.__file__).read_bytes()).hexdigest()
    variant = directory / "changed-consumer.py"
    variant.write_text("# A distinct consumer source identity for the key counterexample.\n")
    with cpu_thread_budget(1):
        model.score_history(view, cap, captured=capture)
        monkeypatch.setattr(owner, '_history_consumer_context', lambda: {
            **actual_consumer, 'changed_source_sha256': hashlib.sha256(variant.read_bytes()).hexdigest()})
        model.score_history(view, cap, captured=capture)
        assert len(calls) == 2
        monkeypatch.setattr(owner, '_history_consumer_context', lambda: actual_consumer)
        native_one = owner._history_native_context()
    with cpu_thread_budget(2):
        assert owner._history_native_context() != native_one
        model.score_history(view, cap, captured=capture)
        assert len(calls) == 3


def test_bounded_lru_and_zero_payload_capacity_preserve_true_capture(actual_history, monkeypatch):
    from train.runtime import cpu_thread_budget
    from transfer import joint_flat_model as owner

    base, view, cap, _, _, capture, _ = actual_history
    calls = _component_counter(monkeypatch)
    payload = capture.logweights.nbytes + capture.logdensity.nbytes
    with cpu_thread_budget(1):
        for capacity in (0, payload - 1):
            model = replace(base)
            object.__setattr__(model, '_verified_history_components', owner._VerifiedHistoryComponents(capacity))
            before = len(calls)
            for _ in range(2):
                assert model.score_history(view, cap, captured=capture).capture.digest == capture.digest
            assert len(calls) == before + 2 and model._verified_history_components.used_bytes == 0
        model = replace(base)
        object.__setattr__(model, '_verified_history_components', owner._VerifiedHistoryComponents(payload * 2, 1))
        context = owner._history_consumer_context()
        monkeypatch.setattr(owner, '_history_consumer_context', lambda: context)
        before = len(calls)
        model.score_history(view, cap, captured=capture)
        context['different_consumer'] = 'f' * 64
        model.score_history(view, cap, captured=capture)
        assert len(model._verified_history_components._entries) == 1
        assert model._verified_history_components.used_bytes == payload
        context.pop('different_consumer')
        model.score_history(view, cap, captured=capture)
        assert len(calls) == before + 3 and len(model._verified_history_components._entries) == 1
