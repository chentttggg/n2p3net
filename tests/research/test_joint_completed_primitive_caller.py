"""Real saved C/P producers reach forward, future and selector callers."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from tests.research.test_joint_flat_history_score_cache import small_history_case


def test_original_saved_capture_reaches_forward_future_and_selector_without_new_sampling(tmp_path, monkeypatch):
    from research.calibration_selection import CalibrationProgramCandidate, _PrefixFits
    from research.joint_flat_replay import (
        JointFlatReplayRecorder,
        _save_capture,
        load_completed_joint_primitives,
    )
    from train.runtime import cpu_thread_budget
    from transfer import ensemble_fit, joint_flat_model
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        _forward_joint_temperature,
        fit_calibration_prefix,
    )
    from transfer.ensemble_views import RecordCalibrationPrefixCompiler
    from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

    def artifact(path):
        return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    with cpu_thread_budget(1):
        base, _, _, original_compiler, _ = small_history_case(tmp_path)
        compiler = RecordCalibrationPrefixCompiler(original_compiler.adapter, budgets=(2,), profiles=(),
            authorization_preprocessing=replace(original_compiler.authorization_preprocessing,
                tmin_ms=-2000/256, tmax_ms=8000/256, n_times=10),
            response_seconds=8/256, pre_roll_seconds=2/256)
        views = compiler.independent_views()
        prefix, earlier = compiler.compile(2), compiler.compile(0)
        held = prefix.complete_groups[0]
        spec = CalibrationExpertProgram('joint', 'joint_flat', joint_flat=base.programme)
        programme = EnsembleProgram.equal_weights('actual-caller', ('joint',))
        sources = {'joint': base.source}
        original_cache = {}
        old_zero = fit_calibration_prefix(earlier, (spec,), programme, source_checkpoints=sources,
            prefix_compiler=compiler, _prefix_fit_cache=original_cache)
        old_final = fit_calibration_prefix(prefix, (spec,), programme, source_checkpoints=sources,
            prefix_compiler=compiler, _prefix_fit_cache=original_cache)
        original_recorder = JointFlatReplayRecorder()
        _forward_joint_temperature(old_final, (held,), (spec,), sources, 'cpu', prefix=prefix,
            prefix_compiler=compiler, temperature_expert_ids=None, producer_recorder=original_recorder,
            producer_context={}, prefix_fit_cache=original_cache)
        original_row = original_recorder.joint_oof[0]
        capture = original_row['captures']['joint'][1]
        models = []
        for budget, fitted in ((0, old_zero), (2, old_final)):
            directory = tmp_path / f'original_C{budget}'
            fitted.save(directory)
            models.append({'K': budget, 'ensemble': artifact(directory / 'ensemble.json')})
        original_row_path, capture_path = tmp_path / 'forward-row.json', tmp_path / 'original-capture.npz'
        original_row_path.write_text(json.dumps(original_row['row']), encoding='utf-8')
        _save_capture(capture_path, capture)
        declaration = {'models': models, 'captures': [{'K': 0, 'expert_id': 'joint',
            'held_record_id': held.view.record_id, 'capture': artifact(capture_path),
            'forward_row': artifact(original_row_path)}]}
        cache, histories = {}, {}
        admitted = load_completed_joint_primitives(declaration, views=views, expert_programs=(spec,),
            source_checkpoints=sources, prefix_fit_cache=cache, history_captures=histories,
            source_preparation=FrozenJointFlatSourcePreparation(), context={})
        assert len(admitted['models']) == 2 and len(admitted['captures']) == 1
        loaded_zero = cache[old_zero._experts[0].fit_digest]
        assert loaded_zero.model_digest == old_zero._experts[0].model_digest
        assert not loaded_zero.predictor._verified_history_components._entries
        component_calls, forbidden_calls = [], []
        original_components = joint_flat_model.collapsed_tail_components

        def components(*args, **kwargs):
            component_calls.append(True)
            return original_components(*args, **kwargs)

        def forbidden(*args, **kwargs):
            forbidden_calls.append(True)
            raise AssertionError('caller attempted a new fit or sampler despite its completed original primitive')

        monkeypatch.setattr(joint_flat_model, 'collapsed_tail_components', components)
        monkeypatch.setattr(joint_flat_model, '_run_prefix', forbidden)
        monkeypatch.setattr(ensemble_fit, '_fit_expert', forbidden)
        model = fit_calibration_prefix(earlier, (spec,), programme, source_checkpoints=sources,
            prefix_compiler=compiler, _prefix_fit_cache=cache)
        with pytest.raises(AssertionError, match='despite its completed original primitive'):
            model.score_experts(held.view, history_context=(compiler, earlier))
        assert len(forbidden_calls) == 1
        final = fit_calibration_prefix(prefix, (spec,), programme, source_checkpoints=sources,
            prefix_compiler=compiler, _prefix_fit_cache=cache)
        forward_recorder = JointFlatReplayRecorder(history_captures=histories)
        _forward_joint_temperature(final, (held,), (spec,), sources, 'cpu', prefix=prefix,
            prefix_compiler=compiler, temperature_expert_ids=None, producer_recorder=forward_recorder,
            producer_context={}, prefix_fit_cache=cache, history_captures=histories)
        assert len(component_calls) == 1 and len(forbidden_calls) == 1
        assert forward_recorder.joint_oof[0]['captures']['joint'][1].digest == capture.digest
        recorder = JointFlatReplayRecorder(history_captures=histories)
        programs = ensemble_fit.fit_calibration_prefix_programs(earlier, (spec,), programme, strategies=('fixed',),
            source_checkpoints=sources, prefix_compiler=compiler, inner_replay_recorder=recorder,
            _prefix_fit_cache=cache, _joint_history_captures=histories)
        future = recorder.score_future(programs, held.view, compiler=compiler, prefix=earlier)
        assert future['fixed']['joint'].probability_available
        candidate = CalibrationProgramCandidate('actual-fixed', (spec,), programme)
        fits = _PrefixFits(sources, 'cpu', {'requested_decision_ids': [held.view.record_id],
            'test_scope': {'start_sample': held.view.raw_scope[0], 'stop_sample': held.view.raw_scope[1]}}, compiler,
            prefix_fit_cache=cache, joint_history_captures=histories)
        _, key = fits.fit(earlier, candidate, held=held, use={'phase': 'exact_caller_counterexample'})
        selected = fits.score(key, held.view)
        assert selected.probability_available
        assert len(component_calls) == 1 and len(forbidden_calls) == 1
        assert cache[old_zero._experts[0].fit_digest].model_digest == old_zero._experts[0].model_digest
        assert cache[old_final._experts[0].fit_digest].model_digest == old_final._experts[0].model_digest
        np.testing.assert_array_equal(future['fixed']['joint'].log_probability,
            original_row['row']['expert_evidence']['joint']['log_probability'])
        (tmp_path / 'caller-verification.json').write_text(json.dumps({'original_models': 2, 'original_P_captures': 1,
            'original_capture_sha256': capture.digest, 'first_consumer_component_recomputations': len(component_calls),
            'deliberately_blocked_missing_capture_sampler_attempts': len(forbidden_calls),
            'new_fit_or_sampler_calls_on_three_reuse_paths': 0,
            'paths': ['forward', 'future_recorder', 'selector_prefix_fits']}, indent=2), encoding='utf-8')
