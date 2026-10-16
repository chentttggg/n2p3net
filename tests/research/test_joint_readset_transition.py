"""Opt-in saved-only vertical case against a real archived-issuer tiny genesis.

N2P3_JOINT_READSET_BASELINE points to its immutable BASELINE.json. Genesis is a
separate frozen-process action; this consumer hard-forbids every new fit/sampler.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_real_original_current_readset_nested_capsule_and_parent(tmp_path, monkeypatch):
    filename = os.environ.get('N2P3_JOINT_READSET_BASELINE')
    if filename is None:
        pytest.skip('requires an explicitly produced immutable original-issuer tiny baseline')
    baseline = json.loads(Path(filename).read_bytes())
    from research.calibration_selection import (
        CalibrationProgramCandidate,
        fit_nested_prefix_selection,
    )
    from research.contracts import semantic_sha256
    from research.joint_flat_replay import (
        JointFlatPackageAdmission,
        JointFlatReplayRecorder,
        _load_capture,
        load_completed_joint_primitives,
        validate_joint_flat_public_parent,
    )
    from research.joint_readset_transition import (
        SCHEMA,
        _IssuedJointReadsetTransition,
        joint_readset_relation,
    )
    from research.prefix_inner_replay import persist_prefix_inner_replay
    from research.prefix_replay_execution import _prepare_history_capabilities
    from research.prefix_replay_inputs import export_prepared_views, open_prepared_inputs
    from research.prefix_selection_replay import replay_prefix_selection
    from tests.research.test_forward_beta_nested_replay import _short_target
    from tests.research.test_joint_source_catalog import _record_issuer
    from train.runtime import cpu_thread_budget
    from transfer import ensemble_fit, joint_flat_model
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        FittedEnsemble,
        fit_calibration_prefix_programs,
    )
    from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation
    from transfer.pool_calibration import FORWARD_BETA_STRATEGY

    def receipt(path):
        path = Path(path)
        data = path.read_bytes()
        return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    before = {row['capture']['path']: receipt(row['capture']['path'])
              for row in baseline['completed_primitives']['captures']}
    for row in baseline['completed_primitives']['models']:
        path = Path(row['ensemble']['path'])
        before[str(path)] = receipt(path)
        model = json.loads(path.read_bytes())['record']['experts'][0]['model']
        before[str(path.parent/model)] = receipt(path.parent/model)
    for key in ('source_codec', 'source_model'):
        before[baseline[key]['path']] = receipt(baseline[key]['path'])
    blocked, components = [], []
    def forbidden(*_args, **_kwargs):
        blocked.append(True)
        raise AssertionError('transition consumer attempted a new source/C fit or sampler')
    monkeypatch.setattr(ensemble_fit, '_fit_expert', forbidden)
    monkeypatch.setattr(joint_flat_model, '_run_prefix', forbidden)
    monkeypatch.setattr(ContinuousDecisionModel, 'fit', forbidden)
    for name in ('collapsed_tail_components', 'factored_tail_components'):
        original = getattr(joint_flat_model, name)
        def counted(*args, _original=original, _name=name, **kwargs):
            components.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(joint_flat_model, name, counted)

    with cpu_thread_budget(1):
        views = _short_target()
        compiler, prefix = views.prefix_compiler, views.calibration[8]
        preparation = FrozenJointFlatSourcePreparation()
        source = preparation.prepare(Path(baseline['source_codec']['path']).read_bytes(),
            expected_sha256=baseline['source_codec']['sha256'])
        expert = CalibrationExpertProgram.from_record(baseline['expert'])
        programme = EnsembleProgram.from_record(baseline['programme'])
        sources, cache, captures = {'joint': source}, {}, {}
        arguments = dict(views=views, expert_programs=(expert,), source_checkpoints=sources,
            prefix_fit_cache=cache, history_captures=captures, source_preparation=preparation, context={})
        with pytest.raises(ValueError, match='original C/model/history/held authority'):
            load_completed_joint_primitives(baseline['completed_primitives'], **arguments)
        loaded = load_completed_joint_primitives(baseline['completed_primitives'], **arguments,
            readset_transition={'schema': SCHEMA, 'original_reader': baseline['actual_canonical_reader']})
        assert len(loaded['captures']) == 15 and all('readset_transition' in row for row in loaded['captures'])
        inner = cache[next(key for key, value in cache.items() if value.model_digest ==
            loaded['models'][0]['model_sha256'])]
        view = prefix.complete_groups[0].view
        earlier = compiler.compile(0)
        cap = compiler.record_read_scope(view, prefix=earlier,
            conditioning_stop=int(view.onsets[0])+view.raw_scope[0]+2)
        old_capture = _load_capture(baseline['completed_primitives']['captures'][0]['capture']['path'])
        with pytest.raises(ValueError, match='another actual history or model'):
            inner.predictor.score_history(view, cap, captured=old_capture)
        relation = joint_readset_relation(compiler, cap, old_capture)
        assert type(relation) is _IssuedJointReadsetTransition
        with pytest.raises(TypeError, match='actual issued transition'):
            inner.predictor.score_history(view, cap, captured=old_capture, _readset_relation=dict(relation.record))
        with pytest.raises(TypeError, match='takes no arguments'):
            replace(relation)
        forged = copy.copy(relation)
        with pytest.raises(TypeError, match='actual private input owner'):
            inner.predictor.score_history(view, cap, captured=old_capture, _readset_relation=forged)
        component_start = len(components)
        scored = inner.predictor.score_history(view, cap, captured=old_capture, _readset_relation=relation)
        assert len(components) == component_start+1 and scored.capture.digest == old_capture.digest
        second = inner.predictor.score_history(view, cap, captured=old_capture, _readset_relation=relation)
        assert len(components) == component_start+1 and second.capture.digest == old_capture.digest
        for field in ('available_sample', 'prefix_identity', 'source_sha256', 'public_selections',
                      'conditioning_sample_support', 'C_record_input_sha256'):
            binding = json.loads(json.dumps(old_capture.binding))
            target = binding['input']['history_capability']
            target[field] = 'forged-authority'
            binding['prefix_input_sha256'] = semantic_sha256(binding['input'])
            altered = replace(old_capture, binding=binding)
            with pytest.raises(ValueError, match='permission field '+field):
                joint_readset_relation(compiler, cap, altered)
        binding = json.loads(json.dumps(old_capture.binding))
        binding['input']['actual_target_axes'][0]['prior_mass'] = [.25,.75]
        binding['prefix_input_sha256'] = semantic_sha256(binding['input'])
        altered = replace(old_capture, binding=binding)
        altered_relation = joint_readset_relation(compiler, cap, altered)
        with pytest.raises(ValueError, match='original numerical input, source, model, q axes or operands'):
            inner.predictor.score_history(view, cap, captured=altered, _readset_relation=altered_relation)
        density = np.array(old_capture.logdensity, copy=True)
        density.flat[0] += 1e-5
        altered = replace(old_capture, logdensity=density)
        altered_relation = joint_readset_relation(compiler, cap, altered)
        with pytest.raises(ValueError, match='posterior components changed'):
            inner.predictor.score_history(view, cap, captured=altered, _readset_relation=altered_relation)
        # This relation cannot authorize an altered actual observed array.
        changed = np.array(cap.samples, copy=True)
        changed[0, 0] += 1e-6
        from transfer.ensemble_views import _register_record_history
        from transfer.noise_model import array_digest

        cap_binding = copy.deepcopy(dict(cap.binding))
        cap_binding['samples_sha256'] = array_digest(changed)
        bad_cap = _register_record_history(compiler, changed, earlier, cap_binding)
        with pytest.raises(ValueError, match='permission field samples_sha256'):
            joint_readset_relation(compiler, bad_cap, old_capture)
        # Restore a real current-issued cap for the remaining full vertical.
        cap = compiler.record_read_scope(view, prefix=earlier,
            conditioning_stop=int(view.onsets[0])+view.raw_scope[0]+2)

        context = {'requested_decision_ids': [view.record_id for view in views.decisions],
            'test_scope': views.ledger['test_scope']}
        recorder = JointFlatReplayRecorder(history_captures=captures)
        programmes = fit_calibration_prefix_programs(prefix, (expert,), programme,
            strategies=('fixed', FORWARD_BETA_STRATEGY), beta_uniform_prior={'a':1.,'b':1.},
            source_checkpoints=sources, prefix_compiler=compiler, inner_replay_recorder=recorder,
            inner_replay_context=context, _prefix_fit_cache=cache, _joint_history_captures=captures)
        assert isinstance(programmes[FORWARD_BETA_STRATEGY], FittedEnsemble)
        assert programmes[FORWARD_BETA_STRATEGY].beta_calibration['n_decisions'] == 4
        for future in views.decisions:
            recorder.score_future(programmes, future, compiler=compiler, prefix=prefix)
        package = persist_prefix_inner_replay(prefix, programmes, tmp_path/'ordinary', recorder=recorder,
            source_checkpoints=sources, context=context)
        candidates = (CalibrationProgramCandidate('raw-fixed', (expert,), programme),
            CalibrationProgramCandidate('explicit-beta', (expert,), programme, FORWARD_BETA_STRATEGY,
                beta_uniform_prior={'a':1.,'b':1.}))
        selected = fit_nested_prefix_selection(prefix, candidates, baseline_name='raw-fixed',
            prefix_compiler=compiler, minimum_selection_rows=1, source_checkpoints=sources,
            context=context, _prefix_fit_cache=cache, _joint_history_captures=captures)
        selected_scores = [selected.score(view).record() for view in views.decisions]
        graph = selected.save(tmp_path/'nested')
        n0 = fit_calibration_prefix_programs(compiler.compile(0), (expert,), programme,
            strategies=('fixed', FORWARD_BETA_STRATEGY), beta_uniform_prior={'a':1.,'b':1.},
            source_checkpoints=sources, prefix_compiler=compiler, _prefix_fit_cache=cache,
            _joint_history_captures=captures)
        assert n0[FORWARD_BETA_STRATEGY].beta_calibration['n_decisions'] == 0

        # Exercise the actual frozen-preparation callsite, then its capsule.
        records = [(package, recorder.prefix_receipt_sha256)]
        records.extend((tmp_path/'nested/models'/key, node['recorder'].prefix_receipt_sha256)
            for key, node in selected._fits.nodes.items())
        preparation_views = _short_target()
        preparation_compiler = preparation_views.prefix_compiler
        component_before_preparation = len(components)
        histories, refusals = _prepare_history_capabilities(preparation_views, (0,2,4,6,8), records)
        assert len(components) == component_before_preparation
        request = {'schema':'n2p3_prefix_replay_input_request/2','preparation_task_id':'readset-transition-tiny',
            'original_source_sha256':views.ledger['source_sha256'],
            'producer_source_sha256':compiler.history_owner_source,
            'participant_key':prefix.target_contract['participant_key'],
            'source_relative_path':prefix.target_contract['source_relative_path'],
            'requested_Ks':[0,8],'actual_Ks':[0,2,4,6,8],'profiles':[],
            'authorization_preprocessing':compiler.authorization_preprocessing.record(),
            'observation_response_seconds':.25,'pre_roll_seconds':.0625,
            'dataset_source':{'explicit_software_fixture':True}}
        capsule = export_prepared_views(preparation_views, (), tmp_path/'capsule', request,
            record_history_scopes=histories, record_history_refusals=refusals)
        capsule_bytes = (tmp_path/'capsule/capsule.json').read_bytes()
        assert len(json.loads(capsule_bytes)['joint_readset_transitions']['relations']) >= 5
        assert preparation_compiler is not compiler
        descriptor = {'schema':'n2p3_prefix_replay_shared_inputs/1','backing':'immutable_bytes',
            'capsule':capsule_bytes,'payload':(tmp_path/'capsule/arrays.bin').read_bytes(),
            'capsule_sha256':capsule['sha256'],'request_sha256':semantic_sha256(request)}
        with monkeypatch.context() as transport_guard:
            from research import joint_readset_transition

            def no_archived_execution(*args, **kwargs):
                raise AssertionError('prepared transport attempted to execute the live archived issuer')
            original_read_bytes = Path.read_bytes
            original_path = Path(baseline['actual_canonical_reader']['path']).resolve()
            def no_external_original(path):
                if path.resolve() == original_path:
                    raise AssertionError('capsule reopened the original external reader locator')
                return original_read_bytes(path)
            transport_guard.setattr(joint_readset_transition, '_original_module', no_archived_execution)
            transport_guard.setattr(Path, 'read_bytes', no_external_original)
            with open_prepared_inputs(descriptor, expected_request_sha=semantic_sha256(request),
                    expected_capsule_sha=capsule['sha256']) as borrowed:
                assert not borrowed.views.scoring_truth
                result = JointFlatPackageAdmission(package).replay(prefix=borrowed.views.calibration[8],
                    prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
                    source_checkpoints=sources)
                assert len(result['scores']) == 1
                replay = replay_prefix_selection(graph, prefix=borrowed.views.calibration[8],
                    prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
                    source_checkpoints=sources, expected_selection_sha256=semantic_sha256(selected.record),
                    expected_graph_sha256=selected.graph_sha256)
                assert replay['zero_new_fits'] and replay['outer_requested'] == 4
        # Fresh live owners, so the parent has to rebuild both issuer endpoints.
        fresh_views = _short_target()
        source_compiler, _ = _record_issuer('source')
        parent = validate_joint_flat_public_parent(package, original_views=fresh_views,
            source_cohort=((source_compiler, source_compiler.compile(8)),))
        assert parent['requested_decisions'] == 1 and parent['new_Gibbs_calls'] == 0
        attacked = tmp_path/'fully-resealed-relation'
        shutil.copytree(package, attacked)
        manifest = attacked/'prefix_inner_replay.json'
        resealed = json.loads(manifest.read_bytes())
        resealed['record']['OOF'][0]['captures']['joint']['readset_transition']['current_numeric_input_sha256'] = '0'*64
        resealed['record_sha256'] = semantic_sha256(resealed['record'])
        manifest.write_text(json.dumps(resealed), encoding='utf-8')
        JointFlatPackageAdmission(attacked)  # Writer consistency alone admits the complete resealed record.
        with pytest.raises(ValueError, match='actual two-issuer reissue'):
            validate_joint_flat_public_parent(attacked, original_views=_short_target(),
                source_cohort=((source_compiler, source_compiler.compile(8)),))
        assert not blocked and selected_scores
        for path, expected in before.items():
            assert receipt(path) == expected
        checkout = Path(__file__).resolve().parents[2]
        runtime_sources = {str(Path(module.__file__).resolve()): receipt(Path(module.__file__).resolve())
            for module in list(sys.modules.values()) if getattr(module, '__file__', None)
            and Path(module.__file__).is_absolute() and Path(module.__file__).suffix == '.py'
            and Path(module.__file__).is_file() and Path(module.__file__).resolve().is_relative_to(checkout)}
        (tmp_path/'TRANSITION_RESULT.json').write_text(json.dumps({'status':'PASS',
            'genesis_counts':baseline['genesis_counts'],'new_consumer_fits':0,'new_consumer_sampler_calls':0,
            'nested_genesis_addendum':baseline['nested_addendum'],
            'original_files_before_equal_after':True,'nested_outer_rows':4,'future_requested':1,
            'first_use_component_computations':len(components),'live_dual_issuer_parent':parent,
            'original_reader_sha256':baseline['actual_canonical_reader']['sha256'],
            'fresh_preparation_numerical_component_calls':0,'capsule_truth_free':True,
            'capsule_without_external_reader_or_old_issuer_execution':True,
            'runtime_source_files':runtime_sources,
            'original_captures_byte_preserved':True},indent=2)+'\n',encoding='utf-8')
