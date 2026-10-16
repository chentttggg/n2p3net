"""New forward Beta programme crosses actual nested/capsule/parent gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from tests.research.test_joint_source_catalog import _record_issuer
from tests.research.test_joint_source_catalog import original_source as _original_source

original_source = _original_source


def _short_target():
    from transfer.ensemble_views import RecordCalibrationPrefixCompiler

    original, samples = _record_issuer('target', seed=91)
    record = original.adapter.recordings[0]
    onsets, labels = record.native_onsets[:10], record.authorized_binary_labels[:10]
    digest = hashlib.sha256(samples.tobytes() + onsets.tobytes() + labels.tobytes()).hexdigest()
    record = replace(record, source_sha256=digest, native_onsets=onsets, authorized_binary_labels=labels,
        selections=record.selections[:5])
    adapter = replace(original.adapter, recordings=(record,), source_artifacts=({'sha256': digest},))
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 8), profiles=(),
        authorization_preprocessing=original.authorization_preprocessing, response_seconds=.25, pre_roll_seconds=.0625)
    return compiler.independent_views()


def test_forward_beta_nested_capsule_parent_rejects_resealed_label_and_raw_forecast(original_source, tmp_path, monkeypatch):
    from research.calibration_selection import (
        CalibrationProgramCandidate,
        fit_nested_prefix_selection,
    )
    from research.contracts import semantic_sha256
    from research.joint_flat_replay import (
        JointFlatPackageAdmission,
        JointFlatReplayRecorder,
        validate_joint_flat_public_parent,
    )
    from research.prefix_inner_replay import persist_prefix_inner_replay
    from research.prefix_replay_inputs import export_prepared_views, open_prepared_inputs
    from research.prefix_selection_replay import replay_prefix_selection
    from train.runtime import cpu_thread_budget
    from transfer import ensemble_fit, joint_flat_model
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        FittedEnsemble,
        fit_calibration_prefix_programs,
    )
    from transfer.joint_flat_programme import JointFlatProgram, JointFlatSamplingProgram
    from transfer.pool_calibration import FORWARD_BETA_STRATEGY

    with cpu_thread_budget(1):
        views = _short_target()
        compiler, prefix = views.prefix_compiler, views.calibration[8]
        assert prefix.budget == 8 and len(prefix.complete_groups) == 4 and len(views.decisions) == 1
        sampler = JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 100., 0., 0., 100., leaf_rows=16)
        expert = CalibrationExpertProgram('joint', 'joint_flat', joint_flat=JointFlatProgram(
            sampler, 2, 'whole_observed_current_q'))
        programme = EnsembleProgram.equal_weights('explicit-forward-beta-development', ('joint',))
        prior, sources = {'a': 1., 'b': 1.}, {'joint': original_source.source}
        cache, captures, records, packages, programmes = {}, {}, {}, {}, {}
        context = {'requested_decision_ids': [view.record_id for view in views.decisions],
            'test_scope': views.ledger['test_scope']}
        for budget in (0, 8):
            current = compiler.compile(budget)
            recorder = JointFlatReplayRecorder(history_captures=captures)
            values = fit_calibration_prefix_programs(current, (expert,), programme,
                strategies=('fixed', FORWARD_BETA_STRATEGY), beta_uniform_prior=prior,
                source_checkpoints=sources, prefix_compiler=compiler, inner_replay_recorder=recorder,
                inner_replay_context=context, _prefix_fit_cache=cache, _joint_history_captures=captures)
            assert isinstance(values[FORWARD_BETA_STRATEGY], FittedEnsemble)
            state = values[FORWARD_BETA_STRATEGY].beta_calibration
            assert state['n_decisions'] == len(current.complete_groups)
            assert values['fixed']._experts[0].model_digest == values[FORWARD_BETA_STRATEGY]._experts[0].model_digest
            for view in views.decisions:
                recorder.score_future(values, view, compiler=compiler, prefix=current)
            package = persist_prefix_inner_replay(current, values, tmp_path / f'K{budget}', recorder=recorder,
                source_checkpoints=sources, context=context)
            records[budget], packages[budget], programmes[budget] = recorder, package, values
        candidates = (CalibrationProgramCandidate('raw-fixed', (expert,), programme),
            CalibrationProgramCandidate('legacy-forward-temperature', (expert,), programme, 'grouped_temperature'),
            CalibrationProgramCandidate('explicit-beta', (expert,), programme, FORWARD_BETA_STRATEGY,
                beta_uniform_prior=prior))
        selected = fit_nested_prefix_selection(prefix, candidates, baseline_name='raw-fixed',
            prefix_compiler=compiler, minimum_selection_rows=1, source_checkpoints=sources,
            context=context, _prefix_fit_cache=cache, _joint_history_captures=captures)
        selected_scores = [selected.score(view).record() for view in views.decisions]
        graph = selected.save(tmp_path / 'nested')
        assert len(selected.record['outer_requested']) == 4
        assert len(selected.record['outer_rows']) == 4
        assert {node['candidate'].strategy for node in selected._fits.nodes.values()} >= {
            'fixed', 'grouped_temperature', FORWARD_BETA_STRATEGY}
        assert any(node['candidate'].strategy == FORWARD_BETA_STRATEGY and node['prefix'].budget == 6
            for node in selected._fits.nodes.values())
        histories = {cap.digest: cap for recorder in (*records.values(),
            *(node['recorder'] for node in selected._fits.nodes.values()))
            for cap in recorder.original_history_capabilities}
        request = {'schema': 'n2p3_prefix_replay_input_request/2', 'preparation_task_id': 'explicit-beta-software-capsule',
            'original_source_sha256': views.ledger['source_sha256'], 'producer_source_sha256': compiler.history_owner_source,
            'participant_key': prefix.target_contract['participant_key'],
            'source_relative_path': prefix.target_contract['source_relative_path'], 'requested_Ks': [0, 8],
            'actual_Ks': [0, 2, 4, 6, 8], 'profiles': [], 'authorization_preprocessing': compiler.authorization_preprocessing.record(),
            'observation_response_seconds': .25, 'pre_roll_seconds': .0625,
            'dataset_source': {'explicit_software_fixture': True}}
        capsule = export_prepared_views(views, (), tmp_path / 'capsule', request, record_history_scopes=histories.values())
        capsule_bytes = (tmp_path / 'capsule/capsule.json').read_bytes()
        descriptor = {'schema': 'n2p3_prefix_replay_shared_inputs/1', 'backing': 'immutable_bytes',
            'capsule': capsule_bytes, 'payload': (tmp_path / 'capsule/arrays.bin').read_bytes(),
            'capsule_sha256': capsule['sha256'], 'request_sha256': semantic_sha256(request)}
        blocked = []
        def forbidden(*args, **kwargs):
            blocked.append(True)
            raise AssertionError('new programme replay attempted an EEG fit or sampler')
        monkeypatch.setattr(ensemble_fit, '_fit_expert', forbidden)
        monkeypatch.setattr(joint_flat_model, '_run_prefix', forbidden)
        with open_prepared_inputs(descriptor, expected_request_sha=semantic_sha256(request),
                expected_capsule_sha=capsule['sha256']) as borrowed:
            assert not borrowed.views.scoring_truth
            replay = replay_prefix_selection(graph, prefix=borrowed.views.calibration[8],
                prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
                source_checkpoints=sources, expected_selection_sha256=semantic_sha256(selected.record),
                expected_graph_sha256=selected.graph_sha256)
            assert replay['zero_new_fits'] and replay['outer_requested'] == 4
            for budget in (0, 8):
                admission = JointFlatPackageAdmission(packages[budget])
                result = admission.replay(prefix=borrowed.views.calibration[budget],
                    prefix_compiler=borrowed.views.prefix_compiler, future_views=borrowed.views.decisions,
                    source_checkpoints=sources)
                assert len(result['scores']) == 1
        parent = validate_joint_flat_public_parent(packages[8], original_views=views,
            source_cohort=((original_source.compiler, original_source.prefix),))
        assert parent['requested_decisions'] == 1 and parent['new_Gibbs_calls'] == 0 and not blocked
        original = programmes[8][FORWARD_BETA_STRATEGY]
        rejected = {}
        for attack in ('authorized_C_target', 'raw_OOF_probability'):
            oof = json.loads(json.dumps(original.beta_oof_record))
            if attack == 'authorized_C_target':
                group = oof['calibration_groups'][0]
                vocabulary = group['view']['decision']['candidate_vocabulary']
                group['target_candidate'] = next(value for value in vocabulary if value != group['target_candidate'])
                oof['fit_authority']['temperature_group_records'] = oof['calibration_groups']
                oof['fit_authority_sha256'] = semantic_sha256(oof['fit_authority'])
            else:
                row = oof['rows'][0]['expert_evidence']['joint']
                row['log_probability'] = list(reversed(row['log_probability']))
            state = ensemble_fit._beta_uniform_oof_state(oof, programme, prior)
            oof['beta_calibration'] = state
            if attack == 'authorized_C_target':
                with pytest.raises(ValueError, match='original numerical model.*C fit authority') as error:
                    FittedEnsemble(original.scorer, original._experts, original.fit_receipts,
                        original.temperature_record, target_contract=original._target_contract,
                        beta_calibration=state, beta_oof_record=oof)
                rejected[attack] = str(error.value)
                continue
            forged = FittedEnsemble(original.scorer, original._experts, original.fit_receipts,
                original.temperature_record, target_contract=original._target_contract,
                beta_calibration=state, beta_oof_record=oof)
            values = {'fixed': programmes[8]['fixed'], FORWARD_BETA_STRATEGY: forged}
            recorder = JointFlatReplayRecorder(history_captures=captures)
            recorder.record_prefix_programs(values, prefix=prefix, expert_programs=(expert,), program=programme,
                source_checkpoints=sources, context=context)
            recorder.joint_oof = [{**entry, 'row': changed}
                for entry, changed in zip(records[8].joint_oof, oof['rows'], strict=True)]
            for view in views.decisions:
                recorder.score_future(values, view, compiler=compiler, prefix=prefix)
            package = persist_prefix_inner_replay(prefix, values, tmp_path / attack, recorder=recorder,
                source_checkpoints=sources, context=context)
            # The whole posterior, forward rows and every writer seal now
            # agree. Only the actual numerical producer disproves the forgery.
            admitted_forgery = JointFlatPackageAdmission(package)
            assert admitted_forgery.programs[FORWARD_BETA_STRATEGY].beta_calibration['n_decisions'] == 4
            with pytest.raises(ValueError, match='captured original forward OOF probabilities changed') as error:
                validate_joint_flat_public_parent(package, original_views=views,
                    source_cohort=((original_source.compiler, original_source.prefix),))
            rejected[attack] = str(error.value)
        assert not blocked
        (tmp_path / 'verification.json').write_text(json.dumps({'prior': prior, 'n0_and_nonempty': [0, 4],
            'outer_requested': 4, 'future_requested': 1, 'nested_candidate_names': [candidate.name for candidate in candidates],
            'capsule_future_truth': False, 'parent_requested': parent['requested_decisions'],
            'new_fit_or_sampler_calls_during_replay': len(blocked), 'rejected_resealed_attacks': rejected,
            'fully_resealed_raw_OOF_admitted_before_actual_numeric_replay': True,
            'authorized_C_target_rejection_stage': 'original_base_numeric_fit_digest',
            'raw_OOF_rejection_stage': 'actual_parent_original_C_P_numerical_replay',
            'selected_future_scores': selected_scores}, indent=2), encoding='utf-8')
