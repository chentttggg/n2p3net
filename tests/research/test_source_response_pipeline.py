"""Actual BNCI parser to typed finite response, graph, capsule and parent."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def actual_source(tmp_path_factory):
    baseline = os.environ.get('N2P3_SOURCE_RESPONSE_BASELINE')
    if baseline is None:
        from tests.research.test_source_response_states import actual_source as fixture
        from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

        return fixture.__wrapped__(tmp_path_factory,FrozenJointFlatSourcePreparation())
    from data.contract import DEFAULT_P300_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from research.source_response_states import SourceInformedResponseParameters
    from transfer.ensemble_views import CalibrationPrefixCompiler

    original = json.loads(Path(baseline).read_bytes())
    directory = Path(original['directory'])
    def compiler(name, relative):
        content = (directory/name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == original[name]
        return CalibrationPrefixCompiler(content,source_sha256=original[name],source_relative_path=relative,
            budgets=(0,120),profiles=(),authorization_preprocessing=preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
            response_seconds=4/256,pre_roll_seconds=4/256)
    parameters = SourceInformedResponseParameters.load(directory/'response-parameters.json',
        expected_sha256=original['response-parameters.json'])
    return compiler('source-input.mat','A01.mat'),compiler('target-input.mat','A02.mat'),parameters


def _target(path, source_target, *, budgets=(0, 119, 120)):
    from transfer.ensemble_views import prepare_bnci_independent_prefixes

    if not path.exists():
        path.write_bytes(source_target._history_source_payload)
    preprocessing = replace(source_target._arguments['authorization_preprocessing'],
        tmin_ms=-1000*4/256, tmax_ms=1000*4/256, n_times=4)
    return prepare_bnci_independent_prefixes(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        budgets=budgets, profiles=(), authorization_preprocessing=preprocessing,
        response_seconds=4/256, pre_roll_seconds=4/256)


def _runtime_sources():
    checkout = Path(__file__).resolve().parents[2]
    receipts = {}
    for module in list(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename is None:
            continue
        path = Path(filename)
        if path.is_absolute() and path.suffix == '.py' and path.is_file() and path.resolve().is_relative_to(checkout):
            data = path.read_bytes()
            receipts[str(path.resolve())] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return receipts


def test_four_arms_two_laws_original_saved_capsule_parent_and_failures(actual_source, tmp_path, monkeypatch):
    from research.calibration_selection import (
        CalibrationProgramCandidate,
        fit_nested_prefix_selection,
    )
    from research.contracts import semantic_sha256
    from research.finite_response_law import FiniteResponseLaw
    from research.history_artifact import source_response_candidate_evidence
    from research.latent_response_reference import ResponseState
    from research.prefix_inner_replay import persist_prefix_inner_replay
    from research.prefix_replay_execution import _prepare_history_capabilities
    from research.prefix_replay_inputs import export_prepared_views, open_prepared_inputs
    from research.prefix_selection_replay import replay_prefix_selection
    from research.source_response_replay import (
        SourceResponsePackageAdmission,
        SourceResponseReplayRecorder,
        validate_source_response_public_parent,
    )
    from research.source_response_states import (
        SourceInformedFiniteResponseProgram,
        SourceInformedResponseModel,
    )
    from tests.research.test_source_response_states import current_view
    from train.runtime import cpu_thread_budget
    from transfer import ensemble_fit
    from transfer.candidate_decision import row_column_target
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        FittedEnsemble,
        fit_calibration_prefix_programs,
    )
    from transfer.source_response_programme import SourceResponseSourceRef

    source_compiler, source_target, parameters = actual_source
    started = time.monotonic()
    def progress(stage, **values):
        (tmp_path/'PROGRESS.json').write_text(json.dumps({'stage':stage,
            'elapsed_seconds':time.monotonic()-started,**values},indent=2)+'\n',encoding='utf-8')
    progress('actual_source_baseline_loaded')
    path = tmp_path/'A02.mat'
    source = parameters.source
    before_source = source.content
    source_ref = SourceResponseSourceRef('target_participant',
        'new_declared_continuation_of_source_native_arrays', 'zero_native_basis_vector')
    law = FiniteResponseLaw.from_states((ResponseState(1.,1.,0.,1.,.6),
        ResponseState(1.,1.,1/1024,1.25,.4)))
    arms = ('nominal','mean_shape','gaussian_moment','finite_shape')
    blocked = []
    def no_source_fit(*args, **kwargs):
        blocked.append(True)
        raise AssertionError('source response caller refitted the original source')
    monkeypatch.setattr(ContinuousDecisionModel, 'fit', no_source_fit)
    c_initializations = []
    score_operations = []
    original_score = SourceInformedResponseModel.score_history
    def count_actual_score(model, *args, **kwargs):
        before = model.operational_receipt()
        completed = False
        try:
            result = original_score(model,*args,**kwargs)
            completed = True
            return result
        finally:
            after = model.operational_receipt()
            score_operations.append({'actual_K':model.binding['fit_label_event_support'][1],
                'control_kind':model.program.control_kind,'scoring_law':model.program.scoring_law,
                'score_completed':completed,
                **{key:after[key]-before[key] for key in after}})
    monkeypatch.setattr(SourceInformedResponseModel,'score_history',count_actual_score)
    fit_expert = ensemble_fit._fit_expert
    def count_c_fit(spec, *args, **kwargs):
        if spec.family == 'source_response':
            event = {'K':kwargs['original_prefix'].budget,'completed':False,
                'control_kind':spec.source_response_program.control_kind,
                'scoring_law':spec.source_response_program.scoring_law}
            c_initializations.append(event)
        result = fit_expert(spec,*args,**kwargs)
        if spec.family == 'source_response':
            event.update(completed=True,nonempty_C_posterior_constructions=sum(
                posterior is not None for posterior in result.predictor.posteriors))
        return result
    monkeypatch.setattr(ensemble_fit,'_fit_expert',count_c_fit)
    checkpoints = tmp_path/'checkpoints'
    checkpoints.mkdir()
    original_expert_score = ensemble_fit._FittedExpert.score
    score_checkpoints = []
    def checkpoint_expert_score(expert, view, **kwargs):
        ordinal = len(score_checkpoints)
        record = {'ordinal':ordinal,'expert_id':expert.program.expert_id,
            'program':expert.program.record(),'actual_view':view.record(),'completed':False}
        score_checkpoints.append(record)
        if expert.predictor is not None:
            record['calibration_samples'] = expert.predictor.calibration_samples
            saved = checkpoints/(expert.model_digest+'.json')
            if not saved.exists():
                expert.predictor.save(saved)
            record['saved_model'] = {'path':str(saved),'sha256':hashlib.sha256(saved.read_bytes()).hexdigest()}
        for key in ('history_recorder','history_capability_recorder','history_refusal_recorder'):
            original_callback = kwargs.get(key)
            def capture(*values, key=key, original_callback=original_callback):
                record[key] = [value.binding if hasattr(value,'binding') else value for value in values]
                if original_callback is not None:
                    original_callback(*values)
            kwargs[key] = capture
        try:
            result = original_expert_score(expert,view,**kwargs)
            record.update(completed=True,evidence=result.record())
            return result
        except Exception as error:
            record['exception'] = {'type':type(error).__name__,'reason':str(error)}
            raise
        finally:
            record.update(C_initializations=c_initializations,score_operations=score_operations,
                runtime_source_files=_runtime_sources())
            (checkpoints/f'score_{ordinal:03d}.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    monkeypatch.setattr(ensemble_fit._FittedExpert,'score',checkpoint_expert_score)
    all_packages, ordinary, all_specs, cache = [], {}, {}, {}
    main_score_outcomes = []
    with cpu_thread_budget(1):
        views = _target(path, source_target)
        compiler = views.prefix_compiler
        for scoring in ('whole_observed_current_q','fixed_current_prior_conditional_tail'):
            specs = tuple(CalibrationExpertProgram(arm,'source_response',source_ref=source_ref,
                source_response_program=SourceInformedFiniteResponseProgram(law,arm,scoring,100,144,
                    512*1024*1024,64)) for arm in arms)
            all_specs[scoring] = specs
            sources = {spec.expert_id: source for spec in specs}
            programme = EnsembleProgram.equal_weights('four-explicit-response-arms', arms)
            for budget in (120,):
                prefix = compiler.compile(budget)
                # This software caller declares one physical decision per C.
                view = current_view(compiler, 0 if budget == 0 else 1)
                context = {'requested_Ks':[budget], 'requested_decision_ids':[view.record_id],
                    'test_scope':{'start_sample':view.raw_scope[0], 'stop_sample':view.raw_scope[1]}}
                recorder = SourceResponseReplayRecorder()
                progress('fit_and_main_score',scoring_law=scoring,actual_K=budget)
                programs = fit_calibration_prefix_programs(prefix,specs,programme,strategies=('fixed',),
                    source_checkpoints=sources,prefix_compiler=compiler,inner_replay_recorder=recorder,
                    inner_replay_context=context,_prefix_fit_cache=cache)
                model = programs['fixed']
                assert model.fit_available
                first = recorder.score_future(programs,view,compiler=compiler,prefix=prefix)['fixed']
                progress('main_score_complete',scoring_law=scoring,actual_K=budget)
                package = persist_prefix_inner_replay(prefix,programs,tmp_path/f'{scoring}_K{budget}',
                    recorder=recorder,source_checkpoints=sources,context=context)
                failures = {name:value.record() for name,value in first.items() if not value.probability_available}
                assert set(first) == set(arms)
                required_positive = {'nominal','finite_shape'} if scoring == 'whole_observed_current_q' else set()
                assert not required_positive.intersection(failures), json.dumps(failures,indent=2)
                for name,value in first.items():
                    if not value.probability_available:
                        failure = json.loads(value.failure_reason)
                        assert value.failure_kind == 'not_available'
                        assert (failure.get('failure_kind') == 'numerical' or failure.get('type') in {
                            'NumericalResolutionError','DiffuseProjectionResolutionError',
                            'HistoryNuisanceResolutionError','FiniteResponseResolutionError','LinAlgError'})
                        assert name in recorder.predictions[0]['permissions']
                    main_score_outcomes.append({'control_kind':name,'scoring_law':scoring,
                        'probability_available':value.probability_available,
                        'failure_reason':value.failure_reason})
                assert all('resolution' in value for value in recorder.predictions[0]['predictions'].values())
                if scoring == 'whole_observed_current_q':
                    for prediction in recorder.predictions[0]['predictions'].values():
                        assert prediction['evidence_kind'] == 'whole_response'
                        assert {'component_full_certificates','component_errors',
                            'component_primitives',
                            'normalized_log_probability_error_estimate','absolute_log_density_error_budget',
                            'actual_increment_scalar_rows','measure'} <= set(prediction['resolution'])
                        assert prediction['resolution']['P_identification_required'] is False
                        assert prediction['resolution']['conditional_ratio_computed'] is False
                        assert 'numerical_resolution' in prediction['core_prediction']
                        assert semantic_sha256(prediction['resolution']) == semantic_sha256(
                            prediction['core_prediction']['numerical_resolution'])
                        primitives = prediction['resolution']['component_primitives']
                        assert len(primitives) == prediction['core_prediction']['evaluated_positive_states']
                        for primitive in primitives:
                            assert primitive['kind'] == 'normalized_C_full_increment'
                            resolution = primitive['resolution']
                            assert resolution['C_rows_reused_in_update'] == resolution['new_flat_dimensions'] == 0
                            assert resolution['actual_increment_scalar_rows'] == prediction['resolution']['actual_increment_scalar_rows']
                            assert 'full' in resolution and 'source_sha256' in resolution
                reloaded = FittedEnsemble.load(package)
                # A new H>1 law instance must compare by programme identity.
                assert reloaded._experts[-1].program.source_response_program.law is not law
                if scoring == 'whole_observed_current_q':
                    caller = source_response_candidate_evidence(reloaded,'finite_shape',compiler,prefix,view)
                    assert semantic_sha256(caller.record()) == semantic_sha256(first['finite_shape'].record())
                all_packages.append((package,recorder.prefix_receipt_sha256))
                ordinary[scoring,budget] = (package,programs,view,recorder)

        partial = compiler.compile(119)
        partial_view = current_view(compiler,1)
        partial_spec = all_specs['whole_observed_current_q'][-1]
        partial_context = {'requested_Ks':[119],'requested_decision_ids':[partial_view.record_id],
            'test_scope':{'start_sample':partial_view.raw_scope[0],'stop_sample':partial_view.raw_scope[1]}}
        partial_recorder = SourceResponseReplayRecorder()
        partial_programs = fit_calibration_prefix_programs(partial,(partial_spec,),
            EnsembleProgram.equal_weights('partial-C-finite',('finite_shape',)),strategies=('fixed',),
            source_checkpoints={'finite_shape':source},prefix_compiler=compiler,
            inner_replay_recorder=partial_recorder,inner_replay_context=partial_context,_prefix_fit_cache=cache)
        partial_scored = partial_recorder.score_future(partial_programs,partial_view,compiler=compiler,prefix=partial)
        progress('partial_C119_complete')
        assert partial_scored['fixed']['finite_shape'].probability_available
        partial_package = persist_prefix_inner_replay(partial,partial_programs,tmp_path/'partial_C119',
            recorder=partial_recorder,source_checkpoints={'finite_shape':source},context=partial_context)
        all_packages.append((partial_package,partial_recorder.prefix_receipt_sha256))

        specs = all_specs['whole_observed_current_q']
        prefix = compiler.compile(120)
        future = current_view(compiler,1)
        rows = np.flatnonzero(source_target._record.selection_index == 1)
        truth = row_column_target(int(source_target._record.target_row[rows[0]]),
            int(source_target._record.target_col[rows[0]]))
        scoped = replace(views,decisions=(future,),scoring_truth={future.decision_id:truth})
        context = {'requested_decision_ids':[future.record_id],
            'test_scope':{'start_sample':future.raw_scope[0], 'stop_sample':future.raw_scope[1]}}
        candidates = tuple(CalibrationProgramCandidate(arm,(spec,),EnsembleProgram.equal_weights(arm,(arm,)))
                           for arm,spec in zip(arms,specs,strict=True))
        selected = fit_nested_prefix_selection(prefix,candidates,baseline_name='nominal',
            prefix_compiler=compiler,minimum_selection_rows=1,source_checkpoints={arm:source for arm in arms},
            context=context,_prefix_fit_cache=cache)
        selected.score(future)
        graph = selected.save(tmp_path/'nested')
        progress('nested_saved')
        assert len(selected.record['outer_rows']) == 1
        all_packages.extend((tmp_path/'nested/models'/key,node['recorder'].prefix_receipt_sha256)
                            for key,node in selected._fits.nodes.items())

        # Real unavailable C and P programmes retain their requested output.
        failure_cases = (
            ('C_budget', replace(specs[0],source_response_program=replace(specs[0].source_response_program,
                maximum_working_bytes=1)),120),
            ('P_budget', replace(specs[0],source_response_program=replace(specs[0].source_response_program,
                maximum_prediction_states=1)),120),
            ('tiny_mean_mass', replace(specs[1],source_response_program=replace(specs[1].source_response_program,
                law=FiniteResponseLaw.from_states((ResponseState(1.,1.,0.,1.,np.nextafter(0.,1.)),
                    ResponseState(1.,1.,0.,1.,np.finfo(float).max))))),120),
            ('conditional_P1',replace(specs[0],source_response_program=replace(specs[0].source_response_program,
                scoring_law='fixed_current_prior_conditional_tail',conditioning_offset_samples=-15)),0))
        for name,spec,budget in failure_cases:
            recorder = SourceResponseReplayRecorder()
            case_view = current_view(compiler,0) if budget == 0 else future
            case_context = {'requested_Ks':[budget],'requested_decision_ids':[case_view.record_id],
                'test_scope':{'start_sample':case_view.raw_scope[0],'stop_sample':case_view.raw_scope[1]}}
            programmes = fit_calibration_prefix_programs(compiler.compile(budget),(spec,),
                EnsembleProgram.equal_weights(name,(spec.expert_id,)),strategies=('fixed',),
                source_checkpoints={spec.expert_id:source},prefix_compiler=compiler,
                inner_replay_recorder=recorder,inner_replay_context=case_context)
            failed = recorder.score_future(programmes,case_view,compiler=compiler,prefix=compiler.compile(budget))
            assert not failed['fixed'][spec.expert_id].probability_available
            assert (programmes['fixed']._experts[0].predictor is None) == (name in {'C_budget','tiny_mean_mass'})
            assert bool(recorder.original_history_capabilities) == (name in {'P_budget','conditional_P1'})
            package = persist_prefix_inner_replay(compiler.compile(budget),programmes,tmp_path/name,
                recorder=recorder,source_checkpoints={spec.expert_id:source},context=case_context)
            all_packages.append((package,recorder.prefix_receipt_sha256))
        progress('typed_failure_packages_saved')

        # Fresh original preparation reissues only real caps, without a model fit.
        blocked_c = []
        original_fit = ensemble_fit._fit_expert
        def no_c_fit(*args, **kwargs):
            blocked_c.append(True)
            raise AssertionError('saved source response replay attempted a new target fit')
        monkeypatch.setattr(ensemble_fit,'_fit_expert',no_c_fit)
        fresh = _target(path,source_target)
        fresh = replace(fresh,decisions=(future,),scoring_truth=scoped.scoring_truth)
        histories,refusals = _prepare_history_capabilities(fresh,(0,119,120),all_packages)
        request = {'schema':'n2p3_prefix_replay_input_request/1','preparation_task_id':'source-response-software',
            'original_source_sha256':fresh.ledger['source_sha256'],
            'producer_source_sha256':fresh.prefix_compiler.history_owner_source,
            'participant_key':prefix.target_contract['participant_key'],
            'source_relative_path':prefix.target_contract['source_relative_path'],
            'requested_Ks':[0,119,120],'actual_Ks':[0,119,120],'profiles':[],
            'authorization_preprocessing':fresh.prefix_compiler._arguments['authorization_preprocessing'].record(),
            'observation_response_seconds':4/256,'pre_roll_seconds':4/256}
        capsule = export_prepared_views(fresh,(),tmp_path/'capsule',request,
            record_history_scopes=histories,record_history_refusals=refusals)
        descriptor = {'schema':'n2p3_prefix_replay_shared_inputs/1','backing':'immutable_bytes',
            'capsule':(tmp_path/'capsule/capsule.json').read_bytes(),'payload':(tmp_path/'capsule/arrays.bin').read_bytes(),
            'capsule_sha256':capsule['sha256'],'request_sha256':semantic_sha256(request)}
        with open_prepared_inputs(descriptor,expected_request_sha=semantic_sha256(request),
                expected_capsule_sha=capsule['sha256']) as borrowed:
            assert not borrowed.views.scoring_truth
            for package,digest in [(partial_package,partial_recorder.prefix_receipt_sha256),*all_packages[-4:]]:
                admission = SourceResponsePackageAdmission(package,expected_record_sha256=digest)
                saved_view = admission.record['future'][0]['view']
                candidates_by_record = [*borrowed.views.decisions,
                    *(g.view for p in borrowed.views.calibration.values() for g in p.complete_groups)]
                actual_view = next(v for v in candidates_by_record if semantic_sha256(v.record()) == semantic_sha256(saved_view))
                result = admission.replay(prefix=borrowed.views.prefix_compiler.compile(admission.record['actual_K']),
                    prefix_compiler=borrowed.views.prefix_compiler,future_views=(actual_view,))
                assert len(result['scores']) == 1
            replay = replay_prefix_selection(graph,prefix=borrowed.views.calibration[120],
                prefix_compiler=borrowed.views.prefix_compiler,future_views=borrowed.views.decisions,
                source_checkpoints={arm:source for arm in arms},
                expected_selection_sha256=semantic_sha256(selected.record),expected_graph_sha256=selected.graph_sha256)
            assert replay['zero_new_fits']
        progress('prepared_replay_complete')
        parent_package = partial_package
        parent = validate_source_response_public_parent(parent_package,original_views=fresh,
            source_cohort=((source_compiler,source_compiler.compile(120)),))
        assert parent['requested_decisions'] == 1 and 'available_probability_nll' in parent['metrics']['fixed']
        conditional_parent = validate_source_response_public_parent(
            ordinary['fixed_current_prior_conditional_tail',120][0],original_views=fresh,
            source_cohort=((source_compiler,source_compiler.compile(120)),))
        assert conditional_parent['requested_decisions'] == 1
        failed_parent = validate_source_response_public_parent(tmp_path/'P_budget',original_views=fresh,
            source_cohort=((source_compiler,source_compiler.compile(120)),))
        assert failed_parent['metrics']['fixed']['probability_coverage'] == 0.
        attacked = tmp_path/'resealed-prediction'
        shutil.copytree(parent_package,attacked)
        payload_path = attacked/'prefix_inner_replay.json'
        payload = json.loads(payload_path.read_bytes())
        prediction = payload['record']['future'][0]['predictions']['finite_shape']
        prediction['core_prediction']['log_probability'] = list(reversed(prediction['core_prediction']['log_probability']))
        payload['record_sha256'] = semantic_sha256(payload['record'])
        payload_path.write_text(json.dumps(payload),encoding='utf-8')
        SourceResponsePackageAdmission(attacked)
        fresh_parent = _target(path,source_target)
        fresh_parent = replace(fresh_parent,decisions=(current_view(fresh_parent.prefix_compiler,1),),
            scoring_truth=scoped.scoring_truth)
        with pytest.raises(ValueError,match='original actual permission, C state or probability'):
            validate_source_response_public_parent(attacked,original_views=fresh_parent,
                source_cohort=((source_compiler,source_compiler.compile(120)),))
        assert not blocked and not blocked_c and source.content == before_source
        whole_primitive_cases = []
        for checkpoint in score_checkpoints:
            callback = checkpoint.get('history_recorder')
            if callback is None or callback[2]['evidence_kind'] != 'whole_response':
                continue
            prediction = callback[2]
            primitive_kind = 'normalized_C_full_increment' if checkpoint['calibration_samples'] else 'common_flat_full_potential'
            primitives = prediction['resolution']['component_primitives']
            assert len(primitives) == prediction['core_prediction']['evaluated_positive_states']
            assert all(primitive['kind'] == primitive_kind for primitive in primitives)
            assert all(('full' if checkpoint['calibration_samples'] else 'C') in primitive['resolution']
                and 'source_sha256' in primitive['resolution'] for primitive in primitives)
            whole_primitive_cases.append({'calibration_samples':checkpoint['calibration_samples'],
                'primitive_kind':primitive_kind,'count':len(primitives)})
        assert {row['primitive_kind'] for row in whole_primitive_cases} == {
            'normalized_C_full_increment','common_flat_full_potential'}
        monkeypatch.setattr(ensemble_fit,'_fit_expert',original_fit)
        (tmp_path/'PIPELINE_RESULT.json').write_text(json.dumps({'status':'PASS','original_genesis_source_fits':1,
            'source_fits_this_invocation':int(os.environ.get('N2P3_SOURCE_RESPONSE_BASELINE') is None),
            'post_source_parameter_refits':0,'sampler_calls':0,'C_initializations':c_initializations,
            'nonempty_C_posterior_constructions':sum(row.get('nonempty_C_posterior_constructions',0)
                for row in c_initializations),
            'score_operational_events':score_operations,
            'score_operational_totals':{key:sum(row[key] for row in score_operations)
                for key in ('C_QR_reconstruction_attempts','C_QR_reconstructions','C_verification_cache_hits')},
            'reader_C_fits':0,'arms':arms,'scoring_laws':list(all_specs),
            'main_score_outcomes':main_score_outcomes,
            'whole_primitive_cases':whole_primitive_cases,
            'main_requested_probability_rows':len(main_score_outcomes),
            'main_probability_available_rows':sum(row['probability_available'] for row in main_score_outcomes),
            'ordinary_main_cases':2,'partial_C119_case':1,'nested_outer_requested':1,
            'failure_cases':[row[0] for row in failure_cases],
            'runtime_source_files':_runtime_sources(),'elapsed_seconds':time.monotonic()-started,
            'source_bytes_unchanged':True,'capsule_truth_free':True,'canonical_parent_metrics':parent['metrics'],
            'main_conditional_parent_metrics':conditional_parent['metrics']},
            indent=2)+'\n',encoding='utf-8')


def test_original_source_catalog_and_generic_plan_contract(actual_source, tmp_path, monkeypatch):
    from data.bnci2014_008_candidate import BNCI2014_008_DATASET_ID
    from experiments.run_candidate_ensemble_study import (
        _joint_sources_for_target,
        _prepare_candidate_plan,
    )
    from research import joint_source_catalog
    from research.joint_source_catalog import _verify_original_cohort
    from research.source_response_states import SourceInformedFiniteResponseProgram
    from tests.research.test_source_response_states import programme
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import CalibrationExpertProgram
    from transfer.source_response_programme import SourceResponseSourceRef

    source_compiler, source_target, parameters = actual_source
    target_path = tmp_path/'A02.mat'
    views = _target(target_path,source_target,budgets=(120,))
    source = parameters.source
    def artifact(path):
        content = path.read_bytes()
        return {'path':str(path),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    codec, original = tmp_path/'source-codec.json',tmp_path/'source-original.npz'
    codec.write_bytes(source.content)
    original.write_bytes(source.original_model_content)
    def forbidden(*args, **kwargs):
        raise AssertionError('source catalogue or metadata plan attempted a new source fit')
    monkeypatch.setattr(ContinuousDecisionModel,'fit',forbidden)
    prefix = views.calibration[120]
    import io
    from types import SimpleNamespace

    summary = _verify_original_cohort(source,ContinuousDecisionModel.load(io.BytesIO(source.original_model_content)),
        ((source_compiler,source_compiler.compile(120)),),prefix.target_contract['participant_key'])
    assert len(summary['source_records']) == 1
    # This baseline predates a source-stage event receipt. Exercise new-family
    # routing at that boundary without inventing an original PID or fit event.
    entry = {'source_id':'actual-software-source','codec':artifact(codec),
        'exclusion':{'target_participant_key':prefix.target_contract['participant_key']}}
    routed = []
    class ExistingSourceRouting:
        def __init__(self, entries, **kwargs):
            assert entries == [entry]
        def resolve(self, source_id, *, target_contract, exclusion_scope, pipeline_pair):
            assert source_id == entry['source_id'] and exclusion_scope == 'target_participant'
            source.assert_target(target_contract)
            routed.append((source_id,exclusion_scope,pipeline_pair))
            return SimpleNamespace(source=source,receipt={'actual_source_seed':None,'resolution_source_fits':0})
    monkeypatch.setattr(joint_source_catalog,'JointFlatSourceCatalog',ExistingSourceRouting)
    entry_path = tmp_path/'source-entry.json'
    entry_path.write_text(json.dumps(entry),encoding='utf-8')
    source_ref = SourceResponseSourceRef('target_participant',
        'new_declared_continuation_of_source_native_arrays','zero_native_basis_vector')
    specs = tuple(CalibrationExpertProgram(arm,'source_response',
        source_response_program=replace(programme(arm),conditioning_offset_samples=64),source_ref=source_ref)
        for arm in ('nominal','mean_shape','gaussian_moment','finite_shape'))
    plan = {'schema':'n2p3_candidate_ensemble_study/1','evidence_scope':'software_correctness',
        'dataset_id':BNCI2014_008_DATASET_ID,'evaluation_design':'independent_prefix',
        'records':[artifact(target_path)],'budgets':[120],'seed_pairs':[[0,0]],
        'authorization_preprocessing':views.prefix_compiler._arguments['authorization_preprocessing'].record(),
        'observation_response_seconds':4/256,'pre_roll_seconds':4/256,
        'experts':[spec.record() for spec in specs],
        'mixtures':[EnsembleProgram.equal_weights(spec.expert_id,(spec.expert_id,)).record() for spec in specs],
        'baseline_mixture':'nominal','calibration_strategies':['fixed'],'source_checkpoints':{},
        'source_seed_catalog':[],'joint_source_catalog':[{'source_id':entry['source_id'],**artifact(entry_path)}],
        'joint_source_bindings':{prefix.target_contract['participant_key']:{spec.expert_id:entry['source_id'] for spec in specs}},
        'joint_source_data_admission':{},'device':'cpu','workers':1,'cpu_threads_per_worker':1,
        'prefix_replay_chunk_size':1}
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan),encoding='utf-8')
    normalized,_,snapshots = _prepare_candidate_plan(plan_path,load_sources=False)
    assert not snapshots
    sources,receipts = _joint_sources_for_target(normalized,specs,prefix.target_contract,pipeline_pair=(0,0))
    assert len(routed) == len(specs)
    assert set(sources) == {spec.expert_id for spec in specs}
    assert all(value.digest == source.digest for value in sources.values())
    assert all(row['actual_source_seed'] is None and row['resolution_source_fits'] == 0 for row in receipts.values())
    for name,change in (
            ('wrong_dataset',{'dataset_id':'Huebner2017'}),
            ('unimplemented_calibration',{'calibration_strategies':['fixed','forward_beta_uniform']}),
            ('mixed_family',{'experts':plan['experts']+[CalibrationExpertProgram('raw','continuous_ar').record()]})):
        candidate = {**plan,**change}
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps(candidate),encoding='utf-8')
        with pytest.raises(ValueError,match='pure finite-family BNCI'):
            _prepare_candidate_plan(path,load_sources=False)
    assert isinstance(specs[0].source_response_program,SourceInformedFiniteResponseProgram)
    (tmp_path/'GENERIC_METADATA_RESULT.json').write_text(json.dumps({'status':'PASS',
        'actual_registered_BNCI_source_cohort_verified':True,'source_role':'original_source_response_and_noise_parameters',
        'source_refits':0,'target_fits':0,'source_randomness':'deterministic','software_evidence_preserved':True,
        'generic_normalization_and_typed_source_resolution':True,
        'catalogue_historical_stage_boundary':'recording_seam_no_invented_PID_or_source_saved_event',
        'original_source_stage_admission_executed':False,'full_generic_run_executed':False,
        'runtime_source_files':_runtime_sources(),
        'unsupported_plan_attacks':3},indent=2)+'\n',encoding='utf-8')
