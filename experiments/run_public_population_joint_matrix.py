"""Complete admitted population -> joint C fit -> original requested held matrix.

The source programme and its independent acceptance are separate inputs. This
driver composes existing issuers, numerical owners, replay and requested metrics;
it does not fit a source, mint acceptance, or reuse a target-C warm start.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = 'experiments.run_public_population_joint_matrix'
KS = (200, 800, 2000)
ARMS = ('G0', 'G1')


def artifact(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def read(reference):
    payload = Path(reference['path']).read_bytes()
    if (len(payload) != reference['bytes']
            or hashlib.sha256(payload).hexdigest() != reference['sha256']):
        raise ValueError('the caller-pinned matrix input changed: '+reference['path'])
    return json.loads(payload)


def pinned(path, expected):
    reference = artifact(path)
    if reference['sha256'] != expected:
        raise ValueError('the caller-pinned matrix digest differs: '+str(path))
    return reference


def write_new(path, record):
    from research.execution import publish_exclusive

    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    publish_exclusive(path, lambda stream: stream.write(payload), byte_size=len(payload))
    return artifact(path)


def failure_record(error):
    """Diagnostic nonfinite values retain explicit symbols; no fit is issued."""
    from research.frozen_metadata import plain_metadata

    def encode(value):
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return 'not_a_number' if math.isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity'
        return value

    result = {'error_type': type(error).__name__, 'reason': str(error)}
    if hasattr(error, 'record'):
        result['original_optimizer_record'] = encode(plain_metadata(error.record))
    return result


def matrix_context(recipe_reference):
    """Metadata-only scope: counts, exact other-person folds and original refs."""
    recipe = read(recipe_reference)
    if (recipe['schema'] != 'public_population_joint_matrix_recipe/1'
            or tuple(recipe['K']) != KS or tuple(recipe['arms']) != ARMS
            or recipe['joint_fit'] != {'maximum_iterations': 2000, 'gradient_tolerance': .001,
                'conditioning_samples': 32, 'block_native_rows': 2048}
            or recipe['inference_configuration']['inference_seed'] != 44719
            or recipe.get('initial_receipt') is not None):
        raise ValueError('the complete original G0/G1 programme and source-only initialization are required')
    source_recipe = read(recipe['source_recipe'])
    matrix_reference = source_recipe['source_matrix']
    matrix = read(matrix_reference)
    people = {row['source_person_id']: row for row in matrix['person_programmes']}
    folds = {row['target_participant_key']: row for row in matrix['fold_programmes']}
    if (matrix['schema'] != 'public_population_source_execution_matrix_candidate/1'
            or len(people) != 13 or len(folds) != 13 or set(people) != set(folds)
            or len(matrix['person_programmes']) != 13 or len(matrix['fold_programmes']) != 13
            or matrix['pending_new_gaussian_score_cells'] != 4914
            or source_recipe['programme_seed'] != matrix['programme_seed']):
        raise ValueError('the exact thirteen-person/thirteen-fold source matrix is required')
    targets = []
    for key, person in people.items():
        fold = folds[key]
        sources = fold['source_person_references']
        if (len(sources) != 12 or set(sources) != set(people)-{key}
                or fold['source_person_count'] != 12 or tuple(fold['target_C_K']) != KS
                or fold['future_requested_characters'] != 63 or fold['source_read_session2'] is not False):
            raise ValueError('each actual target uses exactly the other twelve source people and all 63 held requests')
        targets.append({'participant_key': key, 'fold_id': fold['fold_id'],
            'declaration': person['declaration'], 'requested_held': 63})
    return recipe, source_recipe, matrix_reference, matrix, tuple(targets)


def verify_runtime_source(reference):
    from research.evaluation import verified_executed_snapshot

    read(reference)
    return verified_executed_snapshot(reference['path'], root=ROOT,
        required_files=('experiments/run_public_population_joint_matrix.py',))


def original_target_roster(adapter, *, include_truth=False):
    """Original adapter facts only; no C compile or held sample materialization.

    Adapter construction itself hashes original BDFs and reads headers/markers.
    Panel preparation is therefore an explicitly executed input-preparation
    stage, independent of source fitting, rather than a zero-input-read probe.
    """
    requested = adapter.requested_future_record_ids
    if not requested:
        raise ValueError('the original adapter must declare its independent future recordings')
    rows, truths = [], []
    matched = set()
    for source in adapter.recordings:
        if source.record_id not in requested:
            continue
        matched.add(source.record_id)
        for ordinal, selection in enumerate(source.selections):
            decision = selection.decision
            expected = [int(value) for value in decision.membership.sum(axis=0)]
            rows.append({'record_id': source.record_id, 'source_sha256': source.source_sha256,
                'ordinal': ordinal, 'decision_id': selection.selection_id,
                'candidate_vocabulary': list(decision.candidate_vocabulary),
                'evidence_level': decision.repetitions, 'expected_event_counts': expected})
            if include_truth:
                truths.append({'participant_key': adapter.participant_key,
                    'decision_id': selection.selection_id, 'target_candidate': selection.parent_target_candidate})
    if matched != set(requested) or len(rows) != 63 or len({row['decision_id'] for row in rows}) != 63:
        raise ValueError('the complete original future roster must retain all 63 decisions, including unavailable ones')
    if rows[0]['ordinal'] != 0:
        raise ValueError('the first requested held view is the first selection of its original recording')
    return rows, truths


def prepare_requested_panel(arguments):
    from data.candidate_input import resolve_candidate_dataset_adapter
    from research.contracts import DecisionPlanContract, DecisionPlanEntry, semantic_sha256
    from research.resource_scope import require_managed_phase

    require_managed_phase(arguments['resources'], phase='preparation', device='cpu')
    verify_runtime_source(arguments['source_snapshot'])
    _, _, matrix_reference, _, targets = matrix_context(arguments['recipe'])
    requested, truths, identities = {}, [], []
    vocabulary = None
    for target in targets:
        adapter = resolve_candidate_dataset_adapter(read(target['declaration']))
        if adapter.participant_key != target['participant_key']:
            raise ValueError('the original target adapter differs from the frozen participant identity')
        rows, labels = original_target_roster(adapter, include_truth=True)
        local = {tuple(row['candidate_vocabulary']) for row in rows}
        if len(local) != 1 or vocabulary is not None and next(iter(local)) != vocabulary:
            raise ValueError('the original target panel has a different complete vocabulary')
        vocabulary = next(iter(local))
        requested[target['participant_key']] = {'fold_id': target['fold_id'],
            'declaration': target['declaration'], 'adapter_identity': adapter.digest, 'requests': rows}
        truths.extend(DecisionPlanEntry(**label) for label in labels)
        identities.append({'participant_key': target['participant_key'], 'adapter_identity': adapter.digest})
    plan = DecisionPlanContract(semantic_sha256([row['declaration'] for row in targets]),
        semantic_sha256(identities), tuple(requested), tuple(truths))
    record = {'schema': 'original_population_joint_requested_panel/1', 'source_matrix': matrix_reference,
        'targets': requested, 'decision_plan': plan.record(), 'candidate_vocabulary': list(vocabulary),
        'requested_participants': 13, 'requested_decisions': 819, 'K': list(KS), 'arms': list(ARMS),
        'requested_new_cells': 4914, 'unavailable_requests_removed': False,
        'truth_role': 'controller_evaluation_only_not_fit_or_prediction_arguments',
        'preparation': {'original_input_hash_header_marker_reads': True, 'read_native_calls': 0,
            'source_fits': 0, 'C_fits': 0, 'held_views_materialized': 0},
        'source_snapshot': arguments['source_snapshot']}
    output = Path(arguments['output'])
    output.mkdir(parents=True, exist_ok=False)
    return write_new(output/'REQUESTED_PANEL.json', record)


def target_runtime():
    from types import SimpleNamespace

    from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
    from research.channel_noise_mechanism import NoiseGeometry
    from research.collapsed_continuous_prediction import InferenceQualityPolicy
    from research.contracts import semantic_sha256
    from research.execution import UnidentifiableFitError, atomic_write_json
    from research.frozen_metadata import plain_metadata
    from research.joint_response_noise_calibration import JointCalibrationOptimizationError
    from research.joint_response_scale import NumericalResolutionError
    from research.public_joint_history_execution import consume_frozen_public_joint_history
    from research.public_joint_response_noise import (
        PublicJointFit,
        admit_public_joint_fit,
        fit_public_joint_calibration,
        prepare_public_joint_calibration,
        prepare_public_joint_reference,
        require_public_joint_fit,
    )
    from research.public_population_source import (
        admit_public_population_source,
        load_population_native_response,
        load_source_population_execution,
    )
    from research.public_record_physical_bridge import (
        PublicPhysicalUnavailable,
        admit_public_calibration,
        admit_public_history,
        require_public_physical_input,
        require_public_unavailable,
    )
    from transfer.binary import calibration_budget_requests
    from transfer.ensemble_views import CalibrationPrefixCompiler
    from transfer.native_response_dictionary import NativeResponseDictionary

    return SimpleNamespace(CandidateEvidence=CandidateEvidence, CandidateExpertIdentity=CandidateExpertIdentity,
        NoiseGeometry=NoiseGeometry, InferenceQualityPolicy=InferenceQualityPolicy,
        semantic_sha256=semantic_sha256, atomic_write_json=atomic_write_json, plain_metadata=plain_metadata,
        consume_frozen_public_joint_history=consume_frozen_public_joint_history, PublicJointFit=PublicJointFit,
        admit_public_joint_fit=admit_public_joint_fit, fit_public_joint_calibration=fit_public_joint_calibration,
        prepare_public_joint_calibration=prepare_public_joint_calibration,
        prepare_public_joint_reference=prepare_public_joint_reference, require_public_joint_fit=require_public_joint_fit,
        admit_public_population_source=admit_public_population_source,
        load_population_native_response=load_population_native_response,
        load_source_population_execution=load_source_population_execution,
        PublicPhysicalUnavailable=PublicPhysicalUnavailable, admit_public_calibration=admit_public_calibration,
        admit_public_history=admit_public_history, require_public_physical_input=require_public_physical_input,
        require_public_unavailable=require_public_unavailable, calibration_budget_requests=calibration_budget_requests,
        CalibrationPrefixCompiler=CalibrationPrefixCompiler, NativeResponseDictionary=NativeResponseDictionary,
        expected_fit_errors=(JointCalibrationOptimizationError, UnidentifiableFitError, NumericalResolutionError),
        expected_score_errors=(UnidentifiableFitError, NumericalResolutionError))


def programme_identity(arguments, requested_K, arm):
    return {'participant_key': arguments['target']['participant_key'], 'fold_id': arguments['target']['fold_id'],
        'requested_K': requested_K, 'arm': arm, 'recipe': arguments['recipe'],
        'source_execution': arguments['source_execution'], 'source_acceptance': arguments['source_acceptance'],
        'source_snapshot': arguments['source_snapshot'], 'requested_panel_sha256': arguments['requested_panel_sha256']}


def unavailable_evidence(runtime, fit, view, reason):
    expected = tuple(int(value) for value in view.decision.membership.sum(axis=0))
    observed = tuple(int(value) for value in view.decision.membership[
        view.decision.evidence_indices >= 0].sum(axis=0))
    numerical = runtime.semantic_sha256({'fit': fit.identity, 'raw_input': view.raw_input_digest,
        'reason': reason, 'role': 'explicit_unavailable_output_without_probability'})
    expert = runtime.CandidateExpertIdentity(f'joint-history-{fit.arm}', numerical, fit.identity,
        fit.source.identity, fit.identity, 'public_joint_history/not_available/1', False)
    return runtime.CandidateEvidence(view.decision.candidate_vocabulary, None, view.participant_key,
        view.decision_id, view.physical_support_digest, (expert,), view.decision.repetitions,
        expected, observed, failure_reason=reason, failure_kind='not_available')


def score_one_request(runtime, fit, compiler, prefix, row, *, recipe, directory, identity):
    """Materialize exactly one requested view, then use the existing consumer."""
    source = next(source for source in compiler.sources if source.record_id == row['record_id'])
    if source.source_sha256 != row['source_sha256']:
        raise ValueError('the current original held acquisition differs from its frozen request')
    view = compiler._current_view(source, row['ordinal'])
    if (view.participant_key != identity['participant_key'] or view.decision_id != row['decision_id']
            or tuple(view.decision.candidate_vocabulary) != tuple(row['candidate_vocabulary'])
            or view.decision.repetitions != row['evidence_level']):
        raise ValueError('the original requested view changed its participant/selection/vocabulary/horizon')
    prediction_reference = receipt_reference = None
    refusal = error_record = None
    evidence = None
    if view.samples is None:
        evidence = unavailable_evidence(runtime, fit, view, view.unavailable_reason)
    else:
        history = runtime.admit_public_history(compiler, prefix, view,
            conditioning_stop=int(view.raw_scope[0]))
        if isinstance(history, runtime.PublicPhysicalUnavailable):
            refusal = runtime.plain_metadata(runtime.require_public_unavailable(history).record)
            evidence = unavailable_evidence(runtime, fit, view, refusal['reason'])
        else:
            history = runtime.require_public_physical_input(history)
            policy = recipe['quality_policy']
            quality = runtime.InferenceQualityPolicy(policy['policy_id'], tuple(policy['allowed_estimate_kinds']),
                policy['maximum_replicate_log_se'])
            try:
                prediction, _, receipt_sha = runtime.consume_frozen_public_joint_history(fit,
                    current_view=view, history_input=history, output=directory,
                    inference_configuration=recipe['inference_configuration'], quality_policy=quality)
            except runtime.expected_score_errors as error:
                error_record = failure_record(error)
                evidence = unavailable_evidence(runtime, fit, view, str(error))
            else:
                prediction_reference = artifact(directory/'prediction.json')
                receipt_reference = artifact(directory/'HISTORY_CONSUMER_RECEIPT.json')
                if receipt_reference['sha256'] != receipt_sha:
                    raise ValueError('the completed held consumer receipt changed before matrix publication')
                evidence = prediction.evidence
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'population_joint_requested_cell/1', 'identity': identity,
        'decision_id': row['decision_id'], 'fit_identity': fit.identity,
        'typed_prediction': prediction_reference, 'history_receipt': receipt_reference,
        'unavailable_evidence': None if prediction_reference else evidence.record(),
        'original_public_refusal': refusal, 'numerical_failure': error_record,
        'status': evidence.status, 'truth_supplied_to_prediction': False}
    return write_new(directory/'OUTCOME.json', record)


def fit_and_score_programme(runtime, inputs, *, compiler, prefix, recipe, arguments, requested_K, arm, directory):
    identity = programme_identity(arguments, requested_K, arm)
    directory.mkdir(parents=True, exist_ok=False)
    status = {'schema': 'population_joint_C_programme/1', 'identity': identity,
        'actual_K': prefix.budget, 'original_label_use': runtime.plain_metadata(prefix.label_use),
        'status': 'calibration_running', 'initialization': 'own_frozen_source_only_noise_and_G1_x0',
        'source_execution_identity': arguments['source_execution_identity'], 'requested_held': 63,
        'maximum_iterations': 2000, 'gradient_tolerance': .001, 'target_C_warm_start_reused': False,
        'fit_owner_called': True, 'cell_publications': []}
    runtime.atomic_write_json(directory/'FIT_REPORT.json', status)

    def checkpoint(row):
        evaluation = row['valid_trial']['evaluation']
        if evaluation == 1 or evaluation % 25 == 0:
            runtime.atomic_write_json(directory/'optimization-checkpoint.json', runtime.plain_metadata(row))

    try:
        fit = runtime.fit_public_joint_calibration(inputs, arm, maximum_iterations=2000,
            gradient_tolerance=.001, initial=None, initial_binding=None, evaluation_callback=checkpoint)
    except runtime.expected_fit_errors as error:
        status.update(status='fit_failure', failure=failure_record(error))
        runtime.atomic_write_json(directory/'FIT_REPORT.json', status)
        return status
    path = directory/'joint-fit.npz'
    digest = fit.save(path)
    numeric = runtime.PublicJointFit.load(path, expected_sha256=digest, reference=inputs.reference)
    try:
        runtime.require_public_joint_fit(numeric)
    except TypeError:
        pass
    else:
        raise AssertionError('numeric cold joint state granted live held authority')
    live = runtime.admit_public_joint_fit(numeric, reference=inputs.reference, prefix=prefix,
        calibration_inputs=inputs.calibration_inputs, calibration_programme=inputs)
    runtime.require_public_joint_fit(live)
    if live.identity != fit.identity:
        raise ValueError('the saved/reloaded original-C readmission changed the actual complete fit')
    # The admitted live wrapper owns the cold payload; the original fit is retired.
    del fit, numeric
    status.update(status='fitted_and_readmitted', fit=artifact(path), fit_identity=live.identity,
        full_shared_dimensions=live.source.rank, private_noise_channels=len(live.noise.phi),
        full_innovation_covariance_shape=list(live.noise.innovation_covariance.shape),
        complete_within_factor_shape=list(live.within_factor.shape), completed_held=0)
    runtime.atomic_write_json(directory/'FIT_REPORT.json', status)
    for ordinal, row in enumerate(arguments['requested_roster']):
        publication = score_one_request(runtime, live, compiler, prefix, row, recipe=recipe,
            directory=directory/'held'/f'{ordinal:03}', identity=identity)
        status['cell_publications'].append(publication)
        status['completed_held'] = ordinal+1
        runtime.atomic_write_json(directory/'FIT_REPORT.json', status)
    status['status'] = 'completed_requested_programme'
    runtime.atomic_write_json(directory/'FIT_REPORT.json', status)
    return status


def run_budget(runtime, native, source_execution, *, recipe, source_recipe, arguments, requested_K, directory):
    """One compiler lifetime per K, shared by its G0/G1 programmes only."""
    from data.epochs import PreprocessingSpec

    compiler = runtime.CalibrationPrefixCompiler.from_dataset_source(read(arguments['target']['declaration']),
        budgets=(requested_K,), profiles=(),
        authorization_preprocessing=PreprocessingSpec(**source_recipe['authorization_preprocessing']),
        response_seconds=source_recipe['response_seconds'], pre_roll_seconds=source_recipe['pre_roll_seconds'])
    actual_roster, _ = original_target_roster(compiler.adapter)
    if compiler.adapter.participant_key != arguments['target']['participant_key'] or actual_roster != arguments['requested_roster']:
        raise ValueError('the actual original target roster differs from the frozen requested panel')
    budgets = runtime.calibration_budget_requests(compiler.budgets, compiler.actual_budgets, compiler._adjustments)
    prefix = compiler.compile(budgets[requested_K]['actual_K'])
    directory.mkdir(parents=True, exist_ok=False)
    raw_use = prefix.label_use['raw']
    if raw_use['used_labels'] != prefix.budget or raw_use.get('unavailable_records'):
        for arm in ARMS:
            parent = directory/arm
            parent.mkdir()
            write_new(parent/'FIT_REPORT.json', {'schema': 'population_joint_C_programme/1',
                'identity': programme_identity(arguments, requested_K, arm), 'actual_K': prefix.budget,
                'status': 'calibration_unavailable', 'requested_held': 63,
                'original_label_use': runtime.plain_metadata(prefix.label_use),
                'reason': 'complete_original_C_observations_unavailable', 'fit_attempted': False})
        return
    caps = runtime.admit_public_calibration(compiler, prefix)
    admission = runtime.admit_public_population_source(native, compiler, (prefix,), caps, source_execution=source_execution)
    reference = runtime.prepare_public_joint_reference(admission, strategy='frozen_source_context', noise_scale=1.,
        conditioning_samples=32)
    try:
        inputs = runtime.prepare_public_joint_calibration(reference, prefix, caps, block_native_rows=2048)
    except runtime.expected_fit_errors as error:
        for arm in ARMS:
            parent = directory/arm
            parent.mkdir()
            write_new(parent/'FIT_REPORT.json', {'schema': 'population_joint_C_programme/1',
                'identity': programme_identity(arguments, requested_K, arm), 'actual_K': prefix.budget,
                'status': 'fit_failure', 'requested_held': 63, 'fit_owner_called': False,
                'stage': 'actual_C_numerical_compilation', 'failure': failure_record(error),
                'original_label_use': runtime.plain_metadata(prefix.label_use), 'cell_publications': []})
        return
    for arm in ARMS:
        fit_and_score_programme(runtime, inputs, compiler=compiler, prefix=prefix, recipe=recipe,
            arguments=arguments, requested_K=requested_K, arm=arm, directory=directory/arm)


def run_target_actor(arguments):
    from research.resource_scope import require_managed_phase

    managed = require_managed_phase(arguments['resources'], phase='calibration', device='cpu')
    if (managed.task.resources.native_threads != arguments['resources']['native_threads_per_worker']
            or managed.task.resources.cpu_threads < managed.task.resources.native_threads+1):
        raise ValueError('the target actor must hold its declared native and inter-op CPU phase')
    verify_runtime_source(arguments['source_snapshot'])
    recipe, source_recipe, matrix_reference, matrix, _ = matrix_context(arguments['recipe'])
    runtime = target_runtime()
    source_execution = runtime.load_source_population_execution(arguments['source_execution'],
        trusted_acceptance_reference=arguments['source_acceptance'], expected_matrix_reference=matrix_reference)
    if recipe['source_recipe'] not in source_execution.record['recipes']:
        raise ValueError('the admitted complete source execution used a different original source recipe')
    fold = next(row for row in source_execution.record['folds'] if row['fold_id'] == arguments['target']['fold_id'])
    if fold['target_participant_key'] != arguments['target']['participant_key']:
        raise ValueError('the admitted fold belongs to another actual target')
    dictionary = runtime.NativeResponseDictionary.from_record(read(source_recipe['native_dictionary']))
    native = runtime.load_population_native_response(fold['forecast'], dictionary=dictionary,
        geometry=runtime.NoiseGeometry(**matrix['geometry']), within_reference_dependency=fold['within_reference'],
        noise_reference_dependency=fold['noise_reference'])
    arguments = {**arguments, 'source_execution_identity': source_execution.identity}
    output = Path(arguments['output'])
    output.mkdir(parents=True, exist_ok=False)
    for requested_K in KS:
        run_budget(runtime, native, source_execution, recipe=recipe, source_recipe=source_recipe,
            arguments=arguments, requested_K=requested_K, directory=output/f'K{requested_K}')
        # The completed K's compiler, raw/history caches and live fits are no
        # longer owned. Source/native inputs remain shared by this target actor.
        gc.collect()
    publications = [{'requested_K': requested_K, 'arm': arm,
        'fit_report': artifact(output/f'K{requested_K}'/arm/'FIT_REPORT.json')}
        for requested_K in KS for arm in ARMS]
    return write_new(output/'TARGET_RESULT.json', {'schema': 'population_joint_target_actor/1',
        'participant_key': arguments['target']['participant_key'], 'fold_id': fold['fold_id'],
        'source_execution_identity': source_execution.identity, 'requested_panel_sha256': arguments['requested_panel_sha256'],
        'source_snapshot': arguments['source_snapshot'], 'recipe': arguments['recipe'],
        'requested_C_programmes': 6, 'requested_held_cells': 378, 'programmes': publications, 'new_source_fits': 0,
        'target_C_warm_start_reused': False, 'actual_programme_GPU_claim': 0})


def validate_resource_plan(resources):
    scope = resources.get('resource_scope')
    if (resources.get('schema') != 'public_population_joint_matrix_resources/1'
            or scope not in {'local_development', 'cloud'} or resources.get('device') != 'cpu'):
        raise ValueError('the matrix requires its explicit managed local/cloud CPU resource declaration')
    names = ('target_workers', 'native_threads_per_worker', 'cpu_threads_total',
        'target_worker_reservation_bytes', 'controller_memory_reservation_bytes', 'minimum_free_disk_margin_bytes')
    if any(type(resources.get(name)) is not int or resources[name] < 1 for name in names):
        raise ValueError('matrix worker, native CPU and complete RAM claims must be positive integers')
    if (resources['target_workers'] > 13
            or resources['target_workers']*(resources['native_threads_per_worker']+1)+1 > resources['cpu_threads_total']):
        raise ValueError('all target native/inter-op claims plus the controller must fit the shared CPU budget')
    if scope == 'local_development':
        if (resources.get('whole_machine_utilization_ceiling') != .8
                or type(resources.get('programme_memory_limit_bytes')) is not int
                or resources['programme_memory_limit_bytes'] < 1
                or resources.get('cloud_resource_floor') is not None
                or not resources.get('resource_observation', {}).get('source')):
            raise ValueError('local research retains its declared below-eighty-percent policy and observed budget')
    else:
        headroom = resources.get('memory_headroom_fraction')
        if isinstance(headroom, bool) or not isinstance(headroom, (int, float)) or not .05 <= headroom < 1:
            raise ValueError('cloud admission needs its explicit effective memory headroom')
    return scope


def resolve_matrix_resources(resources, *, cpu_count=None, free_host_bytes=None,
                             host_total_bytes=None, effective_host_bytes=None, free_device_bytes=None):
    """Thin topology adapter to the original capacity resolver and one ledger."""
    from research.cloud_execution import PhaseResources
    from research.population_source_cloud import local_memory_budget
    from research.source_execution import (
        CLOUD_RESOURCE_CONFIG,
        available_host_memory_bytes,
        resolve_source_resources,
        with_cloud_resource_floor,
    )

    scope = validate_resource_plan(resources)
    workers = resources['target_workers']
    native = resources['native_threads_per_worker']
    memory = resources['target_worker_reservation_bytes']
    staging = resources['controller_memory_reservation_bytes']
    common = {'resource_scope': scope, 'cpu_budget': resources['cpu_threads_total'],
        'source_workers': workers, 'source_worker_memory_gib': memory/2**30,
        'target_worker_memory_gib': memory/2**30, 'shared_staging_memory_gib': staging/2**30,
        'target_native_threads_per_worker': native}
    local_observation = None
    if scope == 'cloud':
        central = json.loads(CLOUD_RESOURCE_CONFIG.read_text(encoding='utf-8'))
        plan = {**central, **common, 'target_workers': central['cloud_resource_floor']['gpu_workers'],
            'host_memory_fraction': 1-resources['memory_headroom_fraction']}
        if resources.get('cloud_resource_floor') is not None:
            plan['cloud_resource_floor'] = resources['cloud_resource_floor']
        capacity = resolve_source_resources(with_cloud_resource_floor(plan, 'cuda'), 13, 'cuda',
            cpu_count=cpu_count, free_device_bytes=free_device_bytes, free_host_bytes=free_host_bytes)
        capacity['programme_capacity_device'] = 'cuda'
    else:
        observed_runtime = host_total_bytes is None or free_host_bytes is None
        if observed_runtime:
            import psutil

            observed = psutil.virtual_memory()
            host_total_bytes = int(observed.total) if host_total_bytes is None else host_total_bytes
            free_host_bytes = int(observed.available) if free_host_bytes is None else free_host_bytes
        effective = (min(free_host_bytes, available_host_memory_bytes()) if observed_runtime
            else free_host_bytes) if effective_host_bytes is None else effective_host_bytes
        local_observation = local_memory_budget(resources, total_bytes=host_total_bytes,
            available_bytes=free_host_bytes, effective_available_bytes=effective)
        plan = {**common, 'target_workers': workers,
            'host_memory_fraction': local_observation['effective_host_memory_fraction']}
        capacity = resolve_source_resources(plan, 13, 'cpu', cpu_count=cpu_count, free_host_bytes=free_host_bytes)
        capacity['host_memory_budget_bytes'] = local_observation['effective_programme_memory_limit_bytes']
        capacity['programme_capacity_device'] = 'cpu'
    if (capacity['cpu_budget'] != resources['cpu_threads_total']
            or workers*(native+1)+capacity['coordinator_cpu_threads'] > capacity['cpu_budget']
            or workers*memory+staging > capacity['host_memory_budget_bytes']):
        raise ValueError('the original resource owner cannot admit the complete declared target concurrency')
    capacity['programme_resource_scope'] = scope
    if local_observation is not None:
        capacity['local_host_memory_admission'] = local_observation
    phase = PhaseResources('calibration', native+1, memory, native, 'cpu', interop_threads=1)
    return capacity, phase


def load_requested_panel(reference, *, matrix_reference, targets):
    from research.contracts import DecisionPlanContract

    panel = read(reference)
    expected = {row['participant_key']: row for row in targets}
    plan = DecisionPlanContract.from_record(panel['decision_plan'])
    if (panel['schema'] != 'original_population_joint_requested_panel/1'
            or panel['source_matrix'] != matrix_reference or set(panel['targets']) != set(expected)
            or tuple(panel['K']) != KS or tuple(panel['arms']) != ARMS
            or panel['requested_new_cells'] != 4914 or panel['requested_decisions'] != 819
            or panel['unavailable_requests_removed'] is not False or len(plan.entries) != 819
            or set(plan.requested_participant_keys) != set(expected) or plan.participant_selection_failures):
        raise ValueError('the pinned requested panel must retain all thirteen original participants and 819 decisions')
    vocabulary = tuple(panel['candidate_vocabulary'])
    labels = {(row.participant_key, row.decision_id): row.target_candidate for row in plan.entries}
    seen = set()
    for key, target in panel['targets'].items():
        rows = target['requests']
        if (target['declaration'] != expected[key]['declaration'] or target['fold_id'] != expected[key]['fold_id']
                or len(rows) != 63 or not rows or rows[0]['ordinal'] != 0):
            raise ValueError('each target panel retains its original declaration, fold and complete ordered 63 requests')
        for row in rows:
            cell = key, row['decision_id']
            if (cell in seen or cell not in labels or labels[cell] not in vocabulary
                    or tuple(row['candidate_vocabulary']) != vocabulary
                    or type(row['evidence_level']) is not int or row['evidence_level'] < 1
                    or len(row['expected_event_counts']) != len(vocabulary)
                    or any(type(count) is not int or count < 1 for count in row['expected_event_counts'])):
                raise ValueError('the original full-vocabulary truth/horizon/coverage roster changed')
            seen.add(cell)
    if seen != set(labels):
        raise ValueError('the original requested decision plan and physical roster differ')
    return panel, plan


def _assert_programme_identity(record, expected):
    if record['identity'] != expected:
        raise ValueError('a matrix artifact belongs to another requested source/target/K/arm programme')


def disk_free_bytes(path):
    from shutil import disk_usage

    return disk_usage(path).free


def verify_history_storage(receipt, prediction, *, directory, storage, verified_files):
    """Verify original compact closure file refs once per physical path/SHA.

    This consumes the original owner's declared closure without decoding any
    numeric model or reconstructing a likelihood, proposal or trace.
    """
    path = (directory/'NUMERIC_STORAGE_CLOSURE.json').resolve()
    if Path(receipt['numeric_storage_closure_path']).resolve() != path:
        raise ValueError('the history receipt names another requested cell storage closure')
    reference = pinned(path, receipt['numeric_storage_closure_sha256'])
    closure = read(reference)
    if (closure['schema'] != 'compact_history_cell_storage_closure/1'
            or closure['dense_candidate_potential_or_proposal_roots_stored'] is not False
            or closure['whole_matrix_storage_acceptance'] is not False
            or any(expert.model_digest != closure['target_identity'] for expert in prediction.evidence.expert_identities)):
        raise ValueError('the original numeric closure differs from its actual typed target')
    own = closure['cell_files']
    for name, expected_sha in (('target.json', receipt['complete_numeric_target_sha256']),
            ('inference.json', receipt['inference_result_sha256'])):
        if own[str((directory/name).resolve())]['sha256'] != expected_sha:
            raise ValueError('the storage closure changed its original target/inference publication')

    def verify(file_path, dependency):
        file_path = Path(file_path).resolve()
        key = str(file_path), dependency['sha256']
        if key not in verified_files:
            actual = artifact(file_path)
            if actual['sha256'] != dependency['sha256'] or actual['bytes'] != dependency['bytes']:
                raise ValueError('an original history storage file changed: '+str(file_path))
            verified_files[key] = actual
        elif verified_files[key]['bytes'] != dependency['bytes']:
            raise ValueError('original closure references disagree about one immutable file size')

    for file_path, dependency in own.items():
        verify(file_path, dependency)
        previous = storage['cell_files'].setdefault(file_path, dependency)
        if previous != dependency:
            raise ValueError('cell closures disagree about the same original physical file')
    store_path = str(Path(closure['shared_numeric_store']).resolve())
    expected_store = str((directory.parent/'shared-inference-numeric').resolve())
    if store_path != expected_store:
        raise ValueError('the original numeric closure names another programme shared store')
    union = storage['shared_stores'].setdefault(store_path, {})
    for name, dependency in closure['shared_dependencies'].items():
        if name not in {dependency['sha256']+'.json', dependency['sha256']+'.npz'}:
            raise ValueError('the original shared dependency is not its content-addressed basename')
        verify(Path(store_path)/name, dependency)
        previous = union.setdefault(name, dependency)
        if previous != dependency:
            raise ValueError('same-store closures disagree about an original content-addressed object')
    if (sum(row['bytes'] for row in own.values()) != closure['cell_file_bytes']
            or sum(row['bytes'] for row in closure['shared_dependencies'].values()) != closure['shared_dependency_bytes_once']):
        raise ValueError('the original cell closure changed its declared complete file byte accounting')
    storage['closures'].append(reference)


def target_report_closure(output, target_record, *, target, arguments):
    publications = target_record['programmes']
    expected = {(requested_K, arm) for requested_K in KS for arm in ARMS}
    if (len(publications) != 6
            or {(row['requested_K'], row['arm']) for row in publications} != expected):
        raise ValueError('the retired target publication must bind all six original C programmes')
    reports = {}
    for row in publications:
        requested_K, arm = row['requested_K'], row['arm']
        expected_path = (output/'targets'/target['fold_id']/f'K{requested_K}'/arm/'FIT_REPORT.json').resolve()
        if Path(row['fit_report']['path']).resolve() != expected_path:
            raise ValueError('a target publication points outside its own original C programme')
        record = read(row['fit_report'])
        _assert_programme_identity(record, programme_identity({**arguments, 'target': target}, requested_K, arm))
        if record['status'] not in {'completed_requested_programme', 'fit_failure', 'calibration_unavailable'}:
            raise ValueError('the retired target contains an unfinished original C programme')
        cells = record.get('cell_publications', [])
        if (record['schema'] != 'population_joint_C_programme/1'
                or record['status'] == 'completed_requested_programme' and len(cells) != 63
                or record['status'] != 'completed_requested_programme' and cells):
            raise ValueError('each completed fit binds exactly its 63 held outcomes; failed C grants no prediction')
        reports[requested_K, arm] = (record, row['fit_report'])
    return reports


def collect_metrics(output, *, panel, plan, targets, arguments, recipe, admitted_targets):
    """One existing requested metric owner; missing artifacts retain the roster."""
    from models.candidate_evidence import CandidateEvidence
    from research.candidate_metrics import evaluate_candidate_evidence
    from research.public_joint_history import PublicJointHistoryPrediction
    from transfer.outcomes import CandidateCoverage, DecisionKey

    seed = recipe['inference_configuration']['inference_seed']
    verified_files = {}
    storage = {'schema': 'population_joint_matrix_original_storage_ledger/1',
        'requested_cells': 4914, 'closures': [], 'cell_files': {}, 'shared_stores': {},
        'explicit_unavailable_cells': 0, 'unadmitted_or_missing_cells': 0,
        'expected_fit_failure_cells': 0, 'calibration_unavailable_cells': 0,
        'file_verification': 'stream_SHA_once_per_resolved_physical_path_and_expected_SHA; no_numeric_decode',
        'union_rule': 'same_resolved_store_and_CAS_basename; different_stores_count_physical_copies'}
    horizons, coverage = {}, {}
    for key, target in panel['targets'].items():
        for row in target['requests']:
            decision = DecisionKey(key, row['decision_id'])
            horizons[decision] = row['evidence_level']
            coverage[decision] = CandidateCoverage.from_mappings(
                dict(zip(panel['candidate_vocabulary'], row['expected_event_counts'], strict=True)), {})
    reports_by_target = {}
    for target in targets:
        publication = admitted_targets.get(target['fold_id'])
        if publication is not None:
            reports_by_target[target['fold_id']] = target_report_closure(output, read(publication),
                target=target, arguments=arguments)
    metrics, programmes = {}, []
    for requested_K in KS:
        for arm in ARMS:
            evidence, fit_failures = [], {}
            for target in targets:
                identity = programme_identity({**arguments, 'target': target}, requested_K, arm)
                directory = output/'targets'/target['fold_id']/f'K{requested_K}'/arm
                bound = reports_by_target.get(target['fold_id'], {}).get((requested_K, arm))
                report, report_reference = (None, None) if bound is None else bound
                if report is not None:
                    _assert_programme_identity(report, identity)
                    if report['status'] == 'fit_failure':
                        fit_failures[target['participant_key']] = report['failure']['reason']
                programmes.append({'identity': identity, 'status': None if report is None else report['status'],
                    'fit_report': report_reference})
                rows = panel['targets'][target['participant_key']]['requests']
                publications = [] if report is None else report.get('cell_publications', [])
                for ordinal, row in enumerate(rows):
                    cell = directory/'held'/f'{ordinal:03}'/'OUTCOME.json'
                    if not publications:
                        category = ('expected_fit_failure_cells' if report and report['status'] == 'fit_failure'
                            else 'calibration_unavailable_cells' if report and report['status'] == 'calibration_unavailable'
                            else 'unadmitted_or_missing_cells')
                        storage[category] += 1
                        continue
                    publication = publications[ordinal]
                    if Path(publication['path']).resolve() != cell.resolve():
                        raise ValueError('the bound held artifact differs from its original request ordinal')
                    record = read(publication)
                    _assert_programme_identity(record, identity)
                    if (record['schema'] != 'population_joint_requested_cell/1'
                            or record['decision_id'] != row['decision_id'] or report is None
                            or record['fit_identity'] != report.get('fit_identity')):
                        raise ValueError('a requested cell differs from its actual complete C fit or decision')
                    if record['typed_prediction'] is None:
                        item = CandidateEvidence.from_record(record['unavailable_evidence'])
                        if item.probability_available:
                            raise ValueError('a probability requires the actual typed prediction and replay receipt')
                        storage['explicit_unavailable_cells'] += 1
                    else:
                        dependency = record['typed_prediction']
                        prediction = PublicJointHistoryPrediction.load(dependency['path'], expected_sha256=dependency['sha256'])
                        receipt = read(record['history_receipt'])
                        if (prediction.scope['scoring_law'] != 'whole'
                                or receipt['schema'] != 'same_process_new_joint_fit_complete_history_typed_replay/1'
                                or receipt['joint_fit'] != record['fit_identity']
                                or receipt['typed_prediction_sha256'] != dependency['sha256']
                                or receipt['typed_evidence_status'] != prediction.evidence.status
                                or receipt['future_truth_reads'] != 0):
                            raise ValueError('a typed probability lacks its actual whole-law replay receipt')
                        item = prediction.evidence
                        verify_history_storage(receipt, prediction, directory=cell.parent,
                            storage=storage, verified_files=verified_files)
                    if (item.participant_key != target['participant_key'] or item.decision_id != row['decision_id']
                            or item.status != record['status']
                            or any(expert.fit_digest != record['fit_identity'] for expert in item.expert_identities)):
                        raise ValueError('the typed cell evidence changed its original requested identity or fitted state')
                    evidence.append(item)
            result = evaluate_candidate_evidence({seed: evidence}, decision_plan=plan, seeds=(seed,),
                candidate_vocabulary=panel['candidate_vocabulary'], requested_evidence_levels=horizons,
                requested_coverage=coverage, participant_failures_by_seed={seed: fit_failures})
            if len(result.cells) != 819:
                raise ValueError('the existing requested metric owner lost an original decision')
            metrics[f'K{requested_K}/{arm}'] = result.record()
    if len(programmes) != 78 or sum(value['outcome_accounting_by_seed'][str(seed)]['requested']
            for value in metrics.values()) != 4914:
        raise ValueError('the matrix terminal record lost an original programme or requested cell')
    storage.update(actual_typed_cell_closures=len(storage['closures']),
        cell_file_union_bytes=sum(row['bytes'] for row in storage['cell_files'].values()),
        shared_dependency_union_bytes=sum(row['bytes'] for store in storage['shared_stores'].values() for row in store.values()),
        closure_manifest_bytes=sum(row['bytes'] for row in storage['closures']),
        verified_unique_physical_file_refs=list(verified_files.values()),
        numeric_scope='original_target_inference_trace_and_same_store_CAS; fit_and_source_artifacts_remain_separate')
    if (len(storage['closures'])+sum(storage[name] for name in ('explicit_unavailable_cells',
            'unadmitted_or_missing_cells', 'expected_fit_failure_cells', 'calibration_unavailable_cells')) != 4914):
        raise ValueError('the storage ledger lost a requested original outcome')
    return metrics, programmes, storage


def run_owned_programme(*, mode, recipe_reference, source_snapshot, resources, output,
                        source_execution_reference=None, source_acceptance_reference=None,
                        panel_reference=None, on_observation=None):
    from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
    from research.execution import atomic_write_json
    from research.source_execution import HostMemoryAdmission

    recipe, _, matrix_reference, _, targets = matrix_context(recipe_reference)
    scope = validate_resource_plan(resources)
    capacity, target_phase = resolve_matrix_resources(resources)
    output = Path(output).resolve()
    controller = output.with_name(output.name+('.local' if scope == 'local_development' else '.cloud'))
    if output.exists() or controller.exists():
        raise FileExistsError('the matrix scientific and original-owner controller directories must both be new')
    initial_free_disk_bytes = disk_free_bytes(output.parent)
    if initial_free_disk_bytes < resources['minimum_free_disk_margin_bytes']:
        raise ValueError('the declared free-disk margin is unavailable before the original requested programme')
    common = {'recipe': recipe_reference, 'source_snapshot': source_snapshot, 'resources': resources}
    source_execution_identity = None
    panel = plan = None
    if mode == 'prepare-panel':
        phase = PhaseResources('preparation', target_phase.cpu_threads, target_phase.memory_bytes,
            target_phase.native_threads, 'cpu', interop_threads=1)
        tasks = (CloudTask('requested-panel', ADAPTER+':prepare_requested_panel',
            {**common, 'output': str(output)}, str(output), phase),)
    elif mode == 'run':
        from research.public_population_source import load_source_population_execution

        accepted = load_source_population_execution(source_execution_reference,
            trusted_acceptance_reference=source_acceptance_reference, expected_matrix_reference=matrix_reference)
        if recipe['source_recipe'] not in accepted.record['recipes']:
            raise ValueError('the complete source publication used another original source recipe')
        source_execution_identity = accepted.identity
        panel, plan = load_requested_panel(panel_reference, matrix_reference=matrix_reference, targets=targets)
        common.update(source_execution=source_execution_reference, source_acceptance=source_acceptance_reference,
            requested_panel_sha256=panel_reference['sha256'])
        output.mkdir(parents=True, exist_ok=False)
        write_new(output/'MATRIX_REQUEST.json', {'schema': 'population_joint_matrix_request/1',
            **{key: value for key, value in common.items() if key != 'resources'},
            'source_execution_identity': accepted.identity, 'requested_panel': panel_reference,
            'decision_plan_digest': plan.digest(), 'source_matrix': matrix_reference,
            'requested_targets': 13, 'requested_C_programmes': 78, 'requested_held_cells': 4914,
            'K': list(KS), 'arms': list(ARMS), 'target_C_warm_start_reused': False})
        # Actor arguments carry only the physical request roster. Controller
        # DecisionPlanEntry truth is never a fit/prediction argument.
        tasks = tuple(CloudTask(target['fold_id'], ADAPTER+':run_target_actor',
            {**common, 'target': target,
             'requested_roster': panel['targets'][target['participant_key']]['requests'],
             'output': str(output/'targets'/target['fold_id'])},
            str(output/'targets'/target['fold_id']), target_phase) for target in targets)
    else:
        raise ValueError('matrix mode must be prepare-panel or run')
    controller.mkdir(parents=True, exist_ok=False)
    identity = {'programme': 'population_joint_full_requested_matrix', 'mode': mode,
        **{key: value for key, value in common.items() if key != 'resources'}, 'resource_scope': scope,
        'resource_owner': 'original_source_family_HostMemoryAdmission_and_CloudExecutor',
        'project_capacity_device': capacity['programme_capacity_device'], 'actual_programme_device': 'cpu',
        'actual_GPU_claim': 0, 'ready_GPU_tasks': 0,
        'target_actor_phases': ['calibration', 'evaluation'],
        'aggregate_actor_phase': 'calibration_with_its_same_process_postfit_evaluation',
        'whole_machine_utilization_supervision': 'external_launch_supervisor' if scope == 'local_development' else None}

    def acknowledge(outcome):
        if outcome.status != 'completed':
            return ()
        record = read(outcome.value)
        if mode == 'prepare-panel':
            load_requested_panel(outcome.value, matrix_reference=matrix_reference, targets=targets)
        else:
            target = next(row for row in targets if row['fold_id'] == outcome.task_id)
            if (record['schema'] != 'population_joint_target_actor/1'
                    or record['participant_key'] != target['participant_key']
                    or record['fold_id'] != target['fold_id']
                    or record['source_execution_identity'] != source_execution_identity
                    or record['requested_panel_sha256'] != panel_reference['sha256']
                    or record['source_snapshot'] != source_snapshot or record['recipe'] != recipe_reference
                    or record['requested_C_programmes'] != 6 or record['requested_held_cells'] != 378
                    or record['new_source_fits'] != 0 or record['target_C_warm_start_reused'] is not False):
                raise ValueError('a retired target actor differs from its complete original requested programme')
            target_report_closure(output, record, target=target, arguments=common)
        return ()

    with get_context('spawn').Manager() as manager:
        admission = HostMemoryAdmission(manager, capacity)
        executor = CloudExecutor(admission, max_workers=resources['target_workers'],
            diagnostic_directory=controller, identity=identity)
        run = executor.run(tasks, on_finished=acknowledge, on_observation=on_observation)
        owner_receipt = {'schema': 'owned_population_joint_matrix_execution/1', 'identity': identity,
            'capacity': capacity, 'results': {key: asdict(value) for key, value in run.results.items()},
            'executor_status': run.status, 'diagnostics': run.diagnostics,
            'observation_failures': run.observation_failures, 'settled_resource_observation': admission.record(),
            'actual_programme_GPU_claim': 0, 'ready_GPU_tasks': 0}
        atomic_write_json(controller/'MATRIX_EXECUTION_RECEIPT.json', owner_receipt)
        if mode == 'prepare-panel':
            if run.status != 'completed':
                raise RuntimeError('requested-panel input preparation failed; original owner diagnostics retained')
            return run.results['requested-panel'].value
        from train.runtime import cpu_thread_budget

        try:
            with cpu_thread_budget(max(1, admission.coordinator_threads)):
                metrics, programmes, storage = collect_metrics(output, panel=panel, plan=plan, targets=targets,
                    arguments=common, recipe=recipe, admitted_targets={key: value.value
                        for key, value in run.results.items() if value.status == 'completed'})
            terminal = {'completed_requested_programme', 'fit_failure', 'calibration_unavailable'}
            free_disk_bytes = disk_free_bytes(output)
            storage['disk_margin'] = {'minimum_free_disk_margin_bytes': resources['minimum_free_disk_margin_bytes'],
                'free_bytes_before_launch': initial_free_disk_bytes, 'free_bytes_after_closure_verification': free_disk_bytes,
                'accepted': free_disk_bytes >= resources['minimum_free_disk_margin_bytes']}
            storage_reference = write_new(output/'STORAGE_CLOSURE_LEDGER.json', storage)
            complete = (run.status == 'completed' and all(row['status'] in terminal for row in programmes)
                and storage['disk_margin']['accepted'])
            record = {'schema': 'population_joint_complete_requested_matrix/1',
                'status': 'completed_requested_matrix' if complete else 'execution_incomplete',
                'recipe': recipe_reference, 'source_snapshot': source_snapshot, 'requested_panel': panel_reference,
                'source_execution': source_execution_reference, 'source_acceptance': source_acceptance_reference,
                'source_execution_identity': source_execution_identity,
                'requested_targets': 13, 'requested_C_programmes': 78, 'requested_held_cells': 4914,
                'metrics_by_K_arm': metrics, 'programmes': programmes, 'executor_status': run.status,
                'storage_closure_ledger': storage_reference,
                'resource_owner_receipt': artifact(controller/'MATRIX_EXECUTION_RECEIPT.json'),
                'historical_baseline_cells_role': '9828_original_external_cross_study_diagnostics',
                'evidence_scope': 'public_development_full_requested_matrix',
                'adult_BrainSync_confirmation': False, 'new_source_fits': 0,
                'source_only_initialization': True, 'target_C_warm_start_reused': False}
            result = write_new(output/'MATRIX_RESULT.json', record)
        except BaseException as error:
            atomic_write_json(controller/'MATRIX_TERMINAL_FAILURE.json', failure_record(error))
            raise
    if not complete:
        raise RuntimeError('the complete requested matrix has execution failures; all requested outcomes and partial artifacts retained')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--recipe-sha256', required=True)
    parser.add_argument('--mode', choices=('prepare-panel', 'run'), default='run')
    parser.add_argument('--inspect', action='store_true', help='only inspect recipe/source-matrix JSON; no adapter or model input')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--resources-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--requested-panel', type=Path)
    parser.add_argument('--requested-panel-sha256')
    parser.add_argument('--source-execution', type=Path)
    parser.add_argument('--source-execution-sha256')
    parser.add_argument('--source-acceptance', type=Path)
    parser.add_argument('--source-acceptance-sha256')
    args = parser.parse_args(argv)
    recipe_reference = pinned(args.recipe, args.recipe_sha256)
    recipe, _, _, matrix, targets = matrix_context(recipe_reference)
    if args.inspect:
        record = {'schema': 'population_joint_matrix_code_inspection/1', 'recipe': recipe_reference,
            'source_programme_seed': matrix['programme_seed'], 'inference_seed': recipe['inference_configuration']['inference_seed'],
            'targets': list(targets), 'K': list(KS), 'arms': list(ARMS),
            'requested_C_programmes': 78, 'requested_held_cells': 4914,
            'source_acceptance_issued': False, 'adapter_or_model_inputs_opened': False,
            'fit_calls': 0, 'SMC_calls': 0, 'resource_resolution_performed': False}
        print(json.dumps(record, indent=2))
        return record
    required = ('source_manifest', 'source_manifest_sha256', 'resources', 'resources_sha256', 'output')
    if any(getattr(args, name) is None for name in required):
        parser.error('execution requires an explicit source freeze, resource plan and new output')
    if args.mode == 'run' and any(getattr(args, name) is None for name in
            ('requested_panel', 'requested_panel_sha256', 'source_execution', 'source_execution_sha256',
             'source_acceptance', 'source_acceptance_sha256')):
        parser.error('the full matrix requires separately pinned requested panel, complete source execution and independent source acceptance')
    source_snapshot = pinned(args.source_manifest, args.source_manifest_sha256)
    resources = read(pinned(args.resources, args.resources_sha256))
    validate_resource_plan(resources)
    sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
    verify_runtime_source(source_snapshot)
    source_execution_reference = (pinned(args.source_execution, args.source_execution_sha256) if args.mode == 'run' else None)
    source_acceptance_reference = (pinned(args.source_acceptance, args.source_acceptance_sha256) if args.mode == 'run' else None)
    panel_reference = (pinned(args.requested_panel, args.requested_panel_sha256) if args.mode == 'run' else None)
    result = run_owned_programme(mode=args.mode, recipe_reference=recipe_reference,
        source_snapshot=source_snapshot, resources=resources, output=args.output,
        source_execution_reference=source_execution_reference,
        source_acceptance_reference=source_acceptance_reference, panel_reference=panel_reference)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
