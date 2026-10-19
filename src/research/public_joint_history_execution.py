"""Same-process consumer hook for the genuine frozen joint-fit issuer."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from research.common_cloud_transition_trace import CommonCloudTrace
from research.joint_history_artifact import load_joint_history_target, save_joint_history_target
from research.joint_history_inference import (
    persist_joint_history_smc,
    replay_joint_history_smc,
    run_joint_history_smc,
    save_log_record,
)
from research.joint_history_numeric_store import (
    RecipeHistoryProposal,
    SharedHistoryNumericStore,
)
from research.joint_history_proposal import complete_history_proposal
from research.joint_history_storage_closure import history_storage_closure, history_target_cost
from research.latent_response_reference import ResponseState
from research.public_joint_history import (
    PublicJointHistoryPrediction,
    admit_joint_history_inference,
    build_public_joint_history_target,
    publish_public_joint_history,
)
from research.public_joint_response_noise import require_public_joint_fit
from research.public_native_learner_plan import (
    PublicSelectionLaw,
    admit_public_native_learner,
    original_record_unions,
)
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def nominal_public_joint_laws(calibration_inputs, history_input):
    """Declared uniform actual-vocabulary law; explicit unknown laws stay caller-owned."""
    laws = {}
    for record in original_record_unions(calibration_inputs, history_input):
        for row in record.cap.public_binding['public_physical_history']['selections']:
            vocabulary = tuple(row['candidate_vocabulary'])
            laws[row['selection_id']] = PublicSelectionLaw(vocabulary, tuple(np.repeat(1/len(vocabulary), len(vocabulary))),
                (ResponseState(1., 1., 0., 1., 1.),))
    return laws


def consume_frozen_public_joint_history(fit, *, current_view, history_input, output,
        inference_configuration, quality_policy, law_by_selection=None, progress=None):
    """Publish one new cell; retain partial artifacts on a failed attempt."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        return _consume_frozen_public_joint_history(fit, current_view=current_view,
            history_input=history_input, directory=directory,
            inference_configuration=inference_configuration, quality_policy=quality_policy,
            law_by_selection=law_by_selection, progress=progress)
    except BaseException as error:
        artifacts = {}
        for path in sorted(directory.iterdir()):
            if path.is_file():
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                artifacts[path.name] = {'bytes': path.stat().st_size, 'sha256': digest}
        save_log_record(directory/'HISTORY_CONSUMER_FAILURE.json', {
            'schema': 'public_joint_history_cell_failure/1', 'status': 'failed',
            'error_type': type(error).__name__, 'error': str(error),
            'retained_artifacts': artifacts, 'partial_artifacts_are_completed_cell': False,
            'recovery': 'retain_this_attempt_and_publish_retry_in_a_new_cell_directory',
            'producer': IMPLEMENTATION_SHA256})
        raise


def _consume_frozen_public_joint_history(fit, *, current_view, history_input, directory,
        inference_configuration, quality_policy, law_by_selection=None, progress=None):
    """No fitting or grant minting; caller supplies the same-process live operands."""
    fit = require_public_joint_fit(fit)
    laws = (nominal_public_joint_laws(fit.calibration_inputs, history_input)
        if law_by_selection is None else law_by_selection)
    programme = admit_public_native_learner(fit.source_admission, fit.prefix, fit.calibration_inputs,
        history_input, current_view, law_by_selection=laws)
    numeric_store = SharedHistoryNumericStore(directory.parent/'shared-inference-numeric')
    costs, began = {}, time.perf_counter()
    target, relation = build_public_joint_history_target(fit, programme, progress=progress, numeric_store=numeric_store)
    costs['native_target_compile_seconds'] = time.perf_counter()-began
    actual_target_cost = history_target_cost(target)
    began = time.perf_counter()
    target_sha = save_joint_history_target(target, directory/'target.json')
    costs['compact_target_save_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    proposal, scales, proposal_binding = complete_history_proposal(target, fit.posterior)
    proposal = RecipeHistoryProposal(proposal, target, fit.posterior,
        numeric_store, scales, target_dependency={'path': str((directory/'target.json').resolve()), 'sha256': target_sha})
    costs['proper_proposal_compile_and_recipe_seconds'] = time.perf_counter()-began
    cloud_trace = CommonCloudTrace(proposal.original)
    began = time.perf_counter()
    result, arrays = run_joint_history_smc(target, proposal, scales, observer=cloud_trace, **inference_configuration)
    costs['original_smc_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    result_sha = persist_joint_history_smc(directory, result, arrays, cloud_trace=cloud_trace)
    costs['transition_trace_save_seconds'] = time.perf_counter()-began
    # Persisted replay recipes replace the original proposal and observer arrays.
    del proposal, cloud_trace, arrays
    began = time.perf_counter()
    storage = history_storage_closure(directory/'target.json', target_sha256=target_sha,
        inference_path=directory/'inference.json', inference_sha256=result_sha)
    storage_sha = save_log_record(directory/'NUMERIC_STORAGE_CLOSURE.json', storage)
    costs['storage_closure_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    verified = admit_joint_history_inference(target, directory/'inference.json', expected_sha256=result_sha)
    prediction = publish_public_joint_history(verified, target, quality_policy=quality_policy, scoring_law='whole')
    frontier_layout, q_values = target.frontier_layout, target.q_values
    costs['live_replay_admission_and_typed_save_reload_seconds'] = time.perf_counter()-began
    # The typed value and receipt fields contain metadata, so retire the live table
    # before reconstructing the independent cold table. Publish after both replays.
    del target
    began = time.perf_counter()
    cold_target = load_joint_history_target(directory/'target.json', expected_sha256=target_sha)
    costs['cold_native_target_recompile_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    cold_replay = replay_joint_history_smc(cold_target, result)
    costs['cold_model_proposal_and_cloud_replay_seconds'] = time.perf_counter()-began
    del cold_target
    began = time.perf_counter()
    prediction_sha = prediction.save(directory/'prediction.json')
    cold_prediction = PublicJointHistoryPrediction.load(directory/'prediction.json', expected_sha256=prediction_sha)
    if (metadata_digest(cold_prediction.evidence.expert_identities[0].record()) !=
            metadata_digest(prediction.evidence.expert_identities[0].record())
            or not np.array_equal(cold_prediction.evidence.log_probability, prediction.evidence.log_probability)):
        raise ValueError('cold typed evidence changed the original source/fit/input identity or complete probability axis')
    costs['live_replay_admission_and_typed_save_reload_seconds'] += time.perf_counter()-began
    receipt = {'schema': 'same_process_new_joint_fit_complete_history_typed_replay/1',
        'joint_fit': fit.identity, 'actual_relation': relation, 'proposal_binding': proposal_binding,
        'actual_frontier_layout': frontier_layout, 'inference_result_sha256': result_sha,
        'complete_numeric_target_sha256': target_sha, 'cold_numeric_model_replay': cold_replay,
        'numeric_storage_closure_sha256': storage_sha,
        'numeric_storage_closure_path': str((directory/'NUMERIC_STORAGE_CLOSURE.json').resolve()),
        'actual_numeric_target_cost': actual_target_cost, 'actual_phase_wall_seconds': costs,
        'typed_prediction_sha256': prediction_sha, 'typed_evidence_status': prediction.evidence.status,
        'full_candidate_axis': q_values, 'target_fits_here': 0, 'source_fits_here': 0,
        'old_noise_artifacts_relabelled': False, 'future_truth_reads': 0,
        'source_context_identity': programme.source_context.identity,
        'actual_K': programme.record['actual_K'],
        'original_requested_to_actual_C_budgets': programme.record['original_requested_to_actual_C_budgets'],
        'requested_Ks_for_this_actual_prefix': programme.record['requested_Ks_for_this_actual_prefix'],
        'response_model_family': 'G0_G1_nominal_selection_functional_Gaussian',
        'producer': IMPLEMENTATION_SHA256}
    receipt_sha = save_log_record(directory/'HISTORY_CONSUMER_RECEIPT.json', receipt)
    return prediction, receipt, receipt_sha
