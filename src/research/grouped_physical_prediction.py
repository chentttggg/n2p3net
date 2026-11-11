"""Canonical raw registry + generic grouped integral -> full-session evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from data.brainsync_raw_authority import observed_event_indices, require_raw_capability
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.collapsed_continuous_prediction import (
    CollapsedContinuousPrediction,
    InferenceQuality,
    _verified_result,
)
from research.physical_affine_frontier import require_raw_physical_segment
from transfer.noise_model import metadata_digest


def grouped_physical_input_digest(target, caps, scoring_law):
    return metadata_digest({'target': target.identity, 'caps': [cap.identity_sha256 for cap in caps],
        'law': scoring_law, 'programme': 'physical_grouped_full_history/1'})


def require_current_physical_caps(target, caps):
    caps = tuple(require_raw_capability(cap) for cap in caps)
    first = caps[0].public_binding
    supplied = {cap.identity_sha256 for cap in caps}
    current_inputs = {model.original_input_identity for model, _, _ in target.records
        if any(scope['target_lock'] == first['target_lock_group'] for scope in model.physical_scope_bindings)
        or any(model.current in factor.scope for owned in model.endpoints.values() for factor in owned.factors)}
    if not current_inputs <= supplied:
        raise ValueError('every numerical record contributing to the current target requires its original registered raw capability')
    return {'current_numerical_inputs': sorted(current_inputs), 'registered_current_caps_complete': True}


def publish_grouped_physical(verified_inference, target, original_segments, caps, *,
                             scoring_law, expert_identity, quality_policy):
    result = _verified_result(verified_inference)
    caps = tuple(require_raw_capability(cap) for cap in caps)
    first = caps[0].public_binding
    if (scoring_law not in {'whole', 'conditional'} or result['target_identity'] != target.identity
            or target.rosters[target.current].selection_id != first['target_lock_group']
            or tuple(target.q_values) != caps[0].candidate_vocabulary):
        raise ValueError('grouped physical output requires its actual current session and complete candidate axis')
    expected_programme = metadata_digest({k: first[k] for k in ('participant_id', 'session_id',
        'record_id', 'target_lock_group', 'public_protocol', 'independent_recording_scope_sha256')})
    by_input = {model.original_input_identity: (original, model, p, f)
        for original, (model, p, f) in zip(original_segments, target.records, strict=True)}
    require_current_physical_caps(target, caps)
    observed, event_ids = np.zeros(len(target.q_values), dtype=int), set()
    for cap in caps:
        b = cap.public_binding
        programme = metadata_digest({k: b[k] for k in ('participant_id', 'session_id',
            'record_id', 'target_lock_group', 'public_protocol', 'independent_recording_scope_sha256')})
        original, model, p, f = by_input[cap.identity_sha256]
        require_raw_physical_segment(original, cap)
        if (b['role'] != 'inference' or programme != expected_programme or original.identity != model.base_identity
                or original.origin != b['sample_start'] or not np.array_equal(original.y, cap.eeg)
                or original.record_id != b['record_segment_id']
                or p != b['prefix_stop'] - b['sample_start'] or f != b['full_stop'] - b['sample_start']):
            raise ValueError('generic publication changed the actual independent raw/model/read programme')
        for i in observed_event_indices(cap):
            event = cap.events[i]
            if event.event_id in event_ids:
                raise ValueError('one completed session event cannot enter the requested evidence twice')
            event_ids.add(event.event_id)
            observed += cap.membership[i]
    if (not isinstance(expert_identity, CandidateExpertIdentity) or expert_identity.model_digest != target.identity
            or expert_identity.probability_calibrated
            or expert_identity.score_kind != f'physical_grouped_full_history/{scoring_law}/1'
            or expert_identity.numerical_input_digest != grouped_physical_input_digest(target, caps, scoring_law)):
        raise ValueError('grouped physical prediction requires its caller/source-admitted actual expert identity')
    repetitions = sum(row['requested_repetitions'] for row in first['block_requests'])
    scope = {'decision_unit': 'session_target_lock', 'evaluation_role': 'later_independent_recording',
        'session_id': first['session_id'], 'target_lock_group': first['target_lock_group'],
        'expected_total_events': sum(row['requested_flash_count'] for row in first['block_requests']),
        'completed_event_ids_digest': metadata_digest(sorted(event_ids)),
        'raw_capability_identities': [cap.identity_sha256 for cap in caps],
        'drift_groups': [group.identity for group in target.drift_groups],
        'noise_segment_declarations': [m.noise_segment_declaration for m, _, _ in target.records],
        'actual_exposure_mean_owner': 'generic_physical_affine_callbacks_all_actual_exposures',
        'native_closed_proof_used': False}
    identity = {'candidate_vocabulary': target.q_values, 'participant_key': first['participant_id'],
        'decision_id': first['target_lock_group'], 'physical_support_digest': metadata_digest(scope),
        'expert_identities': (expert_identity,), 'evidence_level': repetitions,
        'expected_event_counts': (repetitions,) * len(target.q_values), 'observed_event_counts': tuple(map(int, observed))}
    kind = result['quality_kind']
    if kind not in quality_policy.allowed_estimate_kinds:
        raise ValueError('frozen inference quality policy does not admit this estimate kind')
    key = f'{scoring_law}_log_probability'
    if key in result['probabilities']:
        evidence = CandidateEvidence.from_log_evidence(np.array(result['probabilities'][key]),
            temperature=1., prior=None, **identity)
    else:
        evidence = CandidateEvidence(log_probability=None, failure_kind='not_available',
            failure_reason=f'{scoring_law}_integral_unavailable', **identity)
    se = result.get('estimated_mc_error', {}).get(f'{scoring_law}_log_probability_se')
    precision = None
    if kind == 'replicated_mc_estimate' and quality_policy.maximum_replicate_log_se is not None:
        precision = bool(se is not None and len(se) == len(target.q_values) and np.isfinite(se).all()
            and np.max(se) <= quality_policy.maximum_replicate_log_se)
    quality = InferenceQuality(kind, result['numerical_error_role'], None if se is None else tuple(se),
        result['finite_sample_certificate'], quality_policy.policy_id, quality_policy.digest(), precision)
    return CollapsedContinuousPrediction(evidence, quality, scoring_law,
        metadata_digest([cap.identity_sha256 for cap in caps]), target.identity,
        verified_inference.admission['result_sha256'], hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope, verified_inference.admission)
