"""Original tempered SMC with explicit true-zero candidate and replay semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.collapsed_continuous_inference import (
    _psi,
    probabilities,
    run_smc,
    save_cloud,
)
from research.frozen_metadata import freeze_metadata
from research.joint_history_numeric_store import history_proposal_from_record
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_NEGATIVE_INFINITY = {'joint_history_number': 'negative_infinity'}


def _same_float64_bytes(actual, saved):
    actual, saved = np.asarray(actual), np.asarray(saved)
    return (actual.dtype == saved.dtype == np.dtype('float64') and actual.shape == saved.shape
        and actual.tobytes(order='C') == saved.tobytes(order='C'))


def _encode(value):
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, float) and np.isneginf(value):
        return dict(_NEGATIVE_INFINITY)
    return value


def _decode(value):
    if isinstance(value, dict):
        if value == _NEGATIVE_INFINITY:
            return -np.inf
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def save_log_record(path, record):
    content = (json.dumps(_encode(record), sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    Path(path).write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def load_log_record(path, *, expected_sha256):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('joint-history numerical artifact changed its external saved identity')
    return _decode(json.loads(content))


def _supported_probabilities(pooled, endpoints, log_prior, scales, compatible):
    active = probabilities(pooled[:, compatible], endpoints, log_prior[compatible], scales[:, compatible])
    result = {}
    for name, values in active.items():
        full = np.full(len(log_prior), -np.inf)
        full[compatible] = values
        result[name] = full
    return result


def _replicate_errors(components, pooled, probabilities_, endpoints, compatible):
    """Original independent-replicate delta SE on the exact compatible support."""
    count, _, _ = components.shape
    values = np.exp(components[:, :, compatible]-pooled[None, :, compatible]).reshape(count, -1)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))/count
    if not np.isfinite(covariance).all():
        raise ArithmeticError('independent replicate covariance is unresolved on positive components')
    width, total = int(compatible.sum()), len(compatible)
    f = endpoints.index('F')
    gradients = {}
    fp = np.exp(probabilities_['whole_log_probability'][compatible])
    gradient = np.zeros((width, covariance.shape[0]))
    gradient[:, f*width:(f+1)*width] = np.eye(width)-np.ones((width, 1))*fp[None, :]
    gradients['whole_log_probability_se'] = gradient
    if 'P' in endpoints:
        p = endpoints.index('P')
        logits = np.zeros_like(gradient)
        logits[:, f*width:(f+1)*width] = np.eye(width)
        logits[:, p*width:(p+1)*width] = -np.eye(width)
        cp = np.exp(probabilities_['conditional_log_probability'][compatible])
        gradients['conditional_log_probability_se'] = (np.eye(width)-np.ones((width, 1))*cp[None, :])@logits
    output = {'exact_zero_C_mass_mask': (~compatible).tolist(),
        'SE_domain': 'log_probability_on_positive_C_compatible_support; null_for_exact_zero'}
    for name, gradient in gradients.items():
        se = np.sqrt(np.maximum(0, np.diag(gradient@covariance@gradient.T)))
        full = [None]*total
        for index, value in zip(np.flatnonzero(compatible), se, strict=True):
            full[index] = float(value)
        output[name] = full
        probability_name = name.replace('log_probability', 'probability')
        probability_values = np.exp(probabilities_[name.removesuffix('_se')][compatible])
        probability_se = np.zeros(total)
        probability_se[compatible] = probability_values*se
        output[probability_name] = probability_se.tolist()
    output['zero_axis_probability_SE_role'] = 'exact_C_constraint_zero; not_a_sampling_estimate'
    return output


def run_joint_history_smc(target, proposal, endpoint_log_scales, **configuration):
    """The original SMC algorithm; current axes are complete, including true zero."""
    # The canonical diagnostic subtracts -inf from -inf for exact-zero axes.
    # Its positive common-target SMC path is unchanged; normalize on the exact
    # compatible support below and retain all original axes in output and cloud.
    with np.errstate(invalid='ignore'):
        result, arrays = run_smc(target, proposal, endpoint_log_scales, **configuration)
    compatible = np.isfinite(target.discrete.log_C_mass)
    pooled, scales = np.array(result['pooled_scaled_log_components']), np.array(result['component_log_scales'])
    estimates = _supported_probabilities(pooled, target.endpoints, target.log_prior, scales, compatible)
    components = np.array([arrays[f"replicate_{row['replicate']}_scaled_log_component"]
        for row in result['replicate_results']])
    result.update({'schema': 'joint_history_original_SMC_full_candidate_result/1',
        'canonical_SMC_schema': result['schema'],
        'probabilities': {key: value.tolist() for key, value in estimates.items()},
        'estimated_mc_error': _replicate_errors(components, pooled, estimates, target.endpoints, compatible),
        'exact_zero_C_mass_mask': (~compatible).tolist(),
        'conditional_support_measure': 'original_C_compatible_current_candidates; original_prior_ratio_only_where_J_P_positive',
        'P_F_log_ratio_on_C_incompatible_candidates': 'undefined; displayed_conditional_probability_is_exact_zero',
        'target_contract': dict(target.binding), 'discrete_integration': target.discrete_record(),
        'posterior_as_proposal_only': True, 'separate_global_history_cartesian_enumeration': False,
        'inference_wrapper_sha256': IMPLEMENTATION_SHA256})
    return result, arrays


def persist_joint_history_smc(directory, result, arrays, *, cloud_trace=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if cloud_trace is None:
        save_cloud(directory/'common_cloud.npz', result, arrays)
    else:
        from research.common_cloud_transition_trace import save_common_cloud_trace

        result['saved_cloud'] = save_common_cloud_trace(directory/'common_cloud.npz', result, arrays, cloud_trace)
    return save_log_record(directory/'inference.json', result)


def replay_joint_history_smc(target, result):
    """Reevaluate actual model, proposal, common Psi and full candidate estimates."""
    if result['schema'] != 'joint_history_original_SMC_full_candidate_result/1' or target.identity != result['target_identity']:
        raise ValueError('joint-history replay requires its original full target and saved result')
    proposal, scales = history_proposal_from_record(result['proper_proposal'], target=target), np.array(result['component_log_scales'])
    descriptor = result['saved_cloud']
    if 'schema' in descriptor:
        from research.common_cloud_transition_trace import SCHEMA, load_common_cloud_trace

        if descriptor['schema'] != SCHEMA:
            raise ValueError('joint-history requires its actual supported common-cloud trace programme')
        trace_metadata = json.loads(Path(descriptor['path']).read_bytes())
        from research.common_cloud_transition_trace import _configuration

        if metadata_digest(trace_metadata['configuration']) != metadata_digest(_configuration(result)):
            raise ValueError('saved trace and inference differ in their original complete SMC programme')
        data = load_common_cloud_trace(descriptor, proposal)
    else:
        cloud = Path(descriptor['path'])
        if hashlib.sha256(cloud.read_bytes()).hexdigest() != descriptor['sha256']:
            raise ValueError('joint-history common cloud changed its saved bytes')
        with np.load(cloud, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
    compatible = np.isfinite(target.discrete.log_C_mass)
    components = []
    for row in result['replicate_results']:
        key = f"replicate_{row['replicate']}"
        points = data[key+'_points']
        joint, log_g = target.evaluate_many(points), proposal.log_density(points)
        psi = _psi(joint, scales)
        if any(not _same_float64_bytes(actual, data[key+suffix]) for actual, suffix in (
                (joint, '_joint'), (log_g, '_log_g'), (psi, '_log_psi'))):
            raise ValueError('actual source/local/history target or proposal changed after inference')
        component = row['log_normalizer']+logsumexp(joint-scales[None, :, :]
            -psi[:, None, None], axis=0)-np.log(result['particles'])
        if not _same_float64_bytes(component, data[key+'_scaled_log_component']):
            raise ValueError('saved positive common-cloud component estimates changed')
        components.append(component)
    pooled = logsumexp(components, axis=0)-np.log(result['replicates'])
    if not _same_float64_bytes(pooled, result['pooled_scaled_log_components']):
        raise ValueError('saved independent-replicate pooling changed')
    estimates = _supported_probabilities(pooled, target.endpoints, target.log_prior, scales, compatible)
    if any(not _same_float64_bytes(value, result['probabilities'][key]) for key, value in estimates.items()):
        raise ValueError('complete current candidate probabilities changed on model replay')
    errors = _replicate_errors(np.array(components), pooled, estimates, target.endpoints, compatible)
    if errors != result['estimated_mc_error'] or (~compatible).tolist() != result['exact_zero_C_mass_mask']:
        raise ValueError('replicate uncertainty or exact-zero candidate support changed')
    return freeze_metadata({'layer': 'joint_history_actual_model_and_common_cloud', 'bitexact': True,
        'target': target.identity, 'source_fits': 0, 'target_noise_fits': 0, 'resampling_calls': 0,
        'cloud_schema': descriptor.get('schema', 'original_complete_cloud_npz'),
        'accept_decisions_recomputed': 0})
