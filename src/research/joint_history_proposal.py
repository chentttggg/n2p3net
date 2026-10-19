"""Truth-free, complete-chart Gaussian proposals for exact history targets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from research.collapsed_continuous_inference import GaussianMixtureProposal, GaussianProposal
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_history_target import gaussian_proposal_from_c
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _lower_factor(factor):
    lower = np.linalg.qr(factor.T, mode='r').T
    return lower*np.where(np.diag(lower) < 0, -1., 1.)[None, :]


def complete_history_proposal(target, C_posterior):
    """One C reference plus one full-union reference per compatible current q.

    Discrete reference assignments are only proposal operands. The target sums
    every positive shape/history/unknown state exactly on its original scopes.
    No current or historical target truth is an argument.
    """
    if target.dimension != target.base_dimension:
        raise TypeError('the production proposal consumes the exact Rao-Blackwell target')
    components, branches, first = [], [], None
    scales = np.zeros((len(target.endpoints), len(target.q_values)))
    original_ids = tuple(C_posterior.drift_coordinate_ids)
    remaining = [tuple(name for name in ids if name not in original_ids)
        for ids in target.drift_ids_by_record.values()]
    remaining = tuple(ids for ids in remaining if ids)
    for iq, q in enumerate(target.q_values):
        if not np.isfinite(target.discrete.log_C_mass[iq]):
            continue
        selected = target.discrete.decode(np.zeros(target.discrete.dimension), iq)
        marginal = target.branch_posterior('F', selected)
        for endpoint_index, endpoint in enumerate(target.endpoints):
            pilot = marginal if endpoint == 'F' else target.branch_posterior(endpoint, selected)
            scales[endpoint_index, iq] = pilot.log_density+target.discrete.log_C_mass[iq]
        rows = list(range(target.rank))+[target.rank+marginal.drift_coordinate_ids.index(name) for name in target.drift_ids]
        mean, factor = marginal.mean[rows], marginal.factor[rows]
        components.append(GaussianProposal(mean, _lower_factor(factor)))
        branches.append({'current_q': q, 'reference_states': {name: vars(value) for name, value in selected.items()},
            'full_union_mean_factor_sha256': array_digest(mean, factor), 'marginal_binding': plain_metadata(marginal.binding)})
        if first is None:
            first = marginal
    extra = []
    for ids in remaining:
        rows = [target.rank+first.drift_coordinate_ids.index(name) for name in ids]
        extra.append((ids, GaussianProposal(first.mean[rows], _lower_factor(first.factor[rows]))))
    C_reference, C_binding = gaussian_proposal_from_c(target, C_posterior, extra)
    components.append(C_reference)
    proposal = GaussianMixtureProposal(tuple(components), np.full(len(components), -np.log(len(components))))
    binding = freeze_metadata({'schema': 'complete_history_truth_free_Gaussian_proposal/1',
        'target': target.identity, 'role': 'proper_numerical_proposal_only_original_target_unchanged',
        'C_posterior_reference': plain_metadata(C_binding), 'candidate_balanced_union_references': branches,
        'positive_component_weights': 'one_equal_weight_per_feasible_current_q_reference_plus_one_C_reference',
        'all_source_and_physical_flat_coordinates_retained': True,
        'history_and_shape_prior_truncation': False, 'target_truth_read': False,
        'endpoint_candidate_log_scales': scales.tolist(),
        'pilot_scale_role': 'truth_free_single_compatible_reference_assignment_Gaussian_integral; not_an_all_history_oracle',
        'original_proposal_numeric_sha256': metadata_digest({'log_weights': array_digest(proposal.log_weights),
            'components': [array_digest(component.mean, component.root) for component in proposal.components]}),
        'producer': IMPLEMENTATION_SHA256})
    return proposal, scales, binding
