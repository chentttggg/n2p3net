"""Truth-free, positive-cloud proposal learning for the original joint target.

This module estimates integration proposals/scales. It does not alter the source,
response/noise model, C posterior, candidate priors or observational likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.special import logsumexp

from research.collapsed_continuous_inference import (
    GaussianMixtureProposal,
    GaussianProposal,
    _white,
)


def physical_flat_reference(target):
    """A known proper numerical reference in original du da coordinates.

    The source coordinates retain N(0,I). Per-record drift is a Gaussian centered
    by the white noise-only least-squares projection. This is a proposal only;
    every response/source/C contribution remains in the exact importance ratio.
    """
    means, roots = [np.zeros(target.rank)], [np.eye(target.rank)]
    for record, (q, r, _) in zip(target.records, target.flat_geometry['F'], strict=True):
        model, stop = record.model, record.full_samples
        white_y = _white(model, model.y[:stop, :, None]).ravel()
        means.append(solve_triangular(r, q.T@white_y, lower=False))
        roots.append(solve_triangular(r, np.eye(len(r)), lower=False))
    mean = np.concatenate(means)
    # Inverse R is a covariance factor but not lower triangular. Cholesky makes
    # the log-density/pcn carrier explicit without changing its covariance.
    factor = block_diag(*roots)
    return GaussianProposal(mean, np.linalg.cholesky(factor@factor.T))


@dataclass(frozen=True)
class PilotResult:
    proposal: GaussianMixtureProposal
    component_log_scales: np.ndarray
    receipt: dict
    arrays: dict


def pilot_from_positive_cloud(target, reference, *, pilot_seed, particles,
                              covariance_reference_fraction):
    """Separate iid pilot draws, all endpoint/q states, complete 1/g correction."""
    if not 0 < covariance_reference_fraction < 1 or particles < 2:
        raise ValueError('declare a positive full-rank numerical proposal reference fraction')
    rng = np.random.default_rng(pilot_seed)
    points = reference.draw(rng, particles)
    log_g = reference.log_density(points)
    joint = target.evaluate_many(points)
    log_importance = joint-log_g[:, None, None]
    scales = logsumexp(log_importance, axis=0)-np.log(particles)
    components, ess_rows = [], []
    reference_covariance = reference.root@reference.root.T
    for endpoint in range(len(target.endpoints)):
        row = []
        for q in range(len(target.q_values)):
            values = log_importance[:, endpoint, q]
            weights = np.exp(values-logsumexp(values))
            mean = weights@points
            centered = points-mean
            covariance = ((1-covariance_reference_fraction)*
                ((centered.T*weights)@centered)+covariance_reference_fraction*reference_covariance)
            components.append(GaussianProposal(mean, np.linalg.cholesky(covariance)))
            row.append(float(1/(weights@weights)))
        ess_rows.append(row)
    proposal = GaussianMixtureProposal(tuple(components), np.full(len(components), -np.log(len(components))))
    return PilotResult(proposal, scales,
        {'schema': 'n2p3_continuous_proposal_pilot/1', 'target_identity': target.identity,
         'pilot_seed': pilot_seed, 'particles': particles,
         'proper_reference': reference.record(), 'component_log_scales': scales.tolist(),
         'covariance_reference_fraction': covariance_reference_fraction,
         'importance_ess_diagnostic': ess_rows, 'candidate_support': list(target.q_values),
         'truth_read': False, 'source_fits': 0, 'target_model_fits': 0,
         'proposal_role': 'numerical_integration_reference_with_exact_density_correction',
         'finite_sample_error_certificate': False},
        {'pilot_points': points, 'pilot_log_g': log_g, 'pilot_joint': joint})
