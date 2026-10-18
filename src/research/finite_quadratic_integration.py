"""Exact finite route: per-record flat Gaussian elimination then shared source.

Complete discrete paths are enumerated only when their predeclared count fits.
Per-record arrowhead elimination avoids an all-record dense continuous matrix.
The numerical identity is a binary64 analytic reference, not an outward error
certificate. Long discrete histories use the separate collapsed-integral route.
"""
from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.special import logsumexp

from research.collapsed_continuous_inference import probabilities
from transfer.noise_model import metadata_digest


def integrate_complete_finite(target, *, maximum_branches):
    if hasattr(target, 'drift_groups'):
        return integrate_grouped_physical_finite(target, maximum_branches=maximum_branches)
    rosters = target.records[0][0].rosters
    count = int(np.prod([len(r.states) for r in rosters], dtype=object))
    if count > maximum_branches:
        raise ValueError(f'complete exact finite route requires {count} paths; declare general continuous integration')
    rank, current = target.rank, target.records[0][0].current
    rows = []
    for coordinates in product(*(range(len(r.states)) for r in rosters)):
        q = rosters[current].states[coordinates[current]].q
        lp = sum(r.log_mass[i] for r, i in zip(rosters, coordinates, strict=True))
        endpoint_values = []
        for endpoint in target.endpoints:
            precision, rhs, constant = np.eye(rank), np.zeros(rank), float(lp)
            for model, p, f in target.records:
                stop = p if endpoint == 'P' else f
                owned = model.endpoints[stop]
                quadratic = owned.common.copy()
                for factor in owned.factors:
                    selected = factor.coefficients[tuple(coordinates[i] for i in factor.scope)]
                    if factor.proper_only:
                        quadratic[:rank+1, :rank+1] += selected
                    else:
                        quadratic += selected
                root = np.array(model.flat_admission[stop]['physical_root'])
                inverse_flat = solve_triangular(root, np.eye(len(root)), lower=False)
                inverse_chart = block_diag(np.eye(rank+1), inverse_flat)
                standardized = inverse_chart.T@quadratic@inverse_chart
                h = -(standardized[1:, 1:]+standardized[1:, 1:].T)
                g = standardized[1:, 0]+standardized[0, 1:]
                h_uu, h_ua, h_aa = h[:rank, :rank], h[:rank, rank:], h[rank:, rank:]
                chol_flat = np.linalg.cholesky(h_aa)
                cross = solve_triangular(chol_flat, h_ua.T, lower=True)
                flat_rhs = solve_triangular(chol_flat, g[rank:], lower=True)
                precision += h_uu-cross.T@cross
                rhs += g[:rank]-cross.T@flat_rhs
                constant += (owned.log_normalizer+standardized[0, 0]+.5*(flat_rhs@flat_rhs)
                    +.5*len(root)*np.log(2*np.pi)-np.log(np.diag(chol_flat)).sum()
                    -model.flat_admission[stop]['log_chart_volume'])
            chol_source = np.linalg.cholesky((precision+precision.T)/2)
            source_rhs = solve_triangular(chol_source, rhs, lower=True)
            # The original normalized phi_r entered once via precision I; its
            # (2pi)^(-r/2) cancels the final source Gaussian integration volume.
            integral = constant+.5*(source_rhs@source_rhs)-np.log(np.diag(chol_source)).sum()
            endpoint_values.append(float(integral))
        rows.append({'coordinates': list(coordinates), 'q': q, 'log_integral_by_endpoint': endpoint_values})
    integrated = np.array([[logsumexp([row['log_integral_by_endpoint'][index] for row in rows if row['q'] == q])
        for q in target.q_values] for index, _ in enumerate(target.endpoints)])
    logp = probabilities(integrated, target.endpoints, target.log_prior)
    return {'schema': 'n2p3_exact_finite_prepared_gaussian_integral/1',
        'target_identity': target.identity, 'q_values': list(target.q_values),
        'endpoints': list(target.endpoints), 'endpoint_refusals': target.refusals,
        'log_current_prior': target.log_prior.tolist(), 'quality_kind': 'analytic_gaussian_reference',
        'finite_sample_certificate': False,
        'numerical_error_role': 'complete_discrete_paths_per_record_flat_Schur_shared_source_binary64',
        'solver_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'maximum_branches': maximum_branches, 'complete_branches': count, 'branch_integrals': rows,
        'integrated_joint': integrated.tolist(), 'probabilities': {key: value.tolist() for key, value in logp.items()},
        'estimated_mc_error': {}, 'source_fits': 0, 'target_model_fits': 0,
        'global_dense_drift_matrix_created': False,
        'largest_gaussian_matrix_dimension': max([rank]+[m.dimension for m, _, _ in target.records]),
        'original_normalized_source_prior_count': 1, 'physical_flat_measure': 'original_per_record_da_once'}


def integrate_grouped_physical_finite(target, *, maximum_branches):
    """Sum all segment quadratics within each physical group BEFORE its da."""
    rosters, rank = target.rosters, target.rank
    count = int(np.prod([len(r.states) for r in rosters], dtype=object))
    if count > maximum_branches:
        raise ValueError(f'complete exact finite route requires {count} paths; declare general continuous integration')
    rows = []
    for coordinates in product(*(range(len(r.states)) for r in rosters)):
        lp = sum(r.log_mass[i] for r, i in zip(rosters, coordinates, strict=True))
        values = []
        for endpoint in target.endpoints:
            precision, rhs, constant = np.eye(rank), np.zeros(rank), float(lp)
            for group in target.drift_groups:
                width = 2 * len(group.channels)
                quadratic, normalizer = np.zeros((1 + rank + width, 1 + rank + width)), 0.
                for model, p, f in target.records:
                    if model.group.group_id != group.group_id:
                        continue
                    owned = model.endpoints[p if endpoint == 'P' else f]
                    local = owned.common.copy()
                    for factor in owned.factors:
                        selected = factor.coefficients[tuple(coordinates[i] for i in factor.scope)]
                        if factor.proper_only:
                            local[:rank + 1, :rank + 1] += selected
                        else:
                            local += selected
                    columns = np.r_[np.arange(rank + 1), rank + 1 + np.array(model.group_columns)]
                    quadratic[np.ix_(columns, columns)] += local
                    normalizer += owned.log_normalizer
                admitted = target.flat_group_admission[endpoint][group.group_id]
                root = np.array(admitted['physical_root'])
                inverse_chart = block_diag(np.eye(rank + 1), solve_triangular(root, np.eye(width), lower=False))
                standardized = inverse_chart.T @ quadratic @ inverse_chart
                h = -(standardized[1:, 1:] + standardized[1:, 1:].T)
                g = standardized[1:, 0] + standardized[0, 1:]
                chol_flat = np.linalg.cholesky(h[rank:, rank:])
                cross = solve_triangular(chol_flat, h[:rank, rank:].T, lower=True)
                flat_rhs = solve_triangular(chol_flat, g[rank:], lower=True)
                precision += h[:rank, :rank] - cross.T @ cross
                rhs += g[:rank] - cross.T @ flat_rhs
                constant += (normalizer + standardized[0, 0] + .5 * (flat_rhs @ flat_rhs)
                    + .5 * width * np.log(2 * np.pi) - np.log(np.diag(chol_flat)).sum()
                    - admitted['log_chart_volume'])
            chol_source = np.linalg.cholesky((precision + precision.T) / 2)
            source_rhs = solve_triangular(chol_source, rhs, lower=True)
            values.append(float(constant + .5 * (source_rhs @ source_rhs) - np.log(np.diag(chol_source)).sum()))
        rows.append({'coordinates': list(coordinates), 'q': rosters[target.current].states[coordinates[target.current]].q,
            'log_integral_by_endpoint': values})
    integrated = np.array([[logsumexp([row['log_integral_by_endpoint'][index] for row in rows if row['q'] == q])
        for q in target.q_values] for index, _ in enumerate(target.endpoints)])
    logp = probabilities(integrated, target.endpoints, target.log_prior)
    return {'schema': 'n2p3_exact_finite_prepared_gaussian_integral/2',
        'target_identity': target.identity, 'q_values': list(target.q_values),
        'endpoints': list(target.endpoints), 'endpoint_refusals': target.refusals,
        'log_current_prior': target.log_prior.tolist(), 'quality_kind': 'analytic_gaussian_reference',
        'finite_sample_certificate': False,
        'numerical_error_role': 'complete_paths_physical_group_flat_Schur_shared_source_binary64',
        'solver_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'maximum_branches': maximum_branches, 'complete_branches': count, 'branch_integrals': rows,
        'integrated_joint': integrated.tolist(), 'probabilities': {key: value.tolist() for key, value in logp.items()},
        'estimated_mc_error': {}, 'source_fits': 0, 'target_model_fits': 0,
        'global_dense_drift_matrix_created': False,
        'largest_gaussian_matrix_dimension': max([rank] + [2 * len(g.channels) for g in target.drift_groups]),
        'original_normalized_source_prior_count': 1,
        'physical_flat_measure': 'one_original_da_per_explicit_drift_group',
        'drift_group_identities': [group.identity for group in target.drift_groups],
        'drift_group_measure_counts': {group.group_id: 1 for group in target.drift_groups},
        'noise_segment_count': len(target.records), 'continuous_coordinate_dimension': target.dimension}


def replay_finite_saved_integral(result):
    if result['schema'] not in {'n2p3_exact_finite_prepared_gaussian_integral/1',
                               'n2p3_exact_finite_prepared_gaussian_integral/2'}:
        raise ValueError('exact finite replay requires its original integral identity')
    rows = result['branch_integrals']
    integrated = np.array([[logsumexp([row['log_integral_by_endpoint'][index]
        for row in rows if row['q'] == q]) for q in result['q_values']]
        for index, _ in enumerate(result['endpoints'])])
    if not np.array_equal(integrated, np.array(result['integrated_joint'])):
        raise ValueError('saved finite branch sum changed the complete positive integral')
    logp = probabilities(integrated, tuple(result['endpoints']), np.array(result['log_current_prior']))
    if any(not np.array_equal(value, np.array(result['probabilities'][key])) for key, value in logp.items()):
        raise ValueError('saved finite probabilities differ from original full-q branch integral')
    return {'layer': 'saved_finite_integral', 'bitexact': True}


def replay_finite_model(target, result):
    replay_finite_saved_integral(result)
    rebuilt = integrate_complete_finite(target, maximum_branches=result['maximum_branches'])
    if metadata_digest(rebuilt) != metadata_digest(result):
        raise ValueError('complete finite result differs from actual compiled-model integration')
    return {'layer': 'finite_model_only', 'bitexact': True, 'source_fits': 0}
