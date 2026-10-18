"""Local conditional posterior from the same QR as the local marginal.

The sealed local-factor owner supplies its marginal/normalizer algebra. This
extension retains the already computed upper rows for conditional moments; no
second QR, source prior, drift prior or fitted parameter is introduced.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from research.gaussian_local_factor import IMPLEMENTATION_SHA256 as LOCAL_CORE_SHA256
from research.gaussian_local_factor import _finish_local_integral
from scipy.linalg import solve_triangular

from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_response_scale import _augmented_qr
from transfer.noise_model import array_digest, finite_real_array, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class LocalGaussianConditional:
    """v|g ~ N(U^-1(rhs-cross@g), U^-1 U^-T), g=[shared,record drift]."""

    upper: np.ndarray
    cross: np.ndarray
    rhs: np.ndarray
    local_dimensions: tuple
    shared_dimension: int
    drift_dimension: int
    binding: dict

    def __post_init__(self):
        upper = finite_real_array(self.upper, name='local conditional full upper precision root')
        cross = finite_real_array(self.cross, name='local conditional shared/drift cross block')
        rhs = finite_real_array(self.rhs, name='local conditional complete right hand side')
        dimensions = tuple(tuple(value) for value in self.local_dimensions)
        width = sum(size for _, size in dimensions)
        if (upper.shape != (width, width) or cross.shape != (width, self.shared_dimension+self.drift_dimension)
                or rhs.shape != (width,) or not np.array_equal(upper, np.triu(upper))
                or np.any(np.diag(upper) == 0)):
            raise ValueError('local conditional must retain every original local/shared/drift coordinate')
        for name, values in (('upper', upper), ('cross', cross), ('rhs', rhs)):
            object.__setattr__(self, name, frozen_array(values))
        object.__setattr__(self, 'local_dimensions', dimensions)
        object.__setattr__(self, 'binding', freeze_metadata(self.binding))

    @property
    def identity(self):
        return metadata_digest({'upper_cross_rhs': array_digest(self.upper, self.cross, self.rhs),
            'local_dimensions': self.local_dimensions, 'shared_dimension': self.shared_dimension,
            'drift_dimension': self.drift_dimension, 'binding': self.binding})

    def mean(self, point):
        point = finite_real_array(point, name='complete shared and same-record drift point')
        if point.shape != (self.shared_dimension+self.drift_dimension,):
            raise ValueError('conditional query order is [all shared source, same-record drift]')
        return solve_triangular(self.upper, self.rhs-self.cross@point, lower=False)

    @cached_property
    def independent_factor(self):
        """All independent posterior directions; zero/weak input columns stay."""
        return frozen_array(solve_triangular(self.upper, np.eye(len(self.upper)), lower=False))


def integrate_local_augmented_root_with_conditional(likelihood_root, *, local_dimensions,
        shared_dimension, drift_dimension, observed_scalar_rows, log_noise_jacobian, binding):
    """One DATA-only augmented root -> (marginal potential, local conditional).

    Parameters and column order match the sealed integrate_local_augmented_root:
    [all independent local v | shared u | same-record drift a | y-mean]. The
    input root has no priors. Only one local standard-normal I is added, and
    the canonical augmented QR runs once. Original raw n/Jacobian remain intact.
    """
    dimensions = tuple(tuple(value) for value in local_dimensions)
    local_width = sum(width for _, width in dimensions)
    width = local_width+shared_dimension+drift_dimension+1
    likelihood = finite_real_array(likelihood_root, name='original prior-free complete data augmented root')
    if (likelihood.ndim != 2 or likelihood.shape[1] != width or len(likelihood) > observed_scalar_rows
            or observed_scalar_rows < 1 or len({name for name, _ in dimensions}) != len(dimensions)
            or any(type(size) is not int or size < 0 for _, size in dimensions)
            or min(shared_dimension, drift_dimension) < 0 or not np.isfinite(log_noise_jacobian)):
        raise ValueError('conditional recovery uses the complete data columns, original n and original noise measure')
    prior = np.zeros((local_width, width))
    prior[:, :local_width] = np.eye(local_width)
    operands = np.vstack((prior, likelihood))
    root = _augmented_qr(operands, max(1, len(operands)))
    reduction = {**plain_metadata(binding),
        'input_rows_role': 'complete_prior_free_augmented_likelihood_root',
        'input_likelihood_root': array_digest(likelihood), 'compressed_likelihood_rows': len(likelihood),
        'local_reduction_extension_producer': IMPLEMENTATION_SHA256,
        'local_integral_core_producer': LOCAL_CORE_SHA256, 'local_augmented_QR_calls': 1,
        'numeric_only_no_raw_or_fit_authority': True}
    potential = _finish_local_integral(root, dimensions, shared_dimension, drift_dimension,
        observed_scalar_rows, log_noise_jacobian, reduction)
    conditional = LocalGaussianConditional(root[:local_width, :local_width],
        root[:local_width, local_width:local_width+shared_dimension+drift_dimension],
        root[:local_width, -1], dimensions, shared_dimension, drift_dimension,
        {'marginal_potential': potential.identity, 'same_augmented_root': array_digest(root),
            'conditional_coordinate_order': ['shared_source', 'same_record_drift'],
            'conditional_local_prior_count': 1, 'shared_or_drift_prior_count_here': 0,
            'local_augmented_QR_calls': 1, 'producer': IMPLEMENTATION_SHA256,
            'numeric_only_no_raw_or_fit_authority': True})
    return potential, conditional
