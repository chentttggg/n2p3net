"""Fixed two-exposure GLS physical reference, never a source observation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from research.channel_noise_mechanism import FittedChannelNoise
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_flat_sparse import compile_sparse_response_image
from research.joint_response_scale import NumericalResolutionError, _augmented_qr
from research.noise_scale_identification import _exact_stored_rank
from scipy.linalg import solve_triangular

from transfer.noise_model import array_digest, finite_real_array, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class TwoExposureWithinReference:
    single_exposure_factor: np.ndarray
    factor: np.ndarray
    binding: dict

    def __post_init__(self):
        single = finite_real_array(self.single_exposure_factor, name='single-exposure GLS factor')
        factor = finite_real_array(self.factor, name='complete two-exposure within factor')
        if (single.ndim != 2 or single.shape[0] != single.shape[1] or not len(single)
                or np.linalg.slogdet(single)[0] == 0):
            raise ValueError('one complete square single-exposure factor is required')
        expected = np.block([[single, np.zeros_like(single)], [-single, single]])
        if not np.array_equal(expected, factor):
            raise ValueError('two-exposure reference changed its common/increment GLS relation')
        object.__setattr__(self, 'single_exposure_factor', frozen_array(single))
        object.__setattr__(self, 'factor', frozen_array(factor))
        object.__setattr__(self, 'binding', freeze_metadata(self.binding))

    @property
    def identity(self):
        return metadata_digest({'binding': self.binding,
            'single_and_full_factor': array_digest(self.single_exposure_factor, self.factor)})

    def scaled_factor(self, x):
        if not np.isfinite(x):
            raise ValueError('within logvariance is a finite dimensionless scalar')
        with np.errstate(over='raise', under='raise', invalid='raise'):
            factor = np.exp(float(x)/2)*self.factor
        if np.any((self.factor != 0) & (factor == 0)) or not np.isfinite(factor).all():
            raise NumericalResolutionError('within factor cannot represent its complete positive support')
        return frozen_array(factor)

    def record(self):
        return {'schema': 'fixed_two_exposure_GLS_within_reference/1',
            'single_exposure_factor': self.single_exposure_factor.tolist(),
            'factor': self.factor.tolist(), 'binding': plain_metadata(self.binding),
            'identity': self.identity, 'numeric_reference_only_no_observation_or_fit_grant': True}

    def save(self, path):
        content = json.dumps(self.record(), sort_keys=True, allow_nan=False).encode()
        Path(path).write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def load(cls, path, *, expected_sha256):
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('within reference changed its external saved identity')
        record = json.loads(content)
        if (record['schema'] != 'fixed_two_exposure_GLS_within_reference/1'
                or record['numeric_reference_only_no_observation_or_fit_grant'] is not True):
            raise ValueError('within reference requires its numeric-only reference role')
        value = cls(record['single_exposure_factor'], record['factor'], record['binding'])
        if value.identity != record['identity']:
            raise ValueError('within reference changed its complete saved factors or policy')
        return value


def two_exposure_gls_reference(basis, source_noise, *, noise_scale, dictionary_identity,
        source_reference_binding, strategy, block_native_rows):
    """Compile a fixed GLS-error template from a declared source noise owner.

    shared_parameter_law_kronecker is a new explicit separable-law reference;
    saved_noise_stream uses the original saved finite whitening map exactly.
    Neither strategy replaces the original source or target numerical state.
    """
    phi = finite_real_array(basis, name='original native reference basis')
    if (type(source_noise) is not FittedChannelNoise or phi.ndim != 2 or not min(phi.shape)
            or not np.isfinite(noise_scale) or noise_scale <= 0
            or type(block_native_rows) is not int or block_native_rows < 1
            or not dictionary_identity):
        raise ValueError('explicit original basis, noise, physical scale and reference programme are required')
    if _exact_stored_rank(phi) != phi.shape[1]:
        raise ValueError('two-exposure GLS reference requires the complete basis column rank')
    source_noise.assert_integrity()
    n, k = phi.shape
    channels = len(source_noise.geometry.channels)
    if strategy == 'shared_parameter_law_kronecker':
        if source_noise.family != 'shared' or n < max(1, source_noise.order):
            raise ValueError('the separable source-law reference requires an actual shared temporal law')
        white_phi = source_noise.temporal[0].whiten(phi)
        upper = _augmented_qr(white_phi, block_native_rows)
        if upper.shape != (k, k) or np.any(np.diag(upper) == 0):
            raise NumericalResolutionError('reference temporal QR is unresolved')
        time_factor = solve_triangular(upper, np.eye(k))
        single = np.kron(time_factor, source_noise._spatial_root)*np.sqrt(noise_scale)
        numerical_role = 'new_shared_parameter_law_separable_reference_not_saved_joint_map_substitution'
    elif strategy == 'saved_noise_stream':
        width = k*channels
        upper = np.empty((0, width))
        for begin in range(0, n, block_native_rows):
            end = min(n, begin+block_native_rows)
            start = 0 if begin < source_noise.conditioning_samples else max(0, begin-source_noise.order)
            raw = np.zeros((end-start, channels, width))
            for column in range(k):
                for channel in range(channels):
                    raw[:, channel, column*channels+channel] = phi[start:end, column]
            owned = compile_sparse_response_image(source_noise, raw, start_native=start, total_native=end)
            first = begin*channels-owned.image.start
            white = owned.image.values[first:first+(end-begin)*channels]/np.sqrt(noise_scale)
            upper = _augmented_qr(np.vstack((upper, white)), max(1, len(upper)+len(white)))
        if upper.shape != (width, width) or np.any(np.diag(upper) == 0):
            raise NumericalResolutionError('reference complete native-stream QR is unresolved')
        single = solve_triangular(upper, np.eye(width))
        numerical_role = 'original_saved_noise_startup_and_innovation_map_stream_reference'
    else:
        raise ValueError('choose an explicit source-law or saved-map GLS reference strategy')
    if not np.isfinite(single).all():
        raise NumericalResolutionError('reference GLS factor is outside binary64 range')
    factor = np.block([[single, np.zeros_like(single)], [-single, single]])
    duration = Fraction(n)/Fraction.from_float(float(source_noise.geometry.sfreq))
    binding = {'schema': 'fixed_two_independent_exposure_reference_policy/1',
        'reference_records': ['unit_non_target_[Phi,0]', 'unit_target_[Phi,Phi]'],
        'independent_reference_noise_records': 2, 'reference_drift': 'fixed_zero',
        'reference_observed_values_or_labels': False, 'source_fit_or_raw_authority_granted': False,
        'coefficient_order': 'common_then_increment_coefficient_major_channel_minor',
        'native_support_each_record': [0, n], 'duration_seconds_exact_ratio': [duration.numerator, duration.denominator],
        'basis_sha256': array_digest(phi), 'basis_shape': phi.shape, 'dictionary_identity': dictionary_identity,
        'noise_geometry': source_noise.geometry.record(), 'source_noise_identity': source_noise._runtime_state_digest(),
        'source_noise_scale': float(noise_scale), 'source_reference_binding': source_reference_binding,
        'strategy': strategy, 'numerical_reference_role': numerical_role,
        'source_repetition_count_divisor': 1, 'source_posterior_factor_used': False,
        'full_support_dimension': len(factor), 'within_logvariance_prior': {'mean': 0., 'sd': 1.,
            'measure': 'dimensionless_log_variance'}, 'producer_sha256': IMPLEMENTATION_SHA256}
    return TwoExposureWithinReference(single, factor, binding)
