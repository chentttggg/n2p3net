"""Private AR1 stationary innovations from two complete small spatial roots.

This is a new numeric producer. It neither loads nor re-signs a legacy saved
prefix root. The native observation law and declared conditioning length stay
unchanged; no dense (prefix*channels)-square factor is constructed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.channel_noise_mechanism import NoiseGeometry
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_response_scale import NumericalResolutionError
from research.markov_stationary_factor import _small_factor_certificate
from research.shared_stationary_factor import _fraction_matrix, _upper
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    frozen_array,
    metadata_digest,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PROGRAMME = 'private_AR1_small_stationary_innovations/1'


class UnrepresentableJointParameterPointError(NumericalResolutionError):
    """A finite optimizer trial has no representable noise/within numeric point."""


@dataclass(frozen=True)
class PrivateAR1Channel:
    """Exact stored AR1 coefficient view owned by the new numeric programme."""

    coefficient: float

    @property
    def coefficients(self):
        return frozen_array([self.coefficient])

    @property
    def reflection_coefficients(self):
        return frozen_array([-self.coefficient])


@dataclass(frozen=True)
class PrivateAR1NoisePoint:
    phi: np.ndarray
    innovation_covariance: np.ndarray
    geometry: NoiseGeometry
    conditioning_samples: int
    provenance: dict
    initial_root: np.ndarray | None = None
    innovation_root: np.ndarray | None = None
    certificate: dict | None = None

    def __post_init__(self):
        phi = finite_real_array(self.phi, name='complete private AR1 vector')
        q = finite_real_array(self.innovation_covariance, name='complete private AR1 innovation covariance')
        c = len(self.geometry.channels)
        if (phi.shape != (c,) or q.shape != (c, c)
                or not np.array_equal(q, q.T) or type(self.conditioning_samples) is not int
                or self.conditioning_samples < 2):
            raise ValueError('the small AR1 producer requires its stable full native phi/Q/geometry')
        if np.any(np.abs(phi) >= 1):
            raise UnrepresentableJointParameterPointError('the current AR1 chart point is outside representable strict stability')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            stationary = q/(1-np.outer(phi, phi))
        supplied_roots = self.initial_root is not None or self.innovation_root is not None
        try:
            initial = np.linalg.cholesky(stationary) if self.initial_root is None else np.asarray(self.initial_root)
            innovation = np.linalg.cholesky(q) if self.innovation_root is None else np.asarray(self.innovation_root)
        except np.linalg.LinAlgError as error:
            raise UnrepresentableJointParameterPointError('the full small AR1 stationary/innovation support is unresolved') from error
        for name, root in (('initial_root', initial), ('innovation_root', innovation)):
            if (root.shape != (c, c) or not np.isfinite(root).all() or np.any(np.diag(root) <= 0)
                    or np.any(np.triu(root, 1) != 0)):
                raise ValueError('both small AR1 roots retain their complete positive lower support')
            object.__setattr__(self, name, frozen_array(root))
        if self.certificate is None and supplied_roots:
            for root, expected in ((initial, stationary), (innovation, q)):
                if np.linalg.norm(root@root.T-expected) > 1e-11*np.linalg.norm(expected):
                    raise ValueError('supplied small AR1 roots differ from their stationary/innovation covariance')
        object.__setattr__(self, 'phi', frozen_array(phi))
        object.__setattr__(self, 'innovation_covariance', frozen_array(q))
        object.__setattr__(self, '_spatial_root', self.innovation_root)
        object.__setattr__(self, 'temporal', tuple(PrivateAR1Channel(float(value)) for value in phi))
        object.__setattr__(self, 'provenance', freeze_metadata(self.provenance))
        object.__setattr__(self, 'certificate', freeze_metadata(self.certificate))
        if self.certificate is not None and metadata_digest(self.small_factor_certificate()) != metadata_digest(self.certificate):
            raise ValueError('the saved small AR1 factors differ from their actual stationary/innovation law certificate')
        object.__setattr__(self, '_runtime_guard', self._runtime_state_digest())

    @property
    def order(self):
        return 1

    @property
    def family(self):
        return 'private'

    def _runtime_state_digest(self):
        return metadata_digest({'programme': PROGRAMME, 'producer': IMPLEMENTATION_SHA256,
            'phi_Q_initial_innovation_roots': array_digest(self.phi, self.innovation_covariance,
                self.initial_root, self.innovation_root), 'geometry': self.geometry.record(),
            'conditioning_samples': self.conditioning_samples, 'provenance': self.provenance,
            'certificate': self.certificate})

    def assert_integrity(self):
        if self._runtime_state_digest() != self._runtime_guard:
            raise ValueError('the frozen small AR1 noise point changed')

    def whiten_prefix(self, values):
        """Whiten all leading native rows, preserving every trailing column."""
        raw = finite_real_array(values, name='complete leading native AR1 observations')
        c = len(self.phi)
        if raw.ndim < 2 or raw.shape[1] != c or not len(raw):
            raise ValueError('small AR1 prefix retains native-time then complete channel order')
        shape, columns = raw.shape, int(np.prod(raw.shape[2:]))
        raw = raw.reshape(len(raw), c, columns)
        white = np.empty_like(raw)
        white[0] = solve_triangular(self.initial_root, raw[0], lower=True)
        if len(raw) > 1:
            residual = raw[1:]-self.phi[None, :, None]*raw[:-1]
            image = solve_triangular(self.innovation_root, residual.transpose(1, 0, 2).reshape(c, -1), lower=True)
            white[1:] = image.reshape(c, len(raw)-1, columns).transpose(1, 0, 2)
        return white.reshape(shape)

    def whitening_logabsdet(self, native_samples):
        if type(native_samples) is not int or native_samples < 0:
            raise ValueError('AR1 log Jacobian uses an original nonnegative native row count')
        if not native_samples:
            return 0.
        return -float(np.log(np.diag(self.initial_root)).sum()
            +(native_samples-1)*np.log(np.diag(self.innovation_root)).sum())

    def small_factor_certificate(self):
        """Exact-law small-factor pullback bounds; no full-prefix certificate."""
        phi, q = [Fraction.from_float(float(value)) for value in self.phi], _fraction_matrix(self.innovation_covariance)
        stationary = [[q[i][j]/(1-phi[i]*phi[j]) for j in range(len(phi))] for i in range(len(phi))]
        initial_error = _small_factor_certificate(self.initial_root, stationary)[0]
        innovation_error = _small_factor_certificate(self.innovation_root, q)[0]
        policy = 1e-8*np.sqrt(len(phi))
        if max(float(initial_error), float(innovation_error)) > policy:
            raise NumericalResolutionError('the saved small AR1 factors do not resolve their complete covariance law')
        return {'schema': 'private_AR1_small_factor_exact_law_pullback_certificate/1',
            'stationary_law': 'S_cd=Q_cd/(1-phi_c*phi_d)',
            'initial_covariance_pullback_F_bound': _upper(initial_error),
            'innovation_covariance_pullback_F_bound': _upper(innovation_error),
            'per_block_covariance_pullback_F_policy': policy,
            'whole_record_covariance_pullback_squared_bound': 'initial_bound^2+(native_samples-1)*innovation_bound^2',
            'native_time_then_complete_channel_order': True, 'legacy_prefix_root_reused': False}

    def freeze(self, provenance):
        """Freeze the exact executed small factors, with an independent law bound."""
        return replace(self, provenance=provenance, certificate=self.small_factor_certificate())

    def record(self):
        return {'schema': PROGRAMME, 'producer': IMPLEMENTATION_SHA256, 'phi': self.phi.tolist(),
            'innovation_covariance': self.innovation_covariance.tolist(), 'initial_root': self.initial_root.tolist(),
            'innovation_root': self.innovation_root.tolist(), 'geometry': self.geometry.record(),
            'conditioning_samples': self.conditioning_samples, 'provenance': plain_metadata(self.provenance),
            'certificate': plain_metadata(self.certificate), 'identity': self._runtime_state_digest(),
            'numeric_only_no_raw_label_or_fit_grant': True}

    def save(self, path):
        content = json.dumps(self.record(), sort_keys=True, allow_nan=False).encode()
        Path(path).write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def from_record(cls, record):
        if (record['schema'] != PROGRAMME or record['producer'] != IMPLEMENTATION_SHA256
                or record['numeric_only_no_raw_label_or_fit_grant'] is not True):
            raise ValueError('small AR1 codec retains its own numeric-only producer')
        value = cls(record['phi'], record['innovation_covariance'], NoiseGeometry(**record['geometry']),
            record['conditioning_samples'], record['provenance'], record['initial_root'], record['innovation_root'], record['certificate'])
        if value._runtime_state_digest() != record['identity']:
            raise ValueError('small AR1 codec changed its actual complete numeric producer state')
        return value

    @classmethod
    def load(cls, path, *, expected_sha256):
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('small AR1 numeric artifact changed its externally frozen bytes')
        return cls.from_record(json.loads(content))
