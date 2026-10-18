"""Streamed original-noise rows -> analytically integrated local Gaussians.

Shared source coordinates and physical record drift remain external. Every local
column has one independent standard-normal prior. This numerical kernel accepts
explicit parameter laws; it fits nothing and grants no observation authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_response_scale import NumericalResolutionError, _augmented_qr
from research.prepared_continuous_frontier import _white_block
from research.response_frontier import LogFactor
from research.structured_response_frontier import StoredNoiseCarrier
from transfer.noise_model import array_digest, finite_real_array, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SharedResponseChart:
    mean: np.ndarray
    factor: np.ndarray
    coordinate_ids: tuple
    channel_ids: tuple
    parameter_binding: dict

    def __post_init__(self):
        mean = finite_real_array(self.mean, name='external shared response mean')
        factor = finite_real_array(self.factor, name='external complete rectangular shared factor')
        coordinates, channels = tuple(self.coordinate_ids), tuple(self.channel_ids)
        if (mean.ndim != 2 or factor.ndim != 3 or factor.shape[:2] != mean.shape
                or factor.shape[2] != len(coordinates) or len(set(coordinates)) != len(coordinates)
                or len(channels) != mean.shape[1] or len(set(channels)) != len(channels)):
            raise ValueError('shared chart requires every named source column and native coefficient/channel coordinate')
        object.__setattr__(self, 'mean', frozen_array(mean))
        object.__setattr__(self, 'factor', frozen_array(factor))
        object.__setattr__(self, 'coordinate_ids', coordinates)
        object.__setattr__(self, 'channel_ids', channels)
        object.__setattr__(self, 'parameter_binding', freeze_metadata(self.parameter_binding))

    @property
    def rank(self):
        return self.factor.shape[2]

    @property
    def identity(self):
        return metadata_digest({'mean_factor': array_digest(self.mean, self.factor),
            'coordinates': self.coordinate_ids, 'channels': self.channel_ids,
            'external_parameter_binding': self.parameter_binding})


@dataclass(frozen=True)
class IndependentLocalResponse:
    selection_id: str
    factor: np.ndarray
    coordinate_ids: tuple
    parameter_binding: dict

    def __post_init__(self):
        factor = finite_real_array(self.factor, name='external zero-mean local rectangular factor')
        coordinates = tuple(self.coordinate_ids)
        if (not self.selection_id or factor.ndim != 3 or factor.shape[2] != len(coordinates)
                or len(set(coordinates)) != len(coordinates)):
            raise ValueError('local response keeps its independent selection and complete named factor columns')
        object.__setattr__(self, 'factor', frozen_array(factor))
        object.__setattr__(self, 'coordinate_ids', coordinates)
        object.__setattr__(self, 'parameter_binding', freeze_metadata(self.parameter_binding))

    @property
    def identity(self):
        return metadata_digest({'selection': self.selection_id, 'zero_mean': True,
            'factor': array_digest(self.factor), 'coordinates': self.coordinate_ids,
            'external_parameter_binding': self.parameter_binding, 'independent_of_other_selections': True})


@dataclass(frozen=True)
class GaussianLocalPotential:
    """Likelihood only in x=(one shared u, one original-record drift a)."""

    residual_root: np.ndarray
    log_normalizer: float
    shared_dimension: int
    drift_dimension: int
    local_dimensions: tuple
    binding: dict

    def __post_init__(self):
        root = finite_real_array(self.residual_root, name='local-integrated complete residual root')
        if (root.ndim != 2 or root.shape[1] != self.shared_dimension+self.drift_dimension+1
                or min(self.shared_dimension, self.drift_dimension) < 0 or not np.isfinite(self.log_normalizer)):
            raise ValueError('compact potential keeps all shared/drift coordinates and its residual column')
        object.__setattr__(self, 'residual_root', frozen_array(root))
        object.__setattr__(self, 'local_dimensions', tuple(tuple(value) for value in self.local_dimensions))
        object.__setattr__(self, 'binding', freeze_metadata(self.binding))

    @property
    def identity(self):
        return metadata_digest({'residual_root': array_digest(self.residual_root),
            'log_normalizer': self.log_normalizer, 'shared_dimension': self.shared_dimension,
            'drift_dimension': self.drift_dimension, 'local_dimensions': self.local_dimensions,
            'binding': self.binding})

    def log_value(self, shared, drift):
        shared, drift = np.asarray(shared, dtype=float), np.asarray(drift, dtype=float)
        if (shared.shape != (self.shared_dimension,) or drift.shape != (self.drift_dimension,)
                or not np.isfinite(shared).all() or not np.isfinite(drift).all()):
            raise ValueError('query retains the complete common source and original-record drift point')
        try:
            with np.errstate(over='raise', invalid='raise'):
                residual = self.residual_root @ np.r_[shared, drift, -1.]
                return float(self.log_normalizer - .5 * (residual @ residual))
        except FloatingPointError as error:
            raise NumericalResolutionError('compact Gaussian log potential is outside binary64 range') from error

    def save(self, path):
        path = Path(path)
        record = {'schema': 'compact_local_Gaussian_potential/1', 'identity': self.identity,
            'log_normalizer': self.log_normalizer, 'shared_dimension': self.shared_dimension,
            'drift_dimension': self.drift_dimension, 'local_dimensions': self.local_dimensions,
            'binding': plain_metadata(self.binding), 'numeric_only_no_raw_or_fit_authority': True}
        np.savez_compressed(path, residual_root=self.residual_root, metadata=np.array(json.dumps(record, allow_nan=False)))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, *, expected_sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('compact local potential changed its external saved byte identity')
        with np.load(path, allow_pickle=False) as archive:
            record = json.loads(str(archive['metadata']))
            if record['schema'] != 'compact_local_Gaussian_potential/1' or record['numeric_only_no_raw_or_fit_authority'] is not True:
                raise ValueError('compact local potential requires its original numeric-only role')
            result = cls(archive['residual_root'], record['log_normalizer'], record['shared_dimension'],
                record['drift_dimension'], record['local_dimensions'], record['binding'])
        if result.identity != record['identity']:
            raise ValueError('compact local potential changed its complete root, constants or latent chart')
        return result


def integrate_local_white_rows(blocks, *, local_dimensions, shared_dimension, drift_dimension,
                               observed_scalar_rows, log_noise_jacobian, binding, qr_block_rows):
    """Same augmented-QR owner, with only the declared local proper prior.

    Blocks contain [all independent local columns | shared | drift | y-mean].
    Raw observation count and noise Jacobian belong to original rows, never the
    compressed root height. Local rank deficiency retains its full prior I.
    """
    dimensions = tuple(tuple(value) for value in local_dimensions)
    if (len({name for name, _ in dimensions}) != len(dimensions)
            or any(type(width) is not int or width < 0 for _, width in dimensions)
            or shared_dimension < 0 or drift_dimension < 0 or observed_scalar_rows < 1
            or not np.isfinite(log_noise_jacobian)):
        raise ValueError('local integration needs complete distinct local charts and original row measure')
    local_width = sum(width for _, width in dimensions)
    width = local_width+shared_dimension+drift_dimension+1
    prior = np.zeros((local_width, width))
    prior[:, :local_width] = np.eye(local_width)
    root, count = prior, 0
    for block in blocks:
        values = finite_real_array(block, name='original whitened local/shared/drift/residual block')
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError('every white block keeps the same complete column order')
        count += len(values)
        root = _augmented_qr(np.vstack((root, values)), qr_block_rows)
    if count != observed_scalar_rows:
        raise ValueError('QR compression cannot replace the original complete observation row count')
    return _finish_local_integral(root, dimensions, shared_dimension, drift_dimension, count,
        log_noise_jacobian, {**plain_metadata(binding), 'qr_block_rows': qr_block_rows,
                            'input_rows_role': 'original_white_observation_rows'})


def integrate_local_augmented_root(likelihood_root, *, local_dimensions, shared_dimension,
        drift_dimension, observed_scalar_rows, log_noise_jacobian, binding):
    """Consume an existing complete DATA QR root, retaining its original n.

    Column order is [local | shared | drift | residual], with no priors already
    inserted. This is a numeric operand seam for existing Gram/QR owners; it
    creates no raw authority. The final residual-only direction must be retained.
    """
    dimensions = tuple(tuple(value) for value in local_dimensions)
    local_width = sum(width for _, width in dimensions)
    width = local_width+shared_dimension+drift_dimension+1
    likelihood = finite_real_array(likelihood_root, name='complete prior-free likelihood augmented root')
    if (likelihood.ndim != 2 or likelihood.shape[1] != width or len(likelihood) > observed_scalar_rows
            or observed_scalar_rows < 1 or len({name for name, _ in dimensions}) != len(dimensions)
            or any(type(value) is not int or value < 0 for _, value in dimensions)
            or min(shared_dimension, drift_dimension) < 0 or not np.isfinite(log_noise_jacobian)):
        raise ValueError('compact data root must retain all named columns, original scalar rows and noise measure')
    prior = np.zeros((local_width, width))
    prior[:, :local_width] = np.eye(local_width)
    operands = np.vstack((prior, likelihood))
    root = _augmented_qr(operands, max(1, len(operands)))
    return _finish_local_integral(root, dimensions, shared_dimension, drift_dimension,
        observed_scalar_rows, log_noise_jacobian, {**plain_metadata(binding),
            'input_rows_role': 'complete_prior_free_augmented_likelihood_root',
            'input_likelihood_root': array_digest(likelihood), 'compressed_likelihood_rows': len(likelihood)})


def _finish_local_integral(root, dimensions, shared_dimension, drift_dimension, observed_scalar_rows,
                            log_noise_jacobian, binding):
    local_width = sum(width for _, width in dimensions)
    local_root = root[:local_width, :local_width]
    diagonal = np.abs(np.diag(local_root))
    if len(diagonal) != local_width or np.any(diagonal == 0):
        raise ValueError('proper local-prior augmented QR is unresolved')
    local_log_volume = float(np.log(diagonal).sum())
    normalizer = float(log_noise_jacobian-.5*observed_scalar_rows*np.log(2*np.pi)-local_log_volume)
    return GaussianLocalPotential(root[local_width:, local_width:], normalizer,
        shared_dimension, drift_dimension, dimensions, {**plain_metadata(binding),
            'observed_scalar_rows': observed_scalar_rows, 'independent_local_dimensions': dimensions,
            'local_prior_count_per_declared_block': 1, 'shared_source_prior_count_here': 0,
            'physical_drift_prior_or_integral_count_here': 0,
            'local_log_volume': local_log_volume, 'original_log_noise_jacobian': float(log_noise_jacobian),
            'numeric_role': 'binary64_reference',
            'finite_sample_or_absolute_nats_certificate': False, 'producer': IMPLEMENTATION_SHA256})


def _intervals(values, count):
    intervals = tuple(tuple(value) for value in values)
    if (not intervals or any(len(value) != 2 or any(type(i) is not int for i in value)
            or not 0 <= value[0] < value[1] <= count for value in intervals)
            or any(right[0] < left[1] for left, right in zip(intervals[:-1], intervals[1:], strict=True))):
        raise ValueError('component observation rows are ordered disjoint intervals of the original record')
    return intervals


def compile_native_local_component(samples, carrier, shared_chart, temporal_callbacks, local_responses, *,
        native_intervals, record_id, drift_group_id, drift_coordinate_ids, nuisance_callback,
        branch_binding, block_native_rows=256):
    """One branch, one white-row component, using original startup/AR history.

    temporal_callbacks[name](start,stop) -> (D,D_error,original_design_binding).
    All callbacks in the component contribute to the one shared mean/factor.
    Each named local response gets separate columns even when factors coincide.
    nuisance_callback(start,stop) returns native [sample,channel,drift] columns.
    Full-record samples stay external; only current work blocks are allocated.
    """
    if type(carrier) is not StoredNoiseCarrier or type(shared_chart) is not SharedResponseChart:
        raise TypeError('native component consumes the original stored noise carrier and explicit shared chart')
    y = finite_real_array(samples, name='original complete native record observations')
    channels, coefficients = len(shared_chart.channel_ids), shared_chart.mean.shape[0]
    if (y.ndim != 2 or not len(y) or y.shape[1] != channels
            or carrier.noise.geometry.channels != shared_chart.channel_ids
            or type(block_native_rows) is not int or block_native_rows < 1
            or not record_id or not drift_group_id):
        raise ValueError('native component requires original channels, recording, drift group and work rows')
    intervals = _intervals(native_intervals, len(y))
    callbacks, locals_ = dict(temporal_callbacks), tuple(local_responses)
    if (not callbacks or len({value.selection_id for value in locals_}) != len(locals_)
            or any(type(value) is not IndependentLocalResponse or value.selection_id not in callbacks
                or value.factor.shape[:2] != shared_chart.mean.shape for value in locals_)):
        raise ValueError('local factor blocks name distinct actual selections in the shared coefficient chart')
    drift_ids = tuple(drift_coordinate_ids)
    if len(set(drift_ids)) != len(drift_ids) or (bool(drift_ids) != (nuisance_callback is not None)):
        raise ValueError('declare the original record drift columns explicitly, including an explicit empty control')
    local_dimensions = tuple((value.selection_id, value.factor.shape[2]) for value in locals_)
    local_width, rank, drift_width = sum(width for _, width in local_dimensions), shared_chart.rank, len(drift_ids)
    row_bindings = []
    input_identity = array_digest(y)
    noise_map = SimpleNamespace(carrier=carrier)

    def blocks():
        for begin, interval_stop in intervals:
            for start in range(begin, interval_stop, block_native_rows):
                end = min(start+block_native_rows, interval_stop)
                raw_start = 0 if start < carrier.noise.conditioning_samples else max(0, start-carrier.noise.order)
                designs, row_ledger = {}, {}
                total = np.zeros((end-raw_start, coefficients))
                for name, callback in callbacks.items():
                    design, error, design_binding = callback(raw_start, end)
                    design = finite_real_array(design, name='original native temporal response design')
                    error = finite_real_array(error, name='original native temporal response design error')
                    if design.shape != total.shape or error.shape != design.shape or np.any(error < 0):
                        raise ValueError('every response retains its full temporal coefficient/error coordinates')
                    designs[name] = design
                    total += design
                    row_ledger[name] = {'design_error_sha256': array_digest(design, error),
                        'original_binding': metadata_digest(design_binding)}
                raw = np.empty((end-raw_start, channels, local_width+rank+drift_width+1))
                offset = 0
                for response in locals_:
                    stop = offset+response.factor.shape[2]
                    raw[:, :, offset:stop] = np.einsum('np,pcr->ncr', designs[response.selection_id], response.factor)
                    offset = stop
                raw[:, :, offset:offset+rank] = np.einsum('np,pcr->ncr', total, shared_chart.factor)
                offset += rank
                if drift_width:
                    nuisance = finite_real_array(nuisance_callback(raw_start, end), name='original physical drift columns')
                    if nuisance.shape != (end-raw_start, channels, drift_width):
                        raise ValueError('physical drift callback changed its complete original coordinate chart')
                    raw[:, :, offset:offset+drift_width] = nuisance
                raw[:, :, -1] = y[raw_start:end] - total @ shared_chart.mean
                white = _white_block(noise_map, raw, raw_start, start, end)
                row_bindings.append({'observed_native_rows': [start, end],
                    'actual_predecessor_read_from_original_rows': [raw_start, start], 'designs': row_ledger,
                    'original_white_block_sha256': array_digest(white)})
                yield white

    logwhite = sum(carrier.leading_log_jacobian(stop)
        - (carrier.leading_log_jacobian(start) if start else 0.) for start, stop in intervals)
    binding = {'schema': 'native_gaussian_local_component/1', 'record_id': record_id,
        'original_observation_sha256': input_identity, 'original_native_samples': len(y),
        'native_intervals': intervals, 'original_noise_carrier': carrier.identity,
        'shared_response_chart': shared_chart.identity, 'shared_coordinate_ids': shared_chart.coordinate_ids,
        'drift_group_id': drift_group_id, 'drift_coordinate_ids': drift_ids,
        'local_response_laws': {value.selection_id: value.identity for value in locals_},
        'branch_binding': branch_binding, 'original_white_blocks': row_bindings,
        'source_noise_or_local_covariance_fits': 0, 'new_observation_grants': 0}
    # The generator appends its actual white-row ledger while QR consumes it.
    result = integrate_local_white_rows(blocks(), local_dimensions=local_dimensions,
        shared_dimension=rank, drift_dimension=drift_width,
        observed_scalar_rows=sum(stop-start for start, stop in intervals)*channels,
        log_noise_jacobian=logwhite, binding=binding, qr_block_rows=max(1, block_native_rows*channels))
    if array_digest(y) != input_identity:
        raise ValueError('original numeric observations changed while compiling their component')
    return result


class GaussianCandidateFactor:
    """Complete declared roster table -> existing pointwise LogFactor."""

    def __init__(self, rosters, scope, table):
        self.scope = tuple(scope)
        if (self.scope != tuple(sorted(set(self.scope))) or not self.scope
                or any(not 0 <= index < len(rosters) for index in self.scope)):
            raise ValueError('candidate factor names its complete ordered original variable scope')
        keys = tuple(product(*(rosters[index].states for index in self.scope)))
        if set(table) != set(keys) or any(type(value) is not GaussianLocalPotential for value in table.values()):
            raise ValueError('candidate factor keeps one integrated potential for every declared positive state')
        self.keys, self.potentials = keys, tuple(table[key] for key in keys)
        self.shape = tuple(len(rosters[index].states) for index in self.scope)
        first = self.potentials[0]
        fields = ('record_id', 'original_observation_sha256', 'original_native_samples', 'native_intervals',
            'original_noise_carrier', 'shared_response_chart', 'shared_coordinate_ids',
            'drift_group_id', 'drift_coordinate_ids', 'local_response_laws')
        for value in self.potentials[1:]:
            if (value.shared_dimension != first.shared_dimension or value.drift_dimension != first.drift_dimension
                    or value.local_dimensions != first.local_dimensions
                    or any(metadata_digest(value.binding[name]) != metadata_digest(first.binding[name]) for name in fields)):
                raise ValueError('candidate alternatives must keep the same observations, priors, noise and drift measure')
        self.identity = metadata_digest({'scope': self.scope, 'states': [[vars(state) for state in key] for key in keys],
            'potentials': [value.identity for value in self.potentials], 'producer': IMPLEMENTATION_SHA256})

    def at(self, shared, drift):
        return LogFactor(self.scope, np.array([value.log_value(shared, drift)
            for value in self.potentials]).reshape(self.shape))


def require_once_record_partition(components, *, record_id, original_native_samples):
    """Validate a likelihood product before sum/product: each white row once."""
    representatives = tuple(value.potentials[0] if type(value) is GaussianCandidateFactor else value for value in components)
    if not representatives or any(type(value) is not GaussianLocalPotential for value in representatives):
        raise TypeError('record partition requires actual integrated numeric potentials/candidate factors')
    first = representatives[0]
    invariant = ('original_observation_sha256', 'original_noise_carrier', 'shared_response_chart',
                 'shared_coordinate_ids', 'drift_group_id', 'drift_coordinate_ids')
    for value in representatives:
        if (value.binding['record_id'] != record_id or value.binding['original_native_samples'] != original_native_samples
                or any(metadata_digest(value.binding[name]) != metadata_digest(first.binding[name]) for name in invariant)):
            raise ValueError('one acquisition uses the same original noise, source and physical drift coordinates')
    intervals = sorted(tuple(interval) for value in representatives for interval in value.binding['native_intervals'])
    local_owners = [name for value in representatives for name, width in value.local_dimensions if width]
    if len(set(local_owners)) != len(local_owners):
        raise ValueError('one selection-local Gaussian must be integrated jointly across every component row it touches')
    stop = 0
    for start, end in intervals:
        if start != stop:
            raise ValueError('component likelihoods must partition every original observation once, without overlaps or gaps')
        stop = end
    if stop != original_native_samples:
        raise ValueError('the complete original-record denominator is not covered')
    return freeze_metadata({'record_id': record_id, 'native_samples': stop, 'native_intervals': intervals,
        'original_noise_startups': 1, 'physical_drift_groups': 1,
        'local_Gaussian_integrals': [value.local_dimensions for value in representatives],
        'shared_source_prior_count_here': 0, 'component_identities': [value.identity for value in representatives]})
