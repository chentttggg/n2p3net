"""Joint C marginal, residual moments and private-AR1/full-Q numeric plugin.

All source, reference and observation components are explicit numerical inputs.
Live original-data admission belongs to the public programme owner. Existing
residual-noise artifacts are accepted only as optional numeric warm starts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.gaussian_local_factor import (
    SharedResponseChart,
    require_once_record_partition,
)
from research.joint_response_noise_reference import TwoExposureWithinReference
from research.joint_response_scale import NumericalResolutionError, _augmented_qr
from research.private_ar1_noise_point import UnrepresentableJointParameterPointError
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    frozen_array,
    metadata_digest,
)
from transfer.temporal_noise import TemporalNoise

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LOG2PI = float(np.log(2*np.pi))


def known_c_bit_design_proof(event_ids, membership, bits, candidate_vocabulary, candidate_prior):
    """All compatible q give the same design on every granted C exposure."""
    events, vocabulary = tuple(event_ids), tuple(candidate_vocabulary)
    matrix = finite_real_array(membership, name='complete original C membership')
    labels = np.asarray(bits)
    prior = finite_real_array(candidate_prior, name='complete original candidate prior')
    if (len(set(events)) != len(events) or len(set(vocabulary)) != len(vocabulary) or len(vocabulary) < 2
            or matrix.shape != (len(events), len(vocabulary)) or labels.shape != (len(events),)
            or not ((matrix == 0) | (matrix == 1)).all() or not ((labels == 0) | (labels == 1)).all()
            or prior.shape != (len(vocabulary),) or np.any(prior <= 0)
            or not np.isclose(prior.sum(), 1., rtol=1e-12, atol=0)):
        raise ValueError('one granted bit per exposure and the original complete candidate axis are required')
    compatible = np.all(matrix == labels[:, None], axis=0)
    if not compatible.any():
        raise ValueError('granted C bits contradict the original membership')
    return freeze_metadata({'schema': 'known_C_bit_design_equivalence/1', 'event_ids': events,
        'candidate_vocabulary': vocabulary, 'compatible_candidates': tuple(vocabulary[i] for i in np.flatnonzero(compatible)),
        'compatibility_mask': compatible.tolist(), 'original_bits_sha256': array_digest(labels),
        'membership_sha256': array_digest(matrix), 'full_candidate_axis_retained': True,
        'common_design': 'sum_all_granted_exposure_bases',
        'increment_design': 'sum_original_bit_times_exposure_basis',
        'conditional_EEG_given_bits_q_sum': 1.,
        'joint_granted_bits_prior_mass': float(prior[compatible].sum()),
        'raw_or_label_authority_granted': False})


@dataclass(frozen=True)
class JointMarginal:
    log_density: float
    upper: np.ndarray
    rhs: np.ndarray
    shared_dimension: int
    drift_coordinate_ids: tuple
    binding: dict

    def __post_init__(self):
        upper = finite_real_array(self.upper, name='shared-source/flat posterior upper')
        rhs = finite_real_array(self.rhs, name='shared-source/flat posterior rhs')
        if (not np.isfinite(self.log_density) or upper.shape != (len(rhs), len(rhs))
                or np.any(np.diag(upper) == 0)):
            raise NumericalResolutionError('complete shared/physical-flat posterior is unresolved')
        object.__setattr__(self, 'upper', frozen_array(upper))
        object.__setattr__(self, 'rhs', frozen_array(rhs))
        object.__setattr__(self, 'drift_coordinate_ids', tuple(self.drift_coordinate_ids))
        object.__setattr__(self, 'binding', freeze_metadata(self.binding))

    @cached_property
    def mean(self):
        return solve_triangular(self.upper, self.rhs)

    @cached_property
    def factor(self):
        return solve_triangular(self.upper, np.eye(len(self.rhs)))

    def component_indices(self, drift_ids):
        return tuple(range(self.shared_dimension))+tuple(self.shared_dimension+
            self.drift_coordinate_ids.index(name) for name in drift_ids)


def joint_calibration_marginal(potentials, *, source_chart, record_samples, input_binding):
    """Combine local-integrated likelihoods, one source prior and physical da."""
    values = tuple(potentials)
    if type(source_chart) is not SharedResponseChart or not values:
        raise TypeError('explicit full source chart and original local-integrated components are required')
    records = dict(record_samples)
    if {value.binding['record_id'] for value in values} != set(records):
        raise ValueError('joint calibration retains every declared original recording')
    partitions = [require_once_record_partition(tuple(v for v in values if v.binding['record_id'] == name),
        record_id=name, original_native_samples=count) for name, count in records.items()]
    drift_ids = tuple(dict.fromkeys(name for value in values for name in value.binding['drift_coordinate_ids']))
    owners = [name for value in values for name, width in value.local_dimensions if width]
    if len(set(owners)) != len(owners):
        raise ValueError('each local Gaussian is integrated once across its entire observed footprint')
    rank, width = source_chart.rank, source_chart.rank+len(drift_ids)
    root = np.zeros((rank, width+1))
    root[:, :rank] = np.eye(rank)
    log_normalizer = 0.
    for value in values:
        if (value.shared_dimension != rank or value.binding['shared_response_chart'] != source_chart.identity
                or tuple(value.binding['shared_coordinate_ids']) != source_chart.coordinate_ids):
            raise ValueError('all C components share the same complete original source coordinates')
        columns = tuple(range(rank))+tuple(rank+drift_ids.index(name)
            for name in value.binding['drift_coordinate_ids'])+(width,)
        block = np.zeros((len(value.residual_root), width+1))
        block[:, columns] = value.residual_root
        root = _augmented_qr(np.vstack((root, block)), max(1, len(root)+len(block)))
        log_normalizer += value.log_normalizer
    if len(root) < width or np.any(np.diag(root[:width, :width]) == 0):
        raise NumericalResolutionError('actual C does not identify its original physical flat coordinates')
    upper, rhs = root[:width, :width], root[:width, width]
    residual = root[width:, width]
    density = (log_normalizer+.5*len(drift_ids)*_LOG2PI
        -float(np.log(np.abs(np.diag(upper))).sum())-.5*float(residual@residual))
    return JointMarginal(density, upper, rhs, rank, drift_ids, {
        'schema': 'joint_C_shared_source_local_response_physical_flat_marginal/1',
        'source_chart_identity': source_chart.identity, 'input_binding': input_binding,
        'record_partitions': partitions, 'component_identities': [value.identity for value in values],
        'source_prior_count': 1, 'C_observation_union_count': 1,
        'physical_flat_coordinate_ids': drift_ids, 'residual_squared': float(residual@residual),
        'raw_or_fit_authority_granted': False, 'producer_sha256': IMPLEMENTATION_SHA256})


@dataclass(frozen=True)
class ResidualMoments:
    mean: np.ndarray
    factor: np.ndarray

    def __post_init__(self):
        mean = finite_real_array(self.mean, name='complete posterior residual mean')
        factor = finite_real_array(self.factor, name='complete posterior residual uncertainty factor')
        if mean.ndim != 1 or factor.ndim != 2 or len(mean) != len(factor):
            raise ValueError('residual moments preserve all original scalar observation rows')
        object.__setattr__(self, 'mean', frozen_array(mean))
        object.__setattr__(self, 'factor', frozen_array(factor))

    def cross_moment(self, left, right):
        return np.outer(self.mean[left], self.mean[right])+self.factor[left]@self.factor[right].T

    def apply_second_moment(self, values):
        values = np.asarray(values, dtype=float)
        mean_image = (self.mean*(self.mean@values) if values.ndim == 1
            else self.mean[:, None]*(self.mean@values)[None, :])
        return mean_image+self.factor@(self.factor.T@values)


def component_posterior_moments(centered_observation, shared_drift_design, local_design,
        *, local_upper, local_cross, local_rhs, component_mean, component_factor):
    """Recover full E[eeT] from the existing local QR's conditional factors.

    g has the true shared-source/record-drift posterior. v|g has mean
    Rvv^-1(rhs-Rvg*g) and independent covariance Rvv^-1 Rvv^-T.
    No global all-local covariance or second local integration is required.
    """
    residual = finite_real_array(centered_observation, name='original centered observations')
    h = finite_real_array(shared_drift_design, name='original shared/flat observation design')
    local = finite_real_array(local_design, name='original whitened or native local design')
    inverse = solve_triangular(local_upper, np.eye(len(local_rhs)))
    regression = solve_triangular(local_upper, local_cross)
    local_mean = solve_triangular(local_upper, local_rhs)-regression@component_mean
    local_factor = np.column_stack((-regression@component_factor, inverse))
    mean_e = residual-h@component_mean-local@local_mean
    factor_e = np.column_stack(((h-local@regression)@component_factor, local@inverse))
    expected_local_square = float(local_mean@local_mean+np.sum(local_factor*local_factor))
    return ResidualMoments(mean_e, factor_e), expected_local_square


@dataclass(frozen=True)
class AR1ResidualStatistics:
    record_id: str
    native_samples: int
    initial: np.ndarray
    previous: np.ndarray
    lag: np.ndarray
    current: np.ndarray


@dataclass(frozen=True)
class AR1InnovationStatistics:
    """Direct E[wwT] and E[w e_previousT], w=e_current-diag(phi)e_previous.

    These are formed from posterior means/factors before second moments are
    accumulated. They are distinct from the complete raw residual diagnostics.
    """

    record_id: str
    native_samples: int
    initial: np.ndarray
    innovation_second: np.ndarray
    innovation_previous: np.ndarray

    def __post_init__(self):
        initial = finite_real_array(self.initial, name='one original stationary startup moment')
        second = finite_real_array(self.innovation_second, name='direct complete innovation second moment')
        cross = finite_real_array(self.innovation_previous, name='direct complete innovation/previous cross moment')
        if (type(self.native_samples) is not int or self.native_samples < 1
                or initial.ndim != 2 or initial.shape[0] != initial.shape[1]
                or second.shape != initial.shape or cross.shape != initial.shape):
            raise ValueError('direct AR1 moments retain one complete original channel grid and record')
        for name, value in (('initial', initial), ('innovation_second', second), ('innovation_previous', cross)):
            object.__setattr__(self, name, frozen_array(value))


def ar1_residual_statistics(moments, *, channels, record_id):
    """One stationary initial moment per record, all consecutive tail moments."""
    if len(moments.mean) % channels:
        raise ValueError('residual scalar rows must preserve their original native channel grid')
    count = len(moments.mean)//channels
    if count < 1:
        raise ValueError('a complete original record has a stationary initial row')
    mean = moments.mean.reshape(count, channels)
    factor = moments.factor.reshape(count, channels, -1)
    def moment(a, b):
        return np.einsum('tc,td->cd', mean[a], mean[b])+np.einsum('tck,tdk->cd', factor[a], factor[b])
    return AR1ResidualStatistics(record_id, count, np.outer(mean[0], mean[0])+factor[0]@factor[0].T,
        moment(slice(None, -1), slice(None, -1)), moment(slice(1, None), slice(None, -1)),
        moment(slice(1, None), slice(1, None)))


@dataclass(frozen=True)
class PrivateAR1Coordinates:
    geometry: NoiseGeometry
    reference_innovation_root: np.ndarray
    conditioning_samples: int
    reference_binding: dict
    numeric_point_programme: str = 'legacy_markov_full_prefix'

    def __post_init__(self):
        root = finite_real_array(self.reference_innovation_root, name='fixed noise unit/reference root')
        c = len(self.geometry.channels)
        if (root.shape != (c, c) or np.any(np.triu(root, 1) != 0) or np.any(np.diag(root) <= 0)
                or type(self.conditioning_samples) is not int or self.conditioning_samples < 2
                or self.numeric_point_programme not in {'legacy_markov_full_prefix', 'small_stationary_AR1_innovations'}):
            raise ValueError('private AR1 coordinates need an explicit proper physical noise reference')
        object.__setattr__(self, 'reference_innovation_root', frozen_array(root))
        object.__setattr__(self, 'reference_binding', freeze_metadata(self.reference_binding))

    @property
    def dimension(self):
        c = len(self.geometry.channels)
        return c+c*(c+1)//2

    @cached_property
    def identity(self):
        if self.numeric_point_programme != 'legacy_markov_full_prefix':
            from research.private_ar1_noise_point import (
                IMPLEMENTATION_SHA256 as POINT_PRODUCER_SHA256,
            )
        else:
            POINT_PRODUCER_SHA256 = None
        programme = ({'numeric_point_programme': self.numeric_point_programme}
            if self.numeric_point_programme != 'legacy_markov_full_prefix' else {})
        if POINT_PRODUCER_SHA256 is not None:
            programme['numeric_point_producer'] = POINT_PRODUCER_SHA256
        return metadata_digest({'geometry': self.geometry.record(), 'conditioning': self.conditioning_samples, **programme,
            'reference_root': array_digest(self.reference_innovation_root), 'reference': self.reference_binding,
            'family': 'private_AR1_full_innovation_covariance', 'eta': 1.,
            'coordinates': 'atanh_AR_and_relative_lower_logdiagonal_cholesky',
            'noise_parameter_role': 'maximum_likelihood_plugin_no_noise_hyperprior'})

    def point(self, parameters):
        parameters = finite_real_array(parameters, name='dimensionless private AR1/full-Q coordinates')
        c = len(self.geometry.channels)
        if parameters.shape != (self.dimension,):
            raise ValueError('all AR and full-Q coordinates are explicit')
        phi, relative = np.tanh(parameters[:c]), np.zeros((c, c))
        with np.errstate(over='raise', under='raise', invalid='raise'):
            for value, (i, j) in zip(parameters[c:], zip(*np.tril_indices(c), strict=True), strict=True):
                relative[i, j] = np.exp(value) if i == j else value
            physical = self.reference_innovation_root@relative
            covariance = physical@physical.T
        if self.numeric_point_programme == 'small_stationary_AR1_innovations':
            from research.private_ar1_noise_point import PrivateAR1NoisePoint

            noise = PrivateAR1NoisePoint(phi, covariance, self.geometry, self.conditioning_samples,
                {'fit_scope': 'numeric_parameter_evaluation', 'joint_parameterization': self.identity})
            return noise, phi, relative
        # TemporalNoise stores the AR polynomial reflection: for AR1 k=-phi.
        temporal = tuple(TemporalNoise.from_reflections([-value], innovation_variance=1.,
            reference_id='joint_C_private_AR1_numeric_point') for value in phi)
        noise = FittedChannelNoise.from_markov_stationary_factors(temporal, covariance,
            geometry=self.geometry, conditioning_samples=self.conditioning_samples, family='private',
            provenance={'fit_scope': 'analytic_reference', 'reference_id': 'joint_C_numeric_parameter_evaluation',
                'records': [], 'joint_parameterization': self.identity})
        return noise, phi, relative

    def warm_start(self, old_noise):
        old_noise.assert_integrity()
        if old_noise.geometry != self.geometry or old_noise.order != 1:
            raise ValueError('warm start requires the actual matching AR1/native geometry')
        relative = solve_triangular(self.reference_innovation_root, old_noise._spatial_root, lower=True)
        lower = [np.log(relative[i, j]) if i == j else relative[i, j]
            for i, j in zip(*np.tril_indices(len(relative)), strict=True)]
        return np.r_[np.arctanh([item.coefficients[0] for item in old_noise.temporal]), lower]

    def score(self, statistics, parameters, point):
        """Fisher identity score with stationary startup and complete E[eeT]."""
        noise, phi, relative = point
        q = noise.innovation_covariance
        q_root = noise._spatial_root
        q_inv = solve_triangular(q_root.T, solve_triangular(q_root, np.eye(len(q)), lower=True))
        denominator = 1-phi[:, None]*phi[None, :]
        initial = q/denominator
        initial_root = np.linalg.cholesky(initial)
        s_inv = solve_triangular(initial_root.T, solve_triangular(initial_root, np.eye(len(q)), lower=True))
        q_gradient, phi_gradient = np.zeros_like(q), np.zeros(len(q))
        for value in statistics:
            a = np.diag(phi)
            tail = value.current-a@value.lag.T-value.lag@a+a@value.previous@a
            q_gradient += .5*(q_inv@tail@q_inv-(value.native_samples-1)*q_inv)
            phi_gradient += np.diag(q_inv@(value.lag-a@value.previous))
            initial_gradient = .5*(s_inv@value.initial@s_inv-s_inv)
            q_gradient += initial_gradient/denominator
            phi_gradient += 2*np.sum(initial_gradient*q*phi[None, :]/denominator**2, axis=1)
        relative_gradient = 2*self.reference_innovation_root.T@q_gradient@self.reference_innovation_root@relative
        packed = [relative_gradient[i, j]*relative[i, j] if i == j else relative_gradient[i, j]
            for i, j in zip(*np.tril_indices(len(q)), strict=True)]
        return np.r_[phi_gradient*(1-phi*phi), packed]


    def score_innovations(self, statistics, parameters, point):
        """Fisher identity using direct innovation factors, including startup."""
        noise, phi, relative = point
        q = noise.innovation_covariance
        q_root = noise._spatial_root
        q_inv = solve_triangular(q_root.T, solve_triangular(q_root, np.eye(len(q)), lower=True))
        denominator = 1-phi[:, None]*phi[None, :]
        initial = q/denominator
        initial_root = np.linalg.cholesky(initial)
        s_inv = solve_triangular(initial_root.T, solve_triangular(initial_root, np.eye(len(q)), lower=True))
        q_gradient, phi_gradient = np.zeros_like(q), np.zeros(len(q))
        for value in statistics:
            if type(value) is not AR1InnovationStatistics:
                raise TypeError('direct score requires explicit innovation statistics, not raw residual moments')
            q_gradient += .5*(q_inv@value.innovation_second@q_inv-(value.native_samples-1)*q_inv)
            phi_gradient += np.diag(q_inv@value.innovation_previous)
            initial_gradient = .5*(s_inv@value.initial@s_inv-s_inv)
            q_gradient += initial_gradient/denominator
            phi_gradient += 2*np.sum(initial_gradient*q*phi[None, :]/denominator**2, axis=1)
        relative_gradient = 2*self.reference_innovation_root.T@q_gradient@self.reference_innovation_root@relative
        packed = [relative_gradient[i, j]*relative[i, j] if i == j else relative_gradient[i, j]
            for i, j in zip(*np.tril_indices(len(q)), strict=True)]
        return np.r_[phi_gradient*(1-phi*phi), packed]


@dataclass(frozen=True)
class JointCalibrationWorkspace:
    potentials: tuple
    record_samples: dict
    posterior_moment_callback: object
    input_identity: str
    innovation_moment_callback: object = None


@dataclass(frozen=True)
class JointCalibrationProblem:
    source: SharedResponseChart
    within_reference: TwoExposureWithinReference
    noise_coordinates: PrivateAR1Coordinates
    component_factory: object
    component_factory_sha256: str
    input_binding: dict
    arm: str

    def __post_init__(self):
        if (type(self.source) is not SharedResponseChart or type(self.within_reference) is not TwoExposureWithinReference
                or type(self.noise_coordinates) is not PrivateAR1Coordinates or self.arm not in {'G0', 'G1'}
                or self.within_reference.factor.shape[0] != self.source.mean.size
                or tuple(self.source.channel_ids) != self.noise_coordinates.geometry.channels
                or plain_metadata(self.within_reference.binding['noise_geometry']) != self.noise_coordinates.geometry.record()
                or self.source.mean.shape[0] != 2*self.within_reference.binding['basis_shape'][1]
                or len(self.component_factory_sha256) != 64):
            raise ValueError('joint C programme explicitly binds complete source/reference/noise/factory and arm')
        object.__setattr__(self, 'input_binding', freeze_metadata(self.input_binding))

    @cached_property
    def identity(self):
        return metadata_digest({'source': self.source.identity, 'reference': self.within_reference.identity,
            'noise_coordinates': self.noise_coordinates.identity, 'factory': self.component_factory_sha256,
            'input': self.input_binding, 'arm': self.arm,
            'G1_log_variance_policy': {'mean': 0., 'sd': 1., 'measure': 'log_variance'},
            'producer': IMPLEMENTATION_SHA256})

    def evaluate(self, parameters, *, gradient=True):
        parameters = np.asarray(parameters, dtype=float)
        noise_width = self.noise_coordinates.dimension
        if parameters.shape != (noise_width+(self.arm == 'G1'),):
            raise ValueError('G0 and G1 retain their own exact declared parameter dimensions')
        try:
            point = self.noise_coordinates.point(parameters[:noise_width])
            x = float(parameters[-1]) if self.arm == 'G1' else None
            within = (self.within_reference.scaled_factor(x) if x is not None
                else np.empty((self.source.mean.size, 0)))
        except (FloatingPointError, NumericalResolutionError) as error:
            raise UnrepresentableJointParameterPointError(str(error)) from error
        workspace = self.component_factory(point[0], within)
        if any(value.binding['original_noise_carrier'] != point[0]._runtime_state_digest()
                for value in workspace.potentials):
            raise ValueError('every original C component must consume the current declared noise point')
        marginal = joint_calibration_marginal(workspace.potentials, source_chart=self.source,
            record_samples=workspace.record_samples, input_binding={**plain_metadata(self.input_binding),
                'workspace_identity': workspace.input_identity, 'problem': self.identity})
        log_prior = -.5*x*x-.5*_LOG2PI if x is not None else 0.
        objective = -marginal.log_density-log_prior
        if not np.isfinite(objective):
            raise UnrepresentableJointParameterPointError('joint parameter point has no finite objective')
        if not gradient:
            return float(objective), marginal
        direct_innovations = workspace.innovation_moment_callback is not None
        callback = workspace.innovation_moment_callback if direct_innovations else workspace.posterior_moment_callback
        statistics, local_second, local_dimension = callback(marginal)
        actual_records = {value.record_id: value.native_samples for value in statistics}
        if len(actual_records) != len(statistics) or actual_records != workspace.record_samples:
            raise ValueError('AR moments retain one startup and every native row of each original record')
        expected_dimension = sum(width for value in workspace.potentials for _, width in value.local_dimensions)
        if local_dimension != expected_dimension or (self.arm == 'G0' and local_dimension):
            raise ValueError('EM moments retain every independent local column exactly once')
        try:
            score_function = self.noise_coordinates.score_innovations if direct_innovations else self.noise_coordinates.score
            score = score_function(statistics, parameters[:noise_width], point)
        except FloatingPointError as error:
            raise UnrepresentableJointParameterPointError('joint parameter point has no finite complete noise score') from error
        if x is not None:
            score = np.r_[score, .5*(local_second-local_dimension)-x]
        if not np.isfinite(score).all():
            raise UnrepresentableJointParameterPointError('joint parameter point has no finite complete gradient')
        return float(objective), frozen_array(-score), marginal


class JointCalibrationOptimizationError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__('the declared joint response/noise programme did not produce a converged plugin')


def fit_joint_calibration(problem, initial, *, initial_binding, maximum_iterations, gradient_tolerance,
        evaluation_callback=None):
    """Reusable numeric optimizer; original C/live fit admission is upstream."""
    initial = finite_real_array(initial, name='explicit joint warm start or external initial point')
    if (type(maximum_iterations) is not int or maximum_iterations < 1
            or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0
            or initial.shape != (problem.noise_coordinates.dimension+(problem.arm == 'G1'),)):
        raise ValueError('declare the actual optimizer programme')
    calls, invalid_trials, last_valid, best_valid, last_invalid = 0, 0, None, None, None
    programme = {'schema': 'joint_response_noise_plugin_fit/1', 'problem_identity': problem.identity,
        'arm': problem.arm, 'source_identity': problem.source.identity,
        'within_reference_identity': problem.within_reference.identity,
        'noise_parameterization': problem.noise_coordinates.identity,
        'within_policy': 'N(0,1)_density_in_dimensionless_log_variance' if problem.arm == 'G1' else 'fixed_zero_G0_no_local_state',
        'optimizer': 'BFGS_with_complete_posterior_EM_score', 'initial': initial.tolist(),
        'initial_binding': plain_metadata(initial_binding), 'maximum_iterations': maximum_iterations,
        'gradient_tolerance': gradient_tolerance, 'producer_sha256': IMPLEMENTATION_SHA256,
        'old_noise_artifact_resigned': False, 'raw_or_live_fit_authority_granted': False}
    def objective(parameters):
        nonlocal calls, invalid_trials, last_valid, best_valid, last_invalid
        calls += 1
        try:
            if not np.isfinite(parameters).all():
                raise UnrepresentableJointParameterPointError('optimizer trial parameters are not finite')
            value, gradient, _ = problem.evaluate(parameters)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise UnrepresentableJointParameterPointError('optimizer trial objective or complete gradient is not finite')
        except UnrepresentableJointParameterPointError as error:
            # Finite-step line searches may probe beyond the binary64 chart.
            # The scientific law is unchanged; these trials have no fit state.
            invalid_trials += 1
            last_invalid = {'parameters': [float(value) if np.isfinite(value) else repr(float(value)) for value in parameters],
                'reason': str(error), 'evaluation': calls, 'phase': 'parameter_point_or_finite_objective_score'}
            return np.inf, np.zeros_like(parameters)
        last_valid = {'parameters': parameters.tolist(), 'negative_log_objective': value,
            'gradient': gradient.tolist(), 'evaluation': calls}
        if best_valid is None or value < best_valid['negative_log_objective']:
            best_valid = last_valid
        if evaluation_callback is not None:
            evaluation_callback(freeze_metadata({'problem_identity': problem.identity,
                'arm': problem.arm, 'valid_trial': last_valid, 'unrepresentable_trials': invalid_trials,
                'best_valid_trial': best_valid, 'last_unrepresentable_trial': last_invalid,
                'numeric_checkpoint_only_not_converged_fit': True}))
        return value, gradient
    try:
        result = minimize(objective, initial, jac=True, method='BFGS',
            options={'maxiter': maximum_iterations, 'gtol': gradient_tolerance})
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        raise JointCalibrationOptimizationError({**programme, 'success': False,
            'evaluations': calls, 'last_valid_trial': last_valid, 'unrepresentable_trials': invalid_trials,
            'best_valid_trial': best_valid, 'last_unrepresentable_trial': last_invalid,
            'failure': {'type': type(error).__name__, 'message': str(error)}}) from error
    record = {**programme,
        'parameters': result.x.tolist(), 'negative_log_objective': float(result.fun),
        'success': bool(result.success), 'status': int(result.status), 'message': str(result.message),
        'iterations': int(result.nit), 'evaluations': int(result.nfev), 'final_gradient': result.jac.tolist(),
        'unrepresentable_trials': invalid_trials}
    if not result.success or not np.isfinite(result.fun):
        record['last_valid_trial'] = last_valid
        record['best_valid_trial'] = best_valid
        record['last_unrepresentable_trial'] = last_invalid
        raise JointCalibrationOptimizationError(record)
    return frozen_array(result.x), freeze_metadata(record)
