"""Original public C authority -> new joint-response/noise fit and frozen state.

The joint fit has its own issuer. Embedded temporal/noise objects are frozen
numeric consumers, never reassigned legacy Burg target-calibration producers.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from research.channel_noise_mechanism import FittedChannelNoise
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.gaussian_local_conditional import LocalGaussianConditional
from research.gaussian_local_factor import GaussianLocalPotential, SharedResponseChart
from research.joint_response_noise_calibration import (
    JointCalibrationProblem,
    JointMarginal,
    PrivateAR1Coordinates,
    fit_joint_calibration,
    known_c_bit_design_proof,
)
from research.joint_response_noise_reference import (
    TwoExposureWithinReference,
    two_exposure_gls_reference,
)
from research.native_joint_sufficient import (
    NativeComponentSpec,
    NativeJointRecordCompiler,
    native_joint_workspace,
)
from research.private_ar1_noise_point import PrivateAR1NoisePoint
from research.public_record_physical_bridge import require_public_physical_input
from research.public_source_context import (
    public_source_context,
    require_public_source_context,
)
from transfer.ensemble_views import calibration_prefix_identity
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_REFERENCES, _INPUTS, _FITS, _LOADED = (WeakKeyDictionary() for _ in range(4))


def _immutable(values):
    values = np.asarray(values, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True, eq=False)
class PublicJointReference:
    source_admission: object
    source: SharedResponseChart
    basis: np.ndarray
    dictionary: object
    geometry: object
    within_reference: TwoExposureWithinReference
    noise_coordinates: PrivateAR1Coordinates
    source_noise: FittedChannelNoise
    record: dict
    source_context: object


def prepare_public_joint_reference(source_admission, *, strategy, noise_scale, conditioning_samples=32):
    context = public_source_context(source_admission)
    admission, source = context.admission, context.source_chart
    geometry, basis, noise = context.geometry, _immutable(context.basis), context.noise_reference
    binding = {'source_context': context.identity, 'source_original_admission': metadata_digest(admission.record),
        'source_role': admission.record['source_role'], 'noise_parameter_origin': noise._runtime_state_digest(),
        'target_observations_used': False}
    if context.within_reference is None:
        reference = two_exposure_gls_reference(basis, noise, noise_scale=noise_scale,
            dictionary_identity=context.dictionary.digest(), source_reference_binding=binding,
            strategy=strategy, block_native_rows=512)
    else:
        if strategy != 'frozen_source_context' or noise_scale != 1.:
            raise ValueError('population Kref is the original fixed source-context artifact, with scale already declared')
        reference = context.within_reference
    coordinates = PrivateAR1Coordinates(geometry, noise._spatial_root*np.sqrt(noise_scale),
        conditioning_samples, {'fixed_source_only_reference': reference.identity, **binding},
        numeric_point_programme='small_stationary_AR1_innovations')
    record = freeze_metadata({'schema': 'live_public_joint_source_only_reference/1',
        'source_chart': source.identity, 'within_reference': reference.identity,
        'noise_coordinates': coordinates.identity, 'basis_sha256': array_digest(basis),
        'dictionary': context.dictionary.digest(), 'source_binding': binding,
        'reference_strategy': strategy, 'source_noise_scale': noise_scale,
        'source_fits': 0, 'target_parameter_fits': 0, 'producer': IMPLEMENTATION_SHA256})
    value = PublicJointReference(admission, source, basis, context.dictionary, geometry,
        reference, coordinates, noise, record, context)
    _REFERENCES[value] = _reference_snapshot(value)
    return value


def _reference_snapshot(value):
    return metadata_digest({'record': value.record, 'context': value.source_context.identity,
        'source': value.source.identity, 'basis': array_digest(value.basis),
        'dictionary': value.dictionary.digest(), 'within': value.within_reference.identity,
        'noise_reference_root': array_digest(value.noise_coordinates.reference_innovation_root),
        'noise_coordinates': value.noise_coordinates.identity, 'geometry': value.geometry.record(),
        'source_noise': value.source_noise._runtime_state_digest()})


def require_public_joint_reference(value):
    if type(value) is not PublicJointReference or value not in _REFERENCES:
        raise TypeError('joint reference requires its original source-only public admission')
    context = require_public_source_context(value.source_context)
    if (value.source_admission is not context.admission or value.source is not context.source_chart
            or value.dictionary is not context.dictionary or value.geometry != context.geometry
            or value.source_noise is not context.noise_reference or _reference_snapshot(value) != _REFERENCES[value]):
        raise ValueError('actual joint source/reference parameters changed after admission')
    return value


@dataclass(frozen=True)
class PublicCTemporalSelection:
    """On-grid empirical native basis with every actual authorized C bit."""

    basis: np.ndarray
    events: tuple
    selection_id: str
    native_samples: int
    binding: dict

    def __call__(self, start, stop):
        if not 0 <= start < stop <= self.native_samples:
            raise ValueError('C temporal query retains its original granted native interval')
        k = self.basis.shape[1]
        design = np.zeros((stop-start, 2*k))
        absolute = np.zeros_like(design)
        for _, onset, bit in self.events:
            left, right = max(start, onset), min(stop, onset+len(self.basis))
            if left < right:
                values = self.basis[left-onset:right-onset]
                design[left-start:right-start, :k] += values
                design[left-start:right-start, k:] += int(bit)*values
                absolute[left-start:right-start, :k] += np.abs(values)
                absolute[left-start:right-start, k:] += int(bit)*np.abs(values)
        gamma = len(self.events)*np.finfo(float).eps/(1-len(self.events)*np.finfo(float).eps)
        return design, gamma*absolute, self.binding

    @property
    def support(self):
        return (min(row[1] for row in self.events),
            min(self.native_samples, max(row[1] for row in self.events)+len(self.basis)))


@dataclass(frozen=True, eq=False)
class PublicJointCalibrationInputs:
    reference: PublicJointReference
    prefix: object
    calibration_inputs: tuple
    compilers: tuple
    record: dict


_C_SELECTION_CALL = PublicCTemporalSelection.__call__
_NATIVE_METHODS = {name: value for name, value in vars(NativeJointRecordCompiler).items()
    if callable(value) and name != '__init__'}


def _compiler_snapshot(compiler):
    if (type(compiler) is not NativeJointRecordCompiler or any(
            getattr(getattr(compiler, name), '__func__', None) is not method
            for name, method in _NATIVE_METHODS.items())
            or any(type(value) is not PublicCTemporalSelection or type(value).__call__ is not _C_SELECTION_CALL
                for value in compiler.designs.values())):
        raise TypeError('the public joint C issuer retains its actual native compiler and empirical design callbacks')
    return {'record': compiler.record_id, 'samples': array_digest(compiler.samples),
        'observation_identity': compiler.observation_identity, 'compiler_identity': compiler.identity,
        'drift_group_id': compiler.drift_group_id,
        'p': compiler.p, 'c': compiler.c, 'n': compiler.n, 'conditioning_samples': compiler.conditioning_samples,
        'block_native_rows': compiler.block_rows, 'branch': compiler.branch_binding, 'supports': compiler.supports,
        'intervals': compiler.intervals, 'compiled_component_indices': compiler.compiled_component_indices,
        'geometry': compiler.geometry.record(), 'drift': compiler.drift_ids,
        'specs': [{'selection_ids': spec.selection_ids, 'native_intervals': spec.native_intervals,
            'tail_root': array_digest(tail), 'moment_root': array_digest(moment)}
            for spec, tail, moment in zip(compiler.specs, compiler.tail_roots, compiler.moment_roots, strict=True)],
        'actual_C_designs': {name: {'events': value.events, 'basis': array_digest(value.basis),
            'selection_id': value.selection_id, 'native_samples': value.native_samples, 'binding': value.binding}
            for name, value in compiler.designs.items()}}


def prepare_public_joint_calibration(reference, prefix, calibration_inputs, *, block_native_rows=2048):
    reference = require_public_joint_reference(reference)
    admission = reference.source_admission
    prefix = admission.compiler.assert_original_prefix(prefix)
    identity = calibration_prefix_identity(prefix)
    inputs = tuple(require_public_physical_input(value) for value in calibration_inputs)
    expected = tuple(value for value in admission.calibration_inputs if value.public_binding['C_prefix_identity'] == identity)
    if (not any(prefix is p for p in admission.prefixes) or len(inputs) != len(expected) or not inputs
            or any(a is not b for a, b in zip(inputs, expected, strict=True))):
        raise ValueError('joint calibration consumes its complete actual same-prefix C grants once')
    compilers, proofs = [], []
    for cap in inputs:
        b = cap.public_binding
        if (b['role'] != 'calibration' or b['sample_start'] != 0
                or b['sample_rate_hz'] != reference.dictionary.sample_rate_hz
                or tuple(b['eeg_channels']) != reference.geometry.channels):
            raise ValueError('joint calibration requires original native-grid C samples and geometry')
        by_event = {name: (onset, bit) for name, onset, bit in cap.label_rows}
        exposures = b['public_physical_history']['exposures']
        if set(by_event) != {event['event_id'] for event in exposures}:
            raise ValueError('every active C exposure requires its exact original authorized binary bit')
        designs, supports = {}, {}
        for row in b['public_physical_history']['selections']:
            name = row['selection_id']
            events = [event for event in exposures if event['block_id'] == name]
            if any(event['membership'] is None for event in events):
                raise ValueError('unknown C physical membership requires its explicit joint identity law')
            labels = []
            for event in events:
                onset, bit = by_event[event['event_id']]
                if onset != event['native_sample'] or bit not in (0, 1):
                    raise ValueError('C bit and original physical exposure native clock differ')
                labels.append((event['event_id'], onset, int(bit)))
            vocabulary = tuple(row['candidate_vocabulary'])
            proof = known_c_bit_design_proof(tuple(event['event_id'] for event in events),
                np.array([event['membership'] for event in events], dtype=int), np.array([bit for _, _, bit in labels]),
                vocabulary, np.repeat(1/len(vocabulary), len(vocabulary)))
            binding = freeze_metadata({'original_C_capability': cap.identity, 'original_selection': name,
                'original_binary_rows': labels, 'known_bit_equivalence': proof,
                'native_grid_basis': array_digest(reference.basis), 'dictionary': reference.dictionary.digest(),
                'native_clock': 'native_sample_index', 'nonlinear_warp': 'not_part_of_G0_G1_functional_law',
                'common_exposure_count': len(events)})
            design = PublicCTemporalSelection(reference.basis, tuple(labels), name, len(cap.samples), binding)
            designs[name], supports[name] = design, design.support
            proofs.append(binding)
        compilers.append(NativeJointRecordCompiler(cap.samples, 2*reference.basis.shape[1], designs, supports,
            reference.geometry, record_id=b['record_id'], drift_group_id=b['record_id'],
            branch_binding={'original_C_capability': cap.identity, 'original_source_sha256': b['source_sha256'],
                'C_prefix_identity': identity, 'full_candidate_axis_retained': True},
            block_native_rows=block_native_rows, conditioning_samples=reference.noise_coordinates.conditioning_samples))
    record = freeze_metadata({'schema': 'original_public_joint_C_inputs/1',
        'source_reference': metadata_digest(reference.record), 'source_chart': reference.source.identity,
        'C_prefix_identity': identity, 'actual_K': prefix.budget, 'actual_granted_bits': sum(len(cap.label_rows) for cap in inputs),
        'original_C_inputs': [cap.identity for cap in inputs], 'original_label_use': prefix.label_use,
        'known_bit_designs': proofs, 'numeric_compilers': [_compiler_snapshot(value) for value in compilers],
        'shared_source_prior_count': 1, 'C_likelihood_count': 1, 'producer': IMPLEMENTATION_SHA256})
    value = PublicJointCalibrationInputs(reference, prefix, inputs, tuple(compilers), record)
    _INPUTS[value] = (reference, prefix, inputs, tuple(compilers), metadata_digest(record))
    return value


def require_public_joint_calibration(value):
    if type(value) is not PublicJointCalibrationInputs or value not in _INPUTS:
        raise TypeError('joint fitting requires its actual original C/source/reference factory')
    reference, prefix, caps, compilers, record = _INPUTS[value]
    require_public_joint_reference(reference)
    if (value.reference is not reference or value.prefix is not prefix or value.calibration_inputs is not caps
            or len(value.compilers) != len(compilers) or any(a is not b for a, b in zip(value.compilers, compilers, strict=True))
            or metadata_digest(value.record) != record):
        raise ValueError('joint calibration changed its actual source/prefix/C/compiler relation')
    for cap in value.calibration_inputs:
        require_public_physical_input(cap)
    if metadata_digest([_compiler_snapshot(compiler) for compiler in value.compilers]) != metadata_digest(value.record['numeric_compilers']):
        raise ValueError('actual C sufficient statistics or response inputs changed after admission')
    return value


@dataclass(frozen=True)
class FrozenJointCComponent:
    record_id: str
    spec: NativeComponentSpec
    drift_ids: tuple
    potential: GaussianLocalPotential
    conditional: LocalGaussianConditional


@dataclass(frozen=True, eq=False)
class PublicJointFit:
    source_admission: object
    prefix: object
    calibration_inputs: tuple
    source: SharedResponseChart
    basis: np.ndarray
    dictionary: object
    within_reference: TwoExposureWithinReference
    noise: PrivateAR1NoisePoint
    parameters: np.ndarray
    arm: str
    posterior: JointMarginal
    components: tuple
    record: dict
    artifact_sha256: str | None = None

    @property
    def within_factor(self):
        return (self.within_reference.scaled_factor(float(self.parameters[-1])) if self.arm == 'G1'
            else np.empty((self.source.mean.size, 0)))

    @property
    def identity(self):
        return metadata_digest({'record': self.record, 'source': self.source.identity,
            'basis': array_digest(self.basis), 'dictionary': self.dictionary.digest(),
            'reference': self.within_reference.identity, 'noise': self.noise._runtime_state_digest(),
            'parameters': array_digest(self.parameters), 'arm': self.arm,
            'posterior': array_digest(self.posterior.upper, self.posterior.rhs, self.posterior.mean, self.posterior.factor),
            'posterior_log_density': self.posterior.log_density,
            'posterior_coordinates': [self.posterior.shared_dimension, self.posterior.drift_coordinate_ids],
            'posterior_binding': self.posterior.binding,
            'components': [{'potential': value.potential.identity, 'conditional': value.conditional.identity,
                'conditional_independent_factor': array_digest(value.conditional.independent_factor),
                'record': value.record_id, 'spec': vars(value.spec), 'drift_ids': value.drift_ids} for value in self.components]})

    def save(self, path):
        return save_public_joint_fit(self, path)

    @classmethod
    def load(cls, path, *, expected_sha256, reference):
        return load_public_joint_fit(path, expected_sha256=expected_sha256, reference=reference)


def _noise_numbers(noise):
    if type(noise) is PrivateAR1NoisePoint:
        return array_digest(noise.phi, noise.innovation_covariance, noise.initial_root, noise.innovation_root)
    return array_digest(noise.innovation_covariance, noise.compiled_prefix_root,
        np.array([value.coefficients for value in noise.temporal]), np.array([value.reflection_coefficients for value in noise.temporal]))


def _freeze_posterior(posterior):
    # These cached properties are writable arrays in the reusable numeric core.
    # The actual public issuer owns immutable query arrays and snapshots them.
    object.__setattr__(posterior, 'mean', _immutable(posterior.mean))
    object.__setattr__(posterior, 'factor', _immutable(posterior.factor))
    return posterior


def _validate_fit_numbers(fit, reference):
    parameters, noise_width = np.asarray(fit.parameters), reference.noise_coordinates.dimension
    if (fit.arm not in ('G0', 'G1') or parameters.shape != (noise_width+(fit.arm == 'G1'),)
            or not np.isfinite(parameters).all() or fit.arm != fit.record['arm']
            or not np.array_equal(parameters, fit.record['optimizer']['parameters'])
            or fit.record['optimizer']['success'] is not True
            or fit.record['optimizer']['noise_parameterization'] != reference.noise_coordinates.identity
            or fit.source is not reference.source or fit.dictionary is not reference.dictionary
            or array_digest(fit.basis) != array_digest(reference.basis)
            or fit.within_reference is not reference.within_reference
            or metadata_digest(fit.record['source_reference']) != metadata_digest(reference.record)):
        raise ValueError('joint artifact retains the complete source/reference/optimizer/arm parameter relation')
    point = reference.noise_coordinates.point(parameters[:noise_width])[0]
    relation = fit.record['noise_point_to_frozen_consumer']
    if (fit.noise.geometry != reference.geometry or _noise_numbers(fit.noise) != _noise_numbers(point)
            or relation['evaluated_point'] != point._runtime_state_digest()
            or relation['frozen_consumer'] != fit.noise._runtime_state_digest()
            or relation['same_complete_numeric_arrays'] != _noise_numbers(point)
            or relation['joint_fit_receipt'] != metadata_digest(fit.record['optimizer'])
            or fit.noise.provenance['actual_joint_fit_scope'] != 'target_calibration'):
        raise ValueError('joint parameters and the frozen numeric noise consumer differ')
    local_width = fit.within_factor.shape[1]
    caps = {row['record_id']: row for row in fit.record['C_inputs']}
    expected_drift = tuple(dict.fromkeys(name for value in fit.components for name in value.drift_ids))
    if (fit.posterior.shared_dimension != fit.source.rank or fit.posterior.drift_coordinate_ids != expected_drift
            or tuple(fit.posterior.binding['component_identities']) != tuple(value.potential.identity for value in fit.components)
            or {value.record_id for value in fit.components} != set(caps)):
        raise ValueError('the saved C posterior retains all shared, original-record drift and local components')
    for value in fit.components:
        potential, conditional = value.potential, value.conditional
        if (potential.shared_dimension != fit.source.rank or potential.drift_dimension != len(value.drift_ids)
                or conditional.shared_dimension != potential.shared_dimension
                or conditional.drift_dimension != potential.drift_dimension
                or conditional.local_dimensions != potential.local_dimensions
                or potential.local_dimensions != tuple((name, local_width) for name in value.spec.selection_ids)
                or potential.binding['shared_response_chart'] != fit.source.identity
                or potential.binding['original_noise_carrier'] != fit.noise._runtime_state_digest()
                or conditional.binding['marginal_potential'] != potential.identity
                or tuple(potential.binding['drift_coordinate_ids']) != value.drift_ids):
            raise ValueError('each saved C component keeps its complete shared/within/noise/local conditional relation')
    fit.noise.assert_integrity()


def fit_public_joint_calibration(inputs, arm, *, maximum_iterations, gradient_tolerance, initial=None, initial_binding=None,
        progress_callback=None, evaluation_callback=None):
    inputs = require_public_joint_calibration(inputs)
    ref = inputs.reference
    last = {'evaluations': 0}

    def factory(noise, within):
        began = time.perf_counter()
        last['evaluations'] += 1
        workspace, reductions = native_joint_workspace(inputs.compilers, noise, within, ref.source)
        original = workspace.innovation_moment_callback
        def capture(marginal):
            moments = original(marginal)
            last['posterior'], last['innovation_moments'] = marginal, moments
            if progress_callback is not None:
                progress_callback({'arm': arm, 'evaluation': last['evaluations'],
                    'log_C_density': marginal.log_density, 'evaluation_seconds': time.perf_counter()-began,
                    'complete_local_dimensions': moments[2]})
            return moments
        last['noise'], last['within'], last['reductions'], last['workspace'] = noise, within, reductions, workspace
        return replace(workspace, innovation_moment_callback=capture)

    problem = JointCalibrationProblem(ref.source, ref.within_reference, ref.noise_coordinates, factory,
        IMPLEMENTATION_SHA256, inputs.record, arm)
    if initial is None:
        if initial_binding is not None:
            raise ValueError('an explicit initial binding accompanies an explicit initial point')
        initial = ref.noise_coordinates.warm_start(ref.source_noise)
        if arm == 'G1':
            initial = np.r_[initial, 0.]
        initial_binding = {'kind': 'source_only_noise_numeric_warm_start',
            'source_noise': ref.source_noise._runtime_state_digest(), 'reference': metadata_digest(ref.record)}
    elif initial_binding is None:
        raise ValueError('a supplied numerical initial point declares its actual origin')
    parameters, optimization = fit_joint_calibration(problem, initial,
        initial_binding=initial_binding,
        maximum_iterations=maximum_iterations, gradient_tolerance=gradient_tolerance,
        evaluation_callback=evaluation_callback)
    # Reevaluate the accepted point to bind the exact final posterior/moments.
    problem.evaluate(parameters)
    actual_noise = last['noise']
    rows = [{'record_id': cap.public_binding['record_id'], 'recording_id': cap.public_binding['record_id'],
        'participant_key': cap.public_binding['participant_key'], 'read_support': [0, len(cap.samples)],
        'original_C_input': cap.identity, 'samples_sha256': array_digest(cap.samples),
        'original_binary_rows': cap.label_rows} for cap in inputs.calibration_inputs]
    provenance = {'fit_scope': 'frozen_production', 'actual_joint_fit_scope': 'target_calibration',
        'source_model_sha256': metadata_digest(optimization), 'records': rows,
        'role': 'new_joint_response_noise_frozen_numeric_consumer_not_Burg_fit',
        'joint_programme': problem.identity, 'joint_optimizer_receipt': metadata_digest(optimization),
        'evaluated_noise_point': actual_noise._runtime_state_digest()}
    frozen_noise = actual_noise.freeze(provenance)
    if _noise_numbers(frozen_noise) != _noise_numbers(actual_noise):
        raise ValueError('joint frozen noise consumer changed the fitted parameters or saved startup root')
    relation = {'schema': 'new_joint_fit_point_to_frozen_numeric_noise_consumer/1',
        'actual_joint_fit_scope': 'target_calibration', 'joint_fit_receipt': metadata_digest(optimization),
        'evaluated_point': actual_noise._runtime_state_digest(), 'frozen_consumer': frozen_noise._runtime_state_digest(),
        'same_complete_numeric_arrays': _noise_numbers(frozen_noise), 'old_Burg_or_fixed_mean_producer_rewritten': False}
    components = []
    for value in last['reductions']:
        potential = replace(value.potential, binding={**plain_metadata(value.potential.binding),
            'original_noise_carrier': frozen_noise._runtime_state_digest(),
            'evaluated_joint_potential': value.potential.identity, 'joint_frozen_noise_relation': relation})
        conditional = replace(value.conditional, binding={**plain_metadata(value.conditional.binding),
            'marginal_potential': potential.identity, 'evaluated_local_conditional': value.conditional.identity,
            'joint_frozen_noise_relation': relation})
        components.append(FrozenJointCComponent(value.compiler.record_id, value.spec, value.drift_ids, potential, conditional))
    posterior = _freeze_posterior(replace(last['posterior'], binding={**plain_metadata(last['posterior'].binding),
        'evaluated_joint_marginal': metadata_digest(last['posterior'].binding),
        'component_identities': [value.potential.identity for value in components], 'joint_frozen_noise_relation': relation}))
    # Raw moments remain diagnostics; the hot fit and replay use direct
    # innovation moments so large flat-intercept covariance is never subtracted.
    statistics, local_second, local_dimension = last['workspace'].posterior_moment_callback(last['posterior'])
    innovation_statistics = last['innovation_moments'][0]
    record = freeze_metadata({'schema': 'actual_public_joint_response_noise_fit/1', 'fit_scope': 'target_calibration',
        'input_programme': inputs.record, 'source_reference': ref.record, 'source_chart': ref.source.identity,
        'within_reference': ref.within_reference.identity, 'C_prefix_identity': calibration_prefix_identity(inputs.prefix),
        'C_inputs': rows, 'source_admission': metadata_digest(ref.source_admission.record),
        'arm': arm, 'optimizer': optimization, 'problem_identity': problem.identity,
        'noise_point_to_frozen_consumer': relation, 'within_local_second': local_second,
        'complete_local_dimensions': local_dimension,
        'complete_record_AR1_statistics': [{'record_id': value.record_id, 'native_samples': value.native_samples,
            'initial': value.initial.tolist(), 'previous': value.previous.tolist(), 'lag': value.lag.tolist(),
            'current': value.current.tolist()} for value in statistics],
        'score_statistics': 'direct_complete_posterior_innovation_factors',
        'complete_record_AR1_innovation_statistics': [{'record_id': value.record_id, 'native_samples': value.native_samples,
            'initial': value.initial.tolist(), 'innovation_second': value.innovation_second.tolist(),
            'innovation_previous': value.innovation_previous.tolist()} for value in innovation_statistics],
        'shared_source_prior_count': 1, 'C_likelihood_count': 1, 'held_data_or_labels_used': False,
        'old_noise_or_source_fit_resigned': False, 'producer': IMPLEMENTATION_SHA256})
    fit = PublicJointFit(ref.source_admission, inputs.prefix, inputs.calibration_inputs, ref.source, ref.basis,
        ref.dictionary, ref.within_reference, frozen_noise, _immutable(parameters), arm, posterior, tuple(components), record)
    _validate_fit_numbers(fit, ref)
    _FITS[fit] = (ref, fit.source_admission, fit.prefix, fit.calibration_inputs, fit.identity)
    return fit


def require_public_joint_fit(fit):
    registered = _FITS.get(fit) if type(fit) is PublicJointFit else None
    if registered is None:
        raise TypeError('joint held consumption requires the new actual public joint-fit issuer')
    reference, admission, prefix, caps, identity = registered
    require_public_joint_reference(reference)
    admission.compiler.assert_original_prefix(prefix)
    if fit.source_admission is not admission or fit.prefix is not prefix or fit.calibration_inputs is not caps or fit.identity != identity:
        raise ValueError('actual joint fitted state or its original C/source input relation changed')
    for cap in caps:
        require_public_physical_input(cap)
    fit.noise.assert_integrity()
    return fit


def _replay_original_c(fit, inputs):
    inputs = require_public_joint_calibration(inputs)
    if metadata_digest(inputs.record) != metadata_digest(fit.record['input_programme']):
        raise ValueError('saved joint fit must consume its actual unchanged complete original C programme')
    reference = inputs.reference
    point = reference.noise_coordinates.point(fit.parameters[:reference.noise_coordinates.dimension])[0]
    workspace, reductions = native_joint_workspace(inputs.compilers, point, fit.within_factor, fit.source)
    if len(reductions) != len(fit.components):
        raise ValueError('saved joint fit changed the complete original C support components')
    for actual, saved in zip(reductions, fit.components, strict=True):
        if (actual.compiler.record_id != saved.record_id or actual.spec != saved.spec
                or actual.drift_ids != saved.drift_ids
                or actual.potential.local_dimensions != saved.potential.local_dimensions
                or actual.potential.binding['original_observation_sha256'] != saved.potential.binding['original_observation_sha256']
                or not np.isclose(actual.potential.log_normalizer, saved.potential.log_normalizer, rtol=2e-12, atol=2e-12)
                or any(not np.allclose(a, b, rtol=2e-11, atol=2e-11) for a, b in (
                    (actual.potential.residual_root, saved.potential.residual_root),
                    (actual.conditional.upper, saved.conditional.upper),
                    (actual.conditional.cross, saved.conditional.cross), (actual.conditional.rhs, saved.conditional.rhs)))):
            raise ValueError('saved joint noise/within/local/shared state differs from the actual original C likelihood')
    problem = JointCalibrationProblem(fit.source, fit.within_reference, reference.noise_coordinates,
        lambda noise, within: workspace, IMPLEMENTATION_SHA256, inputs.record, fit.arm)
    objective, gradient, posterior = problem.evaluate(fit.parameters)
    optimizer = fit.record['optimizer']
    if (problem.identity != optimizer['problem_identity']
            or not np.isclose(objective, optimizer['negative_log_objective'], rtol=2e-12, atol=2e-12)
            or not np.allclose(gradient, optimizer['final_gradient'], rtol=2e-8, atol=2e-8)
            or float(np.max(np.abs(gradient))) > optimizer['gradient_tolerance']+2e-8):
        raise ValueError('saved optimizer receipt differs from the actual original C frozen-point objective/gradient')
    if (not np.allclose(posterior.upper, fit.posterior.upper, rtol=2e-11, atol=2e-11)
            or not np.allclose(posterior.rhs, fit.posterior.rhs, rtol=2e-11, atol=2e-11)
            or not np.allclose(posterior.mean, fit.posterior.mean, rtol=2e-11, atol=2e-11)
            or not np.allclose(posterior.factor, fit.posterior.factor, rtol=2e-11, atol=2e-11)
            or not np.isclose(posterior.log_density, fit.posterior.log_density, rtol=2e-12, atol=2e-12)):
        raise ValueError('saved joint posterior differs from the actual complete original C fixed-parameter replay')


def admit_public_joint_fit(fit, *, reference, prefix, calibration_inputs, calibration_programme=None):
    reference = require_public_joint_reference(reference)
    admission = reference.source_admission
    prefix = admission.compiler.assert_original_prefix(prefix)
    caps = tuple(require_public_physical_input(value) for value in calibration_inputs)
    token = _LOADED.get(fit) if type(fit) is PublicJointFit else None
    if (token is None or token != (fit.artifact_sha256, fit.identity, reference)
            or fit.source.identity != reference.source.identity or fit.within_reference.identity != reference.within_reference.identity
            or fit.record['C_prefix_identity'] != calibration_prefix_identity(prefix)
            or tuple(row['original_C_input'] for row in fit.record['C_inputs']) != tuple(cap.identity for cap in caps)
            or tuple(id(cap) for cap in admission.calibration_inputs if cap.public_binding['C_prefix_identity'] == calibration_prefix_identity(prefix))
                != tuple(id(cap) for cap in caps)):
        raise ValueError('saved joint-fit state must rejoin its actual same-source/reference/C parent')
    _validate_fit_numbers(fit, reference)
    if calibration_programme is None:
        calibration_programme = next((value for value in _INPUTS if value.reference is reference
            and value.prefix is prefix and len(value.calibration_inputs) == len(caps)
            and all(a is b for a, b in zip(value.calibration_inputs, caps, strict=True))), None)
        if calibration_programme is None:
            calibration_programme = prepare_public_joint_calibration(reference, prefix, caps,
                block_native_rows=fit.record['input_programme']['numeric_compilers'][0]['block_native_rows'])
    if (calibration_programme.reference is not reference or calibration_programme.prefix is not prefix
            or tuple(id(cap) for cap in calibration_programme.calibration_inputs) != tuple(id(cap) for cap in caps)):
        raise ValueError('fixed-parameter replay requires this exact original source/reference/prefix/C programme')
    _replay_original_c(fit, calibration_programme)
    live = replace(fit, source_admission=admission, prefix=prefix, calibration_inputs=caps)
    _FITS[live] = (reference, admission, prefix, caps, live.identity)
    return live


def save_public_joint_fit(fit, path):
    if fit in _FITS:
        require_public_joint_fit(fit)
    elif type(fit) is not PublicJointFit or fit not in _LOADED or _LOADED[fit][:2] != (fit.artifact_sha256, fit.identity):
        raise TypeError('save requires an actual issued or unchanged numeric-loaded joint fit')
    path = Path(path)
    arrays = {'parameters': fit.parameters, 'posterior_upper': fit.posterior.upper,
        'posterior_rhs': fit.posterior.rhs, 'posterior_mean': fit.posterior.mean, 'posterior_factor': fit.posterior.factor}
    components = []
    for i, value in enumerate(fit.components):
        key = f'component_{i}'
        arrays.update({key+'_potential_root': value.potential.residual_root, key+'_upper': value.conditional.upper,
            key+'_cross': value.conditional.cross, key+'_rhs': value.conditional.rhs,
            key+'_independent_factor': value.conditional.independent_factor})
        components.append({'record_id': value.record_id, 'spec': vars(value.spec), 'drift_ids': value.drift_ids,
            'potential': {'log_normalizer': value.potential.log_normalizer, 'shared_dimension': value.potential.shared_dimension,
                'drift_dimension': value.potential.drift_dimension, 'local_dimensions': value.potential.local_dimensions,
                'binding': plain_metadata(value.potential.binding)},
            'conditional_binding': plain_metadata(value.conditional.binding)})
    metadata = {'schema': 'public_joint_response_noise_numeric_fit_artifact/1', 'identity': fit.identity,
        'source_identity': fit.source.identity, 'basis_sha256': array_digest(fit.basis), 'dictionary': fit.dictionary.digest(),
        'reference_identity': fit.within_reference.identity, 'arm': fit.arm, 'record': plain_metadata(fit.record),
        'noise': fit.noise.record(),
        'posterior': {'log_density': fit.posterior.log_density, 'shared_dimension': fit.posterior.shared_dimension,
            'drift_coordinate_ids': fit.posterior.drift_coordinate_ids, 'binding': plain_metadata(fit.posterior.binding)},
        'components': components, 'numeric_only_no_live_raw_source_or_C_grant': True}
    arrays['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_public_joint_fit(path, *, expected_sha256, reference):
    path = Path(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('joint fit codec changed its externally frozen bytes')
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        meta = json.loads(str(arrays['metadata']))
        if (meta['schema'] != 'public_joint_response_noise_numeric_fit_artifact/1'
                or meta['numeric_only_no_live_raw_source_or_C_grant'] is not True
                or meta['source_identity'] != reference.source.identity or meta['reference_identity'] != reference.within_reference.identity
                or meta['basis_sha256'] != array_digest(reference.basis) or meta['dictionary'] != reference.dictionary.digest()):
            raise ValueError('joint fit codec requires its complete external original source/reference dependencies')
        noise = PrivateAR1NoisePoint.from_record(meta['noise'])
        p = meta['posterior']
        posterior = _freeze_posterior(JointMarginal(p['log_density'], arrays['posterior_upper'], arrays['posterior_rhs'],
            p['shared_dimension'], tuple(p['drift_coordinate_ids']), p['binding']))
        if (not np.allclose(arrays['posterior_mean'], posterior.mean, rtol=2e-12, atol=2e-12)
                or not np.allclose(arrays['posterior_factor'], posterior.factor, rtol=2e-12, atol=2e-12)):
            raise ValueError('saved posterior query arrays differ from the complete shared/flat solve')
        object.__setattr__(posterior, 'mean', _immutable(arrays['posterior_mean']))
        object.__setattr__(posterior, 'factor', _immutable(arrays['posterior_factor']))
        components = []
        for i, value in enumerate(meta['components']):
            key, data = f'component_{i}', value['potential']
            potential = GaussianLocalPotential(arrays[key+'_potential_root'], data['log_normalizer'],
                data['shared_dimension'], data['drift_dimension'], tuple(map(tuple, data['local_dimensions'])), data['binding'])
            conditional = LocalGaussianConditional(arrays[key+'_upper'], arrays[key+'_cross'], arrays[key+'_rhs'],
                potential.local_dimensions, potential.shared_dimension, potential.drift_dimension, value['conditional_binding'])
            if not np.allclose(arrays[key+'_independent_factor'], conditional.independent_factor, rtol=2e-12, atol=2e-12):
                raise ValueError('saved local conditional factor differs from its complete same-QR triangular solve')
            object.__setattr__(conditional, 'independent_factor', _immutable(arrays[key+'_independent_factor']))
            components.append(FrozenJointCComponent(value['record_id'], NativeComponentSpec(tuple(value['spec']['selection_ids']),
                tuple(map(tuple, value['spec']['native_intervals']))), tuple(value['drift_ids']), potential, conditional))
        fit = PublicJointFit(None, None, (), reference.source, reference.basis, reference.dictionary,
            reference.within_reference, noise, _immutable(arrays['parameters']), meta['arm'], posterior, tuple(components),
            freeze_metadata(meta['record']), expected_sha256)
    if fit.identity != meta['identity']:
        raise ValueError('joint fit codec changed the complete fitted noise/source/local/shared/flat state')
    _validate_fit_numbers(fit, reference)
    _LOADED[fit] = (expected_sha256, fit.identity, reference)
    return fit
