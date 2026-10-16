"""Actual shared-source arrays -> compact original-native history operands.

This numerical builder consumes samples and public clocks already issued by
the original compiler. It does not grant read access, fit noise, or infer a
recording identity from a selection key.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from research.channel_noise_mechanism import (
    UnidentifiableFitError,
    _inverse_root_geometry,
    _roundoff_gamma,
    _triangular_image_error,
)
from research.joint_flat_factor_history import PulseTargetImage, SharedFactorRecord
from transfer.noise_model import array_digest, freeze_metadata, frozen_array


def _lattice_tail_error(temporal, raw, source_error, prefix):
    """The same original full-record PARCOR history, with cellwise errors."""
    if not temporal.order:
        return raw[prefix:].copy(), source_error[prefix:].copy()
    forward, backward = raw[1:].copy(), raw[:-1].copy()
    ef, eb = source_error[1:].copy(), source_error[:-1].copy()
    for index, reflection in enumerate(temporal.reflection_coefficients, 1):
        nf, nb = forward+reflection*backward, backward+reflection*forward
        ne_f = ef+abs(reflection)*eb+_roundoff_gamma(2)*(
            np.abs(forward)+abs(reflection)*np.abs(backward))
        ne_b = eb+abs(reflection)*ef+_roundoff_gamma(2)*(
            np.abs(backward)+abs(reflection)*np.abs(forward))
        if index < temporal.order:
            forward, backward, ef, eb = nf[1:], nb[:-1], ne_f[1:], ne_b[:-1]
        else:
            return nf[prefix-temporal.order:], ne_f[prefix-temporal.order:]
    raise AssertionError('nonzero original AR order must execute its final stage')


def _original_white_cells(noise, raw, errors):
    """Original saved joint startup and tail spatial map; no map replacement."""
    prefix, channels = noise.conditioning_samples, len(noise.geometry.channels)
    white = noise.whiten(raw)
    inverse, _ = _inverse_root_geometry(noise.compiled_prefix_root, lower=True)
    leading = _triangular_image_error(noise.compiled_prefix_root,
        raw[:prefix].reshape(prefix*channels, -1), white[:prefix].reshape(prefix*channels, -1),
        errors[:prefix].reshape(prefix*channels, -1), inverse, accurate=True).reshape(white[:prefix].shape)
    tail, tail_errors = _lattice_tail_error(noise.temporal[0], raw, errors, prefix)
    inverse, _ = _inverse_root_geometry(noise._spatial_root, lower=True)
    trailing = _triangular_image_error(noise._spatial_root, tail.T, white[prefix:].T,
        tail_errors.T, inverse).T
    return white, np.concatenate((leading, trailing))


def _event_columns(onsets, labels, native_samples, basis):
    width = basis.shape[1]
    design = np.zeros((native_samples, 2*width))
    errors = np.zeros_like(design)
    for onset, included in zip(onsets, labels, strict=True):
        right = min(native_samples, onset+len(basis))
        if onset < right:
            value = basis[:right-onset]
            errors[onset:right, :width] += _roundoff_gamma(1)*(
                np.abs(design[onset:right, :width])+np.abs(value))
            design[onset:right, :width] += value
            if included:
                errors[onset:right, width:] += _roundoff_gamma(1)*(
                    np.abs(design[onset:right, width:])+np.abs(value))
                design[onset:right, width:] += value
    return design, errors


@dataclass(frozen=True)
class SharedHistoryKernel:
    """Frozen canonical source factors and measured spatial relation errors."""
    source: object
    white_mean: np.ndarray
    white_mean_errors: np.ndarray
    pulse: np.ndarray
    pulse_errors: np.ndarray
    spatial_factorization_error: float
    proper_prior_covariance_error: float
    binding: dict


def shared_history_kernel(source):
    noise = source.parameters.noise
    if noise.family != 'shared' or any(
            not np.array_equal(t.reflection_coefficients, noise.temporal[0].reflection_coefficients)
            for t in noise.temporal[1:]):
        raise ValueError('separable history requires the original shared temporal source noise')
    channels = len(noise.geometry.channels)
    spatial = source.spatial_covariance_factor
    if not np.array_equal(noise._spatial_root, spatial):
        raise ValueError('fast history requires the same stored spatial factor; use the full physical prior programme')
    spatial_defect = 0.
    inverse, _ = _inverse_root_geometry(noise._spatial_root, lower=True)
    prior_error = source.canonical_factor_binding['saved_prior_chart_relation_error']
    certificate = getattr(source, 'consumer_prior_chart_certificate', None)
    if certificate is None and hasattr(source.parameters.prior, 'root'):
        from research.joint_flat_chart_precision import saved_source_chart_certificate

        certificate = saved_source_chart_certificate(source.temporal_covariance_factor,
            source.spatial_covariance_factor, source.parameters.prior.root)
    if certificate is not None:
        prior_error = min(prior_error, certificate['covariance_relation_error'])
    if not np.isfinite(prior_error) or prior_error > np.sqrt(np.finfo(float).eps):
        raise ValueError('canonical source prior relation requires the full saved physical-prior programme')
    mean = source.parameters.prior.mean.reshape(-1, channels)
    white_mean = solve_triangular(noise._spatial_root, mean.T, lower=True).T
    mean_errors = _triangular_image_error(noise._spatial_root, mean.T, white_mean.T,
        np.zeros_like(mean.T), inverse, accurate=True).T
    scale = np.sqrt(source.parameters.hyperparameters.relative_noise_shape)
    mean_errors = mean_errors/scale+_roundoff_gamma(2)*np.abs(white_mean/scale)
    white_mean = white_mean/scale
    basis, prefix = source.parameters.basis, noise.conditioning_samples
    raw = np.zeros((prefix+len(basis)+noise.order, basis.shape[1]))
    raw[prefix:prefix+len(basis)] = basis
    pulse, pulse_errors = _lattice_tail_error(noise.temporal[0], raw, np.zeros_like(raw), prefix)
    return SharedHistoryKernel(source, frozen_array(white_mean), frozen_array(mean_errors),
        frozen_array(pulse), frozen_array(pulse_errors), spatial_defect, prior_error,
        freeze_metadata({'schema': 'n2p3_joint_flat_shared_history_kernel/1',
            'source_parameters': source.parameters.digest, 'source_codec_sha256': source.digest,
            'canonical_source_factors': source.canonical_factor_binding,
            'spatial_original_map_relation_error_estimate': spatial_defect,
            'proper_prior_covariance_relation_error_estimate': prior_error,
            'consumer_prior_chart_certificate': certificate,
            'saved_temporal_factor_role': 'fixed_exact_invertible_coordinate_operand',
            'source_precision_inverse_defect_role': 'original_source_identity_check_not_a_second_chart_input_error',
            'original_startup_map_replaced': False,
            'white_pulse_sha256': array_digest(pulse), 'white_pulse_error_sha256': array_digest(pulse_errors),
            'zero_padding_role': 'finite_response_difference_only_no_new_observations'}))


def pulse_target_image(kernel, onsets, *, stop, empty_start):
    """One target event subset, sharing the original source pulse and factors."""
    source = kernel.source
    basis_width = source.parameters.basis.shape[1]
    scale = np.sqrt(source.parameters.hyperparameters.relative_noise_shape)
    mapping = source.temporal_covariance_factor[basis_width:]/scale
    mapping_errors = _roundoff_gamma(2)*np.abs(mapping)+kernel.spatial_factorization_error*np.abs(mapping)
    occurred = tuple(int(onset) for onset in onsets if onset < stop)
    start = min(occurred) if occurred else empty_start
    support_stop = min(stop, max(occurred)+len(kernel.pulse)) if occurred else start
    return PulseTargetImage(start, support_stop, occurred, kernel.pulse, kernel.pulse_errors,
        mapping, mapping_errors, kernel.white_mean[basis_width:], kernel.white_mean_errors[basis_width:])


def prepare_shared_record(kernel, samples, onsets, target_mask, *, record_id, native_stop=None):
    """All original channels retained; source prior once; original startup exact.

    Unknown target selections use a zero target_mask in the base and their
    complete public target axes. Known authorized C records supply their actual
    binary labels directly. Neither route copies C posterior into a new prior.
    """
    source, noise = kernel.source, kernel.source.parameters.noise
    samples = np.asarray(samples, dtype=float)
    native_stop = len(samples) if native_stop is None else native_stop
    prefix, channels = noise.conditioning_samples, len(noise.geometry.channels)
    if not prefix < native_stop <= len(samples) or samples.shape[1:] != (channels,):
        raise UnidentifiableFitError('the actual record requires its original startup and observed positive tail')
    samples = samples[:native_stop]
    events = [(int(t), bool(q)) for t, q in zip(onsets, target_mask, strict=True) if t < native_stop]
    design, design_error = _event_columns([t for t, _ in events], [q for _, q in events],
                                          native_stop, source.parameters.basis)
    if np.any(design[:prefix] != 0):
        raise ValueError('shared fast history requires structurally zero ERP design in the original joint startup')
    time = np.arange(native_stop, dtype=float)/noise.geometry.sfreq
    drift = np.column_stack((np.ones(native_stop), time))
    drift_error = np.column_stack((np.zeros(native_stop), _roundoff_gamma(1)*np.abs(time)))
    temporal_rows = np.column_stack((drift, design))
    temporal_errors = np.column_stack((drift_error, design_error))
    white_temporal, white_temporal_errors = _lattice_tail_error(noise.temporal[0], temporal_rows,
                                                             temporal_errors, prefix)
    factor = source.temporal_covariance_factor
    scale = np.sqrt(source.parameters.hyperparameters.relative_noise_shape)
    beta = white_temporal[:, 2:]@factor/scale
    beta_error = white_temporal_errors[:, 2:]@np.abs(factor)/scale
    beta_error += _roundoff_gamma(len(factor)+2)*(np.abs(white_temporal[:, 2:])@np.abs(factor))/scale
    beta_error += kernel.spatial_factorization_error*np.abs(beta)
    z, z_error = white_temporal[:, :2], white_temporal_errors[:, :2]
    z_error += kernel.spatial_factorization_error*np.abs(z)
    mean = source.parameters.prior.mean.reshape(len(factor), channels)
    response = samples-design@mean
    response_error = design_error@np.abs(mean)+_roundoff_gamma(len(factor))*(np.abs(design)@np.abs(mean))
    response_error += _roundoff_gamma(1)*(np.abs(samples)+np.abs(design@mean))
    y, y_error = _original_white_cells(noise, response, response_error)
    y_error = y_error/scale+_roundoff_gamma(2)*np.abs(y/scale)
    y = y/scale
    # a_white -> original physical da: sqrt(shape) * source L_Sigma.
    spatial = scale*source.spatial_covariance_factor
    raw_z = np.einsum('tr,cd->tcrd', drift[:prefix], spatial).reshape(prefix*channels, 2*channels)
    raw_z_error = np.einsum('tr,cd->tcrd', drift_error[:prefix], np.abs(spatial)).reshape(raw_z.shape)
    raw_z_error += _roundoff_gamma(3)*np.abs(raw_z)
    leading = solve_triangular(noise.compiled_prefix_root, raw_z, lower=True)
    inverse, _ = _inverse_root_geometry(noise.compiled_prefix_root, lower=True)
    leading_error = _triangular_image_error(noise.compiled_prefix_root, raw_z, leading, raw_z_error, inverse,
                                            accurate=True)
    leading_error = leading_error/scale+_roundoff_gamma(2)*np.abs(leading/scale)
    leading = leading/scale
    startup_rows = np.column_stack((leading, y[:prefix].ravel()))
    startup_errors = np.column_stack((leading_error, y_error[:prefix].ravel()))
    tail_rows = np.column_stack((z, beta, y[prefix:]))
    tail_errors = np.column_stack((z_error, beta_error, y_error[prefix:]))
    record = SharedFactorRecord(record_id, channels, prefix, startup_rows, startup_errors,
        tail_rows, tail_errors, len(factor), 2)
    binding = {'schema': 'n2p3_joint_flat_actual_shared_record_operands/1',
        'record_id': record_id, 'kernel': kernel.binding, 'native_rows': native_stop,
        'native_channels': channels, 'raw_scalar_rows': native_stop*channels,
        'samples_sha256': array_digest(samples), 'known_event_input_sha256': array_digest(np.asarray(events)),
        'proper_source_prior_occurrences': 'one_global_in_history_not_once_per_record',
        'one_explicit_drift_and_Gamma_for_this_true_record': True,
        'physical_drift_to_white_chart_logabsdet': float(2*np.log(np.diag(spatial)).sum()),
        'original_joint_startup_rows_sha256': array_digest(startup_rows, startup_errors),
        'shared_temporal_tail_rows_sha256': array_digest(tail_rows, tail_errors),
        'parameter_fits': 0, 'read_permission_issued_here': False}
    return record, freeze_metadata(binding)
