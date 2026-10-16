"""Sparse differences under the ORIGINAL finite causal innovation map.

Only a response *difference* is zero outside its physical support. Local zeros
do not represent observed EEG, a new recording, an AR restart or a noise fit.
The real observation uses the original complete record's startup and history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from research.channel_noise_mechanism import (
    FittedChannelNoise,
    _column_norms,
    _inverse_root_geometry,
    _roundoff_gamma,
    _score_whitening_error,
    _score_whitening_geometry,
    _triangular_image_error,
)
from research.joint_flat_chain import SparseResponseImage
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    metadata_digest,
)


@dataclass(frozen=True)
class OwnedSparseImage:
    image: SparseResponseImage
    column_error: np.ndarray
    binding: dict


def whiten_record_prefix(noise, values, *, source_error=None):
    """Original leading block, including prefixes shorter than saved startup."""
    if not isinstance(noise, FittedChannelNoise):
        raise TypeError("the original noise owner is required")
    raw = finite_real_array(values, name="actual original recording prefix operands")
    if raw.ndim != 3 or not len(raw) or raw.shape[1] != len(noise.geometry.channels):
        raise ValueError("prefix operands must preserve native sample / measured channel / column order")
    error = np.zeros_like(raw) if source_error is None else finite_real_array(source_error, name="original operand error")
    if error.shape != raw.shape or np.any(error < 0):
        raise ValueError("source error must follow the original operand cells")
    if len(raw) > noise.conditioning_samples:
        white = noise.whiten(raw)
        bound = _score_whitening_error(noise, raw, white, error, _score_whitening_geometry(noise))
        columns = bound.prefix_columns+bound.tail_columns
    else:
        rows = len(raw)*raw.shape[1]
        root = noise.compiled_prefix_root[:rows, :rows]
        white = solve_triangular(root, raw.reshape(rows, -1), lower=True).reshape(raw.shape)
        inverse, _ = _inverse_root_geometry(root, lower=True)
        columns = _column_norms(_triangular_image_error(root, raw.reshape(rows, -1),
            white.reshape(rows, -1), error.reshape(rows, -1), inverse, accurate=True))
    return white, frozen_array(columns)


def compile_sparse_response_image(noise, values, *, start_native, total_native,
                                  source_error=None):
    """Same W image on its exact innovation support, without a full zero matrix.

    Off-startup images use >=the ORIGINAL compiled-prefix number of zero
    difference rows, so their nonzero values enter the saved tail branch.
    On-startup images use the actual original leading map. Dense startup
    support remains present until its true boundary; tail support is L+AR p.
    """
    if not isinstance(noise, FittedChannelNoise):
        raise TypeError("a frozen original noise owner is required")
    raw = finite_real_array(values, name="original finite response image")
    if (raw.ndim != 3 or raw.shape[1] != len(noise.geometry.channels)
            or type(start_native) is not int or start_native < 0
            or type(total_native) is not int or total_native < 1):
        raise ValueError("the actual native support and measured channel image are required")
    errors = np.zeros_like(raw) if source_error is None else finite_real_array(source_error, name="response image source error")
    if errors.shape != raw.shape or np.any(errors < 0):
        raise ValueError("response image errors must preserve its original cells")
    channels, width = raw.shape[1:]
    if start_native >= total_native or not len(raw):
        empty = SparseResponseImage(0, np.empty((0, width)))
        return OwnedSparseImage(empty, frozen_array(np.zeros(width)), freeze_metadata({
            "program": "structural_zero_before_actual_cutoff", "native_support": [start_native, start_native+len(raw)],
            "actual_native_stop": total_native, "noise_identity": metadata_digest(noise.record())}))
    prefix, order = noise.conditioning_samples, noise.order
    retained = min(len(raw), total_native-start_native)
    stop = min(total_native, start_native+retained+order)
    if start_native < prefix:
        stop = min(total_native, max(stop, prefix))
        working = np.zeros((stop, channels, width))
        working_error = np.zeros_like(working)
        working[start_native:start_native+retained] = raw[:retained]
        working_error[start_native:start_native+retained] = errors[:retained]
        white, bound = whiten_record_prefix(noise, working, source_error=working_error)
        start, value = 0, white
        method = "original_leading_square_map_then_original_finite_tail"
    else:
        # Pad only the *difference* with zeros. The real record keeps its
        # original EEG/history and has exactly one startup/precision variable.
        length = stop-start_native
        working = np.zeros((prefix+length, channels, width))
        working_error = np.zeros_like(working)
        working[prefix:prefix+retained] = raw[:retained]
        working_error[prefix:prefix+retained] = errors[:retained]
        white, bound = whiten_record_prefix(noise, working, source_error=working_error)
        start, value = start_native*channels, white[prefix:]
        method = "homogeneous_zero_difference_pad_then_original_saved_tail_branch"
    image = SparseResponseImage(start, value.reshape(-1, width), bound)
    binding = {"schema": "n2p3_joint_flat_sparse_original_W/1", "program": method,
        "noise_identity": metadata_digest(noise.record()), "actual_raw_image_sha256": array_digest(raw),
        "actual_white_image_sha256": array_digest(image.values), "source_error_sha256": array_digest(errors),
        "native_source_support": [start_native, start_native+len(raw)],
        "native_white_support": [start//channels, start//channels+len(image.values)//channels],
        "actual_native_stop": total_native, "original_stationary_prefix": prefix,
        "temporal_AR_order": order, "new_observed_padding_rows": 0, "new_recordings": 0,
        "noise_fits": 0, "same_original_temporal_and_spatial_operator": True}
    return OwnedSparseImage(image, bound, freeze_metadata(binding))


def response_target_image(prior, basis, onsets, membership, *, start_native, stop_native,
                           observed_channels):
    """Original target-increment Dm/DL image, in source physical coordinates."""
    basis = finite_real_array(basis, name="actual original ERP basis")
    p, channels = len(prior.mean), observed_channels
    if basis.ndim != 2 or p != 2*basis.shape[1]*channels:
        raise ValueError("target response image must match the original common/increment/channel prior")
    temporal = np.zeros((stop_native-start_native, basis.shape[1]))
    absolute_temporal = np.zeros_like(temporal)
    additions = 0
    for onset, included in zip(onsets, membership, strict=True):
        if not included:
            continue
        left, right = max(start_native, int(onset)), min(stop_native, int(onset)+len(basis))
        if left < right:
            temporal[left-start_native:right-start_native] += basis[left-int(onset):right-int(onset)]
            absolute_temporal[left-start_native:right-start_native] += np.abs(basis[left-int(onset):right-int(onset)])
            additions += 1
    root = prior.root.reshape(2*basis.shape[1], channels, p)[basis.shape[1]:]
    mean = prior.mean.reshape(2*basis.shape[1], channels)[basis.shape[1]:]
    values = np.empty((len(temporal), channels, p+1))
    values[:, :, :p] = np.einsum("nk,kcp->ncp", temporal, root)
    values[:, :, p] = temporal@mean
    errors = np.empty_like(values)
    errors[:, :, :p] = _roundoff_gamma(basis.shape[1]) * np.einsum("nk,kcp->ncp", np.abs(temporal), np.abs(root))
    errors[:, :, p] = _roundoff_gamma(basis.shape[1])*(np.abs(temporal)@np.abs(mean))
    temporal_error = _roundoff_gamma(additions)*absolute_temporal
    errors[:, :, :p] += np.einsum("nk,kcp->ncp", temporal_error, np.abs(root))
    errors[:, :, p] += temporal_error@np.abs(mean)
    return values, errors
