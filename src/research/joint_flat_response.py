"""Joint response / recording scale with a declared physical flat da measure.

The existing joint-response QR and Gibbs loop own beta and Gamma sampling.
This factory adds record-local flat nuisance, its physical volume, and a proper
raw-tail prediction using the same recording's prefix drift posterior. Raw and
effective row counts remain separate. A flat marginal is a potential; only
the tail conditional is a normalized density in the actual raw tail values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    FittedChannelNoise,
    _column_norms,
    _conditional_minimizer_norm,
    _diffuse_qr_geometry,
    _inverse_root_geometry,
    _project_diffuse_arrays,
    _response_nuisance_qr,
    _roundoff_gamma,
    _row_dot,
    _row_dot_gamma,
    _score_whitening_error,
    _score_whitening_geometry,
    _triangular_image_error,
)
from research.joint_response_sampling import (
    PredictiveSamplingError,
    _batch_mcse,
    _functional_diagnostics,
    _predictive_policy,
    posterior_diagnostics,
    prepare_model,
)
from research.joint_response_scale import (
    CompiledDecision,
    DecisionBlock,
    _array,
    _augmented_qr,
    _compilation_chart,
    _input_digest,
    _numeric,
    _prefix_cut,
    _root_energy,
    compile_decision,
    compile_owned_decision,
)
from research.student_t_numerics import student_t_log_density
from transfer.noise_model import finite_real_array, freeze_metadata, frozen_array, metadata_digest


@dataclass(frozen=True, init=False, eq=False)
class FlatCompiledDecision(CompiledDecision):
    """Factory-owned quotient plus original physical da and prefix-tail state."""

    nuisance_rank: int
    effective_rows: int
    nuisance_logvolume: float
    measure_logjacobian: float
    nuisance_identity: str
    projection_resolution: dict
    prefix_drift_rhs: np.ndarray | None
    tail_root: np.ndarray | None
    tail_logdet: float | None
    earlier_prefixes: tuple
    energy_error_columns: np.ndarray
    normalizer_error: float
    reference_nu: float
    tail_error_columns: np.ndarray | None
    tail_normalizer_error: float | None
    tail_drift_root: np.ndarray | None
    tail_drift_linear: np.ndarray | None
    tail_drift_error: float | None
    tail_matrix_norm: float | None
    tail_smallest: float | None
    tail_data_norms: np.ndarray | None

    @property
    def density_dimension(self):
        return self.effective_rows

    @property
    def density_logdet(self):
        return self.noise_logdet+2*(self.nuisance_logvolume-self.measure_logjacobian)

    @property
    def likelihood_measure(self):
        return "physical_da_flat_potential"

    def validate_energy(self, charts, residual, energy):
        error = _energy_error(self.augmented_root, self.energy_error_columns, charts, residual, energy)
        sensitivity = (.5 if np.isposinf(self.reference_nu) else
            .5*(self.reference_nu+self.effective_rows)/(self.reference_nu+np.maximum(energy-error, 0)))
        nats = sensitivity*error+self.normalizer_error
        _numeric(error, nats, name="actual flat beta/rate energy error")
        if np.any(nats > np.sqrt(np.finfo(float).eps)*max(1, self.observed_rows)):
            raise DiffuseProjectionResolutionError("flat potential energy is unresolved at the actual beta/rate state")

    def validate_precision(self, charts, precision):
        vectors = np.column_stack((charts, -np.ones(len(charts))))
        residual = vectors @ self.augmented_root.T
        energy = np.einsum("bi,bi->b", residual, residual)
        error = _energy_error(self.augmented_root, self.energy_error_columns, charts, residual, energy)
        nats = .5*precision*error+self.normalizer_error
        _numeric(nats, name="actual conditional Gaussian flat energy error")
        if np.any(nats > np.sqrt(np.finfo(float).eps)*max(1, self.observed_rows)):
            raise DiffuseProjectionResolutionError("conditional flat Gaussian is unresolved at the actual joint scale")


def _energy_error(root, error_columns, charts, residual, energy):
    vectors = np.column_stack((charts, -np.ones(len(charts))))
    arithmetic = _roundoff_gamma(root.shape[1])*(np.abs(vectors) @ np.abs(root).T)
    norm_error = (np.abs(vectors) @ error_columns)+_column_norms(arithmetic.T)
    return (2*np.sqrt(energy)*norm_error+norm_error**2
            +_roundoff_gamma(root.shape[0]+1)*energy)


def _flat_node(white, z, *, decision_id, role, kind, chart, source, mapping, raw,
               nuisance_identity, logdet, measure_logjacobian, chunk_rows, nu, source_errors):
    q, root, scales = _response_nuisance_qr(z)
    d, y, resolution = _project_diffuse_arrays(
        white[:, :-1], white[:, -1], z, q, root, scales,
        response_reference_scale=min(1., np.sqrt(nu)))
    volume = float(np.log(abs(np.diag(root))).sum()+np.log(scales).sum())
    compressed, qr_errors = _augmented_qr(np.column_stack((d, y)), chunk_rows, with_error=True)
    r = z.shape[1]
    geometry = _diffuse_qr_geometry(z, q, root, scales)
    z_error = float(np.linalg.norm(source_errors[white.shape[1]:]/scales))
    smallest = float(np.linalg.svd(root, compute_uv=False)[-1])
    relative = z_error/smallest
    if relative >= 1 or not np.isfinite(relative):
        raise DiffuseProjectionResolutionError("original white flat subspace is unresolved")
    projection_error = 2*relative/(1-relative)
    errors = (np.asarray(resolution["chart_column_error_bounds"])
              if resolution["method"] == "source_scale_thin_QR_complete_prior_noise_chart"
              else np.zeros(white.shape[1]))
    errors = errors+source_errors[:white.shape[1]]+projection_error*_column_norms(white)
    if r == len(white):
        errors = np.zeros(white.shape[1])
    errors += qr_errors
    subspace = 0. if geometry is None else geometry[0]
    if subspace+relative >= 1:
        raise DiffuseProjectionResolutionError("flat coordinate volume is unresolved")
    normalizer_error = r*(-np.log1p(-(subspace+relative)))
    normalizer_error += _roundoff_gamma(4*r+len(white))*(abs(float(logdet))+2*abs(volume)+1)
    _numeric(compressed, logdet, volume, name="physical flat compiled potential")
    result = object.__new__(FlatCompiledDecision)
    fields = {"decision_id": decision_id, "role": role, "kind": kind,
        "augmented_root": frozen_array(compressed), "observed_rows": len(white),
        "row_start": 0, "row_stop": len(white), "noise_logdet": float(logdet),
        "chart_identity": chart, "source_identity": source, "map_identity": mapping,
        "raw_input_identity": raw, "prefix": None, "tail": None,
        "nuisance_rank": z.shape[1], "effective_rows": len(white)-z.shape[1],
        "nuisance_logvolume": volume, "measure_logjacobian": float(measure_logjacobian),
        "nuisance_identity": nuisance_identity, "projection_resolution": freeze_metadata(resolution),
        "prefix_drift_rhs": None, "tail_root": None, "tail_logdet": None, "earlier_prefixes": (),
        "energy_error_columns": frozen_array(errors), "normalizer_error": float(normalizer_error),
        "reference_nu": float(nu), "tail_error_columns": None, "tail_normalizer_error": None,
        "tail_drift_root": None, "tail_drift_linear": None, "tail_drift_error": None,
        "tail_matrix_norm": None, "tail_smallest": None, "tail_data_norms": None}
    for key, value in fields.items():
        object.__setattr__(result, key, value)
    return result, (q, root, scales)


def _flat_white(white, z, *, cut, full_logdet, prefix_logdet, prior, hyperparameters,
                decision_id, role, source, mapping, full_raw, prefix_raw,
                full_z_identity, prefix_z_identity, measure_logjacobian, qr_chunk_rows,
                earlier_prefixes=(), source_errors=None, prefix_source_errors=None):
    """Use full/prefix original rows and the SAME da chart, without root subtraction."""
    chart = _compilation_chart(prior, hyperparameters)
    common = {"decision_id": decision_id, "role": role, "chart": chart,
        "source": source, "mapping": mapping, "measure_logjacobian": measure_logjacobian,
        "chunk_rows": qr_chunk_rows, "nu": hyperparameters.degrees_of_freedom,
        "source_errors": source_errors}
    full, _ = _flat_node(white, z, kind="full", logdet=full_logdet, raw=full_raw,
                         nuisance_identity=full_z_identity, **common)
    if cut is None:
        return full
    prefix_errors = prefix_source_errors[cut]
    prefix, (q, root, scales) = _flat_node(white[:cut], z[:cut], kind="prefix",
        logdet=prefix_logdet, raw=prefix_raw, nuisance_identity=prefix_z_identity,
        **{**common, "source_errors": prefix_errors})
    # Zp=Q R diag(s). In a'=R diag(s) a, prefix a'|u,lambda
    # has mean Q.T(yp-Xp u) and covariance I/lambda.
    drift_rhs = _row_dot(q, white[:cut])
    tail_image = solve_triangular(root.T, (z[cut:]/scales).T, lower=True).T
    r, p = z.shape[1], len(prior.mean)
    # Keep prefix mean in proper PRIOR rows. Avoid constructing H*prefixRHS
    # before integration: its extrapolation is already covered by the same
    # drift posterior and belongs in the augmented posterior metric.
    prior_rows = np.column_stack((np.eye(r), drift_rhs))
    tail_rows = np.column_stack((tail_image, white[cut:]))
    augmented_rows = np.vstack((prior_rows, tail_rows))
    augmented, qr_errors = _augmented_qr(augmented_rows, qr_chunk_rows, with_error=True)
    conditional_root = augmented[r:, r:]
    conditional_logdet = float(full_logdet-prefix_logdet
        +2*np.log(abs(np.diag(augmented[:r, :r]))).sum())
    inverse_bound, inverse_norm = _inverse_root_geometry(root)
    z_reconstruction = float(np.linalg.norm(z[:cut]/scales-q@root))
    z_reconstruction += _roundoff_gamma(r+1)*float(np.linalg.norm(np.abs(q)@np.abs(root)))
    z_reconstruction += float(np.linalg.norm(prefix_errors[p+1:]/scales))
    relative = z_reconstruction*inverse_norm
    if relative >= 1 or not np.isfinite(relative):
        raise DiffuseProjectionResolutionError("prefix flat posterior source geometry is unresolved")
    q_error = (2*relative/(1-relative)+float(np.linalg.norm(_row_dot(q, q)-np.eye(r)))
               +_row_dot_gamma(cut)*float(np.linalg.norm(_row_dot(np.abs(q), np.abs(q)))))
    image_norm = float(np.linalg.norm(tail_image))
    image_error = (float(np.linalg.norm(source_errors[p+1:]/scales))*float(np.linalg.norm(inverse_bound))
                   +relative*image_norm/(1-relative)
                   +_roundoff_gamma(3*r)*image_norm)
    rhs_error = ((q_error+_row_dot_gamma(cut))*_column_norms(white[:cut])
                 +prefix_errors[:p+1])
    local_smallest = float(np.linalg.svd(augmented[:r, :r], compute_uv=False)[-1])
    drift_error = image_error+float(np.linalg.norm(qr_errors[:r]))
    local_relative = drift_error/local_smallest
    if not np.isfinite(local_relative) or local_relative >= 1:
        raise DiffuseProjectionResolutionError("prefix drift predictive factor is unresolved")
    conditional_error = source_errors[:p+1]+rhs_error+qr_errors[r:]
    conditional_normalizer_error = (full.normalizer_error+prefix.normalizer_error
                                    +2*r*(relative/(1-relative)+local_relative/(1-local_relative)))
    _numeric(drift_rhs, conditional_root, conditional_logdet, name="raw tail with prefix flat posterior")
    object.__setattr__(full, "prefix", prefix)
    object.__setattr__(full, "prefix_drift_rhs", frozen_array(drift_rhs))
    object.__setattr__(full, "tail_root", frozen_array(conditional_root))
    object.__setattr__(full, "tail_logdet", conditional_logdet)
    object.__setattr__(full, "tail_error_columns", frozen_array(conditional_error))
    object.__setattr__(full, "tail_normalizer_error", float(conditional_normalizer_error))
    object.__setattr__(full, "tail_drift_root", frozen_array(augmented[:r, :r]))
    object.__setattr__(full, "tail_drift_linear", frozen_array(augmented[:r, r:]))
    object.__setattr__(full, "tail_drift_error", float(drift_error))
    object.__setattr__(full, "tail_matrix_norm", float(np.linalg.norm(augmented[:r, :r], ord=2)))
    object.__setattr__(full, "tail_smallest", local_smallest)
    object.__setattr__(full, "tail_data_norms", frozen_array(_column_norms(augmented_rows[:, r:])))
    earlier = tuple(_flat_node(white[:stop], z[:stop], kind="prefix", logdet=logdet,
        raw=raw, nuisance_identity=z_identity, **{**common, "source_errors": prefix_source_errors[stop]})[0]
        for stop, logdet, raw, z_identity in earlier_prefixes)
    object.__setattr__(full, "earlier_prefixes", earlier)
    return full


def compile_flat_decision(prior, hyperparameters, block, nuisance, *, conditioning_rows=None,
                          measure_logjacobian=0., qr_chunk_rows=2048):
    """Dense oracle entry; Z columns denote physical flat coordinates.

    For Z'=Z T and a=T b, pass log|det T| as measure_logjacobian;
    the same physical da potential and raw conditional are then preserved.
    Empty Z returns the existing proper compiled state without another fitter.
    """
    _compilation_chart(prior, hyperparameters)
    if not isinstance(block, DecisionBlock):
        raise TypeError("a canonical raw decision is required")
    z = _array(nuisance, 2, "actual physical flat design")
    n, p = len(block.samples), len(prior.mean)
    if z.shape[0] != n or z.shape[1] > n or block.design.shape[1] != p:
        raise ValueError("Y/D/Z must share the original physical rows and beta chart")
    _prefix_cut(conditioning_rows, n)
    if not z.shape[1]:
        if measure_logjacobian != 0:
            raise ValueError("empty flat coordinates have unit coordinate volume")
        return compile_decision(prior, hyperparameters, block,
            conditioning_rows=conditioning_rows, qr_chunk_rows=qr_chunk_rows)
    volume = finite_real_array(measure_logjacobian, name="physical da chart Jacobian")
    if volume.ndim != 0:
        raise ValueError("physical da Jacobian must be a finite scalar")
    cut = conditioning_rows
    augmented = np.column_stack((block.design @ prior.root,
        block.samples-block.design @ prior.mean, z))
    source_error = np.column_stack((
        _roundoff_gamma(p)*(np.abs(block.design)@np.abs(prior.root)),
        _roundoff_gamma(p+1)*(np.abs(block.samples)+np.abs(block.design)@np.abs(prior.mean)),
        np.zeros_like(z)))
    raw = augmented/np.sqrt(hyperparameters.relative_noise_shape)
    source_error = source_error/np.sqrt(hyperparameters.relative_noise_shape)+_roundoff_gamma(2)*np.abs(raw)
    white = solve_triangular(block.noise_root, raw, lower=True)
    inverse_bound, _ = _inverse_root_geometry(block.noise_root, lower=True)
    error_matrix = _triangular_image_error(block.noise_root, raw, white, source_error, inverse_bound, accurate=True)
    source_errors = _column_norms(error_matrix)
    prefix_errors = {} if cut is None else {cut: _column_norms(error_matrix[:cut])}
    logdet = 2*np.log(np.diag(block.noise_root)).sum()+n*np.log(hyperparameters.relative_noise_shape)
    prefix_logdet = (None if cut is None else 2*np.log(np.diag(block.noise_root)[:cut]).sum()
                    +cut*np.log(hyperparameters.relative_noise_shape))
    raw = _input_digest(block.samples, block.design, z, block.noise_root)
    leading = None if cut is None else _input_digest(block.samples[:cut], block.design[:cut],
                                                    z[:cut], block.noise_root[:cut, :cut])
    _numeric(white, name="whole dense physical flat whitening")
    return _flat_white(white[:, :p+1], white[:, p+1:], cut=cut,
        full_logdet=logdet, prefix_logdet=prefix_logdet, prior=prior, hyperparameters=hyperparameters,
        decision_id=block.decision_id, role=block.role, source=raw,
        mapping=_input_digest(block.noise_root), full_raw=raw, prefix_raw=leading,
        full_z_identity=_input_digest(z), prefix_z_identity=None if cut is None else _input_digest(z[:cut]),
        measure_logjacobian=float(volume), qr_chunk_rows=qr_chunk_rows, source_errors=source_errors,
        prefix_source_errors=prefix_errors)


def compile_owned_flat_decision(prior, hyperparameters, noise, *, decision_id, role,
        native_samples, raw_batches, conditioning_samples=None, measure_logjacobian=0.,
        qr_chunk_rows=2048, earlier_conditioning_samples=()):
    """Whole same-record B05 whitening of original sample-major Y/D/Z once.

    raw_batches=(start,Y[N,C],D[N,C,p],Z[N,C,r]); batches are reader chunks,
    never independent records. No dense raw covariance is allocated.
    """
    _compilation_chart(prior, hyperparameters)
    if not isinstance(noise, FittedChannelNoise):
        raise TypeError("the original fitted B05 noise owner is required")
    if (type(native_samples) is not int or native_samples < 1
            or not isinstance(decision_id, str) or not decision_id.strip()
            or role not in {"target_calibration", "read_only_inference"}):
        raise ValueError("one physical recording, role and native sample count are required")
    _prefix_cut(conditioning_samples, native_samples)
    earlier_cuts = tuple(earlier_conditioning_samples)
    if (len(set(earlier_cuts)) != len(earlier_cuts) or earlier_cuts
            and (conditioning_samples is None or any(type(v) is not int or not 0 < v < conditioning_samples
                                                     for v in earlier_cuts))):
        raise ValueError("earlier shared-record prefixes must precede the actual prediction prefix")
    batches = tuple(raw_batches)
    if not batches:
        raise ValueError("original source rows are required")
    p, channels = len(prior.mean), len(noise.geometry.channels)
    first_z = finite_real_array(batches[0][3], name="physical flat chart")
    if first_z.ndim != 3:
        raise ValueError("flat image must be sample-major N/C/r")
    rank = first_z.shape[2]
    if rank == 0:
        if measure_logjacobian != 0:
            raise ValueError("empty flat coordinates have unit coordinate volume")
        return compile_owned_decision(prior, hyperparameters, noise, decision_id=decision_id,
            role=role, native_samples=native_samples, raw_batches=[v[:3] for v in batches],
            conditioning_samples=conditioning_samples, qr_chunk_rows=qr_chunk_rows)
    # Retain one augmented array and operand hashes; no second W/noise fit.
    import hashlib

    shapes = ((native_samples, channels), (native_samples, channels, p),
              (native_samples, channels, rank))
    streams = [hashlib.sha256(str((shape, np.dtype(float).str)).encode()) for shape in shapes]
    leading = None if conditioning_samples is None else [hashlib.sha256(
        str(((conditioning_samples, *shape[1:]), np.dtype(float).str)).encode()) for shape in shapes]
    earlier_streams = {cut: [hashlib.sha256(str(((cut, *shape[1:]), np.dtype(float).str)).encode())
                            for shape in shapes] for cut in earlier_cuts}
    white = np.empty((native_samples, channels, p+1+rank))
    source_error = np.empty_like(white)
    offset = 0
    for start, samples, design, nuisance in batches:
        y, d, z = (finite_real_array(value, name=name) for value, name in
            ((samples, "actual Y"), (design, "actual physical D"), (nuisance, "actual physical Z")))
        if (type(start) is not int or start != offset or y.ndim != 2 or not len(y)
                or y.shape[1] != channels or d.shape != (*y.shape, p)
                or z.shape != (*y.shape, rank) or offset+len(y) > native_samples):
            raise ValueError("contiguous original Y/D/Z chunks must share the actual record")
        for digest, value in zip(streams, (y, d, z), strict=True):
            digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        if leading is not None:
            count = min(len(y), max(0, conditioning_samples-offset))
            for digest, value in zip(leading, (y[:count], d[:count], z[:count]), strict=True):
                if count:
                    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        for cut, digests in earlier_streams.items():
            count = min(len(y), max(0, cut-offset))
            if count:
                for digest, value in zip(digests, (y[:count], d[:count], z[:count]), strict=True):
                    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        stop = offset+len(y)
        white[offset:stop, :, :p] = d @ prior.root
        white[offset:stop, :, p] = y-d @ prior.mean
        white[offset:stop, :, p+1:] = z
        source_error[offset:stop, :, :p] = _roundoff_gamma(p)*(np.abs(d)@np.abs(prior.root))
        source_error[offset:stop, :, p] = _roundoff_gamma(p+1)*(np.abs(y)+np.abs(d)@np.abs(prior.mean))
        source_error[offset:stop, :, p+1:] = 0.
        offset = stop
    if offset != native_samples:
        raise ValueError("every declared native row must be actually supplied")
    volume = finite_real_array(measure_logjacobian, name="physical da chart Jacobian")
    if volume.ndim != 0:
        raise ValueError("physical da Jacobian must be scalar")
    mapping = metadata_digest(noise.record())
    source = metadata_digest([v.hexdigest() for v in streams])
    raw = metadata_digest({"source": source, "map": mapping})
    prefix_raw = None if leading is None else metadata_digest({
        "source": metadata_digest([v.hexdigest() for v in leading]), "map": mapping})
    white /= np.sqrt(hyperparameters.relative_noise_shape)
    source_error /= np.sqrt(hyperparameters.relative_noise_shape)
    source_error += _roundoff_gamma(2)*np.abs(white)
    actual_white = noise.whiten(white)
    error_geometry = _score_whitening_geometry(noise)
    error = _score_whitening_error(noise, white, actual_white, source_error, error_geometry)
    source_errors = error.prefix_columns+error.tail_columns
    prefix_errors = {}
    requested_prefixes = (*earlier_cuts, *((conditioning_samples,) if conditioning_samples is not None else ()))
    for stop in requested_prefixes:
        if stop > noise.conditioning_samples:
            # The original causal map's leading output is already available;
            # propagate only its leading operands. No W or AR restart occurs.
            error = _score_whitening_error(noise, white[:stop], actual_white[:stop],
                                           source_error[:stop], error_geometry)
            columns = error.prefix_columns+error.tail_columns
        else:
            width = stop*channels
            leading_root = noise.compiled_prefix_root[:width, :width]
            inverse_bound, _ = _inverse_root_geometry(leading_root, lower=True)
            columns = _column_norms(_triangular_image_error(leading_root,
                white[:stop].reshape(width, -1), actual_white[:stop].reshape(width, -1),
                source_error[:stop].reshape(width, -1), inverse_bound, accurate=True))
        prefix_errors[stop*channels] = columns
    white = actual_white.reshape(native_samples*channels, p+1+rank)
    del source_error
    _numeric(white, name="whole original record flat whitening")
    n = native_samples*channels
    cut = None if conditioning_samples is None else conditioning_samples*channels
    logdet = -2*noise.whitening_logabsdet(native_samples)+n*np.log(hyperparameters.relative_noise_shape)
    prefix_logdet = None if cut is None else -2*noise.whitening_logabsdet(conditioning_samples)+cut*np.log(
        hyperparameters.relative_noise_shape)
    earlier_states = tuple((stop*channels,
        -2*noise.whitening_logabsdet(stop)+stop*channels*np.log(hyperparameters.relative_noise_shape),
        metadata_digest({"source": metadata_digest([v.hexdigest() for v in digests]), "map": mapping}),
        digests[2].hexdigest()) for stop, digests in earlier_streams.items())
    return _flat_white(white[:, :p+1], white[:, p+1:], cut=cut, full_logdet=logdet,
        prefix_logdet=prefix_logdet, prior=prior, hyperparameters=hyperparameters,
        decision_id=decision_id, role=role, source=source, mapping=mapping, full_raw=raw,
        prefix_raw=prefix_raw, full_z_identity=streams[2].hexdigest(),
        prefix_z_identity=None if leading is None else leading[2].hexdigest(),
        measure_logjacobian=float(volume), qr_chunk_rows=qr_chunk_rows, earlier_prefixes=earlier_states,
        source_errors=source_errors, prefix_source_errors=prefix_errors)


def flat_log_potential(record, charts, nu):
    """Gamma-integrated potential with physical da volume and effective rows."""
    return student_t_log_density(_root_energy(record, np.asarray(charts)),
        degrees_of_freedom=nu, dimension=record.density_dimension,
        noise_logdet=record.density_logdet)


@dataclass(frozen=True)
class FlatTailPrediction:
    full: FlatCompiledDecision
    prior: object
    hyperparameters: object
    prefix_already_in_model: bool
    conditioning_reference: FlatCompiledDecision | None


def prepare_flat_prediction(model, full):
    """A same-record continuation or a separately declared recording's prefix.

    Binding is to original prefix Y/D/Z/map/physical-volume operands.
    A same recording never receives a second independent Gamma variable.
    """
    if (not isinstance(full, FlatCompiledDecision) or full.prefix is None or full.tail_root is None
            or full.chart_identity != _compilation_chart(model.prior, model.hyperparameters)):
        raise ValueError("a factory full/prefix flat map under the sampled model is required")
    matches = [record for record in model.blocks if record.decision_id == full.decision_id]
    reference = None
    already = False
    if matches:
        actual = matches[0]
        matching_prefixes = [v for v in (full.prefix, *full.earlier_prefixes)
            if isinstance(actual, FlatCompiledDecision) and v.raw_input_identity == actual.raw_input_identity
            and v.nuisance_identity == actual.nuisance_identity and v.observed_rows == actual.observed_rows
            and v.measure_logjacobian == actual.measure_logjacobian and v.noise_logdet == actual.noise_logdet]
        if len(matching_prefixes) != 1:
            raise ValueError("same-record prediction must preserve the exact original prefix da/map/rows")
        reference = matching_prefixes[0]
        already = reference is full.prefix
    return FlatTailPrediction(full, model.prior, model.hyperparameters, already, reference)


def flat_tail_components(prediction, charts):
    """Ordinary raw-tail t components; a posterior and SAME lambda integrated.

    Tail df is nu+n_prefix-r, shape is (nu+Qprefix)/(nu+n_prefix-r)
    times its prefix-a predictive shape. No flat potential is reported as a
    raw density. Prefix saturation (n_prefix=r) remains proper.
    """
    u = finite_real_array(charts, name="actual retained beta charts")
    p = len(prediction.prior.mean)
    if u.ndim not in (2, 3) or u.shape[-1] != p:
        raise ValueError("retained global charts must match the original physical prior")
    u = u.reshape(-1, p)
    full, nu = prediction.full, prediction.hyperparameters.degrees_of_freedom
    prefix = flat_log_potential(full.prefix, u, nu)
    if prediction.conditioning_reference is not None:
        prefix -= flat_log_potential(prediction.conditioning_reference, u, nu)
    residual = np.column_stack((u, -np.ones(len(u)))) @ full.tail_root.T
    energy = np.einsum("bi,bi->b", residual, residual)
    vectors = np.column_stack((u, -np.ones(len(u))))
    drift_rhs = -vectors @ full.tail_drift_linear.T
    drift = solve_triangular(full.tail_drift_root, drift_rhs.T, lower=False).T
    bounds = np.empty(len(u))
    for index, (point, rhs, value) in enumerate(zip(vectors, drift_rhs, drift, strict=True)):
        data_error = float(np.abs(point)@full.tail_error_columns)
        response_norm = float(np.abs(point)@full.tail_data_norms)
        solve_defect = float(np.linalg.norm(full.tail_drift_root@value-rhs))
        solve_defect += _roundoff_gamma(len(value)+1)*float(np.linalg.norm(
            np.abs(full.tail_drift_root)@np.abs(value)+np.abs(rhs)))
        solve_defect += _roundoff_gamma(len(point))*float(np.linalg.norm(
            np.abs(full.tail_drift_linear)@np.abs(point)))
        bound, _ = _conditional_minimizer_norm(value, full.tail_matrix_norm,
            response_norm, full.tail_smallest, full.tail_drift_error, data_error, solve_defect)
        bounds[index] = data_error+full.tail_drift_error*bound
    arithmetic = _roundoff_gamma(full.tail_root.shape[1])*(np.abs(vectors)@np.abs(full.tail_root).T)
    bounds += _column_norms(arithmetic.T)
    tail_error = 2*np.sqrt(energy)*bounds+bounds**2+_roundoff_gamma(full.tail_root.shape[0]+1)*energy
    dimension = full.observed_rows-full.prefix.observed_rows
    if np.isposinf(nu):
        logs = student_t_log_density(energy, degrees_of_freedom=nu,
            dimension=dimension, noise_logdet=full.tail_logdet)
    else:
        updated_nu = nu+full.prefix.effective_rows
        prefix_energy = _root_energy(full.prefix, u)
        prefix_residual = np.column_stack((u, -np.ones(len(u)))) @ full.prefix.augmented_root.T
        prefix_error = _energy_error(full.prefix.augmented_root, full.prefix.energy_error_columns,
                                     u, prefix_residual, prefix_energy)
        scale = (nu+prefix_energy)/updated_nu
        logs = np.array([student_t_log_density(q/s, degrees_of_freedom=updated_nu,
            dimension=dimension, noise_logdet=full.tail_logdet+dimension*np.log(s))
            for q, s in zip(energy, scale, strict=True)])
    # The compact tail state retains source-scale errors. Admit the actual
    # retained beta/rate points rather than assuming compile admission covers
    # unbounded beta draws or arbitrarily small Gamma rates.
    lower_rate = None if np.isposinf(nu) else nu+np.maximum(prefix_energy-prefix_error, 0)
    sensitivity = (.5 if np.isposinf(nu) else
        .5*(updated_nu+dimension)/(lower_rate+np.maximum(energy-tail_error, 0)))
    nats_error = sensitivity*tail_error+full.tail_normalizer_error
    if not np.isposinf(nu):
        rate_sensitivity = .5*(dimension/lower_rate
            +(updated_nu+dimension)/lower_rate*energy/(lower_rate+energy))
        nats_error += rate_sensitivity*prefix_error
    _numeric(logs, nats_error, name="actual posterior raw-tail density and source error")
    if np.any(nats_error > np.sqrt(np.finfo(float).eps)*dimension):
        raise DiffuseProjectionResolutionError(
            "raw flat tail is unresolved at the actual posterior beta/rate points: "
            f"error={float(nats_error.max()):.17g}, budget={np.sqrt(np.finfo(float).eps)*dimension:.17g}, "
            f"energy={float(energy.max()):.17g}, energy_error={float(tail_error.max()):.17g}, "
            f"normalizer_error={full.tail_normalizer_error:.17g}")
    return np.asarray(prefix), np.asarray(logs)


def predict_flat_tail(samples, prediction, *, batch_size, max_rhat,
                      min_bulk_ess, min_tail_ess):
    """Read-only finite posterior mixture; sampling choices remain explicit."""
    _predictive_policy(max_rhat, min_bulk_ess, min_tail_ess)
    if (prediction.prior is not samples.model.prior
            or prediction.hyperparameters != samples.model.hyperparameters):
        raise ValueError("prediction must use the actual sampled physical prior/noise program")
    diagnostics = posterior_diagnostics(samples)
    if (diagnostics["max_rank_folded_split_rhat"] > max_rhat
            or diagnostics["min_bulk_ess"] < min_bulk_ess
            or diagnostics["min_tail_ess"] < min_tail_ess):
        raise PredictiveSamplingError("caller-declared posterior diagnostics were not met")
    prefix, logs = flat_tail_components(prediction, samples.chart)
    log_weights = (np.full(len(logs), -np.log(len(logs))) if prediction.prefix_already_in_model
                   else prefix-logsumexp(prefix))
    weights = np.exp(log_weights)
    density = float(logsumexp(logs+log_weights))
    influence = (np.exp(logs-density)-1 if prediction.prefix_already_in_model else
        np.exp(prefix+logs-logsumexp(prefix+logs)+np.log(len(logs)))
        -np.exp(prefix-logsumexp(prefix)+np.log(len(logs))))
    return {"log_density": density, "component_tail_logs": frozen_array(logs),
        "prefix_weights": frozen_array(weights), "diagnostics": diagnostics,
        "prefix_log_weights": frozen_array(log_weights),
        "log_density_batch_mcse": _batch_mcse(influence.reshape(samples.chart.shape[:2]), batch_size),
        "log_density_influence_diagnostics": _functional_diagnostics(influence.reshape(samples.chart.shape[:2])),
        "numerical_program": ("Gaussian_limit_flat_da_then_raw_tail_empirical_mixture" if samples.gaussian_limit
                              else "continuous_Gamma_Gibbs_flat_da_then_raw_tail_empirical_mixture"),
        "tail_density_measure": "original_raw_tail_observations",
        "flat_measure": "same_physical_da_full_and_prefix",
        "raw_prefix_rows": prediction.full.prefix.observed_rows,
        "prefix_flat_rank": prediction.full.prefix.nuisance_rank,
        "effective_prefix_rows": prediction.full.prefix.effective_rows,
        "same_record_Gamma_variable": prediction.conditioning_reference is not None,
        "finite_predictive_mixture_proper": True}


def prepare_flat_model(prior, records, hyperparameters):
    return prepare_model(prior, records, hyperparameters)
