"""Native temporal QR statistics for a private-AR1/full-Q joint programme.

Static temporal roots are independent of the fitted noise/within parameters.
Local integration and the global marginal remain with the existing owners.
Residual sufficient moments retain complete same-record adjacent-row pairs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.frozen_metadata import freeze_metadata
from research.gaussian_local_conditional import (
    integrate_local_augmented_root_with_conditional,
)
from research.joint_response_noise_calibration import (
    AR1InnovationStatistics,
    AR1ResidualStatistics,
    JointCalibrationWorkspace,
)
from research.joint_response_scale import _augmented_qr
from research.private_ar1_noise_point import PrivateAR1NoisePoint
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    frozen_array,
    metadata_digest,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeComponentSpec:
    selection_ids: tuple
    native_intervals: tuple


def original_white_components(supports, *, native_samples, conditioning_samples):
    """Conservative complete AR1 white supports; no numerical thresholding."""
    hulls, unobserved = [], []
    for name, (start, stop) in supports.items():
        if not 0 <= start <= stop <= native_samples:
            raise ValueError('response support lies outside its original record')
        if start == stop:
            unobserved.append(name)
            continue
        white_start = 0 if start < conditioning_samples else start
        white_stop = min(native_samples, max(stop+1, conditioning_samples if start < conditioning_samples else 0))
        hulls.append((white_start, white_stop, (name,)))
    grouped = []
    for start, stop, names in sorted(hulls):
        if grouped and start < grouped[-1][1]:
            previous = grouped[-1]
            grouped[-1] = (previous[0], max(previous[1], stop), previous[2]+names)
        else:
            grouped.append((start, stop, names))
    components, gaps, stop = [], [], 0
    for start, end, names in grouped:
        if stop < start:
            gaps.append((stop, start))
        components.append(NativeComponentSpec(names, ((start, end),)))
        stop = end
    if stop < native_samples:
        gaps.append((stop, native_samples))
    if gaps:
        components.append(NativeComponentSpec((), tuple(gaps)))
    if not components:
        components.append(NativeComponentSpec((), ((0, native_samples),)))
    if unobserved:
        index = len(components)-1 if gaps else 0
        value = components[index]
        components[index] = NativeComponentSpec(value.selection_ids+tuple(unobserved), value.native_intervals)
    return tuple(components)


class NativeJointRecordCompiler:
    """One original record: complete temporal features, no live-data authority."""

    def __init__(self, samples, basis_columns, selection_designs, supports, geometry, *,
            record_id, drift_group_id, branch_binding, block_native_rows=2048, conditioning_samples=32,
            component_indices=None):
        self.samples = finite_real_array(samples, name='original complete native observations')
        self.designs, self.supports, self.geometry = dict(selection_designs), dict(supports), geometry
        self.record_id, self.drift_group_id = record_id, drift_group_id
        self.p, self.c, self.n = basis_columns, len(geometry.channels), len(self.samples)
        self.block_rows, self.conditioning_samples = block_native_rows, conditioning_samples
        if (self.samples.shape != (self.n, self.c) or not self.n or not set(self.designs) <= set(self.supports)
                or type(self.p) is not int or self.p < 1 or type(block_native_rows) is not int or block_native_rows < 1):
            raise ValueError('native record compiler keeps its original samples, complete design columns and scopes')
        self.branch_binding = freeze_metadata(branch_binding)
        self.drift_ids = tuple(f'{drift_group_id}:{kind}:{channel}' for kind in ('intercept', 'slope') for channel in geometry.channels)
        self.specs = original_white_components(self.supports, native_samples=self.n, conditioning_samples=conditioning_samples)
        requested = tuple(range(len(self.specs))) if component_indices is None else tuple(component_indices)
        if (len(set(requested)) != len(requested) or any(type(i) is not int or not 0 <= i < len(self.specs) for i in requested)
                or any(not set(self.specs[i].selection_ids) <= set(self.designs) for i in requested)):
            raise ValueError('each requested complete support component has every actual selection callback')
        self.compiled_component_indices = requested
        self.observation_identity = array_digest(self.samples)
        self.intervals = tuple(sorted((start, stop, index) for index, spec in enumerate(self.specs)
            for start, stop in spec.native_intervals))
        cursor = 0
        for start, stop, _ in self.intervals:
            if start != cursor:
                raise ValueError('native white components cover the original record exactly once')
            cursor = stop
        if cursor != self.n:
            raise ValueError('native white components lost original observation rows')
        self.tail_roots, self.moment_roots = [], []
        for i, spec in enumerate(self.specs):
            if i not in requested:
                self.tail_roots.append(None)
                self.moment_roots.append(None)
                continue
            tail = tuple((max(start, conditioning_samples), stop) for start, stop in spec.native_intervals
                if max(start, conditioning_samples) < stop)
            internal = tuple((max(start+1, 1), stop) for start, stop in spec.native_intervals if max(start+1, 1) < stop)
            self.tail_roots.append(self._temporal_root(spec, tail))
            self.moment_roots.append(self._temporal_root(spec, internal))
        self.identity = metadata_digest({'record': record_id, 'samples': self.observation_identity,
            'geometry': geometry.record(), 'p': self.p, 'drift': self.drift_ids,
            'specs': [{'selections': spec.selection_ids, 'intervals': spec.native_intervals,
                'tail_root': None if tail_root is None else array_digest(tail_root),
                'moment_root': None if moment_root is None else array_digest(moment_root)}
                for spec, tail_root, moment_root in zip(self.specs, self.tail_roots, self.moment_roots, strict=True)],
            'branch': self.branch_binding, 'producer': IMPLEMENTATION_SHA256})

    def _feature_width(self, spec):
        return 2*len(spec.selection_ids)*self.p+4+2*self.c

    def _features(self, spec, start, stop):
        if start < 1:
            raise ValueError('paired temporal features require actual original predecessor rows')
        rows = stop-start
        out = np.empty((rows, self._feature_width(spec)))
        offset = 0
        for name in spec.selection_ids:
            design, error, _ = self.designs[name](start-1, stop)
            design, error = np.asarray(design), np.asarray(error)
            if design.shape != (rows+1, self.p) or error.shape != design.shape:
                raise ValueError('actual native temporal design changed its complete coefficient columns')
            out[:, offset:offset+self.p] = design[1:]
            out[:, offset+self.p:offset+2*self.p] = design[:-1]
            offset += 2*self.p
        t = np.arange(start, stop)/self.geometry.sfreq
        previous = np.arange(start-1, stop-1)/self.geometry.sfreq
        out[:, offset:offset+4] = np.column_stack((np.ones(rows), t, np.ones(rows), previous))
        offset += 4
        out[:, offset:offset+self.c] = self.samples[start:stop]
        out[:, offset+self.c:] = self.samples[start-1:stop-1]
        return out

    def _temporal_root(self, spec, intervals):
        root = np.empty((0, self._feature_width(spec)))
        for start, stop in intervals:
            for begin in range(start, stop, self.block_rows):
                end = min(stop, begin+self.block_rows)
                block = np.vstack((root, self._features(spec, begin, end)))
                root = _augmented_qr(block, max(1, len(block)))
        return frozen_array(root)

    def _prefix_physical(self, noise, spec):
        prefix = min(self.n, self.conditioning_samples)
        width = len(spec.selection_ids)*self.p*self.c+2*self.c+1
        raw = np.zeros((prefix, self.c, width))
        for i, name in enumerate(spec.selection_ids):
            design, _, _ = self.designs[name](0, prefix)
            for coefficient in range(self.p):
                for channel in range(self.c):
                    raw[:, channel, i*self.p*self.c+coefficient*self.c+channel] = design[:, coefficient]
        offset = len(spec.selection_ids)*self.p*self.c
        for channel in range(self.c):
            raw[:, channel, offset+channel] = 1.
            raw[:, channel, offset+self.c+channel] = np.arange(prefix)/self.geometry.sfreq
        raw[:, :, -1] = self.samples[:prefix]
        if type(noise) is PrivateAR1NoisePoint:
            white = noise.whiten_prefix(raw).reshape(prefix*self.c, width)
        else:
            white = solve_triangular(noise.compiled_prefix_root[:prefix*self.c, :prefix*self.c],
                raw.reshape(prefix*self.c, width), lower=True)
        return np.vstack([white[start*self.c:min(stop, prefix)*self.c]
            for start, stop in spec.native_intervals if start < prefix])

    def _tail_map(self, noise, spec):
        physical = len(spec.selection_ids)*self.p*self.c
        width = physical+2*self.c+1
        out = np.zeros((self._feature_width(spec), self.c, width))
        inverse = solve_triangular(noise._spatial_root, np.eye(self.c), lower=True)
        ar = np.array([value.coefficients[0] for value in noise.temporal])
        offset = 0
        for i, _ in enumerate(spec.selection_ids):
            for p in range(self.p):
                columns = i*self.p*self.c+p*self.c+np.arange(self.c)
                out[offset+p, :, columns] = inverse.T
                out[offset+self.p+p, :, columns] = (-ar[None, :]*inverse).T
            offset += 2*self.p
        for channel in range(self.c):
            out[offset, :, physical+channel] = inverse[:, channel]
            out[offset+1, :, physical+self.c+channel] = inverse[:, channel]
            out[offset+2, :, physical+channel] = -ar[channel]*inverse[:, channel]
            out[offset+3, :, physical+self.c+channel] = -ar[channel]*inverse[:, channel]
            out[offset+4+channel, :, -1] = inverse[:, channel]
            out[offset+4+self.c+channel, :, -1] = -ar[channel]*inverse[:, channel]
        return out

    def compile(self, noise, within_factor, source_chart):
        if len(self.compiled_component_indices) != len(self.specs):
            raise ValueError('whole-record likelihood requires all complete support components')
        return self._compile_selected(noise, within_factor, source_chart, range(len(self.specs)))

    def compile_component(self, noise, within_factor, source_chart, index):
        """One complete support component; callers cache by its local scope."""
        if index not in self.compiled_component_indices:
            raise ValueError('this numeric compiler did not prepare the requested complete component')
        return self._compile_selected(noise, within_factor, source_chart, (index,))[0]

    def _compile_selected(self, noise, within_factor, source_chart, indices):
        if (noise.geometry != self.geometry or noise.order != 1 or noise.conditioning_samples != self.conditioning_samples
                or source_chart.mean.shape != (self.p, self.c)):
            raise ValueError('native joint components consume their current complete source/noise coordinates')
        within = finite_real_array(within_factor, name='complete current within factor')
        if within.ndim != 2 or within.shape[0] != self.p*self.c:
            raise ValueError('within factor keeps every native physical coefficient')
        result = []
        for index in indices:
            spec, time_root = self.specs[index], self.tail_roots[index]
            width = len(spec.selection_ids)*self.p*self.c+2*self.c+1
            physical_root = np.empty((0, width))
            if any(start < min(self.n, self.conditioning_samples) for start, _ in spec.native_intervals):
                physical_root = _augmented_qr(self._prefix_physical(noise, spec), 2048)
            if len(time_root):
                mapping = self._tail_map(noise, spec)
                tail = (time_root@mapping.reshape(len(mapping), -1)).reshape(-1, width)
                physical_root = _augmented_qr(np.vstack((physical_root, tail)), max(1, len(physical_root)+len(tail)))
            local_width, rank = len(spec.selection_ids)*within.shape[1], source_chart.rank
            augmented = np.zeros((len(physical_root), local_width+rank+2*self.c+1))
            for i, _ in enumerate(spec.selection_ids):
                response = physical_root[:, i*self.p*self.c:(i+1)*self.p*self.c]
                augmented[:, i*within.shape[1]:(i+1)*within.shape[1]] = response@within
                augmented[:, local_width:local_width+rank] += response@source_chart.factor.reshape(self.p*self.c, rank)
                augmented[:, -1] -= response@source_chart.mean.ravel()
            augmented[:, local_width+rank:local_width+rank+2*self.c] = physical_root[:, -2*self.c-1:-1]
            augmented[:, -1] += physical_root[:, -1]
            logwhite = sum(noise.whitening_logabsdet(stop)-(noise.whitening_logabsdet(start) if start else 0.)
                for start, stop in spec.native_intervals)
            binding = {'record_id': self.record_id, 'original_observation_sha256': self.observation_identity,
                'original_native_samples': self.n, 'native_intervals': spec.native_intervals,
                'original_noise_carrier': noise._runtime_state_digest(), 'shared_response_chart': source_chart.identity,
                'shared_coordinate_ids': source_chart.coordinate_ids, 'drift_group_id': self.drift_group_id,
                'drift_coordinate_ids': self.drift_ids,
                'local_response_laws': {name: array_digest(within) for name in spec.selection_ids},
                'branch_binding': self.branch_binding, 'native_compiler': self.identity,
                'physical_data_root': array_digest(physical_root), 'producer': IMPLEMENTATION_SHA256}
            potential, conditional = integrate_local_augmented_root_with_conditional(augmented,
                local_dimensions=tuple((name, within.shape[1]) for name in spec.selection_ids),
                shared_dimension=rank, drift_dimension=2*self.c,
                observed_scalar_rows=sum(stop-start for start, stop in spec.native_intervals)*self.c,
                log_noise_jacobian=logwhite, binding=binding)
            result.append(NativeComponentReduction(self, spec, potential, conditional, frozen_array(physical_root),
                source_chart, frozen_array(within)))
        return tuple(result)


@dataclass(frozen=True)
class NativeComponentReduction:
    compiler: NativeJointRecordCompiler
    spec: NativeComponentSpec
    potential: object
    conditional: object
    physical_data_root: np.ndarray
    source: object
    within_factor: np.ndarray

    @property
    def drift_ids(self):
        return self.compiler.drift_ids


def native_joint_workspace(compilers, noise, within_factor, source_chart):
    reductions = tuple(value for compiler in compilers for value in compiler.compile(noise, within_factor, source_chart))
    by_record = {compiler.record_id: tuple(value for value in reductions if value.compiler is compiler) for compiler in compilers}
    records = {compiler.record_id: compiler.n for compiler in compilers}

    def moments(posterior):
        statistics, local_second, local_width = [], 0., 0
        for compiler in compilers:
            value, second, width = record_posterior_statistics(compiler, by_record[compiler.record_id], posterior)
            statistics.append(value)
            local_second += second
            local_width += width
        return tuple(statistics), local_second, local_width

    def innovation_moments(posterior):
        statistics, local_second, local_width = [], 0., 0
        phi = np.array([value.coefficients[0] for value in noise.temporal])
        for compiler in compilers:
            value, second, width = record_posterior_innovation_statistics(
                compiler, by_record[compiler.record_id], posterior, phi)
            statistics.append(value)
            local_second += second
            local_width += width
        return tuple(statistics), local_second, local_width

    return JointCalibrationWorkspace(tuple(value.potential for value in reductions), records, moments,
        metadata_digest({'compilers': [value.identity for value in compilers],
            'noise': noise._runtime_state_digest(), 'within': array_digest(within_factor), 'source': source_chart.identity}),
        innovation_moments), reductions


def _record_posterior_packs(compiler, reductions, posterior):
    """One complete shared/flat posterior and each local conditional factor."""
    packs, local_second, local_dimension = [], 0., 0
    source, within = reductions[0].source, reductions[0].within_factor
    mean_u, factor_u = posterior.mean[:source.rank], posterior.factor[:source.rank]
    for reduction in reductions:
        conditional = reduction.conditional
        indices = posterior.component_indices(compiler.drift_ids)
        mg, fg = posterior.mean[list(indices)], posterior.factor[list(indices)]
        inverse = conditional.independent_factor
        regression = solve_triangular(conditional.upper, conditional.cross)
        mv = conditional.mean(mg)
        fv = -regression@fg
        local_second += float(mv@mv+np.sum(fv*fv)+np.sum(inverse*inverse))
        local_dimension += len(mv)
        coefficients = []
        for i, _ in enumerate(reduction.spec.selection_ids):
            start, stop = i*within.shape[1], (i+1)*within.shape[1]
            mean = source.mean.ravel()+source.factor.reshape(compiler.p*compiler.c, source.rank)@mean_u+within@mv[start:stop]
            global_factor = source.factor.reshape(compiler.p*compiler.c, source.rank)@factor_u+within@fv[start:stop]
            local_factor = within@inverse[start:stop]
            coefficients.append((mean.reshape(compiler.p, compiler.c),
                global_factor.reshape(compiler.p, compiler.c, -1), local_factor.reshape(compiler.p, compiler.c, -1)))
        drift_mean, drift_factor = mg[source.rank:], fg[source.rank:]
        packs.append((coefficients, drift_mean, drift_factor, len(mv)))
    return packs, local_second, local_dimension


def record_posterior_statistics(compiler, reductions, posterior):
    """Complete raw E[eeT] diagnostics through exact temporal/boundary pairs."""
    packs, local_second, local_dimension = _record_posterior_packs(compiler, reductions, posterior)
    initial = previous = lag = current = np.zeros((compiler.c, compiler.c))
    previous, lag, current = previous.copy(), lag.copy(), current.copy()
    for spec, time_root, pack in zip(compiler.specs, compiler.moment_roots, packs, strict=True):
        if not len(time_root):
            continue
        maps = [_residual_feature_map(compiler, spec, pack, previous=flag) for flag in (False, True)]
        values = [(time_root@mapping.reshape(len(mapping), -1)).reshape(len(time_root), compiler.c, -1) for mapping in maps]
        current += np.einsum('tci,tdi->cd', values[0], values[0])
        previous += np.einsum('tci,tdi->cd', values[1], values[1])
        lag += np.einsum('tci,tdi->cd', values[0], values[1])
    for position, (start, _, owner) in enumerate(compiler.intervals):
        now = _residual_native_row(compiler, compiler.specs[owner], packs[owner], start)
        if start == 0:
            initial = _row_cross(now, now, True)
        else:
            previous_owner = compiler.intervals[position-1][2]
            before = _residual_native_row(compiler, compiler.specs[previous_owner], packs[previous_owner], start-1)
            current += _row_cross(now, now, True)
            previous += _row_cross(before, before, True)
            lag += _row_cross(now, before, owner == previous_owner)
    return AR1ResidualStatistics(compiler.record_id, compiler.n, initial, previous, lag, current), local_second, local_dimension


def record_posterior_innovation_statistics(compiler, reductions, posterior, phi):
    """Form innovations on full mean/factor columns before second products.

    A local conditional factor is shared within its support component. At a
    boundary between different components, only the global posterior columns
    are shared; the two independent local factors retain separate variances.
    """
    phi = finite_real_array(phi, name='current complete AR1 coefficients for direct innovations')
    if phi.shape != (compiler.c,):
        raise ValueError('direct innovations preserve every original channel')
    packs, local_second, local_dimension = _record_posterior_packs(compiler, reductions, posterior)
    second, cross = np.zeros((compiler.c, compiler.c)), np.zeros((compiler.c, compiler.c))
    for spec, time_root, pack in zip(compiler.specs, compiler.moment_roots, packs, strict=True):
        if not len(time_root):
            continue
        now = _residual_feature_map(compiler, spec, pack, previous=False)
        before = _residual_feature_map(compiler, spec, pack, previous=True)
        delta = now-phi[None, :, None]*before
        innovation = (time_root@delta.reshape(len(delta), -1)).reshape(len(time_root), compiler.c, -1)
        previous = (time_root@before.reshape(len(before), -1)).reshape(len(time_root), compiler.c, -1)
        second += np.einsum('tci,tdi->cd', innovation, innovation)
        cross += np.einsum('tci,tdi->cd', innovation, previous)
    initial = np.zeros((compiler.c, compiler.c))
    for position, (start, _, owner) in enumerate(compiler.intervals):
        now = _residual_native_row(compiler, compiler.specs[owner], packs[owner], start)
        if start == 0:
            initial = _row_cross(now, now, True)
            continue
        previous_owner = compiler.intervals[position-1][2]
        before = _residual_native_row(compiler, compiler.specs[previous_owner], packs[previous_owner], start-1)
        mean = now[0]-phi*before[0]
        global_factor = now[1]-phi[:, None]*before[1]
        second += np.outer(mean, mean)+global_factor@global_factor.T
        cross += np.outer(mean, before[0])+global_factor@before[1].T
        previous_local = phi[:, None]*before[2]
        if owner == previous_owner:
            local_factor = now[2]-previous_local
            second += local_factor@local_factor.T
            cross += local_factor@before[2].T
        else:
            second += now[2]@now[2].T+previous_local@previous_local.T
            cross -= previous_local@before[2].T
    return AR1InnovationStatistics(compiler.record_id, compiler.n, initial, second, cross), local_second, local_dimension


def _residual_feature_map(compiler, spec, pack, *, previous):
    coefficients, drift_mean, drift_factor, local_width = pack
    global_width = drift_factor.shape[1]
    out = np.zeros((compiler._feature_width(spec), compiler.c, 1+global_width+local_width))
    offset = 0
    for mean, global_factor, local_factor in coefficients:
        first = offset+(compiler.p if previous else 0)
        out[first:first+compiler.p, :, 0] = -mean
        out[first:first+compiler.p, :, 1:1+global_width] = global_factor
        out[first:first+compiler.p, :, 1+global_width:] = local_factor
        offset += 2*compiler.p
    time = offset+(2 if previous else 0)
    out[time, :, 0], out[time+1, :, 0] = -drift_mean[:compiler.c], -drift_mean[compiler.c:]
    out[time, :, 1:1+global_width] = drift_factor[:compiler.c]
    out[time+1, :, 1:1+global_width] = drift_factor[compiler.c:]
    observed = offset+4+(compiler.c if previous else 0)
    for channel in range(compiler.c):
        out[observed+channel, channel, 0] = 1.
    return out


def _residual_native_row(compiler, spec, pack, row):
    coefficients, drift_mean, drift_factor, local_width = pack
    mean = compiler.samples[row].copy()-drift_mean[:compiler.c]-row/compiler.geometry.sfreq*drift_mean[compiler.c:]
    global_factor = drift_factor[:compiler.c]+row/compiler.geometry.sfreq*drift_factor[compiler.c:]
    local_factor = np.zeros((compiler.c, local_width))
    for name, (mu, gf, lf) in zip(spec.selection_ids, coefficients, strict=True):
        design, _, _ = compiler.designs[name](row, row+1)
        mean -= design[0]@mu
        global_factor = global_factor+np.einsum('p,pcr->cr', design[0], gf)
        local_factor += np.einsum('p,pcr->cr', design[0], lf)
    return mean, global_factor, local_factor


def _row_cross(left, right, same_local):
    return (np.outer(left[0], right[0])+left[1]@right[1].T
        +(left[2]@right[2].T if same_local else 0.))
