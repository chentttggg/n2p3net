"""Bounded-support voltage carrier for the exact pointwise discrete frontier.

One actual record, one shared source coefficient point, complete positive state
rosters. No fitter, permission issuer, outer integration or nats certificate.
The complete sum-product kernel is shared across records and candidate tasks.
Input authority and continuous integration have independent owners.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod

import numpy as np

from data.candidate_task import MembershipDecision
from research.channel_noise_mechanism import FittedChannelNoise
from research.joint_flat_sparse import compile_sparse_response_image, whiten_record_prefix
from research.latent_response_reference import ResponseState, selection_design
from research.response_frontier import (
    FrontierBudgetExceeded,
    LogFactor,
    PointFactors,
    Roster,
    StateKey,
    sum_product,
)
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, frozen_array, metadata_digest


def _finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} differs from the declared finite coordinates")
    return frozen_array(array)


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _elimination_order(rosters, factors, current):
    """Deterministic complete-table min-scope heuristic, without state pruning."""
    scopes = [set(factor.scope) for factor in factors]
    remaining, order = set(range(len(rosters)))-{current}, []
    while remaining:
        def key(variable, current_scopes=scopes):
            bucket = [scope for scope in current_scopes if variable in scope]
            union = set().union(*bucket) if bucket else {variable}
            return prod(len(rosters[index].states) for index in union), rosters[variable].selection_id
        variable = min(remaining, key=key)
        bucket = [scope for scope in scopes if variable in scope]
        union = set().union(*bucket) if bucket else {variable}
        scopes = [scope for scope in scopes if variable not in scope]
        if union-{variable}:
            scopes.append(union-{variable})
        order.append(variable)
        remaining.remove(variable)
    return tuple(order)


def combine_record_factors(points):
    """Add independent-record likelihood factors BEFORE one discrete sum/integral.

    Every point uses the same explicitly ordered latent variables and shared
    coefficient chart/point. Startup, drift and Gaussian constants remain local.
    The admitted discrete prior is the product of the explicit variable roster
    masses. Per-variable priors are inserted once by sum_product. Correlated
    priors need a separate prior-marginal owner and are not an input here.
    """
    points = tuple(points)
    if not points:
        raise ValueError("independent-record composition requires actual point likelihood factors")
    first = points[0]
    record_ids = [point.binding["physical_record_segment_id"] for point in points]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("the same physical record/segment cannot contribute its observations twice")
    for point in points[1:]:
        if (point.current_index != first.current_index or point.current_q_values != first.current_q_values
                or len(point.rosters) != len(first.rosters)
                or any(a.selection_id != b.selection_id or a.states != b.states
                    or not np.array_equal(a.log_mass, b.log_mass)
                    for a, b in zip(point.rosters, first.rosters, strict=True))
                or point.binding["source_coefficient_chart_sha256"] != first.binding["source_coefficient_chart_sha256"]
                or point.binding["fixed_source_point_sha256"] != first.binding["fixed_source_point_sha256"]):
            raise ValueError("record factors must share one actual coefficient point and variable-prior chart")
    factors = tuple(factor for point in points for factor in point.factors)
    return PointFactors(first.rosters, factors, first.current_index, first.current_q_values,
        float(sum(point.log_common for point in points)),
        _elimination_order(first.rosters, factors, first.current_index),
        {"scope": "independent_record_likelihood_product_before_shared_latent_sum_and_integral",
         "records": [point.binding for point in points],
         "variable_prior_owner": "one_global_complete_roster_sum_product",
         "cross_record_noise_pairs": 0, "outer_continuous_integration_executed": False,
         "numerical_error_role": "float64_pointwise_reference_not_production_absolute_nats_certificate"})


@dataclass(frozen=True)
class NativeSelection:
    """One physical response scope: a candidate variable and a local shape variable.

    Shared candidate IDs express an unchanged thought digit across blocks.
    Separate shape IDs preserve independently varying block responses. Variable
    rosters carry the actual effective log masses; this object grants no data
    authority and never reads target/C labels.
    """

    selection_id: str
    membership: MembershipDecision
    onsets: np.ndarray
    candidate_variable: str
    shape_variable: str
    shapes: tuple[ResponseState, ...]

    def __post_init__(self):
        if not isinstance(self.membership, MembershipDecision):
            raise TypeError("native selection requires the actual MembershipDecision contract")
        onsets = np.asarray(self.onsets)
        shapes = tuple(self.shapes)
        if (not self.selection_id or onsets.dtype.kind not in "iu" or onsets.ndim != 1
                or len(onsets) != len(self.membership.event_ids) or not len(onsets)
                or onsets[0] < 0 or np.any(onsets[1:] <= onsets[:-1])
                or not self.candidate_variable or not self.shape_variable
                or self.candidate_variable == self.shape_variable
                or not shapes or any(not isinstance(shape, ResponseState) for shape in shapes)):
            raise ValueError("selection requires ordered native clock and its complete authorized states")
        for shape in shapes:
            values = np.array([shape.common_amplitude, shape.target_amplitude,
                shape.latency_seconds, shape.width_multiplier])
            if (not np.isfinite(values).all() or np.any(values[:2] <= 0)
                    or values[2] < 0 or values[3] <= 0):
                raise ValueError("shape requires positive amplitudes/width and causal latency")
        immutable_onsets = np.frombuffer(np.asarray(onsets, dtype=np.int64).tobytes(), dtype=np.int64)
        object.__setattr__(self, "onsets", immutable_onsets)
        object.__setattr__(self, "shapes", shapes)


@dataclass(frozen=True)
class _PointImage:
    scope: tuple[int, ...]
    widths: tuple[int, ...]
    start: int
    values: np.ndarray  # state x scalar rows; conservative structural support
    image_bindings: tuple[dict, ...]
    raw_support: tuple[int, int]

    @property
    def stop(self):
        return self.start + self.values.shape[1]


class StoredNoiseCarrier:
    """Original saved startup, original lattice tail, original spatial root."""

    def __init__(self, noise, *, eta):
        if not isinstance(noise, FittedChannelNoise):
            raise TypeError("structured carrier consumes one original FittedChannelNoise")
        noise.assert_integrity()
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("noise variance multiplier eta must be positive")
        self.noise, self.eta = noise, float(eta)
        self.scale = float(np.sqrt(eta))
        self.channels = len(noise.geometry.channels)
        self.identity = metadata_digest({"noise_record": noise.record(), "eta": float(eta),
            "program": "original_saved_startup_lattice_spatial_pointwise_voltage/1"})

    def white_observation(self, residual):
        raw = np.asarray(residual, dtype=float)
        white, columns = whiten_record_prefix(self.noise, raw[:, :, None])
        return white[:, :, 0].ravel()/self.scale, columns/self.scale

    def white_difference(self, raw, start, stop):
        owned = compile_sparse_response_image(self.noise, raw[:, :, None],
            start_native=start, total_native=stop)
        return owned.image.start, owned.image.values[:, 0]/self.scale, dict(owned.binding)

    def leading_log_jacobian(self, samples):
        return self.noise.whitening_logabsdet(samples)-.5*samples*self.channels*np.log(self.eta)


class StructuredPointwiseFrontier:
    """Exact discrete algebra with streamed native state designs at fixed u/drift.

    Source mean is [2k,C], source factor [2k,C,r]. Variable rosters are global:
    several blocks/records may reference the same candidate variable and each
    local shape has its own variable. A point holds likelihood factors only;
    sum_product supplies each variable prior once after record factors merge.
    """

    def __init__(self, observed, dictionary, source_mean, source_factor, selections,
                 carrier, variables, *, origin_sample, current_variable_id,
                 current_q_values, maximum_intermediate_entries,
                 record_id, scalar_log_constant=0.):
        if not isinstance(dictionary, NativeResponseDictionary) or not isinstance(carrier, StoredNoiseCarrier):
            raise TypeError("actual native dictionary and stored noise carrier are required")
        dictionary.digest()
        y = np.asarray(observed, dtype=float)
        if y.ndim != 2 or y.shape[1] != carrier.channels or not len(y) or not np.isfinite(y).all():
            raise ValueError("original observed record requires finite measured channel samples")
        if type(origin_sample) is not int or origin_sample < 0:
            raise ValueError("physical origin must be the original native sample coordinate")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("the caller must identify the original physical record/segment")
        if dictionary.sample_rate_hz != carrier.noise.geometry.sfreq:
            raise ValueError("dictionary and original noise must retain the same actual native clock")
        k, c = dictionary.matrix.shape[1], carrier.channels
        mean = _finite(source_mean, (2*k, c), "one shared physical source mean")
        factor = np.asarray(source_factor, dtype=float)
        if factor.ndim != 3 or factor.shape[:2] != mean.shape or not np.isfinite(factor).all():
            raise ValueError("one shared source factor must retain its actual arbitrary latent width")
        selections, current_q_values = tuple(selections), tuple(current_q_values)
        if not selections or any(not isinstance(s, NativeSelection) for s in selections):
            raise ValueError("every selection requires its complete public native schedule")
        selection_ids = tuple(s.selection_id for s in selections)
        variables = tuple(variables)
        ids = tuple(variable.selection_id for variable in variables)
        if (len(set(selection_ids)) != len(selection_ids) or not variables
                or any(not isinstance(v, Roster) for v in variables)
                or len(set(ids)) != len(ids) or current_variable_id not in ids):
            raise ValueError("unique physical selection/variable IDs and explicit current variable are required")
        current = ids.index(current_variable_id)
        q_roles, h_roles = set(), set()
        for selection in selections:
            if selection.candidate_variable not in ids or selection.shape_variable not in ids:
                raise ValueError("response scopes must reference the explicitly declared variables")
            q_roles.add(selection.candidate_variable)
            h_roles.add(selection.shape_variable)
            q_states = variables[ids.index(selection.candidate_variable)].states
            h_states = variables[ids.index(selection.shape_variable)].states
            if (len({s.q for s in q_states}) != len(q_states)
                    or not {s.q for s in q_states} <= set(selection.membership.candidate_vocabulary)
                    or {s.shape for s in h_states} != set(range(len(selection.shapes)))
                    or len(h_states) != len(selection.shapes)):
                raise ValueError("response scope must preserve complete candidate and shape axes")
        if q_roles & h_roles or current_variable_id in h_roles:
            raise ValueError("candidate and shape variables retain distinct physical roles")
        current_roster = variables[current]
        if (not current_q_values or len(set(current_q_values)) != len(current_q_values)
                or len({state.q for state in current_roster.states}) != len(current_roster.states)
                or set(current_q_values) != {state.q for state in current_roster.states}
                or any(set(current_q_values) != set(s.membership.candidate_vocabulary)
                    for s in selections if s.candidate_variable == current_variable_id)):
            raise ValueError("current roster must retain every declared candidate")
        if type(maximum_intermediate_entries) is not int or maximum_intermediate_entries < 1:
            raise ValueError("complete discrete table budget must be positive")
        if not np.isfinite(scalar_log_constant):
            raise ValueError("original scalar log constant must be finite")
        self.y, self.dictionary = frozen_array(y), dictionary
        self.mean, self.factor = mean, frozen_array(factor)
        self.selections, self.carrier, self.origin = selections, carrier, origin_sample
        self.current, self.q_values, self.budget = current, current_q_values, maximum_intermediate_entries
        self.record_id = record_id
        self.constant = float(scalar_log_constant)
        self.rosters, self.variable_ids = variables, ids
        self.identity = metadata_digest({"observed_sha256": array_digest(y),
            "dictionary": dictionary.identity_digest, "mean_sha256": array_digest(mean),
            "factor_sha256": array_digest(factor), "noise": carrier.identity, "origin": origin_sample,
            "selections": [{"id": s.selection_id, "membership": s.membership.record(),
                "onsets_sha256": array_digest(s.onsets), "candidate_variable": s.candidate_variable,
                "shape_variable": s.shape_variable, "shape_parameters": [vars(shape) for shape in s.shapes]}
                for s in selections],
            "variables": [{"id": v.selection_id, "states": [vars(state) for state in v.states],
                "log_mass_sha256": array_digest(v.log_mass)} for v in variables],
            "physical_record_segment_id": record_id,
            "current": current_variable_id, "current_q": current_q_values,
            "scalar_log_constant": self.constant})

    def _support(self, selection, stop):
        rate = Fraction.from_float(self.dictionary.sample_rate_hz)
        length = len(self.dictionary.matrix)
        starts, ends = [], []
        for shape in selection.shapes:
            latency = rate*Fraction.from_float(float(shape.latency_seconds))
            width = Fraction.from_float(float(shape.width_multiplier))
            starts.append(_ceil(Fraction(int(selection.onsets[0]))+latency))
            ends.append(_ceil(Fraction(int(selection.onsets[-1]))+latency+width*length))
        start = max(self.origin, min(starts))
        end = min(self.origin+stop, max(ends))
        begin = min(stop, max(0, start-self.origin))
        return begin, max(begin, min(stop, max(0, end-self.origin)))

    def _local_states(self, selection):
        q = self.variable_ids.index(selection.candidate_variable)
        h = self.variable_ids.index(selection.shape_variable)
        scope = tuple(sorted((q, h)))
        widths = tuple(len(self.rosters[index].states) for index in scope)
        if prod(widths) > self.budget:
            raise FrontierBudgetExceeded("complete response scope exceeds its declared table budget")
        states = []
        for local_indices in np.ndindex(widths):
            chosen = dict(zip(scope, local_indices, strict=True))
            states.append(StateKey(self.rosters[q].states[chosen[q]].q,
                self.rosters[h].states[chosen[h]].shape))
        return scope, widths, tuple(states)

    def _response(self, selection, key, beta, start, end):
        if start >= end:
            return np.empty((0, self.carrier.channels)), None
        column = selection.membership.candidate_vocabulary.index(key.q)
        design, ledger = selection_design(self.dictionary, selection.onsets,
            selection.membership.membership[:, column], origin_sample=self.origin+start,
            observed_samples=end-start, state=selection.shapes[key.shape])
        response = design@beta
        if not np.isfinite(response).all():
            raise ValueError("pointwise native response arithmetic is unresolved")
        return response, metadata_digest(ledger)

    def factors(self, u, drift, stop_samples):
        if type(stop_samples) is not int or not 1 <= stop_samples <= len(self.y):
            raise ValueError("cutoff must retain the actual leading record support")
        u = _finite(u, (self.factor.shape[2],), "shared continuous source point")
        drift = _finite(drift, (2, self.carrier.channels), "physical voltage/intercept and V-per-second slope")
        beta = self.mean+np.einsum("kcr,r->kc", self.factor, u)
        times = (self.origin+np.arange(stop_samples))/self.dictionary.sample_rate_hz
        reference = drift[0]+times[:, None]*drift[1]
        supports, reference_values, local_states = [], [], []
        max_design_bytes = 0
        for selection in self.selections:
            scope, widths, states = self._local_states(selection)
            start, end = self._support(selection, stop_samples)
            value, _ = self._response(selection, states[0], beta, start, end)
            if start < end:
                reference[start:end] += value
            supports.append((start, end))
            reference_values.append(value)
            local_states.append((scope, widths, states))
            max_design_bytes = max(max_design_bytes, max(0, end-start)*2*self.dictionary.matrix.shape[1]*8)
        b, observation_error = self.carrier.white_observation(self.y[:stop_samples]-reference)
        images, max_sparse_workspace = [], 0
        for selection, (scope, widths, states), (start, end), base in zip(
                self.selections, local_states, supports, reference_values, strict=True):
            point_values, bindings, white_start = [], [], None
            for key in states:
                raw, design_binding = self._response(selection, key, beta, start, end)
                offset, white, binding = self.carrier.white_difference(raw-base, start, stop_samples)
                if white_start is not None and offset != white_start:
                    raise ValueError("one selection's structural white supports changed across states")
                white_start = offset
                point_values.append(white)
                bindings.append({"design": design_binding, "image": binding})
                max_sparse_workspace = max(max_sparse_workspace, (max(0, end-start)
                    + self.carrier.noise.conditioning_samples+self.carrier.noise.order)
                    *self.carrier.channels*8)
            images.append(_PointImage(scope, widths, white_start, np.stack(point_values), tuple(bindings), (start, end)))
        factors, edges = [], []
        for image in images:
            active = image.values
            values = active@b[image.start:image.stop]-.5*np.einsum("sr,sr->s", active, active)
            factors.append(LogFactor(image.scope, values.reshape(image.widths)))
        for right, image in enumerate(images):
            for left in range(right):
                other = images[left]
                begin, end = max(image.start, other.start), min(image.stop, other.stop)
                if begin >= end:
                    continue
                scope = tuple(sorted(set(other.scope) | set(image.scope)))
                widths = tuple(len(self.rosters[index].states) for index in scope)
                entries = prod(widths)
                if entries > self.budget:
                    raise FrontierBudgetExceeded("complete actual overlap pair exceeds its declared table budget")
                a = other.values[:, begin-other.start:end-other.start]
                c = image.values[:, begin-image.start:end-image.start]
                # Align shared variables, rather than giving a shared q two
                # independent axes or imposing an artificial equality factor.
                pair = np.empty(widths)
                for coordinates in np.ndindex(widths):
                    selected = dict(zip(scope, coordinates, strict=True))
                    left_flat = np.ravel_multi_index(tuple(selected[v] for v in other.scope), other.widths)
                    right_flat = np.ravel_multi_index(tuple(selected[v] for v in image.scope), image.widths)
                    pair[coordinates] = -np.dot(a[left_flat], c[right_flat])
                factors.append(LogFactor(scope, pair))
                edges.append((self.selections[left].selection_id, self.selections[right].selection_id))
        order = _elimination_order(self.rosters, factors, self.current)
        jacobian = self.carrier.leading_log_jacobian(stop_samples)
        log_common = self.constant+jacobian-.5*len(b)*np.log(2*np.pi)-.5*float(b@b)
        binding = {"scope": "single_record_pointwise_complete_discrete_partition",
            "physical_record_segment_id": self.record_id,
            "original_operands_sha256": self.identity, "noise_carrier_sha256": self.carrier.identity,
            "source_coefficient_chart_sha256": metadata_digest({"mean": array_digest(self.mean),
                "factor": array_digest(self.factor)}),
            "fixed_source_point_sha256": array_digest(u), "physical_drift_sha256": array_digest(drift),
            "original_native_support": [self.origin, self.origin+stop_samples],
            "stop_samples": stop_samples, "scalar_rows": len(b), "edges": edges,
            "startup_samples": self.carrier.noise.conditioning_samples,
            "actual_tail_memory": self.carrier.noise.order,
            "log_whitening_jacobian": float(jacobian), "extra_scalar_log_constant": self.constant,
            "global_source_prior_evaluations": 0, "prior_owner": "one_outer_shared_measure",
            "variable_prior_owner": "global_complete_roster_sum_product_once_after_record_merge",
            "outer_continuous_integration_executed": False,
            "numerical_error_role": "float64_pointwise_reference_not_production_absolute_nats_certificate",
            "support_role": "exact_physical_hulls_then_original_causal_startup_and_finite_tail",
            "actual_support_proofs": [{"selection_id": selection.selection_id,
                "raw_native_hull": image.raw_support,
                "white_scalar_hull": [image.start, image.stop],
                "variable_scope": [self.variable_ids[v] for v in image.scope],
                "state_images": image.image_bindings}
                for selection, image in zip(self.selections, images, strict=True)],
            "observation_whitening_column_error_estimate": observation_error.tolist(),
            "storage_components_bytes": {"pointwise_state_images": sum(i.values.nbytes for i in images),
                "observed_record": self.y.nbytes, "shared_source_mean_and_factor": self.mean.nbytes+self.factor.nbytes,
                "reference_prefix": reference.nbytes, "white_residual": b.nbytes,
                "reference_state_values": sum(x.nbytes for x in reference_values),
                "maximum_one_native_design": max_design_bytes,
                "maximum_one_sparse_white_workspace_lower_component": max_sparse_workspace,
                "discrete_input_factor_tables": sum(f.values.nbytes for f in factors)},
            "storage_receipt_role": "named_array_components_not_total_peak_RSS_or_hard_RAM_certificate"}
        return PointFactors(self.rosters, tuple(factors), self.current, self.q_values,
            float(log_common), order, binding)

    def partition(self, u, drift, stop_samples):
        return sum_product(self.factors(u, drift, stop_samples), maximum_intermediate_entries=self.budget)

    def partition_pair(self, u, drift, prefix_samples, full_samples):
        if not 0 < prefix_samples < full_samples <= len(self.y):
            raise ValueError("P and F require ordered original record cutoffs")
        return self.partition(u, drift, prefix_samples), self.partition(u, drift, full_samples)
