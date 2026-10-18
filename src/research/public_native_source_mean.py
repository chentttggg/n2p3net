"""Native source-prior C mean from original bits and original physical clocks."""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np

from research.cached_response_consumer import ExactMeanResponseSource, _Rows
from research.conditional_response_graph import exact
from research.exact_template_response_consumer import _integer_columns, _integer_row
from research.latent_response_reference import ResponseState
from research.native_c_mean_design import IMPLEMENTATION_SHA256 as ASSEMBLY_SOURCE_SHA256
from research.numeric_response_consumer import _round_rational
from transfer.native_source_response_model import NativeSourceResponseModel
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class NativeTemporalMeanSource:
    """Contract actual D with original native mu, never project source F."""

    def __init__(self, model, *, maximum_cache_bytes=64*1024**2):
        if not isinstance(model, NativeSourceResponseModel):
            raise TypeError('native temporal means require the complete native response model')
        self.model, self.graph = model, model.graph()
        self.channels = len(model.target_observation_geometry.channels)
        self.coefficients = self.graph.coefficient_count
        self.original_mean = tuple(tuple(exact(v) for v in row) for row in
                                   model.source.parameters.prior.mean.reshape(self.coefficients, self.channels))
        self.integer_mean, self.denominators = _integer_columns(self.original_mean)
        self.integer_abs_mean, self.abs_denominators = _integer_columns(
            tuple(tuple(abs(v) for v in row) for row in self.original_mean))
        self.cache = _Rows(maximum_cache_bytes)
        self.generic = ExactMeanResponseSource(self.graph, maximum_cache_bytes=maximum_cache_bytes)
        self.graph_identity = self.graph.digest()

    def compile_mean(self, functionals, *, query_binding):
        values, errors, binding = self.generic.compile_mean(functionals, query_binding=query_binding)
        return values, errors, {k: v for k, v in binding.items() if k != 'cache'}

    def compile_mean_temporal(self, temporal, *, query_binding, temporal_error=None):
        design = np.asarray(temporal, dtype=object)
        if design.ndim != 2 or not len(design) or design.shape[1] != self.coefficients:
            raise ValueError('native mean query requires every common and target-increment coefficient')
        error = (np.zeros_like(design) if temporal_error is None else np.asarray(temporal_error, dtype=object))
        if error.shape != design.shape:
            raise ValueError('native mean design/error shapes differ')
        values, bounds = np.empty((len(design), self.channels)), np.empty((len(design), self.channels))
        row_ids = []
        for i, row in enumerate(design):
            key = tuple(exact(v) for v in row)
            entry = self.cache.get(key)
            if entry is None:
                numeric, denominator = _integer_row(key)
                numerators = numeric@self.integer_mean
                value, bound = np.empty(self.channels), np.empty(self.channels)
                for c, numerator in enumerate(numerators):
                    value[c], bound[c] = _round_rational(Fraction(int(numerator), denominator*self.denominators[c]))
                entry = self.cache.put(key, value, bound)
            values[i], bounds[i] = entry[:2]
            row_ids.append(entry[2])
            exact_error = tuple(exact(v) for v in error[i])
            if any(v < 0 for v in exact_error):
                raise ValueError('native temporal design error must be nonnegative')
            if any(exact_error):
                numeric, denominator = _integer_row(exact_error)
                numerators = numeric@self.integer_abs_mean
                for c, numerator in enumerate(numerators):
                    total = exact(bounds[i, c])+Fraction(int(numerator), denominator*self.abs_denominators[c])
                    rounded, rounding = _round_rational(total)
                    upper = rounded+rounding
                    bounds[i, c] = (float(np.nextafter(upper, np.inf)) if exact(upper) < total else upper)
        shape = values.shape+(1,)
        binding = {'schema': 'native_original_source_temporal_mean/1', 'source_graph': self.graph_identity,
            'actual_query_binding': query_binding, 'exact_temporal_rows': metadata_digest(row_ids),
            'source_mean_sha256': array_digest(self.model.source.parameters.prior.mean),
            'source_factor_projections': 0, 'proper_latent_columns_compiled': 0,
            'arithmetic': 'actual_exact_temporal_query_then_original_native_mean_integer_dot_and_correct_round',
            'producer_sha256': IMPLEMENTATION_SHA256}
        return values.reshape(shape), bounds.reshape(shape), binding


class PublicNativeSourceCMean:
    def __init__(self, model, *, states_by_selection):
        self.model, self.compiler = model, NativeTemporalMeanSource(model)
        self.states = MappingProxyType(dict(states_by_selection))
        for state in self.states.values():
            if (not isinstance(state, ResponseState)
                    or not all(math.isfinite(v) for v in (state.common_amplitude, state.target_amplitude,
                        state.latency_seconds, state.width_multiplier))
                    or min(state.common_amplitude, state.target_amplitude, state.width_multiplier) <= 0
                    or state.latency_seconds < 0):
                raise ValueError('source C mean declares positive amplitudes/width and causal latency before C fitting')
        self.identity = metadata_digest({'source_graph': model.graph().digest(),
            'states_by_original_selection': {name: vars(state) for name, state in self.states.items()},
            'clock_program': 'native_sample_index', 'role': 'fixed_original_source_prior_mean_before_C_response_update',
            'producer_sha256': IMPLEMENTATION_SHA256, 'assembly_source_sha256': ASSEMBLY_SOURCE_SHA256})
        self.last_assembly_performance = None

    def evaluate(self, calibration_input, *, native_start=None, native_stop=None):
        from research.public_record_physical_bridge import require_public_physical_input

        cap = require_public_physical_input(calibration_input)
        b = cap.public_binding
        geometry = self.model.target_observation_geometry
        if (b['role'] != 'calibration' or b['participant_key'] != self.model.target_participant_key
                or (tuple(b['eeg_channels']), b['sample_rate_hz'], b['reference'], b['signal_unit'], b['operator']) !=
                    (geometry.channels, geometry.sfreq, geometry.reference, geometry.unit, geometry.operator_identity)):
            raise ValueError('source C mean requires its actual same-participant native calibration input')
        start = b['sample_start'] if native_start is None else native_start
        stop = b['full_stop'] if native_stop is None else native_stop
        if type(start) is not int or type(stop) is not int or not b['sample_start'] <= start < stop <= b['full_stop']:
            raise ValueError('C mean rows must remain inside their original granted native support')
        events = b['public_physical_history']['exposures']
        labels = {event_id: (onset, bit) for event_id, onset, bit in cap.label_rows}
        if len(labels) != len(cap.label_rows) or set(labels) != {event['event_id'] for event in events}:
            raise ValueError('source C mean requires every original exposure bit exactly once')
        from research.native_c_mean_design import build_native_c_design

        design, errors, ledgers, performance = build_native_c_design(
            self.model.continuation, events, labels, self.states, start=start, stop=stop,
            rate=Fraction(*b['sample_rate_hz_exact_ratio']), clock_for_selection=cap.clock_for_selection)
        self.last_assembly_performance = performance
        values, bounds, compiled = self.compiler.compile_mean_temporal(design, temporal_error=errors,
            query_binding={'C_input_identity': cap.identity, 'fixed_source_mean_programme': self.identity,
                'native_half_open_rows': [start, stop], 'physical_exposure_ledger': ledgers})
        return values[:, :, 0], bounds[:, :, 0], {'source_graph': self.model.graph().digest(),
            'C_input_identity': cap.identity, 'C_fixed_mean_programme': self.identity,
            'source_functional_mean_compiler': compiled, 'native_half_open_rows': [start, stop],
            'clock_program': 'native_sample_index', 'source_function_clock_hz': self.model.continuation.sample_rate_hz,
            'used_label_event_ids': tuple(row['event_id'] for row in ledgers),
            'source_response_C_updates_before_noise_fit': 0, 'source_factor_projections': 0,
            'mean_image_sha256': array_digest(values[:, :, 0], bounds[:, :, 0]), 'future_truth_consumed': False,
            'mean_assembly_producer_sha256': ASSEMBLY_SOURCE_SHA256, 'mean_owner_producer_sha256': IMPLEMENTATION_SHA256}

    def __call__(self, calibration_input):
        return self.evaluate(calibration_input)
