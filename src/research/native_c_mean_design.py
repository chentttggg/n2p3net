"""Exact integer accumulation of original native event response columns."""
from __future__ import annotations

import hashlib
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.conditional_response_graph import exact
from research.exact_template_response_consumer import _integer_columns

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _fraction_rows(numerators, denominators):
    """Lift each distinct exact integer row once, then use the original compiler."""
    cache, rows = {}, []
    for row in numerators:
        key = tuple(row)
        lifted = cache.get(key)
        if lifted is None:
            lifted = tuple(Fraction(int(value), denominator)
                for value, denominator in zip(key, denominators, strict=True))
            cache[key] = lifted
        rows.append(lifted)
    return np.asarray(rows, dtype=object), len(cache)


def build_native_c_design(continuation, events, labels, states, *, start, stop, rate,
                          clock_for_selection):
    """Same native physical rows/basis and exact sums; no source contraction here.

    An integer onset and the native linear clock make each equal response-state
    query repeat across events. Each original evaluated basis/error is lifted
    into common rational column denominators. Integer slice additions perform
    the identical exact sum before returning original Fraction row operands.
    """
    began = time.perf_counter()
    native_rows, width = tuple(range(start, stop)), continuation.matrix.shape[1]
    duration = Fraction(len(continuation.matrix)) / exact(continuation.sample_rate_hz)
    clocks, queries, kernels, programme, ledger = {}, {}, {}, [], []
    query_hits = 0
    for event in events:
        event_id, selection = event['event_id'], event['block_id']
        onset, bit = labels[event_id]
        if onset != event['native_sample'] or bit not in (0, 1):
            raise ValueError('source C bit changed its original exposure/native onset')
        state = states[selection]
        if selection not in clocks:
            clocks[selection] = clock_for_selection(selection)
        clock = clocks[selection]
        first_lag = clock.elapsed_seconds(start, event_id, clock_program='native_sample_index')
        latency, stretch = exact(state.latency_seconds), exact(state.width_multiplier)
        left = max(0, math.ceil((latency - first_lag) * rate))
        right = min(len(native_rows), math.ceil((latency + stretch * duration - first_lag) * rate))
        if left >= right:
            ledger.append({'event_id': event_id, 'original_selection': selection, 'native_onset': onset,
                'authorized_C_bit': int(bit), 'exact_zero_on_requested_rows': True})
            continue
        # These are the original numeric chart's affine native coordinates.
        # Cache scope is this call, after the live owner's original require.
        key = (latency, stretch, first_lag + Fraction(left) / rate, right - left, rate)
        if key not in queries:
            elapsed = clock.elapsed_seconds(native_rows[left:right], event_id, clock_program='native_sample_index')
            query = tuple((lag - latency) / stretch for lag in elapsed)
            queries[key] = continuation.evaluate(query, coordinate='seconds', support='half_open')
        else:
            query_hits += 1
        basis = queries[key]
        common, target = exact(state.common_amplitude), exact(state.target_amplitude)
        kernel_key = key, common, target
        if kernel_key not in kernels:
            values, errors = [], []
            for row, error in zip(basis.matrix, basis.error, strict=True):
                numeric, bound = tuple(exact(v) for v in row), tuple(exact(v) for v in error)
                values.append(tuple(common * v for v in numeric) + tuple(target * v for v in numeric))
                errors.append(tuple(abs(common) * v for v in bound) + tuple(abs(target) * v for v in bound))
            value_n, value_d = _integer_columns(values)
            error_n, error_d = _integer_columns(errors)
            kernels[kernel_key] = value_n, value_d, error_n, error_d
        programme.append((left, right, bit, kernel_key))
        ledger.append({'event_id': event_id, 'original_selection': selection, 'native_onset': onset,
            'authorized_C_bit': int(bit), 'dictionary_image': basis.identity_digest})
    # The generic rational lift also covers non-dyadic declared amplitudes.
    denominators = tuple(math.lcm(*(item[1][j] for item in kernels.values())) for j in range(2 * width))
    error_denominators = tuple(math.lcm(*(item[3][j] for item in kernels.values())) for j in range(2 * width))
    integers, errors = np.zeros((len(native_rows), 2 * width), dtype=object), np.zeros(
        (len(native_rows), 2 * width), dtype=object)
    scaled = {}
    for key, (value_n, value_d, error_n, error_d) in kernels.items():
        values = value_n * np.asarray([d // old for d, old in zip(denominators, value_d, strict=True)], dtype=object)
        error = error_n * np.asarray([d // old for d, old in zip(error_denominators, error_d, strict=True)], dtype=object)
        scaled[key] = values, error
    for left, right, bit, key in programme:
        values, bounds = scaled[key]
        integers[left:right, :width] += values[:, :width]
        errors[left:right, :width] += bounds[:, :width]
        if bit:
            integers[left:right, width:] += values[:, width:]
            errors[left:right, width:] += bounds[:, width:]
    accumulation_seconds = time.perf_counter() - began
    design, distinct_rows = _fraction_rows(integers, denominators)
    error, distinct_error_rows = _fraction_rows(errors, error_denominators)
    return design, error, ledger, {'programme': 'exact_rational_integer_native_C_event_sum/1',
        'events': len(events), 'clock_charts': len(clocks), 'dictionary_queries': len(queries),
        'dictionary_query_hits': query_hits, 'state_kernels': len(kernels),
        'distinct_temporal_rows': distinct_rows, 'distinct_error_rows': distinct_error_rows,
        'integer_accumulation_seconds': accumulation_seconds, 'assembly_seconds': time.perf_counter() - began,
        'original_basis_columns_retained': 2 * width, 'source_factor_projections': 0}
