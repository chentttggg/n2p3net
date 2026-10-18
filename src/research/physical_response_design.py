"""New physical-clock response design over all actual visible exposures.

This numerical plan consumes an original compiled clock chart. It grants no raw
read, chooses no noise law and applies no target/stimulus prior. Negative onset
relative to a later acquisition segment is represented by an exact positive lag.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.exposure_membership import ExposureCandidateMembership
from research.exposure_membership import (
    MissingExposureIdentityError as MissingExposureIdentityError,
)
from research.latent_response_reference import ResponseState
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, metadata_digest


def _ceil(value):
    return -(-value.numerator//value.denominator)


class PhysicalResponsePlan:
    """A declared onset-impulse law on an original rational physical clock.

    `clock_chart` is compiled by the original registered raw/history owner once.
    It contains public coordinates only; no manifest/path/target is consumed.
    A given unknown-exposure assignment is a conditional latent state, not an
    imputation or a prior. The outer factor graph owns any integration law.
    """

    def __init__(self, clock_chart, dictionary, *, clock_program, exposure_response_law, candidate_vocabulary):
        if not isinstance(dictionary, NativeResponseDictionary):
            raise TypeError('physical response uses the unique native rational continuation owner')
        if clock_program not in {'timeline_wall', 'delivery_anchor', 'native_sample_index'}:
            raise ValueError('choose one explicit original physical-clock programme')
        if exposure_response_law != 'onset_impulse_with_actual_duration_provenance/1':
            raise ValueError('an explicit supported exposure-response law is required')
        self.chart, self.dictionary = clock_chart, dictionary
        self.binding = clock_chart.public_binding
        self.history = self.binding['public_physical_history']
        self.exposures = tuple(self.history['exposures'])
        self.clock_program, self.exposure_response_law = clock_program, exposure_response_law
        self._basis_cache = {}
        self._basis_rows = None
        self.vocabulary = tuple(candidate_vocabulary)
        self.membership = ExposureCandidateMembership(self.exposures, self.vocabulary,
            declared_vocabulary=self.history.get('membership_candidate_vocabulary'))
        self.vocabulary = self.membership.vocabulary
        self.identity = metadata_digest({'schema': 'n2p3_physical_response_plan/1',
            'clock_capability_identity': clock_chart.capability_identity_sha256,
            'dictionary': dictionary.digest(), 'clock_program': clock_program,
            'exposure_response_law': exposure_response_law, 'candidate_vocabulary': self.vocabulary,
            'actual_exposure_membership_programme': self.membership.identity,
            'compiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def target_indicator(self, exposure, candidate, unknown_assignments=None):
        return self.membership.target_indicator(exposure['event_id'], candidate, unknown_assignments)

    @staticmethod
    def _shape(state):
        if not isinstance(state, ResponseState):
            raise TypeError('physical design requires the original response state type')
        values = np.array([state.common_amplitude, state.target_amplitude,
            state.latency_seconds, state.width_multiplier])
        if (not np.isfinite(values).all() or np.any(values[:2] <= 0)
                or values[2] < 0 or values[3] <= 0):
            raise ValueError('positive amplitudes/width and causal latency are required')
        return Fraction.from_float(float(values[2])), Fraction.from_float(float(values[3]))

    def response_scope(self, block_id):
        unknown = [event['event_id'] for event in self.exposures
            if event['block_id'] == block_id and event['membership'] is None]
        return {'candidate_variable': self.binding['target_lock_group'],
            'shape_variable': f"{self.binding['record_id']}:block:{block_id}:shape",
            'unknown_exposure_variables': [f'{event}:stimulus_identity' for event in unknown],
            'unknown_exposure_prior_supplied': False}

    def exposure_basis(self, native_rows, exposure_id, state):
        delta, width = self._shape(state)
        rows = tuple(map(int, native_rows))
        if rows != self._basis_rows:
            self._basis_cache.clear()
            self._basis_rows = rows
        key = (rows, exposure_id, delta, width)
        if key in self._basis_cache:
            return self._basis_cache[key]
        exposure = next((event for event in self.exposures if event['event_id'] == exposure_id), None)
        if exposure is None:
            raise ValueError('physical basis requires an actual visible original exposure')
        elapsed = self.chart.elapsed_seconds(rows, exposure_id, clock_program=self.clock_program)
        query = tuple((lag-delta)/width for lag in elapsed)
        evaluated = self.dictionary.evaluate(query, coordinate='seconds', support='half_open')
        result = evaluated, {'event_id': exposure_id, 'physical_exposure_role': exposure['exposure_role'],
            'actual_duration_seconds_exact_ratio': (None if exposure['duration_seconds_exact_ratio'] is None
                else list(exposure['duration_seconds_exact_ratio'])),
            'native_rows': list(rows), 'elapsed_seconds_exact_ratio': [[v.numerator, v.denominator] for v in elapsed],
            'dictionary_evaluation_identity': evaluated.identity_digest,
            'clock_program': self.clock_program, 'response_law': self.exposure_response_law,
            'prior_applied': False, 'new_observation_rows': 0}
        self._basis_cache[key] = result
        return result

    def design(self, block_id, candidate, state, *, native_start, native_stop, unknown_assignments=None):
        self._shape(state)
        if candidate not in self.vocabulary:
            raise ValueError('physical design requires a complete declared current candidate')
        if not self.binding['sample_start'] <= native_start < native_stop <= self.binding['full_stop']:
            raise ValueError('physical design must use actually granted native observations')
        assignments = {} if unknown_assignments is None else unknown_assignments
        rows = tuple(range(native_start, native_stop))
        k = self.dictionary.matrix.shape[1]
        matrix = np.zeros((len(rows), 2*k))
        error, absolute = np.zeros_like(matrix), np.zeros_like(matrix)
        ledgers = []
        selected = [event for event in self.exposures if event['block_id'] == block_id]
        for event in selected:
            included, role = self.target_indicator(event, candidate, assignments)
            basis, ledger = self.exposure_basis(rows, event['event_id'], state)
            common, target = state.common_amplitude*basis.matrix, float(included)*state.target_amplitude*basis.matrix
            matrix[:, :k] += common
            matrix[:, k:] += target
            absolute[:, :k] += abs(common)
            absolute[:, k:] += abs(target)
            error[:, :k] += abs(state.common_amplitude)*basis.error+np.finfo(float).eps*abs(common)
            error[:, k:] += float(included)*abs(state.target_amplitude)*basis.error+np.finfo(float).eps*abs(target)
            ledgers.append({**ledger, 'membership_role': role, 'target_indicator_given_candidate_and_state': bool(included)})
        additions = len(selected)
        gamma = additions*np.finfo(float).eps/(1-additions*np.finfo(float).eps)
        error += gamma*absolute
        return matrix, error, {'schema': 'n2p3_physical_exposure_selection_design/1',
            'plan_identity': self.identity, 'block_id': block_id, 'candidate': candidate,
            'actual_exposure_membership_programme': self.membership.identity,
            'response_scope': self.response_scope(block_id), 'actual_exposures_used': additions,
            'completed_protocol_exposures': sum(e['exposure_role'] == 'completed_protocol_stimulus' for e in selected),
            'interrupted_physical_exposures': sum(e['exposure_role'] == 'interrupted_physical_stimulus' for e in selected),
            'physical_clock_error_role': self.history['clock_error_role'],
            'numerical_error_role': 'interpolation_multiply_sum_reference_estimate_not_complete_nats_certificate',
            'conditional_unknown_assignments': dict(assignments), 'stimulus_prior_applied': False,
            'response_state_parameters': {key: getattr(state, key) for key in
                ('common_amplitude', 'target_amplitude', 'latency_seconds', 'width_multiplier')},
            'design_sha256': array_digest(matrix, error), 'exposures': ledgers}

    def support(self, block_id, states, *, native_start, native_stop):
        """Conservative exact-rational row hulls; no magnitude thresholding."""
        if not self.binding['sample_start'] <= native_start < native_stop <= self.binding['full_stop']:
            raise ValueError('physical support requires actually granted native observation rows')
        rate = Fraction(self.history['native_sample_rate_hz'])
        duration = Fraction(len(self.dictionary.matrix))/Fraction.from_float(self.dictionary.sample_rate_hz)
        hulls = []
        for state in states:
            delta, width = self._shape(state)
            for event in self.exposures:
                if event['block_id'] != block_id:
                    continue
                first_lag = self.chart.elapsed_seconds(native_start, event['event_id'], clock_program=self.clock_program)
                first = max(native_start, native_start+_ceil((delta-first_lag)*rate))
                last = min(native_stop, native_start+_ceil((delta+width*duration-first_lag)*rate))
                if first < last:
                    hulls.append({'event_id': event['event_id'], 'native_half_open_rows': [first, last],
                        'state_latency_seconds': state.latency_seconds, 'state_width': state.width_multiplier})
        return {'scope': self.response_scope(block_id), 'hulls': hulls,
            'raw_native_hull': ([min(h['native_half_open_rows'][0] for h in hulls),
                max(h['native_half_open_rows'][1] for h in hulls)] if hulls else [native_stop, native_stop]),
            'support_proof': 'original_affine_within_segment_physical_clock_exact_rational_half_open',
            'clock_program': self.clock_program, 'source_prior_or_noise_law_changed': False}
