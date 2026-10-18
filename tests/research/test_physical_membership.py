from fractions import Fraction

import numpy as np
import pytest

from research.conditional_response_graph import ExactConditionalResponseGraph
from research.exact_template_response_consumer import ExactTemplateAffineSource
from research.latent_response_reference import ResponseState
from research.numeric_response_consumer import AffineResponseSource
from research.physical_response_design import PhysicalResponsePlan
from research.physical_source_templates import PhysicalTemplateResponse
from research.response_frontier import StateKey
from research.scoped_physical_response import ScopedPhysicalResponse
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, metadata_digest
from transfer.source_response_transport import ResponseCoordinateChart


class _Clock:
    def __init__(self, vocabulary):
        exposures = []
        for identifier, sample, members in (('set', 0, vocabulary[:2]),
                                             ('blank', 3, ()), ('last', 6, vocabulary[-1:])):
            exposures.append({'event_id': identifier, 'sample': sample, 'block_id': 1,
                'membership': tuple(q in members for q in vocabulary),
                'candidate_id': None, 'candidate_options': members,
                'exposure_role': 'completed_protocol_stimulus',
                'duration_seconds_exact_ratio': [1, 1]})
        self.public_binding = {'sample_start': 0, 'full_stop': 8,
            'record_id': 'independent_math_record', 'target_lock_group': 'q',
            'public_physical_history': {'native_sample_rate_hz': 1,
                'clock_error_role': 'exact_mathematical_clock',
                'membership_candidate_vocabulary': vocabulary, 'exposures': exposures}}
        self.capability_identity_sha256 = metadata_digest(self.public_binding)

    def elapsed_seconds(self, rows, identifier, *, clock_program):
        assert clock_program == 'timeline_wall'
        events = self.public_binding['public_physical_history']['exposures']
        onset = next(event['sample'] for event in events if event['event_id'] == identifier)
        if isinstance(rows, int):
            return Fraction(rows - onset)
        return tuple(Fraction(row - onset) for row in rows)


def _plan(count):
    vocabulary = tuple(f'q{i}' for i in range(count))
    basis = np.array([[1.], [2.]])
    dictionary = NativeResponseDictionary.from_source_matrix(basis, sample_rate_hz=1.,
        source_basis_sha256=array_digest(basis), terminal_values=np.zeros(1))
    plan = PhysicalResponsePlan(_Clock(vocabulary), dictionary, clock_program='timeline_wall',
        exposure_response_law='onset_impulse_with_actual_duration_provenance/1',
        candidate_vocabulary=vocabulary)
    return plan, ResponseState(1., 1., 0., 1., 1.)


@pytest.mark.parametrize('count', (9, 32, 36))
def test_set_and_blank_keep_one_common_response_and_all_target_members(count):
    plan, state = _plan(count)
    common = [1., 2., 0., 1., 2., 0., 1., 2.]
    for index, candidate in enumerate(plan.vocabulary):
        expected_target = np.zeros(8)
        if index < 2:
            expected_target[:2] = [1., 2.]
        if index == count - 1:
            expected_target[6:] = [1., 2.]
        design, _, _ = plan.design(1, candidate, state, native_start=0, native_stop=8)
        np.testing.assert_array_equal(design, np.stack((common, expected_target), axis=1))
    assert plan.response_scope(1)['unknown_exposure_variables'] == []


def test_cached_membership_difference_retains_real_target_signal():
    plan, state = _plan(9)
    graph = ExactConditionalResponseGraph(np.array([2., 3.]), np.eye(2),
        ((1, 0), (0, 1)), (0, 0), (), (), ResponseCoordinateChart(((1,),)), 2,
        {'role': 'independent_common_and_target_coefficients'})
    source = ExactTemplateAffineSource(AffineResponseSource(graph),
        native_spatial_map=((1,),), observed_channel_ids=('EEG',))
    response = ScopedPhysicalResponse(plan, 1, 'q', 'h', (state,), None)
    callback = PhysicalTemplateResponse(response, source)

    def selected(candidate):
        return {'q': StateKey(candidate, 'candidate'), 'h': StateKey('h0', 0)}

    a, b, c = map(selected, (plan.vocabulary[0], plan.vocabulary[1], plan.vocabulary[-1]))
    first, _, _ = callback.affine(a, native_start=0, native_stop=5)
    same, _, _ = callback.affine(b, native_start=0, native_stop=5)
    np.testing.assert_array_equal(first, same)
    difference, _, binding = callback.affine_difference(a, c, native_start=0, native_stop=5)
    expected = np.zeros((5, 1, 3))
    expected[:2, 0, 0] = [3., 6.]
    expected[:2, 0, 2] = [1., 2.]
    np.testing.assert_array_equal(difference, expected)
    assert not binding['same_physical_functional_error_cancels']
    zero, error, binding = callback.affine_difference(a, b, native_start=0, native_stop=5)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
    np.testing.assert_array_equal(error, np.zeros_like(error))
    assert binding['same_physical_functional_error_cancels']
