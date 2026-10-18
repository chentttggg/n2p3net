from fractions import Fraction

import numpy as np
import pytest
from scipy.special import logsumexp

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.collapsed_continuous_inference import FlatEndpointUnidentifiedError
from research.finite_quadratic_integration import integrate_complete_finite, replay_finite_model
from research.physical_affine_frontier import (
    GroupedPhysicalTarget,
    PhysicalAffineSegment,
    PhysicalDriftGroup,
    PreparedPhysicalSegment,
    file_sha,
    replay_physical_parent,
)
from research.response_frontier import Roster, StateKey
from research.structured_response_frontier import StoredNoiseCarrier


class _ExactClock:
    def __init__(self, times):
        self.times = times
        self.capability_identity_sha256 = str(times)

    def sample_time_seconds(self, rows, *, clock_program, reference_origin):
        return tuple(Fraction(self.times[row]) - reference_origin for row in rows)


class _StepResponse:
    scope, rank, channels, source_chart = ('q',), 1, 1, 'original_standard_normal_u'

    def __init__(self, values):
        self.values, self.identity = values, str(values)

    def support(self, *, native_start, native_stop):
        return {'raw_native_hull': [native_start, native_stop]}

    def affine(self, selected_states, *, native_start, native_stop):
        step = np.array(self.values[native_start:native_stop])
        coefficient = int(selected_states['q'].q) - 5
        image = np.stack((coefficient * step, step), axis=1)[:, None, :]
        return image, np.zeros_like(image), {'candidate_coefficient': coefficient}


def _segment(values, times, step, origin, group, rosters, record_id):
    noise = FittedChannelNoise.from_parameters(np.empty((1, 0)), np.eye(1),
        geometry=NoiseGeometry(('C',), 1., 'V', 'declared', 'mathematical_I', 1),
        conditioning_samples=2, reference_id='declared_AR0', family='private')
    return PhysicalAffineSegment(np.array(values)[:, None], StoredNoiseCarrier(noise, eta=1.),
        _ExactClock(times), (_StepResponse(step),), rosters, group, record_id=record_id,
        origin_sample=origin, clock_program='timeline_wall', current_variable='q',
        q_values=tuple(str(q) for q in range(1, 10)), maximum_intermediate_entries=9,
        original_input_identity=str((record_id, values)), noise_segment_declaration={
            'law': 'independent_noise_segments_original_stationary_startups/1',
            'noise_segment_id': record_id, 'role': 'declared_mathematical_law'})


def _compile(originals, cuts):
    return [(PreparedPhysicalSegment(model, (p, f), block_rows=1,
        maximum_cache_bytes=4_000_000), p, f)
        for model, (p, f) in zip(originals, cuts, strict=True)]


def _rosters():
    mass = np.r_[np.zeros(8), -900.]
    mass -= logsumexp(mass)
    return (Roster('q', tuple(StateKey(str(q), 'candidate') for q in range(1, 10)), mass),)


def _group(name='physical_record'):
    return PhysicalDriftGroup(name, ('C',), Fraction(0), 'timeline_wall', 'exact_clock')


def test_shared_physical_drift_absolute_integrals_and_canonical_cold_parent(tmp_path):
    rosters, group = _rosters(), _group()
    times, step = (0, 1, 10, 11), (0, 0, 1, 1)
    originals = (_segment([0., 0.], times, step, 0, group, rosters, 'noise0'),
        _segment([1., 1.], times, step, 2, group, rosters, 'noise1'))
    records = _compile(originals, ((2, 2), (0, 2)))
    target = GroupedPhysicalTarget(records)
    result = integrate_complete_finite(target, maximum_branches=9)
    # Direct four-observation Gaussian integral, including physical da volume.
    coefficient = np.arange(1, 10) - 5
    expected_f = rosters[0].log_mass - np.log(2 * np.pi * np.sqrt(408.)) - (1 - coefficient)**2 / 204.
    np.testing.assert_allclose(result['integrated_joint'], np.stack((rosters[0].log_mass, expected_f)),
        rtol=0., atol=2e-12)
    assert target.endpoints == ('P', 'F') and target.dimension == 3
    assert result['drift_group_measure_counts'] == {group.group_id: 1}
    assert result['original_normalized_source_prior_count'] == 1
    assert np.isfinite(result['probabilities']['whole_log_probability']).all()

    independent_originals = (originals[0],
        _segment([1., 1.], times, step, 2, _group('independent_second'), rosters, 'noise1'))
    independent = GroupedPhysicalTarget(_compile(independent_originals, ((2, 2), (0, 2))))
    independent_result = integrate_complete_finite(independent, maximum_branches=9)
    assert independent.endpoints == ('F',)
    assert np.max(abs(np.array(result['probabilities']['whole_log_probability'])
        - np.array(independent_result['probabilities']['whole_log_probability']))) > 1e-4

    cold_records = []
    for index, (model, p, f) in enumerate(records):
        path = tmp_path / f'model_{index}.npz'
        model.save(path)
        metadata = path.with_suffix('.json')
        cold_records.append((PreparedPhysicalSegment.load(metadata,
            expected_metadata_sha256=file_sha(metadata)), p, f))
    cold = GroupedPhysicalTarget(cold_records)
    assert replay_finite_model(cold, result)['bitexact']
    assert replay_physical_parent(originals, cold, result)['bitexact']


def test_shared_drift_uses_exact_joint_time_rank_and_unique_noise_segments():
    rosters, group = _rosters(), _group()
    originals = (_segment([0.], (0, 10), (0, 0), 0, group, rosters, 'one0'),
        _segment([0.], (0, 10), (0, 0), 1, group, rosters, 'one1'))
    records = _compile(originals, ((1, 1), (0, 1)))
    target = GroupedPhysicalTarget(records)
    result = integrate_complete_finite(target, maximum_branches=9)
    assert target.endpoints == ('F',)
    np.testing.assert_allclose(result['integrated_joint'][0], rosters[0].log_mass - np.log(10.),
        rtol=0., atol=2e-12)

    constant = _segment([0., 0., 0.], (1, 1, 1), (0, 0, 0), 0, group, rosters, 'constant')
    with pytest.raises(FlatEndpointUnidentifiedError, match='exact_time_rank'):
        GroupedPhysicalTarget(_compile((constant,), ((1, 3),)))
    records[1][0].noise_segment_declaration = {**records[1][0].noise_segment_declaration,
        'noise_segment_id': records[0][0].noise_segment_declaration['noise_segment_id']}
    with pytest.raises(ValueError, match='one declared noise segment'):
        GroupedPhysicalTarget(records)
