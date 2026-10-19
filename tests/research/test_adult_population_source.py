"""Exact mathematical controls plus original software-source/EDF integration."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research import adult_population_source as module
from research import conditional_response_graph as graph_owner
from research import population_response_forecast as population
from research import public_population_source as source_owner
from research.adult_programme_input import (
    FrozenAdultProgrammeInputs,
    open_adult_programme_inputs,
)
from research.channel_noise_mechanism import NoiseGeometry
from research.source_likelihood_export import SharedLikelihoodStore
from tests.support.adult_programme import build_complete_requests
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest


def geometry(channels, *, unit='V', rate=1000, reference='nose'):
    return NoiseGeometry(tuple(channels), rate, unit, reference, 'unfiltered', len(channels))


@pytest.fixture
def complete_requests(tmp_path):
    return build_complete_requests(tmp_path)


def numerical_native(channels=('Fz', 'Cz', 'Pz'), *, factor=None, unit='V'):
    basis = np.array([[1.], [2.], [-1.], [0.]])
    dictionary = NativeResponseDictionary.from_source_matrix(basis, sample_rate_hz=1000,
        source_basis_sha256=array_digest(basis), terminal_values=np.array([0.]))
    n = 2*len(channels)
    chart = population.ResponseChart.from_definition(tuple(f'coordinate:{i}' for i in range(n)),
        {'numeric_fixture': True})
    if factor is None:
        factor = np.column_stack((2*np.eye(n), np.eye(n)))
    forecast = SimpleNamespace(chart=chart, mean=np.arange(n, dtype=float), factor=factor,
        between_factor=factor[:, :factor.shape[1]//2], mean_factor=factor[:, factor.shape[1]//2:])
    return SimpleNamespace(dictionary=dictionary, geometry=geometry(channels, unit=unit), forecast=forecast)


def generic_policy(native, target, *, gain=None):
    layout = module.PopulationTargetLayout.between(native.geometry, target)
    p, ns, nr = 2*native.dictionary.matrix.shape[1], layout.source_channels, layout.remaining_channels
    gain = tuple((Q(),)*(p*ns) for _ in range(p*nr)) if gain is None else gain
    return module.FixedPopulationConditional('explicit synthetic generic conditional', layout.identity, p,
        native.forecast.chart.identity, native.dictionary.digest(), ((Q(),),)*(p*nr), gain,
        tuple(tuple(Q(i == j)*(i+2) for j in range(p*nr)) for i in range(p*nr)),
        {'origin': 'synthetic_math_control_not_a_product_variance'})


def unit_query(dimension):
    return np.array(module.identity(dimension), dtype=object)[:, None, :]


def covariance_from_image(image):
    factor = image.values.reshape(-1, image.values.shape[-1])[:, 1:]
    return factor@factor.T


def test_layout_is_actual_full_source_quotient_and_delta_chart_equivalent():
    source = geometry(('Fz', 'Cz', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'))
    target = geometry(('Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO8', 'Oz'), rate=250, reference='left earlobe')
    layout = module.PopulationTargetLayout.between(source, target)
    assert layout.source_channels == 8 and layout.remaining_channels == 3
    assert len(layout.common_target_map) == 4
    assert tuple(target.channels[i] for i in layout.completion_target_rows) == ('Fz', 'PO8', 'Oz')
    # Convert accepted (source, delta, PO8-nose, Oz-nose) to this quotient's
    # remainder (Fz-ear, PO8-ear, Oz-ear), conditional on the same full source.
    bs = tuple(tuple(Q(c == s) if c in source.channels else Q() for s in source.channels)
        for c in target.channels)
    bw = tuple((Q(1), Q(c == 'PO8'), Q(c == 'Oz')) for c in target.channels)
    selector = layout.remaining_selector
    left = module.product(selector, bs)
    right = module.product(selector, bw)
    transformed_source = module._add(layout.source_map, module.product(layout.remaining_map, left))
    assert transformed_source == bs
    assert module.product(layout.remaining_map, right) == bw
    assert module._coordinates(module.rows(right), (Q(1), Q(), Q())) is not None
    assert all(c not in source.channels for c in ('PO8', 'Oz'))


def test_generic_full_source_predicts_beyond_common_and_preserves_families():
    native = numerical_native()
    target = geometry(('Fz', 'Cz', 'PO8'), rate=250, reference='left earlobe')
    layout = module.PopulationTargetLayout.between(native.geometry, target)
    p, r, ns = 2, layout.remaining_channels, layout.source_channels
    gain = tuple(tuple(Q(3) if i == r-1 and j == 2 else Q() for j in range(p*ns)) for i in range(p*r))
    policy = generic_policy(native, target, gain=gain)
    numeric = module.AdultPopulationNumericTransport(native, target, policy, {'synthetic': True})
    query = unit_query(numeric.target_dimension)
    image = numeric.compile_target_functionals(query, query_binding={'independent_control': 'full_target'})
    w = module.kronecker(module.identity(p), layout.target_map)
    bs, bw = tuple(row[:ns] for row in layout.target_map), layout.remaining_map
    effective = module._add(module.kronecker(module.identity(p), bs),
        module.product(module.kronecker(module.identity(p), bw), gain))
    expected_mean = np.array(effective, dtype=float)@native.forecast.mean
    expected_cov = (np.array(effective, dtype=float)@native.forecast.factor)@(
        np.array(effective, dtype=float)@native.forecast.factor).T
    expected_cov += np.array(module.product(module.product(module.kronecker(module.identity(p), bw),
        policy.covariance), module.transpose(module.kronecker(module.identity(p), bw))), dtype=float)
    np.testing.assert_allclose(image.values[:, 0, 0], expected_mean, rtol=0, atol=1e-13)
    np.testing.assert_allclose(covariance_from_image(image), expected_cov, rtol=1e-14, atol=1e-13)
    assert numeric.graph.source_factor.shape == native.forecast.factor.shape
    assert numeric.graph.binding['mean_columns'] == (6, 12)
    assert len(w) == numeric.target_dimension and numeric.graph.chart.common_dimension == ns


def test_common_reference_is_exact_submodel_and_needs_explicit_named_policy(tmp_path):
    native = numerical_native()
    target = geometry(('Fz', 'Cz', 'PO8'), rate=250, reference='left earlobe')
    layout = module.PopulationTargetLayout.between(native.geometry, target)
    k0 = tuple(tuple(Q(i == j)*(i+2)+Q(1, 3) for j in range(6)) for i in range(6))
    m0 = tuple(Q(i, 7) for i in range(6))
    policy = module.FixedPopulationConditional.common_reference(name='declared test reference', layout=layout,
        forecast=native.forecast, dictionary=native.dictionary, mean=m0, covariance=k0,
        declaration={'origin': 'independent mathematical fixture'})
    numeric = module.AdultPopulationNumericTransport(native, target, policy, {'synthetic': True})
    common = module.kronecker(module.identity(2), layout.common_target_map)
    bs = module.kronecker(module.identity(2), layout.source_map)
    s = module.product(common, bs)
    j = module.product(module.product(k0, module.transpose(common)),
        module.inverse(module.product(module.product(common, k0), module.transpose(common))))
    residual = module._minus(k0, module.product(module.product(j, common), k0))
    effective = module.product(j, s)
    mean = module._add(tuple((v,) for v in m0), module.product(j,
        module._minus(module.product(s, tuple((module.exact(v),) for v in native.forecast.mean)),
            module.product(common, tuple((v,) for v in m0)))))
    source_factor = module.product(effective, module.rows(native.forecast.factor))
    covariance = module._add(module.product(source_factor, module.transpose(source_factor)), residual)
    query = module.kronecker(module.identity(2), layout.target_map)
    exact_image = numeric.graph.exact_linear_image(query)
    assert exact_image.mean == tuple(row[0] for row in mean)
    assert exact_image.covariance == covariance
    saved = policy.save(tmp_path/'rational-full-policy.json')
    restored = module.FixedPopulationConditional.from_record(module._read(saved))
    assert restored.record() == policy.record()
    diagonal_reference = tuple(tuple(Q((2, 3, 5, 2, 3, 5)[i]) if i == j else Q()
        for j in range(6)) for i in range(6))
    scalar_policy = module.FixedPopulationConditional.common_reference(name='independent hand calculation',
        layout=layout, forecast=native.forecast, dictionary=native.dictionary,
        mean=(0,)*6, covariance=diagonal_reference, declaration={'origin': 'explicit scalar Schur controls'})
    scalar = module.AdultPopulationNumericTransport(native, target, scalar_policy, {})
    scalar_image = scalar.graph.exact_linear_image(query)
    assert scalar_image.mean == (Q(-2, 5), Q(3, 5), Q(), Q(-2, 5), Q(3, 5), Q())
    block = ((Q(14, 5), Q(-6, 5), Q()), (Q(-6, 5), Q(24, 5), Q()), (Q(), Q(), Q(5)))
    assert scalar_image.covariance == tuple(tuple(block[i % 3][j % 3] if i//3 == j//3 else Q()
        for j in range(6)) for i in range(6))
    with pytest.raises(ValueError, match='named fixed'):
        module.FixedPopulationConditional('', layout.identity, 2, native.forecast.chart.identity,
            native.dictionary.digest(), policy.intercept, policy.gain, policy.covariance, {})
    with pytest.raises(ValueError, match='proper'):
        module.FixedPopulationConditional('zero residual', layout.identity, 2, native.forecast.chart.identity,
            native.dictionary.digest(), policy.intercept, policy.gain,
            tuple((Q(),)*4 for _ in range(4)), {'origin': 'negative'})


def test_units_order_multiclock_exact_queries_and_weak_or_zero_source():
    native = numerical_native()
    target = geometry(('PO8', 'Fz', 'Cz'), unit='uV', rate=250, reference='left earlobe')
    policy = generic_policy(native, target)
    numeric = module.AdultPopulationNumericTransport(native, target, policy, {'synthetic': True})
    # The complete joint physical source/target map carries V->uV, not the
    # source Gaussian payload or a renamed reference string.
    assert any(abs(value) == 10**6 for row in numeric.layout.source_map for value in row)
    lags = (Q(-1, 250), Q(), Q(1, 250), Q(1, 2000), Q(7, 2000))
    points = numeric.compile_time_points(lags, group='target_increment')
    assert points.values.shape[:2] == (5, 3)
    assert not points.values[0].any() and not points.values[2].any()
    assert np.isfinite(points.values).all() and np.isfinite(points.absolute_error).all()
    reordered = geometry(('Fz', 'PO8', 'Cz'), unit='uV', rate=250, reference='left earlobe')
    with pytest.raises(ValueError, match='actual target chart'):
        module.AdultPopulationNumericTransport(native, reordered, policy, {})
    weak = np.zeros((6, 1))
    weak[0, 0] = 2.**-1000
    weak_native = numerical_native(factor=weak)
    weak_policy = generic_policy(weak_native, target)
    weak_numeric = module.AdultPopulationNumericTransport(weak_native, target, weak_policy, {})
    assert weak_numeric.graph.source_factor[0, 0] == 2.**-1000
    weak_image = weak_numeric.compile_target_functionals(unit_query(6), query_binding={'case': 'weak source'})
    assert np.any(weak_image.values[:, :, 1])
    underflow_query = np.empty((1, 1, 6), dtype=object)
    underflow_query.fill(Q())
    underflow_query[0, 0, 2] = Q(1, 2**1000)
    underflow = weak_numeric.compile_target_functionals(underflow_query, query_binding={'case': 'outward underflow error'})
    assert underflow.values[0, 0, 1] == 0 and underflow.absolute_error[0, 0, 1] > 0
    zero_native = numerical_native(factor=np.empty((6, 0)))
    zero_policy = generic_policy(zero_native, target)
    zero_numeric = module.AdultPopulationNumericTransport(zero_native, target, zero_policy, {})
    zero = zero_numeric.compile_target_functionals(unit_query(6), query_binding={'case': 'Dirac source'})
    assert zero.values.shape[-1] == 1+len(zero_policy.covariance)
    assert np.any(covariance_from_image(zero))


def test_zero_shared_physical_contrasts_uses_nonempty_full_source_graph():
    native = numerical_native(('Fz', 'Cz'))
    target = geometry(('PO8', 'Oz'), rate=250, reference='left earlobe')
    layout = module.PopulationTargetLayout.between(native.geometry, target)
    assert layout.common_target_map == () and layout.remaining_channels == 2
    policy = module.FixedPopulationConditional.common_reference(name='all unseen explicit reference', layout=layout,
        forecast=native.forecast, dictionary=native.dictionary, mean=(1, 2, 3, 4),
        covariance=module.identity(4), declaration={'origin': 'synthetic'})
    numeric = module.AdultPopulationNumericTransport(native, target, policy, {})
    assert numeric.graph.chart.common_dimension == 2
    image = numeric.compile_target_functionals(unit_query(4), query_binding={})
    np.testing.assert_array_equal(image.values[:, 0, 0], (1, 2, 3, 4))
    np.testing.assert_allclose(covariance_from_image(image), np.eye(4), rtol=0, atol=0)


def artifact(path):
    payload = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def accepted_software_source13(tmp_path):
    fixture = os.environ.get('N2P3_SOFTWARE_SOURCE13')
    if fixture is None:
        pytest.skip('named historical software source13 artifact is an optional integration input')
    root = Path(fixture)
    acceptance_reference = artifact(root/'COMPLETE_SOFTWARE_TRUST_FIXTURE.json')
    trust = json.loads((root/'COMPLETE_SOFTWARE_TRUST_FIXTURE.json').read_bytes())
    assert trust['not_a_scientific_acceptance'] is True
    execution = source_owner.load_source_population_execution(trust['execution'],
        trusted_acceptance_reference=acceptance_reference, expected_matrix_reference=trust['source_matrix'])
    original, isolated = SharedLikelihoodStore(root/'people'), SharedLikelihoodStore(tmp_path/'software-people')
    for person in execution.record['persons']:
        reference = module.SharedLikelihoodReference(**module.plain_metadata(person['shared_likelihood']))
        copied = isolated.put(original.get(reference))
        assert copied.record() == reference.record()
    return execution, isolated


def test_actual_source13_new_adult_fit_save_cold_and_actual_parent_readmission(
        accepted_software_source13, complete_requests, tmp_path):
    execution, store = accepted_software_source13
    calibration, held, scope = complete_requests
    with open_adult_programme_inputs(calibration, held, scope=scope) as inputs:
        source = module.fit_adult_population_source(execution, inputs, recipe_reference=execution.record['recipes'][0],
            shared_store=store, output_directory=tmp_path/'source')
        assert len(source.native.forecast.likelihoods) == 13
        assert len(source.native.forecast.fold.source_participant_keys) == 13
        assert source.native.forecast.fold.target_identity_keys == (inputs.exclusion_contract['participant_key'],)
        assert inputs.exclusion_contract['participant_key'] not in source.native.forecast.fold.source_participant_keys
        assert source.record['original_person_EEG_refits'] == source.record['target_C_statistics_entering_source_learning'] == 0
        assert source.record['programme_seed'] == execution.record['programme_seed']
        assert source.native.forecast.fold.fold_id not in {row['fold_id'] for row in execution.record['folds']}
        policy = generic_policy(source.native, inputs.geometry)
        policy = replace(policy, covariance=tuple(tuple(value+Q(1, 7) for value in row) for row in policy.covariance),
            gain=tuple(tuple(value+(Q(1, 3) if i == 0 and j == 0 else 0)
                for j, value in enumerate(row)) for i, row in enumerate(policy.gain)))
        reference = policy.save(tmp_path/'explicit-fixed-policy.json')
        live = module.build_adult_population_transport(source, conditional_policy_reference=reference)
        cap = inputs.calibration[0].segments[0].capability
        sample = cap.public_binding['sample_start']
        design = ((Q(1), Q()),)
        image = live.compile_response(design, capability=cap, native_rows=(sample,),
            query_binding={'role': 'synthetic actual clock design'})
        assert image.values.shape[:2] == (1, 7)
        record = live.save(tmp_path/'transport.json')
        numeric = module.FrozenAdultPopulationTransport.load(record, shared_store=store)
        # The new B codec consumes a graph whose original producer is the
        # reviewed preimage, while retaining the actual current consumer.
        old_graph = module.plain_metadata(live.numeric.graph.record())
        old_graph['graph_implementation_sha256'] = '8d485cb3106177b7e5a078eb52bcc2571ec4bc9bc6b06e8a71b20c2dfa33c613'
        old_numeric = module.AdultPopulationNumericTransport(source.native, inputs.geometry, policy,
            live.record['binding'], saved_graph_record=old_graph)
        old_record = module.plain_metadata(live.record)
        old_record['graph'], old_record['numeric_compiler'] = old_graph, module._compiler_record(old_numeric.compiler)
        old_reference = module._save(tmp_path/'reviewed-old-graph-producer.json', old_record)
        old_cold = module.FrozenAdultPopulationTransport.load(old_reference, shared_store=store)
        assert old_cold.numeric.graph.record()['graph_implementation_sha256'] == old_graph['graph_implementation_sha256']
        lineage = old_cold.numeric.consumer_record()['graph_consumer']
        assert lineage['graph_producer_sha256'] == old_graph['graph_implementation_sha256']
        assert lineage['graph_consumer_sha256'] == graph_owner.IMPLEMENTATION_SHA256
        with pytest.raises(TypeError, match='original live'):
            module.require_adult_population_transport(numeric)
        with pytest.raises(TypeError, match='original adult full-source'):
            module.require_adult_population_source(numeric.source)
        with pytest.raises(TypeError, match='independently accepted'):
            source_owner.require_complete_source_population_execution(copy.copy(execution))
        adult_reference = inputs.save(tmp_path/'adult-inputs.json')
        original_values = numeric.numeric.compile_time_points((Q(), Q(1, 2000)), group='target_increment')
    restored_adult = FrozenAdultProgrammeInputs.load(adult_reference['path'], expected_sha256=adult_reference['sha256'])
    with restored_adult.re_admit() as actual:
        restored = numeric.re_admit(source_execution=execution, adult_inputs=actual, shared_store=store)
        cold_values = restored.numeric.compile_time_points((Q(), Q(1, 2000)), group='target_increment')
        np.testing.assert_array_equal(cold_values.values, original_values.values)
        np.testing.assert_array_equal(cold_values.absolute_error, original_values.absolute_error)
        assert restored.identity == live.identity
        assert restored.numeric.consumer_record()['adult_population_consumer_sha256'] == module.IMPLEMENTATION_SHA256
        with pytest.raises(TypeError, match='original admitted'):
            restored.compile_response(design, capability=cap, native_rows=(sample,), query_binding={})
    changed_policy = replace(policy, name='a different fixed declaration').save(tmp_path/'different-policy.json')
    changed = module.plain_metadata(live.record)
    changed['fixed_conditional_policy'] = changed_policy
    altered = module._save(tmp_path/'swapped-policy-model.json', changed)
    with pytest.raises(ValueError, match='metadata relationships'):
        module.FrozenAdultPopulationTransport.load(altered, shared_store=store)
    # A numeric source-only model is never accepted as the original execution.
    with pytest.raises(TypeError, match='independently accepted'):
        module.admit_adult_population_source(source.source_reference, source_execution=numeric.source,
            adult_inputs=inputs, shared_store=store)


@pytest.mark.parametrize('excluded', ('identity', 'EEG', 'record'))
def test_original_population_ingress_rejects_adult_source_contamination(excluded):
    chart = population.ResponseChart.from_definition(('response',), {'synthetic': True})
    dependency = population.SourceRecordDependency('source-record', '1'*64, '2'*64, (0, 10), 2)
    likelihood = population.FullResponseLikelihood('source-person', ('adult-person',) if excluded == 'identity' else (),
        chart, np.ones((1, 1)), np.ones(1), 0., (dependency,), '3'*64, '4'*64)
    fold = population.ParticipantFold('adult-synthetic', ('adult-person',), ('source-person',),
        (('source-person', 'source-record', '1'*64, 2),), ('1'*64,) if excluded == 'EEG' else (),
        ('source-record',) if excluded == 'record' else (), '5'*64)
    with pytest.raises(ValueError, match={'identity': 'aliases', 'EEG': 'EEG bytes', 'record': 'excluded original'}[excluded]):
        population.infer_population((likelihood,), fold=fold,
            reference=population.MeanReference(chart, np.zeros(1), np.eye(1), '6'*64),
            templates=(population.VarianceTemplate('source', chart, np.eye(1)),), log_scales=(0.,))
