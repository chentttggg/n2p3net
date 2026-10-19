"""Complete accepted source people -> one adult population -> exact target law.

The source-only fit reuses original person likelihoods and their noise/scale
programme. Fixed conditional policies are explicit mathematical inputs. Cold
numeric models grant no source, raw, label or live-fit authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from research import population_response_forecast as population
from research import public_population_source as source_owner
from research.adult_programme_input import require_adult_programme_inputs
from research.channel_noise_mechanism import NoiseGeometry
from research.conditional_response_graph import (
    ExactConditionalResponseGraph,
    exact,
    identity,
    inverse,
    kronecker,
    product,
    proper,
    rows,
    string_rows,
    transpose,
)
from research.execution import publish_exclusive
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.numeric_response_consumer import IMPLEMENTATION_SHA256 as AFFINE_CONSUMER_SHA256
from research.numeric_response_consumer import AffineResponseSource
from research.source_likelihood_export import SharedLikelihoodReference
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import metadata_digest
from transfer.source_response_transport import ResponseCoordinateChart

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SOURCES, _TRANSPORTS = WeakKeyDictionary(), WeakKeyDictionary()
SOURCE_SCHEMA = 'adult_complete_person_population_source/1'
TRANSPORT_SCHEMA = 'adult_complete_population_fixed_conditional/1'
POLICY_SCHEMA = 'declared_fixed_population_conditional_policy/1'


def _read(reference):
    payload = Path(reference['path']).read_bytes()
    if len(payload) != reference['bytes'] or hashlib.sha256(payload).hexdigest() != reference['sha256']:
        raise ValueError('the original SHA-bound population dependency changed')
    return json.loads(payload)


def _save(path, record):
    path = Path(path).resolve()
    payload = (json.dumps(plain_metadata(record), sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    publish_exclusive(path, lambda stream: stream.write(payload), byte_size=len(payload))
    return {'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def _add(left, right):
    return tuple(tuple(a+b for a, b in zip(row, other, strict=True))
        for row, other in zip(left, right, strict=True))


def _minus(left, right):
    return _add(left, tuple(tuple(-v for v in row) for row in right))


def _rational_rows(value):
    return tuple(tuple(Q(v) if isinstance(v, str) else exact(v) for v in row) for row in value)


def _compiler_record(compiler):
    return {'source_chart': compiler.source_chart, 'graph': compiler.graph.digest(),
        'conditional_lower': string_rows(compiler.lower), 'conditional_diagonal': tuple(map(str, compiler.diagonal)),
        'coordinate_ids': compiler.coordinate_ids, 'rank': compiler.rank,
        'compiler_sha256': AFFINE_CONSUMER_SHA256}


def _coordinates(basis, requested):
    """Exact coordinates in independent original rows, or None outside span."""
    width = len(basis)
    echelon = []
    for index, row in enumerate(basis):
        work, coefficients = list(row), list(identity(width)[index])
        for pivot, reduced, origin in echelon:
            multiplier = work[pivot]
            work = [a-multiplier*b for a, b in zip(work, reduced, strict=True)]
            coefficients = [a-multiplier*b for a, b in zip(coefficients, origin, strict=True)]
        pivot = next(i for i, value in enumerate(work) if value)
        scale = work[pivot]
        echelon.append((pivot, tuple(v/scale for v in work), tuple(v/scale for v in coefficients)))
    work, coefficients = list(requested), [Q()] * width
    for pivot, reduced, origin in echelon:
        multiplier = work[pivot]
        work = [a-multiplier*b for a, b in zip(work, reduced, strict=True)]
        coefficients = [a+multiplier*b for a, b in zip(coefficients, origin, strict=True)]
    return None if any(work) else tuple(coefficients)


def _left_null(matrix, width):
    """Rows spanning ker(matrix.T), including the empty-column case."""
    work = [list(row) for row in transpose(matrix)]
    pivots, position = [], 0
    for column in range(width):
        selected = next((i for i in range(position, len(work)) if work[i][column]), None)
        if selected is None:
            continue
        work[position], work[selected] = work[selected], work[position]
        divisor = work[position][column]
        work[position] = [v/divisor for v in work[position]]
        for i in range(len(work)):
            if i != position:
                multiplier = work[i][column]
                work[i] = [a-multiplier*b for a, b in zip(work[i], work[position], strict=True)]
        pivots.append(column)
        position += 1
    return tuple(tuple(Q(i == free) if i not in pivots else -work[pivots.index(i)][free]
        for i in range(width)) for free in range(width) if free not in pivots)


def _unit_volts(unit):
    units = {'V': Q(1), 'uV': Q(1, 10**6), 'µV': Q(1, 10**6), 'μV': Q(1, 10**6)}
    if unit not in units:
        raise ValueError('population transport requires a declared voltage unit')
    return units[unit]


@dataclass(frozen=True)
class PopulationTargetLayout:
    """An exact quotient of actual source/target electrode-reference rows.

    The first coordinates are every measured source response. Remaining
    independent target rows complete their joint span; they are latent adult
    responses, never extra measured EEG channels. Acquisition references are
    distinct actual potentials, even if their labels happen to match.
    """

    source: NoiseGeometry
    target: NoiseGeometry
    target_map: tuple
    completion_target_rows: tuple[int, ...]
    common_target_map: tuple

    @classmethod
    def between(cls, source, target):
        channels = tuple(dict.fromkeys((*source.channels, *target.channels)))
        nodes = (*channels, 'source:acquisition_reference', 'target:acquisition_reference')

        def observed(geometry, reference):
            scale = 1/_unit_volts(geometry.unit)
            return tuple(tuple(scale*(Q(node == channel)-Q(node == reference)) for node in nodes)
                for channel in geometry.channels)

        ms, mt = observed(source, nodes[-2]), observed(target, nodes[-1])
        basis, added = list(ms), []
        for index, row in enumerate(mt):
            if _coordinates(basis, row) is None:
                basis.append(row)
                added.append(index)
        target_map = tuple(_coordinates(basis, row) for row in mt)
        missing = tuple(tuple(row[len(ms):]) for row in target_map)
        common = _left_null(missing, len(mt))
        return cls(source, target, target_map, tuple(added), common)

    @property
    def source_channels(self):
        return len(self.source.channels)

    @property
    def remaining_channels(self):
        return len(self.completion_target_rows)

    @property
    def latent_channels(self):
        return self.source_channels+self.remaining_channels

    @property
    def source_map(self):
        return tuple(row[:self.source_channels] for row in self.target_map)

    @property
    def remaining_map(self):
        return tuple(row[self.source_channels:] for row in self.target_map)

    @property
    def remaining_selector(self):
        units = identity(len(self.target.channels))
        return tuple(units[i] for i in self.completion_target_rows)

    def record(self):
        return {'schema': 'complete_source_actual_target_response_layout/1',
            'source_geometry': self.source.record(), 'target_geometry': self.target.record(),
            'target_map': string_rows(self.target_map), 'completion_target_rows': self.completion_target_rows,
            'completion_names': tuple(self.target.channels[i] for i in self.completion_target_rows),
            'common_target_map': string_rows(self.common_target_map),
            'latent_role': 'all_source_responses_then_independent_target_response_coordinates',
            'coefficient_order': 'coefficient_major_C_order_P_by_C',
            'reference_relation': 'independent_acquisition_reference_potentials'}

    @property
    def identity(self):
        return metadata_digest(self.record())


@dataclass(frozen=True)
class FixedPopulationConditional:
    """Named fixed w|s Gaussian; mathematical provenance is SHA-bound input."""

    name: str
    layout_identity: str
    coefficient_count: int
    source_chart_identity: str
    dictionary_identity: str
    intercept: tuple
    gain: tuple
    covariance: tuple
    declaration: dict

    def __post_init__(self):
        if not self.name.strip() or not self.declaration or type(self.coefficient_count) is not int or self.coefficient_count < 1:
            raise ValueError('a named fixed conditional policy and its scientific declaration are required')
        for name in ('intercept', 'gain', 'covariance'):
            object.__setattr__(self, name, _rational_rows(getattr(self, name)))
        object.__setattr__(self, 'declaration', freeze_metadata(self.declaration))
        proper(self.covariance)

    def validate(self, layout, forecast, dictionary):
        n, r = self.coefficient_count*layout.source_channels, self.coefficient_count*layout.remaining_channels
        if (self.layout_identity != layout.identity or self.source_chart_identity != forecast.chart.identity
                or self.dictionary_identity != dictionary.digest()
                or self.coefficient_count != 2*dictionary.matrix.shape[1]
                or len(self.intercept) != r or any(len(row) != 1 for row in self.intercept)
                or len(self.gain) != r or any(len(row) != n for row in self.gain)
                or len(self.covariance) != r or any(len(row) != r for row in self.covariance)):
            raise ValueError('fixed conditional policy differs from its full source/function/actual target chart')

    def record(self):
        return {'schema': POLICY_SCHEMA, 'name': self.name, 'layout_identity': self.layout_identity,
            'coefficient_count': self.coefficient_count, 'source_chart_identity': self.source_chart_identity,
            'dictionary_identity': self.dictionary_identity, 'intercept': string_rows(self.intercept),
            'gain': string_rows(self.gain), 'covariance': string_rows(self.covariance),
            'declaration': self.declaration, 'noise_optimization_moves_this_policy': False,
            'raw_or_source_or_label_authority_granted': False}

    @classmethod
    def from_record(cls, record):
        if (record['schema'] != POLICY_SCHEMA or record['noise_optimization_moves_this_policy'] is not False
                or record['raw_or_source_or_label_authority_granted'] is not False):
            raise ValueError('fixed conditional policy changed its fixed mathematical role')
        value = cls(*(record[name] for name in ('name', 'layout_identity', 'coefficient_count',
            'source_chart_identity', 'dictionary_identity', 'intercept', 'gain', 'covariance', 'declaration')))
        if metadata_digest(value.record()) != metadata_digest(record):
            raise ValueError('fixed conditional policy differs from its exact operands')
        return value

    def save(self, path):
        return _save(path, self.record())

    @classmethod
    def common_reference(cls, *, name, layout, forecast, dictionary, mean, covariance, declaration):
        """Embed a declared full target reference, including zero common rank."""
        p = 2*dictionary.matrix.shape[1]
        k0, m0 = rows(covariance), tuple((exact(value),) for value in mean)
        target_width = p*len(layout.target.channels)
        if len(k0) != target_width or len(m0) != target_width:
            raise ValueError('common-reference operands need the complete actual target coefficient chart')
        proper(k0)
        bs, bw = kronecker(identity(p), layout.source_map), kronecker(identity(p), layout.remaining_map)
        common = kronecker(identity(p), layout.common_target_map)
        selector = kronecker(identity(p), layout.remaining_selector)
        if common:
            j = product(product(k0, transpose(common)), inverse(product(product(common, k0), transpose(common))))
            s = product(common, bs)
            js = product(j, s)
            residual = _minus(k0, product(product(j, common), k0))
            offset = _minus(m0, product(product(j, common), m0))
        else:
            js = tuple((Q(),)*(p*layout.source_channels) for _ in range(target_width))
            residual, offset = k0, m0
        gain = product(selector, _minus(js, bs))
        intercept = product(selector, offset)
        conditional = product(product(selector, residual), transpose(selector))
        if bw:
            assert _add(bs, product(bw, gain)) == js
        return cls(name, layout.identity, p, forecast.chart.identity, dictionary.digest(),
            intercept, gain, conditional, {**declaration, 'recipe': 'fixed_common_reference_conditional',
                'full_target_reference_mean': string_rows(m0), 'full_target_reference_covariance': string_rows(k0)})


def _source_parents(execution, inputs, recipe_reference, store):
    execution = source_owner.require_complete_source_population_execution(execution)
    inputs = require_adult_programme_inputs(inputs)
    if metadata_digest(recipe_reference) not in {metadata_digest(row) for row in execution.record['recipes']}:
        raise ValueError('adult source programme must retain an original accepted source recipe')
    recipe, matrix = _read(recipe_reference), _read(execution.matrix_reference)
    if recipe['source_matrix'] != plain_metadata(execution.matrix_reference) or recipe['programme_seed'] != execution.record['programme_seed']:
        raise ValueError('adult source programme changed its original matrix or seed')
    dictionary = NativeResponseDictionary.from_record(_read(recipe['native_dictionary']))
    source_geometry = NoiseGeometry(**matrix['geometry'])
    people = tuple(execution.record['persons'])
    references = tuple(SharedLikelihoodReference(**plain_metadata(row['shared_likelihood'])) for row in people)
    # Hash-bound store values are numerical data factors, never personal priors.
    exported = tuple(store.get(reference) for reference in references)
    exclusion = inputs.exclusion_contract
    identity = inputs.identity_table.record_for(exclusion['source_subject_id'])
    excluded_identities = tuple(sorted(set((*identity.origin_subject_keys, *identity.global_person_keys,
        exclusion['participant_key']))))
    source_records = tuple((ref.participant_key, row['record_id'], row['source_sha256'], row['labels'])
        for ref, value in zip(references, exported, strict=True) for row in value.metadata['block_constants'])
    excluded_eeg = tuple(sorted({row['source_sha256'] for row in exclusion['source_record_roster'].values()}))
    excluded_records = tuple(sorted(set((*exclusion['calibration_record_ids'], *exclusion['held_record_ids'],
        *exclusion['record_scopes']))))
    plan_identity = metadata_digest({'accepted_execution': execution.identity, 'adult_inputs': inputs.identity,
        'source_exclusion': exclusion, 'source_people': tuple(row['participant_key'] for row in people)})
    fold = population.ParticipantFold('adult:'+plan_identity, excluded_identities,
        tuple(row['participant_key'] for row in people), source_records, excluded_eeg, excluded_records, plan_identity)
    # Reuse original source validation of aliases, EEG bytes, labels and roster.
    store.put_fold_references(fold, references)
    return execution, inputs, recipe, dictionary, source_geometry, fold, references


@dataclass(frozen=True, eq=False)
class AdultPopulationSource:
    native: object
    source_execution: object
    adult_inputs: object
    source_reference: dict
    record: dict

    @property
    def identity(self):
        return metadata_digest(self.record)


def _source_owners(value):
    native, forecast = value.native, value.native.forecast
    return (value.record, native, value.source_execution, value.adult_inputs, value.source_reference,
        forecast, native.dictionary, native.geometry, native.within_reference, native.noise_reference,
        forecast.mean, forecast.between_factor, forecast.mean_factor)


def fit_adult_population_source(execution, inputs, *, recipe_reference, shared_store, output_directory):
    """New all-person source-only fit; zero raw EEG fits and zero C statistics."""
    execution, inputs, recipe, dictionary, geometry, fold, references = _source_parents(
        execution, inputs, recipe_reference, shared_store)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    scale_record = _read(recipe['scale_policy'])
    scale = population.LogScaleProgram(*(tuple(scale_record[key]) for key in ('prior_mean', 'prior_sd', 'initial')),
        scale_record['maximum_iterations'], scale_record['gradient_tolerance'], recipe['scale_policy']['sha256'])
    mean, templates, policy, noise, within = source_owner.prepare_fold_response_reference(shared_store,
        fold, references, person_publications=tuple(row['publication'] for row in execution.record['persons']),
        dictionary=dictionary, conditioning_samples=recipe['noise_options']['conditioning_samples'],
        scale_program=scale, output_directory=directory/'reference')
    forecast, fold_reference = source_owner.fit_shared_population_fold(shared_store, fold, references,
        reference=mean, templates=templates, scale_program=scale, frozen_policy_reference=policy)
    publication = source_owner.save_shared_forecast(forecast, directory/'forecast', store=shared_store,
        fold_reference=fold_reference)
    record = {'schema': SOURCE_SCHEMA, 'accepted_source_execution': execution.execution_reference,
        'independent_acceptance': execution.acceptance_reference, 'source_matrix': execution.matrix_reference,
        'original_recipe': recipe_reference, 'programme_seed': execution.record['programme_seed'],
        'adult_programme_identity': inputs.identity, 'source_exclusion': inputs.exclusion_contract,
        'person_publications': tuple(row['publication'] for row in execution.record['persons']),
        'forecast': publication, 'source_fold': fold.record(), 'source_geometry': geometry.record(),
        'within_reference': within, 'noise_reference': noise, 'source_reference_policy': policy,
        'source_scale_policy': recipe['scale_policy'], 'native_dictionary': recipe['native_dictionary'],
        'original_person_EEG_refits': 0, 'target_C_statistics_entering_source_learning': 0,
        'target_held_statistics_entering_source_learning': 0,
        'consumer_sha256': IMPLEMENTATION_SHA256, 'raw_or_label_or_live_authority_granted': False}
    source_reference = _save(directory/'adult-source.json', record)
    return admit_adult_population_source(source_reference, source_execution=execution,
        adult_inputs=inputs, shared_store=shared_store)


@dataclass(frozen=True)
class FrozenAdultPopulationSource:
    native: object
    source_reference: dict
    record: dict

    @classmethod
    def load(cls, reference, *, shared_store=None):
        record = _read(reference)
        if (record['schema'] != SOURCE_SCHEMA or record['raw_or_label_or_live_authority_granted'] is not False
                or record['original_person_EEG_refits'] != 0
                or record['target_C_statistics_entering_source_learning'] != 0
                or record['target_held_statistics_entering_source_learning'] != 0):
            raise ValueError('adult source numeric model changed its source-only role')
        dictionary = NativeResponseDictionary.from_record(_read(record['native_dictionary']))
        native = source_owner.load_population_native_response(record['forecast'], dictionary=dictionary,
            geometry=NoiseGeometry(**record['source_geometry']), within_reference_dependency=record['within_reference'],
            noise_reference_dependency=record['noise_reference'], shared_store=shared_store)
        if metadata_digest(native.forecast.fold.record()) != metadata_digest(record['source_fold']):
            raise ValueError('adult numeric population differs from its original full-person fit')
        policy = _read(record['source_reference_policy'])
        recipe = _read(record['original_recipe'])
        scale_record = _read(recipe['scale_policy'])
        scale = population.LogScaleProgram(*(tuple(scale_record[key]) for key in ('prior_mean', 'prior_sd', 'initial')),
            scale_record['maximum_iterations'], scale_record['gradient_tolerance'], recipe['scale_policy']['sha256'])
        fit = json.loads(native.forecast.variance_fit_json)
        bare_reference = {key: value for key, value in native.forecast.reference.record().items() if key != 'policy_sha256'}
        if (native.forecast.reference.policy_sha256 != record['source_reference_policy']['sha256']
                or metadata_digest(policy['reference']) != metadata_digest(bare_reference)
                or metadata_digest(policy['templates']) != metadata_digest(tuple(t.record() for t in native.forecast.templates))
                or policy['origin'] != 'fold_source_only'
                or sorted(policy['source_likelihoods']) != sorted(row.original_likelihood_artifact_sha256
                    for row in native.forecast.likelihoods)
                or not np.array_equal(native.forecast.reference.factor, native.within_reference.factor)
                or np.any(native.forecast.reference.mean != 0)
                or len(native.forecast.templates) != 1
                or native.forecast.templates[0].name != 'source_two_exposure_full_response'
                or not np.array_equal(native.forecast.templates[0].factor, native.within_reference.factor)
                or metadata_digest(policy['source_person_publications']) != metadata_digest(record['person_publications'])
                or policy['reference_factor'] != 'full_two_independent_unit_exposure_common_increment_GLS_factor'
                or policy['factor_divisor'] != 1 or policy['source_posterior_factor_used'] is not False
                or policy['noise_reference'] != record['noise_reference']
                or policy['within_reference'] != record['within_reference']
                or metadata_digest(policy['scale_program']) != metadata_digest(scale.record())
                or metadata_digest(fit['program']) != metadata_digest(scale.record())
                or record['source_scale_policy'] != recipe['scale_policy']
                or fit['kind'] != 'source_only_log_variance_plugin' or fit['success'] is not True):
            raise ValueError('adult forecast changed its original source noise/reference/scale programme')
        return cls(native, freeze_metadata(reference), freeze_metadata(record))


def admit_adult_population_source(reference, *, source_execution, adult_inputs, shared_store):
    cold = FrozenAdultPopulationSource.load(reference, shared_store=shared_store)
    execution, inputs, _, dictionary, geometry, fold, references = _source_parents(source_execution,
        adult_inputs, cold.record['original_recipe'], shared_store)
    expected = {'accepted_source_execution': execution.execution_reference,
        'independent_acceptance': execution.acceptance_reference, 'source_matrix': execution.matrix_reference,
        'adult_programme_identity': inputs.identity, 'source_exclusion': inputs.exclusion_contract,
        'source_fold': fold.record(), 'source_geometry': geometry.record(),
        'programme_seed': execution.record['programme_seed'],
        'native_dictionary': _read(cold.record['original_recipe'])['native_dictionary'],
        'person_publications': tuple(row['publication'] for row in execution.record['persons'])}
    if any(metadata_digest(cold.record[key]) != metadata_digest(value) for key, value in expected.items()):
        raise ValueError('adult source requires its actual complete source and original adult parents')
    if tuple(row.original_likelihood_artifact_sha256 for row in cold.native.forecast.likelihoods) != tuple(
            ref.sha256 for ref in references) or cold.native.dictionary.digest() != dictionary.digest():
        raise ValueError('adult source changed its original person A/r/c or function continuation')
    value = AdultPopulationSource(cold.native, execution, inputs, cold.source_reference, cold.record)
    _SOURCES[value] = _source_owners(value)
    return value


def require_adult_population_source(value):
    original = _SOURCES.get(value) if type(value) is AdultPopulationSource else None
    if original is None or any(a is not b for a, b in zip(original, _source_owners(value), strict=True)):
        raise TypeError('the original adult full-source admission is required')
    source_owner.require_complete_source_population_execution(value.source_execution)
    require_adult_programme_inputs(value.adult_inputs)
    return value


def _build_transport(native, target_geometry, policy, binding, *, saved_graph_record=None):
    layout = PopulationTargetLayout.between(native.geometry, target_geometry)
    policy.validate(layout, native.forecast, native.dictionary)
    p, c, r = policy.coefficient_count, layout.source_channels, layout.remaining_channels
    chart = ResponseCoordinateChart(tuple(row+(Q(),)*r for row in identity(c)))
    reference_mean = tuple(Q() if spatial < c else policy.intercept[coefficient*r+spatial-c][0]
        for coefficient in range(p) for spatial in range(c+r))
    source_factor = native.forecast.factor
    graph = ExactConditionalResponseGraph(native.forecast.mean, source_factor, identity(p*c),
        reference_mean, policy.gain, policy.covariance, chart, p,
        {**binding, 'function_source_clock_hz': native.dictionary.sample_rate_hz,
            'function_dictionary_sha256': native.dictionary.digest(),
            'target_observation_geometry': target_geometry.record(), 'latent_layout': layout.record(),
            'between_columns': [0, native.forecast.between_factor.shape[1]],
            'mean_columns': [native.forecast.between_factor.shape[1], source_factor.shape[1]],
            'mean_latent_role': 'shared_across_adults_in_a_joint_population_model',
            'between_latent_role': 'one_adult_draw_shared_across_all_C_held_and_history'})
    if saved_graph_record is not None:
        graph = graph.verify_saved_record(saved_graph_record)
    return layout, graph, AffineResponseSource(graph)


@dataclass(frozen=True)
class ExactPopulationTemporalBasis:
    matrix: tuple
    binding: dict


class AdultPopulationNumericTransport:
    """Shared full-source affine law consumed by later C/history compilers."""

    def __init__(self, native, target_geometry, policy, binding, *, saved_graph_record=None):
        self.native, self.policy = native, policy
        self.layout, self.graph, self.compiler = _build_transport(native, target_geometry, policy, binding,
            saved_graph_record=saved_graph_record)
        self._original_numeric_identity = self._numeric_identity()

    def _numeric_identity(self):
        if self.compiler.graph is not self.graph:
            raise ValueError('adult functional compiler changed its actual source graph')
        return metadata_digest({'layout': self.layout.record(), 'policy': self.policy.record(),
            'compiler': _compiler_record(self.compiler), 'dictionary': self.native.dictionary.digest()})

    def consumer_record(self):
        return {'adult_population_consumer_sha256': IMPLEMENTATION_SHA256,
            'graph_consumer': self.graph.consumer_record(), 'affine_consumer_sha256': AFFINE_CONSUMER_SHA256}

    @property
    def target_dimension(self):
        return self.policy.coefficient_count*len(self.layout.target.channels)

    def compile_target_functionals(self, functionals, *, query_binding):
        if self._numeric_identity() != self._original_numeric_identity:
            raise ValueError('adult functional compiler changed its original fixed Gaussian law')
        raw = np.asarray(functionals, dtype=object)
        if raw.ndim != 3 or raw.shape[-1] != self.target_dimension:
            raise ValueError('actual functionals require coefficient-major actual target EEG coordinates')
        physical_map = kronecker(identity(self.policy.coefficient_count), self.layout.target_map)
        combined = product(rows(raw.reshape(-1, self.target_dimension)), physical_map)
        query = np.asarray(combined, dtype=object).reshape((*raw.shape[:2], self.graph.dimension))
        return self.compiler.compile(query, query_binding={'actual_target_functional': query_binding,
            'layout_identity': self.layout.identity, 'source_dictionary': self.native.dictionary.digest(),
            'source_coefficient_order': 'coefficient_major_C_order_P_by_C',
            'source_clock_hz': self.native.dictionary.sample_rate_hz, 'target_geometry': self.layout.target.record()})

    def exact_time_basis(self, seconds):
        """Original physical basis rows for exact exposure/state accumulation.

        This contracts the stored source-native function before binary64. The
        original dictionary's exact sample-ratio metadata determines each row;
        its numerical evaluator remains the independent interpolation control.
        """
        seconds = tuple(seconds)
        dictionary = self.native.dictionary
        evaluation = dictionary.evaluate(seconds, coordinate='seconds', support='half_open')
        metadata = evaluation.metadata()
        k = dictionary.matrix.shape[1]
        samples = tuple(Q(*row) for row in metadata['sample_coordinates_exact_ratio'])
        result = []
        for sample in samples:
            if not 0 <= sample < len(dictionary.matrix):
                result.append((Q(),)*k)
                continue
            lower = sample.numerator//sample.denominator
            weight = sample-lower
            next_row = dictionary.matrix[lower+1] if lower+1 < len(dictionary.matrix) else dictionary.terminal_values
            result.append(tuple((1-weight)*exact(a)+weight*exact(b)
                for a, b in zip(dictionary.matrix[lower], next_row, strict=True)))
        return ExactPopulationTemporalBasis(tuple(result), freeze_metadata({
            'physical_seconds': [str(exact(v)) for v in seconds], 'source_native_clock_hz': dictionary.sample_rate_hz,
            'native_dictionary_evaluation': evaluation.identity_digest, 'source_dictionary': dictionary.digest(),
            'support': 'half_open', 'arithmetic': 'exact_source_native_linear_interpolation_before_contraction'}))

    def compile_time_points(self, seconds, *, group):
        """Exact physical response image, preserving the final entry error."""
        if group not in ('common', 'target_increment'):
            raise ValueError('a response query declares common or target increment')
        basis = self.exact_time_basis(seconds)
        k, channels = self.native.dictionary.matrix.shape[1], len(self.layout.target.channels)
        query = np.empty((len(basis.matrix), channels, self.target_dimension), dtype=object)
        query.fill(Q())
        for i, values in enumerate(basis.matrix):
            for channel in range(channels):
                for basis_index, value in enumerate(values):
                    coefficient = basis_index+(k if group == 'target_increment' else 0)
                    query[i, channel, coefficient*channels+channel] = value
        return self.compile_target_functionals(query, query_binding={'exact_physical_basis': basis.binding,
            'coefficient_group': group,
            'exact_interpolation_before_source_and_conditional_contraction': True})


@dataclass(frozen=True, eq=False)
class AdultPopulationTransport:
    source: AdultPopulationSource
    numeric: AdultPopulationNumericTransport
    policy_reference: dict
    record: dict

    @property
    def identity(self):
        return metadata_digest(self.record)

    def save(self, path):
        require_adult_population_transport(self)
        return _save(path, self.record)

    def compile_response(self, temporal_design, *, capability, native_rows, query_binding):
        require_adult_population_transport(self)
        matches = tuple(segment for record in self.source.adult_inputs.calibration+self.source.adult_inputs.held
            for segment in record.segments if segment.capability is capability)
        if len(matches) != 1:
            raise TypeError('adult response query requires an original admitted C or held raw capability')
        cap, clock = matches[0].capability, matches[0].clock
        b = cap.public_binding
        native_rows = tuple(native_rows)
        design = rows(temporal_design)
        p, c = self.numeric.policy.coefficient_count, len(self.numeric.layout.target.channels)
        if (not native_rows or len(design) != len(native_rows) or any(len(row) != p for row in design)
                or any(type(row) is not int or not b['sample_start'] <= row < b['full_stop'] for row in native_rows)
                or tuple(sorted(set(native_rows))) != native_rows):
            raise ValueError('physical response design retains original increasing granted native rows and all coefficients')
        query = np.empty((len(design), c, p*c), dtype=object)
        query.fill(Q())
        for i, row in enumerate(design):
            for channel in range(c):
                for coefficient, value in enumerate(row):
                    query[i, channel, coefficient*c+channel] = value
        return self.numeric.compile_target_functionals(query, query_binding={
            'adult_source_transport_identity': self.identity, 'raw_capability_identity': cap.identity_sha256,
            'original_clock_identity': metadata_digest(clock.public_binding), 'native_rows': native_rows,
            'physical_design': query_binding, 'target_held_labels_used': False})


def build_adult_population_transport(source, *, conditional_policy_reference, saved_record=None):
    source = require_adult_population_source(source)
    policy = FixedPopulationConditional.from_record(_read(conditional_policy_reference))
    binding = {'adult_source': source.source_reference, 'source_execution': source.source_execution.identity,
        'adult_programme_identity': source.adult_inputs.identity, 'fixed_conditional_policy': conditional_policy_reference,
        'consumer_sha256': IMPLEMENTATION_SHA256}
    if saved_record is not None:
        saved_binding = saved_record['binding']
        if any(metadata_digest(saved_binding[key]) != metadata_digest(value)
                for key, value in binding.items() if key != 'consumer_sha256'):
            raise ValueError('saved transport requires the actual original source, policy and adult parents')
        binding = saved_binding
    numeric = AdultPopulationNumericTransport(source.native, source.adult_inputs.geometry, policy, binding,
        saved_graph_record=None if saved_record is None else saved_record['graph'])
    record = freeze_metadata({'schema': TRANSPORT_SCHEMA, 'source': source.source_reference,
        'fixed_conditional_policy': conditional_policy_reference, 'target_geometry': source.adult_inputs.geometry.record(),
        'adult_programme_identity': source.adult_inputs.identity, 'binding': binding,
        'layout': numeric.layout.record(), 'graph': numeric.graph.record(),
        'numeric_compiler': _compiler_record(numeric.compiler), 'raw_or_label_or_source_or_live_fit_authority_granted': False})
    if saved_record is not None and metadata_digest(record) != metadata_digest(saved_record):
        raise ValueError('saved adult transport differs from its original source/policy/target relation')
    value = AdultPopulationTransport(source, numeric, freeze_metadata(conditional_policy_reference), record)
    _TRANSPORTS[value] = (source, numeric, value.policy_reference, record, numeric.graph.digest())
    return value


def require_adult_population_transport(value):
    original = _TRANSPORTS.get(value) if type(value) is AdultPopulationTransport else None
    if original is None or any(a is not b for a, b in zip(original[:4],
            (value.source, value.numeric, value.policy_reference, value.record), strict=True)):
        raise TypeError('the original live adult population transport admission is required')
    require_adult_population_source(value.source)
    if value.numeric.graph.digest() != original[4]:
        raise ValueError('adult response graph changed its original source or fixed conditional law')
    return value


@dataclass(frozen=True)
class FrozenAdultPopulationTransport:
    numeric: AdultPopulationNumericTransport
    source: FrozenAdultPopulationSource
    record: dict

    @classmethod
    def load(cls, reference, *, shared_store=None):
        record = _read(reference)
        if record['schema'] != TRANSPORT_SCHEMA or record['raw_or_label_or_source_or_live_fit_authority_granted'] is not False:
            raise ValueError('adult transport changed its numeric-only cold role')
        source = FrozenAdultPopulationSource.load(record['source'], shared_store=shared_store)
        relation = {'adult_source': record['source'], 'fixed_conditional_policy': record['fixed_conditional_policy'],
            'adult_programme_identity': record['adult_programme_identity'],
            'source_execution': metadata_digest({'execution': source.record['accepted_source_execution'],
                'independent_acceptance': source.record['independent_acceptance'], 'matrix': source.record['source_matrix']})}
        if any(metadata_digest(record['binding'][key]) != metadata_digest(value) for key, value in relation.items()):
            raise ValueError('cold transport changed its original source/policy/adult metadata relationships')
        policy = FixedPopulationConditional.from_record(_read(record['fixed_conditional_policy']))
        numeric = AdultPopulationNumericTransport(source.native, NoiseGeometry(**record['target_geometry']), policy,
            record['binding'], saved_graph_record=record['graph'])
        if (metadata_digest(numeric.graph.record()) != metadata_digest(record['graph'])
                or metadata_digest(numeric.layout.record()) != metadata_digest(record['layout'])
                or metadata_digest(_compiler_record(numeric.compiler)) != metadata_digest(record['numeric_compiler'])):
            raise ValueError('saved adult transport differs from its actual source/conditional/functional relations')
        return cls(numeric, source, freeze_metadata(record))

    def re_admit(self, *, source_execution, adult_inputs, shared_store):
        source = admit_adult_population_source(self.record['source'], source_execution=source_execution,
            adult_inputs=adult_inputs, shared_store=shared_store)
        live = build_adult_population_transport(source, conditional_policy_reference=self.record['fixed_conditional_policy'],
            saved_record=self.record)
        if metadata_digest(live.record) != metadata_digest(self.record):
            raise ValueError('saved adult transport needs the same original source, fixed policy and complete adult parents')
        return live
