"""Explicit Gaussian representation of a complete transported response model.

The base codec and its original-source representation stay unchanged.  A compact
record names the complete base-model bytes and requires that actual dependency
at read time; standalone bytes embed the complete base owner record once.
Observation permissions, C updates, noise laws and physical clocks stay with
their original callers and are not granted by either codec.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

from research.cached_response_consumer import IMPLEMENTATION_SHA256 as CACHE_SHA256
from research.conditional_response_graph import IMPLEMENTATION_SHA256 as GRAPH_SHA256
from research.conditional_response_graph import rows, string_rows
from research.exact_common_marginal import IMPLEMENTATION_SHA256 as MARGINAL_SHA256
from research.exact_common_marginal import ExactCommonMarginalSource
from research.exact_template_response_consumer import IMPLEMENTATION_SHA256 as TEMPLATE_SHA256
from research.exact_template_response_consumer import ExactTemplateAffineSource
from research.fast_exact_surd import IMPLEMENTATION_SHA256 as ROUNDING_SHA256
from research.fast_marginal_template import IMPLEMENTATION_SHA256 as FAST_TEMPLATE_SHA256
from research.fast_marginal_template import IntegerSurdMarginalTemplate
from research.numeric_response_consumer import IMPLEMENTATION_SHA256 as REFERENCE_SHA256
from transfer.noise_model import metadata_digest, require_sha256
from transfer.transport_graph_model import IMPLEMENTATION_SHA256 as MODEL_SHA256
from transfer.transport_graph_model import FrozenTransportGraphModel

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ORIGINAL_SOURCE = 'original_source_standard_normals/1'
EXACT_COMMON = 'exact_common_marginal_standard_normals/1'
SCHEMA = 'n2p3_frozen_transport_response_programme/1'


def _bytes(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class FrozenTransportResponseProgram:
    model: FrozenTransportGraphModel
    representation: str
    native_spatial_map: tuple | None = None
    observed_channel_ids: tuple = ()

    def __post_init__(self):
        if not isinstance(self.model, FrozenTransportGraphModel):
            raise TypeError('the complete transported model must come from its original owner')
        if self.representation not in {ORIGINAL_SOURCE, EXACT_COMMON}:
            raise ValueError('choose an explicit normalized Gaussian representation')
        if self.native_spatial_map is None:
            if self.observed_channel_ids:
                raise ValueError('channel IDs require an explicit native spatial map')
        else:
            object.__setattr__(self, 'native_spatial_map', rows(self.native_spatial_map))
            object.__setattr__(self, 'observed_channel_ids', tuple(self.observed_channel_ids))
            if (not self.native_spatial_map or len(self.native_spatial_map) != len(self.observed_channel_ids)
                    or len(set(self.observed_channel_ids)) != len(self.observed_channel_ids)
                    or any(len(row) != len(self.model.target_observation_geometry.channels)
                           for row in self.native_spatial_map)):
                raise ValueError('template programme requires its actual native map and complete observed IDs')

    @cached_property
    def affine_source(self):
        reference = self.model.affine_source()
        return reference if self.representation == ORIGINAL_SOURCE else ExactCommonMarginalSource(reference)

    @cached_property
    def compiler(self):
        source = self.affine_source
        if self.native_spatial_map is None:
            return source
        compiler = ExactTemplateAffineSource if self.representation == ORIGINAL_SOURCE else IntegerSurdMarginalTemplate
        return compiler(source, native_spatial_map=self.native_spatial_map,
                        observed_channel_ids=self.observed_channel_ids)

    @cached_property
    def complete_model_sha256(self):
        return hashlib.sha256(self.model.to_bytes()).hexdigest()

    def dependency_record(self):
        source, compiler = self.affine_source, self.compiler
        producers = {'programme': IMPLEMENTATION_SHA256, 'complete_model_codec': MODEL_SHA256,
                     'exact_graph': GRAPH_SHA256, 'reference_functional_compiler': REFERENCE_SHA256}
        if self.representation == EXACT_COMMON:
            producers['exact_common_marginal'] = MARGINAL_SHA256
        if self.native_spatial_map is not None:
            producers.update({'exact_template': TEMPLATE_SHA256, 'query_mean_cache': CACHE_SHA256})
            if self.representation == EXACT_COMMON:
                producers.update({'integer_surd': ROUNDING_SHA256, 'integer_marginal_template': FAST_TEMPLATE_SHA256})
        binding = {'schema': SCHEMA, 'representation': self.representation,
            'complete_model_dependency': {'schema': 'n2p3_frozen_transport_response_graph_model/1',
                                        'content_sha256': self.complete_model_sha256},
            'graph_digest': source.graph.digest(), 'source_chart': source.source_chart,
            'proper_latent_dimensions': source.rank, 'coordinate_ids': list(source.coordinate_ids),
            'source_original_codec_sha256': self.model.source.digest,
            'normalized_standard_gaussian_prior_count': 1,
            'source_null_standard_gaussian_integral': 1,
            'source_null_dimensions_integrated': (source.original_source_rank-source.source_rank
                                                if self.representation == EXACT_COMMON else 0),
            'native_spatial_map': None if self.native_spatial_map is None else string_rows(self.native_spatial_map),
            'observed_channel_ids': list(self.observed_channel_ids),
            'numeric_programme': ('generic_actual_functionals' if self.native_spatial_map is None else
                                 'exact_temporal_templates_with_generic_functional_fallback'),
            'template_header': None if self.native_spatial_map is None else compiler.header,
            'producer_closure': producers, 'raw_or_label_authority_granted': False}
        if self.representation == EXACT_COMMON:
            binding['exact_common_origin_digest'] = metadata_digest(source.origin)
        return binding

    def to_dependency_bytes(self):
        return _bytes(self.dependency_record())

    def to_bytes(self):
        return _bytes({'schema': SCHEMA, 'programme': self.dependency_record(),
                       'complete_model': self.model.record()})

    def save(self, path, *, include_model):
        content = self.to_bytes() if include_model else self.to_dependency_bytes()
        path = Path(path)
        staging = path.with_name(path.name+'.writing')
        staging.write_bytes(content)
        staging.replace(path)
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def from_dependency_bytes(cls, content, *, expected_sha256, model):
        require_sha256(expected_sha256, 'response programme bytes')
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('response programme differs from its external bytes identity')
        record = json.loads(content)
        if record.get('schema') != SCHEMA:
            raise ValueError('unknown response programme schema')
        spatial = (None if record['native_spatial_map'] is None else
                   tuple(tuple(Fraction(v) for v in row) for row in record['native_spatial_map']))
        programme = cls(model, record['representation'], spatial, tuple(record['observed_channel_ids']))
        if programme.complete_model_sha256 != record['complete_model_dependency']['content_sha256']:
            raise ValueError('response programme requires its actual complete model dependency')
        if metadata_digest(programme.dependency_record()) != metadata_digest(record):
            raise ValueError('response programme differs from its complete model, chart or numerical producers')
        return programme

    @classmethod
    def from_bytes(cls, content, *, expected_sha256):
        require_sha256(expected_sha256, 'complete response programme bytes')
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('complete response programme differs from its external bytes identity')
        record = json.loads(content)
        if record.get('schema') != SCHEMA:
            raise ValueError('unknown complete response programme schema')
        dependency = record['programme']
        model = FrozenTransportGraphModel.from_bytes(_bytes(record['complete_model']),
            expected_sha256=dependency['complete_model_dependency']['content_sha256'])
        encoded = _bytes(dependency)
        return cls.from_dependency_bytes(encoded, expected_sha256=hashlib.sha256(encoded).hexdigest(), model=model)

    @classmethod
    def load(cls, path, *, expected_sha256, model=None):
        content = Path(path).read_bytes()
        return (cls.from_bytes(content, expected_sha256=expected_sha256) if model is None else
                cls.from_dependency_bytes(content, expected_sha256=expected_sha256, model=model))
