"""Full original native Gaussian response; numeric parameters grant no data."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from research.channel_noise_mechanism import NoiseGeometry
from research.conditional_response_graph import IMPLEMENTATION_SHA256 as GRAPH_SHA256
from research.conditional_response_graph import (
    ExactConditionalResponseGraph,
    exact,
    identity,
    require_graph_producer,
)
from research.exact_template_response_consumer import IMPLEMENTATION_SHA256 as TEMPLATE_SHA256
from research.exact_template_response_consumer import ExactTemplateAffineSource
from research.numeric_response_consumer import IMPLEMENTATION_SHA256 as NUMERIC_SHA256
from research.numeric_response_consumer import AffineResponseSource
from transfer.joint_flat_programme import FrozenJointFlatSource
from transfer.native_response_dictionary import IMPLEMENTATION_SHA256 as DICTIONARY_SHA256
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, metadata_digest, require_sha256
from transfer.source_response_transport import ResponseCoordinateChart

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LEGACY_IMPLEMENTATION_SHA256 = 'f91f95210ff5a308deb4a040a9ac77b3bfad5432c5b347db1ec4e92b017c7cb2'
SCHEMA = 'n2p3_full_original_native_source_response_model/1'


def _bytes(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class NativeSourceResponseModel:
    source: FrozenJointFlatSource
    continuation: NativeResponseDictionary
    target_observation_geometry: NoiseGeometry
    target_participant_key: str
    _producer_sha256: str = field(default=IMPLEMENTATION_SHA256, repr=False, kw_only=True)
    _graph_producer_sha256: str = field(default=GRAPH_SHA256, repr=False, kw_only=True)

    def __post_init__(self):
        if self._producer_sha256 not in {IMPLEMENTATION_SHA256, LEGACY_IMPLEMENTATION_SHA256}:
            raise ValueError('unknown native response model producer')
        require_graph_producer(self._graph_producer_sha256)
        if not isinstance(self.source, FrozenJointFlatSource) or not isinstance(self.continuation, NativeResponseDictionary):
            raise TypeError('native parameters require their original complete source and dictionary owners')
        geometry = self.source.parameters.noise.geometry
        if self.target_observation_geometry.record() != geometry.record():
            raise ValueError('native source consumption requires the same complete native observation contract')
        if (self.continuation.sample_rate_hz != geometry.sfreq
                or array_digest(self.continuation.matrix) != array_digest(self.source.parameters.basis)):
            raise ValueError('native response dictionary changed original source basis or source clock')
        participants = {row['target_contract']['participant_key']
                        for row in self.source.parameters.binding['source_prefixes']}
        if not self.target_participant_key or self.target_participant_key in participants:
            raise ValueError('native response target is included in its original source participants')
        p = 2*self.source.parameters.basis.shape[1]
        if len(self.source.parameters.prior.mean) != p*len(geometry.channels):
            raise ValueError('native source coefficient order differs from common and target-increment columns')

    @cached_property
    def response_graph(self):
        prior, geometry = self.source.parameters.prior, self.source.parameters.noise.geometry
        channels = len(geometry.channels)
        coefficients = 2*self.source.parameters.basis.shape[1]
        binding = {'schema': SCHEMA, 'role': 'full_native_same_geometry_single_saved_source_Gaussian',
            'source_codec_sha256': self.source.digest,
            'source_parameter_authority_sha256': self.source.parameters.digest,
            'source_parameter_binding_sha256': metadata_digest(self.source.parameters.binding),
            'source_original_native_geometry': geometry.record(),
            'target_observation_geometry': self.target_observation_geometry.record(),
            'target_participant_key': self.target_participant_key,
            'function_source_clock_hz': self.continuation.sample_rate_hz,
            'function_dictionary_sha256': self.continuation.digest(),
            'source_basis_sha256': array_digest(self.source.parameters.basis),
            'coefficient_order': 'coefficient_major_channel_minor_common_then_target_increment',
            'coefficient_roles': {'common': [0, coefficients//2], 'target_increment': [coefficients//2, coefficients]},
            'source_prior_preserved_sha256': array_digest(prior.mean, prior.root),
            'completion_dimensions': 0, 'source_fits': 0,
            'native_model_producer_sha256': self._producer_sha256}
        return ExactConditionalResponseGraph(prior.mean, prior.root, identity(len(prior.mean)),
            tuple(exact(v) for v in prior.mean), (), (), ResponseCoordinateChart(identity(channels)),
            coefficients, binding, _producer_sha256=self._graph_producer_sha256)

    def graph(self):
        return self.response_graph

    @cached_property
    def affine_source(self):
        return AffineResponseSource(self.response_graph)

    @cached_property
    def engine(self):
        geometry = self.target_observation_geometry
        return ExactTemplateAffineSource(self.affine_source, native_spatial_map=np.eye(len(geometry.channels)),
                                         observed_channel_ids=geometry.channels)

    def dependency_record(self):
        graph, engine = self.response_graph, self.engine
        return {'schema': SCHEMA,
            'source_dependency': {'codec_sha256': self.source.digest,
                                  'parameter_authority_sha256': self.source.parameters.digest},
            'continuation': {'source_basis_sha256': array_digest(self.source.parameters.basis),
                'sample_rate_hz': self.continuation.sample_rate_hz,
                'terminal_values': self.continuation.terminal_values.tolist(),
                'dictionary_digest': self.continuation.digest(),
                'dictionary_producer_sha256': DICTIONARY_SHA256},
            'target_observation_geometry': self.target_observation_geometry.record(),
            'target_participant_key': self.target_participant_key, 'exact_graph': graph.record(),
            'source_chart': engine.source_chart, 'proper_latent_dimensions': engine.rank,
            'coordinate_ids': list(engine.coordinate_ids), 'template_header': engine.header,
            'normalized_source_prior_count': 1, 'source_directions_removed': 0,
            'producer_closure': {'native_model': self._producer_sha256,
                'actual_functional_compiler': NUMERIC_SHA256, 'temporal_template': TEMPLATE_SHA256},
            'numeric_parameters_only': True, 'raw_C_or_source_admission_granted': False}

    def consumer_record(self):
        return {'native_model_producer_sha256': self._producer_sha256,
                'native_model_consumer_sha256': IMPLEMENTATION_SHA256,
                **self.response_graph.consumer_record()}

    def to_dependency_bytes(self):
        return _bytes(self.dependency_record())

    def to_bytes(self):
        return _bytes({'schema': SCHEMA, 'native_model': self.dependency_record(),
                       'source_original_codec_base64': base64.b64encode(self.source.content).decode()})

    def save(self, path, *, include_source=False):
        content = self.to_bytes() if include_source else self.to_dependency_bytes()
        path = Path(path)
        staging = path.with_name(path.name+'.writing')
        staging.write_bytes(content)
        staging.replace(path)
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def from_dependency_bytes(cls, content, *, expected_sha256, source):
        require_sha256(expected_sha256, 'native response descriptor')
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('native response descriptor changed its external byte identity')
        record = json.loads(content)
        if (record.get('schema') != SCHEMA or not isinstance(source, FrozenJointFlatSource)
                or source.digest != record['source_dependency']['codec_sha256']):
            raise ValueError('native response descriptor requires its original complete numeric source dependency')
        c = record['continuation']
        continuation = NativeResponseDictionary.from_source_matrix(source.parameters.basis,
            sample_rate_hz=c['sample_rate_hz'], source_basis_sha256=c['source_basis_sha256'],
            terminal_values=c['terminal_values'])
        model = cls(source, continuation, NoiseGeometry(**record['target_observation_geometry']),
                    record['target_participant_key'],
                    _producer_sha256=record['producer_closure']['native_model'],
                    _graph_producer_sha256=record['exact_graph']['graph_implementation_sha256'])
        if metadata_digest(model.dependency_record()) != metadata_digest(record):
            raise ValueError('native response descriptor differs from the actual full parameter and compiler relation')
        return model

    @classmethod
    def from_bytes(cls, content, *, expected_sha256):
        require_sha256(expected_sha256, 'complete native response model')
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('complete native response model changed its external byte identity')
        record = json.loads(content)
        if record.get('schema') != SCHEMA:
            raise ValueError('unknown complete native response schema')
        source = FrozenJointFlatSource.from_bytes(base64.b64decode(record['source_original_codec_base64'], validate=True),
            expected_sha256=record['native_model']['source_dependency']['codec_sha256'])
        descriptor = _bytes(record['native_model'])
        return cls.from_dependency_bytes(descriptor, expected_sha256=hashlib.sha256(descriptor).hexdigest(), source=source)

    @classmethod
    def load(cls, path, *, expected_sha256, source=None):
        content = Path(path).read_bytes()
        return (cls.from_bytes(content, expected_sha256=expected_sha256) if source is None else
                cls.from_dependency_bytes(content, expected_sha256=expected_sha256, source=source))
