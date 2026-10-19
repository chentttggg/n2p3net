"""Self-contained source/geometry response model; no noise or raw authority.

The saved original source codec is decoded with its original owner. Full B09
effect/completion/dictionary states reconstruct the exact graph on cold load.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

from research.channel_noise_mechanism import NoiseGeometry
from research.conditional_response_graph import IMPLEMENTATION_SHA256 as GRAPH_SHA256
from research.conditional_response_graph import require_graph_producer
from research.numeric_response_consumer import IMPLEMENTATION_SHA256 as AFFINE_COMPILER_SHA256
from research.numeric_response_consumer import AffineResponseSource
from transfer.joint_flat_programme import FrozenJointFlatSource
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import metadata_digest, require_sha256
from transfer.source_response_transport import transport_response_prior
from transfer.spatial_signal_prior import (
    ERPPhysicalEffect,
    ERPPhysicalTransport,
    SpatialSignalPrior,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LEGACY_IMPLEMENTATION_SHA256 = "5b56d8adf7f5fbb521db5f6148991b14b60c1f81e27e7ce64833d01f59f61286"


@dataclass(frozen=True)
class FrozenTransportGraphModel:
    source: object
    spatial_transport: object
    completion_prior: object
    continuation: object
    target_observation_geometry: object
    target_participant_key: str
    _producer_sha256: str = field(default=IMPLEMENTATION_SHA256, repr=False, kw_only=True)
    _graph_producer_sha256: str = field(default=GRAPH_SHA256, repr=False, kw_only=True)

    def __post_init__(self):
        if self._producer_sha256 not in {IMPLEMENTATION_SHA256, LEGACY_IMPLEMENTATION_SHA256}:
            raise ValueError("unknown transported response model producer")
        require_graph_producer(self._graph_producer_sha256)

    @property
    def factory_arguments(self):
        return {"spatial_transport": self.spatial_transport, "completion_prior": self.completion_prior,
            "continuation": self.continuation, "target_observation_geometry": self.target_observation_geometry,
            "target_participant_key": self.target_participant_key}

    def graph(self):
        return replace(transport_response_prior(self.source, **self.factory_arguments),
                       _producer_sha256=self._graph_producer_sha256)

    def consumer_record(self):
        return {"model_producer_sha256": self._producer_sha256,
                "model_consumer_sha256": IMPLEMENTATION_SHA256, **self.graph().consumer_record()}

    def affine_source(self):
        graph = self.graph()
        spatial = tuple(tuple(Fraction(v) for v in row) for row in
                        graph.binding["exact_graph_generation"]["conditional_spatial_covariance"])
        return AffineResponseSource(graph, completion_factorization=(self.completion_prior.temporal_covariance, spatial))

    def record(self):
        graph = self.graph()
        return {"schema": "n2p3_frozen_transport_response_graph_model/1",
            "source_original_codec_base64": base64.b64encode(self.source.content).decode("ascii"),
            "source_original_codec_sha256": self.source.digest,
            "source_effect": self.spatial_transport.source_effect.record(),
            "destination_effect": self.spatial_transport.destination_effect.record(),
            "spatial_transport": self.spatial_transport.record(), "completion_prior": self.completion_prior.record(),
            "continuation": self.continuation.record(),
            "target_observation_geometry": self.target_observation_geometry.record(),
            "target_participant_key": self.target_participant_key, "exact_graph": graph.record(),
            "affine_source_chart": self.affine_source().source_chart,
            "response_clock_hz": self.continuation.sample_rate_hz,
            "target_observation_clock_hz": self.target_observation_geometry.sfreq,
            "complete_target_noise_supplied": False, "raw_or_label_authority_granted": False,
            "affine_compiler_sha256": AFFINE_COMPILER_SHA256,
            "model_producer_sha256": self._producer_sha256}

    def to_bytes(self):
        return json.dumps(self.record(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def save(self, path):
        content = self.to_bytes()
        target = Path(path)
        staging = target.with_name(target.name + ".writing")
        staging.write_bytes(content)
        staging.replace(target)
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def from_bytes(cls, content, *, expected_sha256):
        require_sha256(expected_sha256, "frozen transported response model")
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("saved transported model differs from external bytes identity")
        record = json.loads(content)
        if record.get("schema") != "n2p3_frozen_transport_response_graph_model/1":
            raise ValueError("unknown transported response model schema")
        source = FrozenJointFlatSource.from_bytes(base64.b64decode(record["source_original_codec_base64"], validate=True),
            expected_sha256=record["source_original_codec_sha256"])
        source_effect = ERPPhysicalEffect.from_record(record["source_effect"])
        destination_effect = ERPPhysicalEffect.from_record(record["destination_effect"])
        transport = ERPPhysicalTransport.between(source_effect, destination_effect)
        if metadata_digest(transport.record()) != metadata_digest(record["spatial_transport"]):
            raise ValueError("saved transport differs from original source/destination physical effects")
        model = cls(source, transport, SpatialSignalPrior.from_record(record["completion_prior"]),
            NativeResponseDictionary.from_record(record["continuation"]),
            NoiseGeometry(**record["target_observation_geometry"]), record["target_participant_key"],
            _producer_sha256=record["model_producer_sha256"],
            _graph_producer_sha256=record["exact_graph"]["graph_implementation_sha256"])
        if metadata_digest(model.record()) != metadata_digest(record):
            raise ValueError("saved transported model differs from complete original owner relations")
        return model

    @classmethod
    def load(cls, path, *, expected_sha256):
        return cls.from_bytes(Path(path).read_bytes(), expected_sha256=expected_sha256)
