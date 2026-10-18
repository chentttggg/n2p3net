"""Native public C residuals and complete original-fitter noise consumption."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry, ResidualRecord
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.latent_response_reference import ResponseState
from research.public_native_source_admission import require_public_native_source
from research.public_native_source_mean import PublicNativeSourceCMean
from research.public_record_physical_bridge import require_public_physical_input
from research.structured_response_frontier import StoredNoiseCarrier
from transfer.ensemble_views import calibration_prefix_identity
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RECIPE_NAME = 'native_public_fixed_source_mean_private_AR1_C_plugin/1'
SEGMENT_LAW = 'original_record_stationary_startup_C_history_physical_union/1'
_COMPILED, _ADMITTED = WeakKeyDictionary(), WeakKeyDictionary()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry(binding):
    geometry = NoiseGeometry(tuple(binding['eeg_channels']), binding['sample_rate_hz'],
        binding['signal_unit'], binding['reference'], binding['operator'], len(binding['eeg_channels']))
    if (len(geometry.channels) != 31 or geometry.sfreq != 1000. or geometry.unit != 'V'
            or geometry.reference != 'nose' or geometry.operator_identity != 'native_unfiltered_volts/1'):
        raise ValueError('public native C noise requires actual 31 EEG/nose/V/1000 Hz original coordinates')
    return geometry


@dataclass(frozen=True, eq=False)
class CompiledPublicCResiduals:
    source_admission: object
    prefix: object
    inputs: tuple
    mean: object
    residuals: tuple
    programme: dict


def compile_public_c_residuals(source_admission, prefix, inputs, *, mean):
    admission = require_public_native_source(source_admission)
    prefix = admission.compiler.assert_original_prefix(prefix)
    if not any(prefix is item for item in admission.prefixes):
        raise ValueError('C noise requires its exact native source-admitted original prefix')
    inputs = tuple(require_public_physical_input(value) for value in inputs)
    identity = calibration_prefix_identity(prefix)
    expected = tuple(value for value in admission.calibration_inputs
        if value.public_binding['C_prefix_identity'] == identity)
    if len(inputs) != len(expected) or any(a is not b for a, b in zip(inputs, expected, strict=True)):
        raise ValueError('C noise consumes the complete exact original C record set once')
    if type(mean) is not PublicNativeSourceCMean or mean.model is not admission.model:
        raise TypeError('C residuals require the sole source-bound native fixed-mean owner')
    selections = {event['block_id'] for cap in inputs
        for event in cap.public_binding['public_physical_history']['exposures']}
    if any(mean.states.get(selection) != ResponseState(1., 1., 0., 1., 1.) for selection in selections):
        raise ValueError('this fixed-source noise recipe declares the nominal probability-one response state')
    residuals, bindings = [], []
    for cap in inputs:
        b, geometry = cap.public_binding, _geometry(cap.public_binding)
        values, error, binding = mean(cap)
        if (b['role'] != 'calibration' or values.shape != cap.samples.shape or error.shape != values.shape
                or binding['C_input_identity'] != cap.identity
                or binding['source_graph'] != admission.model.graph().digest()
                or binding['source_response_C_updates_before_noise_fit'] != 0
                or set(binding['used_label_event_ids']) != {row[0] for row in cap.label_rows}):
            raise ValueError('fixed native C mean must consume exactly original authorized C bits/raw/source')
        residual = ResidualRecord.from_arrays(samples=cap.samples, design=values,
            kernel_mean=np.eye(len(geometry.channels)), nuisance=np.zeros_like(values),
            record_id=b['record_id'], participant_key=b['participant_key'], recording_id=b['record_id'],
            sample_start=b['sample_start'], geometry=geometry, role='target_calibration',
            source_model_sha256=binding['source_graph'], source_input_sha256=cap.identity)
        # A conservative binary64 subtraction bound accompanies the source mean
        # operand error; it does not alter the residual given to the fitter.
        subtraction_bound = np.nextafter(np.finfo(float).eps *
            (np.abs(cap.samples) + np.abs(values) + np.abs(residual.residual)), np.inf)
        residuals.append(residual)
        bindings.append({'C_input_identity': cap.identity, 'original_public_scope': plain_metadata(b),
            'granted_label_count': len(cap.label_rows), 'positive_label_count': sum(row[2] for row in cap.label_rows),
            'source_mean_binding': plain_metadata(binding), 'source_mean_sha256': array_digest(values),
            'source_mean_error_sha256': array_digest(error), 'source_mean_error_max': float(error.max()),
            'residual_subtraction_bound_sha256': array_digest(subtraction_bound),
            'residual_subtraction_bound_max': float(subtraction_bound.max()), 'residual': residual.ledger()})
    programme = freeze_metadata({'schema': 'native_public_C_noise_programme/1', 'recipe': RECIPE_NAME,
        'C_prefix_identity': identity, 'requested_C_label_budget': prefix.budget,
        'geometry': residuals[0].geometry.record(), 'family': 'private', 'order': 1,
        'conditioning_samples': 32, 'minimum_samples_per_parameter': 20, 'eta': 1.,
        'source_admission_identity': metadata_digest(admission.record),
        'source_graph': admission.model.graph().digest(),
        'source_prior_sha256': array_digest(admission.model.graph().source_mean, admission.model.graph().source_factor),
        'fixed_mean_programme': mean.identity, 'C_inputs': bindings,
        'original_right_censoring': {r.record_id: r.right_censored for r in prefix.calibration_records},
        'segment_law': SEGMENT_LAW, 'noise_C_fit_calls': 1, 'response_C_updates_here': 0,
        'source_response_prior_rescaled': False, 'source_factor_projections': 0,
        'held_sample_reads': 0, 'adapter_producer_sha256': IMPLEMENTATION_SHA256})
    result = CompiledPublicCResiduals(admission, prefix, inputs, mean, tuple(residuals), programme)
    _COMPILED[result] = programme
    return result


@dataclass(frozen=True)
class PublicNativeCNoiseOperand:
    noise: FittedChannelNoise
    programme: dict

    def __post_init__(self):
        self.noise.assert_integrity()
        programme = freeze_metadata(self.programme)
        if (programme['recipe'] != RECIPE_NAME or programme['segment_law'] != SEGMENT_LAW
                or self.noise.geometry.record() != plain_metadata(programme['geometry'])
                or self.noise.family != programme['family'] or self.noise.order != programme['order']
                or self.noise.conditioning_samples != programme['conditioning_samples']
                or metadata_digest(self.noise.provenance['records']) != metadata_digest(
                    [row['residual'] for row in programme['C_inputs']])):
            raise ValueError('complete native noise must retain its actual C fit programme/geometry/ledgers')
        object.__setattr__(self, 'programme', programme)

    @property
    def identity(self):
        return metadata_digest({'noise': self.noise.record(), 'programme': self.programme})

    @property
    def eta(self):
        return self.programme['eta']

    def carrier(self):
        return StoredNoiseCarrier(self.noise, eta=self.eta)

    def initial_state_root(self):
        dimension = self.noise.order * len(self.noise.geometry.channels)
        return self.noise.compiled_prefix_root[:dimension, :dimension]

    def save(self, path, *, existing_noise_payload):
        path, noise_path = Path(path), Path(existing_noise_payload)
        payload = json.loads(noise_path.read_bytes())
        if (path.resolve().parent != noise_path.resolve().parent
                or payload['state_sha256'] != metadata_digest(self.noise.record())
                or metadata_digest(payload['state']) != payload['state_sha256']):
            raise ValueError('native noise descriptor requires its complete exact original local noise bytes')
        record = {'schema': 'native_public_complete_C_noise_operand/1', 'identity': self.identity,
            'programme': plain_metadata(self.programme), 'noise_payload': {'path': noise_path.name,
                'sha256': _sha(noise_path)}, 'noise_record_sha256': payload['state_sha256'],
            'initial_state_root_sha256': array_digest(self.initial_state_root()),
            'initial_state_root_shape': self.initial_state_root().shape,
            'numeric_only_replay_grants_live_C_or_raw_authority': False}
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
        return _sha(path)

    @classmethod
    def load(cls, path, *, expected_sha256):
        path = Path(path)
        if _sha(path) != expected_sha256:
            raise ValueError('native noise operand differs from its saved admitted bytes')
        record = json.loads(path.read_bytes())
        if record['schema'] != 'native_public_complete_C_noise_operand/1':
            raise ValueError('unknown native public noise operand schema')
        noise_path = path.parent / record['noise_payload']['path']
        if _sha(noise_path) != record['noise_payload']['sha256']:
            raise ValueError('complete original native noise payload changed')
        result = cls(FittedChannelNoise.load(noise_path), record['programme'])
        if (result.identity != record['identity']
                or metadata_digest(result.noise.record()) != record['noise_record_sha256']
                or array_digest(result.initial_state_root()) != record['initial_state_root_sha256']):
            raise ValueError('complete cold native noise parameters changed')
        return result


@dataclass(frozen=True, eq=False)
class PublicNativeCNoiseAdmission:
    compiled: CompiledPublicCResiduals
    operand: PublicNativeCNoiseOperand

    def segment_binding(self, physical_input):
        require_public_c_noise(self)
        cap = require_public_physical_input(physical_input)
        b, p = cap.public_binding, self.operand.programme
        admission = self.compiled.source_admission
        roster = admission.record['new_target_union']['source_record_roster']
        if (b['C_prefix_identity'] != p['C_prefix_identity']
                or b['participant_key'] != admission.model.target_participant_key
                or _geometry(b) != self.operand.noise.geometry
                or b['record_id'] not in roster or b['source_sha256'] != roster[b['record_id']]['source_sha256']):
            raise ValueError('noise application requires its original participant/C prefix/native acquisition')
        segment = {key: b[key] for key in ('dataset_id', 'participant_key', 'record_id', 'source_sha256')}
        return {'law': SEGMENT_LAW, 'original_record_noise_segment': metadata_digest(segment),
            'original_acquisition': segment, 'C_prefix_identity': p['C_prefix_identity'],
            'noise_operand': self.operand.identity, 'observation_union': b['observation_union'],
            'consume_each_original_sample_once': True, 'startup_and_drift_once_per_original_record': True}


def admit_fitted_public_c_noise(compiled, noise):
    if type(compiled) is not CompiledPublicCResiduals or _COMPILED.get(compiled) is not compiled.programme:
        raise TypeError('noise admission requires its live original compiled C residual programme')
    require_public_native_source(compiled.source_admission)
    for cap in compiled.inputs:
        require_public_physical_input(cap)
    operand = PublicNativeCNoiseOperand(noise, compiled.programme)
    result = PublicNativeCNoiseAdmission(compiled, operand)
    _ADMITTED[result] = operand
    return result


def require_public_c_noise(value):
    if type(value) is not PublicNativeCNoiseAdmission or _ADMITTED.get(value) is not value.operand:
        raise TypeError('native C noise use requires its live original-source/C-residual admission')
    compiled = value.compiled
    if _COMPILED.get(compiled) is not compiled.programme:
        raise TypeError('native C noise lost its live original compiled programme')
    require_public_native_source(compiled.source_admission)
    for cap in compiled.inputs:
        require_public_physical_input(cap)
    if (metadata_digest([row.ledger() for row in compiled.residuals]) !=
            metadata_digest(value.operand.noise.provenance['records'])):
        raise ValueError('native C noise residual relation changed')
    value.operand.noise.assert_integrity()
    return value
