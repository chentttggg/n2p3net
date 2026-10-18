"""Original public C-noise fit -> same residuals under a new source consumer.

Both live parents remain distinct. Exact original parameters, fitted ledgers and
saved startup roots are consumed without refitting or rewriting provenance.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from research.frozen_metadata import freeze_metadata, plain_metadata
from research.noise_reuse_relation import exact_parameter_law_relation
from research.public_native_source_admission import require_public_native_source
from research.public_native_source_mean import PublicNativeSourceCMean
from research.public_native_target_noise import require_public_c_noise
from research.public_record_physical_bridge import require_public_physical_input
from transfer.ensemble_views import calibration_prefix_identity
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_REUSES = WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class ReusedPublicCNoise:
    original: object
    source_admission: object
    prefix: object
    calibration_inputs: tuple
    fixed_mean: object
    record: dict
    identity: str

    @property
    def noise(self):
        return self.original.operand.noise

    @property
    def eta(self):
        return self.original.operand.eta

    def carrier(self):
        return self.original.operand.carrier()

    def segment_binding(self, cap):
        require_reused_public_c_noise(self)
        cap = require_public_physical_input(cap)
        b = cap.public_binding
        geometry = self.noise.geometry
        roster = self.source_admission.record['new_target_union']['source_record_roster']
        if (b['participant_key'] != self.source_admission.model.target_participant_key
                or b['C_prefix_identity'] != calibration_prefix_identity(self.prefix)
                or b['record_id'] not in roster or b['source_sha256'] != roster[b['record_id']]['source_sha256']
                or (tuple(b['eeg_channels']), b['sample_rate_hz'], b['signal_unit'], b['reference'], b['operator']) != (
                    geometry.channels, geometry.sfreq, geometry.unit, geometry.reference, geometry.operator_identity)):
            raise ValueError('reused public noise requires its exact original acquisition and native coordinates')
        acquisition = {key: b[key] for key in ('dataset_id', 'participant_key', 'record_id', 'source_sha256')}
        return {'law': 'original_record_stationary_startup_C_history_physical_union/1',
            'original_record_noise_segment': metadata_digest(acquisition), 'original_acquisition': acquisition,
            'original_noise_operand': self.record['original_noise_operand'], 'reuse_relation': self.identity,
            'C_prefix_identity': calibration_prefix_identity(self.prefix),
            'consume_each_original_sample_once': True, 'startup_and_drift_once_per_original_record': True}


def admit_reused_public_c_noise(original, source_admission, prefix, calibration_inputs, *, fixed_mean):
    original = require_public_c_noise(original)
    admission = require_public_native_source(source_admission)
    prefix = admission.compiler.assert_original_prefix(prefix)
    prefix_identity = calibration_prefix_identity(prefix)
    inputs = tuple(require_public_physical_input(cap) for cap in calibration_inputs)
    expected = tuple(cap for cap in admission.calibration_inputs
        if cap.public_binding['C_prefix_identity'] == prefix_identity)
    if (not any(prefix is item for item in admission.prefixes) or len(inputs) != len(expected)
            or any(a is not b for a, b in zip(inputs, expected, strict=True))
            or calibration_prefix_identity(original.compiled.prefix) != prefix_identity
            or type(fixed_mean) is not PublicNativeSourceCMean or fixed_mean.model is not admission.model):
        raise ValueError('public noise reuse requires both complete original same-prefix C parents')
    law = exact_parameter_law_relation(original.compiled.source_admission.model.graph(), admission.model.graph())
    before_inputs = original.compiled.inputs
    fitted = original.operand.noise.provenance['records']
    if len(before_inputs) != len(inputs) or len(fitted) != len(inputs):
        raise ValueError('public noise reuse cannot drop an original fitted C record')
    rows = []
    keys = ('dataset_id', 'participant_key', 'record_id', 'source_sha256', 'sample_start', 'full_stop',
            'prefix_stop', 'available_stop', 'sample_rate_hz_exact_ratio', 'signal_unit', 'eeg_channels',
            'reference', 'operator', 'public_physical_history', 'C_label_use_sha256', 'C_prefix_identity')
    for old, new, ledger in zip(before_inputs, inputs, fitted, strict=True):
        require_public_physical_input(old)
        if (any(metadata_digest(old.public_binding[key]) != metadata_digest(new.public_binding[key]) for key in keys)
                or old.label_rows != new.label_rows or not np.array_equal(old.samples, new.samples)):
            raise ValueError('public noise reuse changed original C observations, labels or physical programme')
        old_values, old_error, old_binding = original.compiled.mean(old)
        new_values, new_error, new_binding = fixed_mean(new)
        if (old_binding['source_graph'] != original.compiled.source_admission.model.graph().digest()
                or new_binding['source_graph'] != admission.model.graph().digest()
                or old_values.dtype != new_values.dtype or old_error.dtype != new_error.dtype
                or array_digest(old_values, old_error) != array_digest(new_values, new_error)):
            raise ValueError('the actual fixed original/new C means or complete errors differ')
        channels = len(original.operand.noise.geometry.channels)
        residual = new.samples - new_values @ np.eye(channels) - np.zeros_like(new_values)
        if (ledger['record_id'] != new.public_binding['record_id']
                or tuple(ledger['read_support']) != (0, len(new.samples))
                or array_digest(residual) != ledger['residual_sha256']):
            raise ValueError('public reuse must reproduce the original fitter residual bytes and support')
        rows.append({'original_input': old.identity, 'actual_input': new.identity,
            'original_mean_binding': metadata_digest(old_binding), 'actual_mean_binding': metadata_digest(new_binding),
            'mean_and_error_sha256': array_digest(new_values, new_error),
            'raw_samples_sha256': array_digest(new.samples), 'original_fitted_residual': plain_metadata(ledger),
            'same_numeric_C_residual': True})
    record = freeze_metadata({'schema': 'original_public_C_noise_actual_residual_reuse/1',
        'original_noise_operand': original.operand.identity,
        'original_complete_noise_record': metadata_digest(original.operand.noise.record()),
        'original_C_programme': metadata_digest(original.compiled.programme),
        'actual_source_admission': metadata_digest(admission.record), 'C_prefix_identity': prefix_identity,
        'exact_source_parameter_law': law, 'actual_C_records': rows,
        'noise_refits': 0, 'response_C_updates_here': 0, 'original_fit_provenance_rewritten': False,
        'producer': IMPLEMENTATION_SHA256})
    result = ReusedPublicCNoise(original, admission, prefix, inputs, fixed_mean, record, metadata_digest(record))
    _REUSES[result] = record
    return result


def require_reused_public_c_noise(value):
    if type(value) is not ReusedPublicCNoise or _REUSES.get(value) is not value.record:
        raise TypeError('public noise use requires its original live two-parent residual reuse relation')
    require_public_c_noise(value.original)
    require_public_native_source(value.source_admission)
    for cap in value.calibration_inputs:
        require_public_physical_input(cap)
    return value
