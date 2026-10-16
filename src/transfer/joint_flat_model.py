"""Typed C posterior and captured P-only prediction for the joint-flat expert."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import numpy as np
from threadpoolctl import threadpool_info

from research.channel_noise_mechanism import UnidentifiableFitError
from research.joint_flat_chain import JointHistorySamples, summarize_joint_tail
from research.joint_flat_collapsed_predictive import collapsed_tail_components
from research.joint_flat_factor_history import (
    FactoredTargetAxis,
    factored_tail_components,
    run_factored_history_gibbs,
)
from research.joint_flat_history_builder import (
    prepare_shared_record,
    pulse_target_image,
    shared_history_kernel,
)
from research.joint_response_sampling import (
    SamplingResolutionError,
    _functional_diagnostics,
)
from transfer.joint_flat_programme import (
    FrozenJointFlatSource,
    FrozenJointFlatSourcePreparation,
    JointFlatProgram,
    joint_flat_numeric_producer_identity,
)
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest

_EXECUTED_MODEL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _history_consumer_context():
    """Keep the current consumer separate from the captured numerical producer."""
    return {'executed_model_sha256': _EXECUTED_MODEL_SHA256,
            'current_modules': joint_flat_numeric_producer_identity()}


def _history_native_context():
    pools = [{name: row.get(name) for name in ('filepath', 'version', 'user_api', 'internal_api',
        'prefix', 'num_threads', 'threading_layer', 'architecture')} for row in threadpool_info()]
    pools.sort(key=lambda row: (str(row['filepath']), str(row['internal_api']), str(row['prefix'])))
    torch = sys.modules.get('torch')
    return {'native_pools': pools, 'torch': None if torch is None else {
        'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()}}


class _VerifiedHistoryComponents:
    """One model's bounded, nonserialized memo of byte-verified components.

    The byte bound counts array payloads. The separate entry bound also limits
    key/container overhead. Entries never hold a caller's mutable return value.
    """

    def __init__(self, capacity_bytes=8 * 2**20, max_entries=16):
        if (type(capacity_bytes) is not int or capacity_bytes < 0
                or type(max_entries) is not int or max_entries < 0):
            raise ValueError('history component cache requires nonnegative byte and entry capacities')
        self.capacity_bytes, self.max_entries = capacity_bytes, max_entries
        self.used_bytes, self.hits, self.misses = 0, 0, 0
        self._entries, self._lock = OrderedDict(), RLock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            return entry

    def put(self, key, logweights, logdensity):
        nbytes = logweights.nbytes + logdensity.nbytes
        if not self.max_entries or not self.capacity_bytes or nbytes > self.capacity_bytes:
            return
        entry = frozen_array(logweights), frozen_array(logdensity)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self.used_bytes -= sum(value.nbytes for value in previous)
            while self._entries and (len(self._entries) >= self.max_entries
                    or self.used_bytes + nbytes > self.capacity_bytes):
                _, previous = self._entries.popitem(last=False)
                self.used_bytes -= sum(value.nbytes for value in previous)
            self._entries[key] = entry
            self.used_bytes += nbytes


def _sampling_record(samples):
    return {'seeds': list(samples.seeds), 'burnin': samples.burnin, 'sweeps_per_draw': samples.sweeps_per_draw,
        'degrees_of_freedom': str(samples.degrees_of_freedom), 'raw_prefix_rows': samples.raw_prefix_rows,
        'record_scalar_rows': list(samples.record_scalar_rows), 'elapsed_seconds': samples.elapsed_seconds,
        'qr_updated_leaves': samples.qr_updated_leaves,
        'arrays_sha256': array_digest(samples.coefficients, samples.precisions, samples.targets)}


def _restore_sampling(record, coefficients, precisions, targets):
    if array_digest(coefficients, precisions, targets) != record['arrays_sha256']:
        raise ValueError('captured posterior coefficient/precision/complete-target arrays changed')
    return JointHistorySamples(coefficients, precisions, targets, tuple(record['seeds']), record['burnin'],
        record['sweeps_per_draw'], float(record['degrees_of_freedom']), record['raw_prefix_rows'],
        record['elapsed_seconds'], record['qr_updated_leaves'], tuple(record['record_scalar_rows']))


def _diagnostics(samples, *, program, tail_summary=None):
    values = samples.coefficients
    if not np.isposinf(samples.degrees_of_freedom):
        values = np.concatenate((values, np.log(samples.precisions)), axis=2)
    rows = [_functional_diagnostics(values[:, :, index]) for index in range(values.shape[-1])]
    policy = program.sampler
    for index, value in enumerate(rows):
        if not value.get('constant_at_retained_draws', False) and (
                value['rank_folded_split_rhat'] > policy.max_rhat
                or value['bulk_ess'] < policy.min_bulk_ess or value['tail_ess'] < policy.min_tail_ess):
            raise SamplingResolutionError('actual joint posterior coordinate '+str(index)
                +' requires additional mixing: '+json.dumps(value, sort_keys=True))
    if tail_summary is not None:
        for index, value in enumerate(tail_summary['functional_diagnostics']):
            if not value.get('constant_at_retained_draws', False) and (
                    value['rank_folded_split_rhat'] > policy.max_rhat
                    or value['bulk_ess'] < policy.min_bulk_ess or value['tail_ess'] < policy.min_tail_ess):
                raise SamplingResolutionError('paired current-q predictive functional '+str(index)
                    +' requires additional mixing: '+json.dumps(value, sort_keys=True))
        if max(tail_summary['log_probability_paired_batch_mcse']) > policy.max_log_probability_mcse:
            raise SamplingResolutionError('paired current-q log probabilities require additional retained draws')
    return {'scope': 'actual_retained_coordinate_and_predictive_functional_chains',
        'coordinate_diagnostics': rows, 'policy': policy.record(),
        'retained_prefix_measure_normalized': True, 'posterior_accuracy_method': 'declared_MCMC_diagnostics'}


def _overdispersed_starts(records, axes, program, *, phase, calibration=None, calibration_record_ids=(), record_ids=()):
    k, c = records[0].response_width, records[0].channels
    seeds = program.sampler.stream_seeds(phase+'_initialisation')
    beta = np.empty((len(seeds), k, c))
    drifts = [np.empty((len(seeds), r.drift_width, c)) for r in records]
    precision = np.ones((len(seeds), len(records)))
    targets = np.empty((len(seeds), len(axes)), dtype=np.int64)
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        beta[chain] = program.sampler.start_scale*rng.normal(size=(k, c))
        for index, record in enumerate(records):
            drifts[index][chain] = program.sampler.start_scale*rng.normal(size=(record.drift_width, c))
        if calibration is not None:
            # An original C draw is an initial point only. The P sampler below
            # still uses one source prior and all original C/P likelihood rows.
            old = calibration.coefficients[chain % len(calibration.seeds), -1]
            beta[chain] = old[:k*c].reshape(k, c)
            offset = k*c
            for old_index, old_id in enumerate(calibration_record_ids):
                if old_id in record_ids:
                    new_index = record_ids.index(old_id)
                    count = records[new_index].drift_width*c
                    drifts[new_index][chain] = old[offset:offset+count].reshape(records[new_index].drift_width, c)
                    if not np.isposinf(program.sampler.degrees_of_freedom):
                        precision[chain, new_index] = calibration.precisions[chain % len(calibration.seeds), -1, old_index]
                offset += 2*c
        for index, axis in enumerate(axes):
            logmass = np.full(len(axis.prior_mass), -np.inf)
            positive = axis.prior_mass > 0
            logmass[positive] = np.log(axis.prior_mass[positive])
            targets[chain, index] = np.argmax(logmass+rng.gumbel(size=len(logmass)))
    return beta, drifts, precision, targets


def _run_prefix(records, axes, program, kernel, *, phase, **starts):
    sampler = program.sampler
    return run_factored_history_gibbs(records, axes, degrees_of_freedom=sampler.degrees_of_freedom,
        initial_coefficients=starts['beta'], initial_drifts=starts['drifts'],
        initial_precisions=starts['precision'], initial_targets=starts['targets'], seeds=sampler.stream_seeds(phase+'_transition'),
        burnin=sampler.burnin, draws=sampler.draws_per_chain, sweeps_per_draw=sampler.sweeps_per_draw,
        leaf_rows=sampler.leaf_rows, cache_bytes=sampler.qr_cache_bytes,
        proper_prior_covariance_error=kernel.proper_prior_covariance_error)


@dataclass(frozen=True)
class JointFlatPredictionCapture:
    samples: JointHistorySamples
    logweights: np.ndarray
    logdensity: np.ndarray
    binding: dict
    diagnostics: dict

    def __post_init__(self):
        object.__setattr__(self, 'logweights', frozen_array(self.logweights))
        object.__setattr__(self, 'logdensity', frozen_array(self.logdensity))
        object.__setattr__(self, 'binding', freeze_metadata(self.binding))
        object.__setattr__(self, 'diagnostics', freeze_metadata(self.diagnostics))

    @property
    def digest(self):
        return metadata_digest({'binding': self.binding, 'sampling': _sampling_record(self.samples),
            'components': array_digest(self.logweights, self.logdensity), 'diagnostics': self.diagnostics})


@dataclass(frozen=True)
class JointFlatPrediction:
    log_evidence: np.ndarray
    log_probability: np.ndarray
    input_digest: str
    capture: JointFlatPredictionCapture
    summary: dict


@dataclass(frozen=True)
class JointFlatModel:
    source: FrozenJointFlatSource
    programme: JointFlatProgram
    calibration: JointHistorySamples | None
    calibration_record_ids: tuple[str, ...]
    calibration_inputs: tuple[dict, ...]
    provenance: dict
    _verified_history_components: _VerifiedHistoryComponents = field(
        default_factory=_VerifiedHistoryComponents, init=False, repr=False, compare=False)

    @classmethod
    def fit_prefix(cls, source, programme, *, compiler, prefix):
        from transfer.ensemble_views import (
            CalibrationPrefixCompiler,
            RecordCalibrationPrefixCompiler,
            calibration_prefix_identity,
        )

        if type(compiler) not in {CalibrationPrefixCompiler, RecordCalibrationPrefixCompiler}:
            raise TypeError('joint calibration requires the original dataset prefix compiler')
        prefix = compiler.assert_original_prefix(prefix)
        source.assert_target(prefix.target_contract)
        if programme.source_exclusion_scope == 'whole_target_dataset' and any(
                row['target_contract']['source_dataset_id'] == prefix.target_contract['source_dataset_id']
                for row in source.parameters.binding['source_prefixes']):
            raise ValueError('whole-target-dataset source programme includes the actual target dataset')
        physical_map = compiler.calibration_record_physical_map(prefix)
        kernel = shared_history_kernel(source)
        records, inputs = [], []
        for raw in prefix.calibration_records:
            physical = physical_map['records'][raw.record_id]
            record, binding = prepare_shared_record(kernel, raw.samples, raw.onsets, raw.target_mask,
                                                    record_id=physical['physical_record_id'])
            records.append(record)
            inputs.append({**binding, 'C_record_id': raw.record_id,
                'original_physical_record': physical,
                'actual_C_arrays_sha256': array_digest(raw.samples, raw.onsets, raw.target_mask)})
        calibration = None
        diagnostics = {'source_only_K0': True}
        if records:
            beta, drifts, precision, targets = _overdispersed_starts(records, (), programme, phase='C')
            calibration = _run_prefix(records, (), programme, kernel, phase='C', beta=beta, drifts=drifts,
                                      precision=precision, targets=targets)
            diagnostics = _diagnostics(calibration, program=programme)
        provenance = {'schema': 'n2p3_joint_flat_fitted_prefix/1', 'fit_scope': 'target_calibration',
            'numeric_producer': joint_flat_numeric_producer_identity(),
            'original_C_prefix_identity': calibration_prefix_identity(prefix),
            'target_contract': prefix.target_contract, 'requested_labels': prefix.budget,
            'actual_C_records': inputs, 'source_checkpoint_sha256': source.digest,
            'original_source_model_sha256': source.parameters.binding['source_model_sha256'],
            'kernel': kernel.binding, 'diagnostics': diagnostics,
            'original_C_physical_record_map': physical_map,
            'source_response_or_noise_refits': 0, 'future_labels_consumed': False}
        return cls(source, programme, calibration, tuple(raw.record_id for raw in prefix.calibration_records),
                   tuple(freeze_metadata(row) for row in inputs), freeze_metadata(provenance))

    def digest(self):
        return metadata_digest({'source': self.source.digest, 'programme': self.programme.record(),
            'C_record_ids': self.calibration_record_ids, 'C_inputs': self.calibration_inputs,
            'C_sampling': None if self.calibration is None else _sampling_record(self.calibration),
            'provenance': self.provenance})

    def validate_history_permission(self, view, capability):
        """Check original C/current-label authority without sampling or reading T statistics."""
        from transfer.ensemble_views import AuthorizedRecordPublicHistory

        if not isinstance(capability, AuthorizedRecordPublicHistory):
            raise TypeError('joint prediction requires the original issuer complete-record P/T capability')
        binding = capability.binding
        if (binding['current_decision_id'] != view.decision_id or binding['participant_key'] != view.participant_key
                or binding['source_sha256'] != view.source_sha256
                or metadata_digest(binding['current_view']) != metadata_digest(view.record())
                or binding['samples_sha256'] != array_digest(capability.samples)
                or binding['prefix_identity'] != self.provenance['original_C_prefix_identity']
                or binding['future_truth_consumed'] is not False):
            raise ValueError('record capability changed its original view, C prefix, bytes or hidden-truth boundary')
        self.source.assert_target(self.provenance['target_contract'])
        p_stop, t_stop = binding['conditioning_sample_support'][1], binding['score_sample_support'][1]
        if not 0 < p_stop < t_stop == len(capability.samples):
            raise ValueError('actual original P/T cutoff and supplied native observations changed')
        mapping = binding['C_record_to_original_record']
        if ({raw.record_id for raw in capability.calibration_records} != set(self.calibration_record_ids)
                or len(capability.calibration_records) != len(self.calibration_record_ids)):
            raise ValueError('record capability changed the original fitted C recording roster')
        for raw in capability.calibration_records:
            expected = next((row for row in self.calibration_inputs if row['C_record_id'] == raw.record_id), None)
            digest = array_digest(raw.samples, raw.onsets, raw.target_mask)
            if (expected is None or digest != expected['actual_C_arrays_sha256']
                    or mapping[raw.record_id] != expected['original_physical_record']['physical_record_id']
                    or binding['C_record_input_sha256'][raw.record_id] != digest):
                raise ValueError('record capability differs from original fitted C samples, labels or acquisition map')
            if mapping[raw.record_id] == capability.record_id:
                if len(raw.samples) > p_stop or not np.array_equal(raw.samples, capability.samples[:len(raw.samples)]):
                    raise ValueError('same-record C exceeds or differs from original P observations')
                current = next((selection for selection in binding['public_selections']
                                if selection['selection_id'] == view.decision_id), None)
                if current is None or np.intersect1d(raw.onsets, current['onsets']).size:
                    raise ValueError('independent joint prediction cannot consume current-selection C labels')
        return binding

    def score_history(self, view, capability, *, captured=None, _readset_relation=None):
        binding = self.validate_history_permission(view, capability)
        kernel = shared_history_kernel(self.source)
        p_stop, t_stop = binding['conditioning_sample_support'][1], binding['score_sample_support'][1]
        mapping = binding['C_record_to_original_record']
        records, row_bindings, record_ids = [], [], []
        current_C = None
        for raw in capability.calibration_records:
            original_id = mapping[raw.record_id]
            expected = next((row for row in self.calibration_inputs if row['C_record_id'] == raw.record_id), None)
            if expected is None or expected['actual_C_arrays_sha256'] != array_digest(raw.samples, raw.onsets, raw.target_mask):
                raise ValueError('score capability C inputs differ from the original fitted prefix')
            if (original_id != expected['original_physical_record']['physical_record_id']
                    or binding['C_record_input_sha256'][raw.record_id] != expected['actual_C_arrays_sha256']):
                raise ValueError('record capability changed an original C acquisition map or array identity')
            if original_id == capability.record_id:
                if not np.array_equal(raw.samples, capability.samples[:len(raw.samples)]) or len(raw.samples) > p_stop:
                    raise ValueError('same-record C does not agree with the original available conditioning prefix')
                current_C = raw
                continue
            record, owned = prepare_shared_record(kernel, raw.samples, raw.onsets, raw.target_mask,
                                                  record_id=original_id)
            records.append(record)
            row_bindings.append(owned)
            record_ids.append(original_id)
        public = binding['public_selections']
        onsets = [int(t) for selection in public for t in selection['onsets']]
        record, owned = prepare_shared_record(kernel, capability.samples[:p_stop], onsets,
            np.zeros(len(onsets), dtype=bool), record_id=capability.record_id)
        # Current-record selections belong entirely to their q axes. C-known
        # target increments are not also added to the common base.
        records.append(record)
        row_bindings.append(owned)
        record_ids.append(capability.record_id)
        current_record_index = len(records)-1
        known = {} if current_C is None else dict(zip(map(int, current_C.onsets), map(bool, current_C.target_mask), strict=True))
        current_public = next((selection for selection in public if selection['selection_id'] == view.decision_id), None)
        if current_public is None:
            raise ValueError('original current decision is absent from its issued public history')
        if any(int(onset) in known for onset in current_public['onsets']):
            raise ValueError('independent joint prediction cannot consume labels from its current held selection')
        axes, current_axis = [], None
        for selection in public:
            membership = np.asarray(selection['membership'], dtype=bool)
            times = tuple(map(int, selection['onsets']))
            prior = np.ones(len(selection['candidate_vocabulary']))
            for index, onset in enumerate(times):
                if onset in known:
                    prior *= membership[index] == known[onset]
            if not prior.any():
                raise UnidentifiableFitError('original C labels and public selection membership have no compatible target')
            prior /= prior.sum()
            prefix_images, tail_images = [], []
            for q in range(len(prior)):
                events = [t for t, included in zip(times, membership[:, q], strict=True) if included]
                prefix_images.append(pulse_target_image(kernel, events, stop=p_stop, empty_start=p_stop))
                tail_images.append(pulse_target_image(kernel, events, stop=t_stop, empty_start=t_stop))
            if selection['selection_id'] == view.decision_id:
                current_axis = len(axes)
            axes.append(FactoredTargetAxis(selection['selection_id'], tuple(selection['candidate_vocabulary']), prior,
                tuple(prefix_images), tuple(tail_images), current_record_index))
        if current_axis is None or axes[current_axis].candidate_vocabulary != view.decision.candidate_vocabulary:
            raise ValueError('the original current selection is absent from its full public history')
        calibration_ids = tuple(mapping[key] for key in self.calibration_record_ids)
        input_binding = {'schema': 'n2p3_joint_flat_actual_prefix_prediction/1',
            'numeric_producer': (joint_flat_numeric_producer_identity() if captured is None
                                 else captured.binding['input']['numeric_producer']),
            'model_sha256': self.digest(), 'source_sha256': self.source.digest,
            'programme': self.programme.record(), 'history_capability': binding,
            'P_record_operands': row_bindings, 'P_original_record_ids': record_ids,
            'actual_target_axes': [{'selection_id': axis.selection_id,
                'candidate_vocabulary': list(axis.candidate_vocabulary), 'prior_mass': axis.prior_mass.tolist(),
                'original_record_id': record_ids[axis.record_index]} for axis in axes],
            'current_axis': current_axis, 'tail_values_read_for_prefix_posterior': False}
        relation_digest = None
        if _readset_relation is not None:
            from research.joint_readset_transition import _IssuedJointReadsetTransition

            if type(_readset_relation) is not _IssuedJointReadsetTransition or captured is None:
                raise TypeError('joint captured readset requires its actual issued transition')
            input_binding = _readset_relation.input_relation(capability, captured, input_binding)
            relation_digest = _readset_relation.digest
        input_digest = metadata_digest(input_binding)
        if captured is None:
            beta, drifts, precision, targets = _overdispersed_starts(records, axes, self.programme, phase='P',
                calibration=self.calibration, calibration_record_ids=calibration_ids, record_ids=tuple(record_ids))
            samples = _run_prefix(records, axes, self.programme, kernel, phase='P', beta=beta, drifts=drifts,
                                  precision=precision, targets=targets)
        else:
            if captured.binding['prefix_input_sha256'] != input_digest:
                raise ValueError('captured original P measure belongs to another actual history or model')
            samples = captured.samples
        # Read-only T operands are compiled only after the original P sampler
        # has frozen its measure. They never enter its conditional kernels.
        full, full_binding = prepare_shared_record(kernel, capability.samples, onsets,
            np.zeros(len(onsets), dtype=bool), record_id=capability.record_id)
        offset = p_stop-full.startup_native_rows
        tail_rows, tail_errors = full.tail_rows[offset:], full.tail_errors[offset:]
        noise = self.source.parameters.noise
        # P contains the original complete startup. Each subsequent square
        # innovation contributes the original spatial root once; computing
        # this tail determinant directly avoids subtracting two long logs.
        tail_logdet = (2*tail_rows.shape[0]*float(np.log(np.diag(noise._spatial_root)).sum())
            +tail_rows.shape[0]*full.channels*np.log(self.source.parameters.hyperparameters.relative_noise_shape))
        normalizer_error = _normalizer_error(noise, p_stop, t_stop,
                                            self.source.parameters.hyperparameters.relative_noise_shape)
        common = {'current_axis': current_axis, 'tail_native_start': p_stop,
            'tail_noise_logdet': tail_logdet, 'tail_noise_logdet_error': normalizer_error}
        cache_key, contexts = None, None
        components = None
        if captured is not None:
            contexts = _history_consumer_context(), _history_native_context()
            cache_key = metadata_digest({'model_sha256': self.digest(), 'source_sha256': self.source.digest,
                'input_sha256': input_digest, 'current_view': view.record(), 'complete_history': binding,
                'capture_sha256': captured.digest, 'T_record_operands': full_binding,
                'tail_normalizer': common, 'readset_transition': relation_digest,
                'consumer': contexts[0], 'native': contexts[1]})
            components = self._verified_history_components.get(cache_key)
        if components is not None:
            logweights, logdensity = components
        elif self.programme.sampler.predictive_integration == 'collapsed_beta_all_drift_current_q':
            logweights, logdensity = collapsed_tail_components(records, tail_rows, tail_errors, axes, samples,
                **common, leaf_rows=self.programme.sampler.leaf_rows,
                cache_bytes=self.programme.sampler.qr_cache_bytes,
                prior_covariance_error=kernel.proper_prior_covariance_error)
        else:
            logweights, logdensity = factored_tail_components(records, tail_rows, tail_errors, axes, samples, **common)
        summary = summarize_joint_tail(logweights, logdensity, current_prior=axes[current_axis].prior_mass,
            scoring_law=self.programme.scoring_law, batch_size=self.programme.sampler.batch_size)
        diagnostics = _diagnostics(samples, program=self.programme, tail_summary=summary)
        capture = JointFlatPredictionCapture(samples, logweights, logdensity,
            {'prefix_input_sha256': input_digest, 'input': input_binding, 'T_record_operands': full_binding,
             'raw_T_noise_logdet': tail_logdet, 'raw_T_noise_logdet_error_estimate': normalizer_error}, diagnostics)
        if captured is not None and (not np.array_equal(logweights, captured.logweights)
                                     or not np.array_equal(logdensity, captured.logdensity)):
            raise ValueError('original captured posterior components changed during model-only replay')
        if _readset_relation is not None:
            _readset_relation.confirm_capture(capability, captured, capture)
        if (captured is not None and components is None
                and array_digest(logweights, logdensity) == array_digest(captured.logweights, captured.logdensity)
                and contexts == (_history_consumer_context(), _history_native_context())):
            self._verified_history_components.put(cache_key, logweights, logdensity)
        return JointFlatPrediction(summary['log_evidence'], summary['log_probability'], input_digest, capture, summary)

    def save(self, path):
        state = {'schema': 'n2p3_joint_flat_model/1', 'programme': self.programme.record(),
            'source_sha256': self.source.digest, 'calibration_record_ids': self.calibration_record_ids,
            'calibration_inputs': self.calibration_inputs, 'provenance': self.provenance,
            'calibration_sampling': None if self.calibration is None else _sampling_record(self.calibration),
            'model_sha256': self.digest()}
        arrays = {'metadata_json': np.frombuffer(json.dumps(state, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode('utf8'), dtype=np.uint8), 'source_bytes': np.frombuffer(self.source.content, dtype=np.uint8)}
        if self.calibration is not None:
            arrays.update(C_coefficients=self.calibration.coefficients, C_precisions=self.calibration.precisions,
                          C_targets=self.calibration.targets)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, *, source_preparation=None):
        raw = path if hasattr(path, 'read') else io.BytesIO(Path(path).read_bytes())
        with np.load(raw, allow_pickle=False) as archive:
            state = json.loads(archive['metadata_json'].tobytes())
            if state['schema'] != 'n2p3_joint_flat_model/1':
                raise ValueError('unsupported actual joint prefix model schema')
            preparation = source_preparation or FrozenJointFlatSourcePreparation()
            source = preparation.prepare(archive['source_bytes'].tobytes(), expected_sha256=state['source_sha256'])
            calibration = (None if state['calibration_sampling'] is None else
                _restore_sampling(state['calibration_sampling'], archive['C_coefficients'], archive['C_precisions'], archive['C_targets']))
        result = cls(source, JointFlatProgram.from_record(state['programme']), calibration,
            tuple(state['calibration_record_ids']), tuple(freeze_metadata(row) for row in state['calibration_inputs']),
            freeze_metadata(state['provenance']))
        if result.digest() != state['model_sha256']:
            raise ValueError('saved original joint calibration model changed its numerical state')
        return result


def _normalizer_error(noise, prefix, stop, relative_shape):
    from research.channel_noise_mechanism import _roundoff_gamma

    logs = np.log(np.diag(noise._spatial_root))
    tail_native = stop-prefix
    magnitude = 2*tail_native*float(np.abs(logs).sum())
    magnitude += tail_native*len(logs)*abs(np.log(relative_shape))
    return _roundoff_gamma(len(logs)+5)*magnitude
