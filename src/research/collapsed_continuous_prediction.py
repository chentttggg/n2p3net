"""Independent typed adapter for the continuous programme and original raw cap.

Numerical source/C admission stays with the caller's original model owner. This
layer joins that model's actual operands to the registered truth-free raw scope,
preserves complete candidate counts and carries MC quality outside CandidateEvidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.brainsync_raw_authority import RawCapability, require_raw_capability
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.noise_model import metadata_digest

_VERIFIED_INFERENCES = WeakKeyDictionary()


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class VerifiedContinuousInference:
    result: dict
    admission: dict


def admit_continuous_inference(target, path, *, expected_sha256):
    """Once per publication: external artifact bytes + actual model-only replay."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('continuous result differs from externally admitted producer bytes')
    result = json.loads(content)
    if result['target_identity'] != target.identity:
        raise ValueError('inference artifact requires its actual compiled numerical model')
    if result['schema'] in {'n2p3_exact_finite_prepared_gaussian_integral/1',
                            'n2p3_exact_finite_prepared_gaussian_integral/2'}:
        from research.finite_quadratic_integration import replay_finite_model

        replay = replay_finite_model(target, result)
        cloud_sha = None
    else:
        from research.prepared_continuous_frontier import replay_model_cloud

        replay = replay_model_cloud(target, result)
        cloud_sha = result['saved_cloud']['sha256']
    admission = freeze_metadata({'result_sha256': expected_sha256, 'cloud_sha256': cloud_sha,
        'target_identity': target.identity, 'actual_model_only_replay': replay,
        'admission_source_sha256': _digest_adapter()})
    verified = VerifiedContinuousInference(freeze_metadata(result), admission)
    _VERIFIED_INFERENCES[verified] = admission
    return verified


def _verified_result(verified):
    admission = _VERIFIED_INFERENCES.get(verified) if type(verified) is VerifiedContinuousInference else None
    if admission is None or admission is not verified.admission:
        raise ValueError('typed publication requires original registered inference/model replay admission')
    return verified.result


@dataclass(frozen=True)
class InferenceQualityPolicy:
    policy_id: str
    allowed_estimate_kinds: tuple[str, ...]
    maximum_replicate_log_se: float | None

    def digest(self):
        return metadata_digest(asdict(self))


@dataclass(frozen=True)
class InferenceQuality:
    estimate_kind: str
    numerical_role: str
    all_candidate_log_se: tuple[float, ...] | None
    finite_sample_certificate: bool
    policy_id: str
    policy_digest: str
    declared_precision_target_met: bool | None


@dataclass(frozen=True)
class CollapsedContinuousPrediction:
    evidence: CandidateEvidence
    quality: InferenceQuality
    scoring_law: str
    raw_capability_identity: str
    numerical_model_identity: str
    inference_artifact_identity: str
    adapter_source_sha256: str
    decision_scope: dict
    inference_admission: dict

    def record(self):
        return {'schema': 'n2p3_collapsed_continuous_typed_prediction/2',
            'evidence': self.evidence.record(), 'quality': asdict(self.quality),
            'scoring_law': self.scoring_law, 'raw_capability_identity': self.raw_capability_identity,
            'numerical_model_identity': self.numerical_model_identity,
            'inference_artifact_identity': self.inference_artifact_identity,
            'adapter_source_sha256': self.adapter_source_sha256, 'decision_scope': self.decision_scope,
            'inference_admission': plain_metadata(self.inference_admission)}

    @classmethod
    def from_record(cls, record):
        if record['schema'] != 'n2p3_collapsed_continuous_typed_prediction/2':
            raise ValueError('typed prediction requires its explicit original session decision unit')
        quality = dict(record['quality'])
        if quality['all_candidate_log_se'] is not None:
            quality['all_candidate_log_se'] = tuple(quality['all_candidate_log_se'])
        return cls(CandidateEvidence.from_record(record['evidence']), InferenceQuality(**quality),
            record['scoring_law'], record['raw_capability_identity'], record['numerical_model_identity'],
            record['inference_artifact_identity'], record['adapter_source_sha256'], record['decision_scope'],
            record['inference_admission'])

    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.record(), sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, *, expected_sha256):
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError('typed continuous prediction changed its externally admitted bytes')
        return cls.from_record(json.loads(content))


def _caps(raw_capabilities):
    caps = (raw_capabilities,) if type(raw_capabilities) is RawCapability else tuple(raw_capabilities)
    return tuple(sorted((require_raw_capability(cap) for cap in caps), key=lambda c: c.public_binding['segment_id']))


def numerical_input_digest(original_target, raw_capabilities, scoring_law):
    caps = _caps(raw_capabilities)
    return metadata_digest({'raw_authority': [cap.identity_sha256 for cap in caps],
        'original_numerical_records': [record.model.identity for record in original_target.records],
        'law': scoring_law, 'programme': 'collapsed_continuous_full_history/1'})


def publish_continuous_prediction(verified_inference, original_target, prepared_target, raw_capabilities, *,
                                  scoring_law, expert_identity, quality_policy, evaluation_role,
                                  decision_unit='session_target_lock', block_id=None):
    """Publish full-J probabilities; MC precision flags never delete numeric estimates."""
    result = _verified_result(verified_inference)
    caps = _caps(raw_capabilities)
    if not caps:
        raise ValueError('an original session decision requires registered original raw segments')
    cap = caps[0]
    binding = cap.public_binding
    if decision_unit != 'session_target_lock' or block_id is not None:
        raise ValueError('v4 session target-lock is the decision unit; individual blocks require a separate query contract')
    if evaluation_role != 'later_independent_recording' or any(c.public_binding['role'] != 'inference' for c in caps):
        raise ValueError('this product prediction owner consumes original independent inference-recording caps')
    public_keys = ('participant_id', 'session_id', 'record_id', 'target_lock_group',
        'sample_rate_hz', 'signal_unit', 'eeg_channels', 'reference', 'operator',
        'public_protocol', 'independent_recording_scope_sha256')

    def public_programme(local):
        return {**{k: local[k] for k in public_keys}, 'requested_blocks': [
            {key: row[key] for key in ('block_id', 'requested_flash_count', 'requested_repetitions')}
            for row in local['block_requests']]}

    first_public = metadata_digest(public_programme(binding))
    if (len({c.public_binding['segment_id'] for c in caps}) != len(caps)
            or any(metadata_digest(public_programme(c.public_binding)) != first_public for c in caps)):
        raise ValueError('all contributed segments must belong to one original session/participant/target-lock programme')
    if scoring_law not in {'whole', 'conditional'}:
        raise ValueError('the original whole/conditional law must be explicit')
    if result['target_identity'] != prepared_target.identity:
        raise ValueError('inference result differs from its actual numerical producer')
    if tuple(result['q_values']) != cap.candidate_vocabulary or tuple(prepared_target.q_values) != cap.candidate_vocabulary:
        raise ValueError('typed output must retain the original complete current vocabulary/order')
    if len(original_target.records) != len(prepared_target.records):
        raise ValueError('typed parent changed its full original record set')
    matched = {}
    by_record = {c.public_binding['record_segment_id']: c for c in caps}
    for record, (compiled, p, f) in zip(original_target.records, prepared_target.records, strict=True):
        model = record.model
        if (compiled.base_identity != model.identity or p != record.prefix_samples or f != record.full_samples):
            raise ValueError('compiled model changed original raw/source/C operands')
        current_cap = by_record.get(model.record_id)
        if current_cap is None:
            if any(s.candidate_variable == binding['target_lock_group'] for s in model.selections):
                raise ValueError('every record contributing current-target evidence requires its original raw capability')
            continue
        local = current_cap.public_binding
        geometry = model.carrier.noise.geometry
        if (model.origin != local['sample_start'] or not np.array_equal(model.y, current_cap.eeg)
                or model.dictionary.sample_rate_hz != local['sample_rate_hz']
                or geometry.channels != tuple(local['eeg_channels']) or geometry.unit != local['signal_unit']
                or geometry.reference != local['reference'] or geometry.operator_identity != local['operator']
                or model.variable_ids[model.current] != local['target_lock_group']
                or p+model.origin != local['prefix_stop'] or f+model.origin != local['full_stop']):
            raise ValueError('numerical current record differs from original seven-EEG physical raw scope')
        by_sample = {event.sample: (index, event) for index, event in enumerate(current_cap.events)}
        represented = []
        for selection in model.selections:
            if selection.candidate_variable != local['target_lock_group']:
                raise ValueError('held BrainSync session responses must use its public target-lock group')
            if set(selection.membership.candidate_vocabulary) != set(current_cap.candidate_vocabulary):
                raise ValueError('numerical membership must preserve all original candidate identities')
            columns = [selection.membership.candidate_vocabulary.index(q) for q in current_cap.candidate_vocabulary]
            for row, onset in enumerate(selection.onsets):
                entry = by_sample.get(int(onset))
                if (entry is None or selection.membership.event_ids[row] != entry[1].event_id
                        or not np.array_equal(selection.membership.membership[row, columns], current_cap.membership[entry[0]])):
                    raise ValueError('numerical membership/onset differs from original public events')
                if onset < local['full_stop']:
                    represented.append(int(onset))
        observed_events = [e.sample for e in current_cap.events if e.sample < local['full_stop']]
        if sorted(represented) != sorted(observed_events):
            raise ValueError('numerical current history omitted or duplicated original observed events')
        if model.record_id in matched:
            raise ValueError('an original raw segment cannot contribute twice')
        matched[model.record_id] = model
    if set(matched) != set(by_record):
        raise ValueError('registered raw segments must each match their actual original numerical parent record')
    if (not isinstance(expert_identity, CandidateExpertIdentity)
            or expert_identity.model_digest != prepared_target.identity
            or expert_identity.numerical_input_digest != numerical_input_digest(original_target, caps, scoring_law)
            or expert_identity.score_kind != f'collapsed_continuous_full_history/{scoring_law}/1'
            or expert_identity.probability_calibrated):
        raise ValueError('new programme requires its own original caller expert identity')
    repetitions = sum(row['requested_repetitions'] for row in binding['block_requests'])
    expected = (repetitions,)*len(cap.candidate_vocabulary)
    observed_counts, event_ids = np.zeros(len(cap.candidate_vocabulary), dtype=int), set()
    for current_cap in caps:
        indices = [i for i, event in enumerate(current_cap.events)
            if event.sample < current_cap.public_binding['full_stop']]
        for index in indices:
            if current_cap.events[index].event_id in event_ids:
                raise ValueError('independent segment observations duplicated an original session event')
            event_ids.add(current_cap.events[index].event_id)
        observed_counts += current_cap.membership[indices].sum(axis=0)
    counts = tuple(map(int, observed_counts))
    scope = {'decision_unit': decision_unit, 'evaluation_role': evaluation_role,
        'target_lock_group': binding['target_lock_group'], 'session_id': binding['session_id'],
        'requested_block_ids': [row['block_id'] for row in binding['block_requests']],
        'expected_total_events': sum(row['requested_flash_count'] for row in binding['block_requests']),
        'contributing_segments': [c.public_binding['record_segment_id'] for c in caps],
        'raw_capability_identities': [c.identity_sha256 for c in caps]}
    physical_digest = metadata_digest(scope)
    key = f'{scoring_law}_log_probability'
    identity = {'candidate_vocabulary': cap.candidate_vocabulary,
        'participant_key': binding['participant_id'],
        'decision_id': binding['target_lock_group'],
        'physical_support_digest': physical_digest, 'expert_identities': (expert_identity,),
        'evidence_level': repetitions, 'expected_event_counts': expected, 'observed_event_counts': counts}
    estimate_kind = result['quality_kind']
    if estimate_kind not in quality_policy.allowed_estimate_kinds:
        raise ValueError('the frozen inference quality policy does not admit this estimator kind')
    se = result.get('estimated_mc_error', {}).get(key.replace('_log_probability', '_log_probability_se'))
    if key not in result['probabilities']:
        evidence = CandidateEvidence(log_probability=None, failure_kind='not_available',
            failure_reason=f'{scoring_law}_integral_unavailable', **identity)
    else:
        values = np.array(result['probabilities'][key])
        if values.shape != (len(expected),) or not np.isfinite(values).all():
            raise ValueError('Gaussian continuous inference must preserve every finite current-q log mass')
        # Stabilize only the final complete vector's floating normalization. The
        # original prior already entered J once; no prior/temperature change.
        evidence = CandidateEvidence.from_log_evidence(values, temperature=1., prior=None, **identity)
    precision_met = None
    if estimate_kind == 'replicated_mc_estimate' and quality_policy.maximum_replicate_log_se is not None:
        precision_met = se is not None and len(se) == len(expected) and bool(
            np.isfinite(se).all() and np.max(se) <= quality_policy.maximum_replicate_log_se)
    quality = InferenceQuality(estimate_kind, result['numerical_error_role'],
        None if se is None else tuple(map(float, se)), result['finite_sample_certificate'],
        quality_policy.policy_id, quality_policy.digest(), precision_met)
    return CollapsedContinuousPrediction(evidence, quality, scoring_law,
        metadata_digest([c.identity_sha256 for c in caps]), prepared_target.identity,
        verified_inference.admission['result_sha256'], _digest_adapter(), scope, verified_inference.admission)


def _digest_adapter():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
