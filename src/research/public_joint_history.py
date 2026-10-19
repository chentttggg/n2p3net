"""New joint-fit issuer + original public learner -> complete typed history."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.collapsed_continuous_prediction import InferenceQuality
from research.frozen_metadata import freeze_metadata, plain_metadata
from research.joint_history_inference import (
    load_log_record,
    replay_joint_history_smc,
    save_log_record,
)
from research.joint_history_target import ConstrainedGaussianTable, RaoBlackwellHistoryTarget
from research.latent_response_reference import ResponseState
from research.native_joint_sufficient import IMPLEMENTATION_SHA256 as NATIVE_COMPILER_SHA256
from research.native_joint_sufficient import NativeJointRecordCompiler, original_white_components
from research.public_joint_response_noise import PublicCTemporalSelection
from research.public_native_learner_plan import require_public_native_learner
from research.public_source_context import require_public_source_context
from transfer.noise_model import array_digest, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_TARGETS, _VERIFIED = WeakKeyDictionary(), WeakKeyDictionary()


def _require_fit(fit):
    from research.public_joint_response_noise import require_public_joint_fit

    return require_public_joint_fit(fit)


def _fit_programme_relation(fit, programme):
    fit, programme = _require_fit(fit), require_public_native_learner(programme)
    if (fit.source_admission is not programme.source_admission or fit.prefix is not programme.prefix
            or len(fit.calibration_inputs) != len(programme.calibration_inputs)
            or any(a is not b for a, b in zip(fit.calibration_inputs, programme.calibration_inputs, strict=True))):
        raise ValueError('frozen joint parameters and history share their genuine source and complete same-prefix C')
    context = require_public_source_context(programme.source_context)
    if context.admission is not fit.source_admission or array_digest(
            context.source_chart.mean, context.source_chart.factor) != array_digest(fit.source.mean, fit.source.factor):
        raise ValueError('history consumes the original complete frozen shared source law')
    return fit, programme


def _endpoint_records(programme):
    history = programme.history_input.public_binding
    result = {'P': {}, 'F': {}}
    for record in programme.records:
        name = record.cap.public_binding['record_id']
        full = len(record.samples)
        prefix = full
        if name == history['record_id']:
            c_stop = max([0, *[len(cap.samples) for cap in record.inputs
                if cap.public_binding['role'] == 'calibration']])
            prefix = max(c_stop, history['prefix_stop'])
        result['P'][name], result['F'][name] = prefix, full
    # P and F must use the same original flat measures. A zero-row physical
    # group has no finite flat integral; F remains the primary whole target.
    if any(count < 2 for count in result['P'].values()):
        result.pop('P')
    return result


def _native_g0_g1_callback(selection, selected, basis, native_samples, source_context_identity):
    """Same empirical accumulator as actual C fitting; original membership owner."""
    assignment = selection.law.exposure_assignments
    unknown = (None if assignment is None else assignment.assignments_by_state_id[
        selected[assignment.variable_id].q])
    q = selected[selection.q_variable].q
    events = []
    for event in selection.plan.exposures:
        if event['block_id'] == selection.selection_id:
            bit, _ = selection.plan.target_indicator(event, q, unknown)
            events.append((event['event_id'], event['native_sample'], int(bit)))
    return PublicCTemporalSelection(basis, tuple(events), selection.selection_id, native_samples,
        freeze_metadata({'schema': 'original_public_G0_G1_native_history_empirical_design/1',
            'source_context': source_context_identity, 'original_selection_operator': selection.identity,
            'original_plan': selection.plan.identity, 'original_numeric_binary_rows': events,
            'selected_state': {name: vars(selected[name]) for name in selection.scope},
            'arithmetic_owner': 'same_PublicCTemporalSelection_as_joint_C_fit',
            'new_C_raw_or_label_grant': False}))


def build_public_joint_history_target(fit, programme, *, block_native_rows=2048, progress=None, numeric_store=None):
    fit, programme = _fit_programme_relation(fit, programme)
    nominal = (ResponseState(1., 1., 0., 1., 1.),)
    if fit.arm not in {'G0', 'G1'} or any(law.response_states != nominal for law in programme.law_by_selection.values()):
        raise ValueError('this G0/G1 functional-response programme uses its declared nominal shape; nonlinear W is a separate family')
    positions = {roster.selection_id: i for i, roster in enumerate(programme.rosters)}
    current = positions[programme.current_view.decision_id]
    default = {roster.selection_id: roster.states[0] for roster in programme.rosters}
    endpoints, tables, drift_ids, layout = _endpoint_records(programme), {}, {}, {}
    native_recipes = {}
    all_selections = {selection.selection_id: selection for selection in programme.selections}
    for endpoint, record_counts in endpoints.items():
        endpoint_tables, endpoint_layout = [], []
        for record in programme.records:
            name = record.cap.public_binding['record_id']
            stop = record_counts[name]
            samples = record.samples[:stop]
            selections = {key: value for key, value in all_selections.items() if value.record is record}
            supports = {}
            for key, selection in selections.items():
                begin, end = selection.support()['raw_native_hull']
                supports[key] = min(begin, stop), min(end, stop)
            specs = original_white_components(supports, native_samples=stop,
                conditioning_samples=fit.noise.conditioning_samples)
            for component_index, spec in enumerate(specs):
                active = tuple(key for key in spec.selection_ids if supports[key][0] < supports[key][1])
                scope = tuple(sorted({positions[variable] for key in active for variable in selections[key].scope}))
                entries = {}
                original_constraints = tuple(factor for factor in programme.C_label_factors
                    if set(factor.scope) <= set(scope))
                keys = product(*(programme.rosters[index].states for index in scope)) if scope else ((),)
                for key in keys:
                    selected = {**default, **{programme.rosters[i].selection_id: state
                        for i, state in zip(scope, key, strict=True)}}
                    zero = any(np.isneginf(factor.values[tuple(programme.rosters[i].states.index(
                        selected[programme.rosters[i].selection_id]) for i in factor.scope)])
                        for factor in original_constraints)
                    if zero:
                        entries[key] = None
                        continue
                    callbacks = {selection_id: _native_g0_g1_callback(selections[selection_id], selected,
                        fit.basis, stop, programme.source_context.identity)
                        for selection_id in spec.selection_ids}
                    compiler = NativeJointRecordCompiler(samples, fit.source.mean.shape[0], callbacks, supports,
                        fit.noise.geometry, record_id=name, drift_group_id=name,
                        branch_binding={'public_programme': programme.identity, 'new_joint_fit': fit.identity,
                            'original_union': record.identity, 'original_cap': record.cap.identity,
                            'endpoint': endpoint, 'logical_native_rows': [0, stop],
                            'local_scope': scope, 'selected_state': {key_: vars(value) for key_, value in selected.items()
                                if key_ in {programme.rosters[i].selection_id for i in scope}},
                            'zero_support_selections_integrated_full_prior_once': [key_ for key_ in spec.selection_ids if key_ not in active]},
                        block_native_rows=block_native_rows, conditioning_samples=fit.noise.conditioning_samples,
                        component_indices=(component_index,))
                    reduction = compiler.compile_component(fit.noise, fit.within_factor, fit.source, component_index)
                    entries[key] = reduction.potential
                    if numeric_store is not None:
                        from research.native_joint_numeric_store import store_native_component

                        native_recipes[reduction.potential.identity] = store_native_component(compiler,
                            component_index, fit.noise, fit.within_factor, fit.source, reduction.potential, numeric_store)
                    drift_ids[name] = compiler.drift_ids
                    if progress is not None:
                        progress(endpoint, name, component_index, len(entries))
                if scope:
                    table = ConstrainedGaussianTable(programme.rosters, scope, entries,
                        exact_C_zero_binding={'programme': programme.identity,
                            'original_C_factor_scopes': [factor.scope for factor in original_constraints]})
                else:
                    table = entries[()]
                endpoint_tables.append(table)
                endpoint_layout.append({'record': name, 'native_intervals': spec.native_intervals,
                    'complete_local_selection_ids': spec.selection_ids, 'active_discrete_selection_ids': active,
                    'actual_scope': [programme.rosters[i].selection_id for i in scope],
                    'full_local_table_entries': len(entries), 'exact_C_zero_entries_not_compiled': sum(value is None for value in entries.values())})
        tables[endpoint], layout[endpoint] = tuple(endpoint_tables), endpoint_layout
    target = RaoBlackwellHistoryTarget(fit.source, programme.rosters, current,
        programme.current_view.decision.candidate_vocabulary, programme.C_label_factors, tables,
        endpoints, drift_ids, factory_identity=metadata_digest({'native_compiler': NATIVE_COMPILER_SHA256,
            'new_joint_fit': fit.identity}), input_binding={'public_programme': programme.identity,
                'frozen_joint_fit': fit.identity, 'raw_component_layout': layout,
                'typed_source_context': programme.source_context.identity,
                'source_context_record': programme.source_context.record,
                'C_likelihood_use_count': 1, 'same_selection_local_across_C_P_F': True})
    if numeric_store is not None:
        target.native_component_recipes = freeze_metadata(native_recipes)
        target.native_numeric_store = str(numeric_store.directory)
    record = freeze_metadata({'schema': 'live_public_joint_fit_complete_history_relation/1',
        'new_joint_fit': fit.identity, 'public_programme': programme.identity, 'target': target.identity,
        'original_C_inputs': [cap.identity for cap in fit.calibration_inputs],
        'history_input': programme.history_input.identity, 'source_prior_count': 1,
        'C_response_likelihood_count': 1, 'each_physical_record_union_use_count': 1,
        'physical_record_count': len(programme.records),
        'response_model_family': 'G0_G1_shared_and_selection_functional_Gaussian_nominal_shape',
        'component_layout': layout, 'frontier_layout': target.frontier_layout,
        'actual_numeric_snapshot': metadata_digest(target.numeric_snapshot()),
        'old_fixed_mean_noise_relabelled': False, 'producer': IMPLEMENTATION_SHA256})
    _TARGETS[target] = fit, programme, record
    return target, record


def require_public_joint_history_target(target):
    owned = _TARGETS.get(target)
    if owned is None:
        raise TypeError('typed history output requires its original registered joint-fit/public target')
    fit, programme, record = owned
    _fit_programme_relation(fit, programme)
    if target.identity != record['target']:
        raise ValueError('actual joint history numerical target changed after original publication')
    if metadata_digest(target.numeric_snapshot()) != record['actual_numeric_snapshot']:
        raise ValueError('joint history scientific operands changed after original target admission')
    return fit, programme, record


@dataclass(frozen=True, eq=False)
class VerifiedJointHistoryInference:
    result: dict
    admission: dict


def admit_joint_history_inference(target, path, *, expected_sha256):
    result = load_log_record(path, expected_sha256=expected_sha256)
    replay = replay_joint_history_smc(target, result)
    admission = freeze_metadata({'result_sha256': expected_sha256, 'target': target.identity,
        'actual_model_and_common_cloud_replay': replay, 'producer': IMPLEMENTATION_SHA256})
    value = VerifiedJointHistoryInference(freeze_metadata(result), admission)
    _VERIFIED[value] = admission
    return value


@dataclass(frozen=True)
class PublicJointHistoryPrediction:
    evidence: CandidateEvidence
    quality: InferenceQuality
    scope: dict

    def record(self):
        return {'schema': 'public_joint_history_typed_prediction/1', 'evidence': self.evidence.record(),
            'quality': asdict(self.quality), 'scope': plain_metadata(self.scope),
            'cold_record_grants_live_raw_or_fit_authority': False}

    def save(self, path):
        return save_log_record(path, self.record())

    @classmethod
    def load(cls, path, *, expected_sha256):
        record = load_log_record(path, expected_sha256=expected_sha256)
        if record['schema'] != 'public_joint_history_typed_prediction/1' or record['cold_record_grants_live_raw_or_fit_authority'] is not False:
            raise ValueError('public joint-history prediction requires its explicit numeric-only saved role')
        quality = dict(record['quality'])
        if quality['all_candidate_log_se'] is not None:
            quality['all_candidate_log_se'] = tuple(quality['all_candidate_log_se'])
        return cls(CandidateEvidence.from_record(record['evidence']), InferenceQuality(**quality), freeze_metadata(record['scope']))


def publish_public_joint_history(verified, target, *, quality_policy, scoring_law='whole'):
    if type(verified) is not VerifiedJointHistoryInference or _VERIFIED.get(verified) is not verified.admission:
        raise TypeError('typed public prediction requires actual model/common-cloud replay admission')
    fit, programme, relation = require_public_joint_history_target(target)
    result, view = verified.result, programme.current_view
    if (result['target_identity'] != target.identity or scoring_law not in {'whole', 'conditional'}
            or tuple(result['q_values']) != tuple(view.decision.candidate_vocabulary)):
        raise ValueError('typed joint-history output retains its actual current full candidate request')
    if result['quality_kind'] not in quality_policy.allowed_estimate_kinds:
        raise ValueError('the frozen inference quality policy does not admit this estimate kind')
    decision = view.decision
    expected = tuple(map(int, decision.membership.sum(axis=0)))
    exposures = {event['event_id']: event for event in programme.history_input.public_binding[
        'public_physical_history']['exposures'] if event['block_id'] == view.decision_id}
    observed = np.zeros(len(target.q_values), dtype=int)
    for i, event_id in enumerate(decision.event_ids):
        event = exposures.get(event_id)
        if event is not None:
            if (event['native_sample'] != view.raw_scope[0]+int(view.onsets[i])
                    or tuple(event['membership']) != tuple(decision.membership[i])):
                raise ValueError('typed candidate coverage changed the original physical event schedule')
            observed += decision.membership[i]
    expert = CandidateExpertIdentity(expert_id=f'joint-history-{fit.arm}',
        numerical_input_digest=metadata_digest({'target': target.identity, 'programme': programme.identity,
            'joint_fit': fit.identity, 'scoring_law': scoring_law}), model_digest=target.identity,
        source_digest=programme.source_context.identity, fit_digest=fit.identity,
        score_kind=f'public_joint_history/{scoring_law}/1', probability_calibrated=False)
    identity = {'candidate_vocabulary': target.q_values, 'participant_key': view.participant_key,
        'decision_id': view.decision_id, 'physical_support_digest': view.physical_support_digest,
        'expert_identities': (expert,), 'evidence_level': decision.repetitions,
        'expected_event_counts': expected, 'observed_event_counts': tuple(map(int, observed))}
    key = f'{scoring_law}_log_probability'
    if (not decision.schedule_complete or tuple(observed) != expected or key not in result['probabilities']):
        evidence = CandidateEvidence(log_probability=None, failure_kind='incomplete',
            failure_reason='original_full_schedule_or_requested_integral_incomplete', **identity)
    else:
        evidence = CandidateEvidence(log_probability=np.array(result['probabilities'][key]), **identity)
    se = result.get('estimated_mc_error', {}).get(f'{scoring_law}_log_probability_se')
    precision = None
    if se is not None and quality_policy.maximum_replicate_log_se is not None:
        positive = [value for value, zero in zip(se, result['exact_zero_C_mass_mask'], strict=True) if not zero]
        precision = bool(positive and np.isfinite(positive).all() and max(positive) <= quality_policy.maximum_replicate_log_se)
    quality = InferenceQuality(result['quality_kind'], result['numerical_error_role'],
        None if se is None else tuple(se), result['finite_sample_certificate'],
        quality_policy.policy_id, quality_policy.digest(), precision)
    scope = freeze_metadata({'decision_unit': 'original_public_selection',
        'evaluation_role': 'public_native_development', 'current_view': view.record(),
        'actual_source_C_history_fit_relation': relation, 'inference_admission': verified.admission,
        'actual_K': programme.record['actual_K'],
        'original_requested_to_actual_C_budgets': programme.record['original_requested_to_actual_C_budgets'],
        'requested_Ks_for_this_actual_prefix': programme.record['requested_Ks_for_this_actual_prefix'],
        'source_digest_role': 'complete_typed_live_source_context_identity',
        'scoring_law': scoring_law, 'primary_score': 'normalize_J_F',
        'exact_zero_C_mass_mask': result['exact_zero_C_mass_mask'],
        'conditional_support_measure': result['conditional_support_measure'],
        'P_F_log_ratio_on_C_incompatible_candidates': result['P_F_log_ratio_on_C_incompatible_candidates'],
        'zero_axis_probability_is_exact_constraint': True,
        'producer': IMPLEMENTATION_SHA256})
    return PublicJointHistoryPrediction(evidence, quality, scope)
