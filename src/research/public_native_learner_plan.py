"""One original public C/history union and complete q/shape/identity law.

This live parent compiles no EEG, fits no parameter, and grants no new reads.
The numeric plan retains all candidate axes, including exact zero-likelihood
states established by original C bits. Actual clocks are rejoined by selection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary

import numpy as np

from research.frozen_metadata import freeze_metadata, plain_metadata
from research.latent_response_reference import ResponseState, _admit_law
from research.physical_operator_binding import clock_operator_inputs, plan_operator_inputs
from research.physical_response_design import PhysicalResponsePlan
from research.public_record_physical_bridge import (
    PublicNativeClockChart,
    admit_public_history,
    observation_interval_union,
    require_public_physical_input,
)
from research.public_source_context import public_source_context, require_public_source_context
from research.response_frontier import LogFactor, Roster, StateKey
from research.scoped_physical_response import DeclaredExposureAssignments, ScopedPhysicalResponse
from transfer.binary import calibration_budget_requests
from transfer.ensemble_views import calibration_prefix_identity
from transfer.noise_model import array_digest, frozen_array, metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PROGRAMMES = WeakKeyDictionary()


@dataclass(frozen=True)
class PublicSelectionLaw:
    """Predeclared positive priors; q, response shape and unknown ID stay distinct."""

    candidate_vocabulary: tuple
    candidate_mass: tuple
    response_states: tuple
    exposure_assignments: DeclaredExposureAssignments | None = None

    def __post_init__(self):
        vocabulary, mass = tuple(self.candidate_vocabulary), np.asarray(self.candidate_mass, dtype=float)
        if (len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary)
                or mass.shape != (len(vocabulary),) or not np.isfinite(mass).all()
                or np.any(mass <= 0) or not np.isclose(mass.sum(), 1., rtol=0, atol=2e-15)):
            raise ValueError('declare every original candidate and its normalized positive prior mass')
        object.__setattr__(self, 'candidate_vocabulary', vocabulary)
        object.__setattr__(self, 'candidate_mass', tuple(map(float, mass)))
        if any(type(state) is not ResponseState for state in self.response_states):
            raise TypeError('the public response law uses the original immutable ResponseState values')
        object.__setattr__(self, 'response_states', _admit_law(self.response_states))
        assignment = self.exposure_assignments
        if assignment is not None:
            owned = MappingProxyType({state: MappingProxyType(dict(values))
                for state, values in assignment.assignments_by_state_id.items()})
            object.__setattr__(self, 'exposure_assignments', DeclaredExposureAssignments(
                assignment.variable_id, assignment.roster, owned, assignment.independent_of_current_target))

    def record(self):
        assignment = self.exposure_assignments
        return {'candidate_vocabulary': self.candidate_vocabulary, 'candidate_mass': self.candidate_mass,
            'response_states': [vars(state) for state in self.response_states],
            'unknown_joint_identity': None if assignment is None else {
                'variable_id': assignment.variable_id,
                'states': [vars(state) for state in assignment.roster.states],
                'log_mass': assignment.roster.log_mass.tolist(),
                'assignments': [{'state_id': state, 'events': dict(values)}
                    for state, values in assignment.assignments_by_state_id.items()],
                'independent_of_current_target': assignment.independent_of_current_target}}


def _original_record_key(cap):
    b = cap.public_binding
    return tuple(b[name] for name in ('dataset_id', 'participant_key', 'record_id', 'source_sha256'))


def _roster_record(roster):
    return {'selection_id': roster.selection_id, 'states': [vars(state) for state in roster.states],
            'log_mass': roster.log_mass.tolist()}


@dataclass(frozen=True)
class PublicUnionRecord:
    """The longest original live cap owns the actual union; no raw copy."""

    cap: object
    inputs: tuple
    label_rows: tuple
    identity: str

    @property
    def samples(self):
        return self.cap.samples


def original_record_unions(calibration_inputs, history_input):
    grouped = {}
    for cap in (*calibration_inputs, history_input):
        cap = require_public_physical_input(cap)
        grouped.setdefault(_original_record_key(cap), []).append(cap)
    result = []
    for key, inputs in grouped.items():
        cap = max(inputs, key=lambda item: item.public_binding['full_stop'])
        b = cap.public_binding
        if any(item.public_binding['sample_start'] != 0 for item in inputs):
            raise ValueError('the original public prefix issuer must own record-start observation unions')
        events = {event['event_id']: event for event in b['public_physical_history']['exposures']}
        labels = {}
        for item in inputs:
            other = item.public_binding
            if (not np.array_equal(item.samples, cap.samples[:other['full_stop']])
                    or any(metadata_digest(event) != metadata_digest(events.get(event['event_id']))
                        for event in other['public_physical_history']['exposures'])):
                raise ValueError('C/history overlap changed original samples or physical exposure operands')
            for row in item.label_rows:
                event_id, onset, bit = row
                event = events.get(event_id)
                if (event is None or onset != event['native_sample'] or bit not in (0, 1)
                        or (event_id in labels and labels[event_id] != row)):
                    raise ValueError('original C bits require one consistent observed exposure/native onset')
                labels[event_id] = row
        labels = tuple(labels.values())
        identity = metadata_digest({'original_record': key, 'union_support': [0, b['full_stop']],
            'union_raw_sha256': array_digest(cap.samples), 'longest_original_input': cap.identity,
            'original_inputs': [item.identity for item in inputs], 'original_C_label_rows': labels,
            'noise_startups': 1, 'physical_flat_drift_groups': 1})
        result.append(PublicUnionRecord(cap, tuple(inputs), labels, identity))
    return tuple(result)


class PublicTemporalSelection:
    """Actual original selection chart -> complete native temporal response D."""

    def __init__(self, record, selection_id, law, dictionary):
        self.record, self.selection_id, self.law = record, selection_id, law
        self.plan = PhysicalResponsePlan(record.cap.clock_for_selection(selection_id), dictionary,
            clock_program='native_sample_index',
            exposure_response_law='onset_impulse_with_actual_duration_provenance/1',
            candidate_vocabulary=law.candidate_vocabulary)
        self.q_variable, self.shape_variable = selection_id, selection_id + ':response_shape'
        self.response = ScopedPhysicalResponse(self.plan, selection_id, self.q_variable,
            self.shape_variable, law.response_states, law.exposure_assignments)
        self.scope = self.response.scope
        self.identity = metadata_digest({'union': record.identity, 'original_selection': selection_id,
            'actual_operator': plan_operator_inputs(self.plan), 'scope': self.scope, 'law': law.record(),
            'producer': IMPLEMENTATION_SHA256})

    def temporal(self, selected_states, *, native_start, native_stop):
        return self.response.design(selected_states, native_start=native_start, native_stop=native_stop)

    def support(self):
        return self.response.support(native_start=0, native_stop=len(self.record.samples))

    def label_factor(self, variable_positions):
        """Observed C labels are exact likelihoods, never another response fit."""
        labels = {event: bit for event, _, bit in self.record.label_rows}
        selected = [event for event in self.plan.exposures
            if event['block_id'] == self.selection_id and event['event_id'] in labels]
        if not selected:
            return None
        assignment = self.law.exposure_assignments
        assignment_states = (None,) if assignment is None else tuple(state.q for state in assignment.roster.states)
        values = np.zeros((len(self.law.candidate_vocabulary), len(assignment_states)))
        for iq, q in enumerate(self.law.candidate_vocabulary):
            for iz, state in enumerate(assignment_states):
                assignments = None if assignment is None else assignment.assignments_by_state_id[state]
                if any(self.plan.target_indicator(event, q, assignments)[0] != labels[event['event_id']]
                       for event in selected):
                    values[iq, iz] = -np.inf
        scope = (variable_positions[self.q_variable],)
        if assignment is None:
            values = values[:, 0]
        else:
            scope += (variable_positions[assignment.variable_id],)
            order = np.argsort(scope)
            values, scope = values.transpose(tuple(order)), tuple(scope[i] for i in order)
        if not np.isfinite(values).any():
            raise ValueError('original C labels and declared physical candidate/unknown law have no common support')
        return LogFactor(scope, frozen_array(values))


_SCIENTIFIC_METHODS = {
    'selection': {name: getattr(PublicTemporalSelection, name) for name in ('temporal', 'support', 'label_factor')},
    'response': {name: getattr(ScopedPhysicalResponse, name) for name in ('design', 'support')},
    'plan': {name: getattr(PhysicalResponsePlan, name) for name in ('design', 'support', 'exposure_basis', 'target_indicator')},
}


def _actual_scientific_methods(value, kind):
    """Bind the actual callbacks, while keeping ordinary numerical caches free."""
    result = []
    for name, implementation in _SCIENTIFIC_METHODS[kind].items():
        method = getattr(value, name)
        if getattr(method, '__self__', None) is not value or getattr(method, '__func__', None) is not implementation:
            raise ValueError('the public learner changed its actual numerical callback implementation')
        result.append({'method': name, 'implementation': [implementation.__module__, implementation.__qualname__]})
    return result


def _actual_assignment_inputs(assignment):
    if assignment is None:
        return None
    return {'variable_id': assignment.variable_id, 'roster': _roster_record(assignment.roster),
        'assignments': [{'state_id': state, 'events': dict(values)}
            for state, values in assignment.assignments_by_state_id.items()],
        'independent_of_current_target': assignment.independent_of_current_target}


def actual_public_numeric_state_inputs(selections, rosters, law_by_selection):
    """One pure numeric owner of the scientific inputs actually being consumed.

    Reads live law values, typed state IDs, priors, variable scopes and actual
    response/plan relations. It uses no cached selection/programme identity and
    no raw registry. Working caches are intentionally outside this snapshot.
    """
    selections, rosters, laws = tuple(selections), tuple(rosters), dict(law_by_selection)
    names = tuple(selection.selection_id for selection in selections)
    if len(set(names)) != len(names) or set(names) != set(laws):
        raise ValueError('the numeric state retains one declared law for every original selection')
    state = []
    for selection in selections:
        law, response, plan = selection.law, selection.response, selection.plan
        if (type(law) is not PublicSelectionLaw or type(laws[selection.selection_id]) is not PublicSelectionLaw
                or metadata_digest(law.record()) != metadata_digest(laws[selection.selection_id].record())):
            raise ValueError('the selection law differs from its declared original programme law')
        if (type(selection) is not PublicTemporalSelection or type(response) is not ScopedPhysicalResponse
                or type(plan) is not PhysicalResponsePlan or response.plan is not plan):
            raise ValueError('the actual public response callback is disconnected from its numerical plan')
        state.append({'selection_id': selection.selection_id, 'union_record': selection.record.identity,
            'law': law.record(), 'q_variable': selection.q_variable, 'shape_variable': selection.shape_variable,
            'scope': selection.scope, 'selection_callbacks': _actual_scientific_methods(selection, 'selection'),
            'actual_plan_operator_sha256': metadata_digest(plan_operator_inputs(plan)),
            'plan_callbacks': _actual_scientific_methods(plan, 'plan'),
            'actual_response': {'block_id': response.block_id, 'candidate_variable': response.candidate_variable,
                'shape_variable': response.shape_variable, 'scope': response.scope,
                'shapes': [vars(shape) for shape in response.shapes],
                'unknown_joint_identity': _actual_assignment_inputs(response.exposure_assignments),
                'callbacks': _actual_scientific_methods(response, 'response')}})
    return freeze_metadata({'schema': 'actual_public_learner_numeric_state_inputs/1',
        'rosters': [_roster_record(roster) for roster in rosters],
        'declared_selection_laws': {name: law.record() for name, law in laws.items()},
        'actual_selections': state})


def require_public_numeric_state_snapshot(programme):
    """Pure snapshot comparison, also called by the live original-parent gate."""
    actual = actual_public_numeric_state_inputs(programme.selections, programme.rosters, programme.law_by_selection)
    if metadata_digest(actual) != metadata_digest(programme.record['actual_numeric_state_inputs']):
        raise ValueError('the actual public learner scientific state changed after original admission')
    return actual


@dataclass(frozen=True, eq=False)
class PublicNativeLearnerProgramme:
    source_admission: object
    source_context: object
    prefix: object
    calibration_inputs: tuple
    history_input: object
    current_view: object
    records: tuple
    selections: tuple
    rosters: tuple
    C_label_factors: tuple
    law_by_selection: object
    record: dict
    identity: str

    def save_descriptor(self, path):
        """Small dependency-only state; cold reading it never issues live grants."""
        path = Path(path)
        payload = {'schema': 'public_native_complete_learner_dependencies/2', 'identity': self.identity,
            'programme': plain_metadata(self.record), 'numeric_only': True,
            'EEG_or_source_codec_embedded': False, 'raw_or_C_permission_granted': False}
        path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()


def admit_public_native_learner(source_admission, prefix, calibration_inputs, history_input,
                               current_view, *, law_by_selection):
    context = public_source_context(source_admission)
    admission = context.admission
    prefix = context.compiler.assert_original_prefix(prefix)
    if not any(prefix is item for item in context.prefixes):
        raise ValueError('complete learner C must be one exact native source-admitted prefix')
    prefix_identity = calibration_prefix_identity(prefix)
    inputs = tuple(require_public_physical_input(value) for value in calibration_inputs)
    expected = tuple(value for value in context.calibration_inputs
        if value.public_binding['C_prefix_identity'] == prefix_identity)
    if len(inputs) != len(expected) or any(a is not b for a, b in zip(inputs, expected, strict=True)):
        raise ValueError('complete learner retains every original authorized C record exactly once')
    history = require_public_physical_input(history_input)
    b = history.public_binding
    replayed_history = admit_public_history(context.compiler, prefix, current_view,
        conditioning_stop=b['prefix_stop'])
    if (b['role'] != 'history' or b['C_prefix_identity'] != prefix_identity
            or b['target_lock_group'] != current_view.decision_id
            or b['participant_key'] != current_view.participant_key
            or tuple(b['candidate_vocabulary']) != tuple(current_view.decision.candidate_vocabulary)
            or b['full_stop'] != current_view.available_sample
            or replayed_history.identity != history.identity or replayed_history.samples is not history.samples):
        raise ValueError('complete learner history requires its original current full candidate request')
    unions, laws = original_record_unions(inputs, history), dict(law_by_selection)
    selection_ids = tuple(selection['selection_id'] for record in unions
        for selection in record.cap.public_binding['public_physical_history']['selections'])
    if len(set(selection_ids)) != len(selection_ids) or set(laws) != set(selection_ids):
        raise ValueError('declare exactly one complete law for every occurred C/history character')
    selections, rosters = [], {}
    geometry = context.geometry
    for record in unions:
        rb = record.cap.public_binding
        if (tuple(rb['eeg_channels']), rb['sample_rate_hz'], rb['reference'], rb['signal_unit'], rb['operator']) != (
                geometry.channels, geometry.sfreq, geometry.reference, geometry.unit, geometry.operator_identity):
            raise ValueError('complete learner observations must use the original native source coordinates')
        for row in rb['public_physical_history']['selections']:
            name, law = row['selection_id'], laws[row['selection_id']]
            if type(law) is not PublicSelectionLaw or law.candidate_vocabulary != tuple(row['candidate_vocabulary']):
                raise ValueError('every character law retains its original ordered complete candidate axis')
            selection = PublicTemporalSelection(record, name, law, context.dictionary)
            selections.append(selection)
            local = (Roster(selection.q_variable, tuple(StateKey(q, 'target') for q in law.candidate_vocabulary),
                            np.log(law.candidate_mass)),
                Roster(selection.shape_variable, tuple(StateKey('shape', h) for h in range(len(law.response_states))),
                       np.log([state.probability for state in law.response_states])))
            if law.exposure_assignments is not None:
                local += (law.exposure_assignments.roster,)
            for roster in local:
                old = rosters.get(roster.selection_id)
                if old is not None and metadata_digest(_roster_record(old)) != metadata_digest(_roster_record(roster)):
                    raise ValueError('a shared unknown joint variable cannot change its original positive law')
                rosters[roster.selection_id] = roster
    rosters, selections = tuple(rosters.values()), tuple(selections)
    positions = {roster.selection_id: i for i, roster in enumerate(rosters)}
    factors = tuple(factor for selection in selections if (factor := selection.label_factor(positions)) is not None)
    numeric_state = actual_public_numeric_state_inputs(selections, rosters, laws)
    budget_requests = calibration_budget_requests(context.compiler.budgets,
        context.compiler.actual_budgets, context.compiler._adjustments)
    record = freeze_metadata({'schema': 'public_native_complete_learner_programme/2',
        'source_admission': metadata_digest(admission.record), 'source_context': context.identity,
        'source_context_record': context.record,
        'source_prior_count': 1, 'proper_source_latent_dimensions': context.source_chart.rank,
        'source_prior_sha256': array_digest(context.source_chart.mean.ravel(),
            context.source_chart.factor.reshape(context.source_chart.mean.size, context.source_chart.rank)),
        'C_prefix_identity': prefix_identity, 'actual_K': prefix.budget,
        'original_requested_to_actual_C_budgets': {str(key): value for key, value in budget_requests.items()},
        'requested_Ks_for_this_actual_prefix': [key for key, value in budget_requests.items() if value['actual_K'] == prefix.budget],
        'calibration_inputs': [item.identity for item in inputs], 'history_input': history.identity,
        'current_view': current_view.record(), 'current_selection_id': current_view.decision_id,
        'observation_union': observation_interval_union(tuple({'record_id': cap.public_binding['record_id'],
            'source_sha256': cap.public_binding['source_sha256'], 'read_sample_support': [0, len(cap.samples)]}
            for cap in inputs), {'record_id': b['record_id'], 'source_sha256': b['source_sha256'],
                                'read_sample_support': [0, len(history.samples)]}),
        'union_records': [{'identity': item.identity, 'original_inputs': [cap.identity for cap in item.inputs],
            'original_record': _original_record_key(item.cap), 'samples_shape': item.samples.shape,
            'sample_sha256': array_digest(item.samples), 'authorized_label_rows': item.label_rows,
            'noise_startup_count': 1, 'physical_flat_da_count': 1} for item in unions],
        'rosters': [_roster_record(roster) for roster in rosters],
        'selection_laws': {name: law.record() for name, law in laws.items()},
        'actual_selection_operators': [selection.identity for selection in selections],
        'actual_numeric_state_inputs': numeric_state,
        'C_response_likelihood_use_count': 1, 'C_label_factors': [{'scope': factor.scope,
            'allowed_state_mask': np.isfinite(factor.values).tolist()} for factor in factors],
        'score': 'normalize_J_F', 'current_prefix_candidate_evidence_retained': True,
        'original_inputs_are_live_authority': True, 'producer': IMPLEMENTATION_SHA256})
    result = PublicNativeLearnerProgramme(admission, context, prefix, inputs, history, current_view,
        unions, selections, rosters, factors, MappingProxyType(laws), record, metadata_digest(record))
    _PROGRAMMES[result] = record
    return result


def require_public_native_learner(programme):
    if type(programme) is not PublicNativeLearnerProgramme or _PROGRAMMES.get(programme) is not programme.record:
        raise TypeError('complete public learner requires its original live source/C/history union')
    require_public_numeric_state_snapshot(programme)
    context = require_public_source_context(programme.source_context)
    if context.admission is not programme.source_admission or context.identity != programme.record['source_context']:
        raise ValueError('public learner retains its actual typed source context and original issuer')
    for record in programme.records:
        for cap in record.inputs:
            require_public_physical_input(cap)
    dictionary = context.dictionary
    for selection in programme.selections:
        clock = selection.record.cap.clock_for_selection(selection.selection_id)
        expected = PhysicalResponsePlan(clock, dictionary, clock_program='native_sample_index',
            exposure_response_law='onset_impulse_with_actual_duration_provenance/1',
            candidate_vocabulary=selection.law.candidate_vocabulary)
        if (type(selection.plan.chart) is not PublicNativeClockChart
                or metadata_digest(clock_operator_inputs(selection.plan.chart)) != metadata_digest(clock_operator_inputs(clock))
                or metadata_digest(plan_operator_inputs(selection.plan)) != metadata_digest(plan_operator_inputs(expected))
                or selection.plan.identity != expected.identity):
            raise ValueError('public selection changed its actual original clock, membership or response operator')
        response = selection.response
        if (type(response) is not ScopedPhysicalResponse or response.plan is not selection.plan
                or response.block_id != selection.selection_id or response.candidate_variable != selection.q_variable
                or response.shape_variable != selection.shape_variable or response.shapes != selection.law.response_states
                or response.exposure_assignments is not selection.law.exposure_assignments
                or response.scope != selection.scope):
            raise ValueError('public selection changed the actual variables, state law or unknown identity operator')
    positions = {roster.selection_id: i for i, roster in enumerate(programme.rosters)}
    factors = tuple(factor for selection in programme.selections
        if (factor := selection.label_factor(positions)) is not None)
    if (len(factors) != len(programme.C_label_factors)
            or any(a.scope != b.scope or not np.array_equal(a.values, b.values)
                for a, b in zip(factors, programme.C_label_factors, strict=True))):
        raise ValueError('complete public learner changed its actual original C label likelihoods')
    return programme


def load_public_learner_descriptor(path, *, expected_sha256):
    """Replay the small numeric dependency record, with no live/raw admission."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('public learner descriptor changed its externally frozen bytes')
    record = json.loads(content)
    if (record['schema'] not in {'public_native_complete_learner_dependencies/1', 'public_native_complete_learner_dependencies/2'}
            or metadata_digest(record['programme']) != record['identity']
            or record['numeric_only'] is not True or record['raw_or_C_permission_granted'] is not False):
        raise ValueError('public learner descriptor changed its numerical dependency-only role')
    return freeze_metadata(record)
