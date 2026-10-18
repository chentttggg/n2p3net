"""A declared scope -> physical design callback, including joint exposure state.

Unknown digits are independent latent-variable *roles*, never shape aliases.
One assignment variable can enumerate jointly correlated exposure identities;
its explicit positive roster is inserted by the existing sum-product owner.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.physical_response_design import MissingExposureIdentityError


@dataclass(frozen=True)
class DeclaredExposureAssignments:
    variable_id: str
    roster: object
    assignments_by_state_id: dict
    independent_of_current_target: bool

    def __post_init__(self):
        if (not self.independent_of_current_target or self.variable_id != self.roster.selection_id
                or {s.q for s in self.roster.states} != set(self.assignments_by_state_id)):
            raise ValueError('declare the complete explicit target-independent exposure-assignment roster')


@dataclass(frozen=True)
class ScopedPhysicalResponse:
    plan: object
    block_id: int
    candidate_variable: str
    shape_variable: str
    shapes: tuple
    exposure_assignments: DeclaredExposureAssignments | None

    def __post_init__(self):
        unknown = [e for e in self.plan.exposures if e['block_id'] == self.block_id and e['membership'] is None]
        if unknown and self.exposure_assignments is None:
            raise MissingExposureIdentityError(unknown[0])
        if self.exposure_assignments is not None:
            for assignments in self.exposure_assignments.assignments_by_state_id.values():
                for event in unknown:
                    if assignments.get(event['event_id']) not in event['candidate_options']:
                        raise ValueError('every positive joint assignment retains each actual unknown exposure option')
        if len(set(self.scope)) != len(self.scope):
            raise ValueError('candidate, response shape and exposure identity are distinct variable roles')

    @property
    def scope(self):
        return ((self.candidate_variable, self.shape_variable) if self.exposure_assignments is None else
            (self.candidate_variable, self.shape_variable, self.exposure_assignments.variable_id))

    def design(self, selected_states, *, native_start, native_stop):
        q = selected_states[self.candidate_variable].q
        h = selected_states[self.shape_variable].shape
        assignments = (None if self.exposure_assignments is None else
            self.exposure_assignments.assignments_by_state_id[selected_states[self.exposure_assignments.variable_id].q])
        return self.plan.design(self.block_id, q, self.shapes[h], native_start=native_start,
            native_stop=native_stop, unknown_assignments=assignments)

    def support(self, *, native_start, native_stop):
        return self.plan.support(self.block_id, self.shapes, native_start=native_start, native_stop=native_stop)
