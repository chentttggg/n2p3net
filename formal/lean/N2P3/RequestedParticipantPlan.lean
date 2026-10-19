import N2P3.RequestedOutcomeCounts

/-!
# Requested participants versus planned decisions

Source: research/contracts.py:303-337 and candidate cross-decision runner
participant_operational_endpoints. This starts after identifier/schema/hash
parsing. It expands the actual uniqueness, disjointness and union gates.
Selection failures have no fabricated decision count and remain in the
requested-participant endpoint. No score, fit or cohort partition is an oracle.
-/

namespace N2P3.RequestedParticipantPlan

noncomputable section
attribute [local instance] Classical.propDecidable

def plannedParticipants {Participant Decision : Type} [DecidableEq Participant]
    (entries : List (Participant × Decision)) : Finset Participant := (entries.map Prod.fst).toFinset

inductive Fault where
  | invalidRequested | duplicateDecisions | duplicateSelectionFailures | overlap | unaccountedParticipant
  deriving DecidableEq

def validatePlan {Participant Decision : Type} [DecidableEq Participant] [DecidableEq Decision]
    (requested : List Participant) (entries : List (Participant × Decision)) (failures : List Participant) :
    Except Fault Unit :=
  if requested ≠ [] ∧ requested.Nodup then
    if entries.Nodup then
      if failures.Nodup then
        if Disjoint (plannedParticipants entries) failures.toFinset then
          if plannedParticipants entries ∪ failures.toFinset=requested.toFinset then .ok ()
          else .error .unaccountedParticipant
        else .error .overlap
      else .error .duplicateSelectionFailures
    else .error .duplicateDecisions
  else .error .invalidRequested

theorem checked_plan_iff_actual_source_gates {Participant Decision : Type}
    [DecidableEq Participant] [DecidableEq Decision]
    (requested : List Participant) (entries : List (Participant × Decision)) (failures : List Participant) :
    validatePlan requested entries failures=.ok () ↔
      (requested ≠ [] ∧ requested.Nodup) ∧ entries.Nodup ∧ failures.Nodup ∧
        Disjoint (plannedParticipants entries) failures.toFinset ∧
        plannedParticipants entries ∪ failures.toFinset=requested.toFinset := by
  unfold validatePlan
  split_ifs <;> simp_all

theorem checked_requested_participant_has_exactly_one_terminal_class {Participant Decision : Type}
    [DecidableEq Participant] [DecidableEq Decision]
    (requested : List Participant) (entries : List (Participant × Decision)) (failures : List Participant)
    (accepted : validatePlan requested entries failures=.ok ())
    (participant : Participant) (member : participant ∈ requested) :
    (participant ∈ plannedParticipants entries ∨ participant ∈ failures) ∧
      ¬(participant ∈ plannedParticipants entries ∧ participant ∈ failures) := by
  obtain ⟨_,_,_,disjoint,complete⟩ := (checked_plan_iff_actual_source_gates _ _ _).mp accepted
  have belongs : participant ∈ plannedParticipants entries ∪ failures.toFinset := by
    rw [complete]
    exact List.mem_toFinset.mpr member
  constructor
  · simpa using belongs
  · rintro ⟨planned,failed⟩
    exact Finset.disjoint_left.mp disjoint planned (List.mem_toFinset.mpr failed)

def participantDecisions {Participant Decision : Type} [DecidableEq Participant]
    (entries : List (Participant × Decision)) (participant : Participant) : List (Participant × Decision) :=
  entries.filter (fun entry => decide (entry.1=participant))

theorem planned_participant_has_positive_real_decision_count {Participant Decision : Type}
    [DecidableEq Participant] (entries : List (Participant × Decision)) (participant : Participant)
    (planned : participant ∈ plannedParticipants entries) :
    0 < (participantDecisions entries participant).length := by
  obtain ⟨entry,member,equal⟩ := List.mem_map.mp (List.mem_toFinset.mp planned)
  have kept : entry ∈ participantDecisions entries participant := by
    simp [participantDecisions,member,equal]
  apply List.length_pos_iff.mpr
  intro empty
  simp [empty] at kept

theorem checked_selection_failure_has_zero_planned_decisions {Participant Decision : Type}
    [DecidableEq Participant] [DecidableEq Decision]
    (requested : List Participant) (entries : List (Participant × Decision)) (failures : List Participant)
    (accepted : validatePlan requested entries failures=.ok ())
    (participant : Participant) (failed : participant ∈ failures) :
    participantDecisions entries participant=[] := by
  obtain ⟨_,_,_,disjoint,_⟩ := (checked_plan_iff_actual_source_gates _ _ _).mp accepted
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro entry member
  have kept : entry ∈ entries ∧ entry.1=participant := by
    simpa only [participantDecisions,List.mem_filter,decide_eq_true_eq] using member
  have planned : participant ∈ plannedParticipants entries :=
    List.mem_toFinset.mpr (List.mem_map.mpr ⟨entry,kept.1,kept.2⟩)
  exact Finset.disjoint_left.mp disjoint planned (List.mem_toFinset.mpr failed)

theorem checked_participant_denominator_keeps_selection_failures {Participant Decision : Type}
    [DecidableEq Participant] [DecidableEq Decision]
    (requested : List Participant) (entries : List (Participant × Decision)) (failures : List Participant)
    (accepted : validatePlan requested entries failures=.ok ()) :
    (plannedParticipants entries).card+failures.length=requested.length := by
  obtain ⟨requestedValid,_,failuresUnique,disjoint,complete⟩ :=
    (checked_plan_iff_actual_source_gates _ _ _).mp accepted
  have cardinal := congrArg Finset.card complete
  rw [Finset.card_union_of_disjoint disjoint,List.toFinset_card_of_nodup failuresUnique,
    List.toFinset_card_of_nodup requestedValid.2] at cardinal
  exact cardinal

def fixtureRequested : List ℕ := [0,1,2]

def fixtureEntries : List (ℕ × ℕ) := [(0,0),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8)]

def fixtureFailures : List ℕ := [2]

theorem unequal_decision_counts_and_failed_participant_plan_passes :
    validatePlan fixtureRequested fixtureEntries fixtureFailures=.ok () := by
  apply (checked_plan_iff_actual_source_gates _ _ _).mpr
  refine ⟨by simp [fixtureRequested],by simp [fixtureEntries],by simp [fixtureFailures],?_,?_⟩
  · simp [plannedParticipants,fixtureEntries,fixtureFailures,Finset.disjoint_left]
  · ext participant
    simp [plannedParticipants,fixtureEntries,fixtureFailures,fixtureRequested]
    omega

theorem fixture_has_one_and_nine_decisions_and_one_selection_failure :
    (participantDecisions fixtureEntries 0).length=1 ∧
      (participantDecisions fixtureEntries 1).length=9 ∧
      participantDecisions fixtureEntries 2=[] := by
  simp [participantDecisions,fixtureEntries]

end
end N2P3.RequestedParticipantPlan
