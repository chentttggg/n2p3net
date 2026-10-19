import N2P3.RequestedParticipantPlan
import N2P3.RequestedOutcomeRestriction
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum

/-! Real-valued metric specifications for outcomes.py and the requested
participant endpoints. Division here is ideal real arithmetic; source zero
versus None conventions are explicit. No IEEE bound or source accuracy
guarantee follows. Correct counts are derived from the original outcomes. -/

namespace N2P3.RequestedOutcomeRates

open N2P3.RequestedOutcomes N2P3.RequestedOutcomeRestriction
open N2P3.RequestedParticipantPlan

noncomputable section
attribute [local instance] Classical.propDecidable

def ratio0 (numerator denominator : ℕ) : ℝ :=
  if denominator=0 then 0 else (numerator : ℝ)/denominator

def partialRatio (numerator denominator : ℕ) : Option ℝ :=
  if denominator=0 then none else some ((numerator : ℝ)/denominator)

theorem helper_zero_and_analyzer_none_are_distinct :
    ratio0 0 0=0 ∧ partialRatio 0 0=none := by simp [ratio0,partialRatio]

theorem natural_ratio_bounds (numerator denominator : ℕ) (bounded : numerator ≤ denominator) :
    0 ≤ ratio0 numerator denominator ∧ ratio0 numerator denominator ≤ 1 := by
  by_cases zero : denominator=0
  · simp [ratio0,zero]
  · have positive : 0 < (denominator : ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero zero)
    simp only [ratio0,if_neg zero]
    exact ⟨div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _),
      (div_le_one positive).mpr (Nat.cast_le.mpr bounded)⟩

theorem operational_coverage_valid_factorization (correct valid requested : ℕ)
    (bounded : correct ≤ valid) :
    ratio0 correct requested=ratio0 valid requested*ratio0 correct valid := by
  by_cases emptyValid : valid=0
  · have noCorrect : correct=0 := Nat.eq_zero_of_le_zero (emptyValid ▸ bounded)
    simp [ratio0,emptyValid,noCorrect]
  · by_cases emptyRequested : requested=0
    · simp [ratio0,emptyRequested]
    · have nonzero : (valid : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr emptyValid
      simp only [ratio0,if_neg emptyRequested,if_neg emptyValid]
      symm
      calc
        (valid : ℝ)/requested*((correct : ℝ)/valid)=
            (correct : ℝ)/valid*((valid : ℝ)/requested) := mul_comm _ _
        _=(correct : ℝ)/requested := div_mul_div_cancel₀ nonzero

inductive EndpointFault where
  | failedHasPlannedDecisions | plannedHasNoDenominator
  deriving DecidableEq

def participantEndpoint (selectionFailed : Bool) (correct planned : ℕ) : Except EndpointFault ℝ :=
  if selectionFailed then
    if planned=0 then .ok 0 else .error .failedHasPlannedDecisions
  else
    if planned=0 then .error .plannedHasNoDenominator else .ok (ratio0 correct planned)

def participantCorrect {Participant Decision : Type} [DecidableEq Participant]
    (outcomes : List (Outcome (Participant × Decision))) (level : ℕ) (participant : Participant) : ℕ :=
  (countsAt (outcomes.filter (fun outcome => decide (outcome.key.1=participant))) level).correct

def participantRate {Participant Decision : Type} [DecidableEq Participant]
    (entries : List (Participant × Decision)) (failures : List Participant)
    (outcomes : List (Outcome (Participant × Decision))) (level : ℕ) (participant : Participant) : ℝ :=
  if participant ∈ failures then 0 else
    ratio0 (participantCorrect outcomes level participant) (participantDecisions entries participant).length

theorem selection_failed_rate_is_zero {Participant Decision : Type} [DecidableEq Participant]
    (entries : List (Participant × Decision)) (failures : List Participant)
    (outcomes : List (Outcome (Participant × Decision))) (level : ℕ) (participant : Participant)
    (failed : participant ∈ failures) :
    participantRate entries failures outcomes level participant=0 := by
  simp [participantRate,failed]

theorem checked_participant_endpoint_has_no_assertion_failure {Participant Decision : Type}
    [DecidableEq Participant] [DecidableEq Decision]
    (requested : List Participant) (entries : List (Participant × Decision)) (failures : List Participant)
    (outcomes : List (Outcome (Participant × Decision))) (level : ℕ)
    (plan : validatePlan requested entries failures=.ok ()) (participant : Participant) (member : participant ∈ requested) :
    participantEndpoint (decide (participant ∈ failures)) (participantCorrect outcomes level participant)
      (participantDecisions entries participant).length=
        .ok (participantRate entries failures outcomes level participant) := by
  by_cases failed : participant ∈ failures
  · have empty := checked_selection_failure_has_zero_planned_decisions requested entries failures plan participant failed
    simp [participantEndpoint,participantRate,failed,empty]
  · have classes := checked_requested_participant_has_exactly_one_terminal_class requested entries failures plan participant member
    have planned := classes.1.resolve_right failed
    have positive := planned_participant_has_positive_real_decision_count entries participant planned
    have nonzero : (participantDecisions entries participant).length ≠ 0 := Nat.ne_of_gt positive
    simp [participantEndpoint,participantRate,failed,nonzero]

theorem computed_participant_rate_bounds {Participant Decision : Type}
    [DecidableEq Participant] [DecidableEq Decision]
    (entries : List (Participant × Decision)) (failures : List Participant) (levels : List ℕ)
    (outcomes : List (Outcome (Participant × Decision)))
    (checked : validateKeys entries levels outcomes=.ok ()) (level : ℕ) (atLevel : level ∈ levels)
    (participant : Participant) :
    0 ≤ participantRate entries failures outcomes level participant ∧
      participantRate entries failures outcomes level participant ≤ 1 := by
  by_cases failed : participant ∈ failures
  · simp [participantRate,failed]
  · have bounded := checked_local_correct_never_exceeds_its_actual_request entries levels outcomes checked
      (fun key => decide (key.1=participant)) level atLevel
    simpa only [participantRate,if_neg failed,participantCorrect,participantDecisions] using
      natural_ratio_bounds _ _ bounded

def requestedMacro {Participant : Type} (requested : List Participant) (rate : Participant → ℝ) : Option ℝ :=
  if requested=[] then none else some ((requested.map rate).sum/(requested.length : ℝ))

theorem bounded_rate_sum {Participant : Type} (requested : List Participant) (rate : Participant → ℝ)
    (bounded : ∀ participant ∈ requested, 0 ≤ rate participant ∧ rate participant ≤ 1) :
    0 ≤ (requested.map rate).sum ∧ (requested.map rate).sum ≤ (requested.length : ℝ) := by
  induction requested with
  | nil => simp
  | cons participant rest ih =>
    have head := bounded participant (by simp)
    have tail := ih (fun value member => bounded value (by simp [member]))
    simp only [List.map_cons,List.sum_cons,List.length_cons,Nat.cast_add,Nat.cast_one]
    exact ⟨add_nonneg head.1 tail.1,by simpa only [add_comm] using add_le_add head.2 tail.2⟩

theorem requested_macro_rate_bounds {Participant : Type} (requested : List Participant) (rate : Participant → ℝ)
    (nonempty : requested ≠ [])
    (bounded : ∀ participant ∈ requested, 0 ≤ rate participant ∧ rate participant ≤ 1) :
    ∃ value, requestedMacro requested rate=some value ∧ 0 ≤ value ∧ value ≤ 1 := by
  have positive : 0 < (requested.length : ℝ) := Nat.cast_pos.mpr (List.length_pos_iff.mpr nonempty)
  have sums := bounded_rate_sum requested rate bounded
  refine ⟨(requested.map rate).sum/(requested.length : ℝ),?_,?_,?_⟩
  · simp [requestedMacro,nonempty]
  · exact div_nonneg sums.1 (le_of_lt positive)
  · exact (div_le_one positive).mpr sums.2

def fixtureOutcomes : List (Outcome (ℕ × ℕ)) :=
  fixtureEntries.map (fun key => ⟨key,1,if key.1=0 then .correct else .incorrect⟩)

theorem source_fixed_macro_pool_and_failed_participant_counterexample :
    requestedMacro fixtureRequested (participantRate fixtureEntries fixtureFailures fixtureOutcomes 1)=some (1/3) ∧
      requestedMacro [0,1] (participantRate fixtureEntries fixtureFailures fixtureOutcomes 1)=some (1/2) ∧
      ratio0 (countsAt fixtureOutcomes 1).correct fixtureEntries.length=1/10 := by
  simp [requestedMacro,participantRate,participantCorrect,ratio0,fixtureRequested,
    fixtureFailures,fixtureEntries,fixtureOutcomes,participantDecisions,countsAt,countAt]

end
end N2P3.RequestedOutcomeRates
