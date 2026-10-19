import N2P3.RequestedOutcomes
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Exact candidate-event coverage at the requested evidence level

Source: outcomes.py:37-130 and candidate_decision.expected_candidate_counts.
This is after from_mappings' actual string/int normalization and map uniqueness.
Event-count keys need not be the final target-choice vocabulary (row/column
tasks have a separate target domain). Arbitrary direct raw constructor values
and the native coercion implementation remain outside this natural-count view.
-/

namespace N2P3.RequestedCoverage

open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable

structure Coverage (Candidate : Type) where
  expectedKeys : Finset Candidate
  observedKeys : Finset Candidate
  expected : Candidate → ℕ
  observed : Candidate → ℕ

def observedAt [DecidableEq Candidate] (coverage : Coverage Candidate) (candidate : Candidate) : ℕ :=
  if candidate ∈ coverage.observedKeys then coverage.observed candidate else 0

def missing [DecidableEq Candidate] (coverage : Coverage Candidate) : Finset Candidate :=
  coverage.expectedKeys.filter (fun candidate => observedAt coverage candidate < coverage.expected candidate)

def excess [DecidableEq Candidate] (coverage : Coverage Candidate) : Finset Candidate :=
  coverage.observedKeys.filter (fun candidate => coverage.expected candidate < coverage.observed candidate)

def complete [DecidableEq Candidate] (coverage : Coverage Candidate) : Prop :=
  missing coverage=∅ ∧ excess coverage=∅

def expectedEvents (coverage : Coverage Candidate) : ℕ :=
  ∑ candidate ∈ coverage.expectedKeys, coverage.expected candidate

def observedEvents (coverage : Coverage Candidate) : ℕ :=
  ∑ candidate ∈ coverage.observedKeys, coverage.observed candidate

inductive Fault where
  | unexpectedCandidate
  deriving DecidableEq

def validate [DecidableEq Candidate] (coverage : Coverage Candidate) : Except Fault Unit :=
  if coverage.observedKeys ⊆ coverage.expectedKeys then .ok () else .error .unexpectedCandidate

theorem coverage_key_check_is_the_actual_subset_guard [DecidableEq Candidate] (coverage : Coverage Candidate) :
    validate coverage=.ok () ↔ coverage.observedKeys ⊆ coverage.expectedKeys := by
  unfold validate
  split_ifs <;> simp_all

theorem complete_iff_exact_declared_event_counts [DecidableEq Candidate] (coverage : Coverage Candidate)
    (accepted : validate coverage=.ok ()) :
    complete coverage ↔ ∀ candidate ∈ coverage.expectedKeys, observedAt coverage candidate=coverage.expected candidate := by
  have compatible := (coverage_key_check_is_the_actual_subset_guard coverage).mp accepted
  constructor
  · rintro ⟨noMissing,noExcess⟩ candidate expected
    have lower : ¬ observedAt coverage candidate < coverage.expected candidate := by
      intro short
      have member : candidate ∈ missing coverage := Finset.mem_filter.mpr ⟨expected,short⟩
      simp [noMissing] at member
    by_cases observed : candidate ∈ coverage.observedKeys
    · have upper : ¬ coverage.expected candidate < coverage.observed candidate := by
        intro extra
        have member : candidate ∈ excess coverage := Finset.mem_filter.mpr ⟨observed,extra⟩
        simp [noExcess] at member
      simp only [observedAt,if_pos observed] at lower ⊢
      omega
    · simp only [observedAt,if_neg observed] at lower ⊢
      omega
  · intro exactCounts
    constructor
    · apply Finset.eq_empty_iff_forall_notMem.mpr
      intro candidate member
      have fields := Finset.mem_filter.mp member
      have equal := exactCounts candidate fields.1
      exact (Nat.lt_irrefl _) (equal ▸ fields.2)
    · apply Finset.eq_empty_iff_forall_notMem.mpr
      intro candidate member
      have fields := Finset.mem_filter.mp member
      have equal := exactCounts candidate (compatible fields.1)
      simp only [observedAt,if_pos fields.1] at equal
      exact (Nat.lt_irrefl _) (equal ▸ fields.2)

theorem observed_total_uses_original_observed_map [DecidableEq Candidate] (coverage : Coverage Candidate)
    (accepted : validate coverage=.ok ()) :
    (∑ candidate ∈ coverage.expectedKeys, observedAt coverage candidate)=observedEvents coverage := by
  have compatible := (coverage_key_check_is_the_actual_subset_guard coverage).mp accepted
  have same : (∑ candidate ∈ coverage.observedKeys, observedAt coverage candidate)=
      ∑ candidate ∈ coverage.expectedKeys, observedAt coverage candidate := by
    apply Finset.sum_subset compatible
    intro candidate _ absent
    simp [observedAt,absent]
  rw [←same]
  apply Finset.sum_congr rfl
  intro candidate member
  simp [observedAt,member]

theorem complete_preserves_total_scheduled_event_count [DecidableEq Candidate] (coverage : Coverage Candidate)
    (accepted : validate coverage=.ok ()) (finished : complete coverage) :
    observedEvents coverage=expectedEvents coverage := by
  rw [←observed_total_uses_original_observed_map coverage accepted]
  apply Finset.sum_congr rfl
  exact (complete_iff_exact_declared_event_counts coverage accepted).mp finished

theorem uniform_task_count_is_key_count_times_level (coverage : Coverage Candidate) (level : ℕ)
    (uniform : ∀ candidate ∈ coverage.expectedKeys, coverage.expected candidate=level) :
    expectedEvents coverage=coverage.expectedKeys.card*level := by
  calc
    expectedEvents coverage=(∑ _ ∈ coverage.expectedKeys, level) := Finset.sum_congr rfl uniform
    _=coverage.expectedKeys.card*level := by simp

theorem complete_task_with_declared_events_has_observations [DecidableEq Candidate]
    (coverage : Coverage Candidate) (level : ℕ) (positive : 0 < level)
    (hasCandidates : coverage.expectedKeys.Nonempty)
    (uniform : ∀ candidate ∈ coverage.expectedKeys, coverage.expected candidate=level)
    (accepted : validate coverage=.ok ()) (finished : complete coverage) :
    0 < observedEvents coverage := by
  rw [complete_preserves_total_scheduled_event_count coverage accepted finished,
    uniform_task_count_is_key_count_times_level coverage level uniform]
  exact Nat.mul_pos (Finset.card_pos.mpr hasCandidates) positive

def emptyCoverage : Coverage ℕ := ⟨∅,∅,fun _ => 0,fun _ => 0⟩

theorem empty_complete_helper_is_not_a_frozen_candidate_task :
    validate emptyCoverage=.ok () ∧ complete emptyCoverage ∧ emptyCoverage.expectedKeys.card=0 := by
  simp [validate,complete,missing,excess,emptyCoverage]

def excessiveCoverage : Coverage ℕ :=
  ⟨{0,1},{0,1},fun _ => 1,fun candidate => if candidate=0 then 2 else 1⟩

theorem absence_of_missing_events_does_not_allow_excess :
    validate excessiveCoverage=.ok () ∧ missing excessiveCoverage=∅ ∧ ¬complete excessiveCoverage := by
  simp [validate,missing,excess,complete,observedAt,excessiveCoverage,Finset.filter_insert]

end
end N2P3.RequestedCoverage
