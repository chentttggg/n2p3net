import N2P3.TrainingCrossEntropyRisk
import Mathlib.Data.Finset.Card

/-!
Original L02: actual authorized groups keep requested risk mass even when a
sampled group has no trainable membership loss. An operational tie remains
Some CE under the existing membership loss contract. The typed real losses
here are the finite, default-uniform-prior actual SubjectAdapter caller, not
an unproved extended-log/zero-prior standalone primitive.
Packed P2 counts concern direct one-candidate events and fixed C,R; they do
not assert these counts for a general set-membership multi-candidate flash.
-/
namespace N2P3.RequestedDecisionTrainingRisk
open scoped BigOperators
open N2P3.TrainingCrossEntropyRisk
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def trainable {D : Type*} [Fintype D] (loss : D → Option ℝ) : Finset D :=
  Finset.univ.filter (fun decision => (loss decision).isSome=true)
def requestedRisk {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) : ℝ :=
  risk mass (fun _ => 1) (fun decision => (loss decision).getD 0)
def eligibleMass {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) : ℝ :=
  ∑ decision ∈ trainable loss,mass decision
def eligibleNumerator {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) : ℝ :=
  ∑ decision ∈ trainable loss,mass decision*(loss decision).getD 0
def conditionalRisk {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) : ℝ :=
  eligibleNumerator mass loss/eligibleMass mass loss

theorem actual_requested_risk_keeps_the_original_authorized_mass_denominator
    {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) :
    requestedRisk mass loss=eligibleNumerator mass loss/denominator mass := by
  unfold requestedRisk risk numerator eligibleNumerator trainable
  congr 1
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro decision _
  cases found : loss decision <;> simp [found]

theorem requested_risk_equals_eligible_mass_fraction_times_conditional_risk
    {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ)
    (positiveEligibleMass : eligibleMass mass loss ≠ 0) :
    requestedRisk mass loss=(eligibleMass mass loss/denominator mass)*conditionalRisk mass loss := by
  rw [actual_requested_risk_keeps_the_original_authorized_mass_denominator]
  unfold conditionalRisk
  field_simp

theorem no_trainable_groups_produce_the_actual_zero_requested_risk
    {D : Type*} [Fintype D] (mass : D → ℝ) : requestedRisk mass (fun _ : D => none)=0 := by
  simp [requestedRisk,risk,numerator]

theorem ineligible_group_mass_still_contributes_to_the_authorized_denominator
    {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) (decision : D)
    (none : loss decision=Option.none) :
    decision ∉ trainable loss ∧ (loss decision).getD 0=0 ∧
      coefficient mass (fun _ => 1) decision=mass decision/denominator mass := by
  simp [trainable,none,coefficient]

theorem a_tied_but_trainable_group_stays_in_the_decision_risk_population
    {D : Type*} [Fintype D] (loss : D → Option ℝ) (decision : D) (value : ℝ)
    (some : loss decision=Option.some value) :
    decision ∈ trainable loss ∧ (loss decision).getD 0=value := by simp [trainable,some]

theorem multiplying_all_original_requested_masses_leaves_the_requested_risk_unchanged
    {D : Type*} [Fintype D] (mass : D → ℝ) (loss : D → Option ℝ) (scale : ℝ) (nonzero : scale ≠ 0) :
    requestedRisk (fun decision => scale*mass decision) loss=requestedRisk mass loss := by
  unfold requestedRisk risk numerator denominator
  simp only [mul_one,mul_assoc,← Finset.mul_sum]
  exact mul_div_mul_left _ _ nonzero

def directLabel {C R : ℕ} (truth : Fin C) (row : Fin C × Fin R) : Bool := decide (row.1=truth)

theorem complete_direct_decision_has_exactly_R_positive_event_rows
    {C R : ℕ} (truth : Fin C) :
    (∑ row : Fin C × Fin R,if directLabel truth row then (1 : ℝ) else 0)=(R : ℝ) := by
  rw [Fintype.sum_prod_type]
  simp [directLabel]

theorem complete_direct_decision_has_C_times_R_total_rows (C R : ℕ) :
    Fintype.card (Fin C × Fin R)=C*R := by simp

theorem complete_direct_decision_has_C_minus_one_times_R_negative_event_rows
    {C R : ℕ} (truth : Fin C) :
    (∑ row : Fin C × Fin R,if directLabel truth row then (0 : ℝ) else 1)=((C-1)*R : ℕ) := by
  have total : (∑ row : Fin C × Fin R,if directLabel truth row then (1 : ℝ) else 0)+
      (∑ row : Fin C × Fin R,if directLabel truth row then (0 : ℝ) else 1)=(C*R : ℕ) := by
    rw [← Finset.sum_add_distrib]
    have each : ∀ row : Fin C × Fin R,
        (if directLabel truth row then (1 : ℝ) else 0)+(if directLabel truth row then (0 : ℝ) else 1)=1 := by
      intro row; cases directLabel truth row <;> simp
    simp_rw [each]
    simp
  rw [complete_direct_decision_has_exactly_R_positive_event_rows truth] at total
  have positiveC : 1≤C := by have := truth.isLt; omega
  rw [Nat.cast_mul,Nat.cast_sub positiveC,Nat.cast_one]
  rw [Nat.cast_mul] at total
  nlinarith

def directClassMass {C R : ℕ} (truth : Fin C) (positiveWeight : ℝ) : ℝ :=
  ∑ row : Fin C × Fin R,classWeight positiveWeight (directLabel truth row)

theorem actual_complete_decision_derives_its_class_mass_from_labels_and_repetitions
    {C R : ℕ} (truth : Fin C) (positiveWeight : ℝ) :
    directClassMass (R:=R) truth positiveWeight=(R : ℝ)*((C-1 : ℕ)+positiveWeight) := by
  have split : ∀ row : Fin C × Fin R,classWeight positiveWeight (directLabel truth row)=
      positiveWeight*(if directLabel truth row then (1 : ℝ) else 0)+
        (if directLabel truth row then (0 : ℝ) else 1) := by
    intro row; cases directLabel truth row <;> simp [classWeight]
  simp_rw [directClassMass,split,Finset.sum_add_distrib,← Finset.mul_sum]
  rw [complete_direct_decision_has_exactly_R_positive_event_rows truth,
    complete_direct_decision_has_C_minus_one_times_R_negative_event_rows truth,Nat.cast_mul]
  ring

theorem full_packed_batch_class_mass_is_proportional_to_its_complete_decision_count
    {D C R : ℕ} (truth : Fin D → Fin C) (positiveWeight : ℝ) :
    (∑ decision : Fin D,directClassMass (R:=R) (truth decision) positiveWeight)=
      (D : ℝ)*(R : ℝ)*((C-1 : ℕ)+positiveWeight) := by
  simp_rw [actual_complete_decision_derives_its_class_mass_from_labels_and_repetitions]
  simp only [Finset.sum_const,Finset.card_univ,Fintype.card_fin,nsmul_eq_mul]
  ring

theorem positive_class_weight_and_nonempty_complete_decision_give_positive_class_mass
    {C R : ℕ} (truth : Fin C) (positiveWeight : ℝ) (positive : 0<positiveWeight) (repeated : 0<R) :
    0<directClassMass (R:=R) truth positiveWeight := by
  rw [actual_complete_decision_derives_its_class_mass_from_labels_and_repetitions]
  exact mul_pos (Nat.cast_pos.mpr repeated) (add_pos_of_nonneg_of_pos (Nat.cast_nonneg _) positive)

end
end N2P3.RequestedDecisionTrainingRisk
