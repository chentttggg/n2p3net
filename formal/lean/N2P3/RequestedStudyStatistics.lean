import N2P3.RequestedNLL
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FinCases

/-! Fixed declared-study aggregation after identity and seed-grid checks.
Option.none is an unavailable complete-pair contrast, not an imputed AUC=0.
The point estimate and requested-population identification bounds differ.
Source accepts approximately zero-sum coefficients (numpy.isclose); the exact
zero-sum theorem below therefore has an explicit premise and no native claim.
Bootstrap confidence coverage and external study generalization are unproved. -/

namespace N2P3.RequestedStudyStatistics

open N2P3.RequestedNLL
open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable

def observedMean (rows : List (Option ℝ)) : ℝ :=
  (scoredValues rows).sum/((scoredValues rows).length : ℝ)

def fullStudyAvailable (studies : List (List (Option ℝ))) : Prop :=
  ∀ rows ∈ studies, scoredValues rows ≠ []

def studyEqual (studies : List (List (Option ℝ))) : Option ℝ :=
  if studies=[] ∨ ¬fullStudyAvailable studies then none
  else some ((studies.map observedMean).sum/(studies.length : ℝ))

theorem missing_expected_study_closes_full_study_point_estimate
    (studies : List (List (Option ℝ))) (rows : List (Option ℝ))
    (declared : rows ∈ studies) (missing : scoredValues rows=[]) : studyEqual studies=none := by
  have unavailable : ¬fullStudyAvailable studies := fun available => available rows declared missing
  simp [studyEqual,unavailable]

theorem fixed_study_equal_is_not_pooled_participant_mean :
    studyEqual [[some 1,none],[some 0,some 0,some 0]]=some (1/2 : ℝ) ∧
    scoredMean [some 1,none,some 0,some 0,some 0]=some (1/4 : ℝ) ∧
    fullStudyAvailable [[some 1,none],[some 0,some 0,some 0]] := by
  simp [studyEqual,fullStudyAvailable,observedMean,scoredValues,scoredMean]

theorem full_study_available_does_not_require_every_requested_participant_complete :
    fullStudyAvailable [[some 1,none],[some 0]] ∧
      missingCount [some 1,none]=1 := by
  simp [fullStudyAvailable,scoredValues,missingCount]

def positiveMass {K : ℕ} (coefficients : Fin K → ℝ) : ℝ :=
  ∑ arm, max (coefficients arm) 0

def contrast {K : ℕ} (coefficients values : Fin K → ℝ) : ℝ :=
  ∑ arm, coefficients arm*values arm

theorem each_bounded_auc_contribution_has_signed_coefficient_bounds (coefficient value : ℝ)
    (bounded : 0 ≤ value ∧ value ≤ 1) :
    min coefficient 0 ≤ coefficient*value ∧ coefficient*value ≤ max coefficient 0 := by
  by_cases positive : 0 ≤ coefficient
  · rw [min_eq_right positive,max_eq_left positive]
    exact ⟨mul_nonneg positive bounded.1,by simpa using mul_le_mul_of_nonneg_left bounded.2 positive⟩
  · have negative : coefficient ≤ 0 := le_of_not_ge positive
    rw [min_eq_left negative,max_eq_right negative]
    exact ⟨by simpa using mul_le_mul_of_nonpos_left bounded.2 negative,
      mul_nonpos_of_nonpos_of_nonneg negative bounded.1⟩

theorem auc_contrast_general_bounds_preserve_coefficient_residual {K : ℕ}
    (coefficients values : Fin K → ℝ) (bounded : ∀ arm, 0 ≤ values arm ∧ values arm ≤ 1) :
    (∑ arm,coefficients arm)-positiveMass coefficients ≤ contrast coefficients values ∧
      contrast coefficients values ≤ positiveMass coefficients := by
  have lower : (∑ arm,min (coefficients arm) 0) ≤ contrast coefficients values :=
    Finset.sum_le_sum (fun arm _ =>
      (each_bounded_auc_contribution_has_signed_coefficient_bounds _ _ (bounded arm)).1)
  have identity : (∑ arm,min (coefficients arm) 0)=(∑ arm,coefficients arm)-positiveMass coefficients := by
    unfold positiveMass
    rw [←Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro arm _
    have equal := min_add_max (coefficients arm) 0
    linarith
  rw [identity] at lower
  refine ⟨lower,?_⟩
  exact Finset.sum_le_sum (fun arm _ =>
    (each_bounded_auc_contribution_has_signed_coefficient_bounds _ _ (bounded arm)).2)

theorem exact_zero_sum_auc_contrast_is_bounded_by_positive_mass {K : ℕ}
    (coefficients values : Fin K → ℝ) (zeroSum : (∑ arm,coefficients arm)=0)
    (bounded : ∀ arm, 0 ≤ values arm ∧ values arm ≤ 1) :
    -positiveMass coefficients ≤ contrast coefficients values ∧
      contrast coefficients values ≤ positiveMass coefficients := by
  simpa only [zeroSum,zero_sub] using
    auc_contrast_general_bounds_preserve_coefficient_residual coefficients values bounded

def negativeResidualWeights : Fin 2 → ℝ := ![1,-(1+1/1000000000)]
def extremalAUCs : Fin 2 → ℝ := ![0,1]

theorem approximate_zero_sum_does_not_entail_exact_symmetric_bound :
    (∑ arm,negativeResidualWeights arm)= -(1/1000000000 : ℝ) ∧
    contrast negativeResidualWeights extremalAUCs < -positiveMass negativeResidualWeights := by
  norm_num [negativeResidualWeights,extremalAUCs,contrast,positiveMass,Fin.sum_univ_two]

def missingBounds (B : ℝ) (rows : List (Option ℝ)) : ℝ × ℝ :=
  (((scoredValues rows).sum-(missingCount rows : ℝ)*B)/(rows.length : ℝ),
   ((scoredValues rows).sum+(missingCount rows : ℝ)*B)/(rows.length : ℝ))

def asymmetricMissingBounds (lower upper : ℝ) (rows : List (Option ℝ)) : ℝ × ℝ :=
  (((scoredValues rows).sum+(missingCount rows : ℝ)*lower)/(rows.length : ℝ),
   ((scoredValues rows).sum+(missingCount rows : ℝ)*upper)/(rows.length : ℝ))

def CompletesBetween (lower upper : ℝ) : List (Option ℝ) → List ℝ → Prop :=
  List.Forall₂ (fun given actual =>
    match given with
    | some observed => actual=observed
    | none => lower ≤ actual ∧ actual ≤ upper)

theorem general_completion_preserves_length_and_asymmetric_sum_bounds
    (lower upper : ℝ) (rows : List (Option ℝ)) (completed : List ℝ)
    (completion : CompletesBetween lower upper rows completed) :
    completed.length=rows.length ∧
      (scoredValues rows).sum+(missingCount rows : ℝ)*lower ≤ completed.sum ∧
      completed.sum ≤ (scoredValues rows).sum+(missingCount rows : ℝ)*upper := by
  induction completion with
  | nil => simp [scoredValues,missingCount]
  | @cons given actual rows completed head tail ih =>
    cases given with
    | none =>
      change lower ≤ actual ∧ actual ≤ upper at head
      simp [scoredValues,missingCount] at ih ⊢
      exact ⟨ih.1,by linarith [ih.2.1,head.1],by linarith [ih.2.2,head.2]⟩
    | some observed =>
      change actual=observed at head
      subst actual
      simp [scoredValues,missingCount] at ih ⊢
      exact ⟨ih.1,by linarith [ih.2.1],by linarith [ih.2.2]⟩

theorem asymmetric_requested_bounds_contain_every_allowed_completion
    (lower upper : ℝ) (rows : List (Option ℝ)) (completed : List ℝ) (requested : rows ≠ [])
    (completion : CompletesBetween lower upper rows completed) :
    (asymmetricMissingBounds lower upper rows).1 ≤ completed.sum/(completed.length : ℝ) ∧
      completed.sum/(completed.length : ℝ) ≤ (asymmetricMissingBounds lower upper rows).2 := by
  have actual := general_completion_preserves_length_and_asymmetric_sum_bounds lower upper rows completed completion
  have positive : 0 < (rows.length : ℝ) := Nat.cast_pos.mpr (List.length_pos_iff.mpr requested)
  rw [actual.1]
  exact ⟨(div_le_div_iff_of_pos_right positive).mpr actual.2.1,
    (div_le_div_iff_of_pos_right positive).mpr actual.2.2⟩

theorem signed_coefficient_bounds_are_the_source_negative_and_positive_sums {K : ℕ}
    (coefficients : Fin K → ℝ) :
    (∑ arm,if coefficients arm<0 then coefficients arm else 0)=(∑ arm,coefficients arm)-positiveMass coefficients ∧
    (∑ arm,if 0<coefficients arm then coefficients arm else 0)=positiveMass coefficients := by
  constructor
  · unfold positiveMass
    rw [←Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro arm _
    by_cases negative : coefficients arm<0
    · simp [negative,max_eq_right (le_of_lt negative)]
    · simp [negative,max_eq_left (le_of_not_gt negative)]
  · apply Finset.sum_congr rfl
    intro arm _
    by_cases positive : 0<coefficients arm
    · simp [positive,max_eq_left (le_of_lt positive)]
    · simp [positive,max_eq_right (le_of_not_gt positive)]

theorem declared_study_equal_average_preserves_each_study_bound
    (studies : List (ℝ × ℝ × ℝ))
    (bounded : ∀ study ∈ studies, study.1 ≤ study.2.1 ∧ study.2.1 ≤ study.2.2) :
    (studies.map (fun study => study.1)).sum/(studies.length : ℝ) ≤
      (studies.map (fun study => study.2.1)).sum/(studies.length : ℝ) ∧
    (studies.map (fun study => study.2.1)).sum/(studies.length : ℝ) ≤
      (studies.map (fun study => study.2.2)).sum/(studies.length : ℝ) := by
  have sums : (studies.map (fun study => study.1)).sum ≤ (studies.map (fun study => study.2.1)).sum ∧
      (studies.map (fun study => study.2.1)).sum ≤ (studies.map (fun study => study.2.2)).sum := by
    induction studies with
    | nil => simp
    | cons study rest ih =>
      have head := bounded study (by simp)
      have tail := ih (fun value member => bounded value (by simp [member]))
      simp only [List.map_cons,List.sum_cons]
      exact ⟨add_le_add head.1 tail.1,add_le_add head.2 tail.2⟩
  exact ⟨div_le_div_of_nonneg_right sums.1 (Nat.cast_nonneg _),
    div_le_div_of_nonneg_right sums.2 (Nat.cast_nonneg _)⟩

def CompletesWithin (B : ℝ) : List (Option ℝ) → List ℝ → Prop :=
  List.Forall₂ (fun given actual =>
    match given with
    | some observed => actual=observed
    | none => -B ≤ actual ∧ actual ≤ B)

theorem actual_completion_preserves_requested_length_and_sum_bounds
    (B : ℝ) (rows : List (Option ℝ)) (completed : List ℝ)
    (completion : CompletesWithin B rows completed) :
    completed.length=rows.length ∧
      (scoredValues rows).sum-(missingCount rows : ℝ)*B ≤ completed.sum ∧
      completed.sum ≤ (scoredValues rows).sum+(missingCount rows : ℝ)*B := by
  induction completion with
  | nil => simp [scoredValues,missingCount]
  | @cons given actual rows completed head tail ih =>
    cases given with
    | none =>
      change -B ≤ actual ∧ actual ≤ B at head
      simp [scoredValues,missingCount] at ih ⊢
      exact ⟨ih.1,by linarith [ih.2.1,head.1],by linarith [ih.2.2,head.2]⟩
    | some observed =>
      change actual=observed at head
      subst actual
      simp [scoredValues,missingCount] at ih ⊢
      exact ⟨ih.1,by linarith [ih.2.1],by linarith [ih.2.2]⟩

theorem requested_missing_result_bounds_contain_every_bounded_completion
    (B : ℝ) (rows : List (Option ℝ)) (completed : List ℝ) (requested : rows ≠ [])
    (completion : CompletesWithin B rows completed) :
    (missingBounds B rows).1 ≤ completed.sum/(completed.length : ℝ) ∧
      completed.sum/(completed.length : ℝ) ≤ (missingBounds B rows).2 := by
  have actual := actual_completion_preserves_requested_length_and_sum_bounds B rows completed completion
  have positive : 0 < (rows.length : ℝ) := Nat.cast_pos.mpr (List.length_pos_iff.mpr requested)
  rw [actual.1]
  exact ⟨(div_le_div_iff_of_pos_right positive).mpr actual.2.1,
    (div_le_div_iff_of_pos_right positive).mpr actual.2.2⟩

theorem no_missing_rows_collapse_identification_bounds
    (B : ℝ) (values : List ℝ) :
    missingBounds B (values.map some)=(values.sum/(values.length : ℝ),values.sum/(values.length : ℝ)) := by
  have actual := scored_mean_retains_only_actual_some_outputs values
  simp [missingBounds,actual.1,actual.2]

end
end N2P3.RequestedStudyStatistics
