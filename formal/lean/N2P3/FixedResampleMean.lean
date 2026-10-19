import N2P3.RequestedAggregationWeights

/-! Fixed resample index tables. Keys of the inner mean are DRAW POSITIONS,
not deduplicated original units. A typed index table is a deterministic input;
neither a distribution for it nor scientific independence is assumed. -/

namespace N2P3.FixedResampleMean

open scoped BigOperators
open N2P3.RequestedAggregationWeights (mean)

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def drawMean {n : ℕ} (units : Fin n → ℝ) (indices : Fin n → Fin n) : ℝ :=
  mean Finset.univ (fun position => units (indices position))

def drawWeight {n : ℕ} (indices : Fin n → Fin n) (unit : Fin n) : ℝ :=
  drawMean (fun original => if original=unit then 1 else 0) indices

def multiplicity {n : ℕ} (indices : Fin n → Fin n) (unit : Fin n) : ℕ :=
  (Finset.univ.filter (fun position => indices position=unit)).card

theorem draw_weight_indicator_response_is_actual_multiplicity {n : ℕ}
    (indices : Fin n → Fin n) (unit : Fin n) :
    drawWeight indices unit=(multiplicity indices unit : ℝ)/(n : ℝ) := by
  simp [drawWeight,drawMean,mean,multiplicity,Finset.sum_boole]

theorem actual_draw_unit_coefficients_are_nonnegative {n : ℕ}
    (indices : Fin n → Fin n) (unit : Fin n) : 0 ≤ drawWeight indices unit := by
  rw [draw_weight_indicator_response_is_actual_multiplicity]
  exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

theorem fixed_draw_coefficients_sum_to_one {n : ℕ} (positive : 0<n) (indices : Fin n → Fin n) :
    (∑ unit,drawWeight indices unit)=1 := by
  have nonzero : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.ne_of_gt positive)
  unfold drawWeight drawMean mean
  simp only [Finset.card_univ,Fintype.card_fin]
  rw [←Finset.sum_div,Finset.sum_comm]
  simp [nonzero]

theorem fixed_draw_mean_equals_original_units_times_computed_coefficients {n : ℕ}
    (units : Fin n → ℝ) (indices : Fin n → Fin n) :
    drawMean units indices=∑ unit,units unit*drawWeight indices unit := by
  have numerator : (∑ position,units (indices position))=
      ∑ unit,units unit*(∑ position,if indices position=unit then (1 : ℝ) else 0) := by
    symm
    simp_rw [Finset.mul_sum,mul_ite,mul_one,mul_zero]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro position _
    simp
  unfold drawWeight drawMean mean
  simp only [Finset.card_univ,Fintype.card_fin]
  simp_rw [←mul_div_assoc]
  rw [←Finset.sum_div]
  exact congrArg (fun total : ℝ => total/(n : ℝ)) numerator

def studyDrawMean {Study : Type} (studies : Finset Study) (sizes : Study → ℕ)
    (units : (study : Study) → Fin (sizes study) → ℝ)
    (indices : (study : Study) → Fin (sizes study) → Fin (sizes study)) : ℝ :=
  mean studies (fun study => drawMean (units study) (indices study))

theorem each_study_draw_retains_its_own_unit_denominator {Study : Type}
    (studies : Finset Study) (sizes : Study → ℕ)
    (units : (study : Study) → Fin (sizes study) → ℝ)
    (indices : (study : Study) → Fin (sizes study) → Fin (sizes study)) :
    studyDrawMean studies sizes units indices=
      ∑ study ∈ studies,∑ unit : Fin (sizes study),
        units study unit*(drawWeight (indices study) unit/(studies.card : ℝ)) := by
  unfold studyDrawMean mean
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro study _
  rw [fixed_draw_mean_equals_original_units_times_computed_coefficients,Finset.sum_div]
  apply Finset.sum_congr rfl
  intro unit _
  exact mul_div_assoc _ _ _

theorem stratified_draw_original_unit_coefficient_is_multiplicity_over_study_and_unit_counts
    {Study : Type} (studies : Finset Study) (sizes : Study → ℕ)
    (indices : (study : Study) → Fin (sizes study) → Fin (sizes study))
    (study : Study) (unit : Fin (sizes study)) :
    drawWeight (indices study) unit/(studies.card : ℝ)=
      (multiplicity (indices study) unit : ℝ)/((studies.card : ℝ)*(sizes study : ℝ)) := by
  rw [draw_weight_indicator_response_is_actual_multiplicity,div_div]
  rw [mul_comm]

theorem fixed_study_draw_coefficient_mass_is_one {Study : Type}
    (studies : Finset Study) (sizes : Study → ℕ)
    (indices : (study : Study) → Fin (sizes study) → Fin (sizes study))
    (hasStudies : studies≠∅) (positive : ∀ study ∈ studies,0<sizes study) :
    (∑ study ∈ studies,∑ unit : Fin (sizes study),drawWeight (indices study) unit/(studies.card : ℝ))=1 := by
  have nonzero : (studies.card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr hasStudies))
  calc
    (∑ study ∈ studies,∑ unit : Fin (sizes study),drawWeight (indices study) unit/(studies.card : ℝ))=
        ∑ study ∈ studies,1/(studies.card : ℝ) := by
      apply Finset.sum_congr rfl
      intro study declared
      rw [←Finset.sum_div,fixed_draw_coefficients_sum_to_one (positive study declared)]
    _=1 := by simp [nonzero]

end
end N2P3.FixedResampleMean
