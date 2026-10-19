import N2P3.RequestedAggregationWeights

/-! Actual conditional participant/cell macro NLL, with variable scored-set
sizes. Source-before-proof nll_macro_weight_requirements_01 specifies the
two filters. Missing a decision NLL does not exclude a cell that still has
scored rows; this differs from the complete-seed-grid contrast policy. -/

namespace N2P3.RequestedNLLMacroWeights

open scoped BigOperators
open N2P3.RequestedAggregationWeights (mean)

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

structure Program (Cell Decision : Type) where
  cells : Finset Cell
  decisions : Cell → Finset Decision
  completed : Cell → Bool
  nll : Cell → Decision → Option ℝ

variable {Cell Decision : Type}

def scored (program : Program Cell Decision) (cell : Cell) : Finset Decision :=
  (program.decisions cell).filter (fun decision => (program.nll cell decision).isSome)

def available (program : Program Cell Decision) : Finset Cell :=
  program.cells.filter (fun cell => program.completed cell=true ∧ scored program cell≠∅)

def value (program : Program Cell Decision) (cell : Cell) (decision : Decision) : ℝ :=
  (program.nll cell decision).getD 0

def linearMean (program : Program Cell Decision) (values : Cell → Decision → ℝ) : ℝ :=
  mean (available program) (fun cell => mean (scored program cell) (values cell))

def reported (program : Program Cell Decision) : Option ℝ :=
  if available program=∅ then none else some (linearMean program (value program))

def coefficient (program : Program Cell Decision) (chosenCell : Cell) (chosenDecision : Decision) : ℝ :=
  linearMean program (fun cell decision =>
    if cell=chosenCell then if decision=chosenDecision then 1 else 0 else 0)

theorem available_cells_are_computed_from_source_status_and_actual_some_rows
    (program : Program Cell Decision) (cell : Cell) :
    cell ∈ available program ↔ cell ∈ program.cells ∧ program.completed cell=true ∧
      ∃ decision ∈ program.decisions cell, (program.nll cell decision).isSome=true := by
  simp [available,scored,Finset.nonempty_iff_ne_empty.symm,Finset.filter_nonempty_iff]

theorem coefficient_is_derived_from_the_conditional_mean_operator
    (program : Program Cell Decision) (cell : Cell) (decision : Decision) :
    coefficient program cell decision=
      if cell ∈ available program ∧ decision ∈ scored program cell then
        (1/((scored program cell).card : ℝ))/(available program).card else 0 := by
  have inner (c : Cell) : mean (scored program c) (fun d =>
      if c=cell then if d=decision then 1 else 0 else 0)=
      if c=cell then if decision ∈ scored program c then 1/((scored program c).card : ℝ) else 0 else 0 := by
    by_cases same : c=cell <;> simp [mean,same,ite_div]
  unfold coefficient linearMean
  simp_rw [inner]
  by_cases cellAvailable : cell ∈ available program <;> by_cases decisionScored : decision ∈ scored program cell <;>
    simp [mean,cellAvailable,decisionScored]

theorem source_cell_and_row_denominators_can_vary_without_pooling
    (program : Program Cell Decision) (cell : Cell) (decision : Decision) :
    coefficient program cell decision=
      if cell ∈ available program ∧ decision ∈ scored program cell then
        1/(((available program).card : ℝ)*(scored program cell).card) else 0 := by
  rw [coefficient_is_derived_from_the_conditional_mean_operator]
  split_ifs
  · simp only [div_eq_mul_inv,mul_inv_rev,one_mul]
  · rfl

theorem computed_nll_coefficients_are_nonnegative
    (program : Program Cell Decision) (cell : Cell) (decision : Decision) :
    0 ≤ coefficient program cell decision := by
  rw [coefficient_is_derived_from_the_conditional_mean_operator]
  split_ifs
  · exact div_nonneg (div_nonneg zero_le_one (Nat.cast_nonneg _)) (Nat.cast_nonneg _)
  · exact le_refl _

theorem missing_or_unavailable_rows_have_zero_coefficient
    (program : Program Cell Decision) (cell : Cell) (decision : Decision)
    (excluded : cell ∉ available program ∨ decision ∉ scored program cell) :
    coefficient program cell decision=0 := by
  rcases excluded with cellExcluded | rowExcluded
  · simp [coefficient_is_derived_from_the_conditional_mean_operator,cellExcluded]
  · simp [coefficient_is_derived_from_the_conditional_mean_operator,rowExcluded]

theorem every_retained_cell_gets_equal_total_mass
    (program : Program Cell Decision) (cell : Cell) (selected : cell ∈ available program) :
    (∑ decision ∈ scored program cell,coefficient program cell decision)=1/((available program).card : ℝ) := by
  have nonempty : scored program cell≠∅ := (Finset.mem_filter.mp selected).2.2
  have nonzero : ((scored program cell).card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr nonempty))
  calc
    (∑ decision ∈ scored program cell,coefficient program cell decision)=
      ∑ _decision ∈ scored program cell,(1/((scored program cell).card : ℝ))/(available program).card := by
      apply Finset.sum_congr rfl
      intro decision rowSelected
      simp [coefficient_is_derived_from_the_conditional_mean_operator,selected,rowSelected]
    _=1/((available program).card : ℝ) := by
      simp only [Finset.sum_const,nsmul_eq_mul]
      rw [←mul_div_assoc]
      simp [nonzero]

theorem conditional_macro_coefficient_mass_is_one
    (program : Program Cell Decision) (defined : available program≠∅) :
    (∑ cell ∈ available program,∑ decision ∈ scored program cell,coefficient program cell decision)=1 := by
  have nonzero : ((available program).card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr defined))
  calc
    (∑ cell ∈ available program,∑ decision ∈ scored program cell,coefficient program cell decision)=
      ∑ _cell ∈ available program,1/((available program).card : ℝ) := by
      apply Finset.sum_congr rfl
      intro cell selected
      exact every_retained_cell_gets_equal_total_mass program cell selected
    _=1 := by simp [nonzero]

theorem nested_macro_is_derived_weighted_sum_of_original_scored_rows
    (program : Program Cell Decision) (values : Cell → Decision → ℝ) :
    linearMean program values=
      ∑ cell ∈ available program,∑ decision ∈ scored program cell,
        values cell decision*coefficient program cell decision := by
  unfold linearMean mean
  simp_rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro cell selected
  apply Finset.sum_congr rfl
  intro decision rowSelected
  rw [coefficient_is_derived_from_the_conditional_mean_operator]
  simp [selected,rowSelected,div_eq_mul_inv,mul_assoc]

theorem extending_to_original_requests_adds_only_zero_weight_terms
    (program : Program Cell Decision) (values : Cell → Decision → ℝ) :
    (∑ cell ∈ available program,∑ decision ∈ scored program cell,values cell decision*coefficient program cell decision)=
      ∑ cell ∈ program.cells,∑ decision ∈ program.decisions cell,values cell decision*coefficient program cell decision := by
  have rows (cell : Cell) :
      (∑ decision ∈ scored program cell,values cell decision*coefficient program cell decision)=
        ∑ decision ∈ program.decisions cell,values cell decision*coefficient program cell decision := by
    apply Finset.sum_subset (Finset.filter_subset _ _)
    intro decision _ absent
    rw [missing_or_unavailable_rows_have_zero_coefficient program cell decision (Or.inr absent),mul_zero]
  simp_rw [rows]
  apply Finset.sum_subset (Finset.filter_subset _ _)
  intro cell _ absent
  apply Finset.sum_eq_zero
  intro decision _
  rw [missing_or_unavailable_rows_have_zero_coefficient program cell decision (Or.inl absent),mul_zero]

theorem original_requested_nll_row_coefficient_mass_is_one
    (program : Program Cell Decision) (defined : available program≠∅) :
    (∑ cell ∈ program.cells,∑ decision ∈ program.decisions cell,coefficient program cell decision)=1 := by
  have extension := extending_to_original_requests_adds_only_zero_weight_terms program (fun _ _ => 1)
  simp only [one_mul] at extension
  rw [←extension]
  exact conditional_macro_coefficient_mass_is_one program defined

theorem returned_conditional_nll_macro_has_the_actual_row_weight_expansion
    (program : Program Cell Decision) (defined : available program≠∅) :
    reported program=some
      (∑ cell ∈ program.cells,∑ decision ∈ program.decisions cell,
        value program cell decision*coefficient program cell decision) := by
  rw [reported,if_neg defined,nested_macro_is_derived_weighted_sum_of_original_scored_rows,
    extending_to_original_requests_adds_only_zero_weight_terms]

theorem empty_available_nll_domain_has_no_macro (program : Program Cell Decision)
    (empty : available program=∅) : reported program=none := by
  simp [reported,empty]

end
end N2P3.RequestedNLLMacroWeights
