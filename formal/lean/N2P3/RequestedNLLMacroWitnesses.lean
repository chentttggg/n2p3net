import N2P3.RequestedNLLMacroWeights

/-! Fixed N01-N03 actual Some-domain witnesses; coefficients themselves are
computed by the source mean operator, not installed in the program record. -/

namespace N2P3.RequestedNLLMacroWitnesses

open N2P3.RequestedNLLMacroWeights
open N2P3.RequestedAggregationWeights (mean)

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def program : Program ℕ ℕ where
  cells := {0,1}
  decisions := fun cell => if cell=0 then {0,1} else {0,1,2}
  completed := fun _ => true
  nll := fun cell decision => if cell=0 then if decision=0 then some 1 else none else some 3

theorem actual_some_domains_retain_the_partially_scored_cell :
    scored program 0={0} ∧ scored program 1={0,1,2} ∧ available program={0,1} := by
  simp [scored,available,program,Finset.filter_insert,Finset.filter_singleton]

theorem actual_macro_nll_row_coefficients_are_not_pooled_weights :
    coefficient program 0 0=1/2 ∧ coefficient program 1 0=1/6 ∧ coefficient program 0 1=0 := by
  norm_num [coefficient_is_derived_from_the_conditional_mean_operator,
    actual_some_domains_retain_the_partially_scored_cell.1,
    actual_some_domains_retain_the_partially_scored_cell.2.1,
    actual_some_domains_retain_the_partially_scored_cell.2.2]

theorem conditional_nll_macro_is_two_while_pooled_decision_mean_is_five_halves :
    reported program=some 2 ∧ ((1+3+3+3 : ℝ)/4)=5/2 := by
  have outerMean (values : ℕ → ℝ) : mean ({0,1} : Finset ℕ) values=(values 0+values 1)/2 := by
    simp [mean]
  constructor
  · simp only [reported,linearMean,actual_some_domains_retain_the_partially_scored_cell.2.2,
      Finset.insert_ne_empty,if_false]
    rw [outerMean,actual_some_domains_retain_the_partially_scored_cell.1,
      actual_some_domains_retain_the_partially_scored_cell.2.1]
    norm_num [mean,value,program]
  · norm_num

def emptyProgram : Program ℕ ℕ := {program with nll := fun _ _ => none}

theorem no_scored_cell_has_no_conditional_macro_nll : reported emptyProgram=none := by
  simp [reported,available,scored,emptyProgram,program,Finset.filter_insert,Finset.filter_singleton]

end
end N2P3.RequestedNLLMacroWitnesses
