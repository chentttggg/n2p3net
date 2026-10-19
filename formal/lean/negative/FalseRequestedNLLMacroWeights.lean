import N2P3.RequestedNLLMacroWitnesses

namespace N2P3.NegativeRequestedNLLMacroWeights

open N2P3.RequestedNLLMacroWeights N2P3.RequestedNLLMacroWitnesses

-- N01: unequal actual scored sets make macro and pooled NLL different.
example : reported program=some (5/2) := by
  rw [conditional_nll_macro_is_two_while_pooled_decision_mean_is_five_halves.1]
  norm_num

-- N02: one missing decision does not zero the whole otherwise scored cell.
example : coefficient program 0 0=0 := by
  rw [actual_macro_nll_row_coefficients_are_not_pooled_weights.1]
  norm_num

-- N03: no actual scores means None, not a reported model NLL of0.
example : reported emptyProgram=some 0 := by
  rw [no_scored_cell_has_no_conditional_macro_nll]
  simp

end N2P3.NegativeRequestedNLLMacroWeights
