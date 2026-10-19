import N2P3.CandidateProbability
import Mathlib.Tactic.FinCases

/-!
Strict negative controls for two unjustified probability claims. All concrete
score/prior calculations are proved first; each false equality must leave one
False goal, without an auxiliary import, type, tactic, or helper error.
-/

namespace N2P3.CandidateProbability.Negative

open scoped BigOperators

private def equalScores (_ : Fin 2) : ℝ := 0

private noncomputable def unequalOffset (i : Fin 2) : ℝ := if i = 0 then 0 else Real.log 3

private noncomputable def unequalPrior (i : Fin 2) : ℝ := if i = 0 then 3 / 4 else 1 / 4

private theorem equal_probability_value : probability equalScores 0 = 1 / 2 := by
  rw [probability_eq_exp_div_partition]
  norm_num [partition, equalScores, Fin.sum_univ_succ]

private theorem unequal_offset_probability_value : probability unequalOffset 0 = 1 / 4 := by
  rw [probability_eq_exp_div_partition]
  norm_num [partition, unequalOffset, Fin.sum_univ_succ, Real.exp_log (by norm_num : (0 : ℝ) < 3)]

private theorem unequal_prior_is_legal : IsPrior unequalPrior := by
  constructor
  · intro i
    fin_cases i <;> norm_num [unequalPrior]
  · norm_num [unequalPrior, Fin.sum_univ_succ]

private theorem unequal_prior_posterior_value : posterior unequalPrior equalScores 0 = 3 / 4 := by
  change posterior unequalPrior (fun _ => 0) 0 = 3 / 4
  rw [constant_scores_posterior_eq_prior unequalPrior unequal_prior_is_legal]
  norm_num [unequalPrior]

/-- False: a candidate-dependent offset can be dropped from normalization. -/
example : probability unequalOffset 0 = probability equalScores 0 := by
  rw [unequal_offset_probability_value, equal_probability_value]
  norm_num

/-- False: equal likelihood scores override a nonuniform candidate prior. -/
example : posterior unequalPrior equalScores 0 = probability equalScores 0 := by
  rw [unequal_prior_posterior_value, equal_probability_value]
  norm_num

end N2P3.CandidateProbability.Negative
