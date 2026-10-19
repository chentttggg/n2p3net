import N2P3.RequestedOutcomeRates
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Linarith

/-! Ideal real-valued log-evidence and actual Option-valued scored domain of
run_bnci_continuous_study._cell. Saved floating probabilities may underflow;
this does not refine native exp/logsumexp or serialized probabilities. A tie
may have an NLL, so scoredness is deliberately independent of correct status.
The uniform convention is separately named, never called a model output. -/

namespace N2P3.RequestedNLL

open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable

def normalizer {K : ℕ} (evidence : Fin K → ℝ) : ℝ :=
  ∑ candidate, Real.exp (evidence candidate)

def targetProbability {K : ℕ} (evidence : Fin K → ℝ) (target : Fin K) : ℝ :=
  Real.exp (evidence target)/normalizer evidence

def stableNLL {K : ℕ} (evidence : Fin K → ℝ) (target : Fin K) : ℝ :=
  Real.log (normalizer evidence)-evidence target

theorem log_evidence_normalizer_is_positive {K : ℕ} (evidence : Fin K → ℝ) (target : Fin K) :
    0 < normalizer evidence := by
  apply Finset.sum_pos'
  · intro candidate _
    exact le_of_lt (Real.exp_pos (evidence candidate))
  · exact ⟨target,Finset.mem_univ target,Real.exp_pos (evidence target)⟩

theorem ideal_target_probability_is_positive_and_at_most_one {K : ℕ}
    (evidence : Fin K → ℝ) (target : Fin K) :
    0 < targetProbability evidence target ∧ targetProbability evidence target ≤ 1 := by
  have positive := log_evidence_normalizer_is_positive evidence target
  refine ⟨div_pos (Real.exp_pos _) positive,?_⟩
  apply (div_le_one positive).mpr
  exact Finset.single_le_sum (fun candidate _ => le_of_lt (Real.exp_pos (evidence candidate)))
    (Finset.mem_univ target)

theorem stable_log_evidence_nll_equals_ideal_negative_log_probability {K : ℕ}
    (evidence : Fin K → ℝ) (target : Fin K) :
    stableNLL evidence target = -Real.log (targetProbability evidence target) := by
  have positive := log_evidence_normalizer_is_positive evidence target
  rw [targetProbability,Real.log_div (ne_of_gt (Real.exp_pos _)) (ne_of_gt positive),Real.log_exp]
  unfold stableNLL
  ring

theorem stable_log_evidence_nll_is_nonnegative {K : ℕ}
    (evidence : Fin K → ℝ) (target : Fin K) : 0 ≤ stableNLL evidence target := by
  rw [stable_log_evidence_nll_equals_ideal_negative_log_probability]
  have bounds := ideal_target_probability_is_positive_and_at_most_one evidence target
  exact neg_nonneg.mpr (Real.log_nonpos (le_of_lt bounds.1) bounds.2)

def scoredValues (rows : List (Option ℝ)) : List ℝ := rows.filterMap id

def missingCount (rows : List (Option ℝ)) : ℕ := (rows.filter Option.isNone).length

def scoredMean (rows : List (Option ℝ)) : Option ℝ :=
  if scoredValues rows=[] then none
  else some ((scoredValues rows).sum/((scoredValues rows).length : ℝ))

def uniformRequestedNLL (K : ℕ) (rows : List (Option ℝ)) : Option ℝ :=
  if rows=[] then none
  else some (((scoredValues rows).sum+(missingCount rows : ℝ)*Real.log K)/(rows.length : ℝ))

theorem actual_scored_and_missing_counts_partition_original_rows (rows : List (Option ℝ)) :
    (scoredValues rows).length+missingCount rows=rows.length := by
  induction rows with
  | nil => simp [scoredValues,missingCount]
  | cons head rest ih =>
    cases head <;> simp_all [scoredValues,missingCount] <;> omega

theorem no_scored_rows_have_no_conditional_model_nll (rows : List (Option ℝ))
    (missing : scoredValues rows=[]) : scoredMean rows=none := by
  simp [scoredMean,missing]

theorem scored_mean_retains_only_actual_some_outputs (values : List ℝ) :
    scoredValues (values.map some)=values ∧ missingCount (values.map some)=0 := by
  induction values with
  | nil => simp [scoredValues,missingCount]
  | cons head rest ih => simp_all [scoredValues,missingCount]

theorem missing_rows_do_not_change_conditional_nll (rows : List (Option ℝ)) :
    scoredMean (none::rows)=scoredMean rows := by
  simp [scoredMean,scoredValues]

theorem uniform_convention_uses_original_requested_row_count (K : ℕ) (rows : List (Option ℝ))
    (requested : rows ≠ []) :
    uniformRequestedNLL K rows=some
      (((scoredValues rows).sum+(missingCount rows : ℝ)*Real.log K)/
        (((scoredValues rows).length+missingCount rows : ℕ) : ℝ)) := by
  rw [actual_scored_and_missing_counts_partition_original_rows]
  simp [uniformRequestedNLL,requested]

theorem fixed_nine_choice_uniform_missing_nll_differs_from_model_nll :
    scoredMean [some (-Real.log (9/10)),none]=some (-Real.log (9/10)) ∧
    uniformRequestedNLL 9 [some (-Real.log (9/10)),none]=
      some ((-Real.log (9/10)+Real.log 9)/2) ∧
    -Real.log (9/10) < (-Real.log (9/10)+Real.log 9)/2 := by
  have logs : Real.log (10/9) < Real.log 9 :=
    Real.log_lt_log (by norm_num) (by norm_num)
  have inverse : Real.log (10/9) = -Real.log (9/10) := by
    have identity := Real.log_inv (9/10 : ℝ)
    norm_num at identity ⊢
    exact identity
  rw [inverse] at logs
  refine ⟨?_,?_,by linarith⟩
  · simp [scoredMean,scoredValues]
  · simp [uniformRequestedNLL,scoredValues,missingCount]

end
end N2P3.RequestedNLL
