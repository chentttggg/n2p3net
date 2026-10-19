import N2P3.DecisionLossCalculus

/-! decision_loss.py:935-951: candidate-major cumsum, divide by level
length, CE across candidates, mean across decisions, then level weights.
Only exact-real tensor arithmetic is claimed, not native reshaping/AD. -/
namespace N2P3.DecisionLossReduction
open scoped BigOperators
open N2P3.CandidateProbability
noncomputable section
variable {D C : Type*} [Fintype D] [Fintype C] {R : ℕ}

def prefixSum (evidence : D → C → Fin R → ℝ) (decision : D) (candidate : C)
    (level : Fin R) : ℝ :=
  ∑ row ∈ Finset.univ.filter (fun row : Fin R => row ≤ level), evidence decision candidate row

def prefixScores (evidence : D → C → Fin R → ℝ) (temperature : ℝ)
    (decision : D) (level : Fin R) (candidate : C) : ℝ :=
  (prefixSum evidence decision candidate level / (level.val + 1 : ℕ)) / temperature

def prefixNLL (evidence : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C)
    (decision : D) (level : Fin R) : ℝ :=
  nll (prefixScores evidence temperature decision level) (truth decision)

def packedLoss (evidence : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C)
    (weights : Fin R → ℝ) : ℝ :=
  ∑ level, weights level * ((∑ decision, prefixNLL evidence temperature truth decision level) /
    (Fintype.card D : ℝ))

omit [Fintype D] [Fintype C] in
theorem prefix_reads_only_its_earlier_rows
    (a b : D → C → Fin R → ℝ) (temperature : ℝ) (decision : D) (level : Fin R)
    (same : ∀ candidate row, row ≤ level → a decision candidate row = b decision candidate row) :
    prefixScores a temperature decision level = prefixScores b temperature decision level := by
  funext candidate
  unfold prefixScores
  congr 2
  apply Finset.sum_congr rfl
  intro row member
  exact same candidate row (Finset.mem_filter.mp member).2

theorem packed_is_decision_mean_of_prefix_risks
    (evidence : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C)
    (weights : Fin R → ℝ) :
    packedLoss evidence temperature truth weights =
      (∑ decision, ∑ level, weights level * prefixNLL evidence temperature truth decision level) /
        (Fintype.card D : ℝ) := by
  unfold packedLoss
  simp only [div_eq_mul_inv, ← mul_assoc, Finset.mul_sum]
  rw [← Finset.sum_mul, Finset.sum_comm]

theorem packed_loss_nonnegative [Nonempty C]
    (evidence : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C)
    (weights : Fin R → ℝ) (nonnegative : ∀ level, 0 ≤ weights level) :
    0 ≤ packedLoss evidence temperature truth weights := by
  apply Finset.sum_nonneg
  intro level _
  apply mul_nonneg (nonnegative level)
  apply div_nonneg
  · exact Finset.sum_nonneg (fun decision _ => nll_nonnegative _ _)
  · positivity

theorem final_only_loss_is_actual_last_prefix
    (evidence : D → C → Fin (R+1) → ℝ) (temperature : ℝ) (truth : D → C) :
    packedLoss evidence temperature truth (fun level => if level=Fin.last R then 1 else 0) =
      (∑ decision, prefixNLL evidence temperature truth decision (Fin.last R)) /
        (Fintype.card D : ℝ) := by
  classical
  simp [packedLoss]

theorem uniform_loss_averages_all_prefixes
    (evidence : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C) :
    packedLoss evidence temperature truth (fun _ => 1 / (R : ℝ)) =
      ((∑ level, ∑ decision, prefixNLL evidence temperature truth decision level) /
        (Fintype.card D : ℝ)) / (R : ℝ) := by
  unfold packedLoss
  simp_rw [one_div, mul_comm (R : ℝ)⁻¹, ← div_eq_mul_inv, div_div]
  simp only [div_eq_mul_inv, Finset.sum_mul]

omit [Fintype D] in
theorem zero_scores_have_positive_trainable_loss [Nonempty C]
    (truth : D → C) (decision : D) (level : Fin R) (temperature : ℝ)
    (atLeastTwo : 1 < Fintype.card C) :
    0 < prefixNLL (fun _ _ _ => 0) temperature truth decision level := by
  have zeroScores : prefixScores (fun _ _ _ => (0 : ℝ)) temperature decision level =
      (fun _ : C => 0) := by
    funext candidate
    simp [prefixScores,prefixSum]
  rw [prefixNLL,zeroScores]
  exact N2P3.DecisionLossCalculus.tied_score_ce_positive 0 (truth decision) atLeastTwo

end
end N2P3.DecisionLossReduction
