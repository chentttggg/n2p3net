import N2P3.DecisionLossReduction

/-! Differentiate the actual prefix-mean and decision-mean risk formula.
The gradient is derived from raw mathematical evidence directions, rather
than assumed as a correct result of an automatic differentiation operation. -/
namespace N2P3.DecisionLossChain
open scoped BigOperators
open N2P3.CandidateProbability N2P3.DecisionLossCalculus N2P3.DecisionLossReduction
noncomputable section
variable {D C : Type*} [Fintype D] [Fintype C] {R : ℕ}

def tensorLine (evidence direction : D → C → Fin R → ℝ) (x : ℝ)
    (decision : D) (candidate : C) (row : Fin R) : ℝ :=
  evidence decision candidate row + x * direction decision candidate row

omit [Fintype D] [Fintype C] in
theorem prefix_sum_of_raw_evidence_line (evidence direction : D → C → Fin R → ℝ)
    (x : ℝ) (decision : D) (candidate : C) (level : Fin R) :
    prefixSum (tensorLine evidence direction x) decision candidate level =
      prefixSum evidence decision candidate level + x * prefixSum direction decision candidate level := by
  simp [prefixSum,tensorLine,Finset.sum_add_distrib,Finset.mul_sum]

omit [Fintype D] [Fintype C] in
theorem prefix_scores_of_raw_evidence_line (evidence direction : D → C → Fin R → ℝ)
    (x temperature : ℝ) (decision : D) (level : Fin R) :
    prefixScores (tensorLine evidence direction x) temperature decision level =
      scoreLine (prefixScores evidence temperature decision level)
        (prefixScores direction temperature decision level) x := by
  funext candidate
  simp only [prefixScores,prefix_sum_of_raw_evidence_line,scoreLine,add_div]
  ring

omit [Fintype D] in
theorem prefix_ce_derivative_from_raw_evidence [Nonempty C]
    (evidence direction : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C)
    (decision : D) (level : Fin R) :
    HasDerivAt (fun t => prefixNLL (tensorLine evidence direction t) temperature truth decision level)
      ((∑ candidate, probability (prefixScores evidence temperature decision level) candidate *
        prefixScores direction temperature decision level candidate) -
        prefixScores direction temperature decision level (truth decision)) 0 := by
  have zeroLine : scoreLine (prefixScores evidence temperature decision level)
      (prefixScores direction temperature decision level) 0 = prefixScores evidence temperature decision level := by
    funext candidate
    simp [scoreLine]
  have h := ce_directional_derivative (prefixScores evidence temperature decision level)
    (prefixScores direction temperature decision level) (truth decision) 0
  rw [zeroLine] at h
  convert h using 1
  funext t
  rw [prefixNLL,prefix_scores_of_raw_evidence_line]

def prefixDerivative (evidence direction : D → C → Fin R → ℝ) (temperature : ℝ)
    (truth : D → C) (decision : D) (level : Fin R) : ℝ :=
  (∑ candidate, probability (prefixScores evidence temperature decision level) candidate *
    prefixScores direction temperature decision level candidate) -
    prefixScores direction temperature decision level (truth decision)

theorem packed_ce_derivative_from_raw_evidence [Nonempty C]
    (evidence direction : D → C → Fin R → ℝ) (temperature : ℝ) (truth : D → C)
    (weights : Fin R → ℝ) :
    HasDerivAt (fun t => packedLoss (tensorLine evidence direction t) temperature truth weights)
      (∑ level, weights level * ((∑ decision,
        prefixDerivative evidence direction temperature truth decision level) /
        (Fintype.card D : ℝ))) 0 := by
  exact HasDerivAt.fun_sum (u := Finset.univ) (fun level _ =>
    ((HasDerivAt.fun_sum (u := Finset.univ) (fun decision _ =>
      prefix_ce_derivative_from_raw_evidence evidence direction temperature truth decision level)).div_const
        (Fintype.card D : ℝ)).const_mul (weights level))

end
end N2P3.DecisionLossChain
