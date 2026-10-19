import N2P3.DecisionLossChain

/-! Wrong alternatives fixed in source_contract_01, with inhabited concrete
decision/candidate/repetition sets and no contradictory input hypotheses. -/
namespace N2P3.NegativeDecisionLoss
open N2P3.CandidateProbability N2P3.DecisionLossCalculus N2P3.DecisionLossReduction
noncomputable section

example : deriv (fun t => nll (scoreLine (fun _ : Fin 2 => 0)
    (fun i => if i=0 then 1 else 0) t) 0) 0 = (1/2 : ℝ) := by
  rw [ce_actual_derivative]
  norm_num [scoreLine,probability_eq_exp_div_partition,partition,Fin.sum_univ_two]

example : prefixScores (fun _ : Unit => fun _ : Fin 2 => fun row : Fin 2 =>
    if row=0 then 0 else 5) 1 () 0 0 = 5 := by
  norm_num [prefixScores,prefixSum,Finset.sum_filter,Fin.sum_univ_two]

example : prefixScores (fun _ : Unit => fun _ : Fin 2 => fun _ : Fin 2 => (2 : ℝ))
    1 () 1 0 = 4 := by
  have allRows : Finset.univ.filter (fun row : Fin 2 => row ≤ (1 : Fin 2)) = Finset.univ := by decide
  norm_num [prefixScores,prefixSum,allRows,Fin.sum_univ_two]

example : nll (fun _ : Fin 2 => (0 : ℝ)) 0 = 0 := by
  have positive := tied_score_ce_positive (κ := Fin 2) 0 0 (by decide)
  have different : nll (fun _ : Fin 2 => (0 : ℝ)) 0 ≠ 0 := ne_of_gt positive
  try simp only [different]

end
end N2P3.NegativeDecisionLoss
