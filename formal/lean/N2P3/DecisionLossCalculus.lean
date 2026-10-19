import N2P3.CandidateProbability
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-! Exact-real CE derivatives for decision_loss.py:712,873-876,946-949.
The scores are mathematical tensors; no native AD correctness is assumed or
proved here. The source contract predates proof execution. -/
namespace N2P3.DecisionLossCalculus
open scoped BigOperators
open N2P3.CandidateProbability
noncomputable section
variable {κ : Type*} [Fintype κ] [Nonempty κ]

def scoreLine (scores direction : κ → ℝ) (x : ℝ) (i : κ) : ℝ :=
  scores i + x * direction i

omit [Fintype κ] [Nonempty κ] in
theorem score_line_derivative (scores direction : κ → ℝ) (x : ℝ) (i : κ) :
    HasDerivAt (fun t => scoreLine scores direction t i) (direction i) x := by
  simpa [scoreLine] using ((hasDerivAt_id x).mul_const (direction i)).const_add (scores i)

omit [Nonempty κ] in
theorem partition_line_derivative (scores direction : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => partition (scoreLine scores direction t))
      (∑ i, Real.exp (scoreLine scores direction x i) * direction i) x := by
  exact HasDerivAt.fun_sum (u := Finset.univ)
    (fun i _ => (score_line_derivative scores direction x i).exp)

theorem log_partition_line_derivative (scores direction : κ → ℝ) (x : ℝ) :
    HasDerivAt (fun t => logPartition (scoreLine scores direction t))
      (∑ i, probability (scoreLine scores direction x) i * direction i) x := by
  have h := (partition_line_derivative scores direction x).log
    (partition_positive (scoreLine scores direction x)).ne'
  convert h using 1
  simp only [probability_eq_exp_div_partition, div_eq_mul_inv, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem ce_directional_derivative (scores direction : κ → ℝ) (truth : κ) (x : ℝ) :
    HasDerivAt (fun t => nll (scoreLine scores direction t) truth)
      ((∑ i, probability (scoreLine scores direction x) i * direction i) - direction truth) x := by
  exact (log_partition_line_derivative scores direction x).sub
    (score_line_derivative scores direction x truth)

theorem ce_actual_derivative (scores direction : κ → ℝ) (truth : κ) (x : ℝ) :
    deriv (fun t => nll (scoreLine scores direction t) truth) x =
      (∑ i, probability (scoreLine scores direction x) i * direction i) - direction truth :=
  (ce_directional_derivative scores direction truth x).deriv

theorem ce_single_score_derivative [DecidableEq κ] (scores : κ → ℝ) (truth selected : κ) :
    HasDerivAt (fun t => nll (scoreLine scores (fun i => if i=selected then 1 else 0) t) truth)
      (probability scores selected - if truth=selected then 1 else 0) 0 := by
  have zeroLine : scoreLine scores (fun i => if i=selected then 1 else 0) 0 = scores := by
    funext i
    simp [scoreLine]
  simpa [zeroLine] using
    ce_directional_derivative scores (fun i => if i=selected then 1 else 0) truth 0

theorem tied_score_ce (value : ℝ) (truth : κ) :
    nll (fun _ : κ => value) truth = Real.log (Fintype.card κ : ℝ) := by
  have shift := nll_common_constant (fun _ : κ => (0 : ℝ)) value truth
  simpa [nll,logPartition,partition] using shift

theorem tied_score_ce_positive (value : ℝ) (truth : κ) (atLeastTwo : 1 < Fintype.card κ) :
    0 < nll (fun _ : κ => value) truth := by
  rw [tied_score_ce]
  apply Real.log_pos
  exact_mod_cast atLeastTwo

end
end N2P3.DecisionLossCalculus
