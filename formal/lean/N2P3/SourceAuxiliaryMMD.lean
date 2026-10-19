import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases

/-! Finite off-diagonal empirical MMD. The three matrices are the within-left,
within-right and cross RBF evaluations at a fixed positive detached bandwidth.
The same-batch bandwidth is held fixed for differentiation, without an assertion
of independent sampling. Finite empirical unbiased-estimator values may be negative. -/
namespace N2P3.SourceAuxiliaryMMD
open scoped BigOperators
noncomputable section

def offDiagonalMean {ι : Type*} [Fintype ι] [DecidableEq ι] (k : ι → ι → ℝ) : ℝ :=
  (∑ i, ∑ j, if i = j then 0 else k i j) /
    ((Fintype.card ι : ℝ) * ((Fintype.card ι : ℝ) - 1))

def crossMean {ι κ : Type*} [Fintype ι] [Fintype κ] (k : ι → κ → ℝ) : ℝ :=
  (∑ i, ∑ j, k i j) / ((Fintype.card ι : ℝ) * (Fintype.card κ : ℝ))

def empiricalMMD {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ]
    (left : ι → ι → ℝ) (right : κ → κ → ℝ) (cross : ι → κ → ℝ) : ℝ :=
  offDiagonalMean left + offDiagonalMean right - 2 * crossMean cross

theorem cross_mean_transpose {ι κ : Type*} [Fintype ι] [Fintype κ] (k : ι → κ → ℝ) :
    crossMean (fun j i => k i j) = crossMean k := by
  simp only [crossMean]
  rw [Finset.sum_comm, mul_comm (Fintype.card κ : ℝ)]

theorem empirical_mmd_swap {ι κ : Type*} [Fintype ι] [DecidableEq ι]
    [Fintype κ] [DecidableEq κ] (l : ι → ι → ℝ) (r : κ → κ → ℝ) (c : ι → κ → ℝ) :
    empiricalMMD r l (fun j i => c i j) = empiricalMMD l r c := by
  simp only [empiricalMMD, cross_mean_transpose]
  ring

def twoPointKernel (off : ℝ) (i j : Fin 2) : ℝ := if i = j then 1 else off

theorem two_point_off_diagonal (off : ℝ) : offDiagonalMean (twoPointKernel off) = off := by
  simp [offDiagonalMean, twoPointKernel, Fin.sum_univ_two]
  ring

theorem two_point_cross_mean (off : ℝ) : crossMean (twoPointKernel off) = (1 + off) / 2 := by
  simp [crossMean, twoPointKernel, Fin.sum_univ_two]
  ring

theorem identical_two_point_empirical_mmd (off : ℝ) :
    empiricalMMD (twoPointKernel off) (twoPointKernel off) (twoPointKernel off) = off - 1 := by
  rw [empiricalMMD, two_point_off_diagonal, two_point_cross_mean]
  ring

theorem identical_nonconstant_rbf_empirical_mmd_negative (a : ℝ) (ha : 0 < a) :
    empiricalMMD (twoPointKernel (Real.exp (-a))) (twoPointKernel (Real.exp (-a)))
      (twoPointKernel (Real.exp (-a))) < 0 := by
  rw [identical_two_point_empirical_mmd]
  have h : Real.exp (-a) < 1 := by rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (neg_neg_of_pos ha)
  linarith

theorem identical_constant_empirical_mmd_zero :
    empiricalMMD (twoPointKernel 1) (twoPointKernel 1) (twoPointKernel 1) = 0 := by
  rw [identical_two_point_empirical_mmd]; norm_num

def scalarRBF (sigma x y : ℝ) : ℝ := Real.exp (-((x-y)^2) / (2*sigma^2))

def twoPoints (distance : ℝ) (i : Fin 2) : ℝ := if i = 0 then 0 else distance

theorem two_point_rbf_matrix (distance sigma : ℝ) :
    (fun i j => scalarRBF sigma (twoPoints distance i) (twoPoints distance j)) =
      twoPointKernel (Real.exp (-(distance^2)/(2*sigma^2))) := by
  funext i j
  fin_cases i <;> fin_cases j <;> simp [scalarRBF, twoPoints, twoPointKernel]

theorem actual_identical_two_point_rbf_mmd_negative (distance sigma : ℝ)
    (hd : distance ≠ 0) (hs : 0 < sigma) :
    empiricalMMD
      (fun i j => scalarRBF sigma (twoPoints distance i) (twoPoints distance j))
      (fun i j => scalarRBF sigma (twoPoints distance i) (twoPoints distance j))
      (fun i j => scalarRBF sigma (twoPoints distance i) (twoPoints distance j)) < 0 := by
  rw [two_point_rbf_matrix]
  have ha : 0 < distance^2/(2*sigma^2) := div_pos (sq_pos_of_ne_zero hd) (by positivity)
  simpa only [neg_div] using identical_nonconstant_rbf_empirical_mmd_negative
    (distance^2/(2*sigma^2)) ha

/-- Warmup target weighting is interpolation, not a normalized mixture of losses. -/
def targetWeight (alpha clipped : ℝ) : ℝ := 1 + alpha * (clipped - 1)

theorem target_weight_start (clipped : ℝ) : targetWeight 0 clipped = 1 := by simp [targetWeight]

theorem target_weight_warm (clipped : ℝ) : targetWeight 1 clipped = clipped := by
  unfold targetWeight; ring

theorem target_weight_between (alpha clipped : ℝ) (ha : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hc : 1 ≤ clipped) : 1 ≤ targetWeight alpha clipped ∧ targetWeight alpha clipped ≤ clipped := by
  unfold targetWeight; constructor <;> nlinarith

end
end N2P3.SourceAuxiliaryMMD
