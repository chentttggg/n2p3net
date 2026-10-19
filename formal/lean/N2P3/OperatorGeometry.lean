import N2P3.RetainedBasis
import Mathlib.LinearAlgebra.Matrix.Spectrum
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Real.StarOrdered
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases

/-!
# The source OperatorAwareLDA retained geometry and actual linear score

Source: operator_lda.py:42-68,132-149. The source truncates at the STRICT
relative eigenvalue threshold, adds ridge only in the retained subspace, and
solves a projected contrast. Its tolerance tests do not prove exact support.

An exact symmetric matrix has a genuine spectral representation below, derived
from the pinned Mathlib spectral theorem. No goal equation about weights,
scoring or reconstructed state is a field of a model or an input premise.
Numerical eigh accuracy, float threshold/eigenvalue multiplicities and physical
noise validity are separate. In particular covariance-tolerance acceptance is
never used to infer exact positive semidefiniteness of a raw input matrix.

Scores below are on retained coordinates; singular ambient covariance does
not acquire a full-dimensional Gaussian density. The diagnostic feature gate
uses the Frobenius norm of the complete centered batch, as in NumPy's default
matrix norm; it is not replaced by per-row admission.
-/

namespace N2P3.OperatorGeometry

open Matrix
open scoped BigOperators Matrix

noncomputable section

variable {O K : Type*} [Fintype O] [Fintype K] [DecidableEq O] [DecidableEq K]

def eigenTolerance : ℝ := 1 / 10000000000
def supportTolerance : ℝ := 1 / 1000000
def supportTiny : ℝ := (1 / 2 : ℝ) ^ 1022

abbrev Retained {I : Type*} (values : I → ℝ) (maximum : ℝ) :=
  {i : I // maximum * eigenTolerance < values i}

theorem eigen_tolerance_positive : 0 < eigenTolerance := by norm_num [eigenTolerance]

theorem eigen_tolerance_below_one : eigenTolerance < 1 := by norm_num [eigenTolerance]

theorem positive_maximum_is_retained {I : Type*} (values : I → ℝ) (maximum : ℝ)
    (top : I) (htop : values top = maximum) (hm : 0 < maximum) :
    maximum * eigenTolerance < values top := by
  rw [htop]
  nlinarith [eigen_tolerance_below_one]

theorem retained_has_positive_value {I : Type*} (values : I → ℝ) (maximum : ℝ)
    (hm : 0 < maximum) (i : Retained values maximum) : 0 < values i.val := by
  exact lt_trans (mul_pos hm eigen_tolerance_positive) i.property

theorem retained_nonempty_from_source_maximum {I : Type*} (values : I → ℝ) (maximum : ℝ)
    (top : I) (htop : values top = maximum) (hm : 0 < maximum) :
    Nonempty (Retained values maximum) :=
  ⟨⟨top, positive_maximum_is_retained values maximum top htop hm⟩⟩

/-- Numeric value exactly at the source threshold is discarded. -/
theorem threshold_equality_is_discarded {I : Type*} (values : I → ℝ) (maximum : ℝ)
    (i : I) (h : values i = maximum * eigenTolerance) :
    ¬maximum * eigenTolerance < values i := by rw [h]; exact lt_irrefl _

def thinColumns {I : Type*} (U : Matrix O I ℝ) (values : I → ℝ) (maximum : ℝ) :
    Matrix O (Retained values maximum) ℝ := fun i j => U i j.val

omit [DecidableEq O] in
theorem retained_columns_orthonormal {I : Type*} [Fintype I] [DecidableEq I]
    (U : Matrix O I ℝ) (values : I → ℝ) (maximum : ℝ) (hU : Uᵀ * U = 1) :
    (thinColumns U values maximum)ᵀ * thinColumns U values maximum = 1 := by
  classical
  ext i j
  have h := congrArg (fun A : Matrix I I ℝ => A i.val j.val) hU
  simpa [thinColumns, Matrix.mul_apply, Matrix.transpose_apply, Matrix.one_apply,
    Subtype.ext_iff] using h

/-- Exact real spectral data exists; it is not an assumed output equation. -/
theorem exists_actual_spectral_representation (C : Matrix O O ℝ) (hC : C.IsHermitian) :
    ∃ (U : Matrix O O ℝ) (values : O → ℝ),
      Uᵀ * U = 1 ∧ U * Uᵀ = 1 ∧ C = U * diagonal values * Uᵀ := by
  let U : Matrix O O ℝ := hC.eigenvectorUnitary
  refine ⟨U, hC.eigenvalues, ?_, ?_, ?_⟩
  · simpa only [U, star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial] using
      (unitary.coe_star_mul_self hC.eigenvectorUnitary)
  · simpa only [U, star_eq_conjTranspose, Matrix.conjTranspose_eq_transpose_of_trivial] using
      (unitary.coe_mul_star_self hC.eigenvectorUnitary)
  · simpa [U] using hC.spectral_theorem

def retainedMean (values : K → ℝ) : ℝ := (∑ k, values k) / (Fintype.card K : ℝ)

def regularized (values : K → ℝ) (rho : ℝ) : K → ℝ :=
  fun k => values k + rho * retainedMean values

def projector (V : Matrix O K ℝ) : Matrix O O ℝ := V * Vᵀ

def regularizedCovariance (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ) : Matrix O O ℝ :=
  V * diagonal (regularized values rho) * Vᵀ

def weights (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ) (delta : O → ℝ) : O → ℝ :=
  V *ᵥ (fun k => (Vᵀ *ᵥ delta) k / regularized values rho k)

def intercept (mu0 mu1 w : O → ℝ) : ℝ := -((mu0 + mu1) ⬝ᵥ w) / 2

def logit (x w : O → ℝ) (b : ℝ) : ℝ := x ⬝ᵥ w + b

omit [DecidableEq K] in
theorem retained_mean_positive [Nonempty K] (values : K → ℝ) (hp : ∀ k, 0 < values k) :
    0 < retainedMean values := by
  unfold retainedMean
  apply div_pos
  · exact Finset.sum_pos (fun k _ => hp k) Finset.univ_nonempty
  · exact_mod_cast Fintype.card_pos

omit [DecidableEq K] in
theorem regularized_positive [Nonempty K] (values : K → ℝ) (rho : ℝ)
    (hp : ∀ k, 0 < values k) (hr : 0 ≤ rho) (k : K) :
    0 < regularized values rho k := by
  exact add_pos_of_pos_of_nonneg (hp k) (mul_nonneg hr (retained_mean_positive values hp).le)

omit [Fintype O] [DecidableEq O] [DecidableEq K] in
theorem projector_symmetric (V : Matrix O K ℝ) : (projector V)ᵀ = projector V := by
  simp [projector, Matrix.transpose_mul]

omit [DecidableEq O] in
theorem projector_idempotent (V : Matrix O K ℝ) (hV : Vᵀ * V = 1) :
    projector V * projector V = projector V := by
  simp only [projector, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc Vᵀ V, hV, Matrix.one_mul]

omit [DecidableEq O] in
theorem weights_in_retained_range (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ)
    (delta : O → ℝ) (hV : Vᵀ * V = 1) :
    projector V *ᵥ weights V values rho delta = weights V values rho delta := by
  unfold weights projector
  rw [Matrix.mulVec_mulVec, Matrix.mul_assoc V Vᵀ V, hV, Matrix.mul_one]

omit [DecidableEq O] in
/-- This is the actual equation even when tolerance admits nonzero null contrast. -/
theorem regularized_weights_solve_projected_contrast (V : Matrix O K ℝ) (values : K → ℝ)
    (rho : ℝ) (delta : O → ℝ) (hV : Vᵀ * V = 1)
    (hr : ∀ k, regularized values rho k ≠ 0) :
    regularizedCovariance V values rho *ᵥ weights V values rho delta = projector V *ᵥ delta := by
  unfold regularizedCovariance weights projector
  rw [Matrix.mulVec_mulVec, Matrix.mul_assoc (V * diagonal (regularized values rho)) Vᵀ V,
    hV, Matrix.mul_one, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
  congr 1
  funext k
  rw [Matrix.mulVec_diagonal]
  field_simp [hr k]

omit [DecidableEq O] in
theorem exact_support_is_an_additional_equation (V : Matrix O K ℝ) (values : K → ℝ)
    (rho : ℝ) (delta : O → ℝ) (hV : Vᵀ * V = 1)
    (hr : ∀ k, regularized values rho k ≠ 0) (hd : projector V *ᵥ delta = delta) :
    regularizedCovariance V values rho *ᵥ weights V values rho delta = delta := by
  rw [regularized_weights_solve_projected_contrast V values rho delta hV hr, hd]

omit [DecidableEq O] in
theorem regularized_covariance_annihilates_discarded_vector (V : Matrix O K ℝ)
    (values : K → ℝ) (rho : ℝ) (x : O → ℝ) (hx : Vᵀ *ᵥ x = 0) :
    regularizedCovariance V values rho *ᵥ x = 0 := by
  unfold regularizedCovariance
  rw [← Matrix.mulVec_mulVec, hx, Matrix.mulVec_zero]

omit [DecidableEq O] [DecidableEq K] in
theorem transpose_dot_transport (V : Matrix O K ℝ) (x : O → ℝ) (q : K → ℝ) :
    x ⬝ᵥ (V *ᵥ q) = (Vᵀ *ᵥ x) ⬝ᵥ q := by
  simp only [Matrix.mulVec, dotProduct, Finset.mul_sum, Finset.sum_mul, Matrix.transpose_apply]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro i _
  ring

def retainedEnergy (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ) (x mu : O → ℝ) : ℝ :=
  ∑ k, ((Vᵀ *ᵥ (x - mu)) k)^2 / regularized values rho k

omit [DecidableEq O] [DecidableEq K] in
/-- Actual `X @ w - ((mu0+mu1) @ w)/2` is the retained Mahalanobis contrast.
No exact ambient support condition is needed for this algebraic formula. -/
theorem source_logit_is_retained_quadratic_contrast (V : Matrix O K ℝ) (values : K → ℝ)
    (rho : ℝ) (mu0 mu1 x : O → ℝ) :
    logit x (weights V values rho (mu1 - mu0)) (intercept mu0 mu1 (weights V values rho (mu1 - mu0))) =
      (retainedEnergy V values rho x mu0 - retainedEnergy V values rho x mu1) / 2 := by
  unfold logit intercept weights retainedEnergy
  rw [transpose_dot_transport, transpose_dot_transport]
  simp only [Matrix.mulVec_sub, Matrix.mulVec_add, Pi.sub_apply, Pi.add_apply, dotProduct]
  rw [neg_div, Finset.sum_div, ← sub_eq_add_neg, ← Finset.sum_sub_distrib]
  rw [← Finset.sum_sub_distrib, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro k _
  ring

def vectorNorm (x : O → ℝ) : ℝ := Real.sqrt (∑ i, (x i)^2)

def batchNorm {B : Type*} [Fintype B] (X : Matrix B O ℝ) : ℝ :=
  Real.sqrt (∑ row, ∑ i, (X row i)^2)

def contrastAccepted (V : Matrix O K ℝ) (delta : O → ℝ) : Prop :=
  vectorNorm (delta - projector V *ᵥ delta) ≤ max (vectorNorm delta) supportTiny * supportTolerance

def batchAccepted {B : Type*} [Fintype B] (V : Matrix O K ℝ) (centered : Matrix B O ℝ) : Prop :=
  batchNorm (centered - centered * projector V) ≤ max (batchNorm centered) supportTiny * supportTolerance

omit [DecidableEq O] [DecidableEq K] in
theorem exact_support_passes_source_contrast_gate (V : Matrix O K ℝ) (delta : O → ℝ)
    (h : projector V *ᵥ delta = delta) : contrastAccepted V delta := by
  unfold contrastAccepted vectorNorm
  rw [h, sub_self]
  simp only [Pi.zero_apply, zero_pow (by decide : (2 : ℕ) ≠ 0), Finset.sum_const_zero, Real.sqrt_zero]
  exact mul_nonneg (le_trans (Real.sqrt_nonneg _) (le_max_left _ _)) (by norm_num [supportTolerance])

def sourceLast {n : ℕ} (values : Fin (n + 1) → ℝ) : ℝ := values (Fin.last n)

theorem ascending_eigh_last_is_maximum {n : ℕ} (values : Fin (n + 1) → ℝ)
    (horder : Monotone values) (i : Fin (n + 1)) : values i ≤ sourceLast values := by
  apply horder
  change i.val ≤ n
  have hi := i.isLt
  omega

theorem source_last_retention_is_nonempty {n : ℕ} (values : Fin (n + 1) → ℝ)
    (h : 0 < sourceLast values) : Nonempty (Retained values (sourceLast values)) :=
  retained_nonempty_from_source_maximum values _ (Fin.last n) rfl h

theorem source_retained_regularization_positive {n : ℕ} (values : Fin (n + 1) → ℝ)
    (rho : ℝ) (hlast : 0 < sourceLast values) (hr : 0 ≤ rho)
    (i : Retained values (sourceLast values)) :
    0 < regularized (fun k : Retained values (sourceLast values) => values k.val) rho i := by
  classical
  letI := source_last_retention_is_nonempty values hlast
  apply regularized_positive _ _ _ hr
  intro k
  exact retained_has_positive_value values _ hlast k

omit [DecidableEq O] in
theorem regularized_covariance_posSemidef (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ)
    (h : ∀ k, 0 ≤ regularized values rho k) : (regularizedCovariance V values rho).PosSemidef := by
  have hd : (diagonal (regularized values rho)).PosSemidef := Matrix.PosSemidef.diagonal h
  simpa only [regularizedCovariance, Matrix.conjTranspose_eq_transpose_of_trivial] using
    hd.mul_mul_conjTranspose_same V

omit [DecidableEq O] in
theorem retained_covariance_compression (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ)
    (hV : Vᵀ * V = 1) :
    Vᵀ * regularizedCovariance V values rho * V = diagonal (regularized values rho) := by
  simp only [regularizedCovariance, ← Matrix.mul_assoc]
  rw [hV, Matrix.one_mul, Matrix.mul_assoc, hV, Matrix.mul_one]

omit [DecidableEq O] in
theorem retained_covariance_posDef (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ)
    (hV : Vᵀ * V = 1) (h : ∀ k, 0 < regularized values rho k) :
    (Vᵀ * regularizedCovariance V values rho * V).PosDef := by
  rw [retained_covariance_compression V values rho hV]
  exact Matrix.PosDef.diagonal h

omit [DecidableEq O] in
theorem retained_regularized_solution_unique (V : Matrix O K ℝ) (values : K → ℝ) (rho : ℝ)
    (delta x : O → ℝ) (hV : Vᵀ * V = 1) (hr : ∀ k, regularized values rho k ≠ 0)
    (hx : projector V *ᵥ x = x)
    (hsolve : regularizedCovariance V values rho *ᵥ x = projector V *ᵥ delta) :
    x = weights V values rho delta := by
  have hleft : Vᵀ * regularizedCovariance V values rho = diagonal (regularized values rho) * Vᵀ := by
    simp only [regularizedCovariance, ← Matrix.mul_assoc]
    rw [hV, Matrix.one_mul]
  have hp : Vᵀ * projector V = Vᵀ := by
    rw [projector, ← Matrix.mul_assoc, hV, Matrix.one_mul]
  have hcoords := congrArg (fun y : O → ℝ => Vᵀ *ᵥ y) hsolve
  dsimp only at hcoords
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, hleft, hp, ← Matrix.mulVec_mulVec] at hcoords
  have hcomponent (k : K) : (Vᵀ *ᵥ x) k = (Vᵀ *ᵥ delta) k / regularized values rho k := by
    apply (eq_div_iff (hr k)).mpr
    have h := congrFun hcoords k
    simpa only [Matrix.mulVec_diagonal, mul_comm] using h
  calc
    x = projector V *ᵥ x := hx.symm
    _ = V *ᵥ (Vᵀ *ᵥ x) := by rw [projector, Matrix.mulVec_mulVec]
    _ = weights V values rho delta := by
      unfold weights
      congr 1
      exact funext hcomponent

omit [DecidableEq O] in
theorem two_retained_bases_same_projector_covariance_give_same_weights
    {L : Type*} [Fintype L] [DecidableEq L]
    (V : Matrix O K ℝ) (W : Matrix O L ℝ) (values : K → ℝ) (other : L → ℝ)
    (rho : ℝ) (delta : O → ℝ) (hV : Vᵀ * V = 1) (hW : Wᵀ * W = 1)
    (hr : ∀ k, regularized values rho k ≠ 0) (hs : ∀ k, regularized other rho k ≠ 0)
    (hP : projector V = projector W)
    (hG : regularizedCovariance V values rho = regularizedCovariance W other rho) :
    weights V values rho delta = weights W other rho delta := by
  symm
  apply retained_regularized_solution_unique V values rho delta _ hV hr
  · rw [hP]
    exact weights_in_retained_range W other rho delta hW
  · rw [hG, hP]
    exact regularized_weights_solve_projected_contrast W other rho delta hW hs

omit [Fintype O] [DecidableEq O] in
theorem source_ridge_is_only_on_retained_projector (V : Matrix O K ℝ) (values : K → ℝ)
    (rho : ℝ) : regularizedCovariance V values rho =
      V * diagonal values * Vᵀ + (rho * retainedMean values) • projector V := by
  ext i j
  change (∑ k, (V * diagonal (regularized values rho)) i k * Vᵀ k j) =
    (∑ k, (V * diagonal values) i k * Vᵀ k j) +
      (rho * retainedMean values) * ∑ k, V i k * Vᵀ k j
  simp only [Matrix.mul_diagonal, regularized, Finset.mul_sum]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  ring

def axisBasis : Matrix (Fin 2) (Fin 1) ℝ := fun i _ => if i = 0 then 1 else 0
def axisValues : Fin 1 → ℝ := fun _ => 1
def nearContrast : Fin 2 → ℝ := ![1, 1 / 2000000]
def discardedAxis : Fin 2 → ℝ := ![0, 1]

theorem axis_columns_orthonormal : axisBasisᵀ * axisBasis = 1 := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [axisBasis, Matrix.mul_apply, Fin.sum_univ_two]

theorem axis_projector : projector axisBasis = !![1, 0; 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [projector, axisBasis, Matrix.mul_apply]

theorem axis_regularized_covariance :
    regularizedCovariance axisBasis axisValues 2 = !![3, 0; 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [regularizedCovariance, axisBasis, axisValues, regularized, retainedMean,
      Matrix.mul_apply, Matrix.diagonal_apply]

theorem axis_weights : weights axisBasis axisValues 2 nearContrast = ![1 / 3, 0] := by
  ext i
  fin_cases i <;> norm_num [weights, axisBasis, axisValues, nearContrast, regularized,
    retainedMean, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

theorem near_contrast_has_nonzero_null_component :
    nearContrast - projector axisBasis *ᵥ nearContrast = ![0, 1 / 2000000] := by
  rw [axis_projector]
  ext i
  fin_cases i <;> norm_num [nearContrast, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

omit [Fintype K] [DecidableEq O] [DecidableEq K] in
theorem coordinate_abs_le_vectorNorm (x : O → ℝ) (j : O) : |x j| ≤ vectorNorm x := by
  have hsum : (x j)^2 ≤ ∑ i, (x i)^2 := Finset.single_le_sum (fun i _ => sq_nonneg (x i)) (Finset.mem_univ j)
  have h := Real.sqrt_le_sqrt hsum
  simpa only [Real.sqrt_sq_eq_abs, vectorNorm] using h

theorem near_contrast_source_gate_passes : contrastAccepted axisBasis nearContrast := by
  rw [contrastAccepted, near_contrast_has_nonzero_null_component]
  have hlarge : (1 : ℝ) ≤ vectorNorm nearContrast := by
    simpa only [nearContrast, Matrix.cons_val_zero, abs_one] using
      coordinate_abs_le_vectorNorm nearContrast (0 : Fin 2)
  have hsmall : vectorNorm (![0, 1 / 2000000] : Fin 2 → ℝ) = 1 / 2000000 := by
    have he : (∑ i : Fin 2, ((![0, 1 / 2000000] : Fin 2 → ℝ) i)^2) = (1 / 2000000 : ℝ)^2 := by
      norm_num [Fin.sum_univ_two]
    rw [vectorNorm, he, Real.sqrt_sq (by norm_num)]
  have hm : (1 : ℝ) ≤ max (vectorNorm nearContrast) supportTiny := le_trans hlarge (le_max_left _ _)
  have hscaled := mul_le_mul_of_nonneg_right hm (show 0 ≤ supportTolerance by norm_num [supportTolerance])
  rw [hsmall]
  exact le_trans (by norm_num [supportTolerance] : (1 / 2000000 : ℝ) ≤ 1 * supportTolerance) hscaled

theorem near_contrast_is_not_exactly_supported : projector axisBasis *ᵥ nearContrast ≠ nearContrast := by
  intro h
  have hv := congrFun h (1 : Fin 2)
  rw [axis_projector] at hv
  norm_num [nearContrast, Matrix.mulVec, dotProduct, Fin.sum_univ_two] at hv

theorem actual_weights_solve_only_projected_near_contrast :
    regularizedCovariance axisBasis axisValues 2 *ᵥ weights axisBasis axisValues 2 nearContrast ≠ nearContrast := by
  rw [regularized_weights_solve_projected_contrast _ _ _ _ axis_columns_orthonormal]
  · exact near_contrast_is_not_exactly_supported
  · intro k
    fin_cases k
    norm_num [regularized, retainedMean, axisValues]

theorem ridge_leaves_discarded_axis_zero :
    regularizedCovariance axisBasis axisValues 2 *ᵥ discardedAxis = 0 := by
  rw [axis_regularized_covariance]
  ext i
  fin_cases i <;> norm_num [discardedAxis, Matrix.mulVec, dotProduct, Fin.sum_univ_two]

def nullRow : Matrix (Fin 1) (Fin 2) ℝ := fun _ => ![0, 1]
def dilutedNullRow : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 100000000, 0]

omit [Fintype K] [DecidableEq O] [DecidableEq K] in
theorem entry_abs_le_batchNorm {B : Type*} [Fintype B] (X : Matrix B O ℝ) (row : B) (j : O) :
    |X row j| ≤ batchNorm X := by
  have hr : (X row j)^2 ≤ ∑ i, (X row i)^2 :=
    Finset.single_le_sum (fun i _ => sq_nonneg (X row i)) (Finset.mem_univ j)
  have ht : (∑ i, (X row i)^2) ≤ ∑ b, ∑ i, (X b i)^2 :=
    Finset.single_le_sum (fun b _ => Finset.sum_nonneg (fun i _ => sq_nonneg (X b i))) (Finset.mem_univ row)
  have h := Real.sqrt_le_sqrt (le_trans hr ht)
  simpa only [Real.sqrt_sq_eq_abs, batchNorm] using h

theorem support_tiny_below_one : supportTiny ≤ 1 := by
  exact pow_le_one₀ (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num)

theorem isolated_null_row_refused : ¬batchAccepted axisBasis nullRow := by
  norm_num [batchAccepted, batchNorm, nullRow, axis_projector, Matrix.mul_apply, projector,
    axisBasis, Fin.sum_univ_two, max_eq_left support_tiny_below_one, supportTolerance]

theorem extra_large_row_admits_same_null_row : batchAccepted axisBasis dilutedNullRow := by
  have hn : (100000000 : ℝ) ≤ batchNorm dilutedNullRow := by
    have h := entry_abs_le_batchNorm dilutedNullRow (1 : Fin 2) (0 : Fin 2)
    have he : dilutedNullRow (1 : Fin 2) (0 : Fin 2) = 100000000 := rfl
    rw [he, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 100000000)] at h
    exact h
  have hm := le_trans hn (le_max_left (batchNorm dilutedNullRow) supportTiny)
  have hs := mul_le_mul_of_nonneg_right hm (show 0 ≤ supportTolerance by norm_num [supportTolerance])
  have hres : batchNorm (dilutedNullRow - dilutedNullRow * projector axisBasis) = 1 := by
    rw [axis_projector]
    norm_num [batchNorm, dilutedNullRow, Matrix.mul_apply, Fin.sum_univ_two]
  unfold batchAccepted
  rw [hres]
  exact le_trans (by norm_num [supportTolerance] : (1 : ℝ) ≤ 100000000 * supportTolerance) hs

end

end N2P3.OperatorGeometry
