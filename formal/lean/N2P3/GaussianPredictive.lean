import N2P3.Covariance
import N2P3.ScaleGauge
import Mathlib.Tactic.Abel

/-!
# Correlated calibration/held noise: the predictive premise

The continuous score uses the independent-block expression `Kh + Dh P Dhᵀ`.
This file proves exact real matrix and second-moment identities when the joint
noise has off-diagonal block `C = Cov(held, calibration)`. Disjoint sample scopes
do not establish `C = 0`. The source-bound numerical witness is recorded in
`artifacts/predictive_cross_block_03/report.json`.

The innovation covariance and its zero cross-covariance with calibration noise
are derived, rather than assumed. Zero cross-covariance alone does not prove
independence. The joint-Gaussian conditional-law/disintegration argument and the
full normalized predictive integral remain separate proof obligations. In
particular, the matrix named `predictiveRows` below is an algebraic second-moment
expression, not a theorem that the current source realizes that conditional law.
-/

namespace N2P3.GaussianPredictive

open Matrix MeasureTheory ProbabilityTheory
open N2P3Pilot
open scoped BigOperators Matrix ComplexOrder ProbabilityTheory

noncomputable section

section MatrixIdentities

variable {n m p c : Type*}
variable [Fintype n] [Fintype m] [Fintype p] [Fintype c]
variable [DecidableEq n] [DecidableEq m]

def jointNoise (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) (Kh : Matrix m m ℝ) :
    Matrix (n ⊕ m) (n ⊕ m) ℝ := fromBlocks Kc Cᵀ C Kh

def regression (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) : Matrix m n ℝ := C * Kc⁻¹

def innovationOp (J : Matrix m n ℝ) : Matrix m (n ⊕ m) ℝ := horizontal (-J) 1

def calibrationOp : Matrix n (n ⊕ m) ℝ := horizontal 1 0

def schurNoise (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) (Kh : Matrix m m ℝ) :
    Matrix m m ℝ := Kh - regression Kc C * Cᵀ

def effectiveDesign (J : Matrix m n ℝ) (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ) :
    Matrix m p ℝ := Dh - J * Dc

def predictiveMean (J : Matrix m n ℝ) (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ)
    (Yc : Matrix n c ℝ) (mean : Matrix p c ℝ) : Matrix m c ℝ :=
  Dh * mean + J * (Yc - Dc * mean)

def predictiveRows (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) (Kh : Matrix m m ℝ)
    (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ) (P : Matrix p p ℝ) : Matrix m m ℝ :=
  schurNoise Kc C Kh +
    effectiveDesign (regression Kc C) Dc Dh * P *
      (effectiveDesign (regression Kc C) Dc Dh)ᵀ

omit [Fintype m] [DecidableEq m] in
theorem regression_mul_calibration (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (hKc : Kc.PosDef) : regression Kc C * Kc = C := by
  rw [regression, Matrix.mul_assoc, Matrix.nonsing_inv_mul _
    (N2P3.ScaleGauge.positive_definite_det_unit hKc), Matrix.mul_one]

omit [Fintype p] [Fintype c] in
theorem innovation_mul_joint (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (hKc : Kc.PosDef) :
    innovationOp (regression Kc C) * jointNoise Kc C Kh =
      horizontal 0 (schurNoise Kc C Kh) := by
  rw [innovationOp, jointNoise, horizontal_mul_block]
  simp only [Matrix.neg_mul, Matrix.one_mul, regression_mul_calibration _ _ hKc]
  simp [schurNoise, sub_eq_add_neg, add_comm]

omit [Fintype p] [Fintype c] in
theorem innovation_congruence (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (hKc : Kc.PosDef) :
    innovationOp (regression Kc C) * jointNoise Kc C Kh *
      (innovationOp (regression Kc C))ᵀ = schurNoise Kc C Kh := by
  rw [innovation_mul_joint _ _ _ hKc, innovationOp, horizontal_mul_transpose]
  simp

omit [Fintype p] [Fintype c] in
theorem innovation_calibration_cross (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (hKc : Kc.PosDef) :
    innovationOp (regression Kc C) * jointNoise Kc C Kh *
      (calibrationOp : Matrix n (n ⊕ m) ℝ)ᵀ = 0 := by
  rw [innovation_mul_joint _ _ _ hKc, calibrationOp]
  ext i j
  simp [Matrix.mul_apply, horizontal, Fintype.sum_sum_type]

omit [Fintype n] [Fintype p] [Fintype c] [DecidableEq n] in
theorem innovation_vecMul_injective (J : Matrix m n ℝ) :
    Function.Injective (innovationOp J).vecMul := by
  intro x y h
  funext j
  have hj := congrFun h (Sum.inr j)
  simpa [innovationOp, horizontal, Matrix.vecMul, dotProduct, Matrix.one_apply] using hj

omit [Fintype p] [Fintype c] in
theorem schur_noise_psd (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (hKc : Kc.PosDef)
    (hJoint : (jointNoise Kc C Kh).PosSemidef) : (schurNoise Kc C Kh).PosSemidef := by
  rw [← innovation_congruence _ _ _ hKc]
  exact N2P3Pilot.covariance_pushforward_psd _ _ hJoint

omit [Fintype p] [Fintype c] in
theorem schur_noise_posDef (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (hKc : Kc.PosDef)
    (hJoint : (jointNoise Kc C Kh).PosDef) : (schurNoise Kc C Kh).PosDef := by
  rw [← innovation_congruence _ _ _ hKc]
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    hJoint.mul_mul_conjTranspose_same (innovation_vecMul_injective (regression Kc C))

omit [Fintype m] [Fintype c] [DecidableEq n] [DecidableEq m] in
theorem observation_residual (J : Matrix m n ℝ) (Dc : Matrix n p ℝ)
    (Dh : Matrix m p ℝ) (W : Matrix p c ℝ) (Ec : Matrix n c ℝ) (Eh : Matrix m c ℝ) :
    (Dh * W + Eh) - J * (Dc * W + Ec) =
      effectiveDesign J Dc Dh * W + (Eh - J * Ec) := by
  simp only [effectiveDesign, Matrix.mul_add, Matrix.sub_mul, Matrix.mul_assoc]
  abel

omit [Fintype m] [Fintype c] [DecidableEq n] [DecidableEq m] in
theorem predictive_mean_adjusted (J : Matrix m n ℝ) (Dc : Matrix n p ℝ)
    (Dh : Matrix m p ℝ) (Yc : Matrix n c ℝ) (mean : Matrix p c ℝ) :
    predictiveMean J Dc Dh Yc mean = J * Yc + effectiveDesign J Dc Dh * mean := by
  simp only [predictiveMean, effectiveDesign, Matrix.mul_sub, Matrix.sub_mul,
    Matrix.mul_assoc]
  abel

omit [Fintype m] [Fintype c] [DecidableEq m] in
theorem predictive_rows_expansion (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ) (P : Matrix p p ℝ) :
    predictiveRows Kc C Kh Dc Dh P =
      Kh + Dh * P * Dhᵀ - regression Kc C * Cᵀ -
        Dh * P * Dcᵀ * (regression Kc C)ᵀ -
        regression Kc C * Dc * P * Dhᵀ +
        regression Kc C * Dc * P * Dcᵀ * (regression Kc C)ᵀ := by
  simp only [predictiveRows, schurNoise, effectiveDesign, Matrix.transpose_sub,
    Matrix.transpose_mul, Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_assoc]
  abel

omit [Fintype c] in
theorem predictive_rows_posDef (Kc : Matrix n n ℝ) (C : Matrix m n ℝ)
    (Kh : Matrix m m ℝ) (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ) (P : Matrix p p ℝ)
    (hKc : Kc.PosDef) (hJoint : (jointNoise Kc C Kh).PosDef) (hP : P.PosSemidef) :
    (predictiveRows Kc C Kh Dc Dh P).PosDef := by
  apply (schur_noise_posDef _ _ _ hKc hJoint).add_posSemidef
  simpa only [Matrix.conjTranspose_eq_transpose_of_trivial] using
    hP.mul_mul_conjTranspose_same (effectiveDesign (regression Kc C) Dc Dh)

omit [Fintype m] [DecidableEq m] in
theorem regression_zero (Kc : Matrix n n ℝ) :
    regression Kc (0 : Matrix m n ℝ) = 0 := by simp [regression]

omit [Fintype m] [Fintype p] [Fintype c] [DecidableEq m] in
theorem schur_noise_zero_cross (Kc : Matrix n n ℝ) (Kh : Matrix m m ℝ) :
    schurNoise Kc 0 Kh = Kh := by simp [schurNoise, regression_zero]

omit [Fintype m] [Fintype c] [DecidableEq n] [DecidableEq m] in
theorem predictive_mean_zero_cross (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ)
    (Yc : Matrix n c ℝ) (mean : Matrix p c ℝ) :
    predictiveMean 0 Dc Dh Yc mean = Dh * mean := by simp [predictiveMean]

omit [Fintype m] [Fintype c] [DecidableEq m] in
theorem predictive_rows_zero_cross (Kc : Matrix n n ℝ) (Kh : Matrix m m ℝ)
    (Dc : Matrix n p ℝ) (Dh : Matrix m p ℝ) (P : Matrix p p ℝ) :
    predictiveRows Kc 0 Kh Dc Dh P = N2P3.ScaleGauge.temporalPredictive Dh Kh P := by
  simp [predictiveRows, schur_noise_zero_cross, regression_zero, effectiveDesign,
    N2P3.ScaleGauge.temporalPredictive]

end MatrixIdentities

section RandomVariables

open N2P3.Covariance

variable {Ω n m r s : Type*} [MeasurableSpace Ω]
variable [Fintype n] [Fintype m] [Fintype r] [Fintype s]
variable (μ : Measure Ω) [IsProbabilityMeasure μ]

omit [Fintype n] [Fintype m] [Fintype r] [Fintype s] [IsProbabilityMeasure μ] in
theorem cross_covariance_transpose (X : n → Ω → ℝ) (Z : m → Ω → ℝ) :
    (crossCovarianceMatrix μ X Z)ᵀ = crossCovarianceMatrix μ Z X := by
  ext i j
  exact covariance_comm (X j) (Z i)

omit [Fintype r] [Fintype s] in
theorem cross_covariance_linear_images (X : n → Ω → ℝ) (Z : m → Ω → ℝ)
    (hX : ∀ i, MemLp (X i) 2 μ) (hZ : ∀ i, MemLp (Z i) 2 μ)
    (A : Matrix r n ℝ) (B : Matrix s m ℝ) :
    crossCovarianceMatrix μ (linearImage A X) (linearImage B Z) =
      A * crossCovarianceMatrix μ X Z * Bᵀ := by
  ext i k
  change cov[fun ω => ∑ j, A i j * X j ω, fun ω => ∑ j, B k j * Z j ω; μ] = _
  rw [covariance_linear_combination μ X Z hX hZ]
  simp only [Matrix.mul_apply, Matrix.transpose_apply, crossCovarianceMatrix, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro l _
  ring

variable [DecidableEq n] [DecidableEq m]

def innovation (J : Matrix m n ℝ) (Ec : n → Ω → ℝ) (Eh : m → Ω → ℝ) :
    m → Ω → ℝ := linearImage (innovationOp J) (Sum.elim Ec Eh)

omit [MeasurableSpace Ω] [Fintype r] [Fintype s] [DecidableEq n]
  [IsProbabilityMeasure μ] in
theorem innovation_apply (J : Matrix m n ℝ) (Ec : n → Ω → ℝ) (Eh : m → Ω → ℝ)
    (i : m) (ω : Ω) : innovation J Ec Eh i ω = Eh i ω - linearImage J Ec i ω := by
  simp [innovation, innovationOp, horizontal, linearImage, Fintype.sum_sum_type,
    Finset.sum_neg_distrib, sub_eq_add_neg, add_comm, Matrix.one_apply]

omit [MeasurableSpace Ω] [Fintype r] [Fintype s] [DecidableEq m]
  [IsProbabilityMeasure μ] in
theorem calibration_linear_image (Ec : n → Ω → ℝ) (Eh : m → Ω → ℝ) :
    linearImage (calibrationOp : Matrix n (n ⊕ m) ℝ) (Sum.elim Ec Eh) = Ec := by
  funext i ω
  simp [calibrationOp, horizontal, linearImage, Fintype.sum_sum_type, Matrix.one_apply]

omit [Fintype n] [Fintype m] [Fintype r] [Fintype s]
  [DecidableEq n] [DecidableEq m] [IsProbabilityMeasure μ] in
theorem joint_noise_covariance (Ec : n → Ω → ℝ) (Eh : m → Ω → ℝ)
    (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) (Kh : Matrix m m ℝ)
    (hKc : covarianceMatrix μ Ec = Kc) (hC : crossCovarianceMatrix μ Eh Ec = C)
    (hKh : covarianceMatrix μ Eh = Kh) :
    covarianceMatrix μ (Sum.elim Ec Eh) = jointNoise Kc C Kh := by
  have hc : crossCovarianceMatrix μ Ec Eh = Cᵀ := by
    rw [← hC, cross_covariance_transpose]
  rw [covariance_joint_block, hKc, hKh, hc, Matrix.transpose_transpose]
  rfl

omit [Fintype r] [Fintype s] in
/-- This is an actual covariance equality for finite-second-moment variables;
it uses no Gaussian or independence premise. -/
theorem innovation_covariance (Ec : n → Ω → ℝ) (Eh : m → Ω → ℝ)
    (hEc : ∀ i, MemLp (Ec i) 2 μ) (hEh : ∀ i, MemLp (Eh i) 2 μ)
    (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) (Kh : Matrix m m ℝ)
    (hKc : covarianceMatrix μ Ec = Kc) (hC : crossCovarianceMatrix μ Eh Ec = C)
    (hKh : covarianceMatrix μ Eh = Kh) (hPD : Kc.PosDef) :
    covarianceMatrix μ (innovation (regression Kc C) Ec Eh) = schurNoise Kc C Kh := by
  unfold innovation
  rw [covariance_linear_image μ (Sum.elim Ec Eh)
    (fun i => by cases i with | inl i => exact hEc i | inr i => exact hEh i),
    joint_noise_covariance μ Ec Eh Kc C Kh hKc hC hKh,
    innovation_congruence _ _ _ hPD]

omit [Fintype r] [Fintype s] in
/-- Uncorrelated with calibration noise. Independence still requires the
separate jointly Gaussian argument. -/
theorem innovation_uncorrelated_calibration (Ec : n → Ω → ℝ) (Eh : m → Ω → ℝ)
    (hEc : ∀ i, MemLp (Ec i) 2 μ) (hEh : ∀ i, MemLp (Eh i) 2 μ)
    (Kc : Matrix n n ℝ) (C : Matrix m n ℝ) (Kh : Matrix m m ℝ)
    (hKc : covarianceMatrix μ Ec = Kc) (hC : crossCovarianceMatrix μ Eh Ec = C)
    (hKh : covarianceMatrix μ Eh = Kh) (hPD : Kc.PosDef) :
    crossCovarianceMatrix μ (innovation (regression Kc C) Ec Eh) Ec = 0 := by
  have h := cross_covariance_linear_images μ (Sum.elim Ec Eh) (Sum.elim Ec Eh)
    (fun i => by cases i with | inl i => exact hEc i | inr i => exact hEh i)
    (fun i => by cases i with | inl i => exact hEc i | inr i => exact hEh i)
    (innovationOp (regression Kc C)) (calibrationOp : Matrix n (n ⊕ m) ℝ)
  rw [calibration_linear_image] at h
  change crossCovarianceMatrix μ (innovation (regression Kc C) Ec Eh) Ec =
    innovationOp (regression Kc C) * covarianceMatrix μ (Sum.elim Ec Eh) *
      (calibrationOp : Matrix n (n ⊕ m) ℝ)ᵀ at h
  rw [joint_noise_covariance μ Ec Eh Kc C Kh hKc hC hKh,
    innovation_calibration_cross _ _ _ hPD] at h
  exact h

end RandomVariables

section ScalarWitness

abbrev One := Fin 1

/-- Unit calibration/held variance and correlation 1/2 form a legal positive
definite joint noise covariance. This is not an empirical EEG parameter claim. -/
theorem half_cross_joint_posDef :
    (jointNoise (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)) 1).PosDef := by
  constructor
  · ext i j
    cases i <;> cases j <;>
      simp [jointNoise, Matrix.conjTranspose_apply, Matrix.one_apply, eq_comm]
  · intro x hx
    have hn : 0 < (x (Sum.inl 0)) ^ 2 + (x (Sum.inr 0)) ^ 2 := by
      have hne : x (Sum.inl 0) ≠ 0 ∨ x (Sum.inr 0) ≠ 0 := by
        by_contra h
        push_neg at h
        apply hx
        funext i
        cases i with
        | inl i => fin_cases i; exact h.1
        | inr i => fin_cases i; exact h.2
      rcases hne with h | h
      · exact add_pos_of_pos_of_nonneg (sq_pos_of_ne_zero h) (sq_nonneg _)
      · exact add_pos_of_nonneg_of_pos (sq_nonneg _) (sq_pos_of_ne_zero h)
    simp only [Matrix.mulVec, dotProduct, Fintype.sum_sum_type, Fin.sum_univ_one,
      jointNoise, Matrix.fromBlocks_apply₁₁, Matrix.fromBlocks_apply₁₂,
      Matrix.fromBlocks_apply₂₁, Matrix.fromBlocks_apply₂₂, Matrix.transpose_apply,
      Matrix.one_apply, Matrix.smul_apply, smul_eq_mul, ↓reduceIte, one_mul,
      Pi.star_apply, star_trivial]
    nlinarith [sq_nonneg (x (Sum.inl 0) + x (Sum.inr 0))]

theorem half_cross_ridge_moments :
    N2P3.ScaleGauge.rowCov (1 : Matrix One One ℝ) 1 1 = (1 / 2 : ℝ) • 1 ∧
    N2P3.ScaleGauge.mean (1 : Matrix One One ℝ) ((2 : ℝ) • (1 : Matrix One One ℝ)) 1 1 = 1 := by
  have hp : N2P3.ScaleGauge.rowCov (1 : Matrix One One ℝ) 1 1 = (1 / 2 : ℝ) • 1 := by
    unfold N2P3.ScaleGauge.rowCov
    have hprec : N2P3.ScaleGauge.precision (1 : Matrix One One ℝ) 1 1 = (2 : ℝ) • 1 := by
      ext i j
      fin_cases i
      fin_cases j
      norm_num [N2P3.ScaleGauge.precision, Matrix.mul_apply, Fin.sum_univ_one]
    rw [hprec, N2P3.ScaleGauge.inverse_smul 1 2 (by norm_num) (by simp)]
    simp
  refine ⟨hp, ?_⟩
  rw [N2P3.ScaleGauge.mean, hp]
  ext i j
  fin_cases i
  fin_cases j
  norm_num [Matrix.mul_apply, Fin.sum_univ_one, Matrix.ofNat_apply]

theorem half_cross_corrected_mean :
    predictiveMean (regression (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)))
      (1 : Matrix One One ℝ) 1 (2 • 1) (1 : Matrix One One ℝ) = (3 / 2 : ℝ) • 1 := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [predictiveMean, regression, Matrix.mul_apply, Fin.sum_univ_one]

theorem half_cross_corrected_rows :
    predictiveRows (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)) 1
      (1 : Matrix One One ℝ) 1
      ((1 / 2 : ℝ) • 1) = (7 / 8 : ℝ) • 1 := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [predictiveRows, schurNoise, regression, effectiveDesign,
    Matrix.mul_apply, Fin.sum_univ_one]

theorem half_cross_independent_rows :
    predictiveRows (1 : Matrix One One ℝ) (0 : Matrix One One ℝ) 1 (1 : Matrix One One ℝ) 1
      ((1 / 2 : ℝ) • 1) = (3 / 2 : ℝ) • 1 := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [predictiveRows, schurNoise, regression, effectiveDesign,
    Matrix.mul_apply, Fin.sum_univ_one]

theorem half_cross_predictions_differ :
    predictiveMean (regression (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)))
      (1 : Matrix One One ℝ) 1 (2 • 1) (1 : Matrix One One ℝ) ≠
        predictiveMean (0 : Matrix One One ℝ) (1 : Matrix One One ℝ) 1 (2 • 1)
          (1 : Matrix One One ℝ) ∧
    predictiveRows (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)) 1
      (1 : Matrix One One ℝ) 1 ((1 / 2 : ℝ) • 1) ≠
        predictiveRows (1 : Matrix One One ℝ) (0 : Matrix One One ℝ) 1
          (1 : Matrix One One ℝ) 1 ((1 / 2 : ℝ) • 1) := by
  rw [half_cross_corrected_mean, predictive_mean_zero_cross, Matrix.one_mul,
    half_cross_corrected_rows, half_cross_independent_rows]
  constructor <;> intro h
  · have := congrArg (fun A : Matrix One One ℝ => A 0 0) h
    norm_num at this
  · have := congrArg (fun A : Matrix One One ℝ => A 0 0) h
    norm_num at this

end ScalarWitness

end

end N2P3.GaussianPredictive
