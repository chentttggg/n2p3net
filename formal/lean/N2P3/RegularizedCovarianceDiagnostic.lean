import N2P3.Covariance
import N2P3.RetainedGaussianLaw

/-!
# The actual ridge-regularized diagnostic covariance and fixed-residual density

Source: exact_covariance.py:213-250,282-303,318-335,396-441.
Contract: regularized_diagnostic_source_contract_02.json, fixed before this file.

The source stores raw F K Fᵀ but factors F K Fᵀ + rho I. Its method argument
is named whitened_residuals; the body applies both temporal factors again to
the same passed r. Our density difference uses those passed r coordinates.
It does not restore an already temporally whitened input, refit response state,
integrate kernel uncertainty, certify native Cholesky, or describe raw EEG.
Identity spatial covariance means r is already spatially whitened or the
declared diagnostic has identity channel covariance. These are full-space
regularized Gaussian densities, separate from a singular intrinsic density.
-/
namespace N2P3.RegularizedCovarianceDiagnostic
open Matrix MeasureTheory
open N2P3.MatrixNormalCoordinates N2P3.ScaleGauge N2P3.RetainedGaussianLaw
open scoped BigOperators Matrix ComplexOrder
noncomputable section

variable {n r c : Type*} [Fintype n] [Fintype r] [Fintype c]
  [DecidableEq n] [DecidableEq r] [DecidableEq c]

def rawCovariance (F : Matrix n r ℝ) (K : Matrix r r ℝ) : Matrix n n ℝ := F * K * Fᵀ
def sourceRidge (maximum : ℝ) : ℝ := maximum * (1 / 1000000000000)
def regularized (S : Matrix n n ℝ) (maximum : ℝ) : Matrix n n ℝ :=
  S + sourceRidge maximum • 1

omit [Fintype n] [Fintype r] [Fintype c] [DecidableEq n] [DecidableEq r] [DecidableEq c] in
theorem source_ridge_is_positive (maximum : ℝ) (positive : 0 < maximum) : 0 < sourceRidge maximum := by
  exact mul_pos positive (by norm_num)

omit [Fintype c] [DecidableEq c] [DecidableEq r] [DecidableEq n] in
theorem raw_covariance_is_psd (F : Matrix n r ℝ) (K : Matrix r r ℝ) (positive : K.PosSemidef) :
    (rawCovariance F K).PosSemidef := by
  simpa only [rawCovariance, Matrix.conjTranspose_eq_transpose_of_trivial] using
    positive.mul_mul_conjTranspose_same F

omit [Fintype r] [DecidableEq r] [Fintype c] [DecidableEq c] in
theorem actual_regularized_covariance_is_pd (S : Matrix n n ℝ) (maximum : ℝ)
    (positive : S.PosSemidef) (maximumPositive : 0 < maximum) :
    (regularized S maximum).PosDef := by
  have approximate : Covariance.ApproxPSD S 0 :=
    (Covariance.approx_psd_iff_shift_psd S positive.1 0).mpr (by simpa using positive)
  exact Covariance.ridge_makes_pos_def S 0 (sourceRidge maximum) approximate
    (source_ridge_is_positive maximum maximumPositive)

omit [Fintype r] [DecidableEq r] [Fintype c] [DecidableEq c] in
theorem actual_regularized_covariance_has_a_factor (S : Matrix n n ℝ) (maximum : ℝ)
    (positive : S.PosSemidef) (maximumPositive : 0 < maximum) :
    ∃ H : Matrix n n ℝ, H * Hᵀ = regularized S maximum :=
  covariance_factor_exists _ (actual_regularized_covariance_is_pd S maximum positive maximumPositive)

omit [Fintype n] [Fintype r] [DecidableEq r] [Fintype c] [DecidableEq c] in
theorem actual_ridge_changes_every_diagonal (S : Matrix n n ℝ) (maximum : ℝ) (i : n) :
    regularized S maximum i i = S i i + sourceRidge maximum := by
  simp [regularized]

omit [Fintype n] [Fintype r] [DecidableEq r] [Fintype c] [DecidableEq c] in
theorem positive_ridge_does_not_equal_raw_covariance [Nonempty n]
    (S : Matrix n n ℝ) (maximum : ℝ) (maximumPositive : 0 < maximum) :
    regularized S maximum ≠ S := by
  intro same
  obtain ⟨i⟩ := ‹Nonempty n›
  have diagonal := congrArg (fun A : Matrix n n ℝ => A i i) same
  change regularized S maximum i i = S i i at diagonal
  rw [actual_ridge_changes_every_diagonal] at diagonal
  have positive := source_ridge_is_positive maximum maximumPositive
  linarith

def solveEnergy (H : Matrix n n ℝ) (E : Matrix n c ℝ) : ℝ :=
  ∑ coordinate : n × c, ((H⁻¹ * E) coordinate.1 coordinate.2) ^ 2
def factorScore (H : Matrix n n ℝ) (E : Matrix n c ℝ) : ℝ :=
  -(1 / 2 : ℝ) * solveEnergy H E - (Fintype.card c : ℝ) * ∑ i, Real.log (H i i)
def factorPDF (H : Matrix n n ℝ) (E : Matrix n c ℝ) : ℝ :=
  matrixPDF H⁻¹ (1 : Matrix c c ℝ) 0 (vectorize E)
def dimensionConstant : ℝ := -(Fintype.card (n × c) : ℝ) * Real.log (Real.sqrt (2 * Real.pi))
def sourceCorrection (exactFactor approximateFactor : Matrix n n ℝ) (E : Matrix n c ℝ) : ℝ :=
  -(1 / 2 : ℝ) * (solveEnergy exactFactor E - solveEnergy approximateFactor E) -
    (Fintype.card c : ℝ) * ((∑ i, Real.log (exactFactor i i)) - (∑ i, Real.log (approximateFactor i i)))

omit [Fintype r] [DecidableEq r] in
theorem factor_pdf_is_normalized (H : Matrix n n ℝ) (unit : IsUnit H.det) :
    ∫ x : (n × c) → ℝ, matrixPDF H⁻¹ (1 : Matrix c c ℝ) 0 x = 1 :=
  matrixPDF_integral_one _ _ _ (Matrix.isUnit_nonsing_inv_det H unit) (by simp)

omit [Fintype r] [DecidableEq r] in
theorem regularized_full_space_model_is_normalized (S : Matrix n n ℝ) (maximum : ℝ)
    (positive : S.PosSemidef) (maximumPositive : 0 < maximum) :
    ∃ H : Matrix n n ℝ, H * Hᵀ = regularized S maximum ∧
      ∫ x : (n × c) → ℝ, matrixPDF H⁻¹ (1 : Matrix c c ℝ) 0 x = 1 := by
  obtain ⟨H, factor⟩ := actual_regularized_covariance_has_a_factor S maximum positive maximumPositive
  exact ⟨H, factor, factor_pdf_is_normalized H
    (whiteningFactor_unit H _ factor (actual_regularized_covariance_is_pd S maximum positive maximumPositive))⟩

omit [Fintype r] [DecidableEq r] in
theorem factor_pdf_has_actual_solve_energy (H : Matrix n n ℝ) (E : Matrix n c ℝ) :
    factorPDF H E = |H.det|⁻¹ ^ Fintype.card c *
      (Real.sqrt (2 * Real.pi))⁻¹ ^ Fintype.card (n × c) *
        Real.exp (-(1 / 2 : ℝ) * solveEnergy H E) := by
  rw [factorPDF, matrixPDF, GaussianMeasure.affinePDF_as_quadratic_kernel]
  have piHalf : Real.pi / (1 / 2 : ℝ) = 2 * Real.pi := by ring
  have zeroVector : vectorize (0 : Matrix n c ℝ) = 0 := rfl
  rw [piHalf, zeroVector, sub_zero, whitening_coordinates]
  simp only [whiteningMatrix_abs_det, Matrix.det_nonsing_inv, Ring.inverse_eq_inv, inv_one,
    Matrix.det_one, abs_one, one_pow, mul_one, abs_inv, whitenResidual, Matrix.transpose_one, Matrix.mul_one,
    solveEnergy, vectorize]

omit [Fintype r] [DecidableEq r] [Fintype c] [DecidableEq c] in
theorem lower_factor_has_unit_determinant [LinearOrder n] (H : Matrix n n ℝ)
    (lower : H.BlockTriangular OrderDual.toDual) (diagonal : ∀ i, 0 < H i i) : IsUnit H.det := by
  apply isUnit_iff_ne_zero.mpr
  rw [Matrix.det_of_lowerTriangular H lower]
  exact Finset.prod_ne_zero_iff.mpr (fun i _ => ne_of_gt (diagonal i))

omit [Fintype r] [DecidableEq r] in
theorem factor_pdf_log_is_source_score [LinearOrder n] (H : Matrix n n ℝ) (E : Matrix n c ℝ)
    (lower : H.BlockTriangular OrderDual.toDual) (diagonal : ∀ i, 0 < H i i) :
    Real.log (factorPDF H E) = factorScore H E + dimensionConstant (n := n) (c := c) := by
  have nonzero : H.det ≠ 0 := isUnit_iff_ne_zero.mp (lower_factor_has_unit_determinant H lower diagonal)
  have sqrtPositive : 0 < Real.sqrt (2 * Real.pi) :=
    Real.sqrt_pos.mpr (mul_pos (by norm_num) Real.pi_pos)
  rw [factor_pdf_has_actual_solve_energy]
  rw [Real.log_mul (mul_ne_zero (pow_ne_zero _ (inv_ne_zero (abs_ne_zero.mpr nonzero)))
    (pow_ne_zero _ (inv_ne_zero (ne_of_gt sqrtPositive)))) (Real.exp_ne_zero _),
    Real.log_mul (pow_ne_zero _ (inv_ne_zero (abs_ne_zero.mpr nonzero)))
      (pow_ne_zero _ (inv_ne_zero (ne_of_gt sqrtPositive))),
    Real.log_pow, Real.log_pow, Real.log_inv, Real.log_inv, Real.log_abs, Real.log_exp,
    GaussianLowRank.factor_logdet_equals_diagonal_sum H lower diagonal]
  unfold factorScore dimensionConstant
  ring

omit [Fintype r] [DecidableEq r] in
theorem diagnostic_correction_is_same_coordinate_log_density_difference [LinearOrder n]
    (exactFactor approximateFactor : Matrix n n ℝ) (E : Matrix n c ℝ)
    (exactLower : exactFactor.BlockTriangular OrderDual.toDual) (exactDiagonal : ∀ i, 0 < exactFactor i i)
    (approximateLower : approximateFactor.BlockTriangular OrderDual.toDual)
    (approximateDiagonal : ∀ i, 0 < approximateFactor i i) :
    sourceCorrection exactFactor approximateFactor E =
      Real.log (factorPDF exactFactor E) - Real.log (factorPDF approximateFactor E) := by
  rw [factor_pdf_log_is_source_score exactFactor E exactLower exactDiagonal,
    factor_pdf_log_is_source_score approximateFactor E approximateLower approximateDiagonal]
  unfold sourceCorrection factorScore
  ring

omit [Fintype r] [DecidableEq r] [DecidableEq c] in
theorem exact_column_solutions_recover_source_energy [LinearOrder n]
    (H : Matrix n n ℝ) (E solved : Matrix n c ℝ)
    (lower : H.BlockTriangular OrderDual.toDual) (diagonal : ∀ i, 0 < H i i)
    (equations : H * solved = E) :
    (∑ coordinate : n × c, (solved coordinate.1 coordinate.2) ^ 2) = solveEnergy H E := by
  have unit := lower_factor_has_unit_determinant H lower diagonal
  have recovered : H⁻¹ * E = solved := by
    rw [← equations, ← Matrix.mul_assoc, Matrix.nonsing_inv_mul H unit, Matrix.one_mul]
  rw [solveEnergy, recovered]

omit [Fintype r] [DecidableEq r] in
theorem actual_regularized_diagnostic_is_a_difference_of_normalized_densities [LinearOrder n]
    (S K : Matrix n n ℝ) (maximum : ℝ) (E : Matrix n c ℝ)
    (positive : S.PosSemidef) (maximumPositive : 0 < maximum) (arPositive : K.PosDef)
    (exactFactor approximateFactor : Matrix n n ℝ)
    (exactEquation : exactFactor * exactFactorᵀ = regularized S maximum)
    (approximateEquation : approximateFactor * approximateFactorᵀ = K)
    (exactLower : exactFactor.BlockTriangular OrderDual.toDual) (exactDiagonal : ∀ i, 0 < exactFactor i i)
    (approximateLower : approximateFactor.BlockTriangular OrderDual.toDual)
    (approximateDiagonal : ∀ i, 0 < approximateFactor i i) :
    (∫ x : (n × c) → ℝ, matrixPDF exactFactor⁻¹ (1 : Matrix c c ℝ) 0 x) = 1 ∧
    (∫ x : (n × c) → ℝ, matrixPDF approximateFactor⁻¹ (1 : Matrix c c ℝ) 0 x) = 1 ∧
    sourceCorrection exactFactor approximateFactor E =
      Real.log (factorPDF exactFactor E) - Real.log (factorPDF approximateFactor E) := by
  exact ⟨factor_pdf_is_normalized exactFactor (whiteningFactor_unit exactFactor _ exactEquation
      (actual_regularized_covariance_is_pd S maximum positive maximumPositive)),
    factor_pdf_is_normalized approximateFactor (whiteningFactor_unit approximateFactor K approximateEquation arPositive),
    diagnostic_correction_is_same_coordinate_log_density_difference exactFactor approximateFactor E
      exactLower exactDiagonal approximateLower approximateDiagonal⟩

end
end N2P3.RegularizedCovarianceDiagnostic
