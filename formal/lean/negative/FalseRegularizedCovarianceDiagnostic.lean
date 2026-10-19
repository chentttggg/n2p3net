import N2P3.RegularizedCovarianceDiagnostic

/-! Independent wrong alternatives fixed in source contract02. Both fixtures
are inhabited; no contradictory factor or solver assumptions are supplied. -/
namespace N2P3.NegativeRegularizedCovarianceDiagnostic
open N2P3.RegularizedCovarianceDiagnostic
open scoped BigOperators
noncomputable section

example : regularized (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 = 1 := by
  have different := positive_ridge_does_not_equal_raw_covariance
    (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 (by norm_num)
  simp only [different]

example : sourceCorrection (2 : Matrix (Fin 1) (Fin 1) ℝ) 1
    (0 : Matrix (Fin 1) (Fin 2) ℝ) = -Real.log 2 := by
  have actual : sourceCorrection (2 : Matrix (Fin 1) (Fin 1) ℝ) 1
      (0 : Matrix (Fin 1) (Fin 2) ℝ) = -2 * Real.log 2 := by
    norm_num [sourceCorrection, solveEnergy, Fin.sum_univ_one, Matrix.ofNat_apply]
  have positive : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have different : -2 * Real.log 2 ≠ -Real.log 2 := by linarith
  rw [actual]
  simp only [different]

end
end N2P3.NegativeRegularizedCovarianceDiagnostic
