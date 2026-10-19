import N2P3.BayesDensity
import Mathlib.Tactic.Linarith

/-!
Strict false controls for the accepted Bayesian coefficient-density layer.
These instantiate the actual source kernels/integral theorem at scalar finite
coordinates with K=Sigma=I and positive ridge. They are statistical formula
instances, not assertions that a one-row array is a complete EEG recording.
Each example must leave one False goal and no other error.
-/

namespace N2P3.BayesDensity.Negative

open Matrix MeasureTheory
open N2P3.ScaleGauge N2P3.GaussianConditioning N2P3.MatrixNormalCoordinates
open scoped Matrix BigOperators ComplexOrder

private abbrev Scalar := Fin 1
private abbrev Mat := Matrix Scalar Scalar ℝ

private theorem scalar_data_factor :
    dataFactor (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 = 1 := by
  simp [dataFactor, constantEnergy, mean, rowCov, precision, weightedEnergy]

private theorem scalar_jacobian :
    jacobianNormalizer (1 : Mat) (1 : Mat) = (Real.sqrt (2 * Real.pi))⁻¹ := by
  simp [jacobianNormalizer, whiteningMatrix]

private theorem sqrt_two_pi_gt_one : 1 < Real.sqrt (2 * Real.pi) := by
  apply Real.lt_sqrt_of_sq_lt
  nlinarith [Real.two_le_pi]

private theorem scalar_integral :
    (∫ x, jointCoordinates (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 x) =
      Real.sqrt (2 * Real.pi) := by
  have h := jointCoordinates_integral (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 (1 : Mat) (1 : Mat)
    (by simp [precision]) (by simp) Matrix.PosDef.one (by norm_num) Matrix.PosDef.one
  rw [scalar_data_factor, scalar_jacobian] at h
  simpa only [one_div, inv_inv] using h

private theorem scalar_mode_value :
    jointCoordinates (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1
      (vectorize (mean (0 : Mat) (0 : Mat) (1 : Mat) 1)) = 1 := by
  have hz : mean (0 : Mat) (0 : Mat) (1 : Mat) 1 = 0 := by simp [mean]
  rw [hz]
  change jointKernel (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 (0 : Mat) = 1
  simp [jointKernel, energy, weightedEnergy]

/-- False: c/J may be replaced with c*J. Here J=1/sqrt(2*pi) differs from one. -/
example : (∫ x, jointCoordinates (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 x) =
    dataFactor (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 *
      jacobianNormalizer (1 : Mat) (1 : Mat) := by
  rw [scalar_integral, scalar_data_factor, scalar_jacobian, one_mul]
  have hne : Real.sqrt (2 * Real.pi) ≠ (Real.sqrt (2 * Real.pi))⁻¹ := by
    intro h
    have hpos : 0 < Real.sqrt (2 * Real.pi) := lt_trans (by norm_num) sqrt_two_pi_gt_one
    have hm := mul_inv_cancel₀ (ne_of_gt hpos)
    rw [← h] at hm
    nlinarith [sqrt_two_pi_gt_one]
  simp only [hne]

/-- False: dividing the actual kernel by its mode value produces mass one. -/
example : (∫ x, jointCoordinates (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1 x /
    jointCoordinates (0 : Mat) (0 : Mat) (1 : Mat) (1 : Mat) 1
      (vectorize (mean (0 : Mat) (0 : Mat) (1 : Mat) 1))) = 1 := by
  rw [scalar_mode_value]
  simp only [div_one]
  rw [scalar_integral]
  simp only [ne_of_gt sqrt_two_pi_gt_one]

/-- False: lambda I itself is the source prior factor. At lambda=4 its Gram
is 16I, whereas the correct sqrt(lambda)I factor has Gram 4I. -/
example : (4 : Mat)ᵀ * (4 : Mat) =
    (priorRowFactor (p := Scalar) 4)ᵀ * priorRowFactor 4 := by
  rw [prior_row_factor_gram 4 (by norm_num)]
  have hne : (4 : Mat)ᵀ * (4 : Mat) ≠ (4 : ℝ) • (1 : Mat) := by
    intro h
    have hv := congrArg (fun A : Mat => A 0 0) h
    norm_num [Matrix.mul_apply, Matrix.transpose_apply, Matrix.ofNat_apply, Fin.sum_univ_succ] at hv
  simp only [hne]

end N2P3.BayesDensity.Negative
