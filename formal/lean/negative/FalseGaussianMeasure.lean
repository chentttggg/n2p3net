import N2P3.GaussianMeasure
import Mathlib.Tactic.Linarith

/-!
Deliberately false density-normalization claims. The verifier must reject each
example with one unsolved False goal, rather than an import/type failure.
These are scalar specializations of the source Gaussian density formulas;
they do not assert admission as a complete raw EEG recording.
-/

namespace N2P3.GaussianMeasure.Negative

open MeasureTheory Matrix
open scoped Matrix

/-- False: a Gaussian exponential kernel without its scalar normalizer has mass one. -/
example : (∫ x : ℝ, Real.exp (-(1 / 2 : ℝ) * x ^ 2)) = 1 := by
  rw [integral_gaussian]
  have hne : Real.sqrt (Real.pi / (1 / 2 : ℝ)) ≠ 1 := by
    intro h
    have hsq : (Real.sqrt (Real.pi / (1 / 2 : ℝ))) ^ 2 = (1 : ℝ) ^ 2 :=
      congrArg (fun x : ℝ => x ^ 2) h
    rw [Real.sq_sqrt (div_nonneg Real.pi_pos.le (by norm_num))] at hsq
    norm_num at hsq
    linarith [Real.two_le_pi]
  simp only [hne]

/-- False: omitting the scale-two determinant factor preserves total probability. -/
example : (∫ x : Fin 1 → ℝ,
    productPDF (1 / 2) ((2 : Matrix (Fin 1) (Fin 1) ℝ) *ᵥ x)) = 1 := by
  rw [scale_two_without_jacobian_has_half_mass (1 / 2) (by norm_num)]
  norm_num

/-- False: using the inverse Jacobian as the density factor has the correct orientation. -/
example : (∫ x : Fin 1 → ℝ, (1 / 2 : ℝ) *
    productPDF (1 / 2) ((2 : Matrix (Fin 1) (Fin 1) ℝ) *ᵥ x)) = 1 := by
  rw [integral_const_mul,
    scale_two_without_jacobian_has_half_mass (1 / 2) (by norm_num)]
  norm_num

end N2P3.GaussianMeasure.Negative
