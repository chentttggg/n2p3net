import N2P3.MatrixNormalMoments
import Mathlib.Tactic.NormNum

open Matrix MeasureTheory
open N2P3.GaussianMoments N2P3.MatrixNormalMoments N2P3.GaussianMeasure
open scoped Matrix

/-! Each false proposition is reduced to False using proved actual moments. -/

/-- False: the normalized finite standard Gaussian has variance two. -/
example : N2P3.Covariance.covarianceMatrix (standardMeasure (ι := Fin 1))
    (fun i x => x i) 0 0 = 2 := by
  rw [standard_covariance]
  norm_num

/-- False: translating a normalized Gaussian to mean three keeps mean zero. -/
example : (∫ x : Fin 1 → ℝ, x 0
    ∂affineGaussianMeasure (1 / 2) (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 3)) = 0 := by
  rw [affine_density_mean _ (by simp)]
  norm_num
