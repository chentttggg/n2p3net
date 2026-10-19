import N2P3.GaussianMoments
import N2P3.GaussianAffineLaw
import N2P3.MatrixNormalCoordinates

/-!
# Source mean and inverse-Gram matrices are actual probability moments

Transport the proved finite standard-normal moments through the inverse
whitening map. The resulting probability measure is the actual normalized
matrix density, by GaussianAffineLaw. The source covariance parameters are
identified as moments, beyond the earlier inverse-Gram algebraic identity.
-/

namespace N2P3.MatrixNormalMoments

open MeasureTheory ProbabilityTheory Matrix
open N2P3.GaussianMoments N2P3.GaussianMeasure N2P3.GaussianAffineLaw
open N2P3.MatrixNormalCoordinates N2P3.ScaleGauge
open scoped ProbabilityTheory Matrix BigOperators Kronecker

noncomputable section

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

theorem linear_coordinate_memLp (A : Matrix ι ι ℝ) (i : ι) :
    MemLp (fun z : ι → ℝ => (A *ᵥ z) i) 2 standardMeasure := by
  simpa only [Matrix.mulVec, dotProduct] using
    memLp_finset_sum (μ := standardMeasure) (p := 2)
    (f := fun (j : ι) (z : ι → ℝ) => A i j * z j) Finset.univ
    (fun j _ => (standard_coordinate_memLp j).const_mul (A i j))

theorem affine_coordinate_memLp (A : Matrix ι ι ℝ) (center : ι → ℝ) (i : ι) :
    MemLp (fun z : ι → ℝ => (A *ᵥ z + center) i) 2 standardMeasure := by
  haveI := standardMeasure_probability (ι := ι)
  exact (linear_coordinate_memLp A i).add (memLp_const (center i))

theorem linear_coordinate_mean (A : Matrix ι ι ℝ) (i : ι) :
    (∫ z : ι → ℝ, (A *ᵥ z) i ∂standardMeasure) = 0 := by
  haveI := standardMeasure_probability (ι := ι)
  change (∫ z, ∑ j, A i j * z j ∂standardMeasure) = 0
  rw [integral_finset_sum Finset.univ
    (fun j _ => ((standard_coordinate_memLp j).const_mul _).integrable (by norm_num))]
  simp only [integral_const_mul, standard_mean, mul_zero, Finset.sum_const_zero]

theorem affine_coordinate_mean (A : Matrix ι ι ℝ) (center : ι → ℝ) (i : ι) :
    (∫ z : ι → ℝ, (A *ᵥ z + center) i ∂standardMeasure) = center i := by
  haveI := standardMeasure_probability (ι := ι)
  change (∫ z, (A *ᵥ z) i + center i ∂standardMeasure) = _
  rw [integral_add ((linear_coordinate_memLp A i).integrable (by norm_num))
    (integrable_const _), linear_coordinate_mean]
  simp

theorem affine_coordinate_covariance (A : Matrix ι ι ℝ) (center : ι → ℝ) :
    Covariance.covarianceMatrix (standardMeasure (ι := ι))
      (fun i z => (A *ᵥ z + center) i) = A * Aᵀ := by
  haveI := standardMeasure_probability (ι := ι)
  have h := Covariance.covariance_linear_image standardMeasure
    (fun (i : ι) (z : ι → ℝ) => z i) standard_coordinate_memLp A
  rw [standard_covariance, Matrix.mul_one] at h
  ext i j
  change cov[fun z => (A *ᵥ z) i + center i,
    fun z => (A *ᵥ z) j + center j; standardMeasure] = _
  rw [covariance_add_const_left ((linear_coordinate_memLp A i).integrable (by norm_num)),
    covariance_add_const_right ((linear_coordinate_memLp A j).integrable (by norm_num))]
  exact congrArg (fun M : Matrix ι ι ℝ => M i j) h

theorem affine_density_coordinate_memLp (M : Matrix ι ι ℝ) (hM : IsUnit M.det)
    (center : ι → ℝ) (i : ι) :
    MemLp (fun x : ι → ℝ => x i) 2 (affineGaussianMeasure (1 / 2) M center) := by
  rw [affine_density_is_pushforward (1 / 2) M hM center]
  change MemLp (fun x => x i) 2 (Measure.map (inverseWhitening M center) standardMeasure)
  apply (memLp_map_measure_iff (measurable_pi_apply i).aestronglyMeasurable
    (inverseWhitening_measurable M center).aemeasurable).mpr
  exact affine_coordinate_memLp M⁻¹ center i

theorem affine_density_mean (M : Matrix ι ι ℝ) (hM : IsUnit M.det)
    (center : ι → ℝ) (i : ι) :
    (∫ x : ι → ℝ, x i ∂affineGaussianMeasure (1 / 2) M center) = center i := by
  rw [affine_density_is_pushforward (1 / 2) M hM center]
  rw [integral_map (inverseWhitening_measurable M center).aemeasurable
    (measurable_pi_apply i).aestronglyMeasurable]
  exact affine_coordinate_mean M⁻¹ center i

theorem affine_density_covariance (M : Matrix ι ι ℝ) (hM : IsUnit M.det)
    (center : ι → ℝ) :
    Covariance.covarianceMatrix (affineGaussianMeasure (1 / 2) M center)
      (fun i x => x i) = M⁻¹ * M⁻¹ᵀ := by
  rw [affine_density_is_pushforward (1 / 2) M hM center]
  have h := affine_coordinate_covariance M⁻¹ center
  ext i j
  change cov[fun x : ι → ℝ => x i, fun x : ι → ℝ => x j;
    Measure.map (inverseWhitening M center) standardMeasure] = _
  rw [covariance_map_fun (measurable_pi_apply i).aestronglyMeasurable
    (measurable_pi_apply j).aestronglyMeasurable
    (inverseWhitening_measurable M center).aemeasurable]
  exact congrArg (fun A : Matrix ι ι ℝ => A i j) h

theorem affine_density_covariance_inverse_gram (M : Matrix ι ι ℝ) (hM : IsUnit M.det)
    (center : ι → ℝ) :
    Covariance.covarianceMatrix (affineGaussianMeasure (1 / 2) M center)
      (fun i x => x i) = (Mᵀ * M)⁻¹ := by
  rw [affine_density_covariance M hM center, Matrix.mul_inv_rev,
    Matrix.transpose_nonsing_inv]

section MatrixCoordinates

variable {p c n : Type*} [Fintype p] [Fintype c] [Fintype n]
variable [DecidableEq p] [DecidableEq c] [DecidableEq n]

theorem row_factor_isUnit (R A : Matrix p p ℝ) (hR : Rᵀ * R = A) (hA : A.PosDef) :
    IsUnit R.det := by
  have hprod : IsUnit (R.det * R.det) := by
    simpa only [← hR, Matrix.det_mul, Matrix.det_transpose] using positive_definite_det_unit hA
  exact isUnit_of_mul_isUnit_left hprod

theorem matrix_density_mean (R : Matrix p p ℝ) (L : Matrix c c ℝ)
    (center : Matrix p c ℝ) (hR : IsUnit R.det) (hL : IsUnit L.det) (i : p) (j : c) :
    (∫ x : (p × c) → ℝ, x (i, j) ∂matrixKernelMeasure R L center) = center i j :=
  affine_density_mean (whiteningMatrix R L)
    (isUnit_iff_ne_zero.mpr (whiteningMatrix_det_ne_zero R L hR hL)) (vectorize center) (i, j)

theorem matrix_density_covariance (R A : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (center : Matrix p c ℝ) (hR : Rᵀ * R = A) (hL : L * Lᵀ = Sigma)
    (hA : A.PosDef) (hSigma : Sigma.PosDef) :
    Covariance.covarianceMatrix (matrixKernelMeasure R L center)
      (fun i x => x i) = A⁻¹ ⊗ₖ Sigma := by
  change Covariance.covarianceMatrix
    (affineGaussianMeasure (1 / 2) (whiteningMatrix R L) (vectorize center)) _ = _
  rw [affine_density_covariance_inverse_gram (whiteningMatrix R L)
    (isUnit_iff_ne_zero.mpr (whiteningMatrix_det_ne_zero R L (row_factor_isUnit R A hR hA)
      (whiteningFactor_unit L Sigma hL hSigma))),
    whitening_gram_inverse R A L Sigma hR hL hSigma]

theorem source_posterior_mean (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) (i : p) (j : c) :
    (∫ x : (p × c) → ℝ, x (i, j)
      ∂matrixKernelMeasure R L (mean D Y K lam)) = mean D Y K lam i j :=
  matrix_density_mean R L _ (normalFactor_unit D K lam R hR hK hlam)
    (whiteningFactor_unit L Sigma hL hSigma) i j

theorem source_posterior_covariance (D : Matrix n p ℝ) (Y : Matrix n c ℝ)
    (K : Matrix n n ℝ) (lam : ℝ) (R : Matrix p p ℝ) (L Sigma : Matrix c c ℝ)
    (hR : Rᵀ * R = precision D K lam) (hL : L * Lᵀ = Sigma)
    (hK : K.PosDef) (hlam : 0 < lam) (hSigma : Sigma.PosDef) :
    Covariance.covarianceMatrix (matrixKernelMeasure R L (mean D Y K lam))
      (fun i x => x i) = rowCov D K lam ⊗ₖ Sigma :=
  matrix_density_covariance R (precision D K lam) L Sigma _ hR hL
    (precision_positive_definite D K lam hK hlam) hSigma

end MatrixCoordinates

end

end N2P3.MatrixNormalMoments
