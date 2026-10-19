import N2P3.GaussianMeasure
import N2P3.Covariance
import Mathlib.Probability.Distributions.Gaussian.Basic

/-!
# Actual first and second moments of the normalized finite Gaussian density

Compute the scalar moments from the pinned real Gaussian distribution, then
use finite-product integration to derive the moments of the density already
normalized in GaussianMeasure. Moments and L2 membership are derived from the
measure; no covariance or integrability value is an input hypothesis.
-/

namespace N2P3.GaussianMoments

open MeasureTheory ProbabilityTheory Matrix
open N2P3.GaussianMeasure
open scoped BigOperators ProbabilityTheory ENNReal NNReal Matrix

noncomputable section

theorem standard_scalar_pdf (x : ℝ) :
    scalarPDF (1 / 2) x = gaussianPDFReal 0 1 x := by
  unfold scalarPDF gaussianPDFReal
  have hn : Real.pi / (1 / 2 : ℝ) = 2 * Real.pi * (↑(1 : ℝ≥0) : ℝ) := by
    norm_num
    ring
  have he : -(1 / 2 : ℝ) * x ^ 2 = -(x - 0) ^ 2 / (2 * (↑(1 : ℝ≥0) : ℝ)) := by
    norm_num
    ring
  rw [hn, he]

theorem standard_scalar_first_integrable :
    Integrable (fun x : ℝ => scalarPDF (1 / 2) x * x) := by
  have h := (integrable_mul_exp_neg_mul_sq (b := (1 / 2 : ℝ)) (by norm_num)).const_mul
    ((Real.sqrt (Real.pi / (1 / 2 : ℝ)))⁻¹)
  convert h using 1
  funext x
  unfold scalarPDF
  ring

theorem standard_scalar_second_integrable :
    Integrable (fun x : ℝ => scalarPDF (1 / 2) x * x ^ 2) := by
  have h := (integrable_rpow_mul_exp_neg_mul_sq (b := (1 / 2 : ℝ))
    (by norm_num) (s := 2) (by norm_num)).const_mul
    ((Real.sqrt (Real.pi / (1 / 2 : ℝ)))⁻¹)
  convert h using 1
  funext x
  rw [Real.rpow_two]
  unfold scalarPDF
  ring

theorem standard_scalar_first_moment :
    (∫ x : ℝ, scalarPDF (1 / 2) x * x) = 0 := by
  have h := integral_id_gaussianReal (μ := 0) (v := 1)
  rw [integral_gaussianReal_eq_integral_smul (by norm_num : (1 : ℝ≥0) ≠ 0)] at h
  simpa only [standard_scalar_pdf, smul_eq_mul] using h

theorem standard_scalar_second_moment :
    (∫ x : ℝ, scalarPDF (1 / 2) x * x ^ 2) = 1 := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := 1)
  rw [variance_eq_integral measurable_id'.aemeasurable] at h
  simp only [integral_id_gaussianReal, sub_zero] at h
  rw [integral_gaussianReal_eq_integral_smul (by norm_num : (1 : ℝ≥0) ≠ 0)] at h
  simpa only [standard_scalar_pdf, smul_eq_mul, NNReal.coe_one] using h

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def standardMeasure : Measure (ι → ℝ) :=
  volume.withDensity (fun x => ENNReal.ofReal (productPDF (1 / 2) x))

omit [DecidableEq ι] in
theorem standardMeasure_probability : IsProbabilityMeasure (standardMeasure (ι := ι)) := by
  apply isProbabilityMeasure_iff.mpr
  rw [standardMeasure, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    ← ofReal_integral_eq_lintegral_ofReal (productPDF_integrable (1 / 2) (by norm_num))
      (Filter.Eventually.of_forall (productPDF_nonnegative (1 / 2))),
    productPDF_integral_one (1 / 2) (by norm_num), ENNReal.ofReal_one]

omit [DecidableEq ι] in
theorem standard_integral_product (f : ι → ℝ → ℝ) :
    (∫ x, ∏ i, f i (x i) ∂standardMeasure) =
      ∏ i, ∫ t : ℝ, scalarPDF (1 / 2) t * f i t := by
  rw [standardMeasure, integral_withDensity_eq_integral_toReal_smul
    ((productPDF_continuous (ι := ι) (1 / 2)).measurable.ennreal_ofReal)
    (Filter.Eventually.of_forall (fun _ => ENNReal.ofReal_lt_top))]
  simp_rw [ENNReal.toReal_ofReal (productPDF_nonnegative _ _), smul_eq_mul,
    productPDF, ← Finset.prod_mul_distrib]
  exact integral_fintype_prod_volume_eq_prod
    (fun (i : ι) (t : ℝ) => scalarPDF (1 / 2) t * f i t)

omit [DecidableEq ι] in
theorem standard_integrable_product (f : ι → ℝ → ℝ)
    (hf : ∀ i, Integrable (fun t : ℝ => scalarPDF (1 / 2) t * f i t)) :
    Integrable (fun x : ι → ℝ => ∏ i, f i (x i)) standardMeasure := by
  rw [standardMeasure, integrable_withDensity_iff_integrable_smul'
    ((productPDF_continuous (ι := ι) (1 / 2)).measurable.ennreal_ofReal)
    (Filter.Eventually.of_forall (fun _ => ENNReal.ofReal_lt_top))]
  simp_rw [ENNReal.toReal_ofReal (productPDF_nonnegative _ _), smul_eq_mul,
    productPDF, ← Finset.prod_mul_distrib]
  exact Integrable.fintype_prod hf

def coordinateFactor (i k : ι) (t : ℝ) : ℝ := if k = i then t else 1

theorem coordinate_factor_product (i : ι) (x : ι → ℝ) :
    (∏ k, coordinateFactor i k (x k)) = x i := by
  simp [coordinateFactor]

theorem standard_mean (i : ι) : (∫ x : ι → ℝ, x i ∂standardMeasure) = 0 := by
  have h := standard_integral_product (coordinateFactor i)
  simp only [coordinate_factor_product] at h
  rw [h]
  apply Finset.prod_eq_zero (Finset.mem_univ i)
  simpa only [coordinateFactor, if_pos rfl] using standard_scalar_first_moment

theorem coordinate_integrable (i : ι) : Integrable (fun x : ι → ℝ => x i) standardMeasure := by
  have h := standard_integrable_product (coordinateFactor i) (fun k => by
    by_cases hk : k = i
    · simpa only [coordinateFactor, if_pos hk] using standard_scalar_first_integrable
    · simpa only [coordinateFactor, if_neg hk, mul_one] using
        scalarPDF_integrable (1 / 2) (by norm_num))
  simpa only [coordinate_factor_product] using h

theorem standard_coordinate_second_integrable (i : ι) :
    Integrable (fun x : ι → ℝ => x i ^ 2) standardMeasure := by
  have h := standard_integrable_product (fun k t => if k = i then t ^ 2 else 1) (fun k => by
    by_cases hk : k = i
    · simpa only [if_pos hk] using standard_scalar_second_integrable
    · simpa only [if_neg hk, mul_one] using scalarPDF_integrable (1 / 2) (by norm_num))
  simpa only [Finset.prod_ite_eq', Finset.mem_univ, if_true] using h

theorem standard_coordinate_memLp (i : ι) : MemLp (fun x : ι → ℝ => x i) 2 standardMeasure :=
  (memLp_two_iff_integrable_sq (measurable_pi_apply i).aestronglyMeasurable).mpr
    (standard_coordinate_second_integrable i)

theorem standard_second_moment (i j : ι) :
    (∫ x : ι → ℝ, x i * x j ∂standardMeasure) = if i = j then 1 else 0 := by
  have hp (x : ι → ℝ) :
      (∏ k, coordinateFactor i k (x k) * coordinateFactor j k (x k)) = x i * x j := by
    rw [Finset.prod_mul_distrib, coordinate_factor_product, coordinate_factor_product]
  have h := standard_integral_product (fun k t => coordinateFactor i k t * coordinateFactor j k t)
  simp_rw [hp] at h
  rw [h]
  by_cases hij : i = j
  · subst j
    rw [if_pos rfl]
    apply Finset.prod_eq_one
    intro k _
    by_cases hki : k = i
    · simpa only [coordinateFactor, if_pos hki, ← pow_two] using standard_scalar_second_moment
    · simpa only [coordinateFactor, if_neg hki, one_mul, mul_one] using
        scalarPDF_integral_one (1 / 2) (by norm_num)
  · rw [if_neg hij]
    apply Finset.prod_eq_zero (Finset.mem_univ i)
    simpa only [coordinateFactor, if_pos rfl, if_neg hij, mul_one] using standard_scalar_first_moment

theorem standard_covariance :
    Covariance.covarianceMatrix (standardMeasure (ι := ι)) (fun i x => x i) = 1 := by
  ext i j
  simp only [Covariance.covarianceMatrix, covariance, standard_mean, sub_zero]
  exact standard_second_moment i j

end

end N2P3.GaussianMoments
