import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.MeasureTheory.Integral.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.MeasureTheory.Group.Integral
import Mathlib.MeasureTheory.Measure.WithDensity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Finite Gaussian normalization and its affine Jacobian

This is a measure-theoretic slice using the pinned Mathlib commit
f897ebcf72cd16f89ab4577d0c826cd14afaafc7 (Lean 4.24.0).

The one-coordinate integral is derived from Mathlib's proved Gaussian integral;
the finite product is normalized by Fubini; the affine pullback is normalized
using Lebesgue measure's actual matrix-determinant transformation theorem.
The desired integral value is never an input hypothesis. Integrability and
nonnegativity are proved before constructing a probability measure.

The coordinate space is `ι → ℝ` with product Lebesgue measure. For the source's
nuisance projection this must be the nonredundant retained coordinate type,
not the original N-row array or a redundant frame. This file does not yet
prove a complete conditional Gaussian/disintegration theorem, quotient-space
Lebesgue volume identification, covariance moments, NumPy correspondence, or
IEEE error bounds.
-/

namespace N2P3.GaussianMeasure

open MeasureTheory Matrix
open scoped BigOperators ENNReal Matrix

/-- A normalized scalar Gaussian with positive exponent coefficient b.
At b=1/2 this is the ordinary standard-normal density. -/
noncomputable def scalarPDF (b x : ℝ) : ℝ :=
  (Real.sqrt (Real.pi / b))⁻¹ * Real.exp (-b * x ^ 2)

theorem scalar_normalizer_positive (b : ℝ) (hb : 0 < b) :
    0 < Real.sqrt (Real.pi / b) :=
  Real.sqrt_pos.mpr (div_pos Real.pi_pos hb)

theorem scalarPDF_nonnegative (b x : ℝ) : 0 ≤ scalarPDF b x := by
  exact mul_nonneg (inv_nonneg.mpr (Real.sqrt_nonneg _)) (Real.exp_pos _).le

theorem scalarPDF_continuous (b : ℝ) : Continuous (scalarPDF b) := by
  have h : Continuous (fun x : ℝ => -b * x ^ 2) :=
    continuous_const.mul (continuous_id.pow 2)
  exact continuous_const.mul (Real.continuous_exp.comp h)

theorem scalarPDF_integrable (b : ℝ) (hb : 0 < b) : Integrable (scalarPDF b) :=
  (integrable_exp_neg_mul_sq hb).const_mul _

/-- Genuine Gaussian integration; the integral equality is derived, not assumed. -/
theorem scalarPDF_integral_one (b : ℝ) (hb : 0 < b) : ∫ x, scalarPDF b x = 1 := by
  unfold scalarPDF
  rw [integral_const_mul, integral_gaussian]
  exact inv_mul_cancel₀ (ne_of_gt (scalar_normalizer_positive b hb))

section FiniteProduct

variable {ι : Type*} [Fintype ι]

noncomputable def productPDF (b : ℝ) (x : ι → ℝ) : ℝ := ∏ i, scalarPDF b (x i)

theorem productPDF_nonnegative (b : ℝ) (x : ι → ℝ) : 0 ≤ productPDF b x := by
  exact Finset.prod_nonneg (fun i _ => scalarPDF_nonnegative b (x i))

theorem productPDF_continuous (b : ℝ) : Continuous (productPDF (ι := ι) b) := by
  exact continuous_finset_prod Finset.univ
    (fun i _ => (scalarPDF_continuous b).comp (continuous_apply i))

theorem productPDF_integrable (b : ℝ) (hb : 0 < b) :
    Integrable (productPDF (ι := ι) b) := by
  exact Integrable.fintype_prod (fun _ : ι => scalarPDF_integrable b hb)

/-- Fubini turns the proved scalar normalizer into finite-dimensional mass one. -/
theorem productPDF_integral_one (b : ℝ) (hb : 0 < b) :
    ∫ x : ι → ℝ, productPDF b x = 1 := by
  unfold productPDF
  rw [integral_fintype_prod_volume_eq_pow]
  simp [scalarPDF_integral_one b hb]

/-- The normalized product is exactly the finite Gaussian exponential kernel,
with its dimension-dependent normalizer still present. -/
theorem productPDF_as_quadratic_kernel (b : ℝ) (x : ι → ℝ) :
    productPDF b x = (Real.sqrt (Real.pi / b))⁻¹ ^ Fintype.card ι *
      Real.exp (-b * ∑ i, x i ^ 2) := by
  unfold productPDF scalarPDF
  rw [Finset.prod_mul_distrib]
  simp only [Finset.prod_const, Finset.card_univ]
  rw [← Real.exp_sum, ← Finset.mul_sum]

end FiniteProduct

section AffineJacobian

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The fixed-library pushforward formula, with its inverse absolute determinant
written in the same convention used by the subsequent density transformation. -/
theorem matrix_map_volume (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) :
    Measure.map (Matrix.toLin' M) (volume : Measure (ι → ℝ)) =
      ENNReal.ofReal (|M.det|⁻¹) • volume := by
  simpa only [abs_inv] using Real.map_matrix_volume_pi_eq_smul_volume_pi hM

/-- General affine change of coordinates uses the actual Lebesgue determinant
theorem. No Gaussian integral is smuggled into this Jacobian premise. -/
theorem integral_comp_matrix
    (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) (f : (ι → ℝ) → ℝ)
    (hf : StronglyMeasurable f) :
    (∫ x, f (M *ᵥ x)) = |M.det|⁻¹ * ∫ y, f y := by
  have hlinear : Measurable (Matrix.toLin' M) :=
    (Matrix.toLin' M).continuous_on_pi.measurable
  have hchange := integral_map_of_stronglyMeasurable (μ := volume) hlinear hf
  rw [matrix_map_volume M hM, integral_smul_measure,
    ENNReal.toReal_ofReal (inv_nonneg.mpr (abs_nonneg M.det))] at hchange
  simpa only [Matrix.toLin'_apply, smul_eq_mul] using hchange.symm

/-- Integrability of the transformed function follows independently from the
measure map, so totalization of the Bochner integral cannot fake mass one. -/
theorem integrable_comp_matrix
    (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) {f : (ι → ℝ) → ℝ}
    (hf : Integrable f) : Integrable (fun x => f (M *ᵥ x)) := by
  have hscaled : Integrable f (ENNReal.ofReal (|M.det|⁻¹) • volume) :=
    hf.smul_measure ENNReal.ofReal_ne_top
  rw [← matrix_map_volume M hM] at hscaled
  have hlinear : Measurable (Matrix.toLin' M) :=
    (Matrix.toLin' M).continuous_on_pi.measurable
  simpa only [Function.comp_def, Matrix.toLin'_apply] using hscaled.comp_measurable hlinear

/-- Density after the invertible affine whitening y=M(x-mean). -/
noncomputable def affinePDF (b : ℝ) (M : Matrix ι ι ℝ) (mean x : ι → ℝ) : ℝ :=
  |M.det| * productPDF b (M *ᵥ (x - mean))

theorem affinePDF_nonnegative
    (b : ℝ) (M : Matrix ι ι ℝ) (mean x : ι → ℝ) : 0 ≤ affinePDF b M mean x :=
  mul_nonneg (abs_nonneg M.det) (productPDF_nonnegative b _)

theorem affinePDF_integrable
    (b : ℝ) (hb : 0 < b) (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) (mean : ι → ℝ) :
    Integrable (affinePDF b M mean) := by
  exact ((integrable_comp_matrix M hM (productPDF_integrable b hb)).comp_sub_right mean).const_mul _

/-- Finite-dimensional Gaussian normalization with an actual affine Jacobian. -/
theorem affinePDF_integral_one
    (b : ℝ) (hb : 0 < b) (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) (mean : ι → ℝ) :
    ∫ x, affinePDF b M mean x = 1 := by
  unfold affinePDF
  rw [integral_const_mul,
    integral_sub_right_eq_self (μ := volume) (fun x => productPDF b (M *ᵥ x)) mean,
    integral_comp_matrix M hM (productPDF b) (productPDF_continuous b).stronglyMeasurable,
    productPDF_integral_one b hb, mul_one, mul_inv_cancel₀ (abs_ne_zero.mpr hM)]

/-- The same normalized density is explicitly the Gaussian quadratic kernel. -/
theorem affinePDF_as_quadratic_kernel
    (b : ℝ) (M : Matrix ι ι ℝ) (mean x : ι → ℝ) :
    affinePDF b M mean x = |M.det| *
      (Real.sqrt (Real.pi / b))⁻¹ ^ Fintype.card ι *
        Real.exp (-b * ∑ i, (M *ᵥ (x - mean)) i ^ 2) := by
  unfold affinePDF
  rw [productPDF_as_quadratic_kernel, mul_assoc]

noncomputable def affineGaussianMeasure
    (b : ℝ) (M : Matrix ι ι ℝ) (mean : ι → ℝ) : Measure (ι → ℝ) :=
  volume.withDensity (fun x => ENNReal.ofReal (affinePDF b M mean x))

/-- This constructs a real probability measure, beyond a purely algebraic
kernel identity. Fitted parameters remain fixed during this normalization. -/
theorem affineGaussianMeasure_isProbabilityMeasure
    (b : ℝ) (hb : 0 < b) (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) (mean : ι → ℝ) :
    IsProbabilityMeasure (affineGaussianMeasure b M mean) := by
  apply isProbabilityMeasure_iff.mpr
  unfold affineGaussianMeasure
  rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    ← ofReal_integral_eq_lintegral_ofReal (affinePDF_integrable b hb M hM mean)
      (Filter.Eventually.of_forall (affinePDF_nonnegative b M mean)),
    affinePDF_integral_one b hb M hM mean, ENNReal.ofReal_one]

/-- Omitting the Jacobian genuinely changes total mass. -/
theorem omitted_affine_jacobian_mass
    (b : ℝ) (hb : 0 < b) (M : Matrix ι ι ℝ) (hM : M.det ≠ 0) (mean : ι → ℝ) :
    (∫ x, productPDF b (M *ᵥ (x - mean))) = |M.det|⁻¹ := by
  rw [integral_sub_right_eq_self (μ := volume) (fun x => productPDF b (M *ᵥ x)) mean,
    integral_comp_matrix M hM (productPDF b) (productPDF_continuous b).stronglyMeasurable,
    productPDF_integral_one b hb, mul_one]

end AffineJacobian

/-- A concrete scale-two source-coordinate change loses half the mass without
the determinant factor; normalization cannot be asserted from shape alone. -/
theorem scale_two_without_jacobian_has_half_mass (b : ℝ) (hb : 0 < b) :
    (∫ x : Fin 1 → ℝ, productPDF b (((2 : Matrix (Fin 1) (Fin 1) ℝ)) *ᵥ x)) = (1 / 2 : ℝ) := by
  have h := omitted_affine_jacobian_mass b hb (2 : Matrix (Fin 1) (Fin 1) ℝ)
    (by norm_num [Matrix.det_fin_one, Matrix.ofNat_apply]) (0 : Fin 1 → ℝ)
  simpa [Matrix.det_fin_one, Matrix.ofNat_apply] using h

end N2P3.GaussianMeasure
