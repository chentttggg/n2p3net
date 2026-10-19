import N2P3.GaussianMeasure

/-!
# The affine Gaussian density is an actual transported probability law

Identify the already normalized affine density with the pushforward of the
finite product density by the inverse whitening map. The Jacobian is derived
from Lebesgue measure, not supplied as a distributional premise. This is the
measure identity used to transport actual moments to source matrix coordinates.
-/

namespace N2P3.GaussianAffineLaw

open MeasureTheory Matrix
open N2P3.GaussianMeasure
open scoped Matrix ENNReal BigOperators

noncomputable section

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def inverseWhitening (M : Matrix ι ι ℝ) (center z : ι → ℝ) : ι → ℝ :=
  M⁻¹ *ᵥ z + center

theorem inverseWhitening_measurable (M : Matrix ι ι ℝ) (center : ι → ℝ) :
    Measurable (inverseWhitening M center) :=
  ((Matrix.toLin' M⁻¹).continuous_on_pi.add continuous_const).measurable

theorem whitening_inverseWhitening (M : Matrix ι ι ℝ) (hM : IsUnit M.det)
    (center z : ι → ℝ) : M *ᵥ (inverseWhitening M center z - center) = z := by
  rw [inverseWhitening, add_sub_cancel_right, Matrix.mulVec_mulVec,
    Matrix.mul_nonsing_inv M hM, Matrix.one_mulVec]

theorem inverseWhitening_volume (M : Matrix ι ι ℝ) (hM : IsUnit M.det)
    (center : ι → ℝ) :
    Measure.map (inverseWhitening M center) (volume : Measure (ι → ℝ)) =
      ENNReal.ofReal |M.det| • volume := by
  have hf : Measurable (fun x : ι → ℝ => x + center) :=
    (continuous_id.add continuous_const).measurable
  have hlin : Measurable (Matrix.toLin' M⁻¹) := (Matrix.toLin' M⁻¹).continuous_on_pi.measurable
  change Measure.map ((fun x : ι → ℝ => x + center) ∘ Matrix.toLin' M⁻¹) volume = _
  rw [← Measure.map_map hf hlin,
    matrix_map_volume M⁻¹ (Matrix.isUnit_nonsing_inv_det M hM).ne_zero,
    Measure.map_smul, map_add_right_eq_self]
  simp only [Matrix.det_nonsing_inv, Ring.inverse_eq_inv, abs_inv, inv_inv]

/-- The map of a composed density is the density on the mapped measure. -/
theorem map_composed_density {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (f : α → β) (g : β → ℝ≥0∞)
    (hf : Measurable f) (hg : Measurable g) :
    Measure.map f (μ.withDensity (g ∘ f)) = (Measure.map f μ).withDensity g := by
  apply Measure.ext
  intro s hs
  rw [Measure.map_apply hf hs, withDensity_apply _ (hf hs), withDensity_apply _ hs,
    setLIntegral_map hs hg hf]
  rfl

/-- An equality of measures, not just equality of the total mass. -/
theorem affine_density_is_pushforward (b : ℝ) (M : Matrix ι ι ℝ)
    (hM : IsUnit M.det) (center : ι → ℝ) :
    affineGaussianMeasure b M center =
      Measure.map (inverseWhitening M center)
        (volume.withDensity (fun z => ENNReal.ofReal (productPDF b z))) := by
  let g : (ι → ℝ) → ℝ≥0∞ :=
    fun x => ENNReal.ofReal (productPDF b (M *ᵥ (x - center)))
  have hg : Measurable g := by
    exact ((productPDF_continuous b).comp
      ((Matrix.toLin' M).continuous_on_pi.comp
        (continuous_id.sub continuous_const))).measurable.ennreal_ofReal
  have hcomp : (g ∘ inverseWhitening M center) =
      (fun z => ENNReal.ofReal (productPDF b z)) := by
    funext z
    simp only [g, Function.comp_apply, whitening_inverseWhitening M hM]
  rw [← hcomp, map_composed_density _ _ g (inverseWhitening_measurable M center) hg,
    inverseWhitening_volume M hM center, withDensity_smul_measure]
  unfold affineGaussianMeasure
  have heq : (fun x => ENNReal.ofReal (affinePDF b M center x)) =
      ENNReal.ofReal |M.det| • g := by
    funext x
    simp only [affinePDF, ENNReal.ofReal_mul (abs_nonneg M.det), Pi.smul_apply,
      smul_eq_mul, g]
  rw [heq, withDensity_smul' _ _ ENNReal.ofReal_ne_top]

end

end N2P3.GaussianAffineLaw
