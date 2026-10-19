import N2P3.OcularPointwise
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Tactic.FieldSimp

/-!
# Centered calibration-only OLS behind the frozen ocular coefficients

Source-fixed science_contract_02.json. ocular.py:230-242 selects calibration
samples, centers BOTH design and response, fits without an intercept, then
checks full EOG column rank. Here the independent design injectivity gives
Gram positivity/invertibility before proving the explicit OLS normal equation
and uniqueness. Coefficients are EOG-by-EEG; sklearn.coef_ transposes this image.
Same-design reference subtraction concerns preserve_acquired_eog. It does not
transfer to the changed EOG design of shared measured referencing.
No claim about native SVD rank/rounding or nuisance/brain independence occurs.
-/
namespace N2P3.OcularCalibrationOLS
open Matrix
open scoped Matrix BigOperators ComplexOrder
noncomputable section
variable {S E C : Type*} [Fintype S] [Fintype E] [DecidableEq E]

def sampleMean (x : S → ℝ) : ℝ := (∑ s, x s)/(Fintype.card S : ℝ)

def centered (x : Matrix S C ℝ) : Matrix S C ℝ := fun s c => x s c-sampleMean (fun t => x t c)

theorem empirical_mean_respects_response_subtraction (x y : S → ℝ) :
    sampleMean (fun s => x s-y s)=sampleMean x-sampleMean y := by
  simp only [sampleMean,Finset.sum_sub_distrib,sub_div]

theorem empirical_mean_scales_with_voltage_units (x : S → ℝ) (k : ℝ) :
    sampleMean (fun s => k*x s)=k*sampleMean x := by
  simp only [sampleMean,← Finset.mul_sum]
  ring

theorem centered_calibration_columns_have_zero_sum [Nonempty S] (x : Matrix S C ℝ) (c : C) :
    (∑ s, centered x s c)=0 := by
  have size : (Fintype.card S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp only [centered,Finset.sum_sub_distrib,Finset.sum_const,Finset.card_univ,nsmul_eq_mul,sampleMean]
  field_simp
  ring

theorem centering_respects_response_reference_subtraction (x reference : Matrix S C ℝ) :
    centered (x-reference)=centered x-centered reference := by
  ext s c
  simp only [centered,Matrix.sub_apply,empirical_mean_respects_response_subtraction]
  ring

theorem centering_scales_with_voltage_units (x : Matrix S C ℝ) (k : ℝ) :
    centered (k • x)=k • centered x := by
  ext s c
  simp only [centered,Matrix.smul_apply,smul_eq_mul,empirical_mean_scales_with_voltage_units]
  ring

def gram (D : Matrix S E ℝ) : Matrix E E ℝ := Dᵀ*D

def coefficients (D : Matrix S E ℝ) (Y : Matrix S C ℝ) : Matrix E C ℝ :=
  (gram D)⁻¹*Dᵀ*Y

omit [DecidableEq E] in
theorem independent_full_column_rank_makes_the_actual_Gram_positive_definite
    (D : Matrix S E ℝ) (rank : Function.Injective D.mulVec) : (gram D).PosDef := by
  simpa only [gram,Matrix.conjTranspose_eq_transpose_of_trivial] using Matrix.PosDef.conjTranspose_mul_self D rank

theorem full_column_rank_supplies_an_actual_invertible_Gram_determinant
    (D : Matrix S E ℝ) (rank : Function.Injective D.mulVec) : IsUnit (gram D).det :=
  (Matrix.isUnit_iff_isUnit_det (gram D)).mp
    (independent_full_column_rank_makes_the_actual_Gram_positive_definite D rank).isUnit

theorem explicit_OLS_coefficients_solve_the_normal_equations
    (D : Matrix S E ℝ) (Y : Matrix S C ℝ) (rank : Function.Injective D.mulVec) :
    gram D*coefficients D Y=Dᵀ*Y := by
  unfold coefficients
  rw [← Matrix.mul_assoc,← Matrix.mul_assoc,
    Matrix.mul_nonsing_inv _ (full_column_rank_supplies_an_actual_invertible_Gram_determinant D rank),
    Matrix.one_mul]

theorem independent_full_column_rank_gives_unique_normal_equation_coefficients
    (D : Matrix S E ℝ) (Y : Matrix S C ℝ) (rank : Function.Injective D.mulVec)
    (B : Matrix E C ℝ) (normal : gram D*B=Dᵀ*Y) : B=coefficients D Y := by
  calc
    B = (gram D)⁻¹*(gram D*B) := by
      rw [← Matrix.mul_assoc,
        Matrix.nonsing_inv_mul _ (full_column_rank_supplies_an_actual_invertible_Gram_determinant D rank),
        Matrix.one_mul]
    _ = coefficients D Y := by rw [normal]; simp only [coefficients,Matrix.mul_assoc]

theorem calibration_residual_is_orthogonal_to_the_selected_EOG_design
    (D : Matrix S E ℝ) (Y : Matrix S C ℝ) (rank : Function.Injective D.mulVec) :
    Dᵀ*(Y-D*coefficients D Y)=0 := by
  rw [Matrix.mul_sub,← Matrix.mul_assoc]
  change Dᵀ*Y-gram D*coefficients D Y=0
  rw [explicit_OLS_coefficients_solve_the_normal_equations D Y rank,sub_self]

theorem same_design_OLS_is_linear_in_response_subtraction
    (D : Matrix S E ℝ) (Y reference : Matrix S C ℝ) :
    coefficients D (Y-reference)=coefficients D Y-coefficients D reference := by
  simp only [coefficients,Matrix.mul_sub]

theorem preserved_EOG_reference_before_fit_gives_the_difference_of_native_coefficients
    (EOG : Matrix S E ℝ) (EEG reference : Matrix S C ℝ) :
    coefficients (centered EOG) (centered (EEG-reference))=
      coefficients (centered EOG) (centered EEG)-coefficients (centered EOG) (centered reference) := by
  rw [centering_respects_response_reference_subtraction,same_design_OLS_is_linear_in_response_subtraction]

theorem joint_voltage_unit_change_preserves_OLS_coefficients
    (D : Matrix S E ℝ) (Y : Matrix S C ℝ) (rank : Function.Injective D.mulVec)
    (k : ℝ) (hk : k ≠ 0) : coefficients (k • D) (k • Y)=coefficients D Y := by
  have gramScale : gram (k • D)=k^2 • gram D := by
    simp only [gram,Matrix.transpose_smul,Matrix.smul_mul,Matrix.mul_smul,smul_smul,pow_two]
  letI : Invertible (k^2) := invertibleOfNonzero (pow_ne_zero 2 hk)
  have inverse : (gram (k • D))⁻¹=(k^2)⁻¹ • (gram D)⁻¹ := by
    rw [gramScale,Matrix.inv_smul (gram D) (k^2) (full_column_rank_supplies_an_actual_invertible_Gram_determinant D rank)]
    simp only [invOf_eq_inv]
  have factors : k*(k*(k^2)⁻¹)=1 := by
    rw [← mul_assoc,← sq,mul_inv_cancel₀ (pow_ne_zero 2 hk)]
  simp only [coefficients,inverse,Matrix.transpose_smul,
    Matrix.smul_mul,Matrix.mul_smul,smul_smul,factors,one_smul]

theorem centered_source_fit_preserves_coefficients_under_joint_voltage_unit_change
    (EOG : Matrix S E ℝ) (EEG : Matrix S C ℝ) (rank : Function.Injective (centered EOG).mulVec)
    (k : ℝ) (hk : k ≠ 0) :
    coefficients (centered (k • EOG)) (centered (k • EEG))=
      coefficients (centered EOG) (centered EEG) := by
  rw [centering_scales_with_voltage_units,centering_scales_with_voltage_units]
  exact joint_voltage_unit_change_preserves_OLS_coefficients (centered EOG) (centered EEG) rank k hk

omit [DecidableEq E] in
theorem joint_voltage_unit_change_scales_corrected_physical_voltage
    (B mean eye : E → ℝ) (strength x k : ℝ) :
    OcularPointwise.correct B (fun e => k*mean e) strength (k*x) (fun e => k*eye e)=
      k*OcularPointwise.correct B mean strength x eye := by
  unfold OcularPointwise.correct
  have scaled : (∑ e, B e*(k*eye e-k*mean e))=k*(∑ e, B e*(eye e-mean e)) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro e _
    ring
  rw [scaled]
  ring

end
end N2P3.OcularCalibrationOLS
