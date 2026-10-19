import N2P3.OcularCalibrationOLS
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Conditional harmonic weighted fit and the actual IRLS weight chronology

Source: harmonic.py:92-110. A fit uses weights from the previous update;
the returned weights are newer. Normal equations here use the ACTUAL fit
weights. No claim about optimality under returned weights or global frequency
optimization is made. Nuisance columns condition the fit but only harmonic
columns are subtracted. All native solver/median/rounding bounds stay separate.
-/
namespace N2P3.HarmonicWeightedFit
open Matrix
open scoped BigOperators Matrix
noncomputable section

def huberWeight (residual scale floor multiplier : ℝ) : ℝ :=
  min 1 (multiplier*scale/max |residual| floor)

theorem source_Huber_weights_are_positive_and_bounded_by_one
    (residual scale floor multiplier : ℝ) (floorPositive : 0 < floor)
    (scaleFloor : floor ≤ scale) (multiplierPositive : 0 < multiplier) :
    0 < huberWeight residual scale floor multiplier ∧ huberWeight residual scale floor multiplier ≤ 1 := by
  have scalePositive : 0 < scale := lt_of_lt_of_le floorPositive scaleFloor
  have denominator : 0 < max |residual| floor := lt_of_lt_of_le floorPositive (le_max_right _ _)
  constructor
  · exact lt_min zero_lt_one (div_pos (mul_pos multiplierPositive scalePositive) denominator)
  · exact min_le_left _ _

theorem every_new_MAD_weight_update_remains_strictly_positive
    (residual mad floor multiplier : ℝ) (floorPositive : 0 < floor) (multiplierPositive : 0 < multiplier) :
    0 < huberWeight residual (max mad floor) floor multiplier :=
  (source_Huber_weights_are_positive_and_bounded_by_one residual (max mad floor) floor multiplier
    floorPositive (le_max_right _ _) multiplierPositive).1

theorem square_root_of_each_new_weight_is_a_nonzero_row_scale
    (residual mad floor multiplier : ℝ) (floorPositive : 0 < floor) (multiplierPositive : 0 < multiplier) :
    Real.sqrt (huberWeight residual (max mad floor) floor multiplier) ≠ 0 :=
  (Real.sqrt_pos.mpr (every_new_MAD_weight_update_remains_strictly_positive residual mad floor multiplier
    floorPositive multiplierPositive)).ne'

variable {S E C : Type*} [Fintype S] [Fintype E] [DecidableEq E]

def rowScaled (scale : S → ℝ) (D : Matrix S E ℝ) : Matrix S E ℝ := fun s e => scale s*D s e

omit [Fintype S] [DecidableEq E] in
theorem source_row_scaling_is_the_same_scaling_after_mulVec
    (scale : S → ℝ) (D : Matrix S E ℝ) (x : E → ℝ) :
    (rowScaled scale D).mulVec x=fun s => scale s*(D.mulVec x s) := by
  funext s
  simp only [rowScaled,Matrix.mulVec,dotProduct,Finset.mul_sum,mul_assoc]

omit [Fintype S] [DecidableEq E] in
theorem positive_row_weights_preserve_independent_design_columns
    (scale : S → ℝ) (D : Matrix S E ℝ) (nonzero : ∀ s, scale s ≠ 0)
    (rank : Function.Injective D.mulVec) : Function.Injective (rowScaled scale D).mulVec := by
  intro x y same
  apply rank
  funext s
  apply mul_left_cancel₀ (nonzero s)
  simpa only [source_row_scaling_is_the_same_scaling_after_mulVec] using congrFun same s

theorem each_harmonic_fit_satisfies_normal_equations_for_its_actual_used_weights
    (scale : S → ℝ) (D : Matrix S E ℝ) (Y : Matrix S C ℝ)
    (nonzero : ∀ s, scale s ≠ 0) (rank : Function.Injective D.mulVec) :
    OcularCalibrationOLS.gram (rowScaled scale D)*
      OcularCalibrationOLS.coefficients (rowScaled scale D) (rowScaled scale Y)=
      (rowScaled scale D)ᵀ*rowScaled scale Y :=
  OcularCalibrationOLS.explicit_OLS_coefficients_solve_the_normal_equations _ _
    (positive_row_weights_preserve_independent_design_columns scale D nonzero rank)

def cleanOnlyLine (signal line : ℝ) : ℝ := signal-line

theorem subtracting_only_harmonics_retains_the_fitted_nuisance_and_residual
    (signal nuisance line : ℝ) : cleanOnlyLine signal line=nuisance+(signal-(nuisance+line)) := by
  unfold cleanOnlyLine
  ring

def weightedNormalResidual {I : Type*} [Fintype I] (weights design response : I → ℝ) (beta : ℝ) : ℝ :=
  ∑ i, weights i*design i*(response i-design i*beta)

theorem later_updated_weights_need_not_satisfy_the_normal_equation_of_the_previous_fit :
    weightedNormalResidual (fun _ : Fin 3 => 1) (fun _ => 1)
      ![0,0,10] (10/3)=0 ∧
    weightedNormalResidual ![2,2,1] (fun _ : Fin 3 => 1)
      ![0,0,10] (10/3)= -20/3 := by
  norm_num [weightedNormalResidual,Fin.sum_univ_succ]

theorem fixed_Huber_weight_control : huberWeight 4 1 1 2=1/2 ∧ huberWeight 0 1 1 2=1 := by
  norm_num [huberWeight]

end
end N2P3.HarmonicWeightedFit
