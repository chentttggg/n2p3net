import N2P3.OcularCalibrationOLS
import Mathlib.Tactic.FinCases

namespace N2P3.OcularCalibrationControls
open N2P3.OcularCalibrationOLS Matrix
open scoped Matrix BigOperators
noncomputable section

def signedDesign (amplitude : ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  fun s _ => if s=0 then -amplitude else amplitude

def scalarCoefficient (value : ℝ) : Matrix (Fin 1) (Fin 1) ℝ := fun _ _ => value

theorem signed_nonzero_calibration_design_has_independent_columns (a : ℝ) (ha : a ≠ 0) :
    Function.Injective (signedDesign a).mulVec := by
  intro x y equal
  funext i
  have hi : i=0 := Subsingleton.elim _ _
  subst i
  apply mul_left_cancel₀ ha
  simpa [signedDesign,Matrix.mulVec,dotProduct,Fin.sum_univ_succ] using congrFun equal (0 : Fin 2)

theorem an_independently_generated_linear_response_recovers_its_coefficients
    {S E C : Type*} [Fintype S] [Fintype E] [DecidableEq E]
    (D : Matrix S E ℝ) (rank : Function.Injective D.mulVec) (B : Matrix E C ℝ) :
    coefficients D (D*B)=B := by
  exact (independent_full_column_rank_gives_unique_normal_equation_coefficients D (D*B) rank B
    (by simp only [gram,Matrix.mul_assoc])).symm

theorem signed_design_prediction_is_the_independent_response (a b : ℝ) :
    signedDesign a*scalarCoefficient b=signedDesign (a*b) := by
  ext s c
  fin_cases s <;> fin_cases c <;>
    simp [signedDesign,scalarCoefficient,Matrix.mul_apply]

theorem signed_full_rank_response_recovers_the_declared_coefficient (a b : ℝ) (ha : a ≠ 0) :
    coefficients (signedDesign a) (signedDesign (a*b))=scalarCoefficient b := by
  rw [← signed_design_prediction_is_the_independent_response]
  exact an_independently_generated_linear_response_recovers_its_coefficients _
    (signed_nonzero_calibration_design_has_independent_columns a ha) _

theorem fixed_preserved_reference_changes_the_fitted_coefficient :
    (coefficients (signedDesign 1) (signedDesign 2)) 0 0=2 ∧
    (coefficients (signedDesign 1) (signedDesign 2-signedDesign 1)) 0 0=1 := by
  have two : coefficients (signedDesign 1) (signedDesign 2)=scalarCoefficient 2 := by
    simpa only [one_mul] using signed_full_rank_response_recovers_the_declared_coefficient 1 2 (by norm_num)
  have one : coefficients (signedDesign 1) (signedDesign 1)=scalarCoefficient 1 := by
    simpa only [one_mul] using signed_full_rank_response_recovers_the_declared_coefficient 1 1 (by norm_num)
  rw [same_design_OLS_is_linear_in_response_subtraction,two,one]
  norm_num [scalarCoefficient]

theorem changing_the_shared_EOG_design_does_not_retain_the_same_design_coefficient :
    (coefficients (signedDesign 2) (signedDesign 3)) 0 0=3/2 ∧
    (coefficients (signedDesign 1) (signedDesign 3)) 0 0=3 := by
  have shared : coefficients (signedDesign 2) (signedDesign 3)=scalarCoefficient (3/2) := by
    convert signed_full_rank_response_recovers_the_declared_coefficient 2 (3/2) (by norm_num) using 1
    norm_num
  have preserved : coefficients (signedDesign 1) (signedDesign 3)=scalarCoefficient 3 := by
    simpa only [one_mul] using signed_full_rank_response_recovers_the_declared_coefficient 1 3 (by norm_num)
  rw [shared,preserved]
  norm_num [scalarCoefficient]

theorem dropping_full_rank_admits_distinct_normal_equation_solutions :
    ∃ B Z : Matrix (Fin 1) (Fin 1) ℝ, B ≠ Z ∧
      gram (0 : Matrix (Fin 2) (Fin 1) ℝ)*B=(0 : Matrix (Fin 1) (Fin 1) ℝ) ∧
      gram (0 : Matrix (Fin 2) (Fin 1) ℝ)*Z=(0 : Matrix (Fin 1) (Fin 1) ℝ) := by
  refine ⟨0,1,?_,?_,?_⟩
  · simp
  · simp [gram]
  · simp [gram]

end
end N2P3.OcularCalibrationControls
