import N2P3.OcularCalibrationControls

namespace N2P3.FalseOcularCalibrationOLS
open N2P3.OcularCalibrationOLS N2P3.OcularCalibrationControls Matrix
open scoped Matrix
noncomputable section

theorem false_rank_free_normal_equations_have_no_distinct_solutions :
    ¬∃ B Z : Matrix (Fin 1) (Fin 1) ℝ, B ≠ Z ∧
      gram (0 : Matrix (Fin 2) (Fin 1) ℝ)*B=(0 : Matrix (Fin 1) (Fin 1) ℝ) ∧
      gram (0 : Matrix (Fin 2) (Fin 1) ℝ)*Z=(0 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [dropping_full_rank_admits_distinct_normal_equation_solutions]

theorem false_subtracting_the_EEG_reference_preserves_its_original_coefficient :
    (coefficients (signedDesign 1) (signedDesign 2-signedDesign 1)) 0 0=2 := by
  norm_num [fixed_preserved_reference_changes_the_fitted_coefficient.2]

theorem false_shared_EOG_inherits_the_unchanged_design_coefficient :
    (coefficients (signedDesign 2) (signedDesign 3)) 0 0=3 := by
  norm_num [changing_the_shared_EOG_design_does_not_retain_the_same_design_coefficient.1]

end
end N2P3.FalseOcularCalibrationOLS
