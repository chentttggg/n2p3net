import N2P3.AlignmentUnitScaling
import N2P3.PreprocessingFitScope

namespace N2P3.FalsePreprocessingFitContracts
open N2P3.OcularPointwise N2P3.AlignmentUnitScaling N2P3.PreprocessingFitScope
open scoped BigOperators
noncomputable section

theorem false_replacing_the_calibration_center_is_invisible :
    correct (fun _ : Fin 1 => 1) (fun _ => 0) 1 2 (fun _ => 3)=
      correct (fun _ : Fin 1 => 1) (fun _ => 1) 1 2 (fun _ => 3) := by
  norm_num [correct,Fin.sum_univ_succ]

theorem false_zero_strength_bypasses_measured_reference :
    referenced (fun _ : Fin 1 => 2) (fun _ => 0) 0 true 2 1 (fun _ => 3)=2 := by
  norm_num [referenced,referenceEOG,correct,Fin.sum_univ_succ]

theorem false_reference_policies_have_the_same_native_dependencies :
    referenced (fun _ : Fin 1 => 2) (fun _ => 0) 1 true 2 1 (fun _ => 3)=
      referenced (fun _ : Fin 1 => 2) (fun _ => 0) 1 false 2 1 (fun _ => 3) := by
  norm_num [referenced,referenceEOG,correct,Fin.sum_univ_succ]

theorem false_shared_EOG_anchor_always_has_coefficient_minus_one :
    referenced (fun _ : Fin 1 => 2) (fun _ => 0) 1 true 2 2 (fun _ => 3)-
      referenced (fun _ : Fin 1 => 2) (fun _ => 0) 1 true 2 1 (fun _ => 3)= -1 := by
  norm_num [referenced,referenceEOG,correct,Fin.sum_univ_succ]

theorem false_the_dependency_matrix_contains_the_frozen_DC_offset :
    referenced (fun _ : Fin 1 => 2) (fun _ => 1) 1 true 2 1 (fun _ => 3)=
      dependency (fun _ : Fin 1 => 2) 1 true 2 1 (fun _ => 3) := by
  norm_num [referenced,dependency,referenceEOG,correct,Fin.sum_univ_succ]

theorem false_regularized_EA_always_exactly_whitens_empirical_energy :
    1/regularizedVariance (shrunkEigenvalue (2/5) (4/5)) (1/1000000) (5/2)=1 := by
  norm_num [regularizedVariance,shrunkEigenvalue]

theorem false_the_uncentered_second_moment_ignores_DC_translation :
    secondMoment (fun (_ : Fin 2) (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)) 0 0=
      secondMoment (fun (_ : Fin 2) (_ : Fin 1) (_ : Fin 1) => (2 : ℝ)) 0 0 := by
  norm_num [secondMoment,Fin.sum_univ_succ]

theorem false_rescaling_only_the_input_preserves_dimensionless_components :
    transform (fun (_ _ : Fin 1) => 1) (fun _ => 2) 0=
      transform (fun (_ _ : Fin 1) => 1) (fun _ => 1) 0 := by
  norm_num [transform,Fin.sum_univ_succ]

theorem false_selecting_one_endpoint_admits_an_adjacent_difference :
    eligibleDifference 2 (fun t => decide (t=0)) (fun _ => true) (fun _ => false) 0 := by
  norm_num [eligibleDifference]

theorem false_selecting_both_endpoints_allows_crossing_a_segment_boundary :
    eligibleDifference 2 (fun _ => true) (fun _ => true) (fun t => decide (t=1)) 0 := by
  norm_num [eligibleDifference]

theorem false_unnormalized_overlap_preserves_equal_corrections :
    (∑ _ : Fin 2, (1 : ℝ)*3)=3 := by
  norm_num [Fin.sum_univ_succ]

theorem false_negative_overlap_weights_preserve_the_common_lower_bound :
    0 ≤ overlapCorrection (fun i : Fin 2 => if i=0 then 2 else -1)
      (fun i => if i=0 then 0 else 1) := by
  norm_num [overlapCorrection,Fin.sum_univ_succ]

theorem false_extra_heldout_physical_identity_matches_the_fixed_authorized_order :
    ([0,1] : List ℕ).map (fun row => if row=0 then "a" else "c")=["a","b"] := by
  simp

theorem false_an_absent_domain_has_an_authorized_fit_state :
    "d2" ∈ (["d1"] : List String).toFinset := by
  simp

end
end N2P3.FalsePreprocessingFitContracts
