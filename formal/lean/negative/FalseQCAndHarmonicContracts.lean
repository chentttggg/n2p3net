import N2P3.QCAndHarmonicContracts

namespace N2P3.FalseQCAndHarmonicContracts
open N2P3.QCIntervalContracts N2P3.HarmonicWeightedFit N2P3.HarmonicWindowSupport
open scoped BigOperators
noncomputable section

theorem false_equality_at_the_difference_threshold_is_rejected : differenceFlag 1 0 1 1 := by
  norm_num [differenceFlag]

theorem false_touching_half_open_supports_overlap : overlaps 0 5 5 8 := by
  norm_num [overlaps]

theorem false_a_zero_mixing_row_is_a_valid_observed_output :
    dependencyObserved (fun _ : Fin 1 => 0) (fun _ => True) := by
  simp [dependencyObserved]

theorem false_a_missing_nonzero_dependency_is_observed :
    dependencyObserved (fun _ : Fin 1 => 1) (fun _ => False) := by
  simp [dependencyObserved]

theorem false_cluster_stop_is_the_last_flag_sample : clusterStop 20 5 2=5 := by
  norm_num [clusterStop]

theorem false_lookahead_confirmation_is_always_inside_the_raw_anomaly_interval :
    confirmed 20 (clusterStop 20 5 2) 2 < clusterStop 20 5 2 := by
  norm_num [confirmed,clusterStop]

theorem false_a_finite_pre_guard_certifies_an_unknown_earlier_onset : guardedStart 0 40 5 ≤ 0 := by
  norm_num [guardedStart]

theorem false_a_positive_floor_Huber_update_has_zero_weight : huberWeight 4 1 1 2=0 := by
  norm_num [huberWeight]

theorem false_the_returned_updated_weights_generated_the_last_fit :
    weightedNormalResidual ![2,2,1] (fun _ : Fin 3 => 1) ![0,0,10] (10/3)=0 := by
  norm_num [weightedNormalResidual,Fin.sum_univ_succ]

theorem false_fitted_nuisance_is_subtracted_alongside_harmonics : cleanOnlyLine 10 2=5 := by
  norm_num [cleanOnlyLine]

theorem false_the_final_tail_start_is_absent_from_source_windows : 19 ∉ windowStarts 35 16 16 := by
  norm_num [windowStarts]

theorem false_including_a_Hann_endpoint_preserves_strict_positive_weights :
    0 < interiorHann 16 16 := by
  norm_num [interiorHann,Real.sin_pi]

end
end N2P3.FalseQCAndHarmonicContracts
