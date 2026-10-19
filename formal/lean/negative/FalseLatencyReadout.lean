import N2P3.LatencyReadoutWitnesses

namespace N2P3.NegativeLatencyReadout
open N2P3.LatencyWindowBank N2P3.LatencyContrastAttention N2P3.LatencyReadoutComposition
open N2P3.LatencyReadoutWitnesses
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

-- L01: first sample is substituted for the pooled bin center.
example : timeMs (featureConfig rawConfig 4) 0= -200 := by
  rw [first_ST_bin_is_timestamped_at_its_center_instead_of_its_first_sample.1]
  norm_num

-- L02: the half-open upper endpoint is silently included.
example : (2 : Fin 3) ∈ window miniConfig 3 0 2 := by
  simp [the_literal_half_open_window_excludes_its_upper_sample.2]

-- L03: a one-sample candidate is silently retained.
example : activeOffsets miniConfig 3=[0,1,2] := by
  rw [the_active_bank_keeps_available_two_sample_windows_and_refuses_a_one_sample_reference.1]
  norm_num

-- L04: the shifted candidate is claimed to remain inside the base window.
example : (2 : Fin 3) ∈ candidate miniConfig 3 1 → (2 : Fin 3) ∈ candidate miniConfig 3 0 := by
  simp [a_valid_shifted_candidate_can_include_a_sample_outside_the_base_window.1,
    a_valid_shifted_candidate_can_include_a_sample_outside_the_base_window.2]

-- L05: a different length is accepted by the configured cache.
example : weightsFor (some 3) miniConfig 4 ≠ none := by
  rw [cached_shape_mismatch_does_not_silently_rebuild_for_another_length.1]
  simp

-- L06: raw evidence mean12 is returned instead of contrast10.
example : LatencyContrastAttention.forward ref4 candidates4 signal4 (fun _ => 0) 1 0=12 := by
  rw [the_actual_single_candidate_output_is_evidence_mean_minus_reference_mean.2.2]
  norm_num

-- L07: zero query selects the first candidate instead of marginalizing uniformly.
example : pooled twoContrasts (fun _ => 0) 1 0=2 := by
  rw [zero_query_initialization_averages_distinct_latency_contrasts.2.2]
  norm_num

-- L08: C is substituted for sqrt(C) in the temperature scale.
example : temperatureScale 2 4=8 := by
  rw [actual_temperature_scale_uses_the_square_root_of_channel_count]
  norm_num

-- L09: reference-only changes are incorrectly outside the pooling support.
example : LatencyContrastAttention.forward ref4 candidates4 changedReferenceSignal (fun _ => 0) 1 0=
    LatencyContrastAttention.forward ref4 candidates4 signal4 (fun _ => 0) 1 0 := by
  rw [reference_only_edits_change_the_actual_pool_even_with_unchanged_evidence.1,
    reference_only_edits_change_the_actual_pool_even_with_unchanged_evidence.2]
  norm_num

-- L10: channel-major versus branch-major concatenation is silently exchanged.
example : packed ref4 candidates4 orderedBranchSignals (fun _ _ => 0) 1 1=30 := by
  rw [actual_branch_major_channel_concatenation_keeps_different_coordinates.1]
  norm_num

end
end N2P3.NegativeLatencyReadout
