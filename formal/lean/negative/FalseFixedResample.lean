import N2P3.FixedResampleWitnesses

namespace N2P3.NegativeFixedResample

open N2P3.FixedResampleMean N2P3.FixedResampleBatches N2P3.LinearQuantile
open N2P3.FixedResampleWitnesses

-- R01: deduplicating original unit indices changes the actual mean.
example : drawMean units3 repeatedIndices=3/2 := by
  rw [repeated_draw_positions_retain_their_multiplicity.1]
  norm_num

-- R02: paired_interval's final batch can be shorter than its 4000 cap.
example : takeCount 4001 4000 4000=4000 := by
  rw [paired_batch_tail_is_one_position]
  norm_num

-- R03: _summary's per-study final additive batch can be shorter than1000.
example : takeCount 1001 1000 1000=1000 := by
  rw [study_batch_tail_is_one_position]
  norm_num

-- R04: pooling unequal study sizes is not the equal-study draw.
example : studyDrawMean (Finset.univ : Finset (Fin 2)) unequalStudySizes unequalStudyUnits (fun _ => id)=1/4 := by
  rw [singleton_study_keeps_equal_study_weight_in_every_draw.1]
  norm_num

-- R05: swapping the low endpoint with the high sorted linear quantile.
example : quantile ordered4 (by omega) (1/40) (by norm_num)=237/20 := by
  rw [sorted_linear_quantile_fixture_has_the_independently_required_endpoints.1]
  norm_num

-- R06: interpolating raw input positions before sorting changes the answer.
example : quantile ordered4 (by omega) (1/40) (by norm_num)=111/10 := by
  rw [sorted_linear_quantile_fixture_has_the_independently_required_endpoints.1]
  norm_num

end N2P3.NegativeFixedResample
