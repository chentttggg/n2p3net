import N2P3.SplitBNMomentum

namespace N2P3.NegativeSplitBNMomentum
open N2P3.SplitBNMomentum

example : (factor (some 0) 0).2 = 0 := by
  rw [every_factor_call_increments_before_its_outcome]
  norm_num

example : (factor none (-1)).2 = -1 := by
  rw [every_factor_call_increments_before_its_outcome]
  norm_num

example : (factor none (-2)).1 = .ok 1 := by
  rw [a_negative_counter_can_produce_a_negative_reciprocal_factor]
  norm_num

example : average (1 / 2) 0 10 = 9 := by
  rw [equal_batch_average_is_not_a_pooled_observation_average.1]
  norm_num

end N2P3.NegativeSplitBNMomentum
