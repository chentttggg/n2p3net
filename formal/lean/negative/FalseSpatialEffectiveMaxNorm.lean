import N2P3.SpatialEffectiveMaxNorm

namespace N2P3.NegativeSpatialEffectiveMaxNorm
open N2P3.SpatialEffectiveMaxNorm

example : rowNorm ![3, 4] ≤ 1 := by
  rw [fixed_three_four_raw_row_has_norm_five]
  norm_num

example : effective 1 (1 / 1000000000000) ![3, 4] 0 = 1 := by
  rw [fixed_three_four_source_uses_row_scaling]
  norm_num

example (epsilon : ℝ) (positive : 0 < epsilon) :
    effective (epsilon / 10) epsilon (fun _ : Fin 1 => epsilon / 100) 0 = epsilon / 100 := by
  rw [(tiny_floor_fixture_shrinks_a_row_already_inside_the_ball epsilon positive).2]
  have different : epsilon / 1000 ≠ epsilon / 100 := by intro equal; linarith
  simp only [different]

example (epsilon : ℝ) (positive : 0 < epsilon) :
    effective (epsilon / 10) epsilon
        (effective (epsilon / 10) epsilon (fun _ : Fin 1 => epsilon / 100)) =
      effective (epsilon / 10) epsilon (fun _ : Fin 1 => epsilon / 100) := by
  have different := tiny_floor_fixture_is_not_idempotent epsilon positive
  simp only [different]

end N2P3.NegativeSpatialEffectiveMaxNorm
