import N2P3.NetworkClassifierHeads

namespace N2P3.NegativeNetworkReadoutHeads
open N2P3.NetworkReadoutGeometry N2P3.NetworkClassifierHeads

-- G01: Pm pooling does not occur inside every MST branch.
example : headWidth .fullUnfold (2 * 2) (128 / 4) 8 = 16 := by
  rw [fixed_source_default_readout_widths_separate_full_MS_and_global.1]
  norm_num

-- G02: Source complete pooling uses floor, not ceil.
example : pooledTime 113 4 = .ok 29 := by
  rw [incomplete_source_windows_round_down_and_MS_rejects_the_short_case.1]
  simp only [Except.ok.injEq]
  norm_num

-- G03: Channel-major/time-fast flatten differs from transposed flatten.
example : fullUnfold orderedFeatures 1 = 3 := by
  rw [nonempty_feature_order_is_channel_first_and_time_fast.1]
  norm_num

-- G04: Moving within a complete average window can collide.
example : msFlatten 2 leftImpulse ≠ msFlatten 2 rightImpulse := by
  have impossible : ¬ msFlatten 2 leftImpulse ≠ msFlatten 2 rightImpulse :=
    fun different => different complete_average_pool_collides_while_direct_unfold_retains_the_difference.1
  simp only [impossible]

-- H01: The zero-initialized residual output preserves a nonzero affine base.
example : quadraticHead nonzeroLinear (fun _ => 0) (fun _ : Fin 1 => fun _ => 1)
    (fun _ => fun _ => 1) (fun _ _ => 0) (fun _ => 3) 0 = 0 := by
  rw [(actual_zero_residual_initializations_do_not_make_the_classifier_zero (fun value => value)).1]
  norm_num

-- H02: Matching the P-dependent slope leaves32 allocated cells at defaults.
example : allocatedQuadratic 128 2 8 = allocatedMLP 128 2 16 := by
  rw [source_default_readout_parameter_counts_are_not_exactly_matched.1,
    source_default_readout_parameter_counts_are_not_exactly_matched.2.1]
  norm_num

-- H03: Sixteen trainable residual-output cells still exist when their values0.
example : activeCells (fun _ : Fin 2 × Fin 8 => true) = 0 := by
  rw [source_default_readout_parameter_counts_are_not_exactly_matched.2.2.1]
  norm_num

-- H04: The actual requires_grad filter can report0 with arrays still present.
example : activeCells (fun _ : LinearCells 128 2 => false) = allocatedLinear 128 2 := by
  rw [source_default_readout_parameter_counts_are_not_exactly_matched.2.2.2]
  norm_num [allocatedLinear, LinearCells]

end N2P3.NegativeNetworkReadoutHeads
