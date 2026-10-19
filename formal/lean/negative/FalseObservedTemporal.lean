import N2P3.ObservedTemporal

/- The four source-derived alternatives were fixed before positive proofs. -/
namespace N2P3.NegativeObservedTemporal
open N2P3.ObservedTemporal

example : traceConvolution asymmetricKernel 1 1 asymmetricTrace 0 1 = 4 := by
  rw [source_temporal_operator_is_cross_correlation_without_kernel_reversal]
  norm_num

example : trainingMean measuredMask (packedConvolution identityKernel 1 1 measuredMask measuredRaw) 0 = 3 / 2 := by
  rw [actual_identity_convolution_and_packed_population_have_mean_three]
  norm_num

example : restoredTraining identityKernel 1 1 (fun _ => 1) (fun _ => 0) (1 / 100000)
    measuredMask measuredRaw 0 0 1 0 = bnCell 1 0 3 1 (1 / 100000) 0 := by
  unfold restoredTraining
  rw [missing_scatter_coordinate_is_zero_even_after_BN _ _ _ _ _ _ (by decide)]
  have wrong : ¬((0 : ℝ) = bnCell 1 0 3 1 (1 / 100000) 0) :=
    Ne.symm zero_input_encoded_by_BN_can_be_nonzero_before_scatter
  simp only [wrong]

example : bnCell 1 0 0 0 (1 / 4) 1 = 4 := by
  rw [BN_epsilon_is_inside_the_square_root.1]
  norm_num

end N2P3.NegativeObservedTemporal
