import N2P3.SplitBNStatistics

/- Source-fixed alternatives from contract03. Effective and configured
weights are kept distinct at the explicit Tensor.new_tensor operation. -/
namespace N2P3.NegativeSplitBNStatistics
open N2P3.SplitBNStatistics

private theorem ok_ne_error {ε α : Type} (value : α) (error : ε) :
    (Except.ok value : Except ε α) ≠ .error error := by intro equal; cases equal

private theorem error_ne_ok {ε α : Type} (error : ε) (value : α) :
    (Except.error error : Except ε α) ≠ .ok value := by intro equal; cases equal

example : variance equalWeights separatedMeans zeroVariances = 0 := by
  rw [between_domain_means_contribute_the_missing_variance.2.1]
  norm_num

example : mass nearWeights = 1 := by
  have wrong := near_unit_effective_real_weights_need_not_have_exact_unit_mass.2
  simp only [wrong]

example : 0 ≤ rawVariance nearWeights constantMeans zeroVariances := by
  have wrong : ¬0 ≤ rawVariance nearWeights constantMeans zeroVariances :=
    not_le.mpr near_unit_effective_mass_can_give_negative_unclipped_variance.1
  simp only [wrong]

example : mean nearWeights constantMeans = 1 := by
  have wrong := source_does_not_silently_renormalize_effective_tensor_weights.2
  simp only [wrong]

example : weightedSource (some singleActiveWeight) oneInitializedBuffer = .error .uninitializedActiveDomain := by
  rw [zero_weight_domain_does_not_require_source_initialization]
  simp only [ok_ne_error]

example : weightedSourceWithCast (fun weights => .ok weights) none oneInitializedBuffer = .ok (1, 0) := by
  rw [missing_configured_weights_do_not_invoke_the_tensor_cast]
  simp only [error_ne_ok]

example : weightedSourceWithCast (fun _ => .error "declared tensor cast failure")
    (some singleActiveWeight) (fun _ => ⟨1, 0, 0⟩) = .error .uninitializedActiveDomain := by
  rw [tensor_cast_failure_precedes_every_active_domain_guard _ _ _ _ rfl]
  simp

example : weightedSource (some singleActiveWeight) negativeTrackedBuffer = .error .uninitializedActiveDomain := by
  rw [a_negative_nonzero_count_passes_this_literal_source_guard]
  simp only [ok_ne_error]

end N2P3.NegativeSplitBNStatistics
