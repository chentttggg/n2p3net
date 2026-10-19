import N2P3.LatencyContrastAttention

/-!
The actual ordered active bank is wired to per-MST-branch attention, channel
concatenation and the source affine2 head. This starts with post-MST/adapted
features. ST/MST/ELU/dropout, wrapper authority, BN state and the complete trunk
registry retain their separate original V07 obligations.
-/
namespace N2P3.LatencyReadoutComposition
open N2P3.LatencyWindowBank N2P3.LatencyContrastAttention N2P3.NetworkClassifierHeads
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def sourceCandidates (cfg : Config) (n : ℕ) (k : Fin (activeOffsets cfg n).length) : Finset (Fin n) :=
  candidate cfg n (activeOffsets cfg n)[k]

theorem every_computed_bank_row_has_the_required_two_encoded_samples
    (cfg : Config) (n : ℕ) (k : Fin (activeOffsets cfg n).length) :
    2 ≤ (sourceCandidates cfg n k).card := by
  exact ((actual_active_bank_filters_by_at_least_two_selected_samples cfg n _).mp
    (List.getElem_mem k.isLt)).2

theorem independent_bank_guards_derive_all_reference_and_candidate_normalizations
    (cfg : Config) (n : ℕ) (referenceEnough : 2 ≤ (reference cfg n).card) :
    (∑ t, weight (reference cfg n) t)=1 ∧
      ∀ k : Fin (activeOffsets cfg n).length, (∑ t, weight (sourceCandidates cfg n k) t)=1 := by
  constructor
  · apply nonempty_selected_window_weights_sum_to_one
    exact Finset.card_pos.mp (by omega)
  · intro k
    apply nonempty_selected_window_weights_sum_to_one
    have count := every_computed_bank_row_has_the_required_two_encoded_samples cfg n k
    exact Finset.card_pos.mp (by omega)

theorem nonempty_actual_bank_derives_unit_attention_without_forward_success
    (cfg : Config) (n C : ℕ) (x : Fin C → Fin n → ℝ) (query : Fin C → ℝ)
    (temperature : ℝ) (nonempty : activeOffsets cfg n ≠ []) :
    (∑ k : Fin (activeOffsets cfg n).length,
      attention (contrast (reference cfg n) (sourceCandidates cfg n) x) query temperature k)=1 := by
  have lengthPositive : 0 < (activeOffsets cfg n).length := List.length_pos_iff.mpr nonempty
  letI : Nonempty (Fin (activeOffsets cfg n).length) := Fin.pos_iff_nonempty.mp lengthPositive
  exact actual_latency_attention_sums_to_one _

def branches {K : Type*} [Fintype K] {S C n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n))
    (x : Fin S → Fin C → Fin n → ℝ) (queries : Fin S → Fin C → ℝ) (temperature : ℝ)
    (branch : Fin S) : Fin C → ℝ :=
  LatencyContrastAttention.forward refSamples candidates (x branch) (queries branch) temperature

def packed {K : Type*} [Fintype K] {S C n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n))
    (x : Fin S → Fin C → Fin n → ℝ) (queries : Fin S → Fin C → ℝ) (temperature : ℝ) :
    Fin (S*C) → ℝ := N2P3.NetworkReadoutGeometry.fullUnfold (branches refSamples candidates x queries temperature)

def logits {K : Type*} [Fintype K] {S C n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n))
    (x : Fin S → Fin C → Fin n → ℝ) (queries : Fin S → Fin C → ℝ) (temperature : ℝ)
    (weights : Matrix (Fin 2) (Fin (S*C)) ℝ) (bias : Fin 2 → ℝ) : Fin 2 → ℝ :=
  affine weights bias (packed refSamples candidates x queries temperature)

theorem branch_split_and_concatenation_recover_the_same_source_branch_channel
    {K : Type*} [Fintype K] {S C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin S → Fin C → Fin n → ℝ)
    (queries : Fin S → Fin C → ℝ) (temperature : ℝ) (branch : Fin S) (c : Fin C) :
    packed refSamples candidates x queries temperature (finProdFinEquiv (branch,c))=
      LatencyContrastAttention.forward refSamples candidates (x branch) (queries branch) temperature c := by
  simp [packed,N2P3.NetworkReadoutGeometry.fullUnfold,
    N2P3.MatrixNormalCoordinates.rowMajorVector_at,branches]

theorem each_actual_branch_uses_only_its_own_query_and_encoded_features
    {K : Type*} [Fintype K] {S C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x y : Fin S → Fin C → Fin n → ℝ)
    (queries otherQueries : Fin S → Fin C → ℝ) (temperature : ℝ) (branch : Fin S)
    (sameFeatures : x branch=y branch) (sameQuery : queries branch=otherQueries branch) :
    branches refSamples candidates x queries temperature branch=
      branches refSamples candidates y otherQueries temperature branch := by
  simp only [branches,sameFeatures,sameQuery]

theorem actual_affine_logits_marginalize_each_branch_with_its_own_attention
    {K : Type*} [Fintype K] {S C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin S → Fin C → Fin n → ℝ)
    (queries : Fin S → Fin C → ℝ) (temperature : ℝ)
    (weights : Matrix (Fin 2) (Fin (S*C)) ℝ) (bias : Fin 2 → ℝ) (output : Fin 2) :
    logits refSamples candidates x queries temperature weights bias output=
      (∑ branch, ∑ c, weights output (finProdFinEquiv (branch,c))*
        (∑ k, attention (contrast refSamples candidates (x branch)) (queries branch) temperature k*
          contrast refSamples candidates (x branch) k c))+bias output := by
  exact full_resolution_affine_readout_uses_each_encoded_channel_time_coordinate weights bias
    (branches refSamples candidates x queries temperature) output

theorem actual_logits_have_the_derived_branch_channel_and_signed_time_expansion
    {K : Type*} [Fintype K] {S C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin S → Fin C → Fin n → ℝ)
    (queries : Fin S → Fin C → ℝ) (temperature : ℝ)
    (weights : Matrix (Fin 2) (Fin (S*C)) ℝ) (bias : Fin 2 → ℝ) (output : Fin 2) :
    logits refSamples candidates x queries temperature weights bias output=
      (∑ branch, ∑ c, ∑ t, weights output (finProdFinEquiv (branch,c))*
        signedKernel refSamples candidates
          (attention (contrast refSamples candidates (x branch)) (queries branch) temperature) t*x branch c t)+bias output := by
  unfold logits packed
  rw [full_resolution_affine_readout_uses_each_encoded_channel_time_coordinate]
  congr 1
  apply Finset.sum_congr rfl
  intro branch _
  apply Finset.sum_congr rfl
  intro c _
  rw [branches,the_actual_pool_has_the_derived_attention_conditional_signed_time_kernel,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro t _
  ring

theorem binary_logit_contrast_keeps_the_actual_weight_difference_and_both_biases
    {K : Type*} [Fintype K] {S C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin S → Fin C → Fin n → ℝ)
    (queries : Fin S → Fin C → ℝ) (temperature : ℝ)
    (weights : Matrix (Fin 2) (Fin (S*C)) ℝ) (bias : Fin 2 → ℝ) :
    logits refSamples candidates x queries temperature weights bias 1-
      logits refSamples candidates x queries temperature weights bias 0=
      (∑ feature, (weights 1 feature-weights 0 feature)*
        packed refSamples candidates x queries temperature feature)+(bias 1-bias 0) := by
  simp only [logits,affine,sub_mul,Finset.sum_sub_distrib]
  ring

theorem same_encoded_batch_row_gives_the_same_pool_without_cross_batch_reduction
    {K B : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x y : B → Fin C → Fin n → ℝ)
    (query : Fin C → ℝ) (temperature : ℝ) (batch : B) (same : x batch=y batch) :
    LatencyContrastAttention.forward refSamples candidates (x batch) query temperature=
      LatencyContrastAttention.forward refSamples candidates (y batch) query temperature := by rw [same]

def returned {K : Type*} [Fintype K] {C n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n))
    (x : Fin C → Fin n → ℝ) (query : Fin C → ℝ) (temperature : ℝ) :
    (Fin C → ℝ) × (K → ℝ) :=
  ⟨LatencyContrastAttention.forward refSamples candidates x query temperature,
    attention (contrast refSamples candidates x) query temperature⟩

theorem return_attention_preserves_the_identical_pooled_feature_result
    {K : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin C → Fin n → ℝ)
    (query : Fin C → ℝ) (temperature : ℝ) :
    (returned refSamples candidates x query temperature).1=
      LatencyContrastAttention.forward refSamples candidates x query temperature := rfl

end
end N2P3.LatencyReadoutComposition
