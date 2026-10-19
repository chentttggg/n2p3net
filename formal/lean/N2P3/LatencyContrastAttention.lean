import N2P3.LatencyWindowBank
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
Actual candidate/reference einsums, tanh query, temperature*sqrt(C) softmax
and weighted contrasts. All hypotheses concern individual source inputs and
local bank guards, never a desired final output or whole-forward success.
-/
namespace N2P3.LatencyContrastAttention
open N2P3.LatencyWindowBank
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def contrast {K : Type*} {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin C → Fin n → ℝ) (k : K) (c : Fin C) : ℝ :=
  mean (candidates k) (x c)-mean refSamples (x c)

def score {K : Type*} {C : ℕ} (contrasts : K → Fin C → ℝ) (query : Fin C → ℝ) (k : K) : ℝ :=
  ∑ c, contrasts k c*Real.tanh (query c)

def temperatureScale (temperature : ℝ) (channels : ℕ) : ℝ :=
  temperature*Real.sqrt (channels : ℝ)

def softmax {K : Type*} [Fintype K] (logits : K → ℝ) (k : K) : ℝ :=
  Real.exp (logits k)/(∑ j, Real.exp (logits j))

def attention {K : Type*} [Fintype K] {C : ℕ}
    (contrasts : K → Fin C → ℝ) (query : Fin C → ℝ) (temperature : ℝ) : K → ℝ :=
  softmax (fun k => score contrasts query k/temperatureScale temperature C)

def pooled {K : Type*} [Fintype K] {C : ℕ}
    (contrasts : K → Fin C → ℝ) (query : Fin C → ℝ) (temperature : ℝ) (c : Fin C) : ℝ :=
  ∑ k, attention contrasts query temperature k*contrasts k c

def forward {K : Type*} [Fintype K] {C n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n))
    (x : Fin C → Fin n → ℝ) (query : Fin C → ℝ) (temperature : ℝ) : Fin C → ℝ :=
  pooled (contrast refSamples candidates x) query temperature

theorem positive_temperature_and_channels_derive_the_actual_positive_scale
    (temperature : ℝ) (channels : ℕ) (positiveTemperature : 0 < temperature)
    (positiveChannels : 0 < channels) : 0 < temperatureScale temperature channels := by
  exact mul_pos positiveTemperature (Real.sqrt_pos.mpr (by exact_mod_cast positiveChannels))

theorem the_actual_tanh_query_is_strictly_bounded (query : ℝ) :
    -1 < Real.tanh query ∧ Real.tanh query < 1 := by
  have upper : ∀ x : ℝ, Real.tanh x < 1 := by
    intro x
    rw [Real.tanh_eq_sinh_div_cosh]
    exact (div_lt_one (Real.cosh_pos x)).mpr (Real.sinh_lt_cosh x)
  have lower := upper (-query)
  rw [Real.tanh_neg] at lower
  exact ⟨by linarith,upper query⟩

theorem the_softmax_denominator_is_derived_strictly_positive
    {K : Type*} [Fintype K] [Nonempty K] (logits : K → ℝ) :
    0 < ∑ k, Real.exp (logits k) := by
  exact Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty

theorem every_retained_latency_has_positive_attention
    {K : Type*} [Fintype K] [Nonempty K] (logits : K → ℝ) (k : K) :
    0 < softmax logits k :=
  div_pos (Real.exp_pos _) (the_softmax_denominator_is_derived_strictly_positive logits)

theorem actual_latency_attention_sums_to_one
    {K : Type*} [Fintype K] [Nonempty K] (logits : K → ℝ) :
    (∑ k, softmax logits k)=1 := by
  simp only [softmax,← Finset.sum_div]
  exact div_self (ne_of_gt (the_softmax_denominator_is_derived_strictly_positive logits))

theorem common_score_translation_preserves_all_attention
    {K : Type*} [Fintype K] (logits : K → ℝ) (base : ℝ) :
    softmax (fun k => logits k+base)=softmax logits := by
  funext k
  simp only [softmax,Real.exp_add,← Finset.sum_mul]
  exact mul_div_mul_right _ _ (ne_of_gt (Real.exp_pos base))

theorem score_order_is_attention_order
    {K : Type*} [Fintype K] [Nonempty K] (logits : K → ℝ) (first last : K)
    (ordered : logits first ≤ logits last) : softmax logits first ≤ softmax logits last := by
  exact div_le_div_of_nonneg_right (Real.exp_le_exp.mpr ordered)
    (le_of_lt (the_softmax_denominator_is_derived_strictly_positive logits))

theorem zero_query_initialization_gives_uniform_latency_attention
    {K : Type*} [Fintype K] {C : ℕ} (contrasts : K → Fin C → ℝ) (temperature : ℝ) (k : K) :
    attention contrasts (fun _ => 0) temperature k=1/(Fintype.card K : ℝ) := by
  simp [attention,score,softmax]

theorem zero_query_pools_the_arithmetic_average_of_contrasts
    {K : Type*} [Fintype K] {C : ℕ} (contrasts : K → Fin C → ℝ) (temperature : ℝ) (c : Fin C) :
    pooled contrasts (fun _ => 0) temperature c=(∑ k, contrasts k c)/(Fintype.card K : ℝ) := by
  simp only [pooled,zero_query_initialization_gives_uniform_latency_attention,one_div,
    ← Finset.mul_sum]
  ring

theorem pooled_contrast_stays_inside_independent_candidate_bounds
    {K : Type*} [Fintype K] [Nonempty K] {C : ℕ}
    (contrasts : K → Fin C → ℝ) (query : Fin C → ℝ) (temperature lower upper : ℝ) (c : Fin C)
    (bounds : ∀ k, lower ≤ contrasts k c ∧ contrasts k c ≤ upper) :
    lower ≤ pooled contrasts query temperature c ∧ pooled contrasts query temperature c ≤ upper := by
  have positive : ∀ k, 0 ≤ attention contrasts query temperature k :=
    fun k => (every_retained_latency_has_positive_attention _ k).le
  have mass : (∑ k, attention contrasts query temperature k)=1 := actual_latency_attention_sums_to_one _
  constructor
  · calc
      lower = ∑ k, attention contrasts query temperature k*lower := by rw [← Finset.sum_mul,mass,one_mul]
      _ ≤ _ := Finset.sum_le_sum (fun k _ => mul_le_mul_of_nonneg_left (bounds k).1 (positive k))
  · calc
      _ ≤ ∑ k, attention contrasts query temperature k*upper :=
        Finset.sum_le_sum (fun k _ => mul_le_mul_of_nonneg_left (bounds k).2 (positive k))
      _ = upper := by rw [← Finset.sum_mul,mass,one_mul]

theorem the_source_einsum_subtracts_the_same_reference_after_marginalizing
    {K : Type*} [Fintype K] [Nonempty K] {C n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n))
    (x : Fin C → Fin n → ℝ) (query : Fin C → ℝ) (temperature : ℝ) (c : Fin C) :
    forward refSamples candidates x query temperature c=
      (∑ k, attention (contrast refSamples candidates x) query temperature k*mean (candidates k) (x c))-
      mean refSamples (x c) := by
  unfold forward pooled
  simp only [contrast,mul_sub,Finset.sum_sub_distrib,← Finset.sum_mul]
  have mass : (∑ k, attention (contrast refSamples candidates x) query temperature k)=1 :=
    actual_latency_attention_sums_to_one _
  rw [mass,one_mul]

theorem time_constant_feature_baselines_cancel_before_the_score_and_pool
    {K : Type*} {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin C → Fin n → ℝ) (base : Fin C → ℝ)
    (refNonempty : refSamples.Nonempty) (candidateNonempty : ∀ k, (candidates k).Nonempty) :
    contrast refSamples candidates (fun c t => x c t+base c)=contrast refSamples candidates x := by
  funext k c
  simp only [contrast,adding_a_constant_adds_it_once_to_the_selected_mean _ refNonempty,
    adding_a_constant_adds_it_once_to_the_selected_mean _ (candidateNonempty k)]
  ring

theorem baseline_translation_preserves_attention_and_actual_forward
    {K : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin C → Fin n → ℝ) (base query : Fin C → ℝ)
    (temperature : ℝ) (refNonempty : refSamples.Nonempty) (candidateNonempty : ∀ k, (candidates k).Nonempty) :
    attention (contrast refSamples candidates (fun c t => x c t+base c)) query temperature=
      attention (contrast refSamples candidates x) query temperature ∧
    forward refSamples candidates (fun c t => x c t+base c) query temperature=
      forward refSamples candidates x query temperature := by
  have same := time_constant_feature_baselines_cancel_before_the_score_and_pool
    refSamples candidates x base refNonempty candidateNonempty
  simp only [forward,same,and_self]

theorem time_constant_encoded_features_give_zero_contrast_and_output
    {K : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (base query : Fin C → ℝ) (temperature : ℝ)
    (refNonempty : refSamples.Nonempty) (candidateNonempty : ∀ k, (candidates k).Nonempty) :
    contrast refSamples candidates (fun c _ => base c)=(fun _ _ => 0) ∧
    forward refSamples candidates (fun c _ => base c) query temperature=(fun _ => 0) := by
  have zeros : contrast refSamples candidates (fun c _ => base c)=fun _ _ => 0 := by
    funext k c
    simp only [contrast,a_constant_selected_window_has_that_same_constant_mean _ refNonempty,
      a_constant_selected_window_has_that_same_constant_mean _ (candidateNonempty k),sub_self]
  exact ⟨zeros,by funext c; simp [forward,zeros,pooled]⟩

def support {K : Type*} [Fintype K] {n : ℕ}
    (refSamples : Finset (Fin n)) (candidates : K → Finset (Fin n)) : Finset (Fin n) :=
  refSamples ∪ Finset.univ.biUnion candidates

theorem actual_contrasts_only_read_reference_and_retained_candidate_support
    {K : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x y : Fin C → Fin n → ℝ)
    (same : ∀ c t, t ∈ support refSamples candidates → x c t=y c t) :
    contrast refSamples candidates x=contrast refSamples candidates y := by
  funext k c
  unfold contrast
  congr 1
  · exact selected_window_mean_only_reads_its_own_samples _ _ _
      (fun t h => same c t (by simp [support]; exact Or.inr ⟨k,h⟩))
  · exact selected_window_mean_only_reads_its_own_samples _ _ _
      (fun t h => same c t (Finset.mem_union_left _ h))

theorem changing_only_unused_encoded_times_preserves_attention_and_output
    {K : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x y : Fin C → Fin n → ℝ)
    (query : Fin C → ℝ) (temperature : ℝ)
    (same : ∀ c t, t ∈ support refSamples candidates → x c t=y c t) :
    attention (contrast refSamples candidates x) query temperature=
      attention (contrast refSamples candidates y) query temperature ∧
    forward refSamples candidates x query temperature=forward refSamples candidates y query temperature := by
  have contrasts := actual_contrasts_only_read_reference_and_retained_candidate_support refSamples candidates x y same
  simp only [forward,contrasts,and_self]

def signedKernel {K : Type*} [Fintype K] {n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (alpha : K → ℝ) (t : Fin n) : ℝ :=
  ∑ k, alpha k*(weight (candidates k) t-weight refSamples t)

theorem the_actual_pool_has_the_derived_attention_conditional_signed_time_kernel
    {K : Type*} [Fintype K] {C n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (x : Fin C → Fin n → ℝ)
    (query : Fin C → ℝ) (temperature : ℝ) (c : Fin C) :
    forward refSamples candidates x query temperature c=
      ∑ t, signedKernel refSamples candidates (attention (contrast refSamples candidates x) query temperature) t*x c t := by
  unfold forward pooled contrast mean signedKernel
  simp only [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  simp only [Finset.mul_sum,← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro t _
  ring

theorem the_signed_time_kernel_has_zero_total_mass
    {K : Type*} [Fintype K] {n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (alpha : K → ℝ)
    (refNonempty : refSamples.Nonempty) (candidateNonempty : ∀ k, (candidates k).Nonempty) :
    (∑ t, signedKernel refSamples candidates alpha t)=0 := by
  unfold signedKernel
  rw [Finset.sum_comm]
  apply Finset.sum_eq_zero
  intro k _
  rw [← Finset.mul_sum,Finset.sum_sub_distrib,
    nonempty_selected_window_weights_sum_to_one _ (candidateNonempty k),
    nonempty_selected_window_weights_sum_to_one _ refNonempty,sub_self,mul_zero]

theorem the_signed_kernel_vanishes_outside_the_declared_encoded_support
    {K : Type*} [Fintype K] {n : ℕ} (refSamples : Finset (Fin n))
    (candidates : K → Finset (Fin n)) (alpha : K → ℝ) (t : Fin n)
    (outside : t ∉ support refSamples candidates) : signedKernel refSamples candidates alpha t=0 := by
  have absentRef : t ∉ refSamples := fun h => outside (Finset.mem_union_left _ h)
  have absentCandidate : ∀ k, t ∉ candidates k := by
    intro k h
    apply outside
    simp only [support,Finset.mem_union,Finset.mem_biUnion,Finset.mem_univ,true_and]
    exact Or.inr ⟨k,h⟩
  simp [signedKernel,weight,absentRef,absentCandidate]

end
end N2P3.LatencyContrastAttention
