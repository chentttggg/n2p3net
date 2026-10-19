import N2P3.NetworkClassifierHeads
import Mathlib.Data.List.Nodup
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum

/-!
Actual LatencyMarginalContrastPool grid/mask/cache program, from the fixed
v07_latency_marginal_contract_01.json. Exact finite-real operation interface;
float32/device/dtype realization is a separate numerical refinement. Windows
are half-open and need not be disjoint, pre-stimulus, or fully inside the epoch.
-/
namespace N2P3.LatencyWindowBank
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

structure Config where
  sfreq : ℝ
  tmin : ℝ
  referenceStart : ℝ
  referenceEnd : ℝ
  evidenceStart : ℝ
  evidenceEnd : ℝ
  offsets : List ℝ

def timeMs (cfg : Config) (sample : ℕ) : ℝ :=
  cfg.tmin*1000+(sample : ℝ)*(1000/cfg.sfreq)

def window (cfg : Config) (n : ℕ) (first last : ℝ) : Finset (Fin n) :=
  Finset.univ.filter (fun sample => first ≤ timeMs cfg sample.val ∧ timeMs cfg sample.val < last)

def reference (cfg : Config) (n : ℕ) : Finset (Fin n) :=
  window cfg n cfg.referenceStart cfg.referenceEnd

def candidate (cfg : Config) (n : ℕ) (offset : ℝ) : Finset (Fin n) :=
  window cfg n (cfg.evidenceStart+offset) (cfg.evidenceEnd+offset)

def activeOffsets (cfg : Config) (n : ℕ) : List ℝ :=
  cfg.offsets.filter (fun offset => decide (2 ≤ (candidate cfg n offset).card))

structure Bank (n : ℕ) where
  offsets : List ℝ
  referenceSamples : Finset (Fin n)
  candidateSamples : ℝ → Finset (Fin n)

def computedBank (cfg : Config) (n : ℕ) : Bank n :=
  ⟨activeOffsets cfg n,reference cfg n,candidate cfg n⟩

def build (cfg : Config) (n : ℕ) : Option (Bank n) :=
  if 2 ≤ (reference cfg n).card ∧ activeOffsets cfg n ≠ [] then some (computedBank cfg n) else none

def weightsFor (cachedLength : Option ℕ) (cfg : Config) (actualLength : ℕ) : Option (Bank actualLength) :=
  match cachedLength with
  | none => build cfg actualLength
  | some expected => if expected=actualLength then build cfg actualLength else none

def weight {n : ℕ} (samples : Finset (Fin n)) (t : Fin n) : ℝ :=
  if t ∈ samples then 1/(samples.card : ℝ) else 0

def mean {n : ℕ} (samples : Finset (Fin n)) (x : Fin n → ℝ) : ℝ :=
  ∑ t, weight samples t*x t

theorem actual_grid_is_strictly_ordered (cfg : Config) (positive : 0 < cfg.sfreq)
    (i j : ℕ) (before : i < j) : timeMs cfg i < timeMs cfg j := by
  have step : 0 < 1000/cfg.sfreq := div_pos (by norm_num) positive
  exact add_lt_add_left (mul_lt_mul_of_pos_right (by exact_mod_cast before) step) _

theorem actual_half_open_window_has_the_literal_endpoint_test
    (cfg : Config) (n : ℕ) (first last : ℝ) (t : Fin n) :
    t ∈ window cfg n first last ↔ first ≤ timeMs cfg t.val ∧ timeMs cfg t.val < last := by
  simp [window]

theorem upper_endpoint_is_excluded_even_when_it_is_a_grid_sample
    (cfg : Config) (n : ℕ) (first last : ℝ) (t : Fin n) (endpoint : timeMs cfg t.val=last) :
    t ∉ window cfg n first last := by simp [window,endpoint]

theorem actual_active_bank_filters_by_at_least_two_selected_samples
    (cfg : Config) (n : ℕ) (offset : ℝ) :
    offset ∈ activeOffsets cfg n ↔ offset ∈ cfg.offsets ∧ 2 ≤ (candidate cfg n offset).card := by
  simp [activeOffsets]

theorem active_bank_preserves_the_supplied_offset_order (cfg : Config) (n : ℕ) :
    (activeOffsets cfg n).Sublist cfg.offsets := List.filter_sublist

theorem unique_supplied_offsets_stay_unique (cfg : Config) (n : ℕ) (unique : cfg.offsets.Nodup) :
    (activeOffsets cfg n).Nodup := unique.filter _

theorem one_sample_candidate_is_not_retained
    (cfg : Config) (n : ℕ) (offset : ℝ) (one : (candidate cfg n offset).card=1) :
    offset ∉ activeOffsets cfg n := by simp [activeOffsets,one]

theorem independent_local_reference_and_bank_guards_construct_the_actual_weights
    (cfg : Config) (n : ℕ) (referenceEnough : 2 ≤ (reference cfg n).card)
    (bankNonempty : activeOffsets cfg n ≠ []) :
    build cfg n=some (computedBank cfg n) := by simp [build,referenceEnough,bankNonempty]

theorem too_short_reference_refuses_the_bank (cfg : Config) (n : ℕ)
    (short : (reference cfg n).card < 2) : build cfg n=none := by
  simp [build,Nat.not_le.mpr short]

theorem no_active_candidate_refuses_the_bank (cfg : Config) (n : ℕ)
    (empty : activeOffsets cfg n=[]) : build cfg n=none := by simp [build,empty]

theorem cached_and_dynamic_weights_agree_at_the_same_declared_length (cfg : Config) (n : ℕ) :
    weightsFor (some n) cfg n=weightsFor none cfg n := by simp [weightsFor]

theorem cached_length_mismatch_refuses_before_pooling (cfg : Config) (expected actual : ℕ)
    (different : expected ≠ actual) : weightsFor (some expected) cfg actual=none := by
  simp [weightsFor,different]

theorem dynamic_weights_are_built_from_the_actual_encoded_length (cfg : Config) (actual : ℕ) :
    weightsFor none cfg actual=build cfg actual := rfl

theorem actual_weight_is_nonnegative {n : ℕ} (samples : Finset (Fin n)) (t : Fin n) :
    0 ≤ weight samples t := by unfold weight; split_ifs <;> positivity

theorem actual_weight_vanishes_outside_the_selected_window {n : ℕ}
    (samples : Finset (Fin n)) (t : Fin n) (outside : t ∉ samples) : weight samples t=0 := by
  simp [weight,outside]

theorem nonempty_selected_window_weights_sum_to_one {n : ℕ}
    (samples : Finset (Fin n)) (nonempty : samples.Nonempty) : (∑ t, weight samples t)=1 := by
  have count : (samples.card : ℝ) ≠ 0 := by exact_mod_cast Finset.card_ne_zero.mpr nonempty
  simp [weight,count]

theorem actual_selected_window_mean_is_the_selected_sum_over_its_own_count {n : ℕ}
    (samples : Finset (Fin n)) (x : Fin n → ℝ) :
    mean samples x=(∑ t ∈ samples, x t)/(samples.card : ℝ) := by
  simp only [mean,weight,ite_mul,zero_mul,one_div,← Finset.mul_sum,Finset.sum_ite]
  simp [div_eq_mul_inv,mul_comm]

theorem selected_window_mean_only_reads_its_own_samples {n : ℕ}
    (samples : Finset (Fin n)) (x y : Fin n → ℝ) (same : ∀ t ∈ samples, x t=y t) :
    mean samples x=mean samples y := by
  unfold mean
  apply Finset.sum_congr rfl
  intro t _
  by_cases selected : t ∈ samples
  · rw [same t selected]
  · simp [weight,selected]

theorem adding_a_constant_adds_it_once_to_the_selected_mean {n : ℕ}
    (samples : Finset (Fin n)) (nonempty : samples.Nonempty) (x : Fin n → ℝ) (base : ℝ) :
    mean samples (fun t => x t+base)=mean samples x+base := by
  simp only [mean,mul_add,Finset.sum_add_distrib,← Finset.sum_mul]
  rw [nonempty_selected_window_weights_sum_to_one samples nonempty,one_mul]

theorem a_constant_selected_window_has_that_same_constant_mean {n : ℕ}
    (samples : Finset (Fin n)) (nonempty : samples.Nonempty) (base : ℝ) :
    mean samples (fun _ => base)=base := by
  simp only [mean,← Finset.sum_mul]
  rw [nonempty_selected_window_weights_sum_to_one samples nonempty,one_mul]

theorem sum_of_finite_raw_bin_indices (n : ℕ) :
    (∑ i : Fin n, (i.val : ℝ))=(n : ℝ)*((n : ℝ)-1)/2 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Fin.sum_univ_succ]
    simp only [Fin.val_zero,Nat.cast_zero,Fin.val_succ,Nat.cast_add,Nat.cast_one,zero_add,
      Finset.sum_add_distrib,Finset.sum_const,Finset.card_univ,Fintype.card_fin,nsmul_eq_mul,mul_one]
    rw [ih]
    ring

def featureConfig (cfg : Config) (stride : ℕ) : Config :=
  { cfg with sfreq := cfg.sfreq/(stride : ℝ), tmin := cfg.tmin+((stride : ℝ)-1)/(2*cfg.sfreq) }

theorem actual_ST_feature_time_is_the_complete_raw_bin_average
    (cfg : Config) (stride bin : ℕ) (positiveStride : 0 < stride) (positiveRate : 0 < cfg.sfreq) :
    timeMs (featureConfig cfg stride) bin=
      (∑ i : Fin stride, timeMs cfg (bin*stride+i.val))/(stride : ℝ) := by
  have strideNe : (stride : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt positiveStride
  have rateNe : cfg.sfreq ≠ 0 := ne_of_gt positiveRate
  simp only [timeMs,featureConfig,Nat.cast_add,Nat.cast_mul,add_mul,
    Finset.sum_add_distrib,Finset.sum_const,Finset.card_univ,Fintype.card_fin,nsmul_eq_mul,
    ← Finset.sum_mul,sum_of_finite_raw_bin_indices]
  field_simp
  ring

end
end N2P3.LatencyWindowBank
