import N2P3.ObservedStatistics
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

/-!
# Observed temporal convolution, training normalization and scatter

Source: n2p3net.py:689-708,1130-1164,1182-1202. This is the branch
observed_channel_norm=True. The default architecture's False branch has a
different BN population, explicitly distinguished by the controls below.
The source-fixed contract is v07_observed_temporal_contract_01.json.

Equations for temporal cross-correlation and BN are explicit. Native tensor
indexing, reductions, CUDA/IEEE and running-buffer updates remain separate.
The training population here is ordinary BN, or one already selected domain;
SplitBN's per-domain selection and state machine are separate obligations.
Numerical normalization describes its population>1 native-success domain;
these total mathematical equations do not certify native training admission.
These equations do not assume that an entire encoder or forward pass succeeds.
-/
namespace N2P3.ObservedTemporal
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable

variable {B C T F K : ℕ}

def Slot (mask : Fin B → Fin C → Bool) :=
  {coordinate : Fin B × Fin C // mask coordinate.1 coordinate.2 = true}

instance slotFintype (mask : Fin B → Fin C → Bool) : Fintype (Slot mask) := by
  unfold Slot
  exact Fintype.ofFinite _

def pack (mask : Fin B → Fin C → Bool) (values : Fin B → Fin C → Fin T → ℝ)
    (slot : Slot mask) (time : Fin T) : ℝ := values slot.1.1 slot.1.2 time

def packDomain {Domain : Type} (mask : Fin B → Fin C → Bool) (domains : Fin B → Domain)
    (slot : Slot mask) : Domain := domains slot.1.1

def zeroExtend (trace : Fin T → ℝ) (index : ℤ) : ℝ :=
  if inside : index.toNat < T ∧ 0 ≤ index then trace ⟨index.toNat, inside.1⟩ else 0

def sourcePadding (dilation kernel : ℕ) : ℕ := dilation * (kernel - 1) / 2

def traceConvolution (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ)
    (trace : Fin T → ℝ) (feature : Fin F) (time : Fin T) : ℝ :=
  ∑ lag, weights feature lag * zeroExtend trace
    ((time.val : ℤ) + (dilation : ℤ) * lag.val - padding)

def packedConvolution (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ)
    (mask : Fin B → Fin C → Bool) (values : Fin B → Fin C → Fin T → ℝ)
    (slot : Slot mask) (feature : Fin F) (time : Fin T) : ℝ :=
  traceConvolution weights dilation padding (pack mask values slot) feature time

def trainingMean (mask : Fin B → Fin C → Bool)
    (encoded : Slot mask → Fin F → Fin T → ℝ) (feature : Fin F) : ℝ :=
  ObservedStatistics.maskedMean (fun _ : Slot mask × Fin T => true)
    (fun coordinate => encoded coordinate.1 feature coordinate.2)

def trainingVariance (mask : Fin B → Fin C → Bool)
    (encoded : Slot mask → Fin F → Fin T → ℝ) (feature : Fin F) : ℝ :=
  ObservedStatistics.maskedVariance (fun _ : Slot mask × Fin T => true)
    (fun coordinate => encoded coordinate.1 feature coordinate.2)

def bnCell (gamma beta mean variance epsilon value : ℝ) : ℝ :=
  gamma * ((value - mean) / Real.sqrt (variance + epsilon)) + beta

def trainingEncoded (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ)
    (gamma beta : Fin F → ℝ) (epsilon : ℝ)
    (mask : Fin B → Fin C → Bool) (values : Fin B → Fin C → Fin T → ℝ)
    (slot : Slot mask) (feature : Fin F) (time : Fin T) : ℝ :=
  let convolved := packedConvolution weights dilation padding mask values
  bnCell (gamma feature) (beta feature) (trainingMean mask convolved feature)
    (trainingVariance mask convolved feature) epsilon (convolved slot feature time)

/-- The coordinate order is exactly the returned permute(B,F,C,T). -/
def restore (mask : Fin B → Fin C → Bool) (encoded : Slot mask → Fin F → Fin T → ℝ)
    (batch : Fin B) (feature : Fin F) (channel : Fin C) (time : Fin T) : ℝ :=
  if measured : mask batch channel = true then encoded ⟨(batch, channel), measured⟩ feature time else 0

def restoredTraining (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ)
    (gamma beta : Fin F → ℝ) (epsilon : ℝ)
    (mask : Fin B → Fin C → Bool) (values : Fin B → Fin C → Fin T → ℝ) :
    Fin B → Fin F → Fin C → Fin T → ℝ :=
  restore mask (trainingEncoded weights dilation padding gamma beta epsilon mask values)

def groupedSpatial {M : ℕ} (weights : Fin F → Fin M → Fin C → ℝ)
    (restored : Fin B → Fin F → Fin C → Fin T → ℝ)
    (batch : Fin B) (feature : Fin F) (depth : Fin M) (time : Fin T) : ℝ :=
  ∑ channel, weights feature depth channel * restored batch feature channel time

theorem zero_extension_reads_the_original_in_range_coordinate (trace : Fin T → ℝ) (time : Fin T) :
    zeroExtend trace time.val = trace time := by simp [zeroExtend, time.isLt]

theorem negative_indices_are_padding_zero (trace : Fin T → ℝ) (index : ℤ) (negative : index < 0) :
    zeroExtend trace index = 0 := by
  have outside : ¬(index.toNat < T ∧ 0 ≤ index) := by omega
  simp [zeroExtend, outside]

theorem odd_source_kernel_has_exact_symmetric_padding (dilation kernel : ℕ) (odd : kernel % 2 = 1) :
    2 * sourcePadding dilation kernel = dilation * (kernel - 1) := by
  have width : kernel - 1 = 2 * (kernel / 2) := by omega
  have products : dilation * (kernel - 1) = 2 * (dilation * (kernel / 2)) := by rw [width]; ring
  simp [sourcePadding, products]

theorem source_stride_one_convolution_keeps_time_length (times dilation kernel : ℕ)
    (odd : kernel % 2 = 1) :
    times + 2 * sourcePadding dilation kernel - dilation * (kernel - 1) = times := by
  rw [odd_source_kernel_has_exact_symmetric_padding dilation kernel odd]
  omega

theorem packed_domain_is_the_original_batch_domain {Domain : Type}
    (mask : Fin B → Fin C → Bool) (domains : Fin B → Domain) (slot : Slot mask) :
    packDomain mask domains slot = domains slot.1.1 := rfl

theorem changing_only_missing_values_preserves_every_packed_trace
    (mask : Fin B → Fin C → Bool) (first second : Fin B → Fin C → Fin T → ℝ)
    (same : ∀ batch channel, mask batch channel = true → ∀ time, first batch channel time = second batch channel time) :
    pack mask first = pack mask second := by
  funext slot time
  exact same slot.1.1 slot.1.2 slot.2 time

theorem changing_only_missing_values_preserves_the_actual_temporal_sum
    (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ)
    (mask : Fin B → Fin C → Bool) (first second : Fin B → Fin C → Fin T → ℝ)
    (same : ∀ batch channel, mask batch channel = true → ∀ time, first batch channel time = second batch channel time) :
    packedConvolution weights dilation padding mask first = packedConvolution weights dilation padding mask second := by
  unfold packedConvolution
  rw [changing_only_missing_values_preserves_every_packed_trace mask first second same]

theorem changing_only_missing_values_preserves_training_BN
    (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ) (gamma beta : Fin F → ℝ) (epsilon : ℝ)
    (mask : Fin B → Fin C → Bool) (first second : Fin B → Fin C → Fin T → ℝ)
    (same : ∀ batch channel, mask batch channel = true → ∀ time, first batch channel time = second batch channel time) :
    trainingEncoded weights dilation padding gamma beta epsilon mask first =
      trainingEncoded weights dilation padding gamma beta epsilon mask second := by
  unfold trainingEncoded
  rw [changing_only_missing_values_preserves_the_actual_temporal_sum weights dilation padding mask first second same]

theorem observed_scatter_recovers_its_original_coordinate (mask : Fin B → Fin C → Bool)
    (encoded : Slot mask → Fin F → Fin T → ℝ) (slot : Slot mask) (feature : Fin F) (time : Fin T) :
    restore mask encoded slot.1.1 feature slot.1.2 time = encoded slot feature time := by
  simp [restore, slot.2]

theorem missing_scatter_coordinate_is_zero_even_after_BN (mask : Fin B → Fin C → Bool)
    (encoded : Slot mask → Fin F → Fin T → ℝ) (batch : Fin B) (feature : Fin F) (channel : Fin C) (time : Fin T)
    (missing : mask batch channel = false) : restore mask encoded batch feature channel time = 0 := by
  simp [restore, missing]

theorem changing_only_missing_values_preserves_restored_BFCT
    (weights : Fin F → Fin K → ℝ) (dilation padding : ℕ) (gamma beta : Fin F → ℝ) (epsilon : ℝ)
    (mask : Fin B → Fin C → Bool) (first second : Fin B → Fin C → Fin T → ℝ)
    (same : ∀ batch channel, mask batch channel = true → ∀ time, first batch channel time = second batch channel time) :
    restoredTraining weights dilation padding gamma beta epsilon mask first =
      restoredTraining weights dilation padding gamma beta epsilon mask second := by
  unfold restoredTraining
  rw [changing_only_missing_values_preserves_training_BN weights dilation padding gamma beta epsilon mask first second same]

theorem missing_spatial_term_is_zero (mask : Fin B → Fin C → Bool)
    (encoded : Slot mask → Fin F → Fin T → ℝ) (batch : Fin B) (feature : Fin F) (channel : Fin C) (time : Fin T)
    (weight : ℝ) (missing : mask batch channel = false) :
    weight * restore mask encoded batch feature channel time = 0 := by
  rw [missing_scatter_coordinate_is_zero_even_after_BN mask encoded batch feature channel time missing, mul_zero]

theorem changing_only_missing_values_preserves_grouped_spatial_projection {M : ℕ}
    (temporal : Fin F → Fin K → ℝ) (spatial : Fin F → Fin M → Fin C → ℝ)
    (dilation padding : ℕ) (gamma beta : Fin F → ℝ) (epsilon : ℝ)
    (mask : Fin B → Fin C → Bool) (first second : Fin B → Fin C → Fin T → ℝ)
    (same : ∀ batch channel, mask batch channel = true → ∀ time, first batch channel time = second batch channel time) :
    groupedSpatial spatial (restoredTraining temporal dilation padding gamma beta epsilon mask first) =
      groupedSpatial spatial (restoredTraining temporal dilation padding gamma beta epsilon mask second) := by
  rw [changing_only_missing_values_preserves_restored_BFCT temporal dilation padding gamma beta epsilon mask first second same]

theorem positive_BN_epsilon_and_nonnegative_variance_give_positive_denominator
    (variance epsilon : ℝ) (nonnegative : 0 ≤ variance) (positive : 0 < epsilon) :
    0 < Real.sqrt (variance + epsilon) := by
  apply Real.sqrt_pos.2
  exact add_pos_of_nonneg_of_pos nonnegative positive

theorem observed_BN_population_counts_measured_slots_times_samples
    (mask : Fin B → Fin C → Bool) :
    ObservedStatistics.observedCount (fun _ : Slot mask × Fin T => true) = Fintype.card (Slot mask) * T := by
  simp [ObservedStatistics.observedCount, Fintype.card_prod]

theorem actual_training_variance_is_nonnegative (mask : Fin B → Fin C → Bool)
    (encoded : Slot mask → Fin F → Fin T → ℝ) (feature : Fin F) :
    0 ≤ trainingVariance mask encoded feature :=
  ObservedStatistics.variance_nonnegative _ _

def asymmetricKernel (_feature : Fin 1) (lag : Fin 3) : ℝ := if lag = 0 then 1 else 0
def asymmetricTrace (time : Fin 3) : ℝ := if time = 0 then 1 else if time = 1 then 2 else 4

theorem source_temporal_operator_is_cross_correlation_without_kernel_reversal :
    traceConvolution asymmetricKernel 1 1 asymmetricTrace 0 1 = 1 := by
  norm_num [traceConvolution, asymmetricKernel, asymmetricTrace, zeroExtend, Fin.sum_univ_succ]

theorem observed_trace_mean_differs_from_legacy_zero_filled_population :
    ((2 + 4) / 2 : ℝ) = 3 ∧ ((2 + 4 + 0 + 0) / 4 : ℝ) = 3 / 2 ∧ (3 : ℝ) ≠ 3 / 2 := by norm_num

theorem zero_input_encoded_by_BN_can_be_nonzero_before_scatter :
    bnCell 1 0 3 1 (1 / 100000) 0 ≠ 0 := by
  unfold bnCell
  simp only [one_mul, zero_sub, add_zero]
  exact div_ne_zero (by norm_num) (ne_of_gt (by
    apply Real.sqrt_pos.2
    norm_num))

theorem BN_epsilon_is_inside_the_square_root :
    bnCell 1 0 0 0 (1 / 4) 1 = 2 ∧ (1 / (Real.sqrt 0 + 1 / 4) : ℝ) = 4 := by
  have squareRoot : Real.sqrt (4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
  norm_num [bnCell, Real.sqrt_div, squareRoot]

def measuredMask (_batch : Fin 1) (channel : Fin 2) : Bool := decide (channel = 0)

def onlyMeasuredSlot : Slot measuredMask := ⟨(0, 0), rfl⟩

theorem every_fixed_measured_slot_has_channel_zero (slot : Slot measuredMask) : slot.1.2 = 0 := by
  have measured := slot.2
  change decide (slot.1.2 = 0) = true at measured
  exact of_decide_eq_true measured

theorem every_fixed_measured_slot_has_the_actual_original_coordinates (slot : Slot measuredMask) :
    slot = onlyMeasuredSlot := by
  apply Subtype.ext
  apply Prod.ext
  · exact Subsingleton.elim _ _
  · have measured := slot.2
    change decide (slot.1.2 = 0) = true at measured
    exact of_decide_eq_true measured

instance fixedMeasuredSlotUnique : Unique (Slot measuredMask) where
  default := onlyMeasuredSlot
  uniq := every_fixed_measured_slot_has_the_actual_original_coordinates

def identityKernel (_feature : Fin 1) (lag : Fin 3) : ℝ := if lag = 1 then 1 else 0
def measuredRaw (_batch : Fin 1) (channel : Fin 2) (time : Fin 2) : ℝ :=
  if channel = 0 then if time = 0 then 2 else 4 else 0

theorem fixed_source_population_has_two_native_training_observations :
    ObservedStatistics.observedCount (fun _ : Slot measuredMask × Fin 2 => true) = 2 := by
  rw [observed_BN_population_counts_measured_slots_times_samples]
  simp

theorem actual_identity_convolution_and_packed_population_have_mean_three :
    trainingMean measuredMask (packedConvolution identityKernel 1 1 measuredMask measuredRaw) 0 = 3 := by
  have lastNotMiddle : (2 : Fin 3) ≠ 1 := by decide
  norm_num [trainingMean, ObservedStatistics.maskedMean, ObservedStatistics.observedCount,
    ObservedStatistics.observedSum, Fintype.sum_prod_type, packedConvolution, traceConvolution,
    pack, identityKernel, measuredRaw, zeroExtend, every_fixed_measured_slot_has_channel_zero,
    fixedMeasuredSlotUnique, onlyMeasuredSlot, Fin.sum_univ_succ, lastNotMiddle]

theorem actual_identity_convolution_and_packed_population_have_variance_one :
    trainingVariance measuredMask (packedConvolution identityKernel 1 1 measuredMask measuredRaw) 0 = 1 := by
  have lastNotMiddle : (2 : Fin 3) ≠ 1 := by decide
  norm_num [trainingVariance, ObservedStatistics.maskedVariance, ObservedStatistics.maskedMean,
    ObservedStatistics.observedCount, ObservedStatistics.observedSum, Fintype.sum_prod_type,
    packedConvolution, traceConvolution, pack, identityKernel, measuredRaw, zeroExtend,
    every_fixed_measured_slot_has_channel_zero, fixedMeasuredSlotUnique, onlyMeasuredSlot, Fin.sum_univ_succ, lastNotMiddle]

end
end N2P3.ObservedTemporal
