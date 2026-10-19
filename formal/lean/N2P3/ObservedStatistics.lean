import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fin.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!+# Observed-only channel moments and the final missing-channel overwrite

Source: baselines/deep.py:594-675. Mathematical finite coordinates correspond
to (trial,time) for one channel. NumPy's float64 reductions and float32 casts
remain individual runtime boundaries. Invariance holds for any deterministic
cast of equal mathematical statistics, including a cast codomain with Inf.

The prepared-cell expression describes final values after successful native
subtract/divide operations. It makes no claim that absent coordinates skip
those operations or their exceptions. The independent source contract is
v07_observed_statistics_contract_01.json, written before this module.
-/
namespace N2P3.ObservedStatistics
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable

variable {ι : Type} [Fintype ι]

def observedCount (mask : ι → Bool) : ℕ :=
  ∑ index, if mask index = true then 1 else 0

def observedSum (mask : ι → Bool) (values : ι → ℝ) : ℝ :=
  ∑ index, if mask index = true then values index else 0

def maskedMean (mask : ι → Bool) (values : ι → ℝ) : ℝ :=
  if observedCount mask = 0 then 0 else observedSum mask values / observedCount mask

def maskedVariance (mask : ι → Bool) (values : ι → ℝ) : ℝ :=
  if observedCount mask = 0 then 0 else
    observedSum mask (fun index => (values index - maskedMean mask values) ^ 2) / observedCount mask

def maskedStd (epsilon : ℝ) (mask : ι → Bool) (values : ι → ℝ) : ℝ :=
  if observedCount mask = 0 then 1 else Real.sqrt (max (maskedVariance mask values) 0) + epsilon

def castStatistics {β : Type} (cast : ℝ → β) (epsilon : ℝ) (mask : ι → Bool) (values : ι → ℝ) : β × β :=
  (cast (maskedMean mask values), cast (maskedStd epsilon mask values))

def preparedCell {α β : Type} (subtract : ℝ → β → α) (divide : α → β → α)
    (zero : α) (value : ℝ) (mean scale : β) (observed : Bool) : α :=
  let centered := subtract value mean
  let standardized := divide centered scale
  if observed = true then standardized else zero

theorem observed_sum_depends_only_on_measured_values (mask : ι → Bool) (first second : ι → ℝ)
    (same : ∀ index, mask index = true → first index = second index) :
    observedSum mask first = observedSum mask second := by
  apply Finset.sum_congr rfl
  intro index _
  by_cases measured : mask index = true
  · simp [measured, same index measured]
  · simp [measured]

theorem mean_depends_only_on_measured_values (mask : ι → Bool) (first second : ι → ℝ)
    (same : ∀ index, mask index = true → first index = second index) :
    maskedMean mask first = maskedMean mask second := by
  simp only [maskedMean, observed_sum_depends_only_on_measured_values mask first second same]

theorem variance_depends_only_on_measured_values (mask : ι → Bool) (first second : ι → ℝ)
    (same : ∀ index, mask index = true → first index = second index) :
    maskedVariance mask first = maskedVariance mask second := by
  have mean := mean_depends_only_on_measured_values mask first second same
  have centered := observed_sum_depends_only_on_measured_values mask
    (fun index => (first index - maskedMean mask first)^2)
    (fun index => (second index - maskedMean mask second)^2)
    (by intro index measured; dsimp only; rw [same index measured, mean])
  simp only [maskedVariance, centered]

theorem std_depends_only_on_measured_values (epsilon : ℝ) (mask : ι → Bool) (first second : ι → ℝ)
    (same : ∀ index, mask index = true → first index = second index) :
    maskedStd epsilon mask first = maskedStd epsilon mask second := by
  simp only [maskedStd, variance_depends_only_on_measured_values mask first second same]

theorem each_deterministic_cast_preserves_measured_only_statistics {β : Type}
    (cast : ℝ → β) (epsilon : ℝ) (mask : ι → Bool) (first second : ι → ℝ)
    (same : ∀ index, mask index = true → first index = second index) :
    castStatistics cast epsilon mask first = castStatistics cast epsilon mask second := by
  simp only [castStatistics, mean_depends_only_on_measured_values mask first second same,
    std_depends_only_on_measured_values epsilon mask first second same]

theorem all_absent_has_zero_count (mask : ι → Bool) (absent : ∀ index, mask index = false) :
    observedCount mask = 0 := by simp [observedCount, absent]

theorem repeated_trial_mask_count_is_observed_trials_times_samples
    (trials samples : ℕ) (mask : Fin trials → Bool) :
    observedCount (fun coordinate : Fin trials × Fin samples => mask coordinate.1) =
      observedCount mask * samples := by
  unfold observedCount
  rw [Fintype.sum_prod_type]
  simp only [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro index _
  by_cases measured : mask index = true <;> simp [measured]

theorem zero_observation_mean_and_scale_conventions (epsilon : ℝ) (mask : ι → Bool) (values : ι → ℝ)
    (empty : observedCount mask = 0) : maskedMean mask values = 0 ∧ maskedStd epsilon mask values = 1 := by
  simp [maskedMean, maskedStd, empty]

theorem variance_nonnegative (mask : ι → Bool) (values : ι → ℝ) : 0 ≤ maskedVariance mask values := by
  unfold maskedVariance
  split_ifs
  · exact le_rfl
  · apply div_nonneg _ (Nat.cast_nonneg _)
    apply Finset.sum_nonneg
    intro index _
    by_cases measured : mask index = true
    · simp only [if_pos measured]
      exact sq_nonneg _
    · simp only [if_neg measured]
      exact le_rfl

theorem positive_epsilon_gives_positive_scale (epsilon : ℝ) (positive : 0 < epsilon)
    (mask : ι → Bool) (values : ι → ℝ) : 0 < maskedStd epsilon mask values := by
  unfold maskedStd
  split_ifs
  · exact zero_lt_one
  · exact add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) positive

theorem final_overwrite_removes_any_absent_cell {α β : Type}
    (subtract : ℝ → β → α) (divide : α → β → α) (zero : α)
    (value : ℝ) (mean scale : β) : preparedCell subtract divide zero value mean scale false = zero := by
  simp [preparedCell]

theorem measured_cell_keeps_subtract_then_divide {α β : Type}
    (subtract : ℝ → β → α) (divide : α → β → α) (zero : α)
    (value : ℝ) (mean scale : β) :
    preparedCell subtract divide zero value mean scale true = divide (subtract value mean) scale := by
  simp [preparedCell]

def exampleMask : Fin 2 → Bool := fun index => decide (index = 0)
def exampleValues : Fin 2 → ℝ := fun index => if index = 0 then 2 else 0
def changedAbsentValues : Fin 2 → ℝ := fun index => if index = 0 then 2 else 100

def unmaskedMean (values : Fin 2 → ℝ) : ℝ := (values 0 + values 1) / 2

theorem measured_mean_is_two_for_both_absent_fill_values :
    maskedMean exampleMask exampleValues = 2 ∧ maskedMean exampleMask changedAbsentValues = 2 := by
  norm_num [maskedMean, observedCount, observedSum, exampleMask, exampleValues, changedAbsentValues,
    Fin.sum_univ_succ]

theorem unmasked_mean_changes_with_the_absent_fill_value :
    unmaskedMean exampleValues = 1 ∧ unmaskedMean changedAbsentValues = 51 := by
  norm_num [unmaskedMean, exampleValues, changedAbsentValues]

theorem subtracting_mean_without_remask_creates_nonzero_missing_value :
    ((0 - 2) / 1 : ℝ) = -2 ∧ preparedCell (· - ·) (· / ·) (0 : ℝ) 0 2 1 false = 0 := by
  norm_num [preparedCell]

end
end N2P3.ObservedStatistics
