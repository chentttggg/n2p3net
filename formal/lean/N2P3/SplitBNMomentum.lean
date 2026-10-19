import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Lean.Elab.Tactic.Omega

/-!
# Literal tracked-counter ordering and scalar BN running averages

Source: splitbn.py:352-370,398-429. This exact signed-counter/scalar image
preserves increment-before-reciprocal, including failure after increment.
It does not certify int64 overflow, item/float casts or IEEE/native updates.
Contract: v07_splitbn_momentum_contract_01.json fixed before this module.
-/
namespace N2P3.SplitBNMomentum
noncomputable section
attribute [local instance] Classical.propDecidable

inductive Failure where
  | divisionByZero
  deriving DecidableEq

def factor (momentum : Option ℝ) (counter : ℤ) : Except Failure ℝ × ℤ :=
  let next := counter + 1
  let result : Except Failure ℝ := match momentum with
    | some alpha => .ok alpha
    | none => if next = 0 then .error .divisionByZero else .ok (1 / (next : ℝ))
  (result, next)

def domainFactor (update : Bool) (momentum : Option ℝ) (counter : ℤ) : Except Failure ℝ × ℤ :=
  if update then factor momentum counter else (.ok 0, counter)

def average (alpha old current : ℝ) : ℝ := (1 - alpha) * old + alpha * current

theorem every_factor_call_increments_before_its_outcome (momentum : Option ℝ) (counter : ℤ) :
    (factor momentum counter).2 = counter + 1 := rfl

theorem fixed_momentum_returns_the_declared_scalar_after_increment (alpha : ℝ) (counter : ℤ) :
    factor (some alpha) counter = (.ok alpha, counter + 1) := rfl

theorem zero_momentum_is_not_a_complete_tracked_state_freeze :
    factor (some 0) 0 = (.ok 0, 1) := rfl

theorem reciprocal_zero_failure_keeps_the_already_incremented_counter :
    factor none (-1) = (.error .divisionByZero, 0) := by simp [factor]

theorem a_negative_counter_can_produce_a_negative_reciprocal_factor :
    factor none (-2) = (.ok (-1), -1) := by norm_num [factor]

theorem nonnegative_counter_derives_success_and_positive_reciprocal (counter : ℤ) (nonnegative : 0 ≤ counter) :
    factor none counter = (.ok (1 / ((counter + 1 : ℤ) : ℝ)), counter + 1) ∧
      0 < 1 / ((counter + 1 : ℤ) : ℝ) ∧ 1 / ((counter + 1 : ℤ) : ℝ) ≤ 1 := by
  have positiveInt : 0 < counter + 1 := by omega
  have positiveReal : (0 : ℝ) < ((counter + 1 : ℤ) : ℝ) := by exact_mod_cast positiveInt
  have atLeastOne : (1 : ℝ) ≤ ((counter + 1 : ℤ) : ℝ) := by exact_mod_cast (show 1 ≤ counter + 1 by omega)
  refine ⟨?_, one_div_pos.mpr positiveReal, ?_⟩
  · simp [factor, ne_of_gt positiveInt]
  · exact (div_le_one positiveReal).2 atLeastOne

theorem eval_domain_factor_is_zero_without_a_counter_update (momentum : Option ℝ) (counter : ℤ) :
    domainFactor false momentum counter = (.ok 0, counter) := rfl

theorem training_domain_factor_uses_the_same_counter_transition (momentum : Option ℝ) (counter : ℤ) :
    domainFactor true momentum counter = factor momentum counter := rfl

theorem scalar_average_with_zero_factor_keeps_the_numeric_statistic (old current : ℝ) :
    average 0 old current = old := by simp [average]

theorem scalar_average_with_unit_factor_replaces_the_numeric_statistic (old current : ℝ) :
    average 1 old current = current := by simp [average]

theorem scalar_average_preserves_a_common_lower_bound (alpha old current lower : ℝ)
    (nonnegative : 0 ≤ alpha) (atMostOne : alpha ≤ 1) (oldBound : lower ≤ old) (currentBound : lower ≤ current) :
    lower ≤ average alpha old current := by
  have first := mul_le_mul_of_nonneg_left oldBound (sub_nonneg.mpr atMostOne)
  have second := mul_le_mul_of_nonneg_left currentBound nonnegative
  unfold average
  nlinarith

theorem scalar_average_preserves_a_common_upper_bound (alpha old current upper : ℝ)
    (nonnegative : 0 ≤ alpha) (atMostOne : alpha ≤ 1) (oldBound : old ≤ upper) (currentBound : current ≤ upper) :
    average alpha old current ≤ upper := by
  have first := mul_le_mul_of_nonneg_left oldBound (sub_nonneg.mpr atMostOne)
  have second := mul_le_mul_of_nonneg_left currentBound nonnegative
  unfold average
  nlinarith

theorem legal_factor_preserves_nonnegative_variance (alpha old current : ℝ)
    (nonnegative : 0 ≤ alpha) (atMostOne : alpha ≤ 1) (oldVariance : 0 ≤ old) (currentVariance : 0 ≤ current) :
    0 ≤ average alpha old current :=
  scalar_average_preserves_a_common_lower_bound alpha old current 0 nonnegative atMostOne oldVariance currentVariance

theorem first_cumulative_batch_replaces_the_initial_statistic (old current : ℝ) :
    average (1 / (0 + 1 : ℝ)) old current = current := by simp [average]

theorem cumulative_factor_is_the_batch_count_average (counter : ℕ) (old current : ℝ) :
    average (1 / (counter + 1 : ℝ)) old current = (counter * old + current) / (counter + 1 : ℝ) := by
  have nonzero : (counter + 1 : ℝ) ≠ 0 := by positivity
  unfold average
  field_simp [nonzero]
  ring

theorem equal_batch_average_is_not_a_pooled_observation_average :
    average (1 / 2) 0 10 = 5 ∧ ((2 * 0 + 18 * 10) / (2 + 18) : ℝ) = 9 ∧ (5 : ℝ) ≠ 9 := by
  norm_num [average]

end
end N2P3.SplitBNMomentum
