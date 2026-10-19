import N2P3.RequestedOutcomeRates
import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Data.Real.Archimedean
import Mathlib.Order.Fin.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Exact interface for NumPy's audited default method="linear". Sorting is
an index-permutation primitive with monotone ordered values, preserving every
draw position including ties. No endpoint/order/convex result is assumed.
The native sort/floor/interpolation implementation is not refined here. -/

namespace N2P3.LinearQuantile

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

structure OrderedSample (n : ℕ) where
  raw : Fin n → ℝ
  rank : Equiv.Perm (Fin n)
  ordered : Monotone (fun index => raw (rank index))

def OrderedSample.at {n : ℕ} (sample : OrderedSample n) (index : Fin n) : ℝ := sample.raw (sample.rank index)

def virtualIndex (n : ℕ) (probability : ℝ) : ℝ := ((n-1 : ℕ) : ℝ)*probability

theorem virtual_index_is_inside_the_sample_span (n : ℕ) (probability : ℝ)
    (valid : 0≤probability ∧ probability≤1) :
    0≤virtualIndex n probability ∧ virtualIndex n probability≤((n-1 : ℕ) : ℝ) := by
  unfold virtualIndex
  exact ⟨mul_nonneg (Nat.cast_nonneg _) valid.1,
    by simpa using mul_le_mul_of_nonneg_left valid.2 (Nat.cast_nonneg (n-1))⟩

theorem lower_order_statistic_index_is_legal (n : ℕ) (positive : 0<n) (probability : ℝ)
    (valid : 0≤probability ∧ probability≤1) : ⌊virtualIndex n probability⌋₊<n := by
  have bounded : ⌊virtualIndex n probability⌋₊≤n-1 :=
    Nat.floor_le_of_le (virtual_index_is_inside_the_sample_span n probability valid).2
  omega

def lowerIndex (n : ℕ) (positive : 0<n) (probability : ℝ) (valid : 0≤probability ∧ probability≤1) : Fin n :=
  ⟨⌊virtualIndex n probability⌋₊,lower_order_statistic_index_is_legal n positive probability valid⟩

def upperIndex (n : ℕ) (positive : 0<n) (probability : ℝ) (valid : 0≤probability ∧ probability≤1) : Fin n :=
  ⟨min ((lowerIndex n positive probability valid).val+1) (n-1),by
    have bound := Nat.min_le_right ((lowerIndex n positive probability valid).val+1) (n-1)
    omega⟩

def fraction (n : ℕ) (probability : ℝ) : ℝ := virtualIndex n probability-(⌊virtualIndex n probability⌋₊ : ℝ)

def quantile {n : ℕ} (sample : OrderedSample n) (positive : 0<n)
    (probability : ℝ) (valid : 0≤probability ∧ probability≤1) : ℝ :=
  (1-fraction n probability)*sample.at (lowerIndex n positive probability valid)+
    fraction n probability*sample.at (upperIndex n positive probability valid)

theorem interpolation_fraction_is_derived_from_floor_and_bounded (n : ℕ) (probability : ℝ)
    (valid : 0≤probability ∧ probability≤1) : 0≤fraction n probability ∧ fraction n probability<1 := by
  have nonnegative := (virtual_index_is_inside_the_sample_span n probability valid).1
  have floorLower := Nat.floor_le nonnegative
  have floorUpper := Nat.lt_floor_add_one (virtualIndex n probability)
  unfold fraction
  exact ⟨by linarith,by linarith⟩

theorem bracketing_indices_remain_in_sorted_order (n : ℕ) (positive : 0<n) (probability : ℝ)
    (valid : 0≤probability ∧ probability≤1) :
    lowerIndex n positive probability valid≤upperIndex n positive probability valid := by
  change ⌊virtualIndex n probability⌋₊≤min (⌊virtualIndex n probability⌋₊+1) (n-1)
  exact le_min (by omega) (Nat.floor_le_of_le (virtual_index_is_inside_the_sample_span n probability valid).2)

theorem linear_interpolation_is_between_its_adjacent_order_statistics
    {n : ℕ} (sample : OrderedSample n) (positive : 0<n) (probability : ℝ)
    (valid : 0≤probability ∧ probability≤1) :
    sample.at (lowerIndex n positive probability valid)≤quantile sample positive probability valid ∧
      quantile sample positive probability valid≤sample.at (upperIndex n positive probability valid) := by
  have adjacent := sample.ordered (bracketing_indices_remain_in_sorted_order n positive probability valid)
  have gamma := interpolation_fraction_is_derived_from_floor_and_bounded n probability valid
  unfold quantile OrderedSample.at
  exact ⟨by nlinarith,by nlinarith⟩

theorem zero_quantile_is_the_first_order_statistic {n : ℕ} (sample : OrderedSample n) (positive : 0<n) :
    quantile sample positive 0 (by norm_num)=sample.at ⟨0,positive⟩ := by
  have first : lowerIndex n positive 0 (by norm_num)=⟨0,positive⟩ := by
    apply Fin.ext
    simp [lowerIndex,virtualIndex]
  simp [quantile,fraction,virtualIndex,first]

theorem one_quantile_is_the_last_order_statistic {n : ℕ} (sample : OrderedSample n) (positive : 0<n) :
    quantile sample positive 1 (by norm_num)=sample.at ⟨n-1,by omega⟩ := by
  have last : lowerIndex n positive 1 (by norm_num)=⟨n-1,by omega⟩ := by
    apply Fin.ext
    simp [lowerIndex,virtualIndex]
  simp [quantile,fraction,virtualIndex,last]

theorem quantile_is_inside_the_full_sample_convex_bounds {n : ℕ} (sample : OrderedSample n) (positive : 0<n)
    (probability : ℝ) (valid : 0≤probability ∧ probability≤1) :
    sample.at ⟨0,positive⟩≤quantile sample positive probability valid ∧
      quantile sample positive probability valid≤sample.at ⟨n-1,by omega⟩ := by
  have localBounds := linear_interpolation_is_between_its_adjacent_order_statistics sample positive probability valid
  have first := sample.ordered (show (⟨0,positive⟩ : Fin n)≤lowerIndex n positive probability valid from Nat.zero_le _)
  have last := sample.ordered (show upperIndex n positive probability valid≤(⟨n-1,by omega⟩ : Fin n) from Nat.min_le_right _ _)
  exact ⟨first.trans localBounds.1,localBounds.2.trans last⟩

theorem sorted_interface_preserves_every_original_position {n : ℕ}
    (sample : OrderedSample n) (unit : Fin n) :
    ∃ orderedIndex,sample.rank orderedIndex=unit ∧ sample.at orderedIndex=sample.raw unit := by
  refine ⟨sample.rank.symm unit,sample.rank.apply_symm_apply unit,?_⟩
  simp [OrderedSample.at]

theorem quantile_respects_every_bound_on_original_sample_values {n : ℕ}
    (sample : OrderedSample n) (positive : 0<n) (probability : ℝ) (valid : 0≤probability ∧ probability≤1)
    (lower upper : ℝ) (bounded : ∀ unit,lower≤sample.raw unit ∧ sample.raw unit≤upper) :
    lower≤quantile sample positive probability valid ∧ quantile sample positive probability valid≤upper := by
  have convex := quantile_is_inside_the_full_sample_convex_bounds sample positive probability valid
  exact ⟨(bounded (sample.rank ⟨0,positive⟩)).1.trans convex.1,
    convex.2.trans (bounded (sample.rank ⟨n-1,by omega⟩)).2⟩

theorem ordered_probabilities_give_ordered_linear_quantiles {n : ℕ}
    (sample : OrderedSample n) (positive : 0<n) (left right : ℝ)
    (leftValid : 0≤left ∧ left≤1) (rightValid : 0≤right ∧ right≤1) (ordered : left≤right) :
    quantile sample positive left leftValid≤quantile sample positive right rightValid := by
  have virtualOrder : virtualIndex n left≤virtualIndex n right :=
    mul_le_mul_of_nonneg_left ordered (Nat.cast_nonneg (n-1))
  have floorOrder := Nat.floor_le_floor virtualOrder
  by_cases sameFloor : ⌊virtualIndex n left⌋₊=⌊virtualIndex n right⌋₊
  · have sameLower : lowerIndex n positive left leftValid=lowerIndex n positive right rightValid :=
      Fin.ext sameFloor
    have sameUpper : upperIndex n positive left leftValid=upperIndex n positive right rightValid := by
      apply Fin.ext
      change min (⌊virtualIndex n left⌋₊+1) (n-1)=min (⌊virtualIndex n right⌋₊+1) (n-1)
      rw [sameFloor]
    have fractionOrder : fraction n left≤fraction n right := by
      unfold fraction
      rw [sameFloor]
      linarith
    have adjacent : sample.at (lowerIndex n positive right rightValid)≤sample.at (upperIndex n positive right rightValid) :=
      sample.ordered (bracketing_indices_remain_in_sorted_order n positive right rightValid)
    have product := mul_nonneg (sub_nonneg.mpr fractionOrder) (sub_nonneg.mpr adjacent)
    unfold quantile
    rw [sameLower,sameUpper]
    nlinarith
  · have nextLower : ⌊virtualIndex n left⌋₊+1≤⌊virtualIndex n right⌋₊ := by omega
    have bridge : upperIndex n positive left leftValid≤lowerIndex n positive right rightValid := by
      change min (⌊virtualIndex n left⌋₊+1) (n-1)≤⌊virtualIndex n right⌋₊
      exact (Nat.min_le_left _ _).trans nextLower
    have leftBound := (linear_interpolation_is_between_its_adjacent_order_statistics sample positive left leftValid).2
    have rightBound := (linear_interpolation_is_between_its_adjacent_order_statistics sample positive right rightValid).1
    exact leftBound.trans ((sample.ordered bridge).trans rightBound)

def lowerProbability (confidence : ℝ) : ℝ := (1-confidence)/2
def upperProbability (confidence : ℝ) : ℝ := 1-lowerProbability confidence

theorem confidence_level_produces_legal_ordered_probabilities (confidence : ℝ)
    (valid : 0<confidence ∧ confidence<1) :
    0≤lowerProbability confidence ∧ lowerProbability confidence≤upperProbability confidence ∧ upperProbability confidence≤1 := by
  unfold upperProbability lowerProbability
  exact ⟨by linarith,by linarith,by linarith⟩

def interval {n : ℕ} (sample : OrderedSample n) (positive : 0<n) (confidence : ℝ)
    (valid : 0<confidence ∧ confidence<1) : ℝ × ℝ :=
  let probabilities := confidence_level_produces_legal_ordered_probabilities confidence valid
  (quantile sample positive (lowerProbability confidence) ⟨probabilities.1,probabilities.2.1.trans probabilities.2.2⟩,
   quantile sample positive (upperProbability confidence) ⟨probabilities.1.trans probabilities.2.1,probabilities.2.2⟩)

theorem linear_quantile_interval_has_correct_endpoint_order {n : ℕ} (sample : OrderedSample n)
    (positive : 0<n) (confidence : ℝ) (valid : 0<confidence ∧ confidence<1) :
    (interval sample positive confidence valid).1≤(interval sample positive confidence valid).2 := by
  exact ordered_probabilities_give_ordered_linear_quantiles sample positive _ _ _ _
    (confidence_level_produces_legal_ordered_probabilities confidence valid).2.1

theorem linear_quantile_interval_is_inside_original_sample_bounds {n : ℕ} (sample : OrderedSample n)
    (positive : 0<n) (confidence : ℝ) (valid : 0<confidence ∧ confidence<1)
    (lower upper : ℝ) (bounded : ∀ unit,lower≤sample.raw unit ∧ sample.raw unit≤upper) :
    lower≤(interval sample positive confidence valid).1 ∧ (interval sample positive confidence valid).2≤upper := by
  exact ⟨(quantile_respects_every_bound_on_original_sample_values sample positive _ _ lower upper bounded).1,
    (quantile_respects_every_bound_on_original_sample_values sample positive _ _ lower upper bounded).2⟩

theorem standard_95_confidence_uses_the_source_two_probabilities :
    lowerProbability (19/20)=1/40 ∧ upperProbability (19/20)=39/40 := by
  norm_num [lowerProbability,upperProbability]

end
end N2P3.LinearQuantile
