import N2P3.RequestedOutcomeRates

/-! Pair-count mathematical AUC and its aggregation unit. Source uses
roc_auc_score over the actual scored trial vector; native sklearn/IEEE
refinement is separate. Missing/one-class AUC is undefined, not an operational
decision miss. The fixed macro/pool counterexample is C13 in requirements01. -/

namespace N2P3.RequestedAUC

open N2P3.RequestedOutcomeRates

noncomputable section
attribute [local instance] Classical.propDecidable

def pairCredit (positive negative : ℝ) : ℝ :=
  if negative < positive then 1 else if positive=negative then 1/2 else 0

def pairSum (positives negatives : List ℝ) : ℝ :=
  ((positives.product negatives).map (fun pair => pairCredit pair.1 pair.2)).sum

def auc (positives negatives : List ℝ) : Option ℝ :=
  if positives=[] ∨ negatives=[] then none
  else some (pairSum positives negatives/((positives.length*negatives.length : ℕ) : ℝ))

theorem each_rank_pair_credit_is_bounded (positive negative : ℝ) :
    0 ≤ pairCredit positive negative ∧ pairCredit positive negative ≤ 1 := by
  unfold pairCredit
  split_ifs <;> norm_num

theorem equal_scores_contribute_one_half (score : ℝ) : pairCredit score score=1/2 := by
  simp [pairCredit]

theorem pair_sum_bounds_use_positive_negative_pairs (positives negatives : List ℝ) :
    0 ≤ pairSum positives negatives ∧
      pairSum positives negatives ≤ ((positives.length*negatives.length : ℕ) : ℝ) := by
  have bounds := bounded_rate_sum (positives.product negatives)
    (fun pair => pairCredit pair.1 pair.2) (fun pair _ => each_rank_pair_credit_is_bounded pair.1 pair.2)
  have pairsLength := List.length_product positives negatives
  change (positives.product negatives).length=positives.length*negatives.length at pairsLength
  simpa only [pairSum,pairsLength] using bounds

theorem defined_auc_is_between_zero_and_one (positives negatives : List ℝ)
    (hasPositive : positives ≠ []) (hasNegative : negatives ≠ []) :
    ∃ value, auc positives negatives=some value ∧ 0 ≤ value ∧ value ≤ 1 := by
  have pairsPositive : 0 < positives.length*negatives.length :=
    Nat.mul_pos (List.length_pos_iff.mpr hasPositive) (List.length_pos_iff.mpr hasNegative)
  have denominator : 0 < ((positives.length*negatives.length : ℕ) : ℝ) := Nat.cast_pos.mpr pairsPositive
  have bounds := pair_sum_bounds_use_positive_negative_pairs positives negatives
  refine ⟨pairSum positives negatives/((positives.length*negatives.length : ℕ) : ℝ),?_,?_,?_⟩
  · simp [auc,hasPositive,hasNegative]
  · exact div_nonneg bounds.1 (le_of_lt denominator)
  · exact (div_le_one denominator).mpr bounds.2

theorem empty_positive_or_negative_class_has_no_auc (values : List ℝ) :
    auc [] values=none ∧ auc values []=none := by simp [auc]

theorem source_fixed_macro_auc_differs_from_pooled_auc :
    auc [1] [0]=some 1 ∧ auc [101] [100]=some 1 ∧
      auc [1,101] [0,100]=some (3/4 : ℝ) := by
  norm_num [auc,pairSum,pairCredit,List.product,List.cons_ne_nil]

end
end N2P3.RequestedAUC
