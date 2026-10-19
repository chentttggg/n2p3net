import N2P3.DecisionLossCalculus
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-! SourceObjective's finite exact-real risk, with physical-row first occurrence.
The source-first contract is lean_source_auxiliary_20261018/source_contract_01.json.
Candidate types contain only actual eligible logits. Empty anchors are handled
by activeMean; no log(0) or masked-infinity arithmetic is used in the model.
This does not assert native AD, sampling/BN equivalence, or unit normalization. -/
namespace N2P3.SourceAuxiliaryRisk
open scoped BigOperators
open N2P3.CandidateProbability N2P3.DecisionLossCalculus
noncomputable section

def First {b : ℕ} (rows : Fin b → ℕ) (i : Fin b) : Prop :=
  ∀ j, j < i → rows j ≠ rows i

theorem first_zero {b : ℕ} (rows : Fin (b+1) → ℕ) : First rows 0 := by
  intro j hj
  exact False.elim (not_lt_of_ge (Fin.zero_le j) hj)

theorem later_duplicate_excluded {b : ℕ} (rows : Fin b → ℕ) (i j : Fin b)
    (hij : i < j) (same : rows i = rows j) : ¬ First rows j := by
  intro h
  exact h i hij same

theorem first_representatives_unique {b : ℕ} (rows : Fin b → ℕ) (i j : Fin b)
    (hi : First rows i) (hj : First rows j) (same : rows i = rows j) : i = j := by
  rcases lt_trichotomy i j with h | h | h
  · exact False.elim (hj i h same)
  · exact h
  · exact False.elim (hi j h same.symm)

theorem represented_row_has_first {b : ℕ} (rows : Fin b → ℕ) (i : Fin b) :
    ∃ j, First rows j ∧ rows j = rows i := by
  let P : ℕ → Prop := fun k => ∃ hk : k < b, rows ⟨k,hk⟩ = rows i
  have hex : ∃ k, P k := ⟨i.val, i.isLt, rfl⟩
  obtain ⟨hk, heq⟩ := Nat.find_spec hex
  refine ⟨⟨Nat.find hex, hk⟩, ?_, heq⟩
  intro j hj same
  have hmin := Nat.find_min' hex (show P j.val from ⟨j.isLt, same.trans heq⟩)
  exact (not_lt_of_ge hmin) hj

def Positive {ι : Type*} (keep : ι → Prop) (label domain person : ι → ℕ) (i j : ι) : Prop :=
  keep i ∧ keep j ∧ label i = label j ∧ domain i ≠ domain j ∧ person i ≠ person j

def Negative {ι : Type*} (keep : ι → Prop) (label : ι → ℕ) (i j : ι) : Prop :=
  keep i ∧ keep j ∧ label i ≠ label j

def Candidate {ι : Type*} (keep : ι → Prop) (label domain person : ι → ℕ) (i j : ι) : Prop :=
  Positive keep label domain person i j ∨ Negative keep label i j

theorem positive_symmetric {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ) (i j : ι) :
    Positive keep y d p i j ↔ Positive keep y d p j i := by
  simp only [Positive]; aesop

theorem negative_symmetric {ι : Type*} (keep : ι → Prop) (y : ι → ℕ) (i j : ι) :
    Negative keep y i j ↔ Negative keep y j i := by
  simp only [Negative]; aesop

theorem candidate_no_self {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ) (i : ι) :
    ¬ Candidate keep y d p i i := by simp [Candidate, Positive, Negative]

theorem positive_negative_disjoint {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ) (i j : ι) :
    Positive keep y d p i j → ¬ Negative keep y i j := by
  simp only [Positive, Negative]; aesop

theorem same_class_same_domain_excluded {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ)
    (i j : ι) (hy : y i = y j) (hd : d i = d j) : ¬ Candidate keep y d p i j := by
  simp [Candidate, Positive, Negative, hy, hd]

theorem same_class_same_person_excluded {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ)
    (i j : ι) (hy : y i = y j) (hp : p i = p j) : ¬ Candidate keep y d p i j := by
  simp [Candidate, Positive, Negative, hy, hp]

theorem different_class_is_candidate {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ)
    (i j : ι) (hi : keep i) (hj : keep j) (hy : y i ≠ y j) : Candidate keep y d p i j := by
  exact Or.inr ⟨hi,hj,hy⟩

theorem dropped_row_no_candidate {ι : Type*} (keep : ι → Prop) (y d p : ι → ℕ)
    (i j : ι) (hi : ¬ keep i) : ¬ Candidate keep y d p i j := by
  simp [Candidate, Positive, Negative, hi]

def activeMean {ι : Type*} (active : Finset ι) (loss : ι → ℝ) : ℝ :=
  (∑ i ∈ active, loss i) / max (active.card : ℝ) 1

theorem active_mean_empty {ι : Type*} (loss : ι → ℝ) : activeMean ∅ loss = 0 := by
  simp [activeMean]

theorem active_mean_nonempty {ι : Type*} (a : Finset ι) (loss : ι → ℝ) (ha : a.Nonempty) :
    activeMean a loss = (∑ i ∈ a, loss i) / (a.card : ℝ) := by
  have hc : (1:ℝ) ≤ a.card := by exact_mod_cast ha.card_pos
  simp [activeMean, max_eq_left hc]

theorem active_mean_nonnegative {ι : Type*} (a : Finset ι) (loss : ι → ℝ)
    (h : ∀ i ∈ a, 0 ≤ loss i) : 0 ≤ activeMean a loss := by
  exact div_nonneg (Finset.sum_nonneg h) (le_trans (by norm_num) (le_max_right _ _))

theorem active_mean_congr {ι : Type*} (a : Finset ι) (f g : ι → ℝ)
    (same : ∀ i ∈ a, f i = g i) : activeMean a f = activeMean a g := by
  unfold activeMean
  rw [Finset.sum_congr rfl same]

theorem active_mean_constant {ι : Type*} (a : Finset ι) (ha : a.Nonempty) (c : ℝ) :
    activeMean a (fun _ => c) = c := by
  rw [active_mean_nonempty a _ ha]
  simp only [Finset.sum_const, nsmul_eq_mul]
  exact mul_div_cancel_left₀ c (by exact_mod_cast ha.card_pos.ne')

theorem fixed_row_importance_cancellation {ι : Type*} [Fintype ι]
    (proposal mass loss : ι → ℝ) (hp : ∀ i, proposal i ≠ 0) :
    (∑ i, proposal i * (mass i / proposal i * loss i)) = ∑ i, mass i * loss i := by
  apply Finset.sum_congr rfl
  intro i _
  field_simp [hp i]

def anchorRisk {κ : Type*} [Fintype κ] (scores : κ → ℝ) (positives : Finset κ) : ℝ :=
  logPartition scores - (∑ i ∈ positives, scores i) / (positives.card : ℝ)

theorem anchor_risk_is_mean_nll {κ : Type*} [Fintype κ] (scores : κ → ℝ)
    (positives : Finset κ) (hp : positives.Nonempty) :
    anchorRisk scores positives = (∑ i ∈ positives, nll scores i) / (positives.card : ℝ) := by
  have hn : (positives.card : ℝ) ≠ 0 := by exact_mod_cast hp.card_pos.ne'
  simp only [anchorRisk, nll, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
  field_simp

theorem anchor_risk_nonnegative {κ : Type*} [Fintype κ] [Nonempty κ] (scores : κ → ℝ)
    (positives : Finset κ) (hp : positives.Nonempty) : 0 ≤ anchorRisk scores positives := by
  rw [anchor_risk_is_mean_nll scores positives hp]
  exact div_nonneg (Finset.sum_nonneg (fun i _ => nll_nonnegative scores i)) (Nat.cast_nonneg _)

theorem anchor_risk_common_shift {κ : Type*} [Fintype κ] [Nonempty κ] (scores : κ → ℝ)
    (positives : Finset κ) (hp : positives.Nonempty) (c : ℝ) :
    anchorRisk (fun i => scores i + c) positives = anchorRisk scores positives := by
  rw [anchor_risk_is_mean_nll _ _ hp, anchor_risk_is_mean_nll _ _ hp]
  simp only [nll_common_constant]

theorem singleton_anchor_is_active_zero (score : ℝ) :
    anchorRisk (fun _ : Fin 1 => score) Finset.univ = 0 ∧ (Finset.univ : Finset (Fin 1)).Nonempty := by
  simp [anchorRisk, logPartition, partition, Real.log_exp]

theorem anchor_score_line_derivative {κ : Type*} [Fintype κ] [Nonempty κ]
    (scores direction : κ → ℝ) (positives : Finset κ) (x : ℝ) :
    HasDerivAt (fun t => anchorRisk (scoreLine scores direction t) positives)
      ((∑ i, probability (scoreLine scores direction x) i * direction i) -
        (∑ i ∈ positives, direction i) / (positives.card : ℝ)) x := by
  exact (log_partition_line_derivative scores direction x).sub
    ((HasDerivAt.fun_sum (u := positives) (fun i _ => score_line_derivative scores direction x i)).div_const _)

def normalizedLength (r eps : ℝ) : ℝ := r / max r eps

theorem normalized_length_zero (eps : ℝ) : normalizedLength 0 eps = 0 := by
  simp [normalizedLength]

theorem normalized_length_bounds (r eps : ℝ) (hr : 0 ≤ r) (he : 0 < eps) :
    0 ≤ normalizedLength r eps ∧ normalizedLength r eps ≤ 1 := by
  have hd : 0 < max r eps := lt_of_lt_of_le he (le_max_right _ _)
  exact ⟨div_nonneg hr hd.le, (div_le_one hd).mpr (le_max_left _ _)⟩

theorem subepsilon_length_strictly_below_one (r eps : ℝ) (he : 0 < eps) (hr : r < eps) :
    normalizedLength r eps < 1 := by
  rw [normalizedLength, max_eq_right hr.le]
  exact (div_lt_one he).mpr hr

theorem above_epsilon_length_one (r eps : ℝ) (hr : 0 < r) (he : eps ≤ r) :
    normalizedLength r eps = 1 := by simp [normalizedLength, max_eq_left he, hr.ne']

end
end N2P3.SourceAuxiliaryRisk
