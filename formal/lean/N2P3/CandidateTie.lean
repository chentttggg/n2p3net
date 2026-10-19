import Mathlib.Data.Finset.Max
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FinCases

/-!
# The source's first argmax and absolute-tolerance abstention

The source first takes np.argmax in frozen candidate order, then counts ALL
candidates within atol=1e-10, rtol=0 of that maximum. This file defines those
operations directly, without assuming a unique winner, a nonempty tie band,
or a source/result correspondence as a premise. Nonempty finite candidate
sets and exact real scores are explicit; IEEE NaN/overflow remain separate.
-/

namespace N2P3.CandidateTie

noncomputable section

variable {ι : Type*} [Fintype ι]

def IsMaximum (e : ι → ℝ) (b : ι) : Prop := ∀ j, e j ≤ e b

def closeCandidates (e : ι → ℝ) (b : ι) (tol : ℝ) : Finset ι :=
  Finset.univ.filter (fun j => |e j - e b| ≤ tol)

def IsTied (e : ι → ℝ) (b : ι) (tol : ℝ) : Prop :=
  1 < (closeCandidates e b tol).card

theorem closeCandidates_mem (e : ι → ℝ) (b j : ι) (tol : ℝ) :
    j ∈ closeCandidates e b tol ↔ |e j - e b| ≤ tol := by
  simp [closeCandidates]

theorem maximum_exists [Nonempty ι] (e : ι → ℝ) : ∃ b, IsMaximum e b := by
  obtain ⟨b, _, hb⟩ := Finset.exists_max_image Finset.univ e Finset.univ_nonempty
  exact ⟨b, fun j => hb j (Finset.mem_univ j)⟩

omit [Fintype ι] in
theorem maximum_values_equal (e : ι → ℝ) (a b : ι)
    (ha : IsMaximum e a) (hb : IsMaximum e b) : e a = e b :=
  le_antisymm (hb a) (ha b)

theorem self_in_closeCandidates (e : ι → ℝ) (b : ι) {tol : ℝ} (htol : 0 ≤ tol) :
    b ∈ closeCandidates e b tol := by
  simp [closeCandidates, htol]

theorem close_at_maximum_iff (e : ι → ℝ) (b j : ι) (tol : ℝ)
    (hb : IsMaximum e b) :
    j ∈ closeCandidates e b tol ↔ e b - e j ≤ tol := by
  rw [closeCandidates_mem, abs_of_nonpos (sub_nonpos.mpr (hb j))]
  constructor <;> intro h <;> linarith

theorem tied_iff_close_competitor (e : ι → ℝ) (b : ι) {tol : ℝ}
    (htol : 0 ≤ tol) (hb : IsMaximum e b) :
    IsTied e b tol ↔ ∃ j, j ≠ b ∧ e b - e j ≤ tol := by
  constructor
  · intro ht
    obtain ⟨j, hj, hne⟩ := Finset.exists_mem_ne ht b
    exact ⟨j, hne, (close_at_maximum_iff e b j tol hb).mp hj⟩
  · rintro ⟨j, hne, hj⟩
    exact Finset.one_lt_card.mpr ⟨j, (close_at_maximum_iff e b j tol hb).mpr hj,
      b, self_in_closeCandidates e b htol, hne⟩

theorem not_tied_iff_margin (e : ι → ℝ) (b : ι) {tol : ℝ}
    (htol : 0 ≤ tol) (hb : IsMaximum e b) :
    ¬ IsTied e b tol ↔ ∀ j, j ≠ b → tol < e b - e j := by
  rw [tied_iff_close_competitor e b htol hb]
  push_neg
  rfl

theorem tie_independent_of_argmax (e : ι → ℝ) (a b : ι) (tol : ℝ)
    (ha : IsMaximum e a) (hb : IsMaximum e b) :
    IsTied e a tol ↔ IsTied e b tol := by
  unfold IsTied closeCandidates
  rw [maximum_values_equal e a b ha hb]

theorem closeCandidates_add_constant (e : ι → ℝ) (b : ι) (tol k : ℝ) :
    closeCandidates (fun j => e j + k) b tol = closeCandidates e b tol := by
  ext j
  simp only [closeCandidates_mem, add_sub_add_right_eq_sub]

omit [Fintype ι] in
theorem maximum_add_constant_iff (e : ι → ℝ) (b : ι) (k : ℝ) :
    IsMaximum (fun j => e j + k) b ↔ IsMaximum e b := by
  simp only [IsMaximum, add_le_add_iff_right]

section FirstArgmax

variable [Nonempty ι] [LinearOrder ι]

def maxima (e : ι → ℝ) : Finset ι := by
  classical
  exact Finset.univ.filter (IsMaximum e)

omit [LinearOrder ι] in
theorem maxima_nonempty (e : ι → ℝ) : (maxima e).Nonempty := by
  obtain ⟨b, hb⟩ := maximum_exists e
  exact ⟨b, by simp [maxima, hb]⟩

/-- Min is in candidate index order, matching the first occurrence argmax. -/
def firstArgmax (e : ι → ℝ) : ι := (maxima e).min' (maxima_nonempty e)

theorem firstArgmax_isMaximum (e : ι → ℝ) : IsMaximum e (firstArgmax e) := by
  have h := Finset.min'_mem (maxima e) (maxima_nonempty e)
  simpa only [maxima, Finset.mem_filter, Finset.mem_univ, true_and] using h

theorem firstArgmax_le_any_maximum (e : ι → ℝ) (j : ι) (hj : IsMaximum e j) :
    firstArgmax e ≤ j := by
  exact Finset.min'_le (maxima e) j (by simp [maxima, hj])

def winner (e : ι → ℝ) (tol : ℝ) : Option ι := by
  classical
  exact if IsTied e (firstArgmax e) tol then none else some (firstArgmax e)

theorem winner_none_iff (e : ι → ℝ) (tol : ℝ) :
    winner e tol = none ↔ IsTied e (firstArgmax e) tol := by
  classical
  simp [winner]

theorem winner_some_iff_margin (e : ι → ℝ) (b : ι) {tol : ℝ} (htol : 0 ≤ tol) :
    winner e tol = some b ↔
      IsMaximum e b ∧ ∀ j, j ≠ b → tol < e b - e j := by
  classical
  constructor
  · intro hw
    by_cases ht : IsTied e (firstArgmax e) tol
    · simp [winner, ht] at hw
    · simp only [winner, if_neg ht] at hw
      have heq := Option.some.inj hw
      subst b
      exact ⟨firstArgmax_isMaximum e,
        (not_tied_iff_margin e _ htol (firstArgmax_isMaximum e)).mp ht⟩
  · rintro ⟨hb, hm⟩
    have heq : firstArgmax e = b := by
      by_contra hne
      have hgap := hm (firstArgmax e) hne
      have hmax := firstArgmax_isMaximum e b
      linarith
    unfold winner
    rw [heq, if_neg ((not_tied_iff_margin e b htol hb).mpr hm)]

theorem firstArgmax_add_constant (e : ι → ℝ) (k : ℝ) :
    firstArgmax (fun j => e j + k) = firstArgmax e := by
  have h : maxima (fun j => e j + k) = maxima e := by
    ext j
    simp only [maxima, Finset.mem_filter, Finset.mem_univ, true_and,
      maximum_add_constant_iff]
  simp only [firstArgmax, h]

theorem winner_add_constant (e : ι → ℝ) (tol k : ℝ) :
    winner (fun j => e j + k) tol = winner e tol := by
  classical
  simp [winner, firstArgmax_add_constant, IsTied, closeCandidates_add_constant]

end FirstArgmax

def sourceTolerance : ℝ := 1 / 10000000000

theorem sourceTolerance_positive : 0 < sourceTolerance := by norm_num [sourceTolerance]

def nearTieScores : Fin 2 → ℝ := fun j => if j = 0 then 0 else 1 / 20000000000

theorem nearTie_has_strict_unique_maximum :
    ∀ j : Fin 2, j ≠ 1 → nearTieScores j < nearTieScores 1 := by
  intro j hj
  fin_cases j <;> norm_num [nearTieScores] at *

theorem nearTie_abstains : winner nearTieScores sourceTolerance = none := by
  have hb : IsMaximum nearTieScores 1 := by
    intro j
    fin_cases j <;> norm_num [nearTieScores]
  have ht : IsTied nearTieScores 1 sourceTolerance := by
    apply (tied_iff_close_competitor nearTieScores 1 sourceTolerance_positive.le hb).mpr
    exact ⟨0, by decide, by norm_num [nearTieScores, sourceTolerance]⟩
  apply (winner_none_iff nearTieScores sourceTolerance).mpr
  exact (tie_independent_of_argmax nearTieScores (firstArgmax nearTieScores) 1
    sourceTolerance (firstArgmax_isMaximum nearTieScores) hb).mpr ht

end

end N2P3.CandidateTie
