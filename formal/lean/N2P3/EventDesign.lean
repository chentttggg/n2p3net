import N2P3.RawScope
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.List.FinRange
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.Ring

/-!
# The actual per-event additive ERP design

The operational definition is the source loop: start with zeros and, for each
provided event in list order, add the retained basis slice to the common block
and label times that slice to the differential block. A finite-sum formula is
proved from this fold; it is not supplied as a source-correctness premise.

Onsets here are natural coordinates in the validated record. The caller's
nonnegative/in-record onset gate is a separate contract: the private Python
helper does not itself validate arbitrary negative or out-of-record onsets.
All arithmetic is exact real. No NumPy refinement or permutation-invariant
floating-point accumulation is asserted. The finite dictionary is a declared
observation model, not a claim that scope-wise zero-phase filtering is local.
-/

namespace N2P3.EventDesign

open Matrix
open scoped BigOperators Matrix

noncomputable section

abbrev Basis (width rank : ℕ) := Matrix (Fin width) (Fin rank) ℝ
abbrev Columns (rank : ℕ) := Fin rank ⊕ Fin rank
abbrev Design (samples rank : ℕ) := Matrix (Fin samples) (Columns rank) ℝ

variable {width rank samples : ℕ} {E : Type*}

def retained (width samples onset : ℕ) : ℕ := min width (samples - onset)

/-- The source uses contiguous common columns followed by differential columns. -/
def sourceColumn : Columns rank ≃ Fin (rank + rank) := finSumFinEquiv

theorem source_common_column (j : Fin rank) :
    (sourceColumn (Sum.inl j)).val = j.val := rfl

theorem source_differential_column (j : Fin rank) :
    (sourceColumn (Sum.inr j)).val = rank + j.val := rfl

def sourceArray (D : Design samples rank) : Matrix (Fin samples) (Fin (rank + rank)) ℝ :=
  fun t j => D t (sourceColumn.symm j)

theorem sourceArray_column (D : Design samples rank) (t : Fin samples) (j : Columns rank) :
    sourceArray D t (sourceColumn j) = D t j := by simp [sourceArray]

/-- Exact source slice bounds: guard the onset before taking the natural lag. -/
def responseAt (B : Basis width rank) (onset : ℕ) (t : Fin samples) (j : Fin rank) : ℝ :=
  if h : onset ≤ t.val ∧ t.val < onset + retained width samples onset then
    B ⟨t.val - onset, by
      have hw := Nat.min_le_left width (samples - onset)
      unfold retained at h
      omega⟩ j
  else 0

/-- The finite response kernel at an absolute coordinate, with the same guard. -/
def kernelAt (B : Basis width rank) (onset sample : ℕ) (j : Fin rank) : ℝ :=
  if h : onset ≤ sample ∧ sample < onset + width then
    B ⟨sample - onset, by omega⟩ j
  else 0

theorem retained_endpoint (onset : ℕ) (h : onset ≤ samples) :
    onset + retained width samples onset = min (onset + width) samples :=
  RawScope.observed_response_matches_intersection width samples onset h

theorem retained_complete (onset : ℕ) (h : onset + width ≤ samples) :
    retained width samples onset = width := by unfold retained; omega

theorem responseAt_eq_kernelAt (B : Basis width rank) (onset : ℕ)
    (t : Fin samples) (j : Fin rank) : responseAt B onset t j = kernelAt B onset t.val j := by
  have ht := t.isLt
  have heq : (onset ≤ t.val ∧ t.val < onset + retained width samples onset) ↔
      (onset ≤ t.val ∧ t.val < onset + width) := by unfold retained; omega
  unfold responseAt kernelAt
  split_ifs with h h' h'
  · rfl
  · exact False.elim (h' (heq.mp h))
  · exact False.elim (h (heq.mpr h'))
  · rfl

theorem kernelAt_zero_before (B : Basis width rank) (onset sample : ℕ) (j : Fin rank)
    (h : sample < onset) : kernelAt B onset sample j = 0 := by
  simp [kernelAt, Nat.not_le.mpr h]

theorem kernelAt_zero_after (B : Basis width rank) (onset sample : ℕ) (j : Fin rank)
    (h : onset + width ≤ sample) : kernelAt B onset sample j = 0 := by
  simp [kernelAt, Nat.not_lt.mpr h]

theorem responseAt_restrict {small large : ℕ} (B : Basis width rank) (h : small ≤ large)
    (onset : ℕ) (t : Fin small) (j : Fin rank) :
    responseAt B onset (Fin.castLE h t) j = responseAt B onset t j := by
  rw [responseAt_eq_kernelAt, responseAt_eq_kernelAt]
  rfl

def observedRows (width samples onset : ℕ) : Finset (Fin samples) :=
  Finset.univ.filter (fun t => onset ≤ t.val ∧ t.val < onset + retained width samples onset)

theorem observedRows_card (onset : ℕ) (h : onset ≤ samples) :
    (observedRows width samples onset).card = retained width samples onset := by
  let embed (k : Fin (retained width samples onset)) : Fin samples :=
    ⟨onset + k.val, by
      have he := retained_endpoint (width := width) onset h
      have he' := Nat.min_le_right (onset + width) samples
      omega⟩
  have hc : (Finset.univ : Finset (Fin (retained width samples onset))).card =
      (observedRows width samples onset).card := by
    apply Finset.card_bij (fun k _ => embed k)
    · intro k _
      simp only [observedRows, Finset.mem_filter, Finset.mem_univ, true_and]
      change onset ≤ onset + k.val ∧ onset + k.val < onset + retained width samples onset
      have hk := k.isLt
      omega
    · intro a _ b _ hab
      apply Fin.ext
      have hv := congrArg Fin.val hab
      dsimp [embed] at hv
      omega
    · intro t ht
      simp only [observedRows, Finset.mem_filter, Finset.mem_univ, true_and] at ht
      refine ⟨⟨t.val - onset, by omega⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      dsimp [embed]
      omega
  simpa using hc.symm

/-- Literal `+=` on the two source slices, with untouched rows left unchanged. -/
def sliceUpdate (B : Basis width rank) (onset : ℕ) (label : ℝ)
    (initial : Design samples rank) : Design samples rank :=
  fun t col =>
    if h : onset ≤ t.val ∧ t.val < onset + retained width samples onset then
      let lag : Fin width := ⟨t.val - onset, by
        have hw := Nat.min_le_left width (samples - onset)
        unfold retained at h
        omega⟩
      match col with
      | .inl j => initial t (.inl j) + B lag j
      | .inr j => initial t (.inr j) + label * B lag j
    else initial t col

def eventTerm (B : Basis width rank) (onset : ℕ) (label : ℝ) : Design samples rank :=
  fun t col => match col with
    | .inl j => responseAt B onset t j
    | .inr j => label * responseAt B onset t j

theorem sliceUpdate_eq_add (B : Basis width rank) (onset : ℕ) (label : ℝ)
    (initial : Design samples rank) :
    sliceUpdate B onset label initial = initial + eventTerm B onset label := by
  ext t col
  by_cases h : onset ≤ t.val ∧ t.val < onset + retained width samples onset
  · cases col <;> simp [sliceUpdate, eventTerm, responseAt, h]
  · cases col <;> simp [sliceUpdate, eventTerm, responseAt, h]

/-- This fold is the operational event loop, including overlap accumulation. -/
def loopFrom (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) (initial : Design samples rank) : Design samples rank :=
  events.foldl (fun acc e => sliceUpdate B (onsets e) (labels e) acc) initial

def design (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) : Design samples rank := loopFrom B onsets labels events 0

/-- The sum is derived from the actual sequential update on arbitrary lists. -/
theorem loopFrom_eq_initial_add_sum (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) (initial : Design samples rank) :
    loopFrom B onsets labels events initial =
      initial + (events.map (fun e => eventTerm B (onsets e) (labels e))).sum := by
  induction events generalizing initial with
  | nil => simp [loopFrom]
  | cons e tail ih =>
      change loopFrom B onsets labels tail (sliceUpdate B (onsets e) (labels e) initial) = _
      rw [ih, sliceUpdate_eq_add]
      simp only [List.map_cons, List.sum_cons, add_assoc]

theorem design_eq_sum (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) :
    (design B onsets labels events : Design samples rank) =
      (events.map (fun e => eventTerm B (onsets e) (labels e))).sum := by
  rw [design, loopFrom_eq_initial_add_sum, zero_add]

theorem design_nil (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ) :
    design B onsets labels [] = (0 : Design samples rank) := by simp [design, loopFrom]

theorem design_cons (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (e : E) (events : List E) :
    (design B onsets labels (e :: events) : Design samples rank) =
      eventTerm B (onsets e) (labels e) + design B onsets labels events := by
  simp only [design_eq_sum, List.map_cons, List.sum_cons]

theorem design_append (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (left right : List E) :
    (design B onsets labels (left ++ right) : Design samples rank) =
      design B onsets labels left + design B onsets labels right := by
  simp only [design_eq_sum, List.map_append, List.sum_append]

def common (B : Basis width rank) (onsets : E → ℕ) (events : List E)
    (t : Fin samples) (j : Fin rank) : ℝ :=
  (events.map (fun e => responseAt B (onsets e) t j)).sum

def differential (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ) (events : List E)
    (t : Fin samples) (j : Fin rank) : ℝ :=
  (events.map (fun e => labels e * responseAt B (onsets e) t j)).sum

theorem design_common (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) (t : Fin samples) (j : Fin rank) :
    design B onsets labels events t (.inl j) = common B onsets events t j := by
  induction events with
  | nil => simp [design_nil, common]
  | cons e tail ih => simp [design_cons, eventTerm, common, ih]

theorem design_differential (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) (t : Fin samples) (j : Fin rank) :
    design B onsets labels events t (.inr j) = differential B onsets labels events t j := by
  induction events with
  | nil => simp [design_nil, differential]
  | cons e tail ih => simp [design_cons, eventTerm, differential, ih]

theorem differential_zero (B : Basis width rank) (onsets : E → ℕ) (events : List E)
    (t : Fin samples) (j : Fin rank) : differential B onsets (fun _ => 0) events t j = 0 := by
  simp [differential]

theorem differential_add (B : Basis width rank) (onsets : E → ℕ) (a b : E → ℝ)
    (events : List E) (t : Fin samples) (j : Fin rank) :
    differential B onsets (fun e => a e + b e) events t j =
      differential B onsets a events t j + differential B onsets b events t j := by
  induction events with
  | nil => simp [differential]
  | cons e tail ih =>
      simp only [differential, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]
      ring

theorem differential_smul (B : Basis width rank) (onsets : E → ℕ) (a : E → ℝ) (scale : ℝ)
    (events : List E) (t : Fin samples) (j : Fin rank) :
    differential B onsets (fun e => scale * a e) events t j =
      scale * differential B onsets a events t j := by
  induction events with
  | nil => simp [differential]
  | cons e tail ih =>
      simp only [differential, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]
      ring

theorem differential_sub (B : Basis width rank) (onsets : E → ℕ) (a b : E → ℝ)
    (events : List E) (t : Fin samples) (j : Fin rank) :
    differential B onsets (fun e => a e - b e) events t j =
      differential B onsets a events t j - differential B onsets b events t j := by
  have h := differential_add B onsets a (fun e => -1 * b e) events t j
  rw [differential_smul] at h
  simpa only [neg_one_mul, sub_eq_add_neg] using h

def labelPart (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) : Design samples rank :=
  fun t col => match col with | .inl _ => 0 | .inr j => differential B onsets labels events t j

theorem design_sub_design (B : Basis width rank) (onsets : E → ℕ) (a b : E → ℝ)
    (events : List E) :
    (design B onsets a events : Design samples rank) - design B onsets b events =
      labelPart B onsets (fun e => a e - b e) events := by
  ext t col
  cases col <;> simp [labelPart, design_common, design_differential, differential_sub]

/-- The complete two-block map is affine: its common block is not duplicated. -/
theorem design_affine_add (B : Basis width rank) (onsets : E → ℕ) (a b : E → ℝ)
    (events : List E) :
    (design B onsets (fun e => a e + b e) events : Design samples rank) =
      design B onsets a events + design B onsets b events - design B onsets (fun _ => 0) events := by
  ext t col
  cases col <;> simp [design_common, design_differential, differential_add, differential_zero]

/-- Arbitrary real affine combinations preserve the shared common-response block. -/
theorem design_affine_combination (B : Basis width rank) (onsets : E → ℕ)
    (a b : E → ℝ) (left right : ℝ) (hscale : left + right = 1) (events : List E) :
    (design B onsets (fun e => left * a e + right * b e) events : Design samples rank) =
      left • design B onsets a events + right • design B onsets b events := by
  ext t col
  cases col with
  | inl j =>
      change design B onsets (fun e => left * a e + right * b e) events t (.inl j) =
        left * design B onsets a events t (.inl j) + right * design B onsets b events t (.inl j)
      simp only [design_common]
      rw [← add_mul, hscale, one_mul]
  | inr j =>
      change design B onsets (fun e => left * a e + right * b e) events t (.inr j) =
        left * design B onsets a events t (.inr j) + right * design B onsets b events t (.inr j)
      simp only [design_differential, differential_add, differential_smul]

/-- Cropping the observed rows does not append fake zero-response observations. -/
theorem design_restrict {small large : ℕ} (B : Basis width rank) (h : small ≤ large)
    (onsets : E → ℕ) (labels : E → ℝ) (events : List E) :
    (fun t : Fin small => design (samples := large) B onsets labels events (Fin.castLE h t)) =
      design (samples := small) B onsets labels events := by
  ext t col
  cases col <;> simp only [design_common, design_differential, common, differential]
  · congr 1
    apply List.map_congr_left
    intro e _
    exact responseAt_restrict B h (onsets e) t _
  · congr 1
    apply List.map_congr_left
    intro e _
    rw [responseAt_restrict B h]

theorem eventTerm_mul_coeff {C : Type*} (B : Basis width rank) (onset : ℕ) (label : ℝ)
    (coeff : Matrix (Columns rank) C ℝ) (t : Fin samples) (channel : C) :
    (eventTerm B onset label * coeff) t channel =
      ∑ j : Fin rank, responseAt B onset t j * (coeff (.inl j) channel + label * coeff (.inr j) channel) := by
  simp only [Matrix.mul_apply, Fintype.sum_sum_type, eventTerm]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Coefficient interpretation is derived from the loop, including all overlapping events. -/
theorem design_mul_coeff {C : Type*} (B : Basis width rank) (onsets : E → ℕ) (labels : E → ℝ)
    (events : List E) (coeff : Matrix (Columns rank) C ℝ) (t : Fin samples) (channel : C) :
    (design B onsets labels events * coeff) t channel =
      (events.map (fun e => ∑ j : Fin rank, responseAt B (onsets e) t j *
        (coeff (.inl j) channel + labels e * coeff (.inr j) channel))).sum := by
  induction events with
  | nil => simp [design_nil]
  | cons e tail ih =>
      rw [design_cons, Matrix.add_mul, Matrix.add_apply, eventTerm_mul_coeff, ih]
      rfl

end

end N2P3.EventDesign
