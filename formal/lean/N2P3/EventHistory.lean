import N2P3.CandidateCoverage
import N2P3.EventDesign
import Mathlib.Data.List.OfFn
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Recover complete public event history from the actual admission ingredients

BNCI decoding admits ten repetitions with every physical flash exactly once;
the raw caller intersects a selection with eligible rows, then requires exact
physical counts and checks the preceding/following selection boundaries.

Derive equality of observed and planned EVENT IDENTITIES from subset plus
per-flash counts. Completeness of history is not an input conclusion. A
repetition permutation is the exact mathematical content of the decoder's
sort-and-compare test, not a proof of NumPy/MAT decoding. Temporal order and
neighbor checks then derive zero contribution from other selections under the
declared finite causal response dictionary. This does not prove that the
whole-test zero-phase filter is local or that decision noise is independent.
-/

namespace N2P3.EventHistory

open scoped BigOperators
open N2P3.CandidateCoverage

noncomputable section

section RecoverIdentities

variable {E F : Type*} [DecidableEq E] [DecidableEq F]

def histogram (events : Finset E) (code : E → F) (flash : F) : ℕ :=
  (events.filter (fun e => code e = flash)).card

omit [DecidableEq E] in
/-- Equal physical counts cannot hide a missing identity when observed events
are actually selected from the same planned event set. -/
theorem recover_event_identities (planned observed : Finset E) (code : E → F)
    (hsubset : observed ⊆ planned)
    (hcounts : ∀ f, histogram observed code f = histogram planned code f) :
    observed = planned := by
  apply Finset.Subset.antisymm hsubset
  intro e he
  have hs := Finset.filter_subset_filter (fun j => code j = code e) hsubset
  have hf : observed.filter (fun j => code j = code e) =
      planned.filter (fun j => code j = code e) := by
    apply Finset.eq_of_subset_of_card_le hs
    exact (hcounts (code e)).symm.le
  have hm : e ∈ observed.filter (fun j => code j = code e) := by
    rw [hf]
    exact Finset.mem_filter.mpr ⟨he, rfl⟩
  exact (Finset.mem_filter.mp hm).1

end RecoverIdentities

abbrev Repetition := Fin 10
abbrev Slot := Fin 12
abbrev Event (selections : ℕ) := Fin selections × (Repetition × Slot)

def flashEquiv : Flash ≃ Slot := finSumFinEquiv

theorem flashEquiv_matches_source (f : Flash) : (flashEquiv f).val = physicalCode f := by
  cases f <;> rfl

def RepetitionAccepted (code : Slot → Flash) : Prop :=
  (List.ofFn code).Perm (Finset.univ : Finset Flash).toList

theorem accepted_repetition_bijective (code : Slot → Flash) (h : RepetitionAccepted code) :
    Function.Bijective code := by
  constructor
  · apply List.nodup_ofFn.mp
    exact h.nodup_iff.mpr (Finset.nodup_toList _)
  · intro f
    have hm : f ∈ List.ofFn code := h.mem_iff.mpr (by simp)
    simpa only [List.mem_ofFn] using hm

theorem accepted_repetition_count_one (code : Slot → Flash) (h : RepetitionAccepted code)
    (f : Flash) : (∑ slot, if code slot = f then (1 : ℕ) else 0) = 1 := by
  obtain ⟨slot, hs⟩ := (accepted_repetition_bijective code h).2 f
  have he (j : Slot) : code j = f ↔ j = slot := by
    rw [← hs]
    exact (accepted_repetition_bijective code h).1.eq_iff
  simp_rw [he]
  simp

variable {s : ℕ}

def selectionEvents (selection : Fin s) : Finset (Event s) :=
  Finset.univ.filter (fun e => e.1 = selection)

def observedEvents (selection : Fin s) (eligible : Finset (Event s)) : Finset (Event s) :=
  (selectionEvents selection).filter (fun e => e ∈ eligible)

theorem observed_subset_selection (selection : Fin s) (eligible : Finset (Event s)) :
    observedEvents selection eligible ⊆ selectionEvents selection := Finset.filter_subset _ _

theorem selection_histogram_ten (selection : Fin s) (code : Event s → Flash)
    (hplan : ∀ r, RepetitionAccepted (fun slot => code (selection, r, slot))) (f : Flash) :
    histogram (selectionEvents selection) code f = 10 := by
  simp only [histogram, Finset.card_eq_sum_ones, selectionEvents, Finset.sum_filter,
    Fintype.sum_prod_type]
  have hi (sel : Fin s) :
      (∑ r : Repetition, ∑ slot : Slot,
        if sel = selection then if code (sel, r, slot) = f then (1 : ℕ) else 0 else 0) =
      if sel = selection then 10 else 0 := by
    by_cases hs : sel = selection
    · subst sel
      simp only [if_true]
      simp_rw [accepted_repetition_count_one _ (hplan _) f]
      simp
    · simp [hs]
  simp_rw [hi]
  simp

def coverageFor (selection : Fin s) (eligible : Finset (Event s)) (code : Event s → Flash) :
    Flash → Option ℕ := fun f => some (histogram (observedEvents selection eligible) code f)

/-- Counts come from the proper physical schedule before applying the raw
caller subset selection. This proves the missing-history conclusion. -/
theorem complete_coverage_recovers_selection (selection : Fin s)
    (eligible : Finset (Event s)) (code : Event s → Flash)
    (hplan : ∀ r, RepetitionAccepted (fun slot => code (selection, r, slot)))
    (hcomplete : Complete (fun _ : Flash => 10) (coverageFor selection eligible code)) :
    observedEvents selection eligible = selectionEvents selection := by
  apply recover_event_identities _ _ code (observed_subset_selection selection eligible)
  intro f
  have hc := (complete_iff_exact_counts _ _).mp hcomplete f
  change histogram (observedEvents selection eligible) code f = 10 at hc
  rw [hc, selection_histogram_ten selection code hplan f]

def firstEvent (selection : Fin s) : Event s := (selection, 0, 0)
def lastEvent (selection : Fin s) : Event s := (selection, 9, 11)

def eventIndex (e : Event s) : ℕ := 120 * e.1.val + 12 * e.2.1.val + e.2.2.val

def OrderedOnsets (onset : Event s → ℤ) : Prop :=
  ∀ e f, eventIndex e ≤ eventIndex f → onset e ≤ onset f

def geometry (selection : Fin s) (onset : Event s → ℤ)
    (width preroll testStart testStop : ℤ) : Geometry where
  first := onset (firstEvent selection)
  last := onset (lastEvent selection)
  width := width
  preroll := preroll
  testStart := testStart
  testStop := testStop
  previousLast := if hp : 0 < selection.val then
      some (onset (lastEvent ⟨selection.val - 1, by have := selection.isLt; omega⟩))
    else none
  nextFirst := if hn : selection.val + 1 < s then
      some (onset (firstEvent ⟨selection.val + 1, hn⟩)) else none

theorem other_selection_onset_separated (selection : Fin s) (onset : Event s → ℤ)
    (width preroll testStart testStop : ℤ) (hordered : OrderedOnsets onset)
    (coverage : Flash → Option ℕ)
    (hok : (geometry selection onset width preroll testStart testStop).reason coverage = none)
    (e : Event s) (he : e ∉ selectionEvents selection) :
    onset e + width ≤ (geometry selection onset width preroll testStart testStop).start ∨
      (geometry selection onset width preroll testStart testStop).stop ≤ onset e := by
  let g := geometry selection onset width preroll testStart testStop
  have hne : e.1 ≠ selection := by simpa [selectionEvents] using he
  have hval : e.1.val ≠ selection.val := fun h => hne (Fin.ext h)
  have hs := (Geometry.success_iff g coverage).mp hok
  rcases lt_or_gt_of_ne hval with hbefore | hafter
  · have hp : 0 < selection.val := by omega
    let previous : Fin s := ⟨selection.val - 1, by have := selection.isLt; omega⟩
    have hprev : g.previousLast = some (onset (lastEvent previous)) := by
      simp [g, geometry, hp, previous]
    have hbound := Geometry.no_previous_overlap_orders_response g _ hprev hs.2.2.2.1
    change onset (lastEvent previous) + width ≤ g.start at hbound
    have hindex : eventIndex e ≤ eventIndex (lastEvent previous) := by
      have hr := e.2.1.isLt
      have ht := e.2.2.isLt
      simp only [eventIndex, lastEvent, previous]
      omega
    have ho := hordered e (lastEvent previous) hindex
    exact Or.inl (by change onset e + width ≤ g.start; linarith)
  · have hn : selection.val + 1 < s := by have := e.1.isLt; omega
    let next : Fin s := ⟨selection.val + 1, hn⟩
    have hnext : g.nextFirst = some (onset (firstEvent next)) := by
      simp [g, geometry, hn, next]
    have hbound := Geometry.no_next_overlap_orders_response g _ hnext hs.2.2.2.2
    have hindex : eventIndex (firstEvent next) ≤ eventIndex e := by
      simp only [eventIndex, firstEvent, next]
      omega
    have ho := hordered (firstEvent next) e hindex
    exact Or.inr (by change g.stop ≤ onset e; linarith)

/-- The declared finite causal observed-response model. Its compact support
is a model definition; no locality of the earlier raw IIR filter is inferred. -/
def causalContribution (width onset sample : ℤ) (response : ℤ → ℝ) : ℝ :=
  if onset ≤ sample ∧ sample < onset + width then response (sample - onset) else 0

theorem outside_response_zero (width onset sample start stop : ℤ) (response : ℤ → ℝ)
    (hwindow : start ≤ sample ∧ sample < stop)
    (hsep : onset + width ≤ start ∨ stop ≤ onset) :
    causalContribution width onset sample response = 0 := by
  unfold causalContribution
  have hnot : ¬(onset ≤ sample ∧ sample < onset + width) := by
    rcases hsep with h | h <;> omega
  exact if_neg hnot

theorem accepted_view_response_uses_all_and_only_observed_events
    (selection : Fin s) (eligible : Finset (Event s)) (code : Event s → Flash)
    (hplan : ∀ r, RepetitionAccepted (fun slot => code (selection, r, slot)))
    (onset : Event s → ℤ) (hordered : OrderedOnsets onset)
    (width preroll testStart testStop sample : ℤ) (response : Event s → ℤ → ℝ)
    (hok : (geometry selection onset width preroll testStart testStop).reason
      (coverageFor selection eligible code) = none)
    (hwindow : (geometry selection onset width preroll testStart testStop).start ≤ sample ∧
      sample < (geometry selection onset width preroll testStart testStop).stop) :
    (∑ e : Event s, causalContribution width (onset e) sample (response e)) =
      ∑ e ∈ observedEvents selection eligible,
        causalContribution width (onset e) sample (response e) := by
  have hcomplete := ((Geometry.success_iff _ _).mp hok).1
  rw [complete_coverage_recovers_selection selection eligible code hplan hcomplete]
  symm
  apply Finset.sum_subset (Finset.subset_univ _)
  intro e _ he
  apply outside_response_zero width (onset e) sample _ _ (response e) hwindow
  exact other_selection_onset_separated selection onset width preroll testStart testStop
    hordered _ hok e he

/-! Connect the admission result to the operational EventDesign fold. Absolute
onsets are natural source coordinates; only observed events are passed to the
local slice helper. Other selections occur only in the global guarded kernel.
The two coordinate equalities below describe the actual [start, stop) slice,
not an assumption that its event design is correct. -/

theorem selection_onset_bounds (selection : Fin s) (onset : Event s → ℤ)
    (hordered : OrderedOnsets onset) (e : Event s) (he : e ∈ selectionEvents selection) :
    onset (firstEvent selection) ≤ onset e ∧ onset e ≤ onset (lastEvent selection) := by
  have hsel : e.1 = selection := by simpa [selectionEvents] using he
  have hr := e.2.1.isLt
  have ht := e.2.2.isLt
  constructor
  · apply hordered
    simp only [eventIndex, firstEvent, hsel]
    omega
  · apply hordered
    simp only [eventIndex, lastEvent, hsel]
    omega

theorem successful_start_nonnegative (selection : Fin s) (onset : Event s → ℤ)
    (width preroll testStart testStop : ℤ) (coverage : Flash → Option ℕ)
    (htest : 0 ≤ testStart)
    (hok : (geometry selection onset width preroll testStart testStop).reason coverage = none) :
    0 ≤ (geometry selection onset width preroll testStart testStop).start := by
  exact le_trans htest ((Geometry.success_iff _ _).mp hok).2.1

theorem observed_onsets_inside_slice (selection : Fin s) (eligible : Finset (Event s))
    (onsets : Event s → ℕ) (hordered : OrderedOnsets (fun e => (onsets e : ℤ)))
    (width start samples : ℕ) (preroll testStart testStop : ℤ)
    (hwidth : 0 < width) (hpreroll : 0 ≤ preroll)
    (hstart : (start : ℤ) =
      (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).start)
    (hstop : ((start + samples : ℕ) : ℤ) =
      (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).stop)
    (e : Event s) (he : e ∈ observedEvents selection eligible) :
    start ≤ onsets e ∧ onsets e < start + samples := by
  have hb := selection_onset_bounds selection (fun e => (onsets e : ℤ)) hordered e
    (observed_subset_selection selection eligible he)
  dsimp only at hb
  simp only [geometry, Geometry.start, Geometry.stop] at hstart hstop
  omega

theorem kernelAt_shift {width rank : ℕ} (B : EventDesign.Basis width rank)
    (onset start sample : ℕ) (j : Fin rank) (hstart : start ≤ onset) :
    EventDesign.kernelAt B (onset - start) sample j =
      EventDesign.kernelAt B onset (start + sample) j := by
  have heq : (onset - start ≤ sample ∧ sample < onset - start + width) ↔
      (onset ≤ start + sample ∧ start + sample < onset + width) := by omega
  unfold EventDesign.kernelAt
  split_ifs with h h' h'
  · congr 1
    apply Fin.ext
    change sample - (onset - start) = start + sample - onset
    omega
  · exact False.elim (h' (heq.mp h))
  · exact False.elim (h (heq.mpr h'))
  · rfl

theorem accepted_global_kernel_sum {width rank : ℕ}
    (B : EventDesign.Basis width rank) (selection : Fin s)
    (eligible : Finset (Event s)) (code : Event s → Flash)
    (hplan : ∀ r, RepetitionAccepted (fun slot => code (selection, r, slot)))
    (onsets : Event s → ℕ) (hordered : OrderedOnsets (fun e => (onsets e : ℤ)))
    (preroll testStart testStop : ℤ) (sample : ℕ) (j : Fin rank) (weights : Event s → ℝ)
    (hok : (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).reason
      (coverageFor selection eligible code) = none)
    (hwindow : (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).start ≤
      (sample : ℤ) ∧ (sample : ℤ) <
        (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).stop) :
    (∑ e : Event s, weights e * EventDesign.kernelAt B (onsets e) sample j) =
      ∑ e ∈ observedEvents selection eligible,
        weights e * EventDesign.kernelAt B (onsets e) sample j := by
  have hcomplete := ((Geometry.success_iff _ _).mp hok).1
  rw [complete_coverage_recovers_selection selection eligible code hplan hcomplete]
  symm
  apply Finset.sum_subset (Finset.subset_univ _)
  intro e _ he
  have hs := other_selection_onset_separated selection (fun e => (onsets e : ℤ))
    width preroll testStart testStop hordered _ hok e he
  dsimp only at hs
  have hz : EventDesign.kernelAt B (onsets e) sample j = 0 := by
    rcases hs with hs | hs
    · apply EventDesign.kernelAt_zero_after
      omega
    · apply EventDesign.kernelAt_zero_before
      omega
  rw [hz, mul_zero]

/-- Every column of the actual source += fold equals the corresponding complete
world finite-response sum on the admitted raw slice. Local helper onsets are
proved in range, rather than inserting unsupported future/negative onsets. -/
theorem accepted_source_design_equals_complete_history {width rank samples : ℕ}
    (B : EventDesign.Basis width rank) (selection : Fin s)
    (eligible : Finset (Event s)) (code : Event s → Flash)
    (hplan : ∀ r, RepetitionAccepted (fun slot => code (selection, r, slot)))
    (onsets : Event s → ℕ) (hordered : OrderedOnsets (fun e => (onsets e : ℤ)))
    (labels : Event s → ℝ) (start : ℕ) (preroll testStart testStop : ℤ)
    (hwidth : 0 < width) (hpreroll : 0 ≤ preroll)
    (hstart : (start : ℤ) =
      (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).start)
    (hstop : ((start + samples : ℕ) : ℤ) =
      (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).stop)
    (hok : (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).reason
      (coverageFor selection eligible code) = none)
    (t : Fin samples) (col : EventDesign.Columns rank) :
    EventDesign.design B (fun e => onsets e - start) labels
      (observedEvents selection eligible).toList t col =
      ∑ e : Event s, (match col with | .inl _ => 1 | .inr _ => labels e) *
        EventDesign.kernelAt B (onsets e) (start + t.val) (col.elim id id) := by
  have ht := t.isLt
  have hw : (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).start ≤
      ((start + t.val : ℕ) : ℤ) ∧ ((start + t.val : ℕ) : ℤ) <
        (geometry selection (fun e => (onsets e : ℤ)) width preroll testStart testStop).stop := by
    omega
  have hshift (e : Event s) (he : e ∈ observedEvents selection eligible) (j : Fin rank) :
      EventDesign.responseAt B (onsets e - start) t j =
        EventDesign.kernelAt B (onsets e) (start + t.val) j := by
    rw [EventDesign.responseAt_eq_kernelAt]
    exact kernelAt_shift B (onsets e) start t.val j
      (observed_onsets_inside_slice selection eligible onsets hordered width start samples
        preroll testStart testStop hwidth hpreroll hstart hstop e he).1
  cases col with
  | inl j =>
      rw [EventDesign.design_common]
      simp only [EventDesign.common, Finset.sum_map_toList, one_mul, Sum.elim_inl, id_eq]
      have hsum := accepted_global_kernel_sum B selection eligible code hplan onsets hordered
        preroll testStart testStop (start + t.val) j (fun _ => 1) hok hw
      simp only [one_mul] at hsum
      rw [hsum]
      apply Finset.sum_congr rfl
      intro e he
      exact hshift e he j
  | inr j =>
      rw [EventDesign.design_differential]
      simp only [EventDesign.differential, Finset.sum_map_toList, Sum.elim_inr, id_eq]
      rw [accepted_global_kernel_sum B selection eligible code hplan onsets hordered
        preroll testStart testStop (start + t.val) j labels hok hw]
      apply Finset.sum_congr rfl
      intro e he
      rw [hshift e he j]

/-- The source retains events in onset order. Any duplicate-free enumeration
of exactly the observed identities has the same exact-real loop result. -/
theorem design_of_observed_enumeration {width rank samples : ℕ}
    (B : EventDesign.Basis width rank) (onsets : Event s → ℕ) (labels : Event s → ℝ)
    (selection : Fin s) (eligible : Finset (Event s)) (events : List (Event s))
    (hnodup : events.Nodup) (hids : events.toFinset = observedEvents selection eligible) :
    (EventDesign.design B onsets labels events : EventDesign.Design samples rank) =
      EventDesign.design B onsets labels (observedEvents selection eligible).toList := by
  have hp := List.perm_of_nodup_nodup_toFinset_eq hnodup
    (Finset.nodup_toList (observedEvents selection eligible)) (by simpa using hids)
  simp only [EventDesign.design_eq_sum]
  exact (hp.map _).sum_eq

/-- A complete decoded permutation schedule, independent of response values. -/
def canonicalCode (e : Event s) : Flash := flashEquiv.symm e.2.2

theorem canonical_repetition_accepted (selection : Fin s) (r : Repetition) :
    RepetitionAccepted (fun slot => canonicalCode (selection, r, slot)) := by
  apply List.perm_of_nodup_nodup_toFinset_eq
  · apply List.nodup_ofFn.mpr
    exact flashEquiv.symm.injective
  · exact Finset.nodup_toList _
  · ext f
    simp only [List.mem_toFinset, List.mem_ofFn, Finset.mem_toList, Finset.mem_univ, iff_true]
    exact ⟨flashEquiv f, flashEquiv.symm_apply_apply f⟩

theorem canonical_full_coverage (selection : Fin s) :
    Complete (fun _ : Flash => 10) (coverageFor selection Finset.univ canonicalCode) := by
  rw [complete_iff_exact_counts]
  intro f
  change histogram (observedEvents selection Finset.univ) canonicalCode f = 10
  simp only [observedEvents, Finset.mem_univ, Finset.filter_true]
  exact selection_histogram_ten selection canonicalCode (canonical_repetition_accepted selection) f

/-- A nonempty three-selection witness satisfies BOTH adjacent support gates,
including equality at the half-open boundaries; completeness is derived. -/
theorem canonical_middle_view_admitted :
    (geometry (1 : Fin 3) (fun e => (2 + eventIndex e : ℕ)) 1 0 0 400).reason
      (coverageFor (1 : Fin 3) Finset.univ canonicalCode) = none := by
  rw [Geometry.success_iff]
  refine ⟨canonical_full_coverage _, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [geometry, Geometry.start, Geometry.stop, Geometry.previousOverlap,
      Geometry.nextOverlap, firstEvent, lastEvent, eventIndex]

theorem canonical_onsets_ordered :
    OrderedOnsets (s := 3) (fun e => (2 + eventIndex e : ℕ)) := by
  intro e f hef
  change ((2 + eventIndex e : ℕ) : ℤ) ≤ ((2 + eventIndex f : ℕ) : ℤ)
  exact_mod_cast Nat.add_le_add_left hef 2

end

end N2P3.EventHistory
