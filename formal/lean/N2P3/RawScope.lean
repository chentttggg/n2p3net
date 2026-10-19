import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Finset.Range
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Fin
import Mathlib.Data.Fintype.Card
import Mathlib.Order.Interval.Set.Basic
import Mathlib.Data.Real.Basic
import Lean.Elab.Tactic.Omega
import Mathlib.Tactic.NormNum

/-!
# Finite authorized raw views (V01.a)

This file specifies the original sample and event selection in
`data.preprocess.RawSampleScope`, `crop_raw_scope`, `prefix_raw_scopes`, the
source-record prefix builders, `binary_time_split`, and the distinct continuous
right-censor route. Absolute source-sample indices are natural numbers;
seconds in the availability predicates are exact real numbers.

`extract` contains precisely the selected samples, original event rows and
authorized labels. Channel geometry, annotations, acquisition clocks,
preprocessing configuration and any pre-existing fitted state are FIXED
external inputs to `process`. They are not silently inferred from the arrays.
The processing theorem concerns any function of this extracted input with
those fixed inputs. It is not a refinement proof for MNE/Python and not the
whole-chain information-flow theorem V01.b. In particular, parsing full source
bytes, the source-file digest, physical channel admission and mutable global
state are outside the extensional input theorem.

Half-open support separation is a statement about sample positions. It does
not imply independent physiological observations, independent filtered
decisions or independence of a carried IIR state.
-/

namespace N2P3.RawScope

section Extraction

/-- Positive, in-recording raw support with explicit ordered original event IDs.
The event count is independent of the number of context samples. -/
structure Scope (nSamples nEvents nAuthorized : ℕ) where
  origin : ℕ
  first : ℕ
  stop : ℕ
  afterOrigin : origin ≤ first
  sampleNonempty : first < stop
  inRecording : stop ≤ origin + nSamples
  eventNonempty : 0 < nAuthorized
  eventRow : Fin nAuthorized → Fin nEvents
  chronological : StrictMono eventRow

/-- Original finite recording coordinates; `Event` can include the full MNE row. -/
structure SourceRecord (nSamples nEvents : ℕ) (Sample Event Label : Type*) where
  samples : Fin nSamples → Sample
  events : Fin nEvents → Event
  labels : Fin nEvents → Label

variable {nSamples nEvents nAuthorized : ℕ} {Sample Event Label : Type*}

/-- Mathematical form of the exact start/length postcondition checked after MNE crop. -/
def sampleIndex (s : Scope nSamples nEvents nAuthorized)
    (i : Fin (s.stop - s.first)) : Fin nSamples :=
  ⟨s.first - s.origin + i.val,
    by have := i.isLt; have := s.inRecording; have := s.afterOrigin; omega⟩

/-- Every selected coordinate is inside the declared half-open raw support. -/
theorem sampleIndex_in_scope (s : Scope nSamples nEvents nAuthorized)
    (i : Fin (s.stop - s.first)) :
    s.first ≤ s.origin + (sampleIndex s i).val ∧
      s.origin + (sampleIndex s i).val < s.stop := by
  have := i.isLt
  have := s.afterOrigin
  simp only [sampleIndex, Fin.val_mk]
  omega

/-- The source array uses recording-local coordinates, the declaration absolute MNE coordinates. -/
theorem sampleIndex_absolute_coordinate (s : Scope nSamples nEvents nAuthorized)
    (i : Fin (s.stop - s.first)) :
    s.origin + (sampleIndex s i).val = s.first + i.val := by
  have := s.afterOrigin
  simp only [sampleIndex, Fin.val_mk]
  omega

/-- Every in-scope original coordinate occurs, with the correct local offset. -/
theorem sampleIndex_surjective_on_scope (s : Scope nSamples nEvents nAuthorized)
    (j : Fin nSamples)
    (hj : s.first ≤ s.origin + j.val ∧ s.origin + j.val < s.stop) :
    ∃ i : Fin (s.stop - s.first), sampleIndex s i = j := by
  have := s.afterOrigin
  refine ⟨⟨s.origin + j.val - s.first, by omega⟩, ?_⟩
  apply Fin.ext
  simp only [sampleIndex, Fin.val_mk]
  omega

/-- No sample is duplicated by extraction. -/
theorem sampleIndex_injective (s : Scope nSamples nEvents nAuthorized) :
    Function.Injective (sampleIndex s) := by
  intro i j hij
  apply Fin.ext
  have h := congrArg Fin.val hij
  simp only [sampleIndex, Fin.val_mk] at h
  omega

/-- Authorized event rows remain distinct; their IDs refer to the original ledger. -/
theorem authorized_event_injective (s : Scope nSamples nEvents nAuthorized) :
    Function.Injective s.eventRow := s.chronological.injective

/-- Parent/child authorization has independent numeric and event conditions. -/
def Refines {parentCount childCount : ℕ}
    (child : Scope nSamples nEvents childCount)
    (parent : Scope nSamples nEvents parentCount) : Prop :=
  child.origin = parent.origin ∧ parent.first ≤ child.first ∧ child.stop ≤ parent.stop ∧
    ∀ i, ∃ j, child.eventRow i = parent.eventRow j

theorem nested_raw_support_stays_inside_parent {parentCount childCount : ℕ}
    (child : Scope nSamples nEvents childCount)
    (parent : Scope nSamples nEvents parentCount) (h : Refines child parent) :
    Set.Ico child.first child.stop ⊆ Set.Ico parent.first parent.stop := by
  intro j hj
  exact ⟨le_trans h.2.1 hj.1, lt_of_lt_of_le hj.2 h.2.2.1⟩

theorem refinement_transitive {smallCount middleCount largeCount : ℕ}
    (small : Scope nSamples nEvents smallCount)
    (middle : Scope nSamples nEvents middleCount)
    (large : Scope nSamples nEvents largeCount)
    (hsm : Refines small middle) (hml : Refines middle large) : Refines small large := by
  refine ⟨hsm.1.trans hml.1, le_trans hml.2.1 hsm.2.1,
    le_trans hsm.2.2.1 hml.2.2.1, ?_⟩
  intro i
  obtain ⟨j, hj⟩ := hsm.2.2.2 i
  obtain ⟨k, hk⟩ := hml.2.2.2 j
  exact ⟨k, hj.trans hk⟩

def extract (s : Scope nSamples nEvents nAuthorized)
    (r : SourceRecord nSamples nEvents Sample Event Label) :
    (Fin (s.stop - s.first) → Sample) ×
      (Fin nAuthorized → Event) × (Fin nAuthorized → Label) :=
  (fun i => r.samples (sampleIndex s i),
    (fun i => r.events (s.eventRow i)), fun i => r.labels (s.eventRow i))

/-- Actual selected label count is the explicit event budget, never the raw length. -/
theorem extracted_label_at (s : Scope nSamples nEvents nAuthorized)
    (r : SourceRecord nSamples nEvents Sample Event Label) (i : Fin nAuthorized) :
    (extract s r).2.2 i = r.labels (s.eventRow i) := rfl

/-- After boundary/artifact purge, retained evidence still uses an original
authorized row. The selection map is into the extracted ledger, not the source ledger. -/
theorem retained_evidence_is_authorized {nRetained : ℕ}
    (s : Scope nSamples nEvents nAuthorized) (retained : Fin nRetained → Fin nAuthorized)
    (i : Fin nRetained) : ∃ j, s.eventRow (retained i) = s.eventRow j :=
  ⟨retained i, rfl⟩

/-- Purge can remove evidence, but cannot silently enlarge the authorized K. -/
theorem retained_evidence_count_le_authorized {nRetained : ℕ}
    (retained : Fin nRetained → Fin nAuthorized) (h : Function.Injective retained) :
    nRetained ≤ nAuthorized := by
  simpa only [Fintype.card_fin] using Fintype.card_le_of_injective retained h

/-- The scheduled-to-retained map composes with original row IDs exactly. -/
theorem retained_original_event_ids_injective {nRetained : ℕ}
    (s : Scope nSamples nEvents nAuthorized) (retained : Fin nRetained → Fin nAuthorized)
    (h : Function.Injective retained) : Function.Injective (s.eventRow ∘ retained) :=
  s.chronological.injective.comp h

/-- Equality is required only on authorized sample/event/label coordinates.
This proves equality of the constructed input, rather than assuming it. -/
theorem extracted_input_noninterference (s : Scope nSamples nEvents nAuthorized)
    (a b : SourceRecord nSamples nEvents Sample Event Label)
    (hSamples : ∀ j, s.first ≤ s.origin + j.val →
      s.origin + j.val < s.stop → a.samples j = b.samples j)
    (hEvents : ∀ i, a.events (s.eventRow i) = b.events (s.eventRow i))
    (hLabels : ∀ i, a.labels (s.eventRow i) = b.labels (s.eventRow i)) :
    extract s a = extract s b := by
  apply Prod.ext
  · funext i
    exact hSamples _ (sampleIndex_in_scope s i).1 (sampleIndex_in_scope s i).2
  · apply Prod.ext <;> funext i
    · exact hEvents i
    · exact hLabels i

/-- Fixed metadata/configuration plus the isolated arrays suffice for any
deterministic processor. Any further read by a real caller is a separate obligation. -/
theorem isolated_processing_input_noninterference
    {Metadata Result : Type*} (metadata : Metadata)
    (s : Scope nSamples nEvents nAuthorized)
    (process : Metadata → ((Fin (s.stop - s.first) → Sample) ×
      (Fin nAuthorized → Event) × (Fin nAuthorized → Label)) → Result)
    (a b : SourceRecord nSamples nEvents Sample Event Label)
    (hSamples : ∀ j, s.first ≤ s.origin + j.val →
      s.origin + j.val < s.stop → a.samples j = b.samples j)
    (hEvents : ∀ i, a.events (s.eventRow i) = b.events (s.eventRow i))
    (hLabels : ∀ i, a.labels (s.eventRow i) = b.labels (s.eventRow i)) :
    process metadata (extract s a) = process metadata (extract s b) := by
  rw [extracted_input_noninterference s a b hSamples hEvents hLabels]

/-- Even an onset inside raw context does not grant its target label.
The unselected original row may change while the extracted input stays equal. -/
theorem tail_context_does_not_authorize_label :
    ∃ (s : Scope 2 2 1) (a b : SourceRecord 2 2 ℕ ℕ Bool),
      s.first = 0 ∧ s.stop = 2 ∧
      (∀ i, (s.eventRow i).val = 0) ∧
      a.labels 1 ≠ b.labels 1 ∧ extract s a = extract s b := by
  let s : Scope 2 2 1 := ⟨0, 0, 2, by decide, by decide, by decide, by decide,
    fun _ => 0, by intro i j h; have hi := i.isLt; have hj := j.isLt; omega⟩
  let a : SourceRecord 2 2 ℕ ℕ Bool := ⟨fun i => i.val, fun i => i.val, fun _ => false⟩
  let b : SourceRecord 2 2 ℕ ℕ Bool :=
    ⟨fun i => i.val, fun i => i.val, fun i => if i.val = 1 then true else false⟩
  refine ⟨s, a, b, rfl, rfl, (by intro i; rfl), ?_, ?_⟩
  · decide
  · apply extracted_input_noninterference
    · intro j _ _; rfl
    · intro i; rfl
    · intro i; rfl

end Extraction

section Prefixes

/-- Prefix positions in the authorized parent's ordered event ledger. -/
def prefixRows (n k : ℕ) : Finset (Fin n) := Finset.univ.filter (fun i => i.val < k)

@[simp] theorem mem_prefixRows {n k : ℕ} (i : Fin n) :
    i ∈ prefixRows n k ↔ i.val < k := by simp [prefixRows]

/-- Source stops are already on the MNE grid, with its closed endpoint converted
to an exclusive stop by `+ 1` before taking this maximum. -/
def prefixEnd {n : ℕ} (stops : Fin n → ℕ) (k : ℕ) : ℕ :=
  (prefixRows n k).sup stops

/-- The original closed-end source epoch is represented exactly, including its last sample. -/
theorem closed_epoch_grid_to_half_open (first last : ℕ) :
    Set.Icc first last = Set.Ico first (last + 1) := by
  ext i
  simp only [Set.mem_Icc, Set.mem_Ico]
  omega

@[simp] theorem prefixRows_zero (n : ℕ) : prefixRows n 0 = ∅ := by
  ext i
  simp

theorem prefixEnd_covers_each {n k : ℕ} (stops : Fin n → ℕ) (i : Fin n)
    (hi : i.val < k) : stops i ≤ prefixEnd stops k :=
  Finset.le_sup (mem_prefixRows i |>.mpr hi)

theorem prefixEnd_mono {n k maximum : ℕ} (stops : Fin n → ℕ)
    (hk : k ≤ maximum) : prefixEnd stops k ≤ prefixEnd stops maximum := by
  apply Finset.sup_mono
  intro i hi
  simp only [mem_prefixRows] at hi ⊢
  omega

/-- Lower a child's prefix position into the parent's authorized event ledger. -/
def prefixPosition {n k : ℕ} (hk : k ≤ n) (i : Fin k) : Fin n :=
  ⟨i.val, lt_of_lt_of_le i.isLt hk⟩

/-- Lower a suffix position, preserving each original requested event after Kmax. -/
def suffixPosition {n maximum : ℕ} (hm : maximum ≤ n)
    (i : Fin (n - maximum)) : Fin n :=
  ⟨maximum + i.val, by have := i.isLt; omega⟩

theorem prefixPosition_strictMono {n k : ℕ} (hk : k ≤ n) :
    StrictMono (prefixPosition hk) := by
  intro i j hij
  exact hij

theorem suffixPosition_strictMono {n maximum : ℕ} (hm : maximum ≤ n) :
    StrictMono (suffixPosition hm) := by
  intro i j hij
  change maximum + i.val < maximum + j.val
  exact Nat.add_lt_add_left hij maximum

theorem prefix_suffix_original_ids_disjoint {n k maximum : ℕ}
    (hk : k ≤ maximum) (hm : maximum ≤ n) (i : Fin k) (j : Fin (n - maximum)) :
    prefixPosition (le_trans hk hm) i ≠ suffixPosition hm j := by
  intro heq
  have hval := congrArg Fin.val heq
  have := i.isLt
  simp only [prefixPosition, suffixPosition, Fin.val_mk] at hval
  omega

/-- The parent's explicit ordered event indices are composed with the prefix,
not replaced with fresh global row aliases. Numeric endpoint validation is the
same admission obligation as the Python constructor's source-range check. -/
def prefixScope {nSamples nEvents count k : ℕ}
    (parent : Scope nSamples nEvents count) (stops : Fin count → ℕ)
    (hk : k ≤ count) (hkpos : 0 < k)
    (hfirst : parent.first < prefixEnd stops k)
    (hend : prefixEnd stops k ≤ parent.stop) : Scope nSamples nEvents k where
  origin := parent.origin
  first := parent.first
  stop := prefixEnd stops k
  afterOrigin := parent.afterOrigin
  sampleNonempty := hfirst
  inRecording := le_trans hend parent.inRecording
  eventNonempty := hkpos
  eventRow := fun i => parent.eventRow (prefixPosition hk i)
  chronological := parent.chronological.comp (prefixPosition_strictMono hk)

theorem prefixScope_refines_parent {nSamples nEvents count k : ℕ}
    (parent : Scope nSamples nEvents count) (stops : Fin count → ℕ)
    (hk : k ≤ count) (hkpos : 0 < k)
    (hfirst : parent.first < prefixEnd stops k)
    (hend : prefixEnd stops k ≤ parent.stop) :
    Refines (prefixScope parent stops hk hkpos hfirst hend) parent := by
  exact ⟨rfl, le_rfl, hend, fun i => ⟨prefixPosition hk i, rfl⟩⟩

def heldoutScope {nSamples nEvents count maximum : ℕ}
    (parent : Scope nSamples nEvents count) (stops : Fin count → ℕ)
    (hm : maximum < count)
    (hfirst : parent.first ≤ prefixEnd stops maximum)
    (hend : prefixEnd stops maximum < parent.stop) :
    Scope nSamples nEvents (count - maximum) where
  origin := parent.origin
  first := prefixEnd stops maximum
  stop := parent.stop
  afterOrigin := le_trans parent.afterOrigin hfirst
  sampleNonempty := hend
  inRecording := parent.inRecording
  eventNonempty := by omega
  eventRow := fun i => parent.eventRow (suffixPosition (Nat.le_of_lt hm) i)
  chronological := parent.chronological.comp (suffixPosition_strictMono (Nat.le_of_lt hm))

theorem heldoutScope_refines_parent {nSamples nEvents count maximum : ℕ}
    (parent : Scope nSamples nEvents count) (stops : Fin count → ℕ)
    (hm : maximum < count)
    (hfirst : parent.first ≤ prefixEnd stops maximum)
    (hend : prefixEnd stops maximum < parent.stop) :
    Refines (heldoutScope parent stops hm hfirst hend) parent := by
  exact ⟨rfl, hfirst, le_rfl, fun i => ⟨suffixPosition (Nat.le_of_lt hm) i, rfl⟩⟩

/-- One effective Kmax fixes the complete numerical heldout scope across K.
The constructor has no per-K parameter, matching the actual builder. -/
theorem prefix_and_heldout_scope_samples_disjoint {nSamples nEvents count k maximum : ℕ}
    (parent : Scope nSamples nEvents count) (stops : Fin count → ℕ)
    (hk : k ≤ maximum) (hm : maximum < count) (hkpos : 0 < k)
    (hfirst : parent.first < prefixEnd stops k)
    (hend : prefixEnd stops maximum < parent.stop) :
    let calibration := prefixScope parent stops (le_trans hk (Nat.le_of_lt hm)) hkpos
      hfirst (le_trans (prefixEnd_mono stops hk) (Nat.le_of_lt hend))
    let heldout := heldoutScope parent stops hm
      (le_trans (Nat.le_of_lt hfirst) (prefixEnd_mono stops hk)) hend
    Disjoint (Set.Ico calibration.first calibration.stop)
      (Set.Ico heldout.first heldout.stop) := by
  dsimp [prefixScope, heldoutScope]
  apply Set.disjoint_left.mpr
  intro j hj ht
  have hmono := prefixEnd_mono stops hk
  have hleft := hj.2
  have hright := ht.1
  omega

theorem prefix_and_heldout_original_event_ids_disjoint {nSamples nEvents count k maximum : ℕ}
    (parent : Scope nSamples nEvents count) (hk : k ≤ maximum) (hm : maximum ≤ count)
    (i : Fin k) (j : Fin (count - maximum)) :
    parent.eventRow (prefixPosition (le_trans hk hm) i) ≠
      parent.eventRow (suffixPosition hm j) := by
  intro heq
  exact prefix_suffix_original_ids_disjoint hk hm i j (parent.chronological.injective heq)

/-- K0 has no calibration sample endpoint or target fit labels in this model. -/
@[simp] theorem prefixEnd_zero {n : ℕ} (stops : Fin n → ℕ) :
    prefixEnd stops 0 = 0 := by simp [prefixEnd, prefixRows]

/-- The common heldout raw boundary depends on Kmax, not on individual K. -/
theorem all_prefix_samples_before_common_heldout {n k maximum : ℕ}
    (stops : Fin n → ℕ) (hk : k ≤ maximum) (j : ℕ)
    (hj : j < prefixEnd stops k) : j < prefixEnd stops maximum :=
  lt_of_lt_of_le hj (prefixEnd_mono stops hk)

/-- Epoch offsets may put a requested start before sample zero; signed endpoints
preserve that request for the purge ledger instead of saturating subtraction. -/
def EpochAccepted (maximum boundary endSample row : ℕ) (epochStart epochStop : ℤ) : Prop :=
  maximum ≤ row ∧ (boundary : ℤ) ≤ epochStart ∧ epochStop ≤ (endSample : ℤ)

def EpochPurged (maximum boundary endSample row : ℕ) (epochStart epochStop : ℤ) : Prop :=
  maximum ≤ row ∧ ¬EpochAccepted maximum boundary endSample row epochStart epochStop

theorem later_request_accepted_or_purged
    (maximum boundary endSample row : ℕ) (epochStart epochStop : ℤ) (hlater : maximum ≤ row) :
    EpochAccepted maximum boundary endSample row epochStart epochStop ∨
      EpochPurged maximum boundary endSample row epochStart epochStop := by
  by_cases h : EpochAccepted maximum boundary endSample row epochStart epochStop
  · exact Or.inl h
  · exact Or.inr ⟨hlater, h⟩

theorem accepted_not_purged
    (maximum boundary endSample row : ℕ) (epochStart epochStop : ℤ)
    (h : EpochAccepted maximum boundary endSample row epochStart epochStop) :
    ¬EpochPurged maximum boundary endSample row epochStart epochStop := by
  intro hp
  exact hp.2 h

theorem accepted_outside_every_calibration_prefix
    (k maximum boundary endSample row : ℕ) (epochStart epochStop : ℤ) (hk : k ≤ maximum)
    (h : EpochAccepted maximum boundary endSample row epochStart epochStop) : ¬row < k := by
  have := h.1
  omega

/-- Full epoch support, not just its onset, must fit the isolated test raw. -/
theorem accepted_epoch_support_subset
    (maximum boundary endSample row : ℕ) (epochStart epochStop : ℤ)
    (h : EpochAccepted maximum boundary endSample row epochStart epochStop) :
    Set.Ico epochStart epochStop ⊆ Set.Ico (boundary : ℤ) (endSample : ℤ) := by
  intro j hj
  exact ⟨le_trans h.2.1 hj.1, lt_of_lt_of_le hj.2 h.2.2⟩

theorem prefix_and_accepted_epoch_disjoint {n k maximum : ℕ}
    (stops : Fin n → ℕ) (hk : k ≤ maximum) (first endSample row : ℕ) (epochStart epochStop : ℤ)
    (h : EpochAccepted maximum (prefixEnd stops maximum) endSample row epochStart epochStop) :
    Disjoint (Set.Ico (first : ℤ) (prefixEnd stops k : ℤ)) (Set.Ico epochStart epochStop) := by
  apply Set.disjoint_left.mpr
  intro j hj ht
  have hbound := prefixEnd_mono stops hk
  have hstart := h.2.1
  have hleft := hj.2
  have hright := ht.1
  omega

/-- A request at or after Kmax may still be purged if its epoch crosses raw scope. -/
theorem later_onset_can_still_be_purged :
    EpochPurged 2 10 100 2 9 20 := by
  norm_num [EpochPurged, EpochAccepted]

/-- Signed source-grid requests before sample zero remain rejected requests. -/
theorem negative_epoch_start_is_purged : EpochPurged 1 0 100 1 (-1) 5 := by
  norm_num [EpochPurged, EpochAccepted]

/-- Raw-context authorization is weaker than excluding future stimuli. A legal
epoch-prefix tail may contain the next flash, even though that flash's label is absent. -/
theorem epoch_tail_can_contain_unauthorized_next_flash :
    ∃ (onsets stops : Fin 2 → ℕ),
      onsets 0 < onsets 1 ∧ onsets 1 < prefixEnd stops 1 ∧
      (1 : Fin 2) ∉ prefixRows 2 1 := by
  refine ⟨fun i => if i.val = 0 then 0 else 1, fun _ => 2, ?_, ?_, ?_⟩
  · decide
  · have h := prefixEnd_covers_each (fun _ : Fin 2 => 2) 0 (by decide : (0 : Fin 2).val < 1)
    change 2 ≤ prefixEnd (fun _ : Fin 2 => 2) 1 at h
    change 1 < prefixEnd (fun _ : Fin 2 => 2) 1
    omega
  · simp

end Prefixes

section Availability

/-- Conservative real-time bound before adapter-specific QC/clock/delivery maxima. -/
def offlineAvailable (nominalEndpoint sourceEndpoint recordingEnd : ℝ) : ℝ :=
  max (max nominalEndpoint sourceEndpoint) recordingEnd

theorem offlineAvailable_not_before_any_dependency (nominal source recordingEnd : ℝ) :
    nominal ≤ offlineAvailable nominal source recordingEnd ∧
      source ≤ offlineAvailable nominal source recordingEnd ∧
      recordingEnd ≤ offlineAvailable nominal source recordingEnd := by
  exact ⟨le_trans (le_max_left _ _) (le_max_left _ _),
    le_trans (le_max_right _ _) (le_max_left _ _), le_max_right _ _⟩

/-- A complete-recording reverse pass cannot be presented as available at an
earlier nominal epoch endpoint under this declared conservative policy. -/
theorem offlineAvailable_strictly_after_early_nominal (nominal source recordingEnd : ℝ)
    (h : nominal < recordingEnd) : nominal < offlineAvailable nominal source recordingEnd :=
  lt_of_lt_of_le h (offlineAvailable_not_before_any_dependency _ _ _).2.2

/-- `ready` contains exactly the first Kmax events; hence the boundary is an
actual finite maximum, not an unproved bound on training availability. -/
def maximumAvailability {m : ℕ} (ready : Fin (m + 1) → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty ready

theorem each_availability_le_maximum {m : ℕ} (ready : Fin (m + 1) → ℝ)
    (i : Fin (m + 1)) : ready i ≤ maximumAvailability ready :=
  Finset.le_sup' ready (Finset.mem_univ i)

def TimeAccepted (maximum row : ℕ) (onset tmin boundary : ℝ) : Prop :=
  maximum ≤ row ∧ boundary ≤ onset + tmin

/-- Legal temporal purge is based on actual evidence availability, not onset.
It establishes support ordering and does not assert stochastic independence. -/
theorem timeAccepted_after_all_calibration_availability {m : ℕ}
    (ready : Fin (m + 1) → ℝ) (i : Fin (m + 1)) (row : ℕ) (onset tmin : ℝ)
    (h : TimeAccepted (m + 1) row onset tmin (maximumAvailability ready)) :
    ready i ≤ onset + tmin :=
  le_trans (each_availability_le_maximum ready i) h.2

theorem timeAccepted_excludes_every_smaller_prefix
    (k maximum row : ℕ) (onset tmin boundary : ℝ) (hk : k ≤ maximum)
    (h : TimeAccepted maximum row onset tmin boundary) : ¬row < k := by
  have := h.1
  omega

/-- Merely requiring test onset after training onset is insufficient. -/
theorem onset_order_does_not_imply_availability_order :
    (0 : ℝ) < 1 ∧ ¬TimeAccepted 1 1 1 (-1 / 5) 2 := by
  norm_num [TimeAccepted]

end Availability

section Censoring

/-- The continuous route adds a stricter sample constraint than epoch tails. -/
def censoredStop (authorizedStop nextUnauthorizedOnset : ℕ) : ℕ :=
  min authorizedStop nextUnauthorizedOnset

theorem censor_never_extends_authorized_raw (authorizedStop nextOnset : ℕ) :
    censoredStop authorizedStop nextOnset ≤ authorizedStop := min_le_left _ _

theorem censor_excludes_next_flash_and_future (authorizedStop nextOnset sample : ℕ)
    (h : sample < censoredStop authorizedStop nextOnset) : sample < nextOnset :=
  lt_of_lt_of_le h (min_le_right _ _)

/-- The implementation admits only retained onsets strictly before the stop.
An onset equal to stop is algebraically a zero intersection but is rejected. -/
theorem retained_onset_strictly_before_censored_stop
    (authorizedStop nextOnset onset : ℕ)
    (hauthorized : onset < authorizedStop) (hchronological : onset < nextOnset) :
    onset < censoredStop authorizedStop nextOnset := lt_min hauthorized hchronological

/-- The effective kernel row count is the overlap with observed samples. -/
def observedResponseSamples (width stop onset : ℕ) : ℕ := min width (stop - onset)

/-- Valid when the retained flash onset is inside the selected raw scope. -/
theorem observed_response_matches_intersection (width stop onset : ℕ) (honset : onset ≤ stop) :
    onset + observedResponseSamples width stop onset = min (onset + width) stop := by
  unfold observedResponseSamples
  omega

theorem right_censored_iff_observation_short (width stop onset : ℕ) (honset : onset ≤ stop) :
    observedResponseSamples width stop onset < width ↔ stop < onset + width := by
  unfold observedResponseSamples
  omega

/-- Right censoring changes observation rows; it leaves all K authorized labels.
No response value outside the observed interval is manufactured as evidence. -/
def rightCensoredInput {Sample : Type*} {n k : ℕ}
    (samples : ℕ → Sample) (labels : Fin n → Bool) (selected : Fin k → Fin n)
    (authorizedStop nextOnset : ℕ) :
    (Fin (censoredStop authorizedStop nextOnset) → Sample) × (Fin k → Bool) :=
  (fun i => samples i.val, fun i => labels (selected i))

theorem right_censor_retains_label_selection {n k : ℕ}
    {Sample : Type*} (samples : ℕ → Sample)
    (labels : Fin n → Bool) (selected : Fin k → Fin n)
    (authorizedStop nextOnset : ℕ) :
    (rightCensoredInput samples labels selected authorizedStop nextOnset).2 =
      (fun i : Fin k => labels (selected i)) := by
  rfl

/-- Completing an unobserved tail with zeros can alter a fit statistic. -/
theorem zero_completion_is_not_observed_mean :
    (2 : ℝ) / 1 ≠ (2 + 0) / 2 := by norm_num

/-- Cropping after a full-recording operation can still read an unauthorized
future coordinate, even when the displayed output has only one position. -/
theorem filter_then_crop_can_read_outside_scope :
    ∃ (x y : Fin 2 → ℕ), x 0 = y 0 ∧
      (x 0 + x 1) ≠ (y 0 + y 1) := by
  refine ⟨fun _ => 0, fun i => if i.val = 0 then 0 else 1, rfl, ?_⟩
  decide

end Censoring

end N2P3.RawScope
