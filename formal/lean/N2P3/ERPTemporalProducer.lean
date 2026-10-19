import N2P3.ERPTemporalBasis
import N2P3.ERPState
import Mathlib.Data.List.FinRange
import Mathlib.Data.List.Nodup
import Mathlib.Data.List.Dedup
import Mathlib.Data.List.Sort
import Lean.Elab.Tactic.Omega

/-!
# Source temporal fitting: decoded fields, original events and actual gathers

Source: temporal_transfer.py:146-216; epochs.py:1158-1164,1550-1561.
This is the P12.C/D/E/F/G exact-real connection layer. The input contains
independently decoded X, integer labels, subject IDs, channel names, global and
trial masks, timeline evidence indices, event IDs and per-subject identity.
No whole dataset-admission, genuine fit or receipt-matches-samples oracle is
an input. P12.A/B/K/L remain separate: complete EpochDataset subgates, CLI
scope preparation, physical source authority and actual cache sample relation
are not obtained from these decoded fields.

The event program models every assignment to arbitrary initial np.empty
storage. The literal sorted-evidence equality yields complete writes and the
correct original event for every input row. Participant rows retain request
order; common channels are recomputed locally for each participant. Both
integer class means are built from gathered coordinates before being related
to ERPTemporalBasis.classContrast. Real formulas do not assert NumPy/BLAS
rounding, finite casts, byte encodings or numerical eigh correctness.

The receipt's three hash inputs preserve distinct dtype/shape/value images.
Contiguous byte encoding, Python dtype/shape header encoding and SHA-256 are
small representation boundaries, with no hash injectivity or authenticity
axiom. The parent field is the STORED load identity; it is deliberately not a
fresh decoded-content assertion. The independently executed no-fit cache
probe is finite behavior evidence only, not a fit_plan failure proof.
-/

namespace N2P3.ERPTemporalProducer

open Matrix N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open scoped BigOperators

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Dataset (N C T E : ℕ) where
  values : Fin N → Fin C → Fin T → ℝ
  labels : Fin N → ℤ
  labelDtype : String
  nativeFloat64Dtype : String
  subjects : Fin N → String
  channelNames : Fin C → String
  channelMask : Fin C → Bool
  trialMask : Option (Fin N → Fin C → Bool)
  evidence : Fin E → ℤ
  eventIds : Fin E → String
  identity : String → IdentityImage
  identityPayload : String → Value
  storedParent : Option Value

variable {N C T E : ℕ}

def selectedRows (d : Dataset N C T E) (requested : List (Fin N)) (subject : String) :=
  requested.filter (fun row => d.subjects row = subject)

def participantSubjects (d : Dataset N C T E) (requested : List (Fin N)) : List String :=
  ((requested.map d.subjects).dedup).mergeSort (· ≤ ·)

theorem selected_row_membership (d : Dataset N C T E) (rows : List (Fin N))
    (subject : String) (row : Fin N) :
    row ∈ selectedRows d rows subject ↔ row ∈ rows ∧ d.subjects row = subject := by
  simp [selectedRows]

theorem participant_subject_membership (d : Dataset N C T E) (rows : List (Fin N))
    (subject : String) :
    subject ∈ participantSubjects d rows ↔ ∃ row ∈ rows, d.subjects row = subject := by
  rw [participantSubjects, (List.mergeSort_perm _ _).mem_iff]
  simp

theorem participant_subjects_nodup (d : Dataset N C T E) (rows : List (Fin N)) :
    (participantSubjects d rows).Nodup := by
  exact List.nodup_mergeSort.mpr (List.nodup_dedup _)

theorem selected_preserves_request_order (d : Dataset N C T E) (rows : List (Fin N))
    (subject : String) : (selectedRows d rows subject).Sublist rows := List.filter_sublist

theorem selected_subjects_disjoint (d : Dataset N C T E) (rows : List (Fin N))
    (a b : String) (h : a ≠ b) (row : Fin N)
    (ha : row ∈ selectedRows d rows a) : row ∉ selectedRows d rows b := by
  intro hb
  exact h (((selected_row_membership _ _ _ _).mp ha).2.symm.trans
    ((selected_row_membership _ _ _ _).mp hb).2)

def effectiveObserved (d : Dataset N C T E) (row : Fin N) (channel : Fin C) : Bool :=
  d.channelMask channel && (d.trialMask.map (fun mask => mask row channel)).getD true

def commonChannels (d : Dataset N C T E) (rows : List (Fin N)) : List (Fin C) :=
  (List.finRange C).filter (fun c => rows.all (fun row => effectiveObserved d row c))

theorem common_channel_iff_every_selected_row (d : Dataset N C T E)
    (rows : List (Fin N)) (c : Fin C) :
    c ∈ commonChannels d rows ↔ ∀ row ∈ rows, effectiveObserved d row c = true := by
  simp [commonChannels]

theorem common_channels_are_local (d : Dataset N C T E) (rows : List (Fin N))
    (a : String) (c : Fin C) :
    c ∈ commonChannels d (selectedRows d rows a) ↔
      ∀ row ∈ rows, d.subjects row = a → effectiveObserved d row c = true := by
  simp [common_channel_iff_every_selected_row, selected_row_membership]

def mappedEvents (d : Dataset N C T E) : List (Fin E) :=
  (List.finRange E).filter (fun e => @decide (0 ≤ d.evidence e) (Int.decLe _ _))

def integerLE (a b : ℤ) : Bool := @decide (a ≤ b) (Int.decLe a b)

def sortedEvidenceGate (d : Dataset N C T E) : Prop :=
  ((mappedEvents d).map d.evidence).mergeSort integerLE = (List.range N).map Int.ofNat

def writeInverse (evidence : Fin E → ℤ) :
    List (Fin E) → (Fin N → Option (Fin E)) → (Fin N → Option (Fin E))
  | [], initial => initial
  | event :: tail, initial =>
    writeInverse evidence tail (fun row =>
      if evidence event = (row.val : ℤ) then some event else initial row)

theorem inverse_untouched (evidence : Fin E → ℤ) (events : List (Fin E))
    (initial : Fin N → Option (Fin E)) (row : Fin N)
    (h : ∀ event ∈ events, evidence event ≠ (row.val : ℤ)) :
    writeInverse evidence events initial row = initial row := by
  induction events generalizing initial with
  | nil => rfl
  | cons event tail ih =>
    rw [writeInverse, ih _ (fun e he => h e (by simp [he]))]
    simp [h event (by simp)]

theorem inverse_at_written_event (evidence : Fin E → ℤ) (events : List (Fin E))
    (initial : Fin N → Option (Fin E)) (row : Fin N) (event : Fin E)
    (unique : (events.map evidence).Nodup) (mem : event ∈ events)
    (atRow : evidence event = (row.val : ℤ)) :
    writeInverse evidence events initial row = some event := by
  induction events generalizing initial with
  | nil => simp at mem
  | cons head tail ih =>
    have hu := List.nodup_cons.mp unique
    rcases List.mem_cons.mp mem with he | ht
    · subst event
      rw [writeInverse, inverse_untouched]
      · simp [atRow]
      · intro e he same
        exact hu.1 (List.mem_map.mpr ⟨e, he, same.trans atRow.symm⟩)
    · exact ih _ hu.2 ht

theorem sorted_gate_gives_evidence_permutation (d : Dataset N C T E)
    (h : sortedEvidenceGate d) :
    ((mappedEvents d).map d.evidence).Perm ((List.range N).map Int.ofNat) := by
  have hp := (List.mergeSort_perm ((mappedEvents d).map d.evidence) integerLE).symm
  rw [← h]
  exact hp

theorem sorted_gate_writes_every_row (d : Dataset N C T E)
    (h : sortedEvidenceGate d) (row : Fin N) :
    ∃ event ∈ mappedEvents d, d.evidence event = (row.val : ℤ) := by
  apply List.mem_map.mp
  apply (sorted_gate_gives_evidence_permutation d h).mem_iff.mpr
  exact List.mem_map.mpr ⟨row.val, List.mem_range.mpr row.isLt, rfl⟩

theorem sorted_gate_mapped_evidence_nodup (d : Dataset N C T E)
    (h : sortedEvidenceGate d) : ((mappedEvents d).map d.evidence).Nodup := by
  apply (sorted_gate_gives_evidence_permutation d h).nodup_iff.mpr
  exact List.nodup_range.map Int.ofNat_injective

theorem inverse_complete_from_literal_sorted_gate (d : Dataset N C T E)
    (h : sortedEvidenceGate d) (initial : Fin N → Option (Fin E)) (row : Fin N) :
    ∃ event ∈ mappedEvents d, d.evidence event = (row.val : ℤ) ∧
      writeInverse d.evidence (mappedEvents d) initial row = some event := by
  obtain ⟨event, he, hr⟩ := sorted_gate_writes_every_row d h row
  exact ⟨event, he, hr, inverse_at_written_event _ _ _ _ _
    (sorted_gate_mapped_evidence_nodup d h) he hr⟩

theorem inverse_independent_of_uninitialized_storage (d : Dataset N C T E)
    (h : sortedEvidenceGate d) (initial other : Fin N → Option (Fin E)) :
    writeInverse d.evidence (mappedEvents d) initial =
      writeInverse d.evidence (mappedEvents d) other := by
  funext row
  obtain ⟨event, he, hr⟩ := sorted_gate_writes_every_row d h row
  rw [inverse_at_written_event _ _ _ _ _ (sorted_gate_mapped_evidence_nodup d h) he hr,
    inverse_at_written_event _ _ _ _ _ (sorted_gate_mapped_evidence_nodup d h) he hr]

def gatheredX (d : Dataset N C T E) (rows : List (Fin N)) (channels : List (Fin C)) :
    Fin rows.length → Matrix (Fin channels.length) (Fin T) ℝ :=
  fun i c t => d.values (rows.get i) (channels.get c) t

def gatheredIntegerLabels (d : Dataset N C T E) (rows : List (Fin N)) :=
  fun i : Fin rows.length => d.labels (rows.get i)

def gatheredBoolLabels (d : Dataset N C T E) (rows : List (Fin N)) :=
  fun i : Fin rows.length => decide (gatheredIntegerLabels d rows i = 1)

def integerClassRows (d : Dataset N C T E) (rows : List (Fin N)) (label : ℤ) :=
  Finset.univ.filter (fun i : Fin rows.length => gatheredIntegerLabels d rows i = label)

def binaryLabelGate (d : Dataset N C T E) (rows : List (Fin N)) : Prop :=
  (∀ i, gatheredIntegerLabels d rows i = 0 ∨ gatheredIntegerLabels d rows i = 1) ∧
    2 ≤ (integerClassRows d rows 0).card ∧ 2 ≤ (integerClassRows d rows 1).card

theorem gathered_coordinate_is_original_sample (d : Dataset N C T E)
    (rows : List (Fin N)) (channels : List (Fin C))
    (i : Fin rows.length) (c : Fin channels.length) (t : Fin T) :
    gatheredX d rows channels i c t = d.values (rows.get i) (channels.get c) t := rfl

theorem gathered_label_is_original_label (d : Dataset N C T E) (rows : List (Fin N))
    (i : Fin rows.length) : gatheredIntegerLabels d rows i = d.labels (rows.get i) := rfl

theorem bool_one_rows_equal_integer_one (d : Dataset N C T E) (rows : List (Fin N)) :
    ERPTemporalBasis.classRows (gatheredBoolLabels d rows) true = integerClassRows d rows 1 := by
  ext i
  simp [ERPTemporalBasis.classRows, gatheredBoolLabels, integerClassRows]

theorem bool_zero_rows_equal_integer_zero (d : Dataset N C T E) (rows : List (Fin N))
    (h : ∀ i, gatheredIntegerLabels d rows i = 0 ∨ gatheredIntegerLabels d rows i = 1) :
    ERPTemporalBasis.classRows (gatheredBoolLabels d rows) false = integerClassRows d rows 0 := by
  ext i
  rcases h i with hi | hi <;>
    simp [ERPTemporalBasis.classRows, gatheredBoolLabels, integerClassRows, hi]

def sourceClassMean (d : Dataset N C T E) (rows : List (Fin N))
    (channels : List (Fin C)) (label : ℤ) : Matrix (Fin channels.length) (Fin T) ℝ :=
  fun c t => (∑ i ∈ integerClassRows d rows label, gatheredX d rows channels i c t) /
    ((integerClassRows d rows label).card : ℝ)

def sourceContrast (d : Dataset N C T E) (rows : List (Fin N))
    (channels : List (Fin C)) : Matrix (Fin channels.length) (Fin T) ℝ :=
  sourceClassMean d rows channels 1 - sourceClassMean d rows channels 0

theorem integer_means_bridge_to_existing_class_contrast (d : Dataset N C T E)
    (rows : List (Fin N)) (channels : List (Fin C))
    (h : ∀ i, gatheredIntegerLabels d rows i = 0 ∨ gatheredIntegerLabels d rows i = 1) :
    sourceContrast d rows channels = ERPTemporalBasis.classContrast
      (gatheredX d rows channels) (gatheredBoolLabels d rows) := by
  unfold sourceContrast sourceClassMean ERPTemporalBasis.classContrast ERPTemporalBasis.classMean
  rw [bool_one_rows_equal_integer_one, bool_zero_rows_equal_integer_zero d rows h]

theorem binary_gate_supplies_two_rows_to_both_means (d : Dataset N C T E)
    (rows : List (Fin N)) (h : binaryLabelGate d rows) :
    2 ≤ (ERPTemporalBasis.classRows (gatheredBoolLabels d rows) false).card ∧
      2 ≤ (ERPTemporalBasis.classRows (gatheredBoolLabels d rows) true).card := by
  rw [bool_one_rows_equal_integer_one, bool_zero_rows_equal_integer_zero d rows h.1]
  exact h.2

def sourceGram (d : Dataset N C T E) (rows : List (Fin N))
    (channels : List (Fin C)) : Matrix (Fin T) (Fin T) ℝ :=
  (sourceContrast d rows channels)ᵀ * sourceContrast d rows channels

def sourceEnergy (d : Dataset N C T E) (rows : List (Fin N)) (channels : List (Fin C)) : ℝ :=
  (sourceGram d rows channels).trace

def sourceNormalized (d : Dataset N C T E) (rows : List (Fin N))
    (channels : List (Fin C)) : Matrix (Fin T) (Fin T) ℝ :=
  fun t u => sourceGram d rows channels t u / sourceEnergy d rows channels

theorem literal_gram_division_equals_existing_normalized_gram (d : Dataset N C T E)
    (rows : List (Fin N)) (channels : List (Fin C)) :
    sourceNormalized d rows channels = ERPTemporalBasis.normalizedGram (sourceContrast d rows channels) := by
  ext t u
  rw [ERPTemporalBasis.normalized_gram_coordinate]
  rfl

theorem selected_fields_to_normalized_gram (d : Dataset N C T E)
    (rows : List (Fin N)) (channels : List (Fin C))
    (h : ∀ i, gatheredIntegerLabels d rows i = 0 ∨ gatheredIntegerLabels d rows i = 1) :
    sourceNormalized d rows channels = ERPTemporalBasis.normalizedGram
      (ERPTemporalBasis.classContrast (gatheredX d rows channels) (gatheredBoolLabels d rows)) := by
  rw [literal_gram_division_equals_existing_normalized_gram,
    integer_means_bridge_to_existing_class_contrast d rows channels h]

theorem positive_energy_yields_psd_and_trace_one (d : Dataset N C T E)
    (rows : List (Fin N)) (channels : List (Fin C))
    (energy : 0 < sourceEnergy d rows channels) :
    (sourceNormalized d rows channels).PosSemidef ∧ (sourceNormalized d rows channels).trace = 1 := by
  rw [literal_gram_division_equals_existing_normalized_gram]
  exact ⟨ERPTemporalBasis.normalized_gram_posSemidef _ energy,
    ERPTemporalBasis.normalized_gram_trace _ energy⟩

structure NumericImage where
  dtype : String
  shape : List ℕ
  values : List Value

structure NumericHashPrimitives where
  dtypeShapeHeader : String → List ℕ → List ℕ
  contiguousBytes : NumericImage → List ℕ
  sha256 : List ℕ → String

def numericHash (p : NumericHashPrimitives) (image : NumericImage) : String :=
  p.sha256 (p.dtypeShapeHeader image.dtype image.shape ++ p.contiguousBytes image)

def valuesImage (d : Dataset N C T E) (rows : List (Fin N)) (channels : List (Fin C)) : NumericImage :=
  ⟨d.nativeFloat64Dtype, [rows.length, channels.length, T],
    rows.flatMap (fun row => channels.flatMap (fun c => (List.ofFn (d.values row c)).map Value.number))⟩

def labelsImage (d : Dataset N C T E) (rows : List (Fin N)) : NumericImage :=
  ⟨d.labelDtype, [rows.length], rows.map (fun row => .integer (d.labels row))⟩

def observedImage (d : Dataset N C T E) (rows : List (Fin N)) : NumericImage :=
  ⟨"|b1", [rows.length, C], rows.flatMap (fun row =>
    (List.ofFn (effectiveObserved d row)).map Value.boolean)⟩

def originalEventIds (d : Dataset N C T E) (initial : Fin N → Option (Fin E))
    (rows : List (Fin N)) : List String :=
  rows.map (fun row => ((writeInverse d.evidence (mappedEvents d) initial row).map d.eventIds).getD "<unwritten>")

def receipt (p : NumericHashPrimitives) (d : Dataset N C T E)
    (requested : List (Fin N)) (subject : String) (initial : Fin N → Option (Fin E)) : Fields :=
  let rows := selectedRows d requested subject
  let channels := commonChannels d rows
  let id := d.identity subject
  Fields.ofList [
    ("participant_key", .text id.authority),
    ("origin_keys", .sequence (encodeTexts id.origins)),
    ("global_keys", .sequence (encodeTexts id.globals)),
    ("rows", .sequence (encodeNats (rows.map Fin.val))),
    ("event_ids", .sequence (encodeTexts (originalEventIds d initial rows))),
    ("observed_channels", .sequence (encodeTexts (channels.map d.channelNames))),
    ("values_sha256", .text (numericHash p (valuesImage d rows channels))),
    ("labels_sha256", .text (numericHash p (labelsImage d rows))),
    ("observed_sha256", .text (numericHash p (observedImage d rows))),
    ("parent_cache_attestation", d.storedParent.getD .null),
    ("identity", d.identityPayload subject)]

theorem receipt_rows_follow_selected_input_order (p : NumericHashPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (subject : String)
    (initial : Fin N → Option (Fin E)) :
    (receipt p d requested subject initial).get "rows" =
      some (.sequence (encodeNats ((selectedRows d requested subject).map Fin.val))) := by
  simp [receipt, Fields.get, Fields.ofList]

theorem receipt_parent_is_stored_identity (p : NumericHashPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (subject : String)
    (initial : Fin N → Option (Fin E)) :
    (receipt p d requested subject initial).get "parent_cache_attestation" =
      some (d.storedParent.getD .null) := by
  simp [receipt, Fields.get, Fields.ofList]

theorem observed_hash_image_retains_full_channel_width (d : Dataset N C T E)
    (rows : List (Fin N)) : (observedImage d rows).shape = [rows.length, C] := rfl

theorem values_hash_image_has_only_local_common_channels (d : Dataset N C T E)
    (rows : List (Fin N)) :
    (valuesImage d rows (commonChannels d rows)).shape = [rows.length, (commonChannels d rows).length, T] := rfl

theorem labels_hash_retains_original_dtype (d : Dataset N C T E) (rows : List (Fin N)) :
    (labelsImage d rows).dtype = d.labelDtype := rfl

theorem original_event_names_come_from_complete_writes (d : Dataset N C T E)
    (h : sortedEvidenceGate d) (initial : Fin N → Option (Fin E))
    (row : Fin N) (event : Fin E) (mem : event ∈ mappedEvents d)
    (atRow : d.evidence event = (row.val : ℤ)) :
    ((writeInverse d.evidence (mappedEvents d) initial row).map d.eventIds).getD "<unwritten>" =
      d.eventIds event := by
  rw [inverse_at_written_event _ _ _ _ _ (sorted_gate_mapped_evidence_nodup d h) mem atRow]
  rfl

structure Contribution (T : ℕ) where
  authority : String
  channelCount : ℕ
  contrast : Matrix (Fin channelCount) (Fin T) ℝ
  fitReceipt : Fields
  studies : List String

def literalContribution (p : NumericHashPrimitives) (d : Dataset N C T E)
    (requested : List (Fin N)) (subject : String) (initial : Fin N → Option (Fin E)) :
    Contribution T :=
  let rows := selectedRows d requested subject
  let channels := commonChannels d rows
  ⟨(d.identity subject).authority, channels.length, sourceContrast d rows channels,
    receipt p d requested subject initial, (d.identity subject).studies⟩

/-- Only this layer's explicit local guards occur here. Dataset validation,
shared preprocessing/rank, finite IEEE operations and physical identity are
separate obligations, not success callbacks hidden in this program. -/
def buildContribution (p : NumericHashPrimitives) (d : Dataset N C T E)
    (requested : List (Fin N)) (subject : String) (initial : Fin N → Option (Fin E))
    (seen : List String) : Check (Contribution T) := do
  ensure ((d.identity subject).authority ∉ seen) "source temporal duplicate authority"
  let rows := selectedRows d requested subject
  ensure (binaryLabelGate d rows) "two observations per binary class"
  let channels := commonChannels d rows
  ensure (channels ≠ []) "source temporal local observed channels"
  ensure (0 < sourceEnergy d rows channels) "source temporal positive contrast energy"
  pure (literalContribution p d requested subject initial)

theorem contribution_from_individual_field_guards (p : NumericHashPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (subject : String)
    (initial : Fin N → Option (Fin E)) (seen : List String)
    (duplicate : (d.identity subject).authority ∉ seen)
    (binary : binaryLabelGate d (selectedRows d requested subject))
    (channels : commonChannels d (selectedRows d requested subject) ≠ [])
    (energy : 0 < sourceEnergy d (selectedRows d requested subject)
      (commonChannels d (selectedRows d requested subject))) :
    buildContribution p d requested subject initial seen =
      .ok (literalContribution p d requested subject initial) := by
  simp [buildContribution, ensure, duplicate, binary, channels, energy]

theorem duplicate_authority_refused_before_numeric_work (p : NumericHashPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (subject : String)
    (initial : Fin N → Option (Fin E)) (seen : List String)
    (duplicate : (d.identity subject).authority ∈ seen) :
    buildContribution p d requested subject initial seen =
      .error (.sourceReject "source temporal duplicate authority") := by
  simp [buildContribution, ensure, duplicate]

def collectSubjects (p : NumericHashPrimitives) (d : Dataset N C T E)
    (requested : List (Fin N)) (initial : Fin N → Option (Fin E)) :
    List String → List String → Check (List (Contribution T))
  | [], _ => .ok []
  | subject :: tail, seen => do
    let contribution ← buildContribution p d requested subject initial seen
    let rest ← collectSubjects p d requested initial tail (contribution.authority :: seen)
    pure (contribution :: rest)

theorem collected_length_is_one_per_subject (p : NumericHashPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (initial : Fin N → Option (Fin E))
    (subjects seen : List String) (result : List (Contribution T))
    (h : collectSubjects p d requested initial subjects seen = .ok result) :
    result.length = subjects.length := by
  induction subjects generalizing seen result with
  | nil =>
    have eq : [] = result := by simpa [collectSubjects] using h
    subst result
    rfl
  | cons subject tail ih =>
    cases hb : buildContribution p d requested subject initial seen with
    | error err => simp [collectSubjects, hb] at h
    | ok contribution =>
      cases hr : collectSubjects p d requested initial tail (contribution.authority :: seen) with
      | error err => simp [collectSubjects, hb, hr] at h
      | ok rest =>
        have eq : contribution :: rest = result := by simpa [collectSubjects, hb, hr] using h
        subst result
        simp [ih _ _ hr]

def participantMean (contributions : List (Contribution T)) : Matrix (Fin T) (Fin T) ℝ :=
  fun t u => (∑ p : Fin contributions.length,
    ERPTemporalBasis.gram (contributions.get p).contrast t u /
      ERPTemporalBasis.energy (contributions.get p).contrast) / (contributions.length : ℝ)

theorem actual_equal_mean_matches_participant_average (contributions : List (Contribution T)) :
    participantMean contributions = ERPTemporalBasis.participantAverage
      (fun p : Fin contributions.length => (contributions.get p).channelCount)
      (fun p => (contributions.get p).contrast) := by
  ext t u
  rw [ERPTemporalBasis.participant_average_coordinate]
  simp [participantMean, ERPTemporalBasis.gram_coordinate]

theorem collected_mean_psd_and_trace (contributions : List (Contribution T))
    (nonempty : contributions ≠ [])
    (positive : ∀ p : Fin contributions.length,
      0 < ERPTemporalBasis.energy (contributions.get p).contrast) :
    (participantMean contributions).PosSemidef ∧ (participantMean contributions).trace = 1 := by
  have hp : 0 < contributions.length := List.length_pos_iff.mpr nonempty
  letI : Nonempty (Fin contributions.length) := ⟨⟨0, hp⟩⟩
  rw [actual_equal_mean_matches_participant_average]
  exact ⟨ERPTemporalBasis.participant_average_posSemidef _ _ positive,
    ERPTemporalBasis.participant_average_trace _ _ positive⟩

/-- Individual outputs of the numerical library call. There is deliberately
no exact eigen-equation, sorting, PSD or orthogonality certificate here. -/
structure EighImage (T : ℕ) where
  values : Fin T → ℝ
  vectors : Matrix (Fin T) (Fin T) ℝ

def firstAbsPivot {T : ℕ} (positive : 0 < T) (column : Fin T → ℝ) : Fin T :=
  (List.finRange T).foldl (fun best i => if |column best| < |column i| then i else best)
    ⟨0, positive⟩

def signedColumn {T : ℕ} (positive : 0 < T) (column : Fin T → ℝ) : Fin T → ℝ :=
  let pivot := firstAbsPivot positive column
  fun i => if column pivot < 0 then -column i else column i

theorem sign_loop_makes_its_selected_pivot_nonnegative {T : ℕ} (positive : 0 < T)
    (column : Fin T → ℝ) :
    0 ≤ signedColumn positive column (firstAbsPivot positive column) := by
  simp only [signedColumn]
  split_ifs with negative
  · exact neg_nonneg.mpr negative.le
  · exact le_of_not_gt negative

def retainedIndex (rank : ℕ) (within : rank ≤ T) (column : Fin rank) : Fin T :=
  ⟨T - 1 - column.val, by have := column.isLt; omega⟩

theorem retained_columns_descend (rank : ℕ) (within : rank ≤ T) (column : Fin rank) :
    (retainedIndex rank within column).val = T - 1 - column.val := rfl

def spectralSelect (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T)
    (eigh : EighImage T) : Check (Matrix (Fin T) (Fin rank) ℝ) := do
  let cutoff : Fin T := ⟨T - rank, by omega⟩
  let top : Fin T := ⟨T - 1, by omega⟩
  ensure (¬ eigh.values cutoff ≤ eigh.values top * (1 / 10000000000 : ℝ))
    "source temporal identifiable rank"
  pure (fun i column => signedColumn (by omega)
    (fun t => eigh.vectors t (retainedIndex rank within column)) i)

theorem failed_relative_cutoff_refuses_even_with_positive_trace
    (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T) (eigh : EighImage T)
    (small : eigh.values ⟨T-rank, by omega⟩ ≤
      eigh.values ⟨T-1, by omega⟩ * (1 / 10000000000 : ℝ)) :
    spectralSelect rank positive within eigh =
      .error (.sourceReject "source temporal identifiable rank") := by
  have hl : ¬ eigh.values ⟨T-1, by omega⟩ * (10000000000 : ℝ)⁻¹ <
      eigh.values ⟨T-rank, by omega⟩ := by
    simpa only [one_div] using not_lt.mpr small
  simp [spectralSelect, ensure, hl]

theorem spectral_success_uses_reverse_slice_then_sign
    (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T) (eigh : EighImage T)
    (large : ¬ eigh.values ⟨T-rank, by omega⟩ ≤
      eigh.values ⟨T-1, by omega⟩ * (1 / 10000000000 : ℝ)) :
    spectralSelect rank positive within eigh = .ok (fun i column =>
      signedColumn (by omega) (fun t => eigh.vectors t (retainedIndex rank within column)) i) := by
  have hl : eigh.values ⟨T-1, by omega⟩ * (10000000000 : ℝ)⁻¹ <
      eigh.values ⟨T-rank, by omega⟩ := by
    simpa only [one_div] using lt_of_not_ge large
  simp [spectralSelect, ensure, hl]

def basisValue {R : ℕ} (basis : Matrix (Fin T) (Fin R) ℝ) : Value :=
  .sequence (valuesOfList (List.ofFn (fun i =>
    .sequence (encodeReals (List.ofFn (fun c => basis i c))))))

def contributionStudies (cs : List (Contribution T)) : List String :=
  (cs.flatMap Contribution.studies).dedup.mergeSort (· ≤ ·)

def sourceMetadata {R : ℕ} (preprocessing : Value) (basis : Matrix (Fin T) (Fin R) ℝ)
    (cs : List (Contribution T)) : Fields :=
  Fields.ofList [
    ("schema", .text "n2p3_temporal_basis/1"),
    ("fit_role", .text "source_supervised"),
    ("preprocessing", preprocessing), ("rank", .integer R), ("basis", basisValue basis),
    ("fit_receipts", .sequence (valuesOfList (cs.map (fun c => .object c.fitReceipt)))),
    ("source_studies", .sequence (encodeTexts (contributionStudies cs))),
    ("risk", .text "equal_participant_normalized_ERP_contrast")]

def sourceRecord {R : ℕ} (p : TemporalPrimitives) (preprocessing : Value)
    (basis : Matrix (Fin T) (Fin R) ℝ) (cs : List (Contribution T)) : Fields :=
  let metadata := sourceMetadata preprocessing basis cs
  .cons "digest" (.text (p.metadataHash metadata)) metadata

def finishFromContributions (p : TemporalPrimitives) (preprocessing : Value)
    (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T)
    (eighCall : Matrix (Fin T) (Fin T) ℝ → Check (EighImage T))
    (cs : List (Contribution T)) : Check TemporalModel := do
  ensure (cs ≠ []) "source temporal nonempty scopes"
  let eig ← eighCall (participantMean cs)
  let basis ← spectralSelect rank positive within eig
  ERPState.readTemporal p (.object (sourceRecord p preprocessing basis cs))

theorem empty_scopes_refused_before_eigh (p : TemporalPrimitives) (preprocessing : Value)
    (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T)
    (eighCall : Matrix (Fin T) (Fin T) ℝ → Check (EighImage T)) :
    finishFromContributions p preprocessing rank positive within eighCall [] =
      .error (.sourceReject "source temporal nonempty scopes") := by
  simp [finishFromContributions, ensure]

theorem numeric_library_failure_is_propagated (p : TemporalPrimitives) (preprocessing : Value)
    (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T)
    (eighCall : Matrix (Fin T) (Fin T) ℝ → Check (EighImage T))
    (cs : List (Contribution T)) (nonempty : cs ≠ []) (fault : LedgerFault)
    (failed : eighCall (participantMean cs) = .error fault) :
    finishFromContributions p preprocessing rank positive within eighCall cs = .error fault := by
  simp [finishFromContributions, ensure, nonempty, failed]

theorem source_role_and_risk_are_constructed {R : ℕ} (preprocessing : Value)
    (basis : Matrix (Fin T) (Fin R) ℝ) (cs : List (Contribution T)) :
    (sourceMetadata preprocessing basis cs).get "fit_role" = some (.text "source_supervised") ∧
    (sourceMetadata preprocessing basis cs).get "risk" =
      some (.text "equal_participant_normalized_ERP_contrast") := by
  simp [sourceMetadata, Fields.get, Fields.ofList]

theorem source_record_keeps_each_constructed_receipt {R : ℕ} (preprocessing : Value)
    (basis : Matrix (Fin T) (Fin R) ℝ) (cs : List (Contribution T)) :
    (sourceMetadata preprocessing basis cs).get "fit_receipts" =
      some (.sequence (valuesOfList (cs.map (fun c => .object c.fitReceipt)))) := by
  simp [sourceMetadata, Fields.get, Fields.ofList]

/-- This explicitly starts at the decoded scope boundary, after the separate
full P12.B dataset/row/preprocessing/rank gates. It is not the whole Python
fit_source or its multi-scope/CLI admission refinement. -/
def fitDecodedScope (hash : NumericHashPrimitives) (p : TemporalPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (initial : Fin N → Option (Fin E))
    (preprocessing : Value) (rank : ℕ) (positive : 0 < rank) (within : rank ≤ T)
    (eighCall : Matrix (Fin T) (Fin T) ℝ → Check (EighImage T)) : Check TemporalModel := do
  ensure (sortedEvidenceGate d) "source temporal physical event per row"
  let cs ← collectSubjects hash d requested initial (participantSubjects d requested) []
  finishFromContributions p preprocessing rank positive within eighCall cs

/-- Actual Python dict equality on the decoded finite JSON boundary: numeric
bool/int/float tags can compare equal and dictionary key order is irrelevant.
Reader-specific strict rank/bool guards stay in readTemporal/commonHeader;
this comparison adds no claim that arbitrary compared fields are admitted. -/
def replayRecordEqual (repeated saved : Fields) : Prop :=
  PersonalAdmission.PyEq (.object repeated) (.object saved)

theorem replay_comparison_integer_float_tags :
    replayRecordEqual (Fields.ofList [("cell", .integer 1)])
      (Fields.ofList [("cell", .number 1)]) := by
  simp [replayRecordEqual, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue,
    PersonalAdmission.normalizeFields, Fields.ofList]

theorem replay_comparison_boolean_numeric_tags :
    replayRecordEqual (Fields.ofList [("cell", .boolean true)])
      (Fields.ofList [("cell", .integer 1)]) := by
  simp [replayRecordEqual, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue,
    PersonalAdmission.normalizeFields, Fields.ofList]

theorem structural_fields_distinguish_comparison_equal_numeric_tags :
    Fields.ofList [("cell", .integer 1)] ≠ Fields.ofList [("cell", .number 1)] := by
  intro h
  cases h

theorem replay_comparison_ignores_dictionary_key_order (a b : String)
    (x y : Value) (distinct : a ≠ b) :
    replayRecordEqual (Fields.ofList [(a,x),(b,y)]) (Fields.ofList [(b,y),(a,x)]) :=
  PersonalAdmission.python_object_field_order_irrelevant a b x y distinct

/-- Replay executes the entire decoded graph again; its separate numerical
call can have a different output/failure. This adds no cross-call or platform
eigh determinism assumption. The narrow width equality is a reader field fact,
not a whole reader-success or replay-equality premise. It connects saved.rank
to the actual self.matrix.shape[1] operation on this canonical decoded domain.
That reader derivation, general iterables, CLI scopes and I/O remain J/K. -/
def verifyDecodedScope (hash : NumericHashPrimitives) (p : TemporalPrimitives)
    (d : Dataset N C T E) (requested : List (Fin N)) (initial : Fin N → Option (Fin E))
    (preprocessing : Value) (saved : TemporalModel) (_width : saved.matrix.shape = [T,saved.rank])
    (positive : 0 < saved.rank) (within : saved.rank ≤ T)
    (replayEigh : Matrix (Fin T) (Fin T) ℝ → Check (EighImage T)) : Check Unit := do
  ensure (saved.role = .source) "fixed temporal control cannot replay source fitting"
  let repeated ← fitDecodedScope hash p d requested initial preprocessing saved.rank positive within replayEigh
  ensure (replayRecordEqual repeated.record saved.record) "temporal source replay differs"

theorem fixed_control_cannot_claim_source_replay (hash : NumericHashPrimitives)
    (p : TemporalPrimitives) (d : Dataset N C T E) (requested : List (Fin N))
    (initial : Fin N → Option (Fin E)) (preprocessing : Value) (saved : TemporalModel)
    (width : saved.matrix.shape = [T,saved.rank])
    (positive : 0 < saved.rank) (within : saved.rank ≤ T)
    (replayEigh : Matrix (Fin T) (Fin T) ℝ → Check (EighImage T))
    (fixed : saved.role = .fixed) :
    verifyDecodedScope hash p d requested initial preprocessing saved width positive within replayEigh =
      .error (.sourceReject "fixed temporal control cannot replay source fitting") := by
  simp [verifyDecodedScope, ensure, fixed]

/-! Fixed dimensionless exact-real witness using the preparation's independent
row/event/montage design, before source-fit execution. This is not an admitted
physical EpochDataset: its illustrative identity payload is not a parsed
participant payload. Common offsets are nonzero in both classes; signed
subtraction is checked before Gram, which cannot detect a global sign flip. -/

def fixture : Dataset 8 3 3 10 where
  values := fun row c t =>
    if row.val < 4 then
      if c = 0 then (![5,7,11] : Fin 3 → ℝ) t +
        (if 2 ≤ row.val % 4 ∧ t = 0 then 2 else 0)
      else 0
    else if c = 0 then 0
    else (![13,17,19] : Fin 3 → ℝ) t +
      (if 2 ≤ row.val % 4 ∧ t.val = c.val then 3 else 0)
  labels := fun row => if row.val % 4 < 2 then 0 else 1
  labelDtype := "<i8"
  nativeFloat64Dtype := "<f8"
  subjects := fun row => if row.val < 4 then "A" else "B"
  channelNames := fun c => (!["Fz","Cz","Pz"] : Fin 3 → String) c
  channelMask := fun _ => true
  trialMask := some (fun row c => if row.val < 4 then decide (c = 0) else decide (c ≠ 0))
  evidence := ![2,-1,0,3,1,6,4,-1,7,5]
  eventIds := fun e => "e" ++ toString e.val
  identity := fun subject => ⟨subject, ["origin:" ++ subject], [], ["study:" ++ subject]⟩
  identityPayload := fun subject => .text subject
  storedParent := none

def fixtureRequested : List (Fin 8) := [3,0,7,4,2,1,6,5]

theorem fixture_selected_rows :
    selectedRows fixture fixtureRequested "A" = [3,0,2,1] ∧
    selectedRows fixture fixtureRequested "B" = [7,4,6,5] := by decide

theorem fixture_different_local_montages :
    commonChannels fixture [3,0,2,1] = [0] ∧
    commonChannels fixture [7,4,6,5] = [1,2] ∧
    commonChannels fixture fixtureRequested = [] := by decide

theorem participant_B_montage_is_not_the_global_intersection :
    commonChannels fixture [7,4,6,5] ≠ commonChannels fixture fixtureRequested := by
  rw [fixture_different_local_montages.2.1, fixture_different_local_montages.2.2]
  decide

/-- Pure decoded mask witness. Original row0 observes only channel1; the other
selected A rows observe channel0. Its masked cells are zero. No complete
EpochDataset or physical/QC/identity admission is asserted by this fixture. -/
def fixtureVariableMask : Dataset 8 3 3 10 :=
  { fixture with
    values := fun row c t => if row = 0 then 0 else fixture.values row c t
    trialMask := some (fun row c => if row = 0 then decide (c = 1)
      else if row.val < 4 then decide (c = 0) else decide (c ≠ 0)) }

theorem all_observed_is_not_any_observed_on_varying_rows :
    commonChannels fixtureVariableMask [3,0,2,1] = [] ∧
    ([3,0,2,1] : List (Fin 8)).any (fun row => effectiveObserved fixtureVariableMask row 0) = true := by
  decide

theorem fixture_nontrivial_original_event_gate : sortedEvidenceGate fixture := by
  simp [sortedEvidenceGate, mappedEvents, fixture, List.finRange_succ]
  change List.mergeSort [2,0,3,1,6,4,7,5] (fun a b : ℤ => decide (a ≤ b)) =
    (List.range 8).map Int.ofNat
  rw [List.mergeSort_eq_insertionSort (· ≤ ·)]
  decide

theorem fixture_original_event_names (initial : Fin 8 → Option (Fin 10)) :
    originalEventIds fixture initial [3,0,2,1] = ["e3","e2","e0","e4"] ∧
    originalEventIds fixture initial [7,4,6,5] = ["e8","e6","e5","e9"] := by
  simp [originalEventIds, writeInverse, mappedEvents, fixture, List.finRange_succ]
  decide

theorem fixture_integer_class_sets :
    integerClassRows fixture ([3,0,2,1] : List (Fin 8)) 0 = ({1,3} : Finset (Fin 4)) ∧
    integerClassRows fixture ([3,0,2,1] : List (Fin 8)) 1 = ({0,2} : Finset (Fin 4)) ∧
    integerClassRows fixture ([7,4,6,5] : List (Fin 8)) 0 = ({1,3} : Finset (Fin 4)) ∧
    integerClassRows fixture ([7,4,6,5] : List (Fin 8)) 1 = ({0,2} : Finset (Fin 4)) := by
  refine ⟨?_,?_,?_,?_⟩
  all_goals ext i; fin_cases i <;> norm_num [integerClassRows, gatheredIntegerLabels, fixture] <;> decide

theorem fixture_signed_contrasts :
    sourceContrast fixture [3,0,2,1] [0] =
      (fun _ t => if t = 0 then 2 else 0) ∧
    sourceContrast fixture [7,4,6,5] [1,2] =
      (fun c t => if t.val = c.val + 1 then 3 else 0) := by
  have h02 : (0 : Fin 4) ≠ 2 := by decide
  have h13 : (1 : Fin 4) ≠ 3 := by decide
  have h10 : (1 : Fin 3) ≠ 0 := by decide
  have h20 : (2 : Fin 3) ≠ 0 := by decide
  constructor
  · ext c t
    simp only [sourceContrast, sourceClassMean, fixture_integer_class_sets.1,
      fixture_integer_class_sets.2.1, Matrix.sub_apply]
    fin_cases c; fin_cases t <;>
      norm_num [gatheredX, fixture, h02, h13, h10, h20]
  · ext c t
    simp only [sourceContrast, sourceClassMean, fixture_integer_class_sets.2.2.1,
      fixture_integer_class_sets.2.2.2, Matrix.sub_apply]
    fin_cases c <;> fin_cases t <;>
      norm_num [gatheredX, fixture, h02, h13, h10, h20]

theorem changed_values_do_not_refresh_receipt_parent (p : NumericHashPrimitives)
    (d : Dataset N C T E) (changed : Fin N → Fin C → Fin T → ℝ)
    (requested : List (Fin N)) (subject : String) (initial : Fin N → Option (Fin E)) :
    (receipt p {d with values := changed} requested subject initial).get "parent_cache_attestation" =
      (receipt p d requested subject initial).get "parent_cache_attestation" := by
  rw [receipt_parent_is_stored_identity, receipt_parent_is_stored_identity]


end
end N2P3.ERPTemporalProducer
