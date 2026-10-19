import N2P3.ERPState

/-!
# Personal ERP temporal target identity and complete-study exclusion

Source: temporal_transfer.py:86-129,227-238; identity.py:50-62,126-149,
epochs.py:706-730. Keys are ordinary String values in one common domain:
participant, origin and global categories do not receive disjoint type tags.

The full normalized PreprocessingSpec is represented by Fields. Constructing
that record, decoding canonical participant identities, URL split/unquote on
each origin, and the no-existing-table timeline identity construction remain
small external operations. None is a whole validate_target/admission oracle.
For the existing decoded identity-table route, coverage and materialization
are literal below. Its constructor has already checked nonempty unique local
names and valid identity records; byte parsing and physical identity authority
are not proved here. Source receipts connect to ERPState.readReceipt and its
source-role temporal reader, rather than to a separate permissive set model.
-/

namespace N2P3.PersonalERPIdentity

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState
open N2P3.ERPState

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure ReceiptInput where
  participant : String
  origins : List String
  globals : List String
  events : List String
  valuesDigest : String
  labelsDigest : String
  observedDigest : String
  identity : Value

def receiptFields (r : ReceiptInput) : Fields := Fields.ofList
  [("participant_key", .text r.participant), ("origin_keys", .sequence (encodeTexts r.origins)),
   ("global_keys", .sequence (encodeTexts r.globals)), ("event_ids", .sequence (encodeTexts r.events)),
   ("values_sha256", .text r.valuesDigest), ("labels_sha256", .text r.labelsDigest),
   ("observed_sha256", .text r.observedDigest), ("identity", r.identity)]

def receiptValues : List ReceiptInput → Values
  | [] => .nil
  | r :: tail => .cons (.object (receiptFields r)) (receiptValues tail)

def ReceiptConditions (p : TemporalPrimitives) (r : ReceiptInput) (i : IdentityImage) : Prop :=
  r.participant ≠ "" ∧ r.origins ≠ [] ∧ r.events ≠ [] ∧
  p.eventIdsUniqueHashable (.sequence (encodeTexts r.events)) = .ok true ∧
  r.valuesDigest.toList.length = 64 ∧ r.labelsDigest.toList.length = 64 ∧
  r.observedDigest.toList.length = 64 ∧ p.identity r.identity = .ok i ∧
  p.originsAllowed i.origins = true ∧
  i.authority = r.participant ∧ i.origins = r.origins ∧ i.globals = r.globals

theorem text_sequence_truthy (xs : List String) :
    PyTruthy (.sequence (encodeTexts xs)) ↔ xs ≠ [] := by
  cases xs <;> simp [encodeTexts, PyTruthy]

theorem receipt_reader_from_independent_fields (p : TemporalPrimitives)
    (r : ReceiptInput) (i : IdentityImage) (h : ReceiptConditions p r i) :
    ERPState.readReceipt p (.object (receiptFields r)) = .ok i := by
  rcases h with ⟨hp, ho, he, hu, hv, hl, hm, hi, ha, hap, hao, hag⟩
  have hot := (text_sequence_truthy r.origins).2 ho
  have het := (text_sequence_truthy r.events).2 he
  have hpt : PyTruthy (.text r.participant) := hp
  simp only [String.toList] at hv hl hm
  rw [hao] at ha
  simp [ERPState.readReceipt, receiptFields, objectAt, Fields.ofList, Fields.get,
    fieldDefault, required, ensure, string64, hpt, hot, het, hu, hv, hl, hm,
    hi, ha, hap, hao, hag, PersonalAdmission.PyEq]

def receiptKeys (r : ReceiptInput) : Finset String :=
  (r.participant :: (r.origins ++ r.globals)).toFinset

def identityKeys (i : IdentityImage) : Finset String :=
  (i.authority :: (i.origins ++ i.globals)).toFinset

def sourceKeys (rs : List ReceiptInput) : Finset String :=
  (rs.flatMap (fun r => r.participant :: (r.origins ++ r.globals))).toFinset

def targetKeys (ids : List IdentityImage) : Finset String :=
  (ids.flatMap (fun i => i.authority :: (i.origins ++ i.globals))).toFinset

def targetStudies (ids : List IdentityImage) : Finset String :=
  (ids.flatMap IdentityImage.studies).toFinset

theorem receipt_keys_from_guarded_identity (p : TemporalPrimitives)
    (r : ReceiptInput) (i : IdentityImage) (h : ReceiptConditions p r i) :
    receiptKeys r = identityKeys i := by
  rcases h with ⟨_, _, _, _, _, _, _, _, _, ha, ho, hg⟩
  simp [receiptKeys, identityKeys, ha, ho, hg]

theorem mem_receipt_keys (r : ReceiptInput) (key : String) :
    key ∈ receiptKeys r ↔ key = r.participant ∨ key ∈ r.origins ∨ key ∈ r.globals := by
  simp [receiptKeys]

theorem mem_identity_keys (i : IdentityImage) (key : String) :
    key ∈ identityKeys i ↔ key = i.authority ∨ key ∈ i.origins ∨ key ∈ i.globals := by
  simp [identityKeys]

theorem mem_source_keys (rs : List ReceiptInput) (key : String) :
    key ∈ sourceKeys rs ↔ ∃ r ∈ rs, key ∈ receiptKeys r := by
  simp [sourceKeys, receiptKeys]

theorem mem_target_keys (ids : List IdentityImage) (key : String) :
    key ∈ targetKeys ids ↔ ∃ i ∈ ids, key ∈ identityKeys i := by
  simp [targetKeys, identityKeys]

theorem mem_target_studies (ids : List IdentityImage) (study : String) :
    study ∈ targetStudies ids ↔ ∃ i ∈ ids, study ∈ i.studies := by
  simp [targetStudies]

/-- This parses each canonical origin URI; it does not decide target admission. -/
def originStudyList (decode : String → Check String) : List String → Check (List String)
  | [] => .ok []
  | origin :: tail => do
    let study ← decode origin
    let rest ← originStudyList decode tail
    pure (study :: rest)

def originStudies (decode : String → Check String) (origins : List String) : Check (List String) := do
  let studies ← originStudyList decode origins
  pure (studies.dedup.mergeSort (· ≤ ·))

def CanonicalOrigins (decode : String → Check String) (i : IdentityImage) : Prop :=
  originStudies decode i.origins = .ok i.studies

structure TargetRow where
  localSubject : String
  identity : IdentityImage

structure TargetInput where
  preprocessing : Fields
  displayName : String
  subjects : List String
  identityTable : Option (List TargetRow)
  timeline : Value

structure TargetPrimitives where
  preprocessing : Value → Check Fields
  timelineIdentity : Value → Check (List TargetRow)

def materializeTarget (p : TargetPrimitives) (target : TargetInput) : Check (List IdentityImage) := do
  let rows ← match target.identityTable with
    | some rows => do
      ensure ((rows.map TargetRow.localSubject).toFinset = target.subjects.toFinset)
        "target identity table coverage"
      pure rows
    | none => p.timelineIdentity target.timeline
  pure (rows.map TargetRow.identity)

structure SourceView where
  preprocessing : Value
  receipts : List ReceiptInput
  studies : List String

def validateTarget (p : TargetPrimitives) (source : SourceView) (target : TargetInput) : Check Unit := do
  let expected ← p.preprocessing source.preprocessing
  ensure (PersonalAdmission.PyEq (.object target.preprocessing) (.object expected))
    "temporal target preprocessing"
  let ids ← materializeTarget p target
  ensure (sourceKeys source.receipts ∩ targetKeys ids = ∅) "temporal target participant"
  ensure (source.studies.toFinset ∩ targetStudies ids = ∅) "complete target study"

def targetOfRows (preprocessing : Fields) (display : String) (rows : List TargetRow) : TargetInput :=
  ⟨preprocessing, display, rows.map TargetRow.localSubject, some rows, .null⟩

theorem materialize_explicit_table (p : TargetPrimitives) (pre : Fields)
    (display : String) (rows : List TargetRow) :
    materializeTarget p (targetOfRows pre display rows) = .ok (rows.map TargetRow.identity) := by
  simp [materializeTarget, targetOfRows, ensure]

/-- No disjointness assumption: this is the literal result for arbitrary keys. -/
theorem explicit_table_validation_formula (p : TargetPrimitives) (source : SourceView)
    (pre : Fields) (display : String) (rows : List TargetRow)
    (hp : p.preprocessing source.preprocessing = .ok pre) :
    validateTarget p source (targetOfRows pre display rows) =
      (if sourceKeys source.receipts ∩ targetKeys (rows.map TargetRow.identity) = ∅ then
        if source.studies.toFinset ∩ targetStudies (rows.map TargetRow.identity) = ∅ then .ok ()
        else .error (.sourceReject "complete target study")
       else .error (.sourceReject "temporal target participant")) := by
  by_cases hk : sourceKeys source.receipts ∩ targetKeys (rows.map TargetRow.identity) = ∅
  · by_cases hs : source.studies.toFinset ∩ targetStudies (rows.map TargetRow.identity) = ∅
    · simp [validateTarget, hp, targetOfRows, materializeTarget, PersonalAdmission.PyEq, ensure, hk, hs]
    · simp [validateTarget, hp, targetOfRows, materializeTarget, PersonalAdmission.PyEq, ensure, hk, hs]
  · simp [validateTarget, hp, targetOfRows, materializeTarget, PersonalAdmission.PyEq, ensure, hk]

theorem preprocessing_failure_precedes_materialization (p : TargetPrimitives)
    (source : SourceView) (target : TargetInput) (fault : LedgerFault)
    (hp : p.preprocessing source.preprocessing = .error fault) :
    validateTarget p source target = .error fault := by
  simp [validateTarget, hp]

theorem preprocessing_mismatch_precedes_materialization (p : TargetPrimitives)
    (source : SourceView) (target : TargetInput) (pre : Fields)
    (hp : p.preprocessing source.preprocessing = .ok pre)
    (h : ¬PersonalAdmission.PyEq (.object target.preprocessing) (.object pre)) :
    validateTarget p source target = .error (.sourceReject "temporal target preprocessing") := by
  simp [validateTarget, hp, ensure, h]

theorem shared_key_refuses_explicit_table (p : TargetPrimitives) (source : SourceView)
    (pre : Fields) (display : String) (rows : List TargetRow) (key : String)
    (hp : p.preprocessing source.preprocessing = .ok pre)
    (hs : key ∈ sourceKeys source.receipts)
    (ht : key ∈ targetKeys (rows.map TargetRow.identity)) :
    validateTarget p source (targetOfRows pre display rows) =
      .error (.sourceReject "temporal target participant") := by
  rw [explicit_table_validation_formula p source pre display rows hp]
  have hne : sourceKeys source.receipts ∩ targetKeys (rows.map TargetRow.identity) ≠ ∅ := by
    intro h
    have : key ∈ (∅ : Finset String) := h ▸ Finset.mem_inter.mpr ⟨hs, ht⟩
    simp at this
  simp [hne]

def Rejected (result : Check Unit) : Prop := ∃ stage, result = .error (.sourceReject stage)

theorem shared_study_refuses_explicit_table (p : TargetPrimitives) (source : SourceView)
    (pre : Fields) (display : String) (rows : List TargetRow) (study : String)
    (hp : p.preprocessing source.preprocessing = .ok pre)
    (hs : study ∈ source.studies) (ht : study ∈ targetStudies (rows.map TargetRow.identity)) :
    Rejected (validateTarget p source (targetOfRows pre display rows)) := by
  rw [explicit_table_validation_formula p source pre display rows hp]
  have hne : source.studies.toFinset ∩ targetStudies (rows.map TargetRow.identity) ≠ ∅ := by
    intro h
    have : study ∈ (∅ : Finset String) := h ▸ Finset.mem_inter.mpr ⟨by simpa using hs, ht⟩
    simp at this
  by_cases hk : sourceKeys source.receipts ∩ targetKeys (rows.map TargetRow.identity) = ∅
  · exact ⟨"complete target study", by simp [hk, hne]⟩
  · exact ⟨"temporal target participant", by simp [hk]⟩

def renameRow (rename : String → String) (row : TargetRow) : TargetRow :=
  ⟨rename row.localSubject, row.identity⟩

theorem local_and_display_rename_cannot_change_validation (p : TargetPrimitives)
    (source : SourceView) (pre : Fields) (oldDisplay newDisplay : String)
    (rows : List TargetRow) (rename : String → String) (_hinjective : Function.Injective rename) :
    validateTarget p source (targetOfRows pre newDisplay (rows.map (renameRow rename))) =
      validateTarget p source (targetOfRows pre oldDisplay rows) := by
  simp [validateTarget, targetOfRows, materializeTarget,
    ensure, List.map_map, Function.comp_def, renameRow]

theorem origin_overlap_refuses_despite_authority_change (p : TargetPrimitives)
    (source : SourceView) (pre : Fields) (display : String) (rows : List TargetRow)
    (r : ReceiptInput) (row : TargetRow) (origin : String)
    (hp : p.preprocessing source.preprocessing = .ok pre)
    (hr : r ∈ source.receipts) (ht : row ∈ rows)
    (hsOrigin : origin ∈ r.origins) (htOrigin : origin ∈ row.identity.origins) :
    validateTarget p source (targetOfRows pre display rows) =
      .error (.sourceReject "temporal target participant") := by
  apply shared_key_refuses_explicit_table p source pre display rows origin hp
  · exact (mem_source_keys _ _).2 ⟨r, hr, (mem_receipt_keys _ _).2 (Or.inr (Or.inl hsOrigin))⟩
  · apply (mem_target_keys _ _).2
    exact ⟨row.identity, List.mem_map.mpr ⟨row, ht, rfl⟩,
      (mem_identity_keys _ _).2 (Or.inr (Or.inl htOrigin))⟩

theorem global_overlap_refuses_despite_origin_change (p : TargetPrimitives)
    (source : SourceView) (pre : Fields) (display : String) (rows : List TargetRow)
    (r : ReceiptInput) (row : TargetRow) (global : String)
    (hp : p.preprocessing source.preprocessing = .ok pre)
    (hr : r ∈ source.receipts) (ht : row ∈ rows)
    (hsGlobal : global ∈ r.globals) (htGlobal : global ∈ row.identity.globals) :
    validateTarget p source (targetOfRows pre display rows) =
      .error (.sourceReject "temporal target participant") := by
  apply shared_key_refuses_explicit_table p source pre display rows global hp
  · exact (mem_source_keys _ _).2 ⟨r, hr, (mem_receipt_keys _ _).2 (Or.inr (Or.inr hsGlobal))⟩
  · apply (mem_target_keys _ _).2
    exact ⟨row.identity, List.mem_map.mpr ⟨row, ht, rfl⟩,
      (mem_identity_keys _ _).2 (Or.inr (Or.inr htGlobal))⟩

theorem text_sequences_python_equal_iff (xs ys : List String) :
    PersonalAdmission.PyEq (.sequence (encodeTexts xs)) (.sequence (encodeTexts ys)) ↔ xs = ys := by
  induction xs generalizing ys with
  | nil => cases ys <;> simp [PersonalAdmission.PyEq, PersonalAdmission.normalizeValue,
      PersonalAdmission.normalizeValues, encodeTexts]
  | cons x xs ih =>
    cases ys with
    | nil => simp [PersonalAdmission.PyEq, PersonalAdmission.normalizeValue,
        PersonalAdmission.normalizeValues, encodeTexts]
    | cons y ys =>
      simpa [PersonalAdmission.PyEq, PersonalAdmission.normalizeValue,
        PersonalAdmission.normalizeValues, encodeTexts] using (fun (_ : x = y) => ih ys)

theorem receipt_values_to_list (rs : List ReceiptInput) :
    valuesToList (receiptValues rs) = rs.map (fun r => .object (receiptFields r)) := by
  induction rs with
  | nil => rfl
  | cons r rs ih => simp [receiptValues, valuesToList, ih]

theorem receipts_reader_from_independent_fields (p : TemporalPrimitives)
    (rs : List ReceiptInput) (ids : List IdentityImage)
    (h : List.Forall₂ (ReceiptConditions p) rs ids) :
    ERPState.readReceipts p (valuesToList (receiptValues rs)) = .ok ids := by
  induction h with
  | nil => rfl
  | @cons r i rs ids hhead htail ih =>
    simp only [receiptValues, valuesToList, ERPState.readReceipts]
    rw [receipt_reader_from_independent_fields p r i hhead]
    simp only [Bind.bind, Except.bind, ih, Pure.pure, Except.pure]

theorem source_keys_match_all_guarded_receipt_identities (p : TemporalPrimitives)
    (rs : List ReceiptInput) (ids : List IdentityImage)
    (h : List.Forall₂ (ReceiptConditions p) rs ids) :
    sourceKeys rs = targetKeys ids := by
  induction h with
  | nil => rfl
  | @cons r i rs ids hhead htail ih =>
    have hr := receipt_keys_from_guarded_identity p r i hhead
    have hunion := congrArg₂ (fun (a b : Finset String) => a ∪ b) hr ih
    simpa [sourceKeys, targetKeys, receiptKeys, identityKeys] using hunion

def SourceBasisConditions (p : TemporalPrimitives) (b : FixedTemporalInput) : Prop :=
  p.preprocessing b.preprocessing = .ok ⟨b.times, b.baseline⟩ ∧
  0 < b.rank ∧ b.rank ≤ b.times ∧
  p.metadataArray b.rawBasis = .ok b.matrix ∧
  b.matrix.shape = [b.times, b.rank] ∧ p.orthogonalAllclose b.matrix = true

def sourceMetadata (b : FixedTemporalInput) (rs : List ReceiptInput) (studies : List String) : Fields :=
  Fields.ofList [("schema", .text "n2p3_temporal_basis/1"), ("fit_role", .text "source_supervised"),
    ("preprocessing", b.preprocessing), ("rank", .integer b.rank), ("basis", b.rawBasis),
    ("fit_receipts", .sequence (receiptValues rs)), ("source_studies", .sequence (encodeTexts studies)),
    ("risk", .text "equal_participant_normalized_ERP_contrast")]

def sourceRecord (p : TemporalPrimitives) (b : FixedTemporalInput)
    (rs : List ReceiptInput) (studies : List String) : Fields :=
  .cons "digest" (.text (p.metadataHash (sourceMetadata b rs studies))) (sourceMetadata b rs studies)

def sourceModel (p : TemporalPrimitives) (b : FixedTemporalInput)
    (rs : List ReceiptInput) (studies : List String) : TemporalModel :=
  ⟨sourceRecord p b rs studies, b.matrix, b.times, b.rank, b.baseline, .source⟩

/-- Conditions are small operation results and explicit fields, never whole
source-reader success or target disjointness. The study ledger is computed. -/
theorem source_reader_with_derived_studies (p : TemporalPrimitives) (b : FixedTemporalInput)
    (rs : List ReceiptInput) (ids : List IdentityImage)
    (hb : SourceBasisConditions p b) (hrs : rs ≠ [])
    (hr : List.Forall₂ (ReceiptConditions p) rs ids)
    (hu : (ids.map IdentityImage.authority).Nodup) :
    ERPState.readTemporal p (.object (sourceRecord p b rs (ERPState.canonicalStudies ids))) =
      .ok (sourceModel p b rs (ERPState.canonicalStudies ids)) := by
  rcases hb with ⟨hp, hpos, hmax, hm, hshape, horth⟩
  have hread := receipts_reader_from_independent_fields p rs ids hr
  have hnonempty : valuesToList (receiptValues rs) ≠ [] := by
    simpa [receipt_values_to_list] using hrs
  simp [ERPState.readTemporal, sourceRecord, sourceMetadata, fieldDefault, objectAt,
    Fields.ofList, Fields.get, Fields.erase, required, ensure, readRole, positiveInt,
    hp, hpos, hmax, hm, hshape, horth, hnonempty, hread, hu,
    sourceModel, PersonalAdmission.PyEq]

theorem source_reader_rejects_resigned_false_studies (p : TemporalPrimitives) (b : FixedTemporalInput)
    (rs : List ReceiptInput) (ids : List IdentityImage) (declared : List String)
    (hb : SourceBasisConditions p b) (hrs : rs ≠ [])
    (hr : List.Forall₂ (ReceiptConditions p) rs ids)
    (hu : (ids.map IdentityImage.authority).Nodup)
    (hfalse : declared ≠ ERPState.canonicalStudies ids) :
    ERPState.readTemporal p (.object (sourceRecord p b rs declared)) =
      .error (.sourceReject "temporal source studies") := by
  rcases hb with ⟨hp, hpos, hmax, hm, hshape, horth⟩
  have hread := receipts_reader_from_independent_fields p rs ids hr
  have hnonempty : valuesToList (receiptValues rs) ≠ [] := by
    simpa [receipt_values_to_list] using hrs
  have hneq : ¬PersonalAdmission.PyEq (.sequence (encodeTexts (ERPState.canonicalStudies ids)))
      (.sequence (encodeTexts declared)) := by
    rw [text_sequences_python_equal_iff]
    exact Ne.symm hfalse
  simp [ERPState.readTemporal, sourceRecord, sourceMetadata, fieldDefault, objectAt,
    Fields.ofList, Fields.get, Fields.erase, required, ensure, readRole, positiveInt,
    hp, hpos, hmax, hm, hshape, horth, hnonempty, hread, hu, hneq]

def sourceViewFromReceipts (b : FixedTemporalInput) (rs : List ReceiptInput)
    (ids : List IdentityImage) : SourceView :=
  ⟨b.preprocessing, rs, ERPState.canonicalStudies ids⟩

theorem source_study_set_is_receipt_identity_union (b : FixedTemporalInput)
    (rs : List ReceiptInput) (ids : List IdentityImage) :
    (sourceViewFromReceipts b rs ids).studies.toFinset = targetStudies ids := by
  ext study
  simp [sourceViewFromReceipts, ERPState.canonicalStudies, targetStudies]

def identityFromOrigins (decode : String → Check String) (authority : String)
    (origins globals : List String) : Check IdentityImage := do
  let studies ← originStudies decode origins
  pure ⟨authority, origins, globals, studies⟩

theorem identity_from_individually_decoded_origins (decode : String → Check String)
    (authority : String) (origins globals libraries : List String)
    (h : originStudyList decode origins = .ok libraries) :
    identityFromOrigins decode authority origins globals =
      .ok ⟨authority, origins, globals, libraries.dedup.mergeSort (· ≤ ·)⟩ := by
  simp [identityFromOrigins, originStudies, h]

theorem source_origin_target_global_cannot_be_type_separated (p : TargetPrimitives)
    (source : SourceView) (pre : Fields) (display : String) (rows : List TargetRow)
    (r : ReceiptInput) (row : TargetRow) (key : String)
    (hp : p.preprocessing source.preprocessing = .ok pre)
    (hr : r ∈ source.receipts) (ht : row ∈ rows)
    (hs : key ∈ r.origins) (hglobal : key ∈ row.identity.globals) :
    validateTarget p source (targetOfRows pre display rows) =
      .error (.sourceReject "temporal target participant") := by
  apply shared_key_refuses_explicit_table p source pre display rows key hp
  · exact (mem_source_keys _ _).2 ⟨r, hr, (mem_receipt_keys _ _).2 (Or.inr (Or.inl hs))⟩
  · apply (mem_target_keys _ _).2
    exact ⟨row.identity, List.mem_map.mpr ⟨row, ht, rfl⟩,
      (mem_identity_keys _ _).2 (Or.inr (Or.inr hglobal))⟩

def examplePreprocessing : Fields := Fields.ofList
  [("name", .text "personal-identity-fixture"), ("sfreq", .number 250),
   ("l_freq", .null), ("h_freq", .null), ("tmin_ms", .number 0), ("tmax_ms", .number 4),
   ("n_times", .integer 1), ("baseline_mode", .text "none"), ("signal_unit", .text "V"),
   ("filter_method", .text "iir"), ("filter_order", .integer 4), ("filter_phase", .text "forward"),
   ("causal_iir_initial_state", .text "steady_state_first_sample"),
   ("notch_freqs_hz", .sequence .nil), ("notch_quality_factor", .number 30),
   ("zero_phase_edge_policy", .text "legacy"), ("resample_domain", .text "epoched"),
   ("resample_method", .text "fft"), ("resample_npad", .text "auto"),
   ("resample_window", .text "auto"), ("resample_pad", .text "edge"),
   ("trial_reference_window_ms", .null), ("trial_reference_center", .text "mean"),
   ("trial_reference_scale", .text "none"), ("reject_threshold_v", .null)]

def sourceOrigin : String := "n2p3-origin://source-study/participant/source-a"
def targetOrigin : String := "n2p3-origin://target-study/participant/target-b"
def sameStudyOrigin : String := "n2p3-origin://source-study/participant/target-b"

def exampleOriginStudy : String → Check String
  | "n2p3-origin://source-study/participant/source-a" => .ok "source-study"
  | "n2p3-origin://target-study/participant/target-b" => .ok "target-study"
  | "n2p3-origin://source-study/participant/target-b" => .ok "source-study"
  | _ => .error (.fieldFailure "canonical origin URI")

def exampleSourceIdentity : IdentityImage :=
  ⟨"global-source-a", [sourceOrigin], ["global-source-a"], ["source-study"]⟩
def exampleTargetIdentity : IdentityImage :=
  ⟨"global-target-b", [targetOrigin], ["global-target-b"], ["target-study"]⟩
def exampleSameStudyIdentity : IdentityImage :=
  ⟨"global-target-b", [sameStudyOrigin], ["global-target-b"], ["source-study"]⟩
def exampleSharedOriginIdentity : IdentityImage :=
  ⟨"global-target-b", [sourceOrigin], ["global-target-b"], ["source-study"]⟩
def exampleCrossCategoryIdentity : IdentityImage :=
  ⟨sourceOrigin, [targetOrigin], [sourceOrigin], ["target-study"]⟩

def identityPayload (localName : String) (i : IdentityImage) : Value := .object (Fields.ofList
  [("local_subject_id", .text localName), ("origin_subject_keys", .sequence (encodeTexts i.origins)),
   ("global_person_keys", .sequence (encodeTexts i.globals)), ("identity_status", .text "global_verified")])

def exampleDigest : String := String.mk (List.replicate 64 'g')
def exampleReceipt : ReceiptInput :=
  ⟨"global-source-a", [sourceOrigin], ["global-source-a"], ["source-event-0"],
    exampleDigest, exampleDigest, exampleDigest, identityPayload "source-a" exampleSourceIdentity⟩

/-- Only the consumed finite example operations are supplied here. In
particular these definitions do not certify a physical source fit. -/
def exampleTemporalPrimitives (hash : Fields → String) : TemporalPrimitives :=
  ⟨hash, fun _ => .ok ⟨1, .text "none"⟩,
   fun _ => .ok (realTensor [1, 1] [1]), fun _ => true,
   fun _ _ => realTensor [1, 1] [1],
   fun raw => if raw = identityPayload "source-a" exampleSourceIdentity then .ok exampleSourceIdentity
     else .error (.fieldFailure "unused example identity"),
   fun _ => true, fun _ => .ok true⟩

def exampleBasis : FixedTemporalInput :=
  ⟨.object examplePreprocessing, .sequence (.cons (.sequence (.cons (.number 1) .nil)) .nil),
    1, 1, .text "none", realTensor [1, 1] [1]⟩

def exampleSource : SourceView :=
  sourceViewFromReceipts exampleBasis [exampleReceipt] [exampleSourceIdentity]

def exampleTargetPrimitives : TargetPrimitives :=
  ⟨fun _ => .ok examplePreprocessing,
   fun _ => .error (.fieldFailure "unused timeline identity route")⟩

def exampleRows (identity : IdentityImage) : List TargetRow := [⟨"target-b", identity⟩]

theorem source_and_target_origin_libraries_are_decoded :
    identityFromOrigins exampleOriginStudy "global-source-a" [sourceOrigin] ["global-source-a"] =
      .ok exampleSourceIdentity ∧
    identityFromOrigins exampleOriginStudy "global-target-b" [targetOrigin] ["global-target-b"] =
      .ok exampleTargetIdentity ∧
    identityFromOrigins exampleOriginStudy "global-target-b" [sameStudyOrigin] ["global-target-b"] =
      .ok exampleSameStudyIdentity := by
  simp [identityFromOrigins, originStudies, originStudyList, exampleOriginStudy,
    sourceOrigin, targetOrigin, sameStudyOrigin, exampleSourceIdentity, exampleTargetIdentity,
    exampleSameStudyIdentity]

theorem example_receipt_has_explicit_source_guards (hash : Fields → String) :
    ReceiptConditions (exampleTemporalPrimitives hash) exampleReceipt exampleSourceIdentity := by
  simp [ReceiptConditions, exampleTemporalPrimitives, exampleReceipt, exampleDigest,
    exampleSourceIdentity, sourceOrigin]

theorem nonempty_source_role_reader_constructs (hash : Fields → String) :
    ERPState.readTemporal (exampleTemporalPrimitives hash)
      (.object (sourceRecord (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt]
        (ERPState.canonicalStudies [exampleSourceIdentity]))) =
      .ok (sourceModel (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt]
        (ERPState.canonicalStudies [exampleSourceIdentity])) := by
  apply source_reader_with_derived_studies
  · simp [SourceBasisConditions, exampleTemporalPrimitives, exampleBasis, realTensor]
  · simp
  · exact .cons (example_receipt_has_explicit_source_guards hash) .nil
  · simp

theorem nonempty_two_study_target_is_admitted :
    validateTarget exampleTargetPrimitives exampleSource
      (targetOfRows examplePreprocessing "target-display" (exampleRows exampleTargetIdentity)) = .ok () := by
  simp [validateTarget, exampleTargetPrimitives, exampleSource, sourceViewFromReceipts,
    exampleBasis, materializeTarget, targetOfRows, exampleRows, sourceKeys, targetKeys,
    targetStudies, exampleReceipt, exampleSourceIdentity, exampleTargetIdentity,
    ERPState.canonicalStudies, sourceOrigin, targetOrigin, ensure, PersonalAdmission.PyEq]

theorem different_people_same_study_are_refused :
    validateTarget exampleTargetPrimitives exampleSource
      (targetOfRows examplePreprocessing "renamed-safe-display" (exampleRows exampleSameStudyIdentity)) =
      .error (.sourceReject "complete target study") := by
  simp [validateTarget, exampleTargetPrimitives, exampleSource, sourceViewFromReceipts,
    exampleBasis, materializeTarget, targetOfRows, exampleRows, sourceKeys, targetKeys,
    targetStudies, exampleReceipt, exampleSourceIdentity, exampleSameStudyIdentity,
    ERPState.canonicalStudies, sourceOrigin, sameStudyOrigin, ensure, PersonalAdmission.PyEq]

theorem changed_authority_shared_origin_is_refused :
    validateTarget exampleTargetPrimitives exampleSource
      (targetOfRows examplePreprocessing "renamed" (exampleRows exampleSharedOriginIdentity)) =
      .error (.sourceReject "temporal target participant") := by
  apply origin_overlap_refuses_despite_authority_change _ _ _ _ _ exampleReceipt
    ⟨"target-b", exampleSharedOriginIdentity⟩ sourceOrigin
  all_goals simp [exampleTargetPrimitives, exampleSource, sourceViewFromReceipts,
    exampleRows, exampleReceipt, exampleSharedOriginIdentity]

theorem origin_global_cross_category_is_refused :
    validateTarget exampleTargetPrimitives exampleSource
      (targetOfRows examplePreprocessing "independent-display" (exampleRows exampleCrossCategoryIdentity)) =
      .error (.sourceReject "temporal target participant") := by
  apply source_origin_target_global_cannot_be_type_separated _ _ _ _ _ exampleReceipt
    ⟨"target-b", exampleCrossCategoryIdentity⟩ sourceOrigin
  all_goals simp [exampleTargetPrimitives, exampleSource, sourceViewFromReceipts,
    exampleRows, exampleReceipt, exampleCrossCategoryIdentity]

theorem resigning_fake_study_metadata_still_refuses (hash : Fields → String) :
    ERPState.readTemporal (exampleTemporalPrimitives hash)
      (.object (sourceRecord (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt] ["safe-fiction"])) =
      .error (.sourceReject "temporal source studies") := by
  apply source_reader_rejects_resigned_false_studies _ _ _ [exampleSourceIdentity]
  · simp [SourceBasisConditions, exampleTemporalPrimitives, exampleBasis, realTensor]
  · simp
  · exact .cons (example_receipt_has_explicit_source_guards hash) .nil
  · simp
  · simp [ERPState.canonicalStudies, exampleSourceIdentity]

/-- The target method reads only these three fields from each receipt. Other
receipt fields have already been checked by the source-role record reader. -/
def keyProjection (r : ReceiptInput) : ReceiptInput :=
  ⟨r.participant, r.origins, r.globals, [], "", "", "", .null⟩

def textValue (value : Value) : Check String :=
  match value with
  | .text s => .ok s
  | _ => .error (.fieldFailure "decoded receipt string")

def textsValue (value : Value) : Check (List String) :=
  match value with
  | .sequence values =>
    match decodeTexts values with
    | some strings => .ok strings
    | none => .error (.fieldFailure "decoded receipt strings")
  | _ => .error (.fieldFailure "decoded receipt strings")

def receiptKeyView (raw : Value) : Check ReceiptInput := do
  let f ← objectAt raw "source receipt fields"
  let participant ← textValue (← required f "participant_key")
  let origins ← textsValue (← required f "origin_keys")
  let globals ← textsValue (← required f "global_keys")
  pure ⟨participant, origins, globals, [], "", "", "", .null⟩

def receiptKeyViews : List Value → Check (List ReceiptInput)
  | [] => .ok []
  | raw :: tail => do
    let head ← receiptKeyView raw
    let rest ← receiptKeyViews tail
    pure (head :: rest)

def readSourceView (model : TemporalModel) : Check SourceView := do
  let preprocessing ← required model.record "preprocessing"
  let receipts ← listAt (← required model.record "fit_receipts") "source receipts list"
  let keys ← receiptKeyViews receipts
  let studies ← textsValue (← required model.record "source_studies")
  pure ⟨preprocessing, keys, studies⟩

def validateModelTarget (p : TargetPrimitives) (model : TemporalModel) (target : TargetInput) : Check Unit := do
  let source ← readSourceView model
  validateTarget p source target

theorem receipt_key_view_reads_actual_source_fields (r : ReceiptInput) :
    receiptKeyView (.object (receiptFields r)) = .ok (keyProjection r) := by
  simp [receiptKeyView, receiptFields, keyProjection, objectAt, required,
    Fields.ofList, Fields.get, textValue, textsValue, decode_encode_texts]

theorem receipt_key_views_read_actual_source_fields (rs : List ReceiptInput) :
    receiptKeyViews (valuesToList (receiptValues rs)) = .ok (rs.map keyProjection) := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    simp only [receiptValues, valuesToList, receiptKeyViews]
    rw [receipt_key_view_reads_actual_source_fields]
    simp only [Bind.bind, Except.bind, ih, Pure.pure, Except.pure, List.map_cons]

theorem read_source_view_from_constructed_model (p : TemporalPrimitives) (b : FixedTemporalInput)
    (rs : List ReceiptInput) (studies : List String) :
    readSourceView (sourceModel p b rs studies) = .ok ⟨b.preprocessing, rs.map keyProjection, studies⟩ := by
  simp [readSourceView, sourceModel, sourceRecord, sourceMetadata, required, Fields.ofList,
    Fields.get, listAt, receipt_key_views_read_actual_source_fields, textsValue, decode_encode_texts]

theorem key_projection_preserves_the_common_string_union (rs : List ReceiptInput) :
    sourceKeys (rs.map keyProjection) = sourceKeys rs := by
  simp [sourceKeys, List.flatMap_map, keyProjection]

theorem key_projection_preserves_target_validation (p : TargetPrimitives) (source : SourceView)
    (target : TargetInput) :
    validateTarget p ⟨source.preprocessing, source.receipts.map keyProjection, source.studies⟩ target =
      validateTarget p source target := by
  simp [validateTarget, key_projection_preserves_the_common_string_union]

/-- A real composition through the loaded model's record and each receipt
field. Neither reader success nor target admission is assumed as a premise. -/
theorem source_reader_then_model_target_uses_guarded_receipt_fields
    (p : TemporalPrimitives) (tp : TargetPrimitives) (b : FixedTemporalInput)
    (rs : List ReceiptInput) (ids : List IdentityImage) (target : TargetInput)
    (hb : SourceBasisConditions p b) (hrs : rs ≠ [])
    (hr : List.Forall₂ (ReceiptConditions p) rs ids)
    (hu : (ids.map IdentityImage.authority).Nodup) :
    (ERPState.readTemporal p (.object (sourceRecord p b rs (ERPState.canonicalStudies ids)))).bind
      (fun model => validateModelTarget tp model target) =
      validateTarget tp (sourceViewFromReceipts b rs ids) target := by
  rw [source_reader_with_derived_studies p b rs ids hb hrs hr hu]
  simp only [Except.bind, validateModelTarget]
  rw [read_source_view_from_constructed_model]
  change validateTarget tp
    ⟨b.preprocessing, rs.map keyProjection, ERPState.canonicalStudies ids⟩ target = _
  exact key_projection_preserves_target_validation tp (sourceViewFromReceipts b rs ids) target

theorem existing_table_coverage_failure_is_preserved (p : TargetPrimitives) (target : TargetInput)
    (rows : List TargetRow) (htable : target.identityTable = some rows)
    (hcoverage : (rows.map TargetRow.localSubject).toFinset ≠ target.subjects.toFinset) :
    materializeTarget p target = .error (.sourceReject "target identity table coverage") := by
  simp [materializeTarget, htable, hcoverage, ensure]

theorem missing_table_uses_only_timeline_identity_operation (p : TargetPrimitives)
    (target : TargetInput) (hnone : target.identityTable = none) :
    materializeTarget p target = (p.timelineIdentity target.timeline).map (List.map TargetRow.identity) := by
  simp [materializeTarget, hnone]

theorem nonempty_source_model_and_distinct_target_compose (hash : Fields → String) :
    (ERPState.readTemporal (exampleTemporalPrimitives hash)
      (.object (sourceRecord (exampleTemporalPrimitives hash) exampleBasis [exampleReceipt]
        (ERPState.canonicalStudies [exampleSourceIdentity])))).bind
      (fun model => validateModelTarget exampleTargetPrimitives model
        (targetOfRows examplePreprocessing "target-display" (exampleRows exampleTargetIdentity))) = .ok () := by
  rw [source_reader_then_model_target_uses_guarded_receipt_fields]
  · exact nonempty_two_study_target_is_admitted
  · simp [SourceBasisConditions, exampleTemporalPrimitives, exampleBasis, realTensor]
  · simp
  · exact .cons (example_receipt_has_explicit_source_guards hash) .nil
  · simp

structure CanonicalReceiptRow where
  receipt : ReceiptInput
  decodedLibraries : List String

def canonicalRowIdentity (row : CanonicalReceiptRow) : IdentityImage :=
  ⟨row.receipt.participant, row.receipt.origins, row.receipt.globals,
    row.decodedLibraries.dedup.mergeSort (· ≤ ·)⟩

/-- The origin decoder is called per URI. The identity parser's output is
required to have that computed library list, not an unrelated study field. -/
def CanonicalReceiptGuards (p : TemporalPrimitives) (decode : String → Check String)
    (row : CanonicalReceiptRow) : Prop :=
  originStudyList decode row.receipt.origins = .ok row.decodedLibraries ∧
  ReceiptConditions p row.receipt (canonicalRowIdentity row)

theorem canonical_row_identity_is_computed_from_origin_operations
    (p : TemporalPrimitives) (decode : String → Check String) (row : CanonicalReceiptRow)
    (h : CanonicalReceiptGuards p decode row) :
    identityFromOrigins decode row.receipt.participant row.receipt.origins row.receipt.globals =
      .ok (canonicalRowIdentity row) := by
  simpa only [canonicalRowIdentity] using
    identity_from_individually_decoded_origins decode row.receipt.participant row.receipt.origins
      row.receipt.globals row.decodedLibraries h.1

theorem canonical_rows_supply_the_actual_receipt_reader
    (p : TemporalPrimitives) (decode : String → Check String) (rows : List CanonicalReceiptRow)
    (h : ∀ row ∈ rows, CanonicalReceiptGuards p decode row) :
    List.Forall₂ (ReceiptConditions p) (rows.map CanonicalReceiptRow.receipt)
      (rows.map canonicalRowIdentity) := by
  induction rows with
  | nil => exact .nil
  | cons row rows ih =>
    apply List.Forall₂.cons (h row (by simp)).2
    apply ih
    intro other ho
    exact h other (by simp [ho])

theorem source_reader_with_canonical_origin_operations
    (p : TemporalPrimitives) (decode : String → Check String) (b : FixedTemporalInput)
    (rows : List CanonicalReceiptRow) (hb : SourceBasisConditions p b) (hne : rows ≠ [])
    (hr : ∀ row ∈ rows, CanonicalReceiptGuards p decode row)
    (hu : ((rows.map canonicalRowIdentity).map IdentityImage.authority).Nodup) :
    ERPState.readTemporal p (.object (sourceRecord p b (rows.map CanonicalReceiptRow.receipt)
      (ERPState.canonicalStudies (rows.map canonicalRowIdentity)))) =
      .ok (sourceModel p b (rows.map CanonicalReceiptRow.receipt)
        (ERPState.canonicalStudies (rows.map canonicalRowIdentity))) := by
  apply source_reader_with_derived_studies p b _ _ hb
  · simpa using hne
  · exact canonical_rows_supply_the_actual_receipt_reader p decode rows hr
  · exact hu

theorem canonical_origin_source_reader_then_loaded_target
    (p : TemporalPrimitives) (tp : TargetPrimitives) (decode : String → Check String)
    (b : FixedTemporalInput) (rows : List CanonicalReceiptRow) (target : TargetInput)
    (hb : SourceBasisConditions p b) (hne : rows ≠ [])
    (hr : ∀ row ∈ rows, CanonicalReceiptGuards p decode row)
    (hu : ((rows.map canonicalRowIdentity).map IdentityImage.authority).Nodup) :
    (ERPState.readTemporal p (.object (sourceRecord p b (rows.map CanonicalReceiptRow.receipt)
      (ERPState.canonicalStudies (rows.map canonicalRowIdentity))))).bind
      (fun model => validateModelTarget tp model target) =
      validateTarget tp
        (sourceViewFromReceipts b (rows.map CanonicalReceiptRow.receipt) (rows.map canonicalRowIdentity)) target := by
  apply source_reader_then_model_target_uses_guarded_receipt_fields p tp b _ _ target hb
  · simpa using hne
  · exact canonical_rows_supply_the_actual_receipt_reader p decode rows hr
  · exact hu

def exampleCanonicalReceipt : CanonicalReceiptRow := ⟨exampleReceipt, ["source-study"]⟩

theorem nonempty_receipt_studies_are_canonical_origin_results (hash : Fields → String) :
    CanonicalReceiptGuards (exampleTemporalPrimitives hash) exampleOriginStudy exampleCanonicalReceipt := by
  constructor
  · simp [exampleCanonicalReceipt, exampleReceipt, originStudyList, sourceOrigin, exampleOriginStudy]
  · simpa [exampleCanonicalReceipt, canonicalRowIdentity, exampleReceipt, exampleSourceIdentity] using
      example_receipt_has_explicit_source_guards hash

def exampleSharedGlobalIdentity : IdentityImage :=
  ⟨"global-source-a", [targetOrigin], ["global-source-a"], ["target-study"]⟩

theorem distinct_origins_shared_global_is_refused :
    validateTarget exampleTargetPrimitives exampleSource
      (targetOfRows examplePreprocessing "different-origin" (exampleRows exampleSharedGlobalIdentity)) =
      .error (.sourceReject "temporal target participant") := by
  apply global_overlap_refuses_despite_origin_change _ _ _ _ _ exampleReceipt
    ⟨"target-b", exampleSharedGlobalIdentity⟩ "global-source-a"
  all_goals simp [exampleTargetPrimitives, exampleSource, sourceViewFromReceipts,
    exampleRows, exampleReceipt, exampleSharedGlobalIdentity]

def renamedSameStudyTarget : TargetInput :=
  targetOfRows examplePreprocessing "fictional-independent-study"
    [⟨"renamed-local-person", exampleSameStudyIdentity⟩]

theorem renamed_same_study_is_still_refused :
    validateTarget exampleTargetPrimitives exampleSource renamedSameStudyTarget =
      .error (.sourceReject "complete target study") := by
  simpa [validateTarget, materializeTarget, targetOfRows, renamedSameStudyTarget,
    exampleRows, ensure] using different_people_same_study_are_refused

def differentFrequencyPreprocessing : Fields :=
  .cons "sfreq" (.number 500) (.cons "tmax_ms" (.number 2)
    ((examplePreprocessing.erase "sfreq").erase "tmax_ms"))

theorem equal_time_count_does_not_make_full_preprocessing_equal :
    differentFrequencyPreprocessing.get "n_times" = examplePreprocessing.get "n_times" ∧
    ¬PersonalAdmission.PyEq (.object differentFrequencyPreprocessing) (.object examplePreprocessing) := by
  constructor
  · simp [differentFrequencyPreprocessing, examplePreprocessing, Fields.ofList, Fields.get, Fields.erase]
  · intro h
    change PersonalAdmission.normalizeValue (.object differentFrequencyPreprocessing) =
      PersonalAdmission.normalizeValue (.object examplePreprocessing) at h
    have hf := congrArg (fun value => match value with
      | PersonalAdmission.PyValue.object fields => fields "sfreq"
      | _ => none) h
    have hfield : ("sfreq" : String) ≠ "name" := by decide
    norm_num [PersonalAdmission.normalizeValue, PersonalAdmission.normalizeFields,
      differentFrequencyPreprocessing, examplePreprocessing, Fields.ofList, Fields.erase, hfield] at hf

theorem same_T_different_full_preprocessing_is_refused :
    validateTarget exampleTargetPrimitives exampleSource
      (targetOfRows differentFrequencyPreprocessing "same-T" (exampleRows exampleTargetIdentity)) =
      .error (.sourceReject "temporal target preprocessing") := by
  apply preprocessing_mismatch_precedes_materialization _ _ _ examplePreprocessing
  · rfl
  · exact equal_time_count_does_not_make_full_preprocessing_equal.2

end

end N2P3.PersonalERPIdentity
