import N2P3.PersonalERPData

/-!
# Individual preprocessing execution receipt and bundle checks

Source: epochs.py:523-703 and preprocess.py:36-87. Every receipt field,
raw-scope bound, support span and boundary cause is checked below. A bundle
contains individual receipts only; the source rejects nested bundles.

The finite unique-key JSON record domain follows the saved personal metadata
copy. Per-array NumPy float conversion, semantic SHA, Python floating round,
isclose and scalar arithmetic are explicit small operations. They do not
certify genuine execution, raw producer identity, exact IEEE behavior or the
relation of recorded event hashes to the samples that actually executed.
No whole receipt, binding, raw-scope or preprocessing validator is assumed.
-/

namespace N2P3.PersonalERPExecution

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

theorem bind_success {α β : Type} (first : Check α) (next : α → Check β) (value : β) :
    (first >>= next)=.ok value ↔ ∃ input, first=.ok input ∧ next input=.ok value := by
  cases first <;> simp

theorem ensure_success (condition : Prop) (site : String) (completion : Unit)
    (h : ensure condition site=.ok completion) : condition := by
  by_cases hc : condition
  · exact hc
  · simp [ensure,hc] at h

structure Primitives where
  strip : String → String
  floorSamples : FloorSampleCount
  windowSamples : WindowSampleCount
  semanticHash : Value → Check String
  roundOffset : ℝ → ℝ → Check ℤ
  closeRates : ℝ → ℝ → Bool
  roundResampled : ℤ → ℝ → ℝ → Check ℤ
  divide : ℝ → ℝ → Check ℝ
  floatArray : Value → Check (ArrayImage Atom)

def text (value : Value) (site : String) : Check String :=
  match value with
  | .text s => .ok s
  | _ => .error (.fieldFailure site)

def integer (value : Value) (site : String) : Check ℤ :=
  match value with
  | .integer z => .ok z
  | _ => .error (.fieldFailure site)

def integers (site : String) : List Value → Check (List ℤ)
  | [] => .ok []
  | value :: tail => do
    let z ← integer value site
    let rest ← integers site tail
    pure (z :: rest)

def texts (site : String) : List Value → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let s ← text value site
    let rest ← texts site tail
    pure (s :: rest)

def textList (value : Value) (site : String) : Check (List String) := do
  let values ← listAt value site
  texts site values

def integerList (value : Value) (site : String) : Check (List ℤ) := do
  let values ← listAt value site
  integers site values

def nonemptyText (value : Value) (site : String) : Check String := do
  let s ← text value site
  ensure (s ≠ "") site
  pure s

def hexDigest (value : Value) (site : String) : Check String := do
  let s ← text value site
  ensure (LowerHex64 s) site
  pure s

def operatorIdentity : Value := .object (Fields.ofList
  [("schema",.text "n2p3_preprocessing_operator/1"),("backend",.text "mne_raw"),
   ("operations",.sequence (encodeTexts
      ["continuous_filter","source_grid_closed_epoch","epoch_fft_resample","first_T_crop",
       "half_open_baseline","float32"])),
   ("availability",.text "max_nominal_and_source_endpoint_or_record_end/1")])

structure RawScope where
  name : String
  start : ℤ
  stop : ℤ
  eventIndices : List ℤ

def readRawScope (strip : String → String) (value : Value) : Check RawScope := do
  let fields ← objectAt value "raw scope mapping"
  ensure (ExactKeys fields ["schema","name","start_sample","stop_sample","event_indices"] ∧
    fieldDefault fields "schema"=.text "n2p3_raw_sample_scope/1") "raw scope complete schema"
  let name ← text (fieldDefault fields "name") "raw scope name"
  ensure (strip name ≠ "") "nonblank raw scope name"
  let start ← integer (fieldDefault fields "start_sample") "raw scope integer start"
  let stop ← integer (fieldDefault fields "stop_sample") "raw scope integer stop"
  ensure (0 ≤ start ∧ start < stop) "increasing nonnegative raw scope bounds"
  let indices ← integerList (fieldDefault fields "event_indices") "raw scope integer events"
  ensure (indices ≠ [] ∧ (∀ i ∈ indices, 0 ≤ i) ∧ indices.Pairwise (· < ·))
    "nonempty increasing nonnegative raw scope events"
  pure ⟨name,start,stop,indices⟩

def checkImplementation (fields : Fields) : Check Unit := do
  let implementation ← objectAt (fieldDefault fields "implementation") "execution implementation mapping"
  let _ ← nonemptyText (fieldDefault implementation "python") "execution Python version"
  let _ ← nonemptyText (fieldDefault implementation "mne") "execution MNE version"
  let _ ← nonemptyText (fieldDefault implementation "numpy") "execution NumPy version"
  let _ ← nonemptyText (fieldDefault implementation "scipy") "execution SciPy version"
  let sources ← objectAt (fieldDefault implementation "source_sha256") "execution source hashes mapping"
  ensure (ExactKeys sources ["preprocess.py","epochs.py"]) "execution exact source hash names"
  let _ ← hexDigest (fieldDefault sources "preprocess.py") "execution preprocess source SHA"
  let _ ← hexDigest (fieldDefault sources "epochs.py") "execution epochs source SHA"
  pure ()

structure SourceGrid where
  rate : ℝ
  first : ℤ
  length : ℤ

def readSourceGrid (context : Fields) : Check SourceGrid := do
  let rate ← strictReal (atomFromValue (fieldDefault context "sample_rate_hz"))
  ensure (0 < rate) "positive finite source sample rate"
  let first ← integer (fieldDefault context "first_samp") "integer source first sample"
  let length ← integer (fieldDefault context "n_times") "integer source sample count"
  ensure (0 < length) "positive source sample count"
  pure ⟨rate,first,length⟩

def checkSourceBands (declaration : DeclarationScalars) (raw : RawPreprocessing) (source : SourceGrid) : Check Unit := do
  ensure (∀ frequency ∈ declaration.lFreq, frequency < source.rate/2) "high pass below source Nyquist"
  ensure (∀ frequency ∈ declaration.hFreq, frequency < source.rate/2) "low pass below source Nyquist"
  let _ ← readNotches raw.notchFrequencies raw.notchQuality (some source.rate)
  pure ()

def expectedEpochGrid (p : Primitives) (declaration : DeclarationScalars) (source : SourceGrid) : Check Fields := do
  let start ← p.roundOffset declaration.tmin source.rate
  let stop ← p.roundOffset declaration.tmax source.rate
  let samples := stop-start+1
  let resample := !(p.closeRates source.rate declaration.sfreq)
  let resampled ← if resample then p.roundResampled samples declaration.sfreq source.rate else .ok samples
  let requestedTmin ← p.divide declaration.tmin 1000
  let actualTmin ← p.divide start source.rate
  ensure (declaration.nTimes ≤ resampled) "resampled samples cover requested output"
  pure (Fields.ofList
    [("start_offset_samples",.integer start),("end_offset_samples",.integer stop),
     ("source_n_times",.integer samples),("resampled_n_times",.integer resampled),
     ("output_n_times",.integer declaration.nTimes),("requested_tmin_s",.number requestedTmin),
     ("actual_source_tmin_s",.number actualTmin),("resampled",.boolean resample)])

structure OutputGeometry where
  channelNames : List String
  rows : ℤ
  channels : ℤ
  times : ℤ
  scheduled : ℤ

def readOutputGeometry (output : Fields) (declaration : DeclarationScalars) : Check OutputGeometry := do
  let names ← textList (fieldDefault output "channel_names") "output channel names"
  ensure (names ≠ [] ∧ names.Nodup) "nonempty distinct output channel names"
  let shape ← integerList (fieldDefault output "shape") "output shape integer dimensions"
  let dimensions ← match shape with
    | [rows,channels,times] => .ok (rows,channels,times)
    | _ => .error (.fieldFailure "output shape rank three")
  ensure (0 ≤ dimensions.1 ∧ 0 ≤ dimensions.2.1 ∧ 0 ≤ dimensions.2.2) "nonnegative output shape"
  ensure (dimensions.2.1=(names.length : ℤ) ∧ dimensions.2.2=declaration.nTimes ∧
    fieldDefault output "dtype"=.text "float32") "physical output geometry and dtype"
  let scheduled ← integer (fieldDefault output "scheduled_events") "integer scheduled output count"
  ensure (dimensions.1 ≤ scheduled) "scheduled events cover output rows"
  let _ ← hexDigest (fieldDefault output "event_samples_sha256") "source event sample digest"
  let _ ← hexDigest (fieldDefault output "selected_event_indices_sha256") "selected event index digest"
  match output.get "scheduled_event_indices_sha256" with
  | none => pure ()
  | some digest => do
    let _ ← hexDigest digest "scheduled event index digest"
    pure ()
  pure ⟨names,dimensions.1,dimensions.2.1,dimensions.2.2,scheduled⟩

def checkRawScopeContext (p : Primitives) (context output : Fields) (source : SourceGrid)
    (geometry : OutputGeometry) : Check Unit := do
  match context.get "raw_scope" with
  | none => pure ()
  | some value =>
    let record ← objectAt value "raw scope context mapping"
    let scope ← readRawScope p.strip (fieldDefault record "declaration")
    let first ← integer (fieldDefault record "parent_first_samp") "raw scope parent first sample"
    let length ← integer (fieldDefault record "parent_n_times") "raw scope parent length"
    let events ← integer (fieldDefault record "parent_event_count") "raw scope parent event count"
    ensure (0 < length ∧ 0 < events) "positive parent sample and event counts"
    ensure (first ≤ scope.start ∧ scope.stop ≤ first+length) "raw scope parent containment"
    ensure (source.first=scope.start ∧ source.length=scope.stop-scope.start) "executed raw crop grid"
    ensure (scope.eventIndices.getLastD 0 < events ∧ (scope.eventIndices.length : ℤ)=geometry.scheduled)
      "raw scope event range and denominator"
    let eventHash ← p.semanticHash (.sequence (valuesOfList (scope.eventIndices.map Value.integer)))
    ensure (fieldDefault output "scheduled_event_indices_sha256"=.text eventHash)
      "raw scope actual scheduled index digest"
    ensure (fieldDefault record "operator_order"=.text "crop_original_raw_before_filtering_and_epoching")
      "crop raw before filtering and epoching"

abbrev Span := ℤ × ℤ

def readSpans (length : ℤ) (previous : ℤ) : List Value → Check (List Span)
  | [] => .ok []
  | value :: tail => do
    let parts ← integerList value "filter support interval integer bounds"
    let span ← match parts with
      | [start,stop] => .ok (start,stop)
      | _ => .error (.fieldFailure "filter support interval pair")
    ensure (previous ≤ span.1 ∧ span.1 < span.2 ∧ span.2 ≤ length) "ordered contained filter support"
    let spans ← readSpans length span.2 tail
    pure (span :: spans)

def boundaryCauses : List String :=
  ["record_start","record_end","acquisition_discontinuity","artifact","unclassified_reset"]

def checkCauses (value : Value) (sample length : ℤ) : Check Unit := do
  let causes ← textList value "filter boundary cause names"
  ensure (causes ≠ [] ∧ (∀ cause ∈ causes, cause ∈ boundaryCauses) ∧ causes.Pairwise (· < ·))
    "sorted unique nonempty known boundary causes"
  ensure (("record_start" ∈ causes ↔ sample=0) ∧ ("record_end" ∈ causes ↔ sample=length))
    "record causes exactly at natural boundaries"

def checkBoundaryRows (length : ℤ) : List Value → List Span → Check Unit
  | [],[] => .ok ()
  | value :: tail,span :: spans => do
    let fields ← objectAt value "filter boundary row mapping"
    ensure (ExactKeys fields ["start_sample","stop_sample","left_causes","right_causes"])
      "filter boundary exact fields"
    let start ← integer (fieldDefault fields "start_sample") "boundary integer start"
    let stop ← integer (fieldDefault fields "stop_sample") "boundary integer stop"
    ensure ((start,stop)=span) "boundary coordinates agree with support"
    checkCauses (fieldDefault fields "left_causes") start length
    checkCauses (fieldDefault fields "right_causes") stop length
    checkBoundaryRows length tail spans
  | _,_ => .error (.sourceReject "filter boundary count matches support")

def checkFilterSupports (context : Fields) (source : SourceGrid) (geometry : OutputGeometry) : Check Unit := do
  let segments ← objectAt (fieldDefault context "filter_segments") "per-channel filter support mapping"
  ensure (ExactKeys segments geometry.channelNames) "exact per-channel filter support names"
  let boundaries ← match context.get "filter_segment_boundaries" with
    | none => .ok none
    | some value => do
      let fields ← objectAt value "per-channel boundary mapping"
      ensure (ExactKeys fields geometry.channelNames) "exact boundary channel names"
      pure (some fields)
  let rec check : List String → Check Unit
    | [] => .ok ()
    | name :: tail => do
      let values ← listAt (fieldDefault segments name) "filter support span list"
      let spans ← readSpans source.length 0 values
      match boundaries with
      | none => pure ()
      | some fields =>
        let rows ← listAt (fieldDefault fields name) "filter boundary row list"
        checkBoundaryRows source.length rows spans
      check tail
  check geometry.channelNames

def expectedFilterBackend (raw : RawPreprocessing) (declaration : DeclarationScalars) : String :=
  let active := declaration.lFreq.isSome || declaration.hFreq.isSome || !declaration.notches.isEmpty
  if !active then "identity"
  else if raw.phase=.text "forward" then "scipy.sosfilt"
  else if raw.edgePolicy=.text "mne_reflect_limited" ∨ declaration.notches=[] then "mne_reflect_limited"
  else "scipy.sosfiltfilt_odd"

def checkFilter (p : Primitives) (filtering : Fields) (raw : RawPreprocessing)
    (declaration : DeclarationScalars) : Check Unit := do
  let sos ← p.floatArray (fieldDefault filtering "sos")
  if sos.values ≠ [] then
    ensure ((∃ rows, sos.shape=[rows,6]) ∧ (∀ value ∈ sos.values, finiteNumber value=true)) "finite SOS rows by six"
    ensure (∀ row ∈ List.range (sos.shape.headD 0),
      (sos.values[row*6+3]?).bind atomNumber=some 1) "normalized SOS denominator a0"
  let active := declaration.lFreq.isSome || declaration.hFreq.isSome || !declaration.notches.isEmpty
  let backend := expectedFilterBackend raw declaration
  ensure ((decide (sos.values ≠ []))=active ∧ fieldDefault filtering "backend"=.text backend)
    "filter presence and backend match declaration"
  let padding ← integer (fieldDefault filtering "padding_limit_samples") "integer filter padding"
  ensure ((backend="identity" ∨ backend="scipy.sosfilt") → padding=0) "identity or forward has zero padding"
  ensure (0 ≤ padding ∧ fieldDefault filtering "initial_state"=raw.initialState) "filter boundary and initial state"

structure Receipt where
  original : Fields
  declaration : Fields
  source : SourceGrid
  geometry : OutputGeometry

def checkIndividual (p : Primitives) (expectedArgs : Fields) (fields : Fields) : Check Receipt := do
  ensure (fieldDefault fields "schema"=.text "n2p3_preprocessing_execution/1") "individual execution receipt schema"
  let hash ← p.semanticHash (.object (fields.erase "receipt_sha256"))
  ensure (fieldDefault fields "receipt_sha256"=.text hash) "execution receipt semantic SHA"
  let declared ← objectAt (fieldDefault fields "declaration") "complete execution declaration mapping"
  let declared ← readCompletePreprocessingArgs declared
  let raw := rawPreprocessing declared
  let declaration ← checkPreprocessing p.strip p.floorSamples p.windowSamples raw
  ensure (PersonalAdmission.PyEq (.object (expectedArgs.erase "name")) (.object (declared.erase "name")))
    "executed physical declaration excluding display name"
  ensure (PersonalAdmission.PyEq (fieldDefault fields "semantic_identity") operatorIdentity) "supported operator identity"
  checkImplementation fields
  let context ← objectAt (fieldDefault fields "source_context") "execution source context mapping"
  let grid ← objectAt (fieldDefault fields "epoch_grid") "execution epoch grid mapping"
  let output ← objectAt (fieldDefault fields "output") "execution output mapping"
  let filtering ← objectAt (fieldDefault fields "filter") "execution filter mapping"
  let source ← readSourceGrid context
  checkSourceBands declaration raw source
  let expectedGrid ← expectedEpochGrid p declaration source
  ensure (PersonalAdmission.PyEq (.object grid) (.object expectedGrid)) "actual closed-source epoch grid"
  let geometry ← readOutputGeometry output declaration
  checkRawScopeContext p context output source geometry
  checkFilterSupports context source geometry
  checkFilter p filtering raw declaration
  let _ ← nonemptyText (fieldDefault fields "timing_source") "nonempty execution timing source"
  pure ⟨fields,declared,source,geometry⟩

def checkChildren (p : Primitives) (expectedArgs : Fields) : List Value → Check (List Receipt)
  | [] => .ok []
  | value :: tail => do
    let fields ← objectAt value "individual bundle child mapping"
    ensure (fieldDefault fields "schema"=.text "n2p3_preprocessing_execution/1") "bundle child individual receipt schema"
    let receipt ← checkIndividual p expectedArgs fields
    let receipts ← checkChildren p expectedArgs tail
    pure (receipt :: receipts)

structure CheckedRecord where
  original : Fields
  receipts : List Receipt
  unverifiedInputs : Value

def checkRecord (p : Primitives) (expectedArgs : Fields) (value : Value) : Check CheckedRecord := do
  let fields ← objectAt value "execution receipt dictionary"
  if fieldDefault fields "schema"=.text "n2p3_preprocessing_execution_bundle/1" then
    let children ← listAt (fieldDefault fields "recordings") "execution bundle receipt list"
    let unknown ← integer (fieldDefault fields "unverified_inputs") "bundle nonbool unknown count"
    ensure (0 ≤ unknown ∧ (children ≠ [] ∨ unknown ≠ 0)) "nonempty execution bundle or unknown inputs"
    let receipts ← checkChildren p expectedArgs children
    pure ⟨fields,receipts,.integer unknown⟩
  else
    let receipt ← checkIndividual p expectedArgs fields
    pure ⟨fields,[receipt],(fields.get "unverified_inputs").getD (.integer 0)⟩

structure Binding where
  executionIdentity : Option Value
  provenanceHash : String

def transferableIdentity (record : CheckedRecord) : Option Value :=
  if PyTruthy record.unverifiedInputs ∨ record.receipts=[] then none else some operatorIdentity

/-- Absence returns an empty binding without validating the declaration. -/
def executionBinding (p : Primitives) (expectedArgs : Fields) (value : Option Value) : Check (Option Binding) := do
  match value with
  | none | some .null => pure none
  | some value =>
    let record ← checkRecord p expectedArgs value
    let digest ← p.semanticHash (.object record.original)
    pure (some ⟨transferableIdentity record,digest⟩)

def bindingFields : Option Binding → Fields
  | none => .nil
  | some binding => Fields.ofList
    [("execution_identity",binding.executionIdentity.getD .null),
     ("execution_provenance_sha256",.text binding.provenanceHash)]

theorem absent_execution_does_not_validate_preprocessing (p : Primitives) (expectedArgs : Fields) :
    executionBinding p expectedArgs none=.ok none := rfl

theorem absent_execution_binding_is_empty : bindingFields none=.nil := rfl

theorem unknown_input_never_gets_supported_identity (record : CheckedRecord)
    (h : PyTruthy record.unverifiedInputs) : transferableIdentity record=none := by
  simp [transferableIdentity,h]

theorem no_receipt_never_gets_supported_identity (record : CheckedRecord)
    (h : record.receipts=[]) : transferableIdentity record=none := by
  simp [transferableIdentity,h]

theorem supported_identity_needs_all_inputs_evidenced (record : CheckedRecord) :
    transferableIdentity record=some operatorIdentity ↔ ¬PyTruthy record.unverifiedInputs ∧ record.receipts ≠ [] := by
  by_cases hu : PyTruthy record.unverifiedInputs
  · simp [transferableIdentity,hu]
  · by_cases hr : record.receipts=[] <;> simp [transferableIdentity,hu,hr]

theorem supported_identity_not_record_identity (a b : CheckedRecord)
    (ha : ¬PyTruthy a.unverifiedInputs ∧ a.receipts ≠ [])
    (hb : ¬PyTruthy b.unverifiedInputs ∧ b.receipts ≠ []) :
    transferableIdentity a=transferableIdentity b := by
  rw [(supported_identity_needs_all_inputs_evidenced a).2 ha,
    (supported_identity_needs_all_inputs_evidenced b).2 hb]

theorem read_source_grid_proves_positive_rate_and_length (context : Fields) (grid : SourceGrid)
    (h : readSourceGrid context=.ok grid) : 0 < grid.rate ∧ 0 < grid.length := by
  unfold readSourceGrid at h
  rcases (bind_success _ _ _).mp h with ⟨rate,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hr,h⟩
  have hrate := ensure_success _ _ completion hr
  rcases (bind_success _ _ _).mp h with ⟨first,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨length,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hl,h⟩
  have hlength := ensure_success _ _ completion hl
  have he : (⟨rate,first,length⟩ : SourceGrid)=grid := Except.ok.inj h
  rw [←he]
  exact ⟨hrate,hlength⟩

theorem read_raw_scope_proves_explicit_bounds_and_events (strip : String → String)
    (value : Value) (scope : RawScope) (h : readRawScope strip value=.ok scope) :
    strip scope.name ≠ "" ∧ 0 ≤ scope.start ∧ scope.start < scope.stop ∧ scope.eventIndices ≠ [] ∧
      (∀ event ∈ scope.eventIndices, 0 ≤ event) ∧ scope.eventIndices.Pairwise (· < ·) := by
  unfold readRawScope at h
  rcases (bind_success _ _ _).mp h with ⟨fields,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨_,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨name,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hn,h⟩
  have hname := ensure_success _ _ completion hn
  rcases (bind_success _ _ _).mp h with ⟨start,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨stop,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hb,h⟩
  have hbounds := ensure_success _ _ completion hb
  rcases (bind_success _ _ _).mp h with ⟨indices,_,h⟩
  rcases (bind_success _ _ _).mp h with ⟨completion,hi,h⟩
  have hevents := ensure_success _ _ completion hi
  have he : (⟨name,start,stop,indices⟩ : RawScope)=scope := Except.ok.inj h
  rw [←he]
  exact ⟨hname,hbounds.1,hbounds.2,hevents.1,hevents.2.1,hevents.2.2⟩

end

end N2P3.PersonalERPExecution
