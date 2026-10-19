import N2P3.PersonalERPInput

/-!
# Saved raw requests, prospective sessions and explicit reference stages

Source: calibrated.py:665-750,815-824; brainsync.py:113-150,645-668.
These are literal field/control components. Frozen QC/EOG readers, finalized
BIDS/clock fields, the numerical raw stages and final Input composition remain
separate definitions to connect before P06 acceptance. A reference view below
is a projection of the fields actually consumed by resolve_brainsync_reference,
never a predicate certifying an entire fitted model or physical recording.
-/

namespace N2P3.PersonalERPRaw

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def tupleValue : Value → Check (List Value)
  | .sequence values => .ok (valuesToList values)
  | .text value => .ok (value.toList.map (fun char => .text (String.mk [char])))
  | .object fields => .ok ((fieldKeys fields).map Value.text)
  | _ => .error (.fieldFailure "iterable tuple conversion")

def hashableJSON : Value → Prop
  | .sequence _ | .object _ => False
  | _ => True

def readCalibrationIds : List Value → Check (List Value)
  | [] => .ok []
  | value :: tail => do
    let row ← objectAt value "calibration session row mapping"
    let id ← required row "session_id"
    ensure (hashableJSON id) "hashable calibration session id"
    let rest ← readCalibrationIds tail
    pure (id :: rest)

def sessionGuard (metadata : Fields) (id : String) (started : PersonalAdmission.Time) : Check Unit := do
  let sessions ← required metadata "calibration_sessions"
  let sessions ← listAt sessions "calibration session list"
  let ids ← readCalibrationIds sessions
  ensure (.text id ∉ ids) "calibration recording cannot be reused for inference"
  let available ← required metadata "calibration_available_timestamp_s"
  let available ← match numericScalar available with
    | some available => .ok available
    | none => .error (.fieldFailure "numeric calibration availability")
  ensure (¬started.notLater available) "inference starts strictly after calibration availability"

def sessionMetadata (ids : List String) (available : ℝ) : Fields := Fields.ofList
  [("calibration_sessions",.sequence (PersonalAdmission.calibrationSessionRows ids)),
   ("calibration_available_timestamp_s",.number available)]

/-- Per-cell results of the actual group-local to_numpy(dtype=float)
conversion. Neither finiteness, agreement nor successful conversion is
assumed. Missing session keys are dropped before these cells are consumed.
Present keys here are the official producer's String subdomain; arbitrary
pandas object grouping/stringification remains a separate representation step. -/
structure TimingRow where
  session : Option String
  started : Check PersonalAdmission.Time
  ended : Check PersonalAdmission.Time
  labelAvailable : Check PersonalAdmission.Time

structure TimingTable where
  columns : List String
  rows : List TimingRow

structure ScopeSession where
  id : String
  rows : ℕ
  started : ℝ
  ended : Option ℝ
  labelAvailable : Option ℝ

def readTimeCells : List (Check PersonalAdmission.Time) → Check (List PersonalAdmission.Time)
  | [] => .ok []
  | cell :: tail => do
    let value ← cell
    let values ← readTimeCells tail
    pure (value :: values)

def readUniformTime (cells : List (Check PersonalAdmission.Time)) : Check ℝ := do
  let times ← readTimeCells cells
  match times with
  | .finite first :: tail =>
    ensure (∀ time ∈ tail, time=.finite first) "consistent finite session timestamp"
    pure first
  | _ => .error (.sourceReject "consistent finite session timestamp")

def timingIds (rows : List TimingRow) : List String :=
  (rows.filterMap TimingRow.session).foldl PersonalAdmission.insertFresh []

def readScopeGroup (rows : List TimingRow) (calibration : Bool) (id : String) : Check ScopeSession := do
  let group := rows.filter (fun row => decide (row.session=some id))
  let started ← readUniformTime (group.map TimingRow.started)
  if calibration then
    let ended ← readUniformTime (group.map TimingRow.ended)
    let available ← readUniformTime (group.map TimingRow.labelAvailable)
    ensure (started < ended ∧ ended ≤ available) "calibration availability after complete recording"
    pure ⟨id,group.length,started,some ended,some available⟩
  else pure ⟨id,group.length,started,none,none⟩

def readScopeGroups (rows : List TimingRow) (calibration : Bool) : List String → Check (List ScopeSession)
  | [] => .ok []
  | id :: tail => do
    let group ← readScopeGroup rows calibration id
    let groups ← readScopeGroups rows calibration tail
    pure (group :: groups)

def sessionScope (table : TimingTable) (calibration : Bool) : Check (List ScopeSession) := do
  let required := if calibration then
    ["session","session_start_timestamp_s","session_end_timestamp_s","calibration_available_timestamp_s"]
    else ["session","session_start_timestamp_s"]
  ensure (required ⊆ table.columns) "required session timing columns"
  let groups ← readScopeGroups table.rows calibration (timingIds table.rows)
  pure (groups.mergeSort (fun a b => decide (a.started < b.started ∨ a.started=b.started ∧ a.id ≤ b.id)))

def checkScopeSessions (metadata : Fields) : List ScopeSession → Check Unit
  | [] => .ok ()
  | group :: tail => do
    sessionGuard metadata group.id (.finite group.started)
    checkScopeSessions metadata tail

def validateInferenceScope (metadata : Fields) (table : TimingTable) : Check Unit := do
  let sessions ← sessionScope table false
  checkScopeSessions metadata sessions

def expectedProcessingModel (contract : Fields) (key : String) : Check Value := do
  let processing ← objectAt ((contract.get key).getD (.object .nil)) "saved processing mapping"
  if fieldDefault processing "status"=.text "applied" then required processing "model" else pure .null

structure SavedRequest where
  original : Fields
  contract : Fields
  channels : List Value
  preprocessingArgs : Fields
  ocularRecord : Value
  continuousRecord : Value

def checkSavedRequest (metadata : Fields) : Check SavedRequest := do
  let request ← objectAt (fieldDefault metadata "input_request") "saved input request mapping"
  let contract ← objectAt (fieldDefault metadata "input_contract") "saved physical input mapping"
  let channels ← required request "channel_names"
  let channels ← tupleValue channels
  let expected ← required contract "channel_names"
  let expected ← tupleValue expected
  ensure (PersonalAdmission.PyEq (.sequence (valuesOfList channels)) (.sequence (valuesOfList expected)))
    "saved ordered request channels agree with contract"
  let ocular ← expectedProcessingModel contract "ocular_artifact_processing"
  ensure (PersonalAdmission.PyEq (fieldDefault request "ocular_model") ocular) "saved request ocular model"
  let continuous ← expectedProcessingModel contract "continuous_artifact_processing"
  ensure (PersonalAdmission.PyEq (fieldDefault request "continuous_qc_model") continuous) "saved request continuous model"
  let preprocessing ← objectAt (fieldDefault contract "preprocessing") "saved preprocessing kwargs mapping"
  let preprocessing ← readPreprocessingArgs preprocessing
  pure ⟨request,contract,channels,preprocessing,fieldDefault request "ocular_model",
    fieldDefault request "continuous_qc_model"⟩

structure OcularReferenceView where
  reference : Option String

structure ContinuousReferenceView where
  reference : Option String
  channelLocal : Bool

def canonicalReference (p : PersonalERPInput.Primitives) (value : Value) : Check (Option String) := do
  if value=.null then pure none
  else
    let rendered ← p.pythonString value
    let channel ← p.canonicalChannel rendered
    pure (some channel)

def canonicalAnchorList (p : PersonalERPInput.Primitives) : List Value → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let reference ← canonicalReference p value
    let rest ← canonicalAnchorList p tail
    pure (reference.toList ++ rest)

def resolveReference (p : PersonalERPInput.Primitives) (requested mode : Value)
    (ocular : Option OcularReferenceView) (continuous : Option ContinuousReferenceView) : Check (Option String) := do
  ensure (mode=.text "auto" ∨ mode=.text "native") "BrainSync reference mode"
  let anchors ← canonicalAnchorList p [requested,
    ((ocular.bind OcularReferenceView.reference).map Value.text).getD .null,
    ((continuous.bind ContinuousReferenceView.reference).map Value.text).getD .null]
  ensure (anchors.toFinset.card ≤ 1) "request ocular and QC references agree"
  ensure (mode ≠ .text "native" ∨ anchors=[]) "native reference has no measured anchor"
  let reference := anchors.head?
  match ocular with
  | none => pure ()
  | some model => ensure (model.reference=reference) "requested reference matches fitted ocular reference"
  pure reference

def rawReference (ocular : Option OcularReferenceView) (continuous : Option ContinuousReferenceView) : Option String :=
  match ocular with
  | some model => model.reference
  | none => continuous.bind (fun model => if model.channelLocal then model.reference else none)

def needsPostReference (reference : Option String) (ocular : Option OcularReferenceView)
    (continuous : Option ContinuousReferenceView) : Bool :=
  reference.isSome && decide (rawReference ocular continuous ≠ reference)

def checkSavedReference (contract : Fields) (reference : Option String) : Check Unit := do
  let source ← required contract "source_reference"
  match reference with
  | some anchor => ensure (source=.text ("measured_channel:" ++ anchor)) "saved measured raw reference"
  | none =>
    let source ← text source "source reference startswith string"
    ensure (PersonalERPInput.startsWith "measured_channel:" source=false) "native saved raw reference"

structure RecordingCallbackFields where
  subject : String
  session : String
  started : PersonalAdmission.Time
  manifest : Fields
  acquisitionReference : String
  acquisitionDetails : Fields

def acquisitionReferenceMatches (references : List Value) (label : String) (details : Fields) : Check Bool := do
  let rec find : List Value → Check Bool
    | [] => .ok false
    | value :: tail => do
      let row ← objectAt value "acquisition reference row mapping"
      let actualLabel ← required row "label"
      if actualLabel=.text label then
        let actualDetails ← required row "details"
        if PersonalAdmission.PyEq actualDetails (.object details) then pure true else find tail
      else find tail
  find references

def checkAcquisitionReference (request : Fields) (recording : RecordingCallbackFields) : Check Unit := do
  let references := (request.get "acquisition_references").getD (.sequence .nil)
  if PyTruthy references then
    let references ← listAt references "acquisition reference list"
    let matched ← acquisitionReferenceMatches references recording.acquisitionReference recording.acquisitionDetails
    ensure (matched=true) "recording acquisition reference matches calibration"

def simulationStatus (manifest : Fields) : String :=
  let transport := match fieldDefault manifest "device" with
    | .object device => fieldDefault device "transport"
    | _ => .null
  if transport=.text "mock" then "declared_mock" else "not_declared"

def recordingAuthority (p : PersonalERPInput.Primitives) (subject : String) : Check String :=
  PersonalERPInput.originKey p "BrainSync-GTN" subject

def checkRecordingIdentityTimeSimulationReference (p : PersonalERPInput.Primitives) (metadata : Fields)
    (recording : RecordingCallbackFields) : Check Unit := do
  let participant ← recordingAuthority p recording.subject
  ensure (fieldDefault metadata "participant_key"=.text participant) "raw recording authoritative participant"
  sessionGuard metadata recording.session recording.started
  let contract ← objectAt (fieldDefault metadata "input_contract") "raw callback input contract mapping"
  ensure (fieldDefault contract "simulation_status"=.text (simulationStatus recording.manifest)) "raw recording simulation status"
  let request ← objectAt (fieldDefault metadata "input_request") "raw callback input request mapping"
  checkAcquisitionReference request recording

theorem tuple_of_JSON_string_is_characters (value : String) :
    tupleValue (.text value)=.ok (value.toList.map (fun char => .text (String.mk [char]))) := rfl

theorem missing_processing_status_does_not_activate_model (contract : Fields) (key : String)
    (h : contract.get key=none) : expectedProcessingModel contract key=.ok .null := by
  simp [expectedProcessingModel,h,objectAt,fieldDefault,Fields.get]

theorem ocular_reference_takes_raw_reference_priority (ocular : OcularReferenceView)
    (continuous : Option ContinuousReferenceView) : rawReference (some ocular) continuous=ocular.reference := rfl

theorem no_resolved_reference_needs_no_post_reference (ocular : Option OcularReferenceView)
    (continuous : Option ContinuousReferenceView) : needsPostReference none ocular continuous=false := rfl

theorem no_acquisition_reference_budget_skips_only_that_check (recording : RecordingCallbackFields) :
    checkAcquisitionReference .nil recording=.ok () := by
  simp [checkAcquisitionReference,Fields.get,PyTruthy]

theorem mock_transport_has_explicit_simulation_status (tail : Fields) :
    simulationStatus (.cons "device" (.object (.cons "transport" (.text "mock") .nil)) tail)="declared_mock" := by
  simp [simulationStatus,fieldDefault,Fields.get]

theorem read_canonical_calibration_ids (ids : List String) :
    readCalibrationIds (valuesToList (PersonalAdmission.calibrationSessionRows ids))=.ok (ids.map Value.text) := by
  induction ids with
  | nil => rfl
  | cons id tail ih =>
    simp [PersonalAdmission.calibrationSessionRows,valuesToList,readCalibrationIds,objectAt,required,
      Fields.ofList,Fields.get,hashableJSON,ensure,ih]

theorem canonical_session_guard_formula (ids : List String) (available : ℝ) (id : String)
    (started : PersonalAdmission.Time) :
    sessionGuard (sessionMetadata ids available) id started=
      if id ∈ ids then .error (.sourceReject "calibration recording cannot be reused for inference")
      else if started.notLater available then
        .error (.sourceReject "inference starts strictly after calibration availability") else .ok () := by
  simp [sessionGuard,sessionMetadata,required,Fields.ofList,Fields.get,listAt,
    read_canonical_calibration_ids,numericScalar,ensure]
  split_ifs <;> simp_all

theorem reused_session_refused_even_after_labels (ids : List String) (available later : ℝ) (id : String)
    (h : id ∈ ids) : sessionGuard (sessionMetadata ids available) id (.finite later)=
      .error (.sourceReject "calibration recording cannot be reused for inference") := by
  simp [canonical_session_guard_formula,h]

theorem equality_with_label_availability_refused (ids : List String) (available : ℝ) (id : String)
    (h : id ∉ ids) : sessionGuard (sessionMetadata ids available) id (.finite available)=
      .error (.sourceReject "inference starts strictly after calibration availability") := by
  simp [canonical_session_guard_formula,h,PersonalAdmission.Time.notLater]

theorem direct_nan_start_still_passes_comparison (ids : List String) (available : ℝ) (id : String)
    (h : id ∉ ids) : sessionGuard (sessionMetadata ids available) id .nan=.ok () := by
  simp [canonical_session_guard_formula,h,PersonalAdmission.Time.notLater]

theorem empty_column_complete_metadata_still_passes_scope (metadata : Fields) :
    validateInferenceScope metadata ⟨["session","session_start_timestamp_s"],[]⟩=.ok () := by
  simp [validateInferenceScope,sessionScope,ensure,readScopeGroups,timingIds,checkScopeSessions]

theorem all_missing_session_keys_skip_timestamp_conversions (rows : List TimingRow)
    (h : ∀ row ∈ rows, row.session=none) : timingIds rows=[] := by
  have he : rows.filterMap TimingRow.session=[] := List.filterMap_eq_nil_iff.mpr h
  simp [timingIds,he]

theorem all_missing_session_keys_still_pass_scope (metadata : Fields) (rows : List TimingRow)
    (h : ∀ row ∈ rows, row.session=none) :
    validateInferenceScope metadata ⟨["session","session_start_timestamp_s"],rows⟩=.ok () := by
  simp [validateInferenceScope,sessionScope,ensure,all_missing_session_keys_skip_timestamp_conversions rows h,
    readScopeGroups,checkScopeSessions]

end

end N2P3.PersonalERPRaw
