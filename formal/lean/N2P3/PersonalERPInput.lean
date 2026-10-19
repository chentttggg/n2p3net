import N2P3.PersonalERPExecution
import N2P3.PersonalERPIdentity

/-!
# Physical dataset, participant and temporal input composition

Source: calibrated.py:465-491,752-766; epochs.py:325-471,706-730;
identity.py:29-149,182-304; exclusions.py:19-67; temporal_transfer.py:227-238.
All physical/identity/execution gates are explicit field checks. Individual
string conversion, channel normalization, URI quote/split/unquote, decimal
parsing, ndarray norms and floating scalar operations remain local operations.

The context carries the mutable identity-table cache. A no-table source
construction is not silently substituted for the existing-table coverage
branch. In particular cleaning a subject label can affect the next materialize
call; no immutability or stronger metadata schema is invented for the source.
JSON/array images are logical field representations, not proof of genuine raw
producer identity or actual byte-level Python/NumPy implementation refinement.
This is P06 work in progress; constructor/raw/entrypoint connections are still
required before a complete admission claim.
-/

namespace N2P3.PersonalERPInput

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPExecution

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure URIImage where
  scheme : String
  netloc : String
  path : String
  pathParts : List String

structure Primitives where
  execution : PersonalERPExecution.Primitives
  pythonString : Value → Check String
  canonicalChannel : String → Check String
  coordinateNorm : CoordinateNorm
  quoteURI : String → String
  unquoteURI : String → String
  splitURI : String → Check URIImage
  decimalInteger : String → Option ℤ

def cleanString (p : Primitives) (value : String) : Check String := do
  let cleaned := p.execution.strip value
  ensure (cleaned ≠ "" ∧ ∀ char ∈ cleaned.toList, char.toNat ≠ 0) "nonempty identifier without NUL"
  pure cleaned

def cleanIdentifier (p : Primitives) (value : Value) : Check String := do
  let value ← p.pythonString value
  cleanString p value

def cleanStrings (p : Primitives) : List String → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let s ← cleanString p value
    let rest ← cleanStrings p tail
    pure (s :: rest)

structure IdentityRecord where
  localSubject : String
  origins : List String
  globals : List String
  status : String

def identityRecordFields (row : IdentityRecord) : Fields := Fields.ofList
  [("local_subject_id",.text row.localSubject),("origin_subject_keys",.sequence (encodeTexts row.origins)),
   ("global_person_keys",.sequence (encodeTexts row.globals)),("identity_status",.text row.status)]

def readIdentityRecord (p : Primitives) (value : Value) : Check IdentityRecord := do
  let fields ← objectAt value "participant identity record mapping"
  ensure (ExactKeys fields ["local_subject_id","origin_subject_keys","global_person_keys","identity_status"])
    "participant identity exact fields"
  let localSubject ← cleanIdentifier p (fieldDefault fields "local_subject_id")
  let origins ← textList (fieldDefault fields "origin_subject_keys") "origin key string list"
  let globals ← textList (fieldDefault fields "global_person_keys") "global key string list"
  let origins ← cleanStrings p origins
  let globals ← cleanStrings p globals
  let origins := origins.dedup.mergeSort (· ≤ ·)
  let globals := globals.dedup.mergeSort (· ≤ ·)
  let status ← p.pythonString (fieldDefault fields "identity_status")
  ensure (status ∈ ["source_verified","global_verified","cross_study_unknown"]) "identity status enum"
  ensure (status="source_verified" → origins ≠ []) "source identity needs origins"
  ensure (status="global_verified" → origins ≠ [] ∧ globals ≠ []) "global identity needs source and global keys"
  ensure (globals ≠ [] → status="global_verified") "global keys require verified status"
  ensure (status="cross_study_unknown" → origins=[] ∧ globals=[]) "unknown identity has no authority keys"
  pure ⟨localSubject,origins,globals,status⟩

def readIdentityRecords (p : Primitives) : List Value → Check (List IdentityRecord)
  | [] => .ok []
  | value :: tail => do
    let row ← readIdentityRecord p value
    let rows ← readIdentityRecords p tail
    pure (row :: rows)

def identityCanonicalPayload (rows : List IdentityRecord) : Fields := Fields.ofList
  [("schema",.text "n2p3net_dataset_identity_table/1"),
   ("records",.sequence (valuesOfList (rows.map (fun row => .object (identityRecordFields row)))))]

def checkIdentityTableConstructor (rows : List IdentityRecord) : Check (List IdentityRecord) := do
  let sorted := rows.mergeSort (fun a b => a.localSubject ≤ b.localSubject)
  ensure (sorted ≠ [] ∧ (sorted.map IdentityRecord.localSubject).Nodup) "nonempty unique identity table subjects"
  pure sorted

def readIdentityPayload (p : Primitives) (value : Value) : Check (List IdentityRecord) := do
  let fields ← objectAt value "identity table payload mapping"
  ensure (ExactKeys fields ["schema","records","digest"]) "identity payload exact fields"
  let values ← listAt (fieldDefault fields "records") "identity record list"
  let rows ← readIdentityRecords p values
  let schema ← p.pythonString (fieldDefault fields "schema")
  ensure (schema="n2p3net_dataset_identity_table/1") "dataset identity schema"
  let rows ← checkIdentityTableConstructor rows
  let digest ← p.execution.semanticHash (.object (identityCanonicalPayload rows))
  ensure (fieldDefault fields "digest"=.text digest) "identity table canonical digest"
  pure rows

def authorityKey (row : IdentityRecord) : Check String :=
  let keys := if row.globals=[] then row.origins else row.globals
  match keys with
  | [key] => .ok key
  | _ => .error (.sourceReject "exactly one source_or_global authority key")

def originKey (p : Primitives) (source subject : String) : Check String := do
  let source ← cleanString p source
  let subject ← cleanString p subject
  pure ("n2p3-origin://" ++ p.quoteURI source ++ "/participant/" ++ p.quoteURI subject)

def startsWith (stem value : String) : Bool := stem.toList.isPrefixOf value.toList

def removePrefix (stem value : String) : String :=
  if startsWith stem value then String.mk (value.toList.drop stem.toList.length) else value

def sourceParticipantExcluded (p : Primitives) (source subject : String) : Bool :=
  let parsed := p.decimalInteger (removePrefix "subject_" subject)
  if source="BI2014a" ∨ source="BrainInvaders2014a" ∨ startsWith "BI2014a-candidate:" source then
    parsed.any (fun subject => decide (subject ∈ ([62,63,64] : List ℤ)))
  else if source="BI2015a" ∨ source="BrainInvaders2015a" ∨ startsWith "BI2015a-candidate:" source then
    parsed.any (fun subject => decide (subject ∈ ([1,27,44,45,46,47,48,49,50] : List ℤ)))
  else false

def checkOriginAllowed (p : Primitives) (origin : String) : Check Unit := do
  let parsed ← p.splitURI origin
  if parsed.scheme="n2p3-origin" ∧ parsed.pathParts.length=3 ∧ parsed.pathParts[1]?=some "participant" then
    let subject ← match parsed.pathParts[2]? with
      | some subject => .ok subject
      | none => .error (.fieldFailure "origin participant path")
    ensure (sourceParticipantExcluded p (p.unquoteURI parsed.netloc) (p.unquoteURI subject)=false)
      "author admitted source participant"

def checkOriginsAllowed (p : Primitives) : List String → Check Unit
  | [] => .ok ()
  | origin :: tail => do
    checkOriginAllowed p origin
    checkOriginsAllowed p tail

def checkIdentityExclusions (p : Primitives) : List IdentityRecord → Check Unit
  | [] => .ok ()
  | row :: tail => do
    checkOriginsAllowed p row.origins
    checkIdentityExclusions p tail

def checkScheduledExclusions (p : Primitives) {M : ℕ} (timeline : TimelineImage M) : Check Unit :=
  ensure (∀ i, sourceParticipantExcluded p (timeline.datasets i) (timeline.subjects i)=false)
    "scheduled source participants author admitted"

structure Context (N C T M : ℕ) where
  data : DatasetImage N C T M
  preprocessingArgs : Fields
  provenance : Fields
  identityCache : Option (List IdentityRecord)
  lineage : LineageImage
  attestation : Option Value

def subjects {N C T M : ℕ} (context : Context N C T M) : List String := List.ofFn context.data.subjects

def checkDataset (p : Primitives) {N C T M : ℕ} (context : Context N C T M) (requireLabels : Bool) : Check Unit := do
  checkPhysicalCoordinates context.data.historical context.data.coordinateSystem (context.provenance.get "data_alignment")
  match context.identityCache with
  | none => pure ()
  | some rows => checkIdentityExclusions p rows
  checkScheduledExclusions p context.data.timeline
  let args ← readPreprocessingArgs context.preprocessingArgs
  let _ ← checkDatasetBody p.execution.strip p.canonicalChannel p.execution.floorSamples p.execution.windowSamples
    p.coordinateNorm (rawPreprocessing args) context.data requireLabels
  let _ ← executionBinding p.execution args (context.provenance.get "preprocessing_execution")
  checkDatasetTail (subjects context) (context.identityCache.map (fun rows => rows.map IdentityRecord.localSubject))
    context.lineage context.attestation

def sourceForRow {N M : ℕ} (timeline : TimelineImage M) (row : Fin N) : Check String :=
  match (List.finRange M).find? (fun event => decide (timeline.statuses event="available" ∧
      timeline.evidence event=(row.val : ℤ))) with
  | none => .error (.fieldFailure "available event for original epoch row")
  | some event => .ok (timeline.datasets event)

def sourceRowPairs (p : Primitives) {N C T M : ℕ} (context : Context N C T M) :
    List (Fin N) → Check (List (String × String))
  | [] => .ok []
  | row :: tail => do
    let localSubject ← cleanString p (context.data.subjects row)
    let source ← sourceForRow context.data.timeline row
    let origin ← originKey p source (context.data.subjects row)
    let rest ← sourceRowPairs p context tail
    pure ((localSubject,origin) :: rest)

def constructSourceRows (pairs : List (String × String)) : Check (List IdentityRecord) := do
  ensure (pairs ≠ []) "source identity aligned nonempty rows"
  let locals := (pairs.map Prod.fst).dedup
  checkIdentityTableConstructor (locals.map fun localSubject =>
    ⟨localSubject,((pairs.filter (fun row => decide (row.1=localSubject))).map Prod.snd).dedup.mergeSort (· ≤ ·),
      [],"source_verified"⟩)

def materializeIdentity (p : Primitives) {N C T M : ℕ} (context : Context N C T M) :
    Check (List IdentityRecord × Context N C T M) := do
  match context.identityCache with
  | some rows =>
    ensure ((rows.map IdentityRecord.localSubject).toFinset=(subjects context).toFinset)
      "materialized identity exact original subject coverage"
    pure (rows,context)
  | none =>
    checkTimeline p.execution.strip context.data.timeline N
    let pairs ← sourceRowPairs p context (List.finRange N)
    let rows ← constructSourceRows pairs
    pure (rows,{ context with identityCache := some rows })

def recordFor (p : Primitives) (rows : List IdentityRecord) (subject : String) : Check IdentityRecord := do
  let subject ← cleanString p subject
  match rows.filter (fun row => decide (row.localSubject=subject)) with
  | [row] => pure row
  | _ => .error (.sourceReject "unique participant record for local subject")

def participant (p : Primitives) {N C T M : ℕ} (context : Context N C T M) :
    Check (String × String × Context N C T M) := do
  let locals := (subjects context).dedup.mergeSort (· ≤ ·)
  let subject ← match locals with
    | [subject] => .ok subject
    | _ => .error (.sourceReject "exactly one personalized participant")
  let materialized ← materializeIdentity p context
  let row ← recordFor p materialized.1 subject
  let key ← authorityKey row
  pure (subject,key,materialized.2)

def atomJSON : Atom → Check Value
  | .boolean b => .ok (.boolean b)
  | .integer z => .ok (.integer z)
  | .real r => .ok (.number r)
  | .null => .ok .null
  | _ => .error (.fieldFailure "finite JSON scalar encoding")

def atomJSONList : List Atom → Check (List Value)
  | [] => .ok []
  | atom :: tail => do
    let value ← atomJSON atom
    let values ← atomJSONList tail
    pure (value :: values)

def coordinateJSON {C : ℕ} (coordinates : Fin C → Fin 3 → Atom) : List (Fin C) → Check (List Value)
  | [] => .ok []
  | channel :: tail => do
    let row ← atomJSONList (List.ofFn (coordinates channel))
    let rows ← coordinateJSON coordinates tail
    pure (.sequence (valuesOfList row) :: rows)

def inputContract (p : Primitives) {N C T M : ℕ} (context : Context N C T M) : Check Fields := do
  let reference ← text (fieldDefault context.provenance "source_reference") "explicit physical reference string"
  ensure (p.execution.strip reference ≠ "") "explicit nonblank physical reference"
  let positions ← coordinateJSON context.data.coordinates (List.finRange C)
  let args ← readPreprocessingArgs context.preprocessingArgs
  pure (Fields.ofList
    [("channel_names",.sequence (encodeTexts context.data.channelNames)),
     ("channel_positions_m",.sequence (valuesOfList positions)),
     ("preprocessing",.object (preprocessingRecord args)),("source_reference",.text reference),
     ("n_channels",.integer C),("n_times",.integer T),
     ("simulation_status",(context.provenance.get "simulation_status").getD (.text "not_declared")),
     ("ocular_artifact_processing",(context.provenance.get "ocular_artifact_processing").getD
       (.object (Fields.ofList [("status",.text "not_applied")]))),
     ("continuous_artifact_processing",(context.provenance.get "continuous_artifact_processing").getD
       (.object (Fields.ofList [("status",.text "not_applied")])))])

def originStudy (p : Primitives) (origin : String) : Check String := do
  let parsed ← p.splitURI origin
  ensure (parsed.scheme="n2p3-origin" ∧ parsed.netloc ≠ "" ∧ startsWith "/participant/" parsed.path)
    "canonical source library origin"
  pure (p.unquoteURI parsed.netloc)

def rowStudies (p : Primitives) (row : IdentityRecord) : Check (List String) := do
  ensure (row.origins ≠ []) "source library needs canonical origins"
  PersonalERPIdentity.originStudies (originStudy p) row.origins

def identityKeyView (row : IdentityRecord) : Check IdentityImage := do
  let authority ← authorityKey row
  pure ⟨authority,row.origins,row.globals,[]⟩

def identityKeyViews : List IdentityRecord → Check (List IdentityImage)
  | [] => .ok []
  | row :: tail => do
    let view ← identityKeyView row
    let views ← identityKeyViews tail
    pure (view :: views)

def allIdentityStudies (p : Primitives) : List IdentityRecord → Check (List String)
  | [] => .ok []
  | row :: tail => do
    let studies ← rowStudies p row
    let rest ← allIdentityStudies p tail
    pure (studies ++ rest)

def temporalTarget (p : Primitives) (model : TemporalModel) {N C T M : ℕ} (context : Context N C T M) :
    Check (Context N C T M) := do
  let source ← PersonalERPIdentity.readSourceView model
  let expected ← objectAt source.preprocessing "temporal source preprocessing mapping"
  let expected ← readPreprocessingArgs expected
  let target ← readPreprocessingArgs context.preprocessingArgs
  ensure (PersonalAdmission.PyEq (.object target) (.object expected)) "temporal target preprocessing"
  let materialized ← materializeIdentity p context
  let targetIds ← identityKeyViews materialized.1
  ensure (PersonalERPIdentity.sourceKeys source.receipts ∩ PersonalERPIdentity.targetKeys targetIds=∅)
    "temporal target participant"
  let studies ← allIdentityStudies p materialized.1
  ensure (source.studies.toFinset ∩ studies.toFinset=∅) "complete target study"
  pure materialized.2

def validateInput (p : Primitives) (metadata : Fields) (basis : Option TemporalModel)
    {N C T M : ℕ} (context : Context N C T M) : Check (Context N C T M) := do
  checkDataset p context false
  let actual ← inputContract p context
  ensure (PersonalAdmission.PyEq (.object actual) (fieldDefault metadata "input_contract")) "full physical input contract"
  let savedBinding ← objectAt ((metadata.get "preprocessing_execution_binding").getD (.object .nil))
    "saved preprocessing execution binding mapping"
  let expected := fieldDefault savedBinding "execution_identity"
  if expected ≠ .null then
    let args ← readPreprocessingArgs context.preprocessingArgs
    let actualBinding ← executionBinding p.execution args (context.provenance.get "preprocessing_execution")
    let actualIdentity := match actualBinding with
      | none => .null
      | some binding => binding.executionIdentity.getD .null
    ensure (PersonalAdmission.PyEq actualIdentity expected) "compatible actual preprocessing execution evidence"
  let context ← match basis with
    | none => .ok context
    | some basis => temporalTarget p basis context
  let resolved ← participant p context
  ensure (fieldDefault metadata "participant_key"=.text resolved.2.1) "personal model authoritative participant"
  pure resolved.2.2

theorem global_authority_has_source_priority_only_when_no_globals (row : IdentityRecord)
    (h : row.globals=[]) : authorityKey row=
      match row.origins with
      | [key] => .ok key
      | _ => .error (.sourceReject "exactly one source_or_global authority key") := by
  simp [authorityKey,h]

theorem multiple_global_keys_refused (localSubject origin first second status : String) (tail : List String) :
    authorityKey ⟨localSubject,[origin],first :: second :: tail,status⟩=
      .error (.sourceReject "exactly one source_or_global authority key") := rfl

theorem identity_cache_coverage_is_not_cleaned_subject_coverage {N C T M : ℕ}
    (p : Primitives) (context : Context N C T M) (rows : List IdentityRecord)
    (hc : context.identityCache=some rows)
    (h : (rows.map IdentityRecord.localSubject).toFinset ≠ (subjects context).toFinset) :
    materializeIdentity p context=.error (.sourceReject "materialized identity exact original subject coverage") := by
  simp [materializeIdentity,hc,ensure,h]

theorem nonfinite_unobserved_position_still_cannot_encode_input_contract :
    atomJSON .nan=.error (.fieldFailure "finite JSON scalar encoding") := rfl

theorem existing_identity_table_is_used_without_rebuilding {N C T M : ℕ}
    (p : Primitives) (context : Context N C T M) (rows : List IdentityRecord)
    (hc : context.identityCache=some rows)
    (h : (rows.map IdentityRecord.localSubject).toFinset=(subjects context).toFinset) :
    materializeIdentity p context=.ok (rows,context) := by
  simp [materializeIdentity,hc,ensure,h]

theorem single_global_key_overrides_multiple_origins (localSubject first second key status : String)
    (tail : List String) :
    authorityKey ⟨localSubject,first :: second :: tail,[key],status⟩=.ok key := rfl

def withScopeFields (metadata : Fields) (sessions available : Value) : Fields :=
  .cons "calibration_sessions" sessions (.cons "calibration_available_timestamp_s" available metadata)

/-- The literal ordinary input method contains no inference-session call.
This does not certify the altered metadata for the model constructor. -/
theorem ordinary_input_ignores_chronology_fields (p : Primitives) (metadata : Fields)
    (sessions available : Value) (basis : Option TemporalModel)
    {N C T M : ℕ} (context : Context N C T M) :
    validateInput p (withScopeFields metadata sessions available) basis context=
      validateInput p metadata basis context := by
  simp [validateInput,withScopeFields,fieldDefault,Fields.get]

theorem missing_physical_reference_refused_before_geometry_encoding (p : Primitives)
    {N C T M : ℕ} (context : Context N C T M) (h : context.provenance.get "source_reference"=none) :
    inputContract p context=.error (.fieldFailure "explicit physical reference string") := by
  simp [inputContract,fieldDefault,h,text]

theorem blank_physical_reference_refused_before_geometry_encoding (p : Primitives)
    {N C T M : ℕ} (context : Context N C T M) (reference : String)
    (h : context.provenance.get "source_reference"=some (.text reference))
    (hb : p.execution.strip reference="") :
    inputContract p context=.error (.sourceReject "explicit nonblank physical reference") := by
  simp [inputContract,fieldDefault,h,text,ensure,hb]

end

end N2P3.PersonalERPInput
