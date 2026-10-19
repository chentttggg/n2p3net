import N2P3.PersonalERPGuardedViews

/-!
# Public input continuation retaining all independent raw timeline arrays

Source: calibrated.py:465-491,752-766; epochs.py:706-730;
identity.py:286-312; temporal_transfer.py:227-238.
This route requires no common-length Timeline or Timeline-constructor premise.
Unused/mutated timeline vectors remain in raw, including source-accepted
wrong-length status_details. Actual native mask/index operations occur at
the places that consume those arrays. Only the already-guarded positions
receive C-by-3 views for physical input-contract encoding.

Object-array allocation, actual bytes/aliasing, full numeric platform behavior
and inherited exception classes retain the established representation bounds.
The inherited URI quoteURI leaf is pure/total and does not refine Python's
unpaired-surrogate UnicodeEncodeError. Frozen Fields/atomJSON also do not
cover arbitrary NaN-bearing Python JSON objects or their complete rejection
order; the separate RawJSON bridge remains required. Thus the broader ARRAY
domain here is not a claim about every arbitrary Python object/string input.
Identity mutations persist within one returned State. Repeated calls must
explicitly thread withReturnedIdentity below; no persistent heap/alias behavior
of repeated calls on the same immutable raw argument is claimed.
No complete validator, physical-validity, identity-success or default-model
primitive is introduced.
-/

namespace N2P3.PersonalERPRawInputTail
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.ERPTemporalNPZ N2P3.ERPTemporalArrayViews N2P3.PersonalERPArrayIngress

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure State (raw : RawDataset) where
  checked : Validated raw
  identityCache : Option (List PersonalERPInput.IdentityRecord)

def withReturnedIdentity (raw : RawDataset) (state : State raw) : RawDataset :=
  {raw with identityCache:=state.identityCache}

theorem returned_identity_is_explicit_next_input_state (raw : RawDataset) (state : State raw) :
    (withReturnedIdentity raw state).identityCache=state.identityCache ∧
    (withReturnedIdentity raw state).timeline=raw.timeline := ⟨rfl,rfl⟩

def subjectStrings (p : Operations) (raw : RawDataset) : R (List String) := do
  let strings ← p.cast .string raw.subjects
  pure (strings.1.cells.map cellText)

def inputContract (p : Operations) (raw : RawDataset) (checked : Validated raw) : R Fields := do
  let reference ← PersonalERPModelRuntime.text (fieldDefault raw.provenance "source_reference") "explicit physical reference string"
  require (p.input.data.input.original.execution.strip reference ≠ "") "explicit nonblank physical reference"
  let positions := PersonalERPGuardedViews.numericNative raw.positions checked.positionDType
  let coordinates := PersonalERPGuardedViews.matrixOf positions checked.tensor.C 3 checked.positionShape
  let positions ← liftCheck (PersonalERPInput.coordinateJSON coordinates (List.finRange checked.tensor.C))
  let args ← liftCheck (readPreprocessingArgs raw.preprocessingArgs)
  pure (Fields.ofList
    [("channel_names",.sequence (encodeTexts raw.channelNames)),
     ("channel_positions_m",.sequence (valuesOfList positions)),
     ("preprocessing",.object (preprocessingRecord args)),("source_reference",.text reference),
     ("n_channels",.integer checked.tensor.C),("n_times",.integer checked.tensor.T),
     ("simulation_status",(raw.provenance.get "simulation_status").getD (.text "not_declared")),
     ("ocular_artifact_processing",(raw.provenance.get "ocular_artifact_processing").getD
       (.object (Fields.ofList [("status",.text "not_applied")]))),
     ("continuous_artifact_processing",(raw.provenance.get "continuous_artifact_processing").getD
       (.object (Fields.ofList [("status",.text "not_applied")])))])

def scatter (N : ℕ) : List (ℤ × Cell) → (current : List Cell) → R {cells : List Cell // cells.length=current.length}
  | [],current => .ok ⟨current,rfl⟩
  | (index,value) :: tail,current => do
    -- Only nonnegative evidence is selected at the actual calling site.
    if 0 ≤ index ∧ index < (N : ℤ) then
      let rest ← scatter N tail (current.set index.toNat value)
      pure ⟨rest.1,by simpa only [List.length_set] using rest.2⟩
    else
      .error (.inherited (.fieldFailure "source identity native evidence assignment index"))

def sourcePairs (p : Operations) (raw : RawDataset) (N : ℕ) : R (List (String × String)) := do
  -- materialize_dataset_identity repeats actual timeline.validate here.
  let _ ← timelineValidate p raw.timeline N
  let evidence ← p.cast .int64 (raw.timeline.arrays 7)
  let indices := evidence.1.cells.map cellInteger
  let available := indices.map (fun z => decide (0 ≤ z))
  let datasets ← p.cast .string (raw.timeline.arrays 10)
  let selected ← masked datasets.1 available "source identity source rows"
  let kept := (indices.zip available).filterMap (fun pair => if pair.2 then some pair.1 else none)
  let aligned ← scatter N (kept.zip selected) (List.replicate N (.atom .null))
  let subjects ← subjectStrings p raw
  let objectArray : RawArray :=
    { kind := .object, dtype := "object", shape := [N], cells := aligned.1,
      storageGeometry := by simpa only [List.length_replicate,List.prod_cons,List.prod_nil,Nat.mul_one] using aligned.2,
      dtypeContents := by intro cell member; trivial }
  let sourceStrings ← p.cast .string objectArray
  require (subjects.length=sourceStrings.1.cells.length ∧ subjects ≠ []) "source identity aligned nonempty rows"
  let pairs := subjects.zip (sourceStrings.1.cells.map cellText)
  pairs.mapM (fun pair => do
    let localSubject ← liftCheck (PersonalERPInput.cleanString p.input.data.input.original pair.1)
    let source ← liftCheck (PersonalERPInput.cleanString p.input.data.input.original pair.2)
    let origin ← liftCheck (PersonalERPInput.originKey p.input.data.input.original source localSubject)
    pure (localSubject,origin))

def materializeIdentity (p : Operations) (raw : RawDataset) (state : State raw) :
    R (List PersonalERPInput.IdentityRecord × State raw) := do
  match state.identityCache with
  | some rows =>
    let subjects ← subjectStrings p raw
    require ((rows.map PersonalERPInput.IdentityRecord.localSubject).toFinset=subjects.toFinset)
      "materialized identity exact original subject coverage"
    pure (rows,state)
  | none =>
    let pairs ← sourcePairs p raw state.checked.tensor.N
    let rows ← liftCheck (PersonalERPInput.constructSourceRows pairs)
    pure (rows,{state with identityCache:=some rows})

def participant (p : Operations) (raw : RawDataset) (state : State raw) : R (String × String × State raw) := do
  let subjects ← subjectStrings p raw
  let unique := subjects.dedup.mergeSort (· ≤ ·)
  let subject ← match unique with
    | [subject] => pure subject
    | _ => .error (.sourceValueError "exactly one personalized participant")
  let materialized ← materializeIdentity p raw state
  let row ← liftCheck (PersonalERPInput.recordFor p.input.data.input.original materialized.1 subject)
  let authority ← liftCheck (PersonalERPInput.authorityKey row)
  pure (subject,authority,materialized.2)

def temporalTarget (p : Operations) (basis : TemporalModel) (raw : RawDataset) (state : State raw) : R (State raw) := do
  let source ← liftCheck (PersonalERPIdentity.readSourceView basis)
  let expected ← PersonalERPModelRuntime.objectAt source.preprocessing "temporal source preprocessing mapping"
  let expected ← liftCheck (readPreprocessingArgs expected)
  let target ← liftCheck (readPreprocessingArgs raw.preprocessingArgs)
  require (PersonalAdmission.PyEq (.object target) (.object expected)) "temporal target preprocessing"
  let materialized ← materializeIdentity p raw state
  let identities ← liftCheck (PersonalERPInput.identityKeyViews materialized.1)
  require (PersonalERPIdentity.sourceKeys source.receipts ∩ PersonalERPIdentity.targetKeys identities=∅)
    "temporal target participant"
  let studies ← liftCheck (PersonalERPInput.allIdentityStudies p.input.data.input.original materialized.1)
  require (source.studies.toFinset ∩ studies.toFinset=∅) "complete target study"
  pure materialized.2

def validateInput (p : Operations) (metadata : Fields) (basis : Option TemporalModel) (raw : RawDataset) : R (State raw) := do
  let checked ← PersonalERPArrayIngress.validate p raw false
  let actual ← inputContract p raw checked
  require (PersonalAdmission.PyEq (.object actual) (fieldDefault metadata "input_contract")) "full physical model input contract"
  let saved ← PersonalERPModelRuntime.objectAt
    ((metadata.get "preprocessing_execution_binding").getD (.object .nil)) "saved execution binding mapping"
  let expected := fieldDefault saved "execution_identity"
  if expected ≠ .null then
    let args ← liftCheck (readPreprocessingArgs raw.preprocessingArgs)
    let binding ← PersonalERPExecutionRuntime.executionBinding p.input args (raw.provenance.get "preprocessing_execution")
    let identity := match binding with | none => .null | some binding => binding.executionIdentity.getD .null
    require (PersonalAdmission.PyEq identity expected) "actual compatible preprocessing execution identity"
  let state : State raw := ⟨checked,raw.identityCache⟩
  let state ← match basis with
    | none => pure state
    | some basis => temporalTarget p basis raw state
  let resolved ← participant p raw state
  require (fieldDefault metadata "participant_key"=.text resolved.2.1) "actual authoritative model participant"
  pure resolved.2.2

theorem ordinary_raw_input_ignores_chronology (p : Operations) (metadata : Fields)
    (sessions available : Value) (basis : Option TemporalModel) (raw : RawDataset) :
    validateInput p (PersonalERPInput.withScopeFields metadata sessions available) basis raw=
      validateInput p metadata basis raw := by
  simp [validateInput,PersonalERPInput.withScopeFields,fieldDefault,Fields.get]

end
end N2P3.PersonalERPRawInputTail
