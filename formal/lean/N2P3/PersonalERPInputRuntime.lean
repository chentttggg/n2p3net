import N2P3.PersonalERPExecutionRuntime

/-!
# Actual preprocessing/receipt atoms on the mutable personal input route

Source: epochs.py:325-471,706-730; calibrated.py:752-766;
temporal_transfer.py:227-238. Physical, exclusion, data, execution, identity
and lineage checks remain explicit definitions. The context carries the real
identity cache across temporal validation and participant materialization.
No dataset/input-validation success primitive or chronology premise is used.
Lower array/URI/JSON conversions remain inherited representation boundaries.
-/

namespace N2P3.PersonalERPInputRuntime
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def checkDataset (p : PersonalERPExecutionRuntime.Primitives) {N C T M : ℕ}
    (context : PersonalERPInput.Context N C T M) (requireLabels : Bool) : R Unit := do
  let input := p.data.input.original
  liftCheck (checkPhysicalCoordinates context.data.historical context.data.coordinateSystem (context.provenance.get "data_alignment"))
  match context.identityCache with
  | none => pure ()
  | some rows => liftCheck (PersonalERPInput.checkIdentityExclusions input rows)
  liftCheck (PersonalERPInput.checkScheduledExclusions input context.data.timeline)
  let args ← liftCheck (readPreprocessingArgs context.preprocessingArgs)
  let _ ← PersonalERPDataRuntime.checkDatasetBody p.data args context.data requireLabels
  let _ ← PersonalERPExecutionRuntime.executionBinding p args (context.provenance.get "preprocessing_execution")
  liftCheck (checkDatasetTail (PersonalERPInput.subjects context)
    (context.identityCache.map (fun rows => rows.map PersonalERPInput.IdentityRecord.localSubject)) context.lineage context.attestation)

def validateAfterDataset (p : PersonalERPExecutionRuntime.Primitives) (metadata : Fields) (basis : Option TemporalModel)
    {N C T M : ℕ} (context : PersonalERPInput.Context N C T M) : R (PersonalERPInput.Context N C T M) := do
  let actual ← liftCheck (PersonalERPInput.inputContract p.data.input.original context)
  require (PersonalAdmission.PyEq (.object actual) (fieldDefault metadata "input_contract")) "full physical model input contract"
  let saved ← PersonalERPModelRuntime.objectAt
    ((metadata.get "preprocessing_execution_binding").getD (.object .nil)) "saved execution binding mapping"
  let expected := fieldDefault saved "execution_identity"
  if expected ≠ .null then
    let args ← liftCheck (readPreprocessingArgs context.preprocessingArgs)
    let binding ← PersonalERPExecutionRuntime.executionBinding p args (context.provenance.get "preprocessing_execution")
    let identity := match binding with | none => .null | some binding => binding.executionIdentity.getD .null
    require (PersonalAdmission.PyEq identity expected) "actual compatible preprocessing execution identity"
  let context ← match basis with
    | none => pure context
    | some basis => liftCheck (PersonalERPInput.temporalTarget p.data.input.original basis context)
  let resolved ← liftCheck (PersonalERPInput.participant p.data.input.original context)
  require (fieldDefault metadata "participant_key"=.text resolved.2.1) "actual authoritative model participant"
  pure resolved.2.2

def validateInput (p : PersonalERPExecutionRuntime.Primitives) (metadata : Fields) (basis : Option TemporalModel)
    {N C T M : ℕ} (context : PersonalERPInput.Context N C T M) : R (PersonalERPInput.Context N C T M) := do
  checkDataset p context false
  validateAfterDataset p metadata basis context

theorem ordinary_input_still_ignores_chronology (p : PersonalERPExecutionRuntime.Primitives) (metadata : Fields)
    (sessions available : Value) (basis : Option TemporalModel) {N C T M : ℕ}
    (context : PersonalERPInput.Context N C T M) :
    validateInput p (PersonalERPInput.withScopeFields metadata sessions available) basis context=
      validateInput p metadata basis context := by
  simp [validateInput,validateAfterDataset,PersonalERPInput.withScopeFields,fieldDefault,Fields.get]

end
end N2P3.PersonalERPInputRuntime
