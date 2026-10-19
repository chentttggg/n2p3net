import N2P3.PersonalERPQuality
import N2P3.PersonalERPRecipe
import N2P3.PersonalERPAvailability

/-!
# Personal ERP constructor and saved-request composition

Source: calibrated.py:456-458,547-589,620-646,662-713.
The outer constructor expands JSON, decision, quality, execution and physical
ERP gates before the final all-state finite test. Tensor CPU copies, device
parsing and each tensor-to-NumPy conversion remain individual runtime
operations. An unrelated nonfinite named tensor is retained until the final
guard; it is not silently discarded with the ERP prefix projection.
This is P06 work in progress; raw recording/build/post-reference and producer
composition are still separately required for complete admission.
-/

namespace N2P3.PersonalERPAdmission

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalState
open N2P3.PersonalERPData N2P3.PersonalERPExecution N2P3.PersonalERPRaw
open N2P3.PersonalERPModels N2P3.PersonalERPQuality

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure TorchTensor where
  dtype : String
  image : ArrayImage Atom

structure TorchState where
  entries : List (String × TorchTensor)
  unique : (entries.map Prod.fst).Nodup

structure Primitives where
  quality : PersonalERPQuality.Primitives
  temporal : TemporalPrimitives
  reference : ReferencePrimitives
  finiteJSON : Value → Check Value
  detachCPUClone : TorchTensor → Check TorchTensor
  device : String → Check String
  numpy : TorchTensor → Check Tensor
  tensorIsFinite : TorchTensor → Check Bool

def copyTorchEntries (p : Primitives) : List (String × TorchTensor) → Check (List (String × TorchTensor))
  | [] => .ok []
  | (name,value) :: tail => do
    let copied ← p.detachCPUClone value
    let rest ← copyTorchEntries p tail
    pure ((name,copied) :: rest)

def lookupTorch (name : String) : List (String × TorchTensor) → Option TorchTensor
  | [] => none
  | (key,value) :: tail => if name=key then some value else lookupTorch name tail

def requiredTorch (state : List (String × TorchTensor)) (name : String) : Check TorchTensor :=
  match lookupTorch name state with
  | none => .error (.fieldFailure ("missing personal tensor:" ++ name))
  | some tensor => .ok tensor

def selectedIndices (channels : ℕ) (tensor : TorchTensor) : Check (List ℕ) := do
  ensure (tensor.dtype="torch.int64") "ERP selected dtype"
  ensure (tensor.image.shape=[tensor.image.values.length] ∧ tensor.image.values ≠ []) "ERP selected rank/nonempty"
  let values ← tensor.image.values.mapM fun atom => do
    let value ← strictInteger atom
    ensure (0 ≤ value) "ERP selected nonnegative integer"
    pure value.toNat
  ensure (values.Nodup ∧ ∀ index ∈ values, index < channels) "ERP selected unique/range"
  pure values

def erpNumpyState (p : Primitives) : List (String × TorchTensor) → Check NamedState
  | [] => .ok []
  | (name,tensor) :: tail => do
    match ERPState.stripERP name with
    | none => erpNumpyState p tail
    | some inner => do
      let value ← p.numpy tensor
      let rest ← erpNumpyState p tail
      pure ((inner,value) :: rest)

def checkDecision (metadata : Fields) : Check Unit := do
  let value := fieldDefault metadata "decision_score"
  if value ≠ .null then
    let fields ← objectAt value "decision score mapping"
    ensure (fieldDefault fields "method"=.text "membership_llr_sum/1") "decision score method"
    ensure (fieldDefault fields "prior"=.text "uniform") "decision score prior"
    ensure (fieldDefault fields "coverage"=.text "complete_physical_schedule") "decision physical schedule coverage"
    ensure (fieldDefault fields "probability_calibrated"=.boolean false) "decision strict uncalibrated probability flag"
    let temperature ← numeric (fieldDefault fields "temperature") "decision numeric finite temperature"
    ensure (0 < temperature) "decision positive temperature"

def qualityProvenance : EpochQuality → Check Fields
  | .personal model => objectAt model.provenance "personal epoch quality fit provenance mapping"
  | .fold model => .ok model.provenance

def readQuality (p : Primitives) (metadata : Fields) : Check (Option EpochQuality) := do
  let value := fieldDefault metadata "epoch_quality"
  if value=.null then pure none
  else
    let fields ← objectAt value "calibrated epoch quality envelope mapping"
    let modelValue := fieldDefault fields "model"
    let digest ← p.quality.models.recovery.input.execution.semanticHash modelValue
    ensure (fieldDefault fields "model_digest"=.text digest) "calibrated epoch quality digest"
    let requiredModel ← required fields "model"
    let model ← epochQualityFromRecord p.quality requiredModel
    pure (some model)

def checkPersonalRawQuality (p : Primitives) (processingValue : Value) (quality : EpochQuality) : Check Unit := do
  match quality with
  | .fold _ => pure ()
  | .personal _ =>
    let processing ← objectAt processingValue "personal epoch QC continuous processing dictionary"
    ensure (fieldDefault processing "status"=.text "applied") "personal epoch QC requires applied raw QC"
    let modelValue := fieldDefault processing "model"
    let model ← objectAt modelValue "personal raw QC model dictionary"
    ensure (fieldDefault model "schema"=.text "n2p3_continuous_qc/4") "personal epoch QC raw constant schema four"
    let spec ← objectAt (fieldDefault model "spec") "personal raw QC spec dictionary"
    ensure (fieldDefault spec "channel_local"=.boolean true) "personal raw QC strict channel local"
    ensure (fieldDefault spec "flatline_min_duration_s" ≠ .null) "personal raw QC explicit flatline duration"
    let digest ← p.quality.models.recovery.input.execution.semanticHash modelValue
    ensure (fieldDefault processing "model_digest"=.text digest) "personal raw QC model digest"
    let _ ← continuousQCFromRecord p.quality.models modelValue
    pure ()

def checkQualityFitFields (p : Primitives) (provenance expected : Fields) : List String → Check Unit
  | [] => .ok ()
  | key :: rest => do
    let actual ← p.finiteJSON (.object (Fields.ofList [("value",fieldDefault provenance key)]))
    let saved ← p.finiteJSON (.object (Fields.ofList [("value",fieldDefault expected key)]))
    ensure (PersonalAdmission.PyEq actual saved) "epoch quality exact normalized fit field"
    checkQualityFitFields p provenance expected rest

def checkQualityBudget (metadata contract : Fields) (quality : EpochQuality) (provenance : Fields) : Check Unit := do
  ensure (PersonalAdmission.PyEq (.integer (epochQualityChannels quality)) (fieldDefault contract "n_channels"))
    "calibrated epoch quality channel geometry"
  let calibration ← required metadata "calibration_sessions"
  let calibration ← listAt calibration "calibration session rows"
  let ids ← readCalibrationIds calibration
  let fitIds ← readHashableTuple ((provenance.get "session_ids").getD (.sequence .nil))
  ensure (∀ id ∈ fitIds, ∃ saved ∈ ids, PersonalAdmission.PyEq id saved)
    "epoch quality calibration session subset"

structure Predictor where
  metadata : Fields
  state : List (String × TorchTensor)
  device : String
  quality : Option EpochQuality
  bound : PersonalERPRecipe.Bound

def allStateFinite (p : Primitives) : List (String × TorchTensor) → Check Unit
  | [] => .ok ()
  | (_,value) :: tail => do
    let finite ← p.tensorIsFinite value
    ensure (finite=true) "personal predictor all state finite"
    allStateFinite p tail

def constructERP (p : Primitives) (metadataValue : Value) (state : TorchState) (deviceName : String) : Check Predictor := do
  let metadataValue ← p.finiteJSON metadataValue
  let metadata ← objectAt metadataValue "calibrated predictor JSON metadata mapping"
  let state ← copyTorchEntries p state.entries
  let device ← p.device deviceName
  checkDecision metadata
  let quality ← readQuality p metadata
  let contractValue ← required metadata "input_contract"
  let contract ← objectAt contractValue "calibrated input contract dictionary"
  ensure (PyTruthy (fieldDefault metadata "participant_key")) "calibrated participant key truthiness"
  let preprocessingValue ← required contract "preprocessing"
  let preprocessingFields ← objectAt preprocessingValue "calibrated preprocessing kwargs mapping"
  let args ← readPreprocessingArgs preprocessingFields
  let binding ← executionBinding p.quality.models.recovery.input.execution args (metadata.get "preprocessing_execution")
  ensure (PersonalAdmission.PyEq (.object (bindingFields binding))
    ((metadata.get "preprocessing_execution_binding").getD (.object .nil))) "saved full preprocessing execution binding"
  match quality with
  | none => pure ()
  | some fitted => do
    checkPersonalRawQuality p ((contract.get "continuous_artifact_processing").getD (.object .nil)) fitted
    let reference ← required contract "source_reference"
    let channels ← required contract "channel_names"
    let preprocessing ← required contract "preprocessing"
    let expected := Fields.ofList [("participant_key",fieldDefault metadata "participant_key"),
      ("source_reference",reference),("channel_names",channels),("preprocessing",preprocessing)]
    let provenance ← qualityProvenance fitted
    checkQualityFitFields p provenance expected ["participant_key","source_reference","channel_names","preprocessing"]
    checkQualityBudget metadata contract fitted provenance
  let kindValue ← required metadata "model_kind"
  let kind ← text kindValue "personal model kind string"
  ensure (kind ∈ ["target_only_structured_erp","source_temporal_target_erp"]) "personal ERP constructor branch"
  let channelValue ← required contract "n_channels"
  let channelCount ← numeric channelValue "calibrated comparable channel count"
  let selectedTensor ← requiredTorch state "waveform_channels"
  -- The source compares indices to the saved numeric channel count; it does
  -- not require exact integer n_channels before this branch.
  ensure (selectedTensor.dtype="torch.int64") "ERP selected dtype"
  ensure (selectedTensor.image.shape=[selectedTensor.image.values.length] ∧ selectedTensor.image.values ≠ [])
    "ERP selected rank/nonempty"
  let selected ← selectedTensor.image.values.mapM fun atom => do
    let index ← strictInteger atom
    ensure (0 ≤ index ∧ (index : ℝ) < channelCount) "ERP selected signed index range"
    pure index.toNat
  ensure (selected.Nodup) "ERP selected unique indices"
  let recordValue ← required metadata "erp_artifact"
  let record ← objectAt recordValue "personal ERP artifact mapping"
  let erpState ← erpNumpyState p state
  let model ← CurrentERPState.fromState p.temporal p.reference record erpState
  let recipeValue ← required metadata "recipe"
  let recipeFields ← objectAt recipeValue "personal ERP recipe kwargs"
  let recipe ← PersonalERPRecipe.fromKwargs recipeFields
  let timeValue ← required contract "n_times"
  ensure (PersonalAdmission.PyEq (.integer model.channels) (.integer selected.length) ∧
    PersonalAdmission.PyEq (.integer model.times) timeValue) "ERP input shape physical correspondence"
  let rawBaseline ← required preprocessingFields "baseline_mode"
  ensure (PersonalAdmission.PyEq model.header.baseline rawBaseline ∧
    recipe.family=model.header.family ∧ recipe.width=model.header.binWidth) "ERP physical recipe state correspondence"
  let sourceFlag := fieldDefault metadata "source_pretraining_used"
  let source := PersonalERPRecipe.usesSource model
  ensure (sourceFlag=.boolean source ∧ kind=PersonalERPRecipe.modelKind source) "ERP strict source role"
  match model.header.temporal with
  | none => pure ()
  | some basis => ensure (recipe.temporalKind=some (if source then "source_supervised" else "fixed_cosine") ∧
      recipe.temporalRank=some basis.rank) "ERP temporal recipe basis kind rank"
  allStateFinite p state
  pure ⟨metadata,state,device,quality,⟨recipe,selected,model⟩⟩

def predictorBasis (predictor : Predictor) : Option TemporalModel := predictor.bound.model.header.temporal

def validatePredictorInput (p : Primitives) (predictor : Predictor) {N C T M : ℕ}
    (context : PersonalERPInput.Context N C T M) : Check (PersonalERPInput.Context N C T M) :=
  PersonalERPInput.validateInput p.quality.models.recovery.input predictor.metadata (predictorBasis predictor) context

structure ResolvedRequest where
  fields : PersonalERPRaw.SavedRequest
  ocular : Option EOGModel
  continuous : Option ContinuousQC
  reference : Option String
  postReference : Bool

def readSavedRequest (p : Primitives) (metadata : Fields) : Check ResolvedRequest := do
  let request ← checkSavedRequest metadata
  let _ ← checkPreprocessing p.quality.models.recovery.input.execution.strip
    p.quality.models.recovery.input.execution.floorSamples p.quality.models.recovery.input.execution.windowSamples
    (rawPreprocessing request.preprocessingArgs)
  let ocular ← if request.ocularRecord=.null then .ok none else do
    let model ← readEOGModel p.quality.models.recovery.input request.ocularRecord
    pure (some model)
  let continuous ← if request.continuousRecord=.null then .ok none else do
    let model ← continuousQCFromRecord p.quality.models request.continuousRecord
    pure (some model)
  let ocularView := ocular.map eogReferenceView
  let continuousView := continuous.map (fun model => (⟨model.spec.reference,model.spec.channelLocal⟩ : ContinuousReferenceView))
  let reference ← resolveReference p.quality.models.recovery.input (fieldDefault request.original "reference_channel")
    ((request.original.get "reference_mode").getD (.text "auto")) ocularView continuousView
  checkSavedReference request.contract reference
  pure ⟨request,ocular,continuous,reference,needsPostReference reference ocularView continuousView⟩

theorem strict_probability_flag_rejects_integer_zero :
    ensure (Value.integer 0=.boolean false) "decision strict uncalibrated probability flag"=
      .error (.sourceReject "decision strict uncalibrated probability flag") := by simp [ensure]

theorem decision_temperature_retains_bool_numeric_alias : numeric (.boolean true) "decision numeric finite temperature"=.ok 1 := rfl

theorem no_reference_post_stage_is_false (request : PersonalERPRaw.SavedRequest) (ocular : Option EOGModel)
    (continuous : Option ContinuousQC) :
    (ResolvedRequest.mk request ocular continuous none false).postReference=false := rfl

end

end N2P3.PersonalERPAdmission
