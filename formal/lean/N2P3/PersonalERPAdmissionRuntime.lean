import N2P3.PersonalERPInputRuntime
import N2P3.PersonalERPQualityRuntime
import N2P3.PersonalERPAdmission
import N2P3.PersonalERPActualConsumer

/-!
# Actual personal ERP constructor and saved raw-request route

Source: calibrated.py:547-589,620-646,662-713. This composes runtime QC,
preprocessing/receipt and actual recipe scalar checks at their original slots.
The selected Torch tensor is checked for dtype/rank, uniqueness and negativity
before contract n_channels is accessed for the final Torch bound comparison.
The latter retains the actual library operand tags and may fail independently.
All unrelated named state tensors reach the final finite test. No entire
constructor-success or physical-input-validity premise is introduced.
-/

namespace N2P3.PersonalERPAdmissionRuntime
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalState
open N2P3.PersonalERPData N2P3.PersonalERPRaw N2P3.PersonalERPModels N2P3.PersonalERPQuality
open N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.PersonalERPAdmission

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  original : PersonalERPAdmission.Primitives
  finite : PersonalERPModelRuntime.Finite
  arrays : PersonalERPModelRuntime.Arrays
  recipeScalar : PersonalERPScalar.Primitives
  mathFinite : Value → Except AtomicException Bool
  numpyTupleFiniteAll : List Value → Except AtomicException Bool
  numpyFloatFiniteAll : List Atom → Except AtomicException Bool
  floatScalar : Value → R Atom
  divide : Value → Value → R ℝ
  floorSamples : Value → Value → Value → R ℤ
  windowSamples : ℝ → ℝ → Value → R Bool
  roundOffset : Value → Value → R ℤ
  closeRates : Value → Value → R Bool
  roundResampled : ℤ → Value → Value → R ℤ
  selectedUniqueCount : TorchTensor → R ℕ
  selectedAnyNegative : TorchTensor → R Bool
  selectedAnyTooHigh : TorchTensor → Value → R Bool

def Primitives.quality (p : Primitives) : PersonalERPQualityRuntime.Primitives :=
  ⟨p.original.quality,p.finite,p.arrays⟩

def Primitives.data (p : Primitives) : PersonalERPDataRuntime.Primitives :=
  ⟨⟨p.original.quality.models.recovery.input,p.finite⟩,p.mathFinite,p.numpyTupleFiniteAll,
    p.numpyFloatFiniteAll,p.floatScalar,p.divide,p.floorSamples,p.windowSamples⟩

def Primitives.execution (p : Primitives) : PersonalERPExecutionRuntime.Primitives :=
  ⟨p.data,p.arrays,p.roundOffset,p.closeRates,p.roundResampled⟩

def checkDecision (finite : PersonalERPModelRuntime.Finite) (metadata : Fields) : R Unit := do
  let value := fieldDefault metadata "decision_score"
  if value ≠ .null then
    let fields ← PersonalERPModelRuntime.objectAt value "decision score.get mapping"
    require (fieldDefault fields "method"=.text "membership_llr_sum/1") "decision method"
    require (fieldDefault fields "prior"=.text "uniform") "decision uniform prior"
    require (fieldDefault fields "coverage"=.text "complete_physical_schedule") "decision complete physical coverage"
    require (fieldDefault fields "probability_calibrated"=.boolean false) "decision strict probability False"
    let temperature := fieldDefault fields "temperature"
    require (builtinNumeric temperature=true) "decision temperature isinstance int float"
    let temperature ← numpyNumeric finite temperature "decision temperature NumPy finite"
    require (0 < temperature) "decision positive temperature"

def readQuality (p : Primitives) (metadata : Fields) : R (Option EpochQuality) := do
  let value := fieldDefault metadata "epoch_quality"
  if value=.null then pure none
  else
    let fields ← PersonalERPModelRuntime.objectAt value "epoch quality envelope.get mapping"
    let hash ← p.quality.models.recovery.input.execution.semanticHash (fieldDefault fields "model")
    require (fieldDefault fields "model_digest"=.text hash) "epoch quality actual model digest"
    let modelValue ← PersonalERPModelRuntime.required fields "model"
    let model ← PersonalERPQualityRuntime.epochQualityFromRecord p.quality modelValue
    pure (some model)

def checkPersonalRawQuality (p : Primitives) (value : Value) (quality : EpochQuality) : R Unit := do
  match quality with
  | .fold _ => pure ()
  | .personal _ =>
    let processing ← PersonalERPModelRuntime.objectAt value "personal raw QC processing mapping"
    require (fieldDefault processing "status"=.text "applied") "personal quality applied continuous QC"
    let modelValue := fieldDefault processing "model"
    let model ← PersonalERPModelRuntime.objectAt modelValue "personal raw QC model mapping"
    require (fieldDefault model "schema"=.text "n2p3_continuous_qc/4") "personal quality raw QC schema four"
    let spec ← PersonalERPModelRuntime.objectAt (fieldDefault model "spec") "personal raw QC spec mapping"
    require (fieldDefault spec "channel_local"=.boolean true) "personal quality raw QC strict local flag"
    require (fieldDefault spec "flatline_min_duration_s" ≠ .null) "personal quality raw exact constant duration"
    let hash ← p.quality.models.recovery.input.execution.semanticHash modelValue
    require (fieldDefault processing "model_digest"=.text hash) "personal raw QC model actual digest"
    -- Source catches KeyError/TypeError/ValueError from this profile parser.
    -- Refined scalar and enum errors are represented as source ValueError;
    -- inherited conversion faults keep their explicitly unresolved class.
    let result := PersonalERPModelRuntime.continuousQCFromRecord p.quality.models modelValue
    match result with
    | .ok _ => pure ()
    | .error (.atomic _ _ .typeError) => .error (.sourceValueError "invalid personal raw QC caught TypeError")
    | .error (.atomic _ _ (.other "KeyError")) => .error (.sourceValueError "invalid personal raw QC caught KeyError")
    | .error (.atomic _ _ (.other "ValueError")) => .error (.sourceValueError "invalid personal raw QC caught ValueError")
    | .error (.sourceValueError _) => .error (.sourceValueError "invalid personal raw QC caught ValueError")
    | .error fault => .error fault

def qualityFitFields (p : Primitives) (provenance expected : Fields) : List String → R Unit
  | [] => .ok ()
  | key :: tail => do
    let actual ← liftCheck (p.original.finiteJSON (.object (Fields.ofList [("value",fieldDefault provenance key)])))
    let saved ← liftCheck (p.original.finiteJSON (.object (Fields.ofList [("value",fieldDefault expected key)])))
    require (PersonalAdmission.PyEq actual saved) "quality normalized fit input field"
    qualityFitFields p provenance expected tail

def qualityBudget (metadata contract : Fields) (quality : EpochQuality) (provenance : Fields) : R Unit := do
  require (PersonalAdmission.PyEq (.integer (epochQualityChannels quality)) (fieldDefault contract "n_channels")) "quality exact channel count"
  let sessionsValue ← PersonalERPModelRuntime.required metadata "calibration_sessions"
  let sessions ← liftCheck (ERPState.listAt sessionsValue "calibration session rows")
  let ids ← liftCheck (readCalibrationIds sessions)
  let fit ← PersonalERPModelRuntime.readHashableTuple ((provenance.get "session_ids").getD (.sequence .nil))
  require (∀ id ∈ fit, ∃ saved ∈ ids, PersonalAdmission.PyEq id saved) "quality calibration session subset"

def selected (p : Primitives) (state : List (String × TorchTensor)) (contract : Fields) : R (List ℕ) := do
  let tensor ← liftCheck (requiredTorch state "waveform_channels")
  require (tensor.dtype="torch.int64") "ERP selected dtype"
  require (tensor.image.shape=[tensor.image.values.length] ∧ tensor.image.values ≠ []) "ERP selected rank nonempty"
  let unique ← p.selectedUniqueCount tensor
  require (unique=tensor.image.values.length) "ERP selected actual Torch uniqueness"
  let negative ← p.selectedAnyNegative tensor
  require (negative=false) "ERP selected actual Torch nonnegative"
  let channels ← PersonalERPModelRuntime.required contract "n_channels"
  let tooHigh ← p.selectedAnyTooHigh tensor channels
  require (tooHigh=false) "ERP selected actual Torch upper bound"
  tensor.image.values.mapM (fun value => do
    let index ← liftCheck (PersonalERPData.strictInteger value)
    pure index.toNat)

def finishERP (p : Primitives) (metadata : Fields) (state : List (String × TorchTensor))
    (device : String) (quality : Option EpochQuality) (contract rawArgs : Fields) : R Predictor := do
  let kindValue ← PersonalERPModelRuntime.required metadata "model_kind"
  let kind ← PersonalERPModelRuntime.text kindValue "personal ERP kind string"
  require (kind ∈ ["target_only_structured_erp","source_temporal_target_erp"]) "personal actual ERP branch"
  let indices ← selected p state contract
  let recordValue ← PersonalERPModelRuntime.required metadata "erp_artifact"
  let record ← PersonalERPModelRuntime.objectAt recordValue "actual ERP artifact mapping"
  let numpyState ← liftCheck (erpNumpyState p.original state)
  let model ← liftCheck (CurrentERPState.fromState p.original.temporal p.original.reference record numpyState)
  let kwargsValue ← PersonalERPModelRuntime.required metadata "recipe"
  let kwargs ← PersonalERPModelRuntime.objectAt kwargsValue "actual recipe kwargs mapping"
  let recipe ← PersonalERPScalar.fromKwargsScalar p.recipeScalar kwargs
  let times ← PersonalERPModelRuntime.required contract "n_times"
  require (model.channels=indices.length ∧ PersonalAdmission.PyEq (.integer model.times) times) "actual ERP input shape physical contract"
  let baseline ← PersonalERPModelRuntime.required rawArgs "baseline_mode"
  require (PersonalAdmission.PyEq model.header.baseline baseline ∧ recipe.family=model.header.family ∧ recipe.width=model.header.binWidth)
    "actual ERP baseline recipe correspondence"
  let source := PersonalERPRecipe.usesSource model
  require (fieldDefault metadata "source_pretraining_used"=.boolean source ∧ kind=PersonalERPRecipe.modelKind source)
    "actual ERP strict source role"
  match model.header.temporal with
  | none => pure ()
  | some basis => require (recipe.temporalKind=some (if source then "source_supervised" else "fixed_cosine") ∧
      recipe.temporalRank=some basis.rank) "actual ERP temporal recipe basis kind rank"
  liftCheck (allStateFinite p.original state)
  pure ⟨metadata,state,device,quality,⟨recipe,indices,model⟩⟩

def constructERP (p : Primitives) (metadataValue : Value) (state : TorchState) (deviceName : String) : R Predictor := do
  let metadataValue ← liftCheck (p.original.finiteJSON metadataValue)
  let metadata ← PersonalERPModelRuntime.objectAt metadataValue "personal JSON metadata mapping"
  let state ← liftCheck (copyTorchEntries p.original state.entries)
  let device ← liftCheck (p.original.device deviceName)
  checkDecision p.finite metadata
  let quality ← readQuality p metadata
  let contractValue ← PersonalERPModelRuntime.required metadata "input_contract"
  let contract ← PersonalERPModelRuntime.objectAt contractValue "personal input contract dictionary"
  require (PyTruthy (fieldDefault metadata "participant_key")) "personal participant truthiness"
  let rawArgs ← PersonalERPModelRuntime.required contract "preprocessing"
  let rawArgs ← PersonalERPModelRuntime.objectAt rawArgs "personal preprocessing kwargs mapping"
  let args ← liftCheck (readPreprocessingArgs rawArgs)
  let binding ← PersonalERPExecutionRuntime.executionBinding p.execution args (metadata.get "preprocessing_execution")
  require (PersonalAdmission.PyEq (.object (PersonalERPExecution.bindingFields binding))
    ((metadata.get "preprocessing_execution_binding").getD (.object .nil))) "personal saved entire execution binding"
  match quality with
  | none => pure ()
  | some quality => do
    checkPersonalRawQuality p ((contract.get "continuous_artifact_processing").getD (.object .nil)) quality
    let reference ← PersonalERPModelRuntime.required contract "source_reference"
    let channels ← PersonalERPModelRuntime.required contract "channel_names"
    let preprocessing ← PersonalERPModelRuntime.required contract "preprocessing"
    let expected := Fields.ofList [("participant_key",fieldDefault metadata "participant_key"),
      ("source_reference",reference),("channel_names",channels),("preprocessing",preprocessing)]
    let provenance ← liftCheck (PersonalERPAdmission.qualityProvenance quality)
    qualityFitFields p provenance expected ["participant_key","source_reference","channel_names","preprocessing"]
    qualityBudget metadata contract quality provenance
  finishERP p metadata state device quality contract rawArgs

def validatePredictorInput (p : Primitives) (model : Predictor) {N C T M : ℕ}
    (context : PersonalERPInput.Context N C T M) : R (PersonalERPInput.Context N C T M) :=
  PersonalERPInputRuntime.validateInput p.execution model.metadata (predictorBasis model) context

def readSavedRequest (p : Primitives) (metadata : Fields) : R ResolvedRequest := do
  let request ← liftCheck (checkSavedRequest metadata)
  let _ ← PersonalERPDataRuntime.checkPreprocessing p.data request.preprocessingArgs
  let ocular ← if request.ocularRecord=.null then pure none else
    (PersonalERPModelRuntime.readEOGModel p.quality.models.recovery.input request.ocularRecord).map some
  let continuous ← if request.continuousRecord=.null then pure none else
    (PersonalERPModelRuntime.continuousQCFromRecord p.quality.models request.continuousRecord).map some
  let ocularView := ocular.map eogReferenceView
  let continuousView := continuous.map (fun model => (⟨model.spec.reference,model.spec.channelLocal⟩ : ContinuousReferenceView))
  let mode := (request.original.get "reference_mode").getD (.text "auto")
  setOption "raw saved reference mode" [.text "auto",.text "native"] mode
  let reference ← liftCheck (resolveReference p.original.quality.models.recovery.input
    (fieldDefault request.original "reference_channel") mode ocularView continuousView)
  liftCheck (checkSavedReference request.contract reference)
  pure ⟨request,ocular,continuous,reference,needsPostReference reference ocularView continuousView⟩

def constructAndValidate (p : Primitives) (metadata : Value) (state : TorchState) (device : String)
    {N C T M : ℕ} (context : PersonalERPInput.Context N C T M) : R (Predictor × PersonalERPInput.Context N C T M) := do
  let model ← constructERP p metadata state device
  let context ← validatePredictorInput p model context
  pure (model,context)

def noQualityMetadata (authority : String) (contract recipe artifact : Fields) : Fields := Fields.ofList
  [("participant_key",.text authority),("input_contract",.object contract),("recipe",.object recipe),
   ("erp_artifact",.object artifact),("model_kind",.text "target_only_structured_erp"),
   ("source_pretraining_used",.boolean false)]

theorem copied_entries_from_individual_clone_operations (p : PersonalERPAdmission.Primitives)
    (state : List (String × TorchTensor))
    (h : ∀ entry ∈ state, p.detachCPUClone entry.2=.ok entry.2) :
    copyTorchEntries p state=.ok state := by
  induction state with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name,tensor⟩
    have hc := h (name,tensor) (by simp)
    have ht : ∀ entry ∈ tail, p.detachCPUClone entry.2=.ok entry.2 := by
      intro entry he
      exact h entry (by simp [he])
    simp [copyTorchEntries,hc,ih ht]

theorem prefix_from_independent_fields_reaches_actual_ERP_continuation (p : Primitives)
    (authority : String) (contract recipe artifact rawArgs : Fields) (state : TorchState) (requested actualDevice : String)
    (hAuthority : authority ≠ "") (hArgs : contract.get "preprocessing"=some (.object rawArgs))
    (hKnown : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (hJSON : p.original.finiteJSON (.object (noQualityMetadata authority contract recipe artifact))=
      .ok (.object (noQualityMetadata authority contract recipe artifact)))
    (hCopies : ∀ entry ∈ state.entries, p.original.detachCPUClone entry.2=.ok entry.2)
    (hDevice : p.original.device requested=.ok actualDevice) :
    constructERP p (.object (noQualityMetadata authority contract recipe artifact)) state requested=
      finishERP p (noQualityMetadata authority contract recipe artifact) state.entries actualDevice none contract rawArgs := by
  have hCopied := copied_entries_from_individual_clone_operations p.original state.entries hCopies
  unfold constructERP
  rw [hJSON,hCopied,hDevice]
  simp [noQualityMetadata,checkDecision,readQuality,
    PersonalERPModelRuntime.objectAt,PersonalERPModelRuntime.required,liftCheck,StateLedger.required,
    fieldDefault,Fields.ofList,Fields.get,ERPState.objectAt,PyTruthy,hAuthority,require,
    readPreprocessingArgs,ensure,hKnown,PersonalERPExecutionRuntime.executionBinding,
    PersonalERPExecution.bindingFields,PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,hArgs]

theorem wrong_selector_dtype_precedes_missing_channel_bound (p : Primitives) (state : List (String × TorchTensor))
    (tensor : TorchTensor) (contract : Fields)
    (ht : lookupTorch "waveform_channels" state=some tensor) (hd : tensor.dtype ≠ "torch.int64") :
    selected p state contract=.error (.sourceValueError "ERP selected dtype") := by
  simp [selected,requiredTorch,ht,liftCheck,require,hd]

end
end N2P3.PersonalERPAdmissionRuntime
