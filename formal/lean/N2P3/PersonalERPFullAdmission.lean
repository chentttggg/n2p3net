import N2P3.PersonalERPAdmissionRuntime
import N2P3.PersonalFullRecipeOutcome

/-!
# ERP admission with raw NumPy transport and the full recipe constructor

Source: calibrated.py:547-642,660-661; structured_erp.py:171-230.
Contract: full_admission_connection_contract_03.json, fixed before this file.

JSON -> all clones -> device precedes the first metadata.get. The raw artifact
value is fetched before all Tensor.numpy arguments, while artifact.get and
float64 casts occur only inside the subsequent reader. No finiteness/type
normalization is inserted into Tensor.numpy. Each array's float64 conversion
and copy precedes the next array; every copy precedes the finite guard.

This is the ERP branch, not the neural/global branch dispatcher. Earlier
QC/execution leaves and the inherited finite-reader xDAWN reference predicates
retain their stated codec/numeric/exception boundaries. Array images describe
decoded observations, not native bytes/aliasing or arbitrary Python objects.
-/

namespace N2P3.PersonalERPFullAdmission
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPAdmission

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure NativeArray where
  dtype : String
  image : ArrayImage Atom

abbrev NativeArrays := List (String × NativeArray)

inductive Failure where
  | runtime (fault : PersonalERPScalar.Fault)
  | recipe (fault : PersonalRecipePython.Fault)
  | native (operation field : String) (exception : PersonalRecipePython.Exception)
  deriving DecidableEq

abbrev AdmissionResult (α : Type) := Except Failure α

def liftRuntime (value : PersonalERPScalar.Result α) : AdmissionResult α :=
  value.mapError Failure.runtime

def liftCheck (value : Check α) : AdmissionResult α :=
  value.mapError (fun fault => .runtime (.inherited fault))

def liftRecipe (value : PersonalRecipePython.Result α) : AdmissionResult α :=
  value.mapError Failure.recipe

def liftNative (operation field : String) (value : Except PersonalRecipePython.Exception α) : AdmissionResult α :=
  value.mapError (Failure.native operation field)

def require (condition : Prop) (site : String) : AdmissionResult Unit :=
  liftRuntime (PersonalERPScalar.require condition site)

structure Operations where
  admission : PersonalERPAdmissionRuntime.Primitives
  recipe : PersonalRecipePython.Primitives
  numpy : TorchTensor → Except PersonalRecipePython.Exception NativeArray
  asFloat64 : NativeArray → Except PersonalRecipePython.Exception PersonalERPFitOutcome.Array64
  copyFloat64 : PersonalERPFitOutcome.Array64 → Except PersonalRecipePython.Exception PersonalERPFitOutcome.Array64
  temporalMatrix : TemporalModel → Except PersonalRecipePython.Exception NativeArray

/-- Builtin JSON nonobjects lack .get; this check is performed at its actual
method access, never at the annotation or JSON return point. -/
def getMapping (value : Value) (site : String) : AdmissionResult Fields :=
  match value with
  | .object fields => .ok fields
  | _ => .error (.native "get" site .attributeError)

def contractDictionary (value : Value) : AdmissionResult Fields :=
  match value with
  | .object fields => .ok fields
  | _ => .error (.runtime (.sourceValueError "calibrated model lacks participant and input contracts"))

def kwargsMapping (value : Value) (site : String) : AdmissionResult Fields :=
  match value with
  | .object fields => .ok fields
  | _ => .error (.native "keyword expansion" site .typeError)

def indexedMapping (value : Value) (site : String) : AdmissionResult Fields :=
  match value with
  | .object fields => .ok fields
  | _ => .error (.native "string-key subscription" site .typeError)

def erpKind : Value → AdmissionResult String
  | .text "target_only_structured_erp" => .ok "target_only_structured_erp"
  | .text "source_temporal_target_erp" => .ok "source_temporal_target_erp"
  | _ => .error (.runtime (.unrefined "non-ERP/global model-kind dispatcher"))

def numpyState (p : Operations) : List (String × TorchTensor) → AdmissionResult NativeArrays
  | [] => .ok []
  | (name,tensor) :: tail => do
    match ERPState.stripERP name with
    | none => numpyState p tail
    | some key =>
      let array ← liftNative "Tensor.numpy" key (p.numpy tensor)
      let rest ← numpyState p tail
      pure ((key,array)::rest)

def castAll (p : Operations) : NativeArrays → AdmissionResult PersonalERPFitOutcome.Arrays
  | [] => .ok []
  | (key,array) :: tail => do
    let converted ← liftNative "np.asarray(float64)" key (p.asFloat64 array)
    let copied ← liftNative "ndarray.copy" key (p.copyFloat64 converted)
    let rest ← castAll p tail
    pure ((key,copied)::rest)

def readNativeOutcome (p : Operations) (recordValue : Value) (arrays : NativeArrays) : AdmissionResult Model := do
  let record ← getMapping recordValue "ERP artifact"
  liftCheck (ensure (fieldDefault record "schema"=.text "n2p3_structured_erp/1" ∧
    fieldDefault record "feature_order"=.text "channel_prime") "ERP schema/order")
  let temporal ← liftCheck (optionalTemporal p.admission.original.temporal record)
  let header ← liftCheck (commonHeader p.admission.original.temporal record temporal)
  let (channels,times) ← liftCheck (readGeometry record)
  liftCheck (ensure (0 < times/header.binWidth ∧ PersonalAdmission.PyEq
    (fieldDefault record "discarded_tail_samples") (.integer (times%header.binWidth))) "ERP binning")
  let names := arrays.map Prod.fst
  liftCheck (ensure (names ⊆ expectedKeys header ∧ expectedKeys header ⊆ names) "ERP state keys")
  let copied ← castAll p arrays
  liftCheck (ensure (PersonalERPFitOutcome.arraysFinite copied=true) "ERP state contains nonfinite values")
  liftCheck (PersonalERPFitOutcome.finishFinite p.admission.original.reference record header channels times
    (PersonalERPFitOutcome.decodeArrays copied))

structure Predictor where
  metadata : Fields
  state : List (String × TorchTensor)
  device : String
  quality : Option PersonalERPQuality.EpochQuality
  erp : PersonalFullRecipeConsumer.Bound

def temporalColumns (p : Operations) (basis : TemporalModel) : AdmissionResult ℕ := do
  let matrix ← liftNative "TemporalBasis.matrix" "temporal recipe rank" (p.temporalMatrix basis)
  match matrix.image.shape with
  | _ :: columns :: _ => pure columns
  | _ => .error (.native "shape[1]" "temporal recipe rank" (.other "IndexError"))

def bindActual (p : Operations) (metadata contract : Fields) (indices : List ℕ)
    (model : Model) (recipe : PersonalTargetRecipe.Recipe) : AdmissionResult PersonalFullRecipeConsumer.Bound := do
  let times ← liftRuntime (PersonalERPModelRuntime.required contract "n_times")
  require (model.channels=indices.length ∧ PersonalAdmission.PyEq (.integer model.times) times)
    "actual ERP input shape physical contract"
  let preprocessingValue ← liftRuntime (PersonalERPModelRuntime.required contract "preprocessing")
  let preprocessing ← indexedMapping preprocessingValue "original contract preprocessing"
  let baseline ← liftRuntime (PersonalERPModelRuntime.required preprocessing "baseline_mode")
  require (PersonalAdmission.PyEq model.header.baseline baseline) "actual ERP baseline correspondence"
  require (recipe.initialization.text=model.header.family.text) "actual ERP recipe family correspondence"
  require (recipe.width=model.header.binWidth) "actual ERP recipe width correspondence"
  let source := PersonalERPRecipe.usesSource model
  require (fieldDefault metadata "source_pretraining_used"=.boolean source) "actual ERP strict source flag"
  let kind ← liftRuntime (PersonalERPModelRuntime.required metadata "model_kind")
  require (PersonalAdmission.PyEq kind (.text "source_temporal_target_erp") ↔ source=true)
    "actual ERP strict source kind"
  match model.header.temporal with
  | none => pure ()
  | some basis =>
    let expected := if source then "source_supervised" else "fixed_cosine"
    require (recipe.raw.temporalKind=.text expected) "actual ERP temporal kind"
    let columns ← temporalColumns p basis
    require (recipe.raw.temporalRank=.integer columns) "actual ERP temporal rank"
  pure ⟨recipe,indices,model⟩

def finishERP (p : Operations) (metadata : Fields) (state : List (String × TorchTensor))
    (device : String) (quality : Option PersonalERPQuality.EpochQuality) (contract : Fields) : AdmissionResult Predictor := do
  let kindValue ← liftRuntime (PersonalERPModelRuntime.required metadata "model_kind")
  let _ ← erpKind kindValue
  let indices ← liftRuntime (PersonalERPAdmissionRuntime.selected p.admission state contract)
  let recordValue ← liftRuntime (PersonalERPModelRuntime.required metadata "erp_artifact")
  let arrays ← numpyState p state
  let model ← readNativeOutcome p recordValue arrays
  let kwargsValue ← liftRuntime (PersonalERPModelRuntime.required metadata "recipe")
  let kwargs ← kwargsMapping kwargsValue "TargetRecipe kwargs"
  let recipe ← liftRecipe (PersonalTargetRecipe.fromKwargs p.recipe (PersonalRecipePython.embedFields kwargs))
  let bound ← bindActual p metadata contract indices model recipe
  liftCheck (allStateFinite p.admission.original state)
  pure ⟨metadata,state,device,quality,bound⟩

def constructERP (p : Operations) (metadataValue : Value) (state : TorchState) (deviceName : String) : AdmissionResult Predictor := do
  let metadataValue ← liftCheck (p.admission.original.finiteJSON metadataValue)
  let state ← liftCheck (copyTorchEntries p.admission.original state.entries)
  let device ← liftCheck (p.admission.original.device deviceName)
  let metadata ← getMapping metadataValue "decision_score metadata"
  liftRuntime (PersonalERPAdmissionRuntime.checkDecision p.admission.finite metadata)
  let quality ← liftRuntime (PersonalERPAdmissionRuntime.readQuality p.admission metadata)
  let contractValue ← liftRuntime (PersonalERPModelRuntime.required metadata "input_contract")
  let contract ← contractDictionary contractValue
  require (PyTruthy (fieldDefault metadata "participant_key")) "calibrated model lacks participant and input contracts"
  let rawArgsValue ← liftRuntime (PersonalERPModelRuntime.required contract "preprocessing")
  let rawArgs ← kwargsMapping rawArgsValue "PreprocessingSpec kwargs"
  let args ← liftCheck (readPreprocessingArgs rawArgs)
  let binding ← liftRuntime (PersonalERPExecutionRuntime.executionBinding p.admission.execution args
    (metadata.get "preprocessing_execution"))
  require (PersonalAdmission.PyEq (.object (PersonalERPExecution.bindingFields binding))
    ((metadata.get "preprocessing_execution_binding").getD (.object .nil))) "personal saved entire execution binding"
  match quality with
  | none => pure ()
  | some quality =>
    liftRuntime (PersonalERPAdmissionRuntime.checkPersonalRawQuality p.admission
      ((contract.get "continuous_artifact_processing").getD (.object .nil)) quality)
    let reference ← liftRuntime (PersonalERPModelRuntime.required contract "source_reference")
    let channels ← liftRuntime (PersonalERPModelRuntime.required contract "channel_names")
    let preprocessing ← liftRuntime (PersonalERPModelRuntime.required contract "preprocessing")
    let expected := Fields.ofList [("participant_key",fieldDefault metadata "participant_key"),
      ("source_reference",reference),("channel_names",channels),("preprocessing",preprocessing)]
    let provenance ← liftCheck (PersonalERPAdmission.qualityProvenance quality)
    liftRuntime (PersonalERPAdmissionRuntime.qualityFitFields p.admission provenance expected
      ["participant_key","source_reference","channel_names","preprocessing"])
    liftRuntime (PersonalERPAdmissionRuntime.qualityBudget metadata contract quality provenance)
  finishERP p metadata state device quality contract

theorem numpy_failure_stops_before_remaining_tensors (p : Operations) (name key : String)
    (tensor : TorchTensor) (tail : List (String × TorchTensor)) (exception : PersonalRecipePython.Exception)
    (strip : ERPState.stripERP name=some key) (failure : p.numpy tensor=.error exception) :
    numpyState p ((name,tensor)::tail)=.error (.native "Tensor.numpy" key exception) := by
  simp [numpyState,strip,liftNative,failure]

theorem float64_failure_stops_before_copy_and_remaining_arrays (p : Operations) (key : String)
    (array : NativeArray) (tail : NativeArrays) (exception : PersonalRecipePython.Exception)
    (failure : p.asFloat64 array=.error exception) :
    castAll p ((key,array)::tail)=.error (.native "np.asarray(float64)" key exception) := by
  simp [castAll,liftNative,failure]

theorem array_copy_failure_stops_before_remaining_casts (p : Operations) (key : String)
    (array : NativeArray) (converted : PersonalERPFitOutcome.Array64) (tail : NativeArrays)
    (exception : PersonalRecipePython.Exception) (cast : p.asFloat64 array=.ok converted)
    (copy : p.copyFloat64 converted=.error exception) :
    castAll p ((key,array)::tail)=.error (.native "ndarray.copy" key exception) := by
  simp [castAll,liftNative,cast,copy]

theorem device_failure_precedes_nonmapping_metadata (p : Operations) (raw normalized : Value)
    (state : TorchState) (copied : List (String × TorchTensor)) (device : String) (fault : LedgerFault)
    (json : p.admission.original.finiteJSON raw=.ok normalized)
    (clones : copyTorchEntries p.admission.original state.entries=.ok copied)
    (failure : p.admission.original.device device=.error fault) :
    constructERP p raw state device=.error (.runtime (.inherited fault)) := by
  simp [constructERP,liftCheck,json,clones,failure]

def noQualityRawMetadata (authority : String) (contract kwargs : Fields) (artifact : Value) : Fields :=
  Fields.ofList [("participant_key",.text authority),("input_contract",.object contract),
    ("recipe",.object kwargs),("erp_artifact",artifact),("model_kind",.text "target_only_structured_erp"),
    ("source_pretraining_used",.boolean false)]

theorem prefix_from_fields_and_each_copy_operation (p : Operations) (authority : String)
    (contract kwargs rawArgs : Fields) (artifact : Value) (state : TorchState) (requested device : String)
    (authorityNonempty : authority ≠ "")
    (preprocessing : contract.get "preprocessing"=some (.object rawArgs))
    (known : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (json : p.admission.original.finiteJSON (.object (noQualityRawMetadata authority contract kwargs artifact))=
      .ok (.object (noQualityRawMetadata authority contract kwargs artifact)))
    (copies : ∀ entry ∈ state.entries, p.admission.original.detachCPUClone entry.2=.ok entry.2)
    (deviceResult : p.admission.original.device requested=.ok device) :
    constructERP p (.object (noQualityRawMetadata authority contract kwargs artifact)) state requested=
      finishERP p (noQualityRawMetadata authority contract kwargs artifact) state.entries device none contract := by
  have copied := PersonalERPAdmissionRuntime.copied_entries_from_individual_clone_operations
    p.admission.original state.entries copies
  unfold constructERP
  rw [json]
  simp only [liftCheck,Except.mapError,Except.bind,Bind.bind] at copied ⊢
  rw [copied,deviceResult]
  simp [noQualityRawMetadata,getMapping,contractDictionary,kwargsMapping,
    PersonalERPAdmissionRuntime.checkDecision,PersonalERPAdmissionRuntime.readQuality,
    PersonalERPModelRuntime.required,StateLedger.required,fieldDefault,Fields.ofList,Fields.get,
    PyTruthy,authorityNonempty,readPreprocessingArgs,known,ensure,require,liftRuntime,
    PersonalERPScalar.liftCheck,PersonalERPScalar.require,
    PersonalERPExecutionRuntime.executionBinding,PersonalERPExecution.bindingFields,
    PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,preprocessing]

theorem device_error_from_each_clone_precedes_metadata_get (p : Operations) (raw normalized : Value)
    (state : TorchState) (device : String) (fault : LedgerFault)
    (json : p.admission.original.finiteJSON raw=.ok normalized)
    (copies : ∀ entry ∈ state.entries, p.admission.original.detachCPUClone entry.2=.ok entry.2)
    (failure : p.admission.original.device device=.error fault) :
    constructERP p raw state device=.error (.runtime (.inherited fault)) := by
  exact device_failure_precedes_nonmapping_metadata p raw normalized state state.entries device fault json
    (PersonalERPAdmissionRuntime.copied_entries_from_individual_clone_operations
      p.admission.original state.entries copies) failure

def unitSelector : TorchTensor :=
  ⟨"torch.int64",⟨.integer,[1],[.integer 0],rfl⟩⟩

def oneERPTensorState (tensor : TorchTensor) : TorchState :=
  ⟨[("waveform_channels",unitSelector),("erp.coefficient",tensor)],by simp⟩

def oneChannelContract (rawArgs : Fields) : Fields :=
  Fields.ofList [("n_channels",.integer 1),("preprocessing",.object rawArgs)]

theorem selector_from_three_native_atoms (p : Operations) (tensor : TorchTensor) (rawArgs : Fields)
    (unique : p.admission.selectedUniqueCount unitSelector=.ok 1)
    (negative : p.admission.selectedAnyNegative unitSelector=.ok false)
    (upper : p.admission.selectedAnyTooHigh unitSelector (.integer 1)=.ok false) :
    PersonalERPAdmissionRuntime.selected p.admission (oneERPTensorState tensor).entries
      (oneChannelContract rawArgs)=.ok [0] := by
  simp only [unitSelector] at unique negative upper
  simp [PersonalERPAdmissionRuntime.selected,oneERPTensorState,unitSelector,oneChannelContract,
    requiredTorch,lookupTorch,PersonalERPModelRuntime.required,StateLedger.required,
    Fields.ofList,Fields.get,PersonalERPScalar.require,PersonalERPScalar.liftCheck,
    unique,negative,upper,strictInteger]

theorem numpy_error_precedes_raw_artifact_mapping_and_headers (p : Operations) (tensor : TorchTensor)
    (artifact : Value) (rawArgs : Fields) (exception : PersonalRecipePython.Exception)
    (device : String)
    (unique : p.admission.selectedUniqueCount unitSelector=.ok 1)
    (negative : p.admission.selectedAnyNegative unitSelector=.ok false)
    (upper : p.admission.selectedAnyTooHigh unitSelector (.integer 1)=.ok false)
    (failure : p.numpy tensor=.error exception) :
    finishERP p (noQualityRawMetadata "authority" (oneChannelContract rawArgs) .nil artifact)
      (oneERPTensorState tensor).entries device none (oneChannelContract rawArgs)=
      .error (.native "Tensor.numpy" "coefficient" exception) := by
  have selected := selector_from_three_native_atoms p tensor rawArgs unique negative upper
  dsimp only [oneERPTensorState] at selected
  simp [finishERP,noQualityRawMetadata,erpKind,PersonalERPModelRuntime.required,StateLedger.required,
    Fields.ofList,Fields.get,selected,liftRuntime,PersonalERPScalar.liftCheck,numpyState,oneERPTensorState,
    ERPState.stripERP,liftNative,failure]

theorem full_constructor_numpy_error_from_independent_atoms (p : Operations) (tensor : TorchTensor)
    (artifact : Value) (rawArgs : Fields) (requested device : String)
    (exception : PersonalRecipePython.Exception) (known : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (json : p.admission.original.finiteJSON (.object (noQualityRawMetadata "authority"
      (oneChannelContract rawArgs) .nil artifact))=.ok (.object (noQualityRawMetadata "authority"
        (oneChannelContract rawArgs) .nil artifact)))
    (copies : ∀ entry ∈ (oneERPTensorState tensor).entries,
      p.admission.original.detachCPUClone entry.2=.ok entry.2)
    (deviceResult : p.admission.original.device requested=.ok device)
    (unique : p.admission.selectedUniqueCount unitSelector=.ok 1)
    (negative : p.admission.selectedAnyNegative unitSelector=.ok false)
    (upper : p.admission.selectedAnyTooHigh unitSelector (.integer 1)=.ok false)
    (failure : p.numpy tensor=.error exception) :
    constructERP p (.object (noQualityRawMetadata "authority" (oneChannelContract rawArgs) .nil artifact))
      (oneERPTensorState tensor) requested=.error (.native "Tensor.numpy" "coefficient" exception) := by
  rw [prefix_from_fields_and_each_copy_operation p "authority" (oneChannelContract rawArgs)
    .nil rawArgs artifact (oneERPTensorState tensor) requested device (by decide)
    (by simp [oneChannelContract,Fields.ofList,Fields.get]) known json copies deviceResult]
  exact numpy_error_precedes_raw_artifact_mapping_and_headers p tensor artifact rawArgs exception device
    unique negative upper failure

end
end N2P3.PersonalERPFullAdmission
