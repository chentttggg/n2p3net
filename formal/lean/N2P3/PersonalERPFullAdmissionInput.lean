import N2P3.PersonalERPFullAdmissionSuccess
import N2P3.PersonalERPNineIdentityWitness

/-! The same returned model metadata and temporal basis drive raw admission.
The nine-event fixture is an admission witness, not a fitted or discriminating
model or a raw EDF/R01 producer. Source contract precedes proof execution. -/
namespace N2P3.PersonalERPFullAdmissionInput
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPAdmission N2P3.PersonalERPFullAdmission
open N2P3.PersonalERPFullAdmissionSuccess N2P3.PersonalERPNineInputWitness
noncomputable section
set_option maxHeartbeats 1000000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map

def savedPreprocessing : Fields := preprocessingRecord resolvedPreprocessing

theorem saved_preprocessing_has_known_keywords :
    fieldKeys savedPreprocessing ⊆ preprocessingKeywords := by
  simp [savedPreprocessing,preprocessingRecord,resolvedPreprocessing,
    PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,fieldDefault,Fields.ofList,Fields.get,
    Fields.erase,fieldKeys,preprocessingKeywords,preprocessingDefaults]

theorem saved_preprocessing_baseline_is_none :
    savedPreprocessing.get "baseline_mode" = some (.text "none") := by
  simp [savedPreprocessing,preprocessingRecord,resolvedPreprocessing,
    PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,fieldDefault,Fields.ofList,Fields.get,Fields.erase]

def metadata (f : PersonalERPRecipe.ERPFields) : Fields :=
  noQualityRawMetadata expectedAuthority expectedContract (PersonalERPRecipe.recordFields f.raw)
    (.object (plainRecord f.family f.width 4 1 2 (.text "none")))

def expectedPredictor (f : PersonalERPRecipe.ERPFields) (device : String)
    (coefficientTensor interceptTensor : TorchTensor)
    (coefficient : Fin (1 * (2 / f.width)) → ℝ) (intercept : ℝ) : PersonalERPFullAdmission.Predictor :=
  ⟨metadata f,(state coefficientTensor interceptTensor [0]).entries,device,none,
    ⟨(arguments f).recipe none,[0],PersonalERPRecipe.plainModel f 1 2 4 (.text "none") coefficient intercept⟩⟩

/-- Every successful-library statement here is one source atomic operation;
the constructor, recipe graph, input validator and their loops remain derived. -/
structure ModelAtoms (p : Operations) (f : PersonalERPRecipe.ERPFields) (requested device : String)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (coefficient : Fin (1 * (2 / f.width)) → ℝ) (intercept : ℝ) : Prop where
  width : 0 < f.width
  bins : 0 < 2 / f.width
  family : f.family ≠ .temporal ∧ f.family ≠ .xdawn
  kind : f.kind = none
  rank : f.rank = none
  fields : PersonalTargetRecipeFields.Conditions (arguments f)
  atoms : PersonalTargetRecipeFields.AtomicSuccess p.recipe (arguments f)
  json : p.admission.original.finiteJSON (.object (metadata f)) = .ok (.object (metadata f))
  cloneCoefficient : p.admission.original.detachCPUClone coefficientTensor = .ok coefficientTensor
  cloneIntercept : p.admission.original.detachCPUClone interceptTensor = .ok interceptTensor
  cloneSelector : p.admission.original.detachCPUClone (selectedTorch [0]) = .ok (selectedTorch [0])
  deviceResult : p.admission.original.device requested = .ok device
  unique : p.admission.selectedUniqueCount (selectedTorch [0]) = .ok 1
  negative : p.admission.selectedAnyNegative (selectedTorch [0]) = .ok false
  upper : p.admission.selectedAnyTooHigh (selectedTorch [0]) (.integer 1) = .ok false
  numpyCoefficient : p.numpy coefficientTensor = .ok coefficientArray
  numpyIntercept : p.numpy interceptTensor = .ok interceptArray
  castCoefficient : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient)
  copyCoefficient : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient)
  castIntercept : p.asFloat64 interceptArray = .ok (finiteIntercept intercept)
  copyIntercept : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept)
  finiteCoefficient : p.admission.original.tensorIsFinite coefficientTensor = .ok true
  finiteIntercept : p.admission.original.tensorIsFinite interceptTensor = .ok true
  finiteSelector : p.admission.original.tensorIsFinite (selectedTorch [0]) = .ok true

theorem constructor_uses_the_actual_nine_key_contract
    (p : Operations) (f : PersonalERPRecipe.ERPFields) (requested device : String)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (coefficient : Fin (1 * (2 / f.width)) → ℝ) (intercept : ℝ)
    (h : ModelAtoms p f requested device coefficientTensor interceptTensor coefficientArray interceptArray coefficient intercept) :
    constructERP p (.object (metadata f)) (state coefficientTensor interceptTensor [0]) requested =
      .ok (expectedPredictor f device coefficientTensor interceptTensor coefficient intercept) := by
  exact constructERP_plain_for_any_contract_from_its_read_fields_and_atomic_operations
    p expectedAuthority f [0] 1 2 savedPreprocessing expectedContract (.text "none") requested device
    coefficient intercept coefficientTensor interceptTensor coefficientArray interceptArray
    (by decide) saved_preprocessing_has_known_keywords (by simp) h.width (by decide) h.bins h.family
    (Or.inr rfl) h.kind h.rank saved_preprocessing_baseline_is_none
    (by simp [expectedContract,Fields.ofList,Fields.get])
    (by simp [expectedContract,Fields.ofList,Fields.get])
    (by simp [expectedContract,savedPreprocessing,Fields.ofList,Fields.get])
    h.fields h.atoms h.json h.cloneCoefficient h.cloneIntercept h.cloneSelector h.deviceResult
    h.unique h.negative h.upper h.numpyCoefficient h.numpyIntercept h.castCoefficient h.copyCoefficient
    h.castIntercept h.copyIntercept h.finiteCoefficient h.finiteIntercept h.finiteSelector

def validateReturnedModel (p : Operations) (predictor : PersonalERPFullAdmission.Predictor) :
    PersonalERPScalar.Result (PersonalERPRawInputTail.State raw) :=
  PersonalERPRawInputTail.validateInput (PersonalERPNineInputWitness.operations p.admission.execution)
    predictor.metadata predictor.erp.model.header.temporal raw

def constructThenValidate (p : Operations) (f : PersonalERPRecipe.ERPFields)
    (coefficientTensor interceptTensor : TorchTensor) (requested : String) :
    AdmissionResult (PersonalERPFullAdmission.Predictor × PersonalERPRawInputTail.State raw) := do
  let predictor ← constructERP p (.object (metadata f)) (state coefficientTensor interceptTensor [0]) requested
  let checked ← liftRuntime (validateReturnedModel p predictor)
  pure (predictor,checked)

theorem returned_model_admits_actual_nine_event_input
    (p : Operations) (f : PersonalERPRecipe.ERPFields) (device : String)
    (coefficientTensor interceptTensor : TorchTensor)
    (coefficient : Fin (1 * (2 / f.width)) → ℝ) (intercept : ℝ)
    (leaves : Leaves p.admission.execution) :
    ∃ checked : PersonalERPRawInputTail.State raw,
      validateReturnedModel p (expectedPredictor f device coefficientTensor interceptTensor coefficient intercept) = .ok checked ∧
      checked.checked.tensor.N = 9 ∧ checked.identityCache = some expectedIdentityRows := by
  exact complete_raw_input_from_individual_metadata_fields_and_scalar_operations
    p.admission.execution leaves (metadata f)
    (by simp [metadata,noQualityRawMetadata,Fields.ofList,Fields.get])
    (by simp [metadata,noQualityRawMetadata,Fields.ofList,Fields.get])
    (by simp [metadata,noQualityRawMetadata,Fields.ofList,Fields.get])

theorem complete_constructor_to_raw_input_from_individual_operations
    (p : Operations) (f : PersonalERPRecipe.ERPFields) (requested device : String)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (coefficient : Fin (1 * (2 / f.width)) → ℝ) (intercept : ℝ)
    (h : ModelAtoms p f requested device coefficientTensor interceptTensor coefficientArray interceptArray coefficient intercept)
    (leaves : Leaves p.admission.execution) :
    ∃ checked : PersonalERPRawInputTail.State raw,
      constructThenValidate p f coefficientTensor interceptTensor requested =
        .ok (expectedPredictor f device coefficientTensor interceptTensor coefficient intercept,checked) ∧
      checked.checked.tensor.N = 9 ∧ checked.identityCache = some expectedIdentityRows := by
  obtain ⟨checked,admitted,nine,identity⟩ :=
    returned_model_admits_actual_nine_event_input p f device coefficientTensor interceptTensor coefficient intercept leaves
  refine ⟨checked,?_,nine,identity⟩
  rw [constructThenValidate,constructor_uses_the_actual_nine_key_contract p f requested device
    coefficientTensor interceptTensor coefficientArray interceptArray coefficient intercept h]
  simp [liftRuntime,admitted]

end
end N2P3.PersonalERPFullAdmissionInput
