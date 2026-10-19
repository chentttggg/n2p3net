import N2P3.PersonalRecipeRecordJSON

/-! Source-field producer relationship: actual TargetRecipe construction,
asdict of its normalized typed adapter, conditional omission, metadata recipe
JSON, kwargs reconstruction, and binding. All success premises are individual
fields or single copy/cast/finite/comparison/codec replies. -/
namespace N2P3.PersonalRecipeRecordProducer
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipeRecord N2P3.PersonalRecipeRecordJSON
noncomputable section
set_option maxHeartbeats 800000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def constructed (f : Arguments) (adapter : Option FeatureConfig) : Arguments :=
  { f with adapter := match adapter with | none => .raw .null | some config => .existing config }

theorem constructed_fields_are_actual_constructor_output (f : Arguments) (adapter : Option FeatureConfig) :
    (constructed f adapter).raw = (f.recipe adapter).raw := rfl

def loaded (f : Arguments) (temperature prior width : Bool) : Arguments :=
  { f with
    batchSize := normalValue f.batchSize, lr := normalValue f.lr, weightDecay := normalValue f.weightDecay,
    epochs := normalValue f.epochs, maxUpdates := normalValue f.maxUpdates, seed := normalValue f.seed,
    refit := normalValue f.refit,
    adapter := if f.adapter = .raw .null then .raw .null else .raw (normalValue (adapterRecordValue f.adapter))
    lossWeight := normalValue f.lossWeight
    temperature := if f.lossWeight = .null ∧ temperature = true then .floating (.finite 1) else normalValue f.temperature
    prior := if prior then .floating (.finite 0) else normalValue f.prior
    width := if width then 1 else f.width
    rank := if f.kind = none then none else f.rank }

theorem loaded_fields_are_derived_from_source_fields_and_defaults (f : Arguments) (temperature prior width : Bool) :
    (loaded f temperature prior width).raw = restored (normalRaw f.raw) temperature prior width := by
  have ha := nested_adapter_None_identity_after_JSON f.raw
  cases hk : f.kind <;> cases hr : f.rank <;>
    cases temperature <;> cases prior <;> cases width <;>
      simp [loaded, Arguments.raw, restored, normalRaw, normalValue, adapterRecordValue,
        JSON_normalization_preserves_None_identity, hk, hr] at ha ⊢
  all_goals simp only [ha]

theorem actual_record_constructor_success_from_loaded_fields_and_each_operation
    (copy : CopyOperations) (p : Primitives) (f : Arguments) (temperature prior width : Bool)
    (sourceCopies : CopyFieldsFacts copy (allRecordFields f.raw))
    (equalities : EqualityAtoms p f.raw temperature prior width)
    (json : JSONFieldsFacts p (allRecordFields f.raw))
    (adapter : Option FeatureConfig)
    (structureFields : Conditions (loaded f temperature prior width))
    (atoms : AtomicSuccess p (loaded f temperature prior width))
    (adapterFields : AdapterFields p (loaded f temperature prior width).raw adapter) :
    saveThenReconstruct copy p f.raw = .ok ((loaded f temperature prior width).recipe adapter) := by
  rw [actual_record_to_reconstructed_constructor_formula copy p f.raw temperature prior width sourceCopies equalities json,
    ← loaded_fields_are_derived_from_source_fields_and_defaults,
    complete_constructor_from_independent_fields p (loaded f temperature prior width) adapter structureFields atoms adapterFields]
  rfl

def constructSaveThenReconstruct (copy : CopyOperations) (p : Primitives) (r : Raw) : RecordResult Recipe := do
  let original ← (validate p r).mapError PersonalRecipeRecord.Fault.operation
  saveThenReconstruct copy p original.raw

theorem genuine_constructor_record_JSON_reconstruction_from_individual_fields
    (copy : CopyOperations) (p : Primitives) (f : Arguments)
    (sourceAdapter loadedAdapter : Option FeatureConfig) (temperature prior width : Bool)
    (sourceConditions : Conditions f) (sourceAtoms : AtomicSuccess p f)
    (sourceAdapterFields : AdapterFields p f.raw sourceAdapter)
    (copies : CopyFieldsFacts copy (allRecordFields (constructed f sourceAdapter).raw))
    (equalities : EqualityAtoms p (constructed f sourceAdapter).raw temperature prior width)
    (json : JSONFieldsFacts p (allRecordFields (constructed f sourceAdapter).raw))
    (loadedConditions : Conditions (loaded (constructed f sourceAdapter) temperature prior width))
    (loadedAtoms : AtomicSuccess p (loaded (constructed f sourceAdapter) temperature prior width))
    (loadedAdapterFields : AdapterFields p (loaded (constructed f sourceAdapter) temperature prior width).raw loadedAdapter) :
    constructSaveThenReconstruct copy p f.raw =
      .ok ((loaded (constructed f sourceAdapter) temperature prior width).recipe loadedAdapter) := by
  rw [constructSaveThenReconstruct,
    complete_constructor_from_independent_fields p f sourceAdapter sourceConditions sourceAtoms sourceAdapterFields]
  simp only [Except.mapError, Bind.bind, Except.bind]
  rw [← constructed_fields_are_actual_constructor_output]
  exact actual_record_constructor_success_from_loaded_fields_and_each_operation copy p
    (constructed f sourceAdapter) temperature prior width copies equalities json loadedAdapter
    loadedConditions loadedAtoms loadedAdapterFields

theorem reloaded_initialization_is_the_original_initialization (f : Arguments) (temperature prior width : Bool) :
    (loaded f temperature prior width).initialization = f.initialization := rfl

theorem reloaded_width_preserves_the_source_integer_comparison (f : Arguments) (temperature prior width : Bool)
    (widthComparison : width = true → f.width = 1) : (loaded f temperature prior width).width = f.width := by
  cases width
  · rfl
  · exact (widthComparison rfl).symm

theorem reloaded_temporal_declaration_is_unchanged (f : Arguments) (temperature prior width : Bool)
    (sourceAdapter loadedAdapter : Option FeatureConfig) :
    ((loaded f temperature prior width).recipe loadedAdapter).temporal = (f.recipe sourceAdapter).temporal := by
  cases hk : f.kind <;> cases hr : f.rank <;> simp [loaded, Arguments.recipe, hk, hr]

theorem reloaded_plain_binding_from_original_independent_physical_fields
    (f : Arguments) (temperature prior width : Bool) (adapter : Option FeatureConfig)
    (physical : N2P3.PersonalERPRecipe.Physical) (indices : List ℕ) (model : N2P3.ERPState.Model)
    (ht : model.header.temporal = none) (hc : model.channels = indices.length) (hn : model.times = physical.times)
    (hb : N2P3.PersonalAdmission.PyEq model.header.baseline physical.baseline)
    (hf : f.initialization.text = model.header.family.text) (hw : f.width = model.header.binWidth)
    (widthComparison : width = true → f.width = 1) :
    PersonalFullRecipeConsumer.bindERP ((loaded f temperature prior width).recipe adapter)
      "target_only_structured_erp" (.boolean false) physical indices model =
      .ok ⟨(loaded f temperature prior width).recipe adapter, indices, model⟩ := by
  apply PersonalFullRecipeConsumer.bind_plain_from_independent_fields _ physical indices model ht hc hn hb hf
  exact (reloaded_width_preserves_the_source_integer_comparison f temperature prior width widthComparison).trans hw

end
end N2P3.PersonalRecipeRecordProducer
