import N2P3.PersonalRecipeRecordControls

/-! Nonfinite/unsupported retained JSON values and an independently failing
single deepcopy leaf. These are declared operation witnesses, not new Python
source probe results or universal library implementation proofs. -/
namespace N2P3.PersonalRecipeRecordFailures
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipeRecord N2P3.PersonalRecipeRecordJSON N2P3.PersonalRecipeRecordProducer
open N2P3.PersonalRecipeObserved N2P3.PersonalRecipeControls N2P3.PersonalRecipeRecordControls
noncomputable section
set_option maxRecDepth 4096
set_option exponentiation.threshold 1024
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def nanLR : Arguments := { base with lr := .floating .nan }
def numpySeed : Arguments := { base with seed := .numpyInteger 0 }
def integerDefaults : Arguments := { base with temperature := .integer 1, prior := .integer 0 }
def sourceAdapter : Arguments := { base with weights := "adapter" }
def storedAdapter : Arguments := constructed sourceAdapter (some defaultConfig)
def rebuiltAdapter : Arguments :=
  { base with
    weights := "adapter"
    adapter := .raw (.object (PersonalFeatureTransferConfig.kwargsFields {})) }

local macro "constructor_fields" : tactic => `(tactic|
  constructor <;> norm_num +decide [nanLR, numpySeed, integerDefaults, sourceAdapter, rebuiltAdapter, base, Initialization.isLDA,
    Initialization.isERP, isBuiltinBool])

local macro "single_operations" : tactic => `(tactic|
  (constructor
   · constructor <;> norm_num +decide [nanLR, numpySeed, integerDefaults, sourceAdapter, rebuiltAdapter, base, Arguments.raw,
       isBuiltinBool, recipeScalarTag, primitives, numpyFinite, numberFinite,
       ordinaryCompare, numberOf, compareNumbers, numberLt, numberLe, huge100, huge400]
   all_goals norm_num +decide [nanLR, numpySeed, integerDefaults, sourceAdapter, rebuiltAdapter, base, Arguments.raw, project,
     Initialization.isLDA, primitives, stripObserved, mathFinite, numberFinite, ordinaryCompare,
     numberOf, compareNumbers, numberLt, numberLe, numberEq, huge100, huge400]))

theorem admitted_nan_learning_rate_still_reaches_record :
    validate primitives nanLR.raw = .ok (nanLR.recipe none) := D01_direct_nan_lr_constructs

theorem unchecked_numpy_seed_reaches_record :
    validate primitives numpySeed.raw = .ok (numpySeed.recipe none) := by
  apply complete_constructor_from_independent_fields primitives numpySeed none
  · constructor_fields
  · single_operations
  · exact .absent rfl (by simp [numpySeed, base, Arguments.raw])

theorem integer_defaults_are_admitted_before_record :
    validate primitives integerDefaults.raw = .ok (integerDefaults.recipe none) := by
  apply complete_constructor_from_independent_fields primitives integerDefaults none
  · constructor_fields
  · single_operations
  · exact .absent rfl (by simp [integerDefaults, base, Arguments.raw])

theorem nan_lr_individual_copy_facts : CopyFieldsFacts copies (allRecordFields nanLR.raw) := by
  norm_num +decide [nanLR, base, Arguments.raw, allRecordFields, Fields.ofList, adapterRecordValue]
  repeat' constructor

theorem numpy_seed_individual_copy_facts : CopyFieldsFacts copies (allRecordFields numpySeed.raw) := by
  norm_num +decide [numpySeed, base, Arguments.raw, allRecordFields, Fields.ofList, adapterRecordValue]
  repeat' constructor

theorem integer_defaults_individual_copy_facts : CopyFieldsFacts copies (allRecordFields integerDefaults.raw) := by
  norm_num +decide [integerDefaults, base, Arguments.raw, allRecordFields, Fields.ofList, adapterRecordValue]
  repeat' constructor

theorem nan_lr_default_equalities_are_separate_operations : EqualityAtoms primitives nanLR.raw true true true := by
  constructor <;> norm_num +decide [nanLR, base, Arguments.raw, primitives,
    ordinaryCompare, numberOf, compareNumbers, numberEq]

theorem numpy_seed_default_equalities_are_separate_operations : EqualityAtoms primitives numpySeed.raw true true true := by
  constructor <;> norm_num +decide [numpySeed, base, Arguments.raw, primitives,
    ordinaryCompare, numberOf, compareNumbers, numberEq]

theorem retained_nan_lr_is_rejected_by_metadata_JSON :
    recordJSON copies primitives nanLR.raw =
      .error (.operation ⟨"lr", .jsonDump, .valueError⟩) := by
  rw [recordJSON, record_image_from_individual_copy_and_equality_atoms copies primitives nanLR.raw true true true
    nan_lr_individual_copy_facts nan_lr_default_equalities_are_separate_operations]
  norm_num +decide [image, nanLR, base, Arguments.raw, allRecordFields, erase, Fields.ofList,
    PersonalRecipeJSON.normalizeFields, PersonalRecipeJSON.normalize, atom,
    primitives, jsonInteger, huge100]

theorem retained_numpy_integer_seed_is_rejected_by_metadata_JSON :
    recordJSON copies primitives numpySeed.raw =
      .error (.operation ⟨"seed", .jsonDump, .typeError⟩) := by
  rw [recordJSON, record_image_from_individual_copy_and_equality_atoms copies primitives numpySeed.raw true true true
    numpy_seed_individual_copy_facts numpy_seed_default_equalities_are_separate_operations]
  norm_num +decide [image, numpySeed, base, Arguments.raw, allRecordFields, erase, Fields.ofList,
    PersonalRecipeJSON.normalizeFields, PersonalRecipeJSON.normalize, atom,
    primitives, jsonInteger, huge100]

theorem genuine_nan_constructor_then_metadata_JSON_error :
    constructSaveThenReconstruct copies primitives nanLR.raw =
      .error (.operation ⟨"lr", .jsonDump, .valueError⟩) := by
  rw [constructSaveThenReconstruct, admitted_nan_learning_rate_still_reaches_record]
  simp only [Except.mapError, Bind.bind, Except.bind]
  change saveThenReconstruct copies primitives nanLR.raw = _
  simp [saveThenReconstruct, retained_nan_lr_is_rejected_by_metadata_JSON]

theorem genuine_numpy_seed_constructor_then_metadata_JSON_error :
    constructSaveThenReconstruct copies primitives numpySeed.raw =
      .error (.operation ⟨"seed", .jsonDump, .typeError⟩) := by
  rw [constructSaveThenReconstruct, unchecked_numpy_seed_reaches_record]
  simp only [Except.mapError, Bind.bind, Except.bind]
  change saveThenReconstruct copies primitives numpySeed.raw = _
  simp [saveThenReconstruct, retained_numpy_integer_seed_is_rejected_by_metadata_JSON]

def failingCopy : CopyOperations := ⟨fun _ => .error (.other "declared single deepcopy failure")⟩

theorem deepcopy_of_a_later_omitted_prior_still_happens_before_any_comparison (p : Primitives) :
    record failingCopy p { base.raw with prior := .numpyFloat64 (.finite 0) } =
      .error (.deepcopy "source_prior_weight" (.other "declared single deepcopy failure")) := by
  simp [record, copyFields, copyValue, copyScalar, failingCopy, allRecordFields,
    base, Arguments.raw, Fields.ofList, adapterRecordValue]

theorem integer_default_field_image_reloads_as_base_arguments : loaded integerDefaults true true true = base := by
  norm_num +decide [loaded, integerDefaults, base, normalValue]

theorem integer_defaults_JSON_leaf_facts : JSONFieldsFacts primitives (allRecordFields integerDefaults.raw) := by
  norm_num +decide [integerDefaults, base, Arguments.raw, allRecordFields, Fields.ofList,
    adapterRecordValue, JSONFieldsFacts, JSONLeafFacts, primitives, jsonInteger, huge100]

theorem integer_defaults_record_equality_atoms : EqualityAtoms primitives integerDefaults.raw true true true := by
  constructor <;> norm_num +decide [integerDefaults, base, Arguments.raw, primitives,
    ordinaryCompare, numberOf, compareNumbers, numberEq]

theorem genuine_integer_default_constructor_saves_and_reloads_builtin_float_defaults :
    constructSaveThenReconstruct copies primitives integerDefaults.raw = .ok (base.recipe none) := by
  have hc : constructed integerDefaults none = integerDefaults := rfl
  have hl : loaded (constructed integerDefaults none) true true true = base := by
    rw [hc, integer_default_field_image_reloads_as_base_arguments]
  have h := genuine_constructor_record_JSON_reconstruction_from_individual_fields copies primitives
    integerDefaults none none true true true (by constructor_fields) (by single_operations)
    (.absent rfl (by simp [integerDefaults, base, Arguments.raw]))
    (by simpa only [hc] using integer_defaults_individual_copy_facts)
    (by simpa only [hc] using integer_defaults_record_equality_atoms)
    (by simpa only [hc] using integer_defaults_JSON_leaf_facts)
    (by simpa only [hl] using all_eight_fixture_structural_fields .source)
    (by simpa only [hl] using all_eight_fixture_single_operations .source)
    (by simpa only [hl] using all_eight_source_adapters_are_absent .source)
  simpa only [hl] using h

theorem typed_adapter_source_copy_facts : CopyFieldsFacts copies (allRecordFields storedAdapter.raw) := by
  norm_num +decide [storedAdapter, constructed, sourceAdapter, base, Arguments.raw, allRecordFields,
    Fields.ofList, adapterRecordValue, defaultConfig, PersonalFeatureTransferConfig.kwargsFields,
    PersonalFeatureTransferConfig.asRaw]
  repeat' constructor

theorem typed_adapter_source_JSON_leaf_facts : JSONFieldsFacts primitives (allRecordFields storedAdapter.raw) := by
  norm_num +decide [storedAdapter, constructed, sourceAdapter, base, Arguments.raw, allRecordFields,
    Fields.ofList, adapterRecordValue, defaultConfig, PersonalFeatureTransferConfig.kwargsFields,
    PersonalFeatureTransferConfig.asRaw, PersonalFeatureTransferConfig.quarter, PersonalFeatureTransferConfig.threeTenths,
    JSONFieldsFacts, JSONLeafFacts, primitives, jsonInteger, huge100]

theorem typed_adapter_record_default_comparison_atoms : EqualityAtoms primitives storedAdapter.raw true true true := by
  constructor <;> norm_num +decide [storedAdapter, constructed, sourceAdapter, base, Arguments.raw,
    primitives, ordinaryCompare, numberOf, compareNumbers, numberEq]

theorem typed_adapter_record_JSON_reconstructs_its_ten_field_dictionary :
    loaded storedAdapter true true true = rebuiltAdapter := by
  norm_num +decide [loaded, storedAdapter, constructed, sourceAdapter, rebuiltAdapter, base,
    normalValue, normalFields, Fields.ofList, adapterRecordValue, defaultConfig,
    PersonalFeatureTransferConfig.kwargsFields, PersonalFeatureTransferConfig.asRaw]
  intro h
  cases h

theorem typed_adapter_reconstruction_has_the_same_normalized_recipe :
    rebuiltAdapter.recipe (some defaultConfig) = sourceAdapter.recipe (some defaultConfig) := rfl

theorem genuine_typed_adapter_record_JSON_and_dictionary_constructor_roundtrip :
    constructSaveThenReconstruct copies primitives sourceAdapter.raw = .ok (sourceAdapter.recipe (some defaultConfig)) := by
  have hl : loaded (constructed sourceAdapter (some defaultConfig)) true true true = rebuiltAdapter :=
    typed_adapter_record_JSON_reconstructs_its_ten_field_dictionary
  have h := genuine_constructor_record_JSON_reconstruction_from_individual_fields copies primitives
    sourceAdapter (some defaultConfig) (some defaultConfig) true true true
    (by constructor_fields) (by single_operations)
    (.defaultConfig defaultConfig rfl rfl default_config_individual_casts default_config_individual_guards rfl rfl)
    typed_adapter_source_copy_facts typed_adapter_record_default_comparison_atoms typed_adapter_source_JSON_leaf_facts
    (by rw [hl]; constructor_fields) (by rw [hl]; single_operations)
    (by rw [hl]; exact .dictionary {} defaultConfig rfl rfl
          default_config_individual_casts default_config_individual_guards rfl rfl)
  simpa only [hl, typed_adapter_reconstruction_has_the_same_normalized_recipe] using h

theorem retained_nan_record_JSON_error_is_before_even_a_bad_selector_and_nonfinite_inner :
    PersonalRecipeRecordConsumer.consumeRecord copies primitives nanLR.raw N2P3.ERPState.fixturePrimitives
      N2P3.ERPState.fixtureReference N2P3.PersonalERPFitOutcome.identityCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false)
      (N2P3.PersonalERPFitOutcome.artifactRecord N2P3.PersonalERPFitOutcome.boundaryArtifact)
      (N2P3.PersonalERPFitOutcome.producer N2P3.PersonalERPFitOutcome.boundaryPayload.arrays
        N2P3.PersonalERPCurrentConsumer.floatingSelector) =
      .error (.inl (.operation ⟨"lr", .jsonDump, .valueError⟩)) := by
  simp [PersonalRecipeRecordConsumer.consumeRecord, retained_nan_lr_is_rejected_by_metadata_JSON]

theorem actual_default_source_record_reaches_physical_binding_before_outer :
    PersonalRecipeRecordConsumer.consumeRecord copies primitives base.raw N2P3.ERPState.fixturePrimitives
      N2P3.ERPState.fixtureReference N2P3.PersonalERPFitOutcome.identityCopy N2P3.PersonalERPCurrentConsumer.fixturePhysical
      "target_only_structured_erp" (.boolean false) N2P3.PersonalERPCurrentConsumer.fixtureRecord
      N2P3.PersonalERPFitOutcome.nonERPExtraState =
      .error (.inr (.core (.inherited (.sourceReject "ERP state recipe physical contract")))) := by
  have hl : loaded base true true true = base := all_eight_default_loaded_arguments_are_the_same_fields Initialization.source
  apply PersonalRecipeRecordConsumer.actual_saved_nonERP_record_reaches_physical_binding_before_outer copies primitives
    base true true true (all_eight_saved_fields_have_individual_copy_facts .source)
    (all_eight_default_comparisons_are_individually_true .source)
    (all_eight_JSON_fields_have_individual_integer_codec_facts .source) none
  · simpa only [hl] using all_eight_fixture_structural_fields .source
  · simpa only [hl] using all_eight_fixture_single_operations .source
  · simpa only [hl] using all_eight_source_adapters_are_absent .source
  · exact Or.inl rfl

end
end N2P3.PersonalRecipeRecordFailures
