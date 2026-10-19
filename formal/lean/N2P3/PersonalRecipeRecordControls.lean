import N2P3.PersonalRecipeRecordConsumer

/-! Independent source-field witnesses for record omission and normalization.
These use the already declared single scalar operations and a declared single
NumPy-scalar deepcopy response. They run no Python source or native probes. -/
namespace N2P3.PersonalRecipeRecordControls
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipeRecord N2P3.PersonalRecipeRecordJSON N2P3.PersonalRecipeRecordProducer
open N2P3.PersonalRecipeObserved N2P3.PersonalRecipeControls
noncomputable section
set_option maxRecDepth 4096
set_option exponentiation.threshold 1024
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def copies : CopyOperations := ⟨fun _ => .ok ()⟩

theorem all_eight_saved_fields_have_individual_copy_facts (initialization : Initialization) :
    CopyFieldsFacts copies (allRecordFields (fixture initialization).raw) := by
  cases initialization <;>
    norm_num +decide [fixture, lda, base, Initialization.isLDA, Arguments.raw, allRecordFields,
      Fields.ofList, adapterRecordValue]
  all_goals repeat' constructor

theorem all_eight_default_comparisons_are_individually_true (initialization : Initialization) :
    EqualityAtoms primitives (fixture initialization).raw true true true := by
  constructor <;> cases initialization <;>
    norm_num +decide [fixture, lda, base, Initialization.isLDA, Arguments.raw, primitives,
      ordinaryCompare, numberOf, compareNumbers, numberEq]

theorem all_eight_JSON_fields_have_individual_integer_codec_facts (initialization : Initialization) :
    JSONFieldsFacts primitives (allRecordFields (fixture initialization).raw) := by
  cases initialization <;>
    norm_num +decide [fixture, lda, base, Initialization.isLDA, Arguments.raw, allRecordFields,
      Fields.ofList, adapterRecordValue, JSONFieldsFacts, JSONLeafFacts, primitives, jsonInteger, huge100]

theorem all_eight_default_loaded_arguments_are_the_same_fields (initialization : Initialization) :
    loaded (fixture initialization) true true true = fixture initialization := by
  cases initialization <;>
    norm_num +decide [loaded, fixture, lda, base, Initialization.isLDA, normalValue]

theorem all_eight_default_constructed_arguments_are_the_same_fields (initialization : Initialization) :
    constructed (fixture initialization) none = fixture initialization := by
  cases initialization <;> rfl

theorem all_eight_source_adapters_are_absent (initialization : Initialization) :
    AdapterFields primitives (fixture initialization).raw none := by
  apply AdapterFields.absent
  · cases initialization <;> rfl
  · cases initialization <;> simp [fixture, lda, base, Initialization.isLDA, Arguments.raw]

theorem all_eight_genuine_constructors_save_JSON_and_reconstruct (initialization : Initialization) :
    constructSaveThenReconstruct copies primitives (fixture initialization).raw =
      .ok ((fixture initialization).recipe none) := by
  have hc := all_eight_default_constructed_arguments_are_the_same_fields initialization
  have hl := all_eight_default_loaded_arguments_are_the_same_fields initialization
  have hcombined : loaded (constructed (fixture initialization) none) true true true = fixture initialization := by
    rw [hc, hl]
  have h := genuine_constructor_record_JSON_reconstruction_from_individual_fields copies primitives
    (fixture initialization) none none true true true
    (all_eight_fixture_structural_fields initialization) (all_eight_fixture_single_operations initialization)
    (all_eight_source_adapters_are_absent initialization)
    (by simpa only [hc] using all_eight_saved_fields_have_individual_copy_facts initialization)
    (by simpa only [hc] using all_eight_default_comparisons_are_individually_true initialization)
    (by simpa only [hc] using all_eight_JSON_fields_have_individual_integer_codec_facts initialization)
    (by simpa only [hcombined] using all_eight_fixture_structural_fields initialization)
    (by simpa only [hcombined] using all_eight_fixture_single_operations initialization)
    (by simpa only [hcombined] using all_eight_source_adapters_are_absent initialization)
  simpa only [hcombined] using h

def baseKeys : List String := ["name", "initialization", "input_statistics", "bn_buffers", "bn_affine",
  "weights", "dropout", "batch_size", "lr", "weight_decay", "epochs", "max_updates", "seed", "refit_full_prefix"]

theorem source_actual_saved_record_has_fourteen_fields :
    (record copies primitives base.raw).map Fields.keys = .ok baseKeys := by
  rw [record_image_from_individual_copy_and_equality_atoms copies primitives base.raw true true true
    (all_eight_saved_fields_have_individual_copy_facts .source)
    (all_eight_default_comparisons_are_individually_true .source)]
  norm_num +decide [image, base, Arguments.raw, allRecordFields, erase, Fields.ofList, Fields.keys, baseKeys]

theorem source_saved_record_does_not_contain_an_explicit_default_temperature :
    (record copies primitives base.raw).map (Fields.get "candidate_loss_temperature") = .ok none := by
  rw [record_image_from_individual_copy_and_equality_atoms copies primitives base.raw true true true
    (all_eight_saved_fields_have_individual_copy_facts .source)
    (all_eight_default_comparisons_are_individually_true .source)]
  norm_num +decide [image, base, Arguments.raw, allRecordFields, erase, Fields.ofList, Fields.get]

theorem nonNone_loss_keeps_default_temperature_in_the_actual_field_list :
    (image { base.raw with lossWeight := .floating (.finite 0) } true true true).get
      "candidate_loss_temperature" = some (.floating (.finite 1)) := by
  have hn : (Value.floating (.finite 0)) ≠ .null := by intro h; cases h
  norm_num +decide [image, base, Arguments.raw, allRecordFields, erase, Fields.ofList, Fields.get, hn]

theorem typed_adapter_record_contains_exactly_the_ten_constructor_keys :
    (adapterRecordValue (.existing defaultConfig)) =
      .object (PersonalFeatureTransferConfig.kwargsFields (PersonalFeatureTransferConfig.asRaw defaultConfig)) := rfl

theorem typed_adapter_asdict_image_contains_no_record_schema_key :
    (PersonalFeatureTransferConfig.kwargsFields (PersonalFeatureTransferConfig.asRaw defaultConfig)).get "schema" = none := by
  norm_num +decide [PersonalFeatureTransferConfig.kwargsFields, Fields.ofList, Fields.get]

theorem omitted_integer_zero_and_one_reconstruct_builtin_float_defaults :
    (restored (normalRaw { base.raw with temperature := .integer 1, prior := .integer 0 }) true true true).temperature =
        .floating (.finite 1) ∧
    (restored (normalRaw { base.raw with temperature := .integer 1, prior := .integer 0 }) true true true).prior =
        .floating (.finite 0) := by
  norm_num +decide [restored, normalRaw, base, Arguments.raw, normalValue]

theorem retained_numpy_float64_is_normalized_before_kwargs_reconstruction :
    (normalRaw { base.raw with prior := .numpyFloat64 (.finite (1 / 2)) }).prior = .floating (.finite (1 / 2)) := rfl

end
end N2P3.PersonalRecipeRecordControls
