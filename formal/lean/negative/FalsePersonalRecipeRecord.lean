import N2P3.PersonalRecipeRecordFailures

/-! Source-derived false alternatives for actual asdict omission, numeric-tag
normalization, unsupported metadata values and producer/consumer order. -/
namespace N2P3.NegativePersonalRecipeRecord
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipeRecord N2P3.PersonalRecipeRecordJSON N2P3.PersonalRecipeRecordProducer
open N2P3.PersonalRecipeObserved N2P3.PersonalRecipeControls N2P3.PersonalRecipeRecordControls
open N2P3.PersonalRecipeRecordFailures
noncomputable section
attribute [local instance] Classical.propDecidable

example : (record copies primitives base.raw).map Fields.keys = .ok ((allRecordFields base.raw).keys) := by
  rw [source_actual_saved_record_has_fourteen_fields]
  simp +decide [baseKeys, allRecordFields, Fields.keys, Fields.ofList]

example : (image { base.raw with lossWeight := .floating (.finite 0) } true true true).get
    "candidate_loss_temperature" = none := by
  rw [nonNone_loss_keeps_default_temperature_in_the_actual_field_list]
  simp only [reduceCtorEq]

example : record failingCopy primitives { base.raw with prior := .numpyFloat64 (.finite 0) } =
    .ok (image { base.raw with prior := .numpyFloat64 (.finite 0) } true true true) := by
  rw [deepcopy_of_a_later_omitted_prior_still_happens_before_any_comparison]
  simp only [reduceCtorEq]

example : (restored (normalRaw { base.raw with temperature := .integer 1, prior := .integer 0 }) true true true).prior =
    .integer 0 := by
  rw [omitted_integer_zero_and_one_reconstruct_builtin_float_defaults.2]
  have hn : (Value.floating (.finite 0)) ≠ .integer 0 := by intro h; cases h
  simp only [hn]

example : (normalRaw { base.raw with prior := .numpyFloat64 (.finite (1 / 2)) }).prior =
    .numpyFloat64 (.finite (1 / 2)) := by
  rw [retained_numpy_float64_is_normalized_before_kwargs_reconstruction]
  have hn : (Value.floating (.finite (1 / 2))) ≠ .numpyFloat64 (.finite (1 / 2)) := by intro h; cases h
  simp only [hn]

example : recordJSON copies primitives numpySeed.raw = .ok (image (normalRaw numpySeed.raw) true true true) := by
  rw [retained_numpy_integer_seed_is_rejected_by_metadata_JSON]
  simp only [reduceCtorEq]

example : (PersonalFeatureTransferConfig.kwargsFields (PersonalFeatureTransferConfig.asRaw defaultConfig)).get "schema" =
    some (.text "n2p3_feature_transfer/1") := by
  rw [typed_adapter_asdict_image_contains_no_record_schema_key]
  simp only [reduceCtorEq]

example : PersonalRecipeRecordConsumer.consumeRecord copies primitives nanLR.raw N2P3.ERPState.fixturePrimitives
    N2P3.ERPState.fixtureReference N2P3.PersonalERPFitOutcome.identityCopy ⟨1, 1, .text "none"⟩
    "target_only_structured_erp" (.boolean false)
    (N2P3.PersonalERPFitOutcome.artifactRecord N2P3.PersonalERPFitOutcome.boundaryArtifact)
    (N2P3.PersonalERPFitOutcome.producer N2P3.PersonalERPFitOutcome.boundaryPayload.arrays
      N2P3.PersonalERPCurrentConsumer.floatingSelector) =
    .error (.inr (.core (.inherited (.sourceReject "ERP selected dtype")))) := by
  rw [retained_nan_record_JSON_error_is_before_even_a_bad_selector_and_nonfinite_inner]
  simp +decide

example : constructSaveThenReconstruct copies primitives integerDefaults.raw = .ok (integerDefaults.recipe none) := by
  rw [genuine_integer_default_constructor_saves_and_reloads_builtin_float_defaults]
  have hn : base.recipe none ≠ integerDefaults.recipe none := by
    intro h
    have ht := congrArg (fun r : Recipe => r.raw.temperature) h
    change Value.floating (.finite 1) = Value.integer 1 at ht
    cases ht
  simp only [Except.ok.injEq, hn]

example : PersonalRecipeRecordConsumer.consumeRecord copies primitives base.raw N2P3.ERPState.fixturePrimitives
    N2P3.ERPState.fixtureReference N2P3.PersonalERPFitOutcome.identityCopy N2P3.PersonalERPCurrentConsumer.fixturePhysical
    "target_only_structured_erp" (.boolean false) N2P3.PersonalERPCurrentConsumer.fixtureRecord
    N2P3.PersonalERPFitOutcome.nonERPExtraState =
    .error (.inr (.core (.inherited (.sourceReject "calibrated predictor contains nonfinite state")))) := by
  rw [actual_default_source_record_reaches_physical_binding_before_outer]
  simp +decide

end
end N2P3.NegativePersonalRecipeRecord
