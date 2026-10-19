import N2P3.PersonalFullRecipeOutcome

/-! Strict false alternatives fixed by the source plan: config/scalar order,
conversion/range order, tuple/list hashability, indexed config, real nonERP
continuations, width ownership, declaration/name order, directAPI/JSON NaN,
and inner-reader/full-constructor/physical-binding/outer-finite priority. -/
namespace N2P3.NegativePersonalFullRecipe
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipeObserved N2P3.PersonalRecipeControls
noncomputable section
attribute [local instance] Classical.propDecidable

example : validate primitives { base.raw with
    weights := .text "adapter", adapter := .raw (.object (.cons "unknown_probe_key" (.integer 1) .nil)),
    temperature := .integer huge100 } = .error ⟨"candidate_loss_temperature", .numpyFinite, .typeError⟩ := by
  rw [C01_config_keyword_precedes_huge_scalar]
  simp +decide

example : PersonalFeatureTransferConfig.construct primitives
    { invariant := .integer 0, maxInvariant := .integer huge400 } =
      .error ⟨"feature-transfer channel counts", .validation, .valueError⟩ := by
  rw [C04_float_overflow_before_channel_bound]
  simp +decide

example : PersonalFeatureTransferConfig.construct primitives
    { maxInvariant := .integer huge400, maxTarget := .list .nil } =
      .error ⟨"max_target_residual", .validation, .typeError⟩ := by
  rw [C05_first_float_before_later_real_tag]
  simp +decide

example : PersonalFeatureTransferConfig.construct primitives { dropMode := .list .nil } =
    .error ⟨"source_drop_mode", .validation, .valueError⟩ := by
  rw [C11_list_drop_mode_unhashable]
  simp +decide

example : PersonalFeatureTransferConfig.construct primitives { dropMode := .tuple .nil } =
    .error ⟨"source_drop_mode", .setMembership, .typeError⟩ := by
  rw [C12_empty_tuple_drop_mode_hashable_invalid]
  simp +decide

example : validate primitives { base.raw with
    weights := .text "adapter", adapter := .raw (.object (.cons "source_domain_count" (.integer 2) .nil)),
    temperature := .integer huge100 } = .ok (base.recipe none) := by
  rw [C14_indexed_config_before_huge_scalar]
  simp only [reduceCtorEq]

example : validate primitives base.raw = .error ⟨"only ERP initializations", .validation, .valueError⟩ := by
  rw [N01_source_default_constructs]
  simp only [reduceCtorEq]

example : validate primitives { base.raw with binWidth := .integer 2 } = .ok (base.recipe none) := by
  rw [N08_source_nonunit_bin_rejected]
  simp only [reduceCtorEq]

example : validate primitives { base.raw with temporalKind := .list .nil } =
    .error ⟨"temporal basis kind", .setMembership, .typeError⟩ := by
  rw [N09_non_temporal_list_declaration_valueerror]
  simp +decide

example : validate primitives { base.raw with
    name := .integer 0, initialization := .text "scratch", inputStatistics := .text "source" } =
      .error ⟨"scratch calibration input statistics", .validation, .valueError⟩ := by
  rw [N13_name_attribute_error_before_scratch_policy]
  simp +decide

example : validate primitives { base.raw with lr := .floating .nan } =
    .error ⟨"adapter lr finite", .validation, .valueError⟩ := by
  rw [D01_direct_nan_lr_constructs]
  simp only [reduceCtorEq]

example : PersonalRecipeJSON.jsonThenRecipe primitives
    (.cons "name" (.text "full_recipe_probe") (.cons "lr" (.floating .nan) .nil)) =
      .ok (({ base with lr := .floating .nan }).recipe none) := by
  rw [D05_json_nan_rejected_before_constructor]
  simp only [reduceCtorEq]

example : PersonalFullRecipeOutcome.consume primitives ERPState.fixturePrimitives ERPState.fixtureReference
    PersonalERPFitOutcome.identityCopy ⟨1, 1, .text "none"⟩ "target_only_structured_erp" (.boolean false)
    (.cons "unknown_recipe_key" (.integer 0) .nil) (PersonalERPFitOutcome.artifactRecord PersonalERPFitOutcome.boundaryArtifact)
    (PersonalERPFitOutcome.producer PersonalERPFitOutcome.boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.core (.constructor ⟨"TargetRecipe kwargs", .keywordConstruction, .typeError⟩)) := by
  rw [PersonalFullRecipeOutcome.nonfinite_inner_output_precedes_all_full_constructor_fields]
  simp +decide

example : PersonalFullRecipeOutcome.consume primitives ERPState.fixturePrimitives ERPState.fixtureReference
    PersonalERPFitOutcome.identityCopy PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (allRecordFields { base.raw with temperature := .integer huge100 })
    PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.core (.inherited (.sourceReject "calibrated predictor contains nonfinite state"))) := by
  rw [PersonalFullRecipeOutcome.raw_huge_temperature_precedes_outer_nonfinite_gate]
  simp +decide

example : PersonalFullRecipeOutcome.consume primitives ERPState.fixturePrimitives ERPState.fixtureReference
    PersonalERPFitOutcome.identityCopy PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (allRecordFields base.raw) PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.core (.inherited (.sourceReject "calibrated predictor contains nonfinite state"))) := by
  rw [PersonalFullRecipeOutcome.source_constructs_then_physical_mismatch_precedes_outer]
  simp +decide

example : PersonalFullRecipeOutcome.consume primitives ERPState.fixturePrimitives ERPState.fixtureReference
    PersonalERPFitOutcome.identityCopy PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (allRecordFields (fixture .pooled).raw) PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .ok ⟨(fixture .pooled).recipe none, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by
  rw [PersonalFullRecipeOutcome.legal_pooled_fields_reach_outer_nonfinite_gate]
  simp only [reduceCtorEq]

end
end N2P3.NegativePersonalFullRecipe
