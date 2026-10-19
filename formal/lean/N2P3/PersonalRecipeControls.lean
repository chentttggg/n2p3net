import N2P3.PersonalTargetRecipeFields
import N2P3.PersonalRecipeObserved
import N2P3.PersonalRecipeJSON

/-!
# Independent witnesses for the predeclared 35 constructor/source controls

The case plan predates its single source probe. These witnesses instantiate
the full executable constructor, not a whole-constructor oracle. Single
numeric observations remain declared backend witnesses. This does not prove
universal Python/NumPy/JSON implementation refinement or all P01 callers.
-/
namespace N2P3.PersonalRecipeControls
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe
open N2P3.PersonalTargetRecipeFields N2P3.PersonalRecipeObserved
noncomputable section
set_option maxRecDepth 4096
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def base : Arguments := { name := "full_recipe_probe", initialization := .source }

def lda (initialization : Initialization) : Arguments :=
  { base with
    initialization := initialization, weights := "none", affine := "frozen",
    buffers := "source", dropout := "off",
    kind := if initialization = .temporal then some "fixed_cosine" else none,
    rank := if initialization = .temporal then some 1 else none }

def fixture (initialization : Initialization) : Arguments :=
  if initialization.isLDA then lda initialization else { base with initialization := initialization }

def defaultConfig : FeatureConfig :=
  ⟨4, 4, 4, 9, PersonalFeatureTransferConfig.quarter, PersonalFeatureTransferConfig.quarter,
    0, 0, PersonalFeatureTransferConfig.threeTenths, .text "sample"⟩

theorem default_config_individual_casts :
    PersonalFeatureTransferConfig.CastFields primitives {} defaultConfig := by
  constructor
  · constructor <;> norm_num +decide [primitives, intCast, defaultConfig, isBuiltinBool, isIntegral]
  all_goals norm_num +decide [primitives, floatCast, defaultConfig, isBuiltinBool, isReal]

theorem default_config_individual_guards :
    PersonalFeatureTransferConfig.GuardFields primitives defaultConfig := by
  constructor <;> norm_num +decide [defaultConfig, primitives, mathFinite, numberFinite,
    ordinaryCompare, numberOf, compareNumbers, numberEq, numberLe,
    PersonalFeatureTransferConfig.quarter, PersonalFeatureTransferConfig.zero,
    PersonalFeatureTransferConfig.twoTenths, PersonalFeatureTransferConfig.threeTenths,
    PersonalFeatureTransferConfig.half]

theorem default_config_built_from_fields :
    PersonalFeatureTransferConfig.construct primitives {} = .ok defaultConfig :=
  PersonalFeatureTransferConfig.config_from_raw_fields_and_individual_operations
    primitives {} defaultConfig default_config_individual_casts default_config_individual_guards

theorem all_eight_fixture_structural_fields (initialization : Initialization) :
    Conditions (fixture initialization) := by
  cases initialization <;> constructor <;>
    norm_num +decide [fixture, lda, base, Initialization.isLDA, Initialization.isERP, isBuiltinBool]

theorem all_eight_fixture_single_operations (initialization : Initialization) :
    AtomicSuccess primitives (fixture initialization) := by
  constructor
  · cases initialization <;> constructor <;> norm_num +decide [fixture, lda, base, Arguments.raw,
      Initialization.isLDA, isBuiltinBool, recipeScalarTag,
      primitives, numpyFinite, numberFinite, ordinaryCompare, numberOf, compareNumbers, numberLt, numberLe]
  all_goals
    cases initialization <;>
      norm_num +decide [fixture, lda, base, Arguments.raw, project, Initialization.isLDA,
        primitives, stripObserved, mathFinite, numberFinite, ordinaryCompare, numberOf,
        compareNumbers, numberLt, numberLe, numberEq]

theorem every_initialization_constructs_from_independent_fields (initialization : Initialization) :
    validate primitives (fixture initialization).raw = .ok ((fixture initialization).recipe none) := by
  apply complete_constructor_from_independent_fields primitives (fixture initialization) none
    (all_eight_fixture_structural_fields initialization) (all_eight_fixture_single_operations initialization)
  apply AdapterFields.absent
  · cases initialization <;> rfl
  · cases initialization <;> simp [fixture, lda, base, Arguments.raw, Initialization.isLDA]

theorem N01_source_default_constructs : validate primitives base.raw = .ok (base.recipe none) := by
  exact every_initialization_constructs_from_independent_fields .source

theorem N02_scratch_default_constructs :
    validate primitives { base.raw with initialization := .text "scratch" } =
      .ok ((fixture .scratch).recipe none) := by
  exact every_initialization_constructs_from_independent_fields .scratch

theorem N03_source_adapter_constructs :
    validate primitives { base.raw with
      weights := .text "adapter" } =
      .ok (({ base with weights := "adapter" }).recipe (some defaultConfig)) := by
  apply complete_constructor_from_independent_fields primitives { base with weights := "adapter" }
    (some defaultConfig)
  · constructor <;> norm_num +decide [base, Initialization.isLDA, Initialization.isERP, isBuiltinBool]
  · constructor
    · have a := (all_eight_fixture_single_operations .source).toPrefixAtoms
      exact ⟨a.loss, a.temperatureTag, a.temperatureFinite, a.temperatureNegative,
        a.priorTag, a.priorFinite, a.priorNegative, a.temperaturePositive⟩
    all_goals
      norm_num +decide [base, Arguments.raw, project, primitives, stripObserved, mathFinite,
        numberFinite, ordinaryCompare, numberOf, compareNumbers, numberLt, numberLe, numberEq,
        Initialization.isLDA]
  · exact .defaultConfig defaultConfig rfl rfl default_config_individual_casts
      default_config_individual_guards rfl rfl

local macro "config_control" : tactic => `(tactic|
  norm_num +decide [PersonalFeatureTransferConfig.construct, PersonalFeatureTransferConfig.readIntegral,
    PersonalFeatureTransferConfig.readReal, PersonalFeatureTransferConfig.finitePositive,
    PersonalFeatureTransferConfig.dropBound, PersonalFeatureTransferConfig.fromKwargs,
    PersonalFeatureTransferConfig.readArgs, PersonalFeatureTransferConfig.getDefault,
    PersonalFeatureTransferConfig.names, PersonalFeatureTransferConfig.kwargsFields,
    PersonalFeatureTransferConfig.quarter, PersonalFeatureTransferConfig.twoTenths,
    PersonalFeatureTransferConfig.threeTenths, PersonalFeatureTransferConfig.half,
    PersonalFeatureTransferConfig.zero, primitives, intCast, floatCast, floatInteger,
    mathFinite, numberFinite, ordinaryCompare, numberOf, compareNumbers, numberLe, numberEq,
    isBuiltinBool, isIntegral, isReal, require, atom, rejectComparison, compareAtom,
    setOption, setContains, builtinHashable, allHashable, Fields.ofList, Fields.keys, Fields.get])

theorem C01_config_keyword_precedes_huge_scalar :
    validate primitives { base.raw with
      weights := .text "adapter",
      adapter := .raw (.object (.cons "unknown_probe_key" (.integer 1) .nil)),
      temperature := .integer huge100 } =
      .error ⟨"FeatureTransferConfig kwargs", .keywordConstruction, .typeError⟩ := by
  apply bad_config_keyword_precedes_every_scalar primitives _ _ rfl
  norm_num +decide [Fields.keys, PersonalFeatureTransferConfig.names,
    PersonalFeatureTransferConfig.kwargsFields, Fields.ofList]

theorem C02_bool_config_precedes_huge_scalar :
    validate primitives { base.raw with
      weights := .text "adapter",
      adapter := .raw (.object (.cons "invariant_bottleneck_channels" (.boolean true) .nil)),
      temperature := .integer huge100 } =
      .error ⟨"invariant_bottleneck_channels", .validation, .typeError⟩ := by
  unfold validate prepareAdapter convertAdapter
  config_control

theorem C03_all_integers_before_reals :
    PersonalFeatureTransferConfig.construct primitives
      { maxInvariant := .list .nil, sourceCount := .boolean false } =
      .error ⟨"source_domain_count", .validation, .typeError⟩ := by config_control

theorem C04_float_overflow_before_channel_bound :
    PersonalFeatureTransferConfig.construct primitives
      { invariant := .integer 0, maxInvariant := .integer huge400 } =
      .error ⟨"max_invariant_residual", .floatCast, .overflowError⟩ := by config_control

theorem C05_first_float_before_later_real_tag :
    PersonalFeatureTransferConfig.construct primitives
      { maxInvariant := .integer huge400, maxTarget := .list .nil } =
      .error ⟨"max_invariant_residual", .floatCast, .overflowError⟩ := by config_control

theorem C06_channel_bound_before_nan_residual :
    PersonalFeatureTransferConfig.construct primitives
      { invariant := .integer 0, maxInvariant := .floating .nan } =
      .error ⟨"feature-transfer channel counts", .validation, .valueError⟩ := by config_control

theorem C07_kernel_parity_before_nan_residual :
    PersonalFeatureTransferConfig.construct primitives
      { kernel := .integer 4, maxInvariant := .floating .nan } =
      .error ⟨"target_kernel_size", .validation, .valueError⟩ := by config_control

theorem C08_nonfinite_residual_rejected :
    PersonalFeatureTransferConfig.construct primitives { maxInvariant := .floating .nan } =
      .error ⟨"max_invariant_residual", .validation, .valueError⟩ := by config_control

theorem C09_private_count_before_nan_drop :
    PersonalFeatureTransferConfig.construct primitives { privateCount := .integer 1, drop := .floating .nan } =
      .error ⟨"private_domain_count", .validation, .valueError⟩ := by config_control

theorem C10_drop_real_tag_before_count_range :
    PersonalFeatureTransferConfig.construct primitives { sourceCount := .integer 1, drop := .list .nil } =
      .error ⟨"source_drop_p", .validation, .typeError⟩ := by config_control

theorem C11_list_drop_mode_unhashable :
    PersonalFeatureTransferConfig.construct primitives { dropMode := .list .nil } =
      .error ⟨"source_drop_mode", .setMembership, .typeError⟩ := by config_control

theorem C12_empty_tuple_drop_mode_hashable_invalid :
    PersonalFeatureTransferConfig.construct primitives { dropMode := .tuple .nil } =
      .error ⟨"source_drop_mode", .validation, .valueError⟩ := by config_control

theorem C13_tuple_with_list_drop_mode_unhashable :
    PersonalFeatureTransferConfig.construct primitives { dropMode := .tuple (.cons (.list .nil) .nil) } =
      .error ⟨"source_drop_mode", .setMembership, .typeError⟩ := by config_control

theorem C14_indexed_config_before_huge_scalar :
    validate primitives { base.raw with
      weights := .text "adapter",
      adapter := .raw (.object (.cons "source_domain_count" (.integer 2) .nil)),
      temperature := .integer huge100 } =
      .error ⟨"target residual unindexed FeatureTransferConfig", .validation, .valueError⟩ := by
  unfold validate prepareAdapter convertAdapter requireUnindexed
  config_control

theorem C15_nonadapter_config_before_huge_scalar :
    validate primitives { base.raw with
      weights := .text "none", adapter := .raw (.object .nil),
      temperature := .integer huge100 } =
      .error ⟨"target_adapter_config requires adapter weights", .validation, .valueError⟩ := by
  unfold validate prepareAdapter convertAdapter
  config_control

theorem C16_default_config_then_huge_scalar :
    validate primitives { base.raw with
      weights := .text "adapter", temperature := .integer huge100 } =
      .error ⟨"candidate_loss_temperature", .numpyFinite, .typeError⟩ := by
  have hp : prepareAdapter primitives { base.raw with
      weights := .text "adapter", temperature := .integer huge100 } = .ok (some defaultConfig) := by
    simp [prepareAdapter, convertAdapter, base, Arguments.raw,
      default_config_built_from_fields, requireUnindexed, defaultConfig, require]
  rw [validate, hp]
  norm_num +decide [base, Arguments.raw, require, scalarPrefix, scalarField,
    isBuiltinBool, recipeScalarTag, atom, primitives, numpyFinite]

local macro "recipe_control" : tactic => `(tactic|
  norm_num +decide [validate, prepareAdapter, convertAdapter, scalarPrefix, scalarField, remainder,
    options, initializationOptions, classifyInitialization, checkName, positiveBuiltinInt,
    subjectChecks, project, Arguments.raw, Arguments.recipe, fixture, lda, base,
    Initialization.text, Initialization.isLDA, Initialization.isERP,
    primitives, numpyFinite, mathFinite, numberFinite, ordinaryCompare, numberOf,
    compareNumbers, numberLt, numberLe, numberEq, stripObserved,
    require, atom, rejectComparison, compareAtom, setOption, setContains,
    builtinHashable, allHashable, isBuiltinBool, recipeScalarTag, isStrictInt])

theorem N04_scratch_statistics_before_bad_width :
    validate primitives { base.raw with
      initialization := .text "scratch",
      inputStatistics := .text "source", binWidth := .integer 0 } =
      .error ⟨"scratch calibration input statistics", .validation, .valueError⟩ := by recipe_control

theorem N05_waveform_frozen_guard_before_bad_width :
    validate primitives { base.raw with initialization := .text "waveform_lda", binWidth := .integer 0 } =
      .error ⟨"LDA frozen treatment", .validation, .valueError⟩ := by recipe_control

theorem N06_waveform_nonunit_bin_rejected :
    validate primitives { (lda .waveform).raw with binWidth := .integer 2 } =
      .error ⟨"erp_bin_width belongs only to ERP", .validation, .valueError⟩ := by recipe_control

theorem N07_source_feature_nonunit_bin_rejected :
    validate primitives { (lda .sourceFeature).raw with binWidth := .integer 2 } =
      .error ⟨"erp_bin_width belongs only to ERP", .validation, .valueError⟩ := by recipe_control

theorem N08_source_nonunit_bin_rejected :
    validate primitives { base.raw with binWidth := .integer 2 } =
      .error ⟨"erp_bin_width belongs only to ERP", .validation, .valueError⟩ := by recipe_control

theorem N09_non_temporal_list_declaration_valueerror :
    validate primitives { base.raw with temporalKind := .list .nil } =
      .error ⟨"non-temporal declaration", .validation, .valueError⟩ := by
  recipe_control
  have hn : (Value.list .nil) ≠ .null := by intro he; cases he
  simp only [if_neg hn]

theorem N10_temporal_list_kind_typeerror_before_rank :
    validate primitives { (lda .temporal).raw with temporalKind := .list .nil, temporalRank := .null } =
      .error ⟨"temporal basis kind", .setMembership, .typeError⟩ := by recipe_control

theorem N11_empty_tuple_initialization_invalid_value :
    validate primitives { base.raw with initialization := .tuple .nil } =
      .error ⟨"recipe initialization", .validation, .valueError⟩ := by recipe_control

theorem N12_tuple_with_list_initialization_typeerror :
    validate primitives { base.raw with initialization := .tuple (.cons (.list .nil) .nil) } =
      .error ⟨"recipe initialization", .setMembership, .typeError⟩ := by recipe_control

theorem N13_name_attribute_error_before_scratch_policy :
    validate primitives { base.raw with
      name := .integer 0, initialization := .text "scratch",
      inputStatistics := .text "source" } =
      .error ⟨"recipe name.strip", .nameStrip, .attributeError⟩ := by recipe_control

theorem D01_direct_nan_lr_constructs :
    validate primitives { base.raw with lr := .floating .nan } =
      .ok (({ base with lr := .floating .nan }).recipe none) := by recipe_control

theorem D02_direct_nan_epochs_constructs :
    validate primitives { base.raw with epochs := .floating .nan } =
      .ok (({ base with epochs := .floating .nan }).recipe none) := by recipe_control

theorem D03_direct_nan_temperature_rejected :
    validate primitives { base.raw with temperature := .floating .nan } =
      .error ⟨"candidate_loss_temperature", .validation, .valueError⟩ := by recipe_control

theorem D04_direct_nan_weight_decay_constructs :
    validate primitives { base.raw with weightDecay := .floating .nan } =
      .ok (({ base with weightDecay := .floating .nan }).recipe none) := by recipe_control

theorem D05_json_nan_rejected_before_constructor :
    PersonalRecipeJSON.jsonThenRecipe primitives
      (.cons "name" (.text "full_recipe_probe") (.cons "lr" (.floating .nan) .nil)) =
      .error ⟨"lr", .jsonDump, .valueError⟩ := rfl

/-- This begins at the parser's declared NaN output, not at allow_nan=False
serialization. It does not certify the Python text parser. -/
theorem D06_decoded_parser_nan_temperature_rejected :
    fromKwargs primitives (.cons "name" (.text "full_recipe_probe")
      (.cons "candidate_loss_temperature" (.floating .nan) .nil)) =
      .error ⟨"candidate_loss_temperature", .validation, .valueError⟩ := by
  unfold fromKwargs readArgs
  norm_num +decide [Fields.keys, keywordNames, allRecordFields, Fields.ofList, Fields.get, getDefault]
  recipe_control

end
end N2P3.PersonalRecipeControls
