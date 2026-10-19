import N2P3.PersonalTargetRecipe

/-!
# Independent field/atomic-operation premises for all recipe treatments

This file derives constructors from raw fields, individual FeatureTransferConfig
casts/guards and single scalar operations. It supplies no whole configuration,
recipe, adapter-validation or desired-result primitive. Arbitrary extended
values still pass through PersonalTargetRecipe's executable error paths.
-/
namespace N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Arguments where
  name : String
  initialization : Initialization
  statistics : String := "target_prefix"
  buffers : String := "online"
  affine : String := "trainable"
  weights : String := "non_bn"
  dropout : String := "checkpoint"
  batchSize : Value := .integer 32
  lr : Value := .floating (.finite (1152921504606847 / 1152921504606846976))
  weightDecay : Value := .floating (.finite (7378697629483821 / 73786976294838206464))
  epochs : Value := .integer 60
  maxUpdates : Value := .null
  seed : Value := .integer 0
  refit : Value := .boolean false
  adapter : AdapterArgument := .raw .null
  lossWeight : Value := .null
  temperature : Value := .floating (.finite 1)
  prior : Value := .floating (.finite 0)
  width : ℕ := 1
  kind : Option String := none
  rank : Option ℕ := none

def Arguments.raw (f : Arguments) : Raw := {
  name := .text f.name, initialization := .text f.initialization.text,
  inputStatistics := .text f.statistics, buffers := .text f.buffers,
  affine := .text f.affine, weights := .text f.weights, dropout := .text f.dropout,
  batchSize := f.batchSize, lr := f.lr, weightDecay := f.weightDecay,
  epochs := f.epochs, maxUpdates := f.maxUpdates, seed := f.seed, refit := f.refit,
  adapter := f.adapter, lossWeight := f.lossWeight, temperature := f.temperature,
  prior := f.prior, binWidth := .integer f.width,
  temporalKind := match f.kind with | none => .null | some kind => .text kind,
  temporalRank := match f.rank with | none => .null | some rank => .integer rank }

inductive AdapterFields (p : Primitives) (r : Raw) : Option FeatureConfig → Prop
  | absent (field : r.adapter = .raw .null) (weights : r.weights ≠ .text "adapter") : AdapterFields p r none
  | defaultConfig (c : FeatureConfig) (field : r.adapter = .raw .null) (weights : r.weights = .text "adapter")
      (casts : PersonalFeatureTransferConfig.CastFields p {} c)
      (guards : PersonalFeatureTransferConfig.GuardFields p c)
      (sourceCount : c.sourceCount = 0) (privateCount : c.privateCount = 0) : AdapterFields p r (some c)
  | dictionary (raw : PersonalFeatureTransferConfig.Raw) (c : FeatureConfig)
      (field : r.adapter = .raw (.object (PersonalFeatureTransferConfig.kwargsFields raw)))
      (weights : r.weights = .text "adapter")
      (casts : PersonalFeatureTransferConfig.CastFields p raw c)
      (guards : PersonalFeatureTransferConfig.GuardFields p c)
      (sourceCount : c.sourceCount = 0) (privateCount : c.privateCount = 0) : AdapterFields p r (some c)
  | existing (c : FeatureConfig) (field : r.adapter = .existing c) (weights : r.weights = .text "adapter")
      (sourceCount : c.sourceCount = 0) (privateCount : c.privateCount = 0) : AdapterFields p r (some c)

theorem adapter_preparation_from_independent_fields (p : Primitives) (r : Raw)
    (config : Option FeatureConfig) (h : AdapterFields p r config) : prepareAdapter p r = .ok config := by
  cases h with
  | absent ha hw => simp [prepareAdapter, convertAdapter, ha, hw]
  | defaultConfig c ha hw hc hg hs hp =>
    have hf := PersonalFeatureTransferConfig.config_from_raw_fields_and_individual_operations p {} c hc hg
    simp [prepareAdapter, convertAdapter, ha, hw, hf, requireUnindexed, require, hs, hp]
  | dictionary raw c ha hw hc hg hs hp =>
    have hf := PersonalFeatureTransferConfig.kwargs_config_from_independent_fields p raw c hc hg
    simp [prepareAdapter, convertAdapter, ha, hw, hf, requireUnindexed, require, hs, hp]
  | existing c ha hw hs hp =>
    simp [prepareAdapter, convertAdapter, ha, hw, requireUnindexed, require, hs, hp]

structure PrefixAtoms (p : Primitives) (r : Raw) : Prop where
  loss : r.lossWeight = .null ∨
    (isBuiltinBool r.lossWeight = false ∧ recipeScalarTag r.lossWeight = true ∧
      p.numpyFinite r.lossWeight = .ok true ∧ p.compare .lt r.lossWeight (.integer 0) = .ok false)
  temperatureTag : isBuiltinBool r.temperature = false ∧ recipeScalarTag r.temperature = true
  temperatureFinite : p.numpyFinite r.temperature = .ok true
  temperatureNegative : p.compare .lt r.temperature (.integer 0) = .ok false
  priorTag : isBuiltinBool r.prior = false ∧ recipeScalarTag r.prior = true
  priorFinite : p.numpyFinite r.prior = .ok true
  priorNegative : p.compare .lt r.prior (.integer 0) = .ok false
  temperaturePositive : p.compare .le r.temperature (.integer 0) = .ok false

theorem scalar_prefix_from_each_atomic_response (p : Primitives) (r : Raw) (h : PrefixAtoms p r) :
    scalarPrefix p r = .ok () := by
  have hl : scalarField p "candidate_loss_weight" true r.lossWeight = .ok () := by
    rcases h.loss with hl | ⟨hb, ht, hf, hn⟩
    · simp [scalarField, hl]
    · have hnull : r.lossWeight ≠ .null := by intro he; simp [he, recipeScalarTag] at ht
      simp [scalarField, hnull, require, hb, ht, atom, hf, rejectComparison, compareAtom, hn]
  rw [scalarPrefix, hl]
  simp [scalarField, require, h.temperatureTag,
    h.temperatureFinite, h.temperatureNegative, h.priorTag, h.priorFinite, h.priorNegative,
    atom, rejectComparison, compareAtom, h.temperaturePositive]

structure Conditions (f : Arguments) : Prop where
  statistics : f.statistics ∈ ["source", "target_prefix"]
  buffers : f.buffers ∈ ["source", "calibration", "online"]
  affine : f.affine ∈ ["frozen", "trainable"]
  weights : f.weights ∈ ["none", "head", "non_bn", "spatial", "temporal", "adapter"]
  dropout : f.dropout ∈ ["off", "checkpoint"]
  scratch : f.initialization = .scratch → f.statistics = "target_prefix"
  frozen : f.initialization.isLDA = true → f.weights = "none" ∧ f.affine = "frozen" ∧ f.dropout = "off"
  noLDALoss : f.initialization.isLDA = true → f.lossWeight = .null
  width : 0 < f.width
  owner : f.width = 1 ∨ f.initialization.isERP = true
  temporal :
    (f.initialization = .temporal ∧ f.width = 1 ∧ ∃ kind rank,
      f.kind = some kind ∧ f.rank = some rank ∧ kind ∈ ["fixed_cosine", "source_supervised"] ∧ 0 < rank) ∨
    (f.initialization ≠ .temporal ∧ f.kind = none ∧ f.rank = none)
  statisticsOnly : f.weights = "none" → f.affine = "frozen" ∧ f.buffers ≠ "online" ∧ f.dropout = "off"
  maxTag : f.maxUpdates = .null ∨
    (isBuiltinBool f.maxUpdates = false ∧ (isStrictInt f.maxUpdates = true ∨
      (match f.maxUpdates with | .numpyInteger _ => True | _ => False)))
  refit : isBuiltinBool f.refit = true

structure AtomicSuccess (p : Primitives) (f : Arguments) : Prop extends PrefixAtoms p f.raw where
  name : ∃ stripped, p.strip f.name = .ok stripped ∧ stripped ≠ ""
  priorMath : p.mathFinite f.prior = .ok true
  epochs : p.compare .lt f.epochs (.integer 1) = .ok false
  batch : p.compare .lt f.batchSize (.integer 1) = .ok false
  lr : p.compare .le f.lr (.floating (.finite 0)) = .ok false
  max : f.maxUpdates ≠ .null → p.compare .lt f.maxUpdates (.integer 1) = .ok false
  lossMath : p.mathFinite (project f.raw).lossWeight = .ok true
  lossBound : p.compare .le (.floating (.finite 0)) (project f.raw).lossWeight = .ok true
  temperatureMath : p.mathFinite f.temperature = .ok true
  temperatureBound : p.compare .lt (.floating (.finite 0)) f.temperature = .ok true
  ldaPrior : f.initialization.isLDA = true → p.compare .ne f.prior (.integer 0) = .ok false

theorem literal_text_option_from_field_membership (field : String) (allowed : List (Option String))
    (text : String) (h : some text ∈ allowed) : setOption field allowed (.text text) = .ok () := by
  simp [setOption, setContains, builtinHashable, require, h]

theorem all_six_option_sets_from_independent_fields (f : Arguments) (h : Conditions f) : options f.raw = .ok () := by
  have hi : some f.initialization.text ∈ initializationOptions := by
    cases f.initialization <;> simp [Initialization.text, initializationOptions]
  have h1 := literal_text_option_from_field_membership "recipe initialization" _ _ hi
  have h2 := literal_text_option_from_field_membership "recipe input_statistics"
    [some "source", some "target_prefix"] f.statistics (by simpa using h.statistics)
  have h3 := literal_text_option_from_field_membership "recipe bn_buffers"
    [some "source", some "calibration", some "online"] f.buffers (by simpa using h.buffers)
  have h4 := literal_text_option_from_field_membership "recipe bn_affine"
    [some "frozen", some "trainable"] f.affine (by simpa using h.affine)
  have h5 := literal_text_option_from_field_membership "recipe weights"
    [some "none", some "head", some "non_bn", some "spatial", some "temporal", some "adapter"]
    f.weights (by simpa using h.weights)
  have h6 := literal_text_option_from_field_membership "recipe dropout"
    [some "off", some "checkpoint"] f.dropout (by simpa using h.dropout)
  simp [options, Arguments.raw, h1, h2, h3, h4, h5, h6]

theorem projected_subject_checks_from_each_atomic_response (p : Primitives) (f : Arguments)
    (h : Conditions f) (a : AtomicSuccess p f) : subjectChecks p (project f.raw) = .ok () := by
  have hheadMember : some (project f.raw).head ∈
      [some "linear", some "mlp16", some "classifier_fine", some "full_fine", some "adapter"] := by
    simp only [project]
    split_ifs <;> simp
  have hhead := literal_text_option_from_field_membership "adapter head_kind" _ _ hheadMember
  have hb : setOption "adapter batchnorm_buffers" [none, some "source", some "calibration", some "online"]
      (project f.raw).buffers = .ok () :=
    literal_text_option_from_field_membership _ _ f.buffers (by simpa using h.buffers)
  have haf : setOption "adapter batchnorm_affine" [none, some "frozen", some "trainable"]
      (project f.raw).affine = .ok () :=
    literal_text_option_from_field_membership _ _ f.affine (by simpa using h.affine)
  have hw : setOption "adapter trainable_weights"
      [none, some "none", some "head", some "non_bn", some "spatial", some "temporal", some "adapter"]
      (project f.raw).weights = .ok () :=
    literal_text_option_from_field_membership _ _ f.weights (by simpa using h.weights)
  have hd : setOption "adapter dropout_mode" [some "legacy", some "off", some "checkpoint"]
      (project f.raw).dropout = .ok () := by
    apply literal_text_option_from_field_membership
    have hd0 : f.dropout = "off" ∨ f.dropout = "checkpoint" := by simpa using h.dropout
    rcases hd0 with hd0 | hd0 <;> simp [hd0]
  have hv : setOption "adapter validation_behavior"
      [some "select_epoch", some "select_epoch_full_trajectory", some "diagnostics_only"] (.text "select_epoch") = .ok () := by
    simp [setOption, setContains, builtinHashable, require]
  have hi : setOption "adapter input_statistics" [some "auto", some "source", some "target_prefix", some "shrinkage"]
      (project f.raw).inputStatistics = .ok () := by
    apply literal_text_option_from_field_membership
    have hi0 : f.statistics = "source" ∨ f.statistics = "target_prefix" := by simpa using h.statistics
    rcases hi0 with hi0 | hi0 <;> simp [hi0]
  have hnone : (project f.raw).weights ≠ .text "none" ∨
      ((project f.raw).affine ≠ .text "trainable" ∧ (project f.raw).buffers ≠ .text "online") := by
    by_cases hn : f.weights = "none"
    · obtain ⟨haff, hbuf, _⟩ := h.statisticsOnly hn
      simp [project, Arguments.raw, hn, haff, hbuf]
    · simp [project, Arguments.raw, hn]
  have hrole : (project f.raw).weights ≠ .text "adapter" ∨ (project f.raw).head = "adapter" := by
    by_cases hn : f.weights = "adapter" <;> simp [project, Arguments.raw, hn]
  have hlegacy : f.dropout ≠ "legacy" := by
    have hd0 : f.dropout = "off" ∨ f.dropout = "checkpoint" := by simpa using h.dropout
    rcases hd0 with hd0 | hd0 <;> simp [hd0]
  have hp : p.compare .lt f.prior (.integer 0) = .ok false := a.priorNegative
  have hlm : p.mathFinite (if f.lossWeight = .null then .floating (.finite 0) else f.lossWeight) = .ok true := a.lossMath
  have hlb : p.compare .le (.floating (.finite 0))
      (if f.lossWeight = .null then .floating (.finite 0) else f.lossWeight) = .ok true := a.lossBound
  have hnoneOK : require ((project f.raw).weights ≠ .text "none" ∨
      ((project f.raw).affine ≠ .text "trainable" ∧ (project f.raw).buffers ≠ .text "online"))
      "adapter statistics-only" = .ok () := by simp [require, hnone]
  have hroleOK : require ((project f.raw).weights ≠ .text "adapter" ∨ (project f.raw).head = "adapter")
      "adapter head role" = .ok () := by simp [require, hrole]
  have hstructOK : require ((project f.raw).buffers ≠ .null ∧ (project f.raw).affine ≠ .null ∧
      (project f.raw).dropout ≠ .text "legacy") "adapter structural" = .ok () := by
    simp [require, project, Arguments.raw, hlegacy]
  unfold subjectChecks
  rw [hhead, hb, haf, hw, hd, hv, hi, hnoneOK, hroleOK, hstructOK]
  by_cases hm : f.maxUpdates = .null
  · norm_num [project, Arguments.raw, hm, require, atom, rejectComparison, compareAtom,
      a.priorMath, hp, a.epochs, a.batch, a.lr, hlm, hlb, a.temperatureMath,
      a.temperatureBound, h.refit, setContains, builtinHashable]
  · have htag := h.maxTag.resolve_left hm
    have hcmp := a.max hm
    norm_num [project, Arguments.raw, hm, htag.1, hcmp, require, atom, rejectComparison, compareAtom,
      a.priorMath, hp, a.epochs, a.batch, a.lr, hlm, hlb, a.temperatureMath,
      a.temperatureBound, h.refit, setContains, builtinHashable]
    all_goals
      split_ifs with ht
      · rfl
      · exact False.elim (ht htag.2)

def Arguments.recipe (f : Arguments) (adapter : Option FeatureConfig) : Recipe :=
  ⟨{ f.raw with adapter := match adapter with | none => .raw .null | some config => .existing config },
    adapter, f.initialization, f.width,
    (match f.kind, f.rank with | some kind, some rank => some (.text kind, rank) | _, _ => none),
    project f.raw⟩

theorem all_initialization_remainders_from_fields_and_atoms (p : Primitives) (f : Arguments)
    (adapter : Option FeatureConfig) (h : Conditions f) (a : AtomicSuccess p f) :
    remainder p f.raw adapter = .ok (f.recipe adapter) := by
  have ho := all_six_option_sets_from_independent_fields f h
  obtain ⟨stripped, hstrip, hnonempty⟩ := a.name
  have hn : checkName p f.raw.name = .ok () := by
    simp [checkName, Arguments.raw, atom, hstrip, require, hnonempty]
  have hc : classifyInitialization f.raw.initialization = f.initialization := supported_initialization_text_classifies _
  have hs := projected_subject_checks_from_each_atomic_response p f h a
  have hfr := h.frozen
  have hloss := h.noLDALoss
  have hprior := a.ldaPrior
  have hscratch := h.scratch
  have howner := h.owner
  have hstatistics := h.statisticsOnly
  have hwidthInt : 1 ≤ (f.width : ℤ) := by exact_mod_cast h.width
  have htemporal := h.temporal
  simp only [remainder, ho, hn, hc, hs]
  clear ho hn hs hstrip hnonempty stripped a h
  rcases htemporal with ⟨hinit, hwidth, kind, rank, hkind, hrank, hallowed, hpos⟩ | ⟨hnot, hkind, hrank⟩
  · have hkindCheck : setOption "temporal basis kind" [some "fixed_cosine", some "source_supervised"]
        (.text kind) = .ok () := literal_text_option_from_field_membership _ _ _ (by simpa using hallowed)
    have hrankInt : 1 ≤ (rank : ℤ) := by exact_mod_cast hpos
    by_cases hnone : f.weights = "none"
    · obtain ⟨haffine, hbuffers, hdropout⟩ := hstatistics hnone
      simp_all [Arguments.raw, Arguments.recipe, Initialization.isLDA, Initialization.isERP,
        positiveBuiltinInt, require, rejectComparison, compareAtom, atom]; rfl
    · simp_all [Arguments.raw, Initialization.isLDA]
  · by_cases hnone : f.weights = "none"
    · obtain ⟨haffine, hbuffers, hdropout⟩ := hstatistics hnone
      cases hi : f.initialization <;>
        simp_all [Arguments.raw, Arguments.recipe, Initialization.isLDA, Initialization.isERP,
          positiveBuiltinInt, require, rejectComparison, compareAtom, atom] <;> rfl
    · cases hi : f.initialization <;>
        simp_all [Arguments.raw, Arguments.recipe, Initialization.isLDA, Initialization.isERP,
          positiveBuiltinInt, require] <;> rfl

/-- All successful premises are raw structural fields or ONE backend operation.
The feature configuration is built from its individual fields before the
scalar loop. Source/scratch/waveform/source_feature are real continuations. -/
theorem complete_constructor_from_independent_fields (p : Primitives) (f : Arguments)
    (adapter : Option FeatureConfig) (h : Conditions f) (a : AtomicSuccess p f)
    (ha : AdapterFields p f.raw adapter) : validate p f.raw = .ok (f.recipe adapter) := by
  have hp := adapter_preparation_from_independent_fields p f.raw adapter ha
  have hs := scalar_prefix_from_each_atomic_response p f.raw a.toPrefixAtoms
  have hr := all_initialization_remainders_from_fields_and_atoms p f adapter h a
  simp [validate, hp, hs, hr]

theorem complete_json_kwargs_constructor_from_fields (p : Primitives) (f : Arguments)
    (adapter : Option FeatureConfig) (value : Value)
    (h : Conditions f) (a : AtomicSuccess p f) (ha : AdapterFields p f.raw adapter)
    (hj : f.raw.adapter = .raw value) :
    fromKwargs p (allRecordFields f.raw) = .ok (f.recipe adapter) := by
  rw [fromKwargs, json_kwargs_reconstruct_independent_raw_fields f.raw value hj]
  exact complete_constructor_from_independent_fields p f adapter h a ha

theorem generated_feature_kwargs_have_unique_names (r : PersonalFeatureTransferConfig.Raw) :
    (Fields.keys (PersonalFeatureTransferConfig.kwargsFields r)).Nodup := by
  simp [PersonalFeatureTransferConfig.kwargsFields, Fields.ofList, Fields.keys]

theorem generated_recipe_field_image_has_unique_names (r : Raw) :
    (Fields.keys (allRecordFields r)).Nodup := by
  simp [allRecordFields, Fields.ofList, Fields.keys]

end
end N2P3.PersonalTargetRecipeFields
