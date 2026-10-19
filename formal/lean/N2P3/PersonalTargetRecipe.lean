import N2P3.PersonalFeatureTransferConfig

/-!
# Full TargetRecipe initialization branches and early feature-config creation

Source: transfer/recipes.py:17-144 and subject_adapter.py:62-165. The adapter
dict/default constructor occurs before the scalar loop. All eight supported
initializations share that loop, the six option sets, name.strip and their
literal subsequent branch guards. The SubjectAdapterConfig projection is
fieldwise and its variable checks remain explicit. There is no whole config,
recipe, non-ERP continuation or adapter-success oracle.

The extended builtin/NumPy scalar observations retain the operation boundaries
from PersonalRecipePython. Full arbitrary Python classes, NumPy array-valued
comparisons, backend conversion/IEEE implementations and JSON codec refinement
remain named obligations; this is not a complete P01 acceptance assertion.
-/
namespace N2P3.PersonalTargetRecipe
open N2P3.PersonalRecipePython
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev FeatureConfig := PersonalFeatureTransferConfig.Config

inductive AdapterArgument where
  | raw (value : Value)
  /-- A normalized constructor-produced field snapshot, not an arbitrary
  isinstance-compatible object. Subclass overrides, object.__setattr__ and
  mutated/non-normalized typed fields remain separate explicit obligations. -/
  | existing (config : FeatureConfig)

structure Raw where
  name : Value := .null
  initialization : Value := .text "source"
  inputStatistics : Value := .text "target_prefix"
  buffers : Value := .text "online"
  affine : Value := .text "trainable"
  weights : Value := .text "non_bn"
  dropout : Value := .text "checkpoint"
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
  binWidth : Value := .integer 1
  temporalKind : Value := .null
  temporalRank : Value := .null

/-- The asdict image of a typed adapter contains its ten constructor fields,
not FeatureTransferConfig.record's separate schema key. Normalized direct-API
typed snapshots are modeled by AdapterArgument.existing and validate; JSON kwargs
reconstruct their independent dictionary image through the real constructor. -/
def adapterRecordValue : AdapterArgument → Value
  | .raw value => value
  | .existing config => .object (PersonalFeatureTransferConfig.kwargsFields (PersonalFeatureTransferConfig.asRaw config))

def allRecordFields (r : Raw) : Fields := Fields.ofList
  [("name", r.name), ("initialization", r.initialization), ("input_statistics", r.inputStatistics),
    ("bn_buffers", r.buffers), ("bn_affine", r.affine), ("weights", r.weights), ("dropout", r.dropout),
    ("batch_size", r.batchSize), ("lr", r.lr), ("weight_decay", r.weightDecay), ("epochs", r.epochs),
    ("max_updates", r.maxUpdates), ("seed", r.seed), ("refit_full_prefix", r.refit),
    ("target_adapter_config", adapterRecordValue r.adapter), ("candidate_loss_weight", r.lossWeight),
    ("candidate_loss_temperature", r.temperature), ("source_prior_weight", r.prior),
    ("erp_bin_width", r.binWidth), ("temporal_basis_kind", r.temporalKind), ("temporal_rank", r.temporalRank)]

def keywordNames : List String := Fields.keys (allRecordFields {})
def getDefault (fields : Fields) (name : String) (default : Value) : Value := (fields.get name).getD default

def readArgs (fields : Fields) : Result Raw := do
  if ¬fields.keys ⊆ keywordNames then
    .error ⟨"TargetRecipe kwargs", .keywordConstruction, .typeError⟩
  else
    let name ← match fields.get "name" with
      | none => .error ⟨"TargetRecipe required name", .keywordConstruction, .typeError⟩
      | some name => .ok name
    pure {
      name := name
      initialization := getDefault fields "initialization" (.text "source")
      inputStatistics := getDefault fields "input_statistics" (.text "target_prefix")
      buffers := getDefault fields "bn_buffers" (.text "online")
      affine := getDefault fields "bn_affine" (.text "trainable")
      weights := getDefault fields "weights" (.text "non_bn")
      dropout := getDefault fields "dropout" (.text "checkpoint")
      batchSize := getDefault fields "batch_size" (.integer 32)
      lr := getDefault fields "lr" (.floating (.finite (1152921504606847 / 1152921504606846976)))
      weightDecay := getDefault fields "weight_decay" (.floating (.finite (7378697629483821 / 73786976294838206464)))
      epochs := getDefault fields "epochs" (.integer 60)
      maxUpdates := getDefault fields "max_updates" .null
      seed := getDefault fields "seed" (.integer 0)
      refit := getDefault fields "refit_full_prefix" (.boolean false)
      adapter := .raw (getDefault fields "target_adapter_config" .null)
      lossWeight := getDefault fields "candidate_loss_weight" .null
      temperature := getDefault fields "candidate_loss_temperature" (.floating (.finite 1))
      prior := getDefault fields "source_prior_weight" (.floating (.finite 0))
      binWidth := getDefault fields "erp_bin_width" (.integer 1)
      temporalKind := getDefault fields "temporal_basis_kind" .null
      temporalRank := getDefault fields "temporal_rank" .null }

inductive PreparedAdapter where
  | absent
  | config (value : FeatureConfig)
  | other (value : Value)

def convertAdapter (p : Primitives) : AdapterArgument → Result PreparedAdapter
  | .existing config => .ok (.config config)
  | .raw (.object fields) => .config <$> PersonalFeatureTransferConfig.fromKwargs p fields
  | .raw .nonstringKeyDict => .error ⟨"FeatureTransferConfig kwargs", .keywordConstruction, .typeError⟩
  | .raw .null => .ok .absent
  | .raw value => .ok (.other value)

def requireUnindexed (config : FeatureConfig) : Result FeatureConfig := do
  require (config.sourceCount = 0 ∧ config.privateCount = 0) "target residual unindexed FeatureTransferConfig"
  pure config

def prepareAdapter (p : Primitives) (r : Raw) : Result (Option FeatureConfig) := do
  let converted ← convertAdapter p r.adapter
  if r.weights = .text "adapter" then
    match converted with
    | .absent => some <$> (PersonalFeatureTransferConfig.construct p {} >>= requireUnindexed)
    | .config config => some <$> requireUnindexed config
    | .other _ => .error ⟨"target residual unindexed FeatureTransferConfig", .validation, .valueError⟩
  else
    match converted with
    | .absent => .ok none
    | _ => .error ⟨"target_adapter_config requires adapter weights", .validation, .valueError⟩

def scalarField (p : Primitives) (field : String) (skipNone : Bool) (value : Value) : Result Unit := do
  if skipNone && value = .null then pure ()
  else
    require (isBuiltinBool value = false ∧ recipeScalarTag value = true) field
    let finite ← atom field .numpyFinite (p.numpyFinite value)
    require (finite = true) field
    rejectComparison p field .lt value (.integer 0)

def scalarPrefix (p : Primitives) (r : Raw) : Result Unit := do
  scalarField p "candidate_loss_weight" true r.lossWeight
  scalarField p "candidate_loss_temperature" false r.temperature
  scalarField p "source_prior_weight" false r.prior
  rejectComparison p "candidate_loss_temperature positive" .le r.temperature (.integer 0)

def initializationOptions : List (Option String) :=
  [some "source", some "scratch", some "waveform_lda", some "source_feature_lda",
    some "pooled_lda", some "toeplitz_lda", some "temporal_lda", some "xdawn_erp"]

def options (r : Raw) : Result Unit := do
  setOption "recipe initialization" initializationOptions r.initialization
  setOption "recipe input_statistics" [some "source", some "target_prefix"] r.inputStatistics
  setOption "recipe bn_buffers" [some "source", some "calibration", some "online"] r.buffers
  setOption "recipe bn_affine" [some "frozen", some "trainable"] r.affine
  setOption "recipe weights"
    [some "none", some "head", some "non_bn", some "spatial", some "temporal", some "adapter"] r.weights
  setOption "recipe dropout" [some "off", some "checkpoint"] r.dropout

inductive Initialization where
  | source | scratch | waveform | sourceFeature | pooled | toeplitz | temporal | xdawn
  deriving DecidableEq

def Initialization.text : Initialization → String
  | .source => "source"
  | .scratch => "scratch"
  | .waveform => "waveform_lda"
  | .sourceFeature => "source_feature_lda"
  | .pooled => "pooled_lda"
  | .toeplitz => "toeplitz_lda"
  | .temporal => "temporal_lda"
  | .xdawn => "xdawn_erp"

/-- This pure classification is used only after the literal option gate. Its
fallback is unreachable there; it is not an added source error or admission. -/
def classifyInitialization : Value → Initialization
  | .text "scratch" => .scratch
  | .text "waveform_lda" => .waveform
  | .text "source_feature_lda" => .sourceFeature
  | .text "pooled_lda" => .pooled
  | .text "toeplitz_lda" => .toeplitz
  | .text "temporal_lda" => .temporal
  | .text "xdawn_erp" => .xdawn
  | _ => .source

def Initialization.isERP : Initialization → Bool
  | .pooled | .toeplitz | .temporal | .xdawn => true
  | _ => false

def Initialization.isLDA : Initialization → Bool
  | .waveform | .sourceFeature | .pooled | .toeplitz | .temporal | .xdawn => true
  | _ => false

def checkName (p : Primitives) : Value → Result Unit
  | .text text => do
    let stripped ← atom "recipe name.strip" .nameStrip (p.strip text)
    require (stripped ≠ "") "target recipe requires a name"
  | _ => .error ⟨"recipe name.strip", .nameStrip, .attributeError⟩

def positiveBuiltinInt (field : String) : Value → Result ℕ
  | .integer value => do
    require (1 ≤ value) field
    pure value.toNat
  | _ => .error ⟨field, .validation, .valueError⟩

structure Projection where
  head : String
  inputStatistics : Value
  buffers : Value
  affine : Value
  weights : Value
  dropout : Value
  batchSize : Value
  lr : Value
  weightDecay : Value
  epochs : Value
  maxUpdates : Value
  seed : Value
  refit : Value
  prior : Value
  lossWeight : Value
  temperature : Value
  freezeStats : Bool

def project (r : Raw) : Projection := {
  head := if r.weights = .text "adapter" then "adapter"
    else if r.weights ∈ [.text "non_bn", .text "spatial", .text "temporal"] then "full_fine"
    else "classifier_fine"
  inputStatistics := r.inputStatistics, buffers := r.buffers, affine := r.affine,
  weights := r.weights, dropout := r.dropout, batchSize := r.batchSize,
  lr := r.lr, weightDecay := r.weightDecay, epochs := r.epochs,
  maxUpdates := r.maxUpdates, seed := r.seed, refit := r.refit,
  prior := r.prior, lossWeight := if r.lossWeight = .null then .floating (.finite 0) else r.lossWeight,
  temperature := r.temperature, freezeStats := decide (r.buffers ≠ .text "online") }

def subjectChecks (p : Primitives) (a : Projection) : Result Unit := do
  let priorFinite ← atom "adapter source_prior_weight" .mathFinite (p.mathFinite a.prior)
  require (priorFinite = true) "adapter source_prior_weight"
  rejectComparison p "adapter source_prior_weight" .lt a.prior (.integer 0)
  setOption "adapter head_kind" [some "linear", some "mlp16", some "classifier_fine", some "full_fine", some "adapter"]
    (.text a.head)
  rejectComparison p "adapter epochs" .lt a.epochs (.integer 1)
  rejectComparison p "adapter batch_size" .lt a.batchSize (.integer 1)
  rejectComparison p "adapter lr" .le a.lr (.floating (.finite 0))
  setOption "adapter batchnorm_buffers" [none, some "source", some "calibration", some "online"] a.buffers
  setOption "adapter batchnorm_affine" [none, some "frozen", some "trainable"] a.affine
  setOption "adapter trainable_weights"
    [none, some "none", some "head", some "non_bn", some "spatial", some "temporal", some "adapter"] a.weights
  setOption "adapter dropout_mode" [some "legacy", some "off", some "checkpoint"] a.dropout
  setOption "adapter validation_behavior"
    [some "select_epoch", some "select_epoch_full_trajectory", some "diagnostics_only"] (.text "select_epoch")
  if a.maxUpdates ≠ .null then
    rejectComparison p "adapter max_updates positive" .lt a.maxUpdates (.integer 1)
    require (isBuiltinBool a.maxUpdates = false ∧
      (isStrictInt a.maxUpdates = true ∨ (match a.maxUpdates with | .numpyInteger _ => True | _ => False)))
      "adapter max_updates integer"
  require (1 ≤ (128 : ℤ)) "adapter fixed bn_reestimation_batch_size"
  require (a.weights ≠ .text "none" ∨ (a.affine ≠ .text "trainable" ∧ a.buffers ≠ .text "online"))
    "adapter statistics-only"
  let structural ← setContains "adapter structural weights" [some "spatial", some "temporal", some "adapter"] a.weights
  if structural then
    require (a.buffers ≠ .null ∧ a.affine ≠ .null ∧ a.dropout ≠ .text "legacy") "adapter structural"
  require (a.weights ≠ .text "adapter" ∨ a.head = "adapter") "adapter head role"
  -- val_group_fraction=None short circuits its two comparisons.
  require (1 ≤ (2 : ℤ) ∧ (2 : ℤ) ≤ 6) "adapter fixed validation groups"
  require (1 ≤ (6 : ℤ) ∧ (0 : ℝ) ≤ 1 / 1000000) "adapter fixed early stopping"
  require (isBuiltinBool a.refit = true) "adapter refit_full_prefix"
  -- freeze_batchnorm_stats is a Bool produced by the literal projection.
  setOption "adapter input_statistics" [some "auto", some "source", some "target_prefix", some "shrinkage"] a.inputStatistics
  require ((0 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1) "adapter fixed target_stat_weight"
  let lossFinite ← atom "adapter candidate_loss_weight" .mathFinite (p.mathFinite a.lossWeight)
  require (lossFinite = true) "adapter candidate_loss_weight"
  let lossNonnegative ← compareAtom p "adapter candidate_loss_weight" .le (.floating (.finite 0)) a.lossWeight
  require (lossNonnegative = true) "adapter candidate_loss_weight"
  let tempFinite ← atom "adapter candidate_loss_temperature" .mathFinite (p.mathFinite a.temperature)
  require (tempFinite = true) "adapter candidate_loss_temperature"
  let tempPositive ← compareAtom p "adapter candidate_loss_temperature" .lt (.floating (.finite 0)) a.temperature
  require (tempPositive = true) "adapter candidate_loss_temperature"
  require ((0 : ℝ) ≤ 0) "adapter fixed AS-MMD lambda"
  -- lambda=0 short circuits the source's conditional batch_size<2 test.
  require (1 ≤ (1 : ℤ)) "adapter fixed AS-MMD warmup"
  require ((0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 6) "adapter fixed AS-MMD target bounds"

structure Recipe where
  raw : Raw
  adapter : Option FeatureConfig
  initialization : Initialization
  width : ℕ
  temporal : Option (Value × ℕ)
  projection : Projection

def remainder (p : Primitives) (r : Raw) (adapter : Option FeatureConfig) : Result Recipe := do
  options r
  checkName p r.name
  let initialization := classifyInitialization r.initialization
  if initialization = .scratch then
    require (r.inputStatistics = .text "target_prefix") "scratch calibration input statistics"
  if initialization.isLDA then
    require (r.weights = .text "none" ∧ r.affine = .text "frozen" ∧ r.dropout = .text "off") "LDA frozen treatment"
  if initialization.isLDA then
    require (r.lossWeight = .null) "LDA no optimized loss"
    rejectComparison p "LDA no source-prior loss" .ne r.prior (.integer 0)
  let width ← positiveBuiltinInt "erp_bin_width" r.binWidth
  require (width = 1 ∨ initialization.isERP = true) "erp_bin_width belongs only to ERP"
  let temporal ← if initialization = .temporal then do
    setOption "temporal basis kind" [some "fixed_cosine", some "source_supervised"] r.temporalKind
    let rank ← positiveBuiltinInt "temporal_rank" r.temporalRank
    require (width = 1) "temporal bin_width"
    pure (some (r.temporalKind, rank))
  else do
    require (r.temporalKind = .null ∧ r.temporalRank = .null) "non-temporal declaration"
    pure none
  if r.weights = .text "none" then
    require (r.affine = .text "frozen" ∧ r.buffers ≠ .text "online" ∧ r.dropout = .text "off") "statistics-only treatment"
  let projection := project r
  subjectChecks p projection
  let normalized := { r with adapter := match adapter with | none => .raw .null | some config => .existing config }
  pure ⟨normalized, adapter, initialization, width, temporal, projection⟩

def validate (p : Primitives) (r : Raw) : Result Recipe := do
  let adapter ← prepareAdapter p r
  scalarPrefix p r
  remainder p r adapter

def fromKwargs (p : Primitives) (fields : Fields) : Result Recipe := do
  let raw ← readArgs fields
  validate p raw

theorem json_kwargs_reconstruct_independent_raw_fields (r : Raw) (value : Value)
    (h : r.adapter = .raw value) : readArgs (allRecordFields r) = .ok r := by
  cases r
  simp_all [readArgs, allRecordFields, adapterRecordValue, keywordNames,
    Fields.ofList, Fields.keys, Fields.get, getDefault]

theorem supported_initialization_text_classifies (initialization : Initialization) :
    classifyInitialization (.text initialization.text) = initialization := by
  cases initialization <;> simp [classifyInitialization, Initialization.text]

/-- The literal set gate itself derives the classification domain for arbitrary
extended values. No legal-family assumption bypasses container/hash errors. -/
theorem initialization_gate_derives_one_of_eight_texts (value : Value)
    (h : setOption "recipe initialization" initializationOptions value = .ok ()) :
    ∃ initialization : Initialization, value = .text initialization.text := by
  cases value with
  | text text =>
    have hm : some text ∈ initializationOptions := by
      by_contra hn
      simp [setOption, setContains, builtinHashable, require, hn] at h
    simp only [initializationOptions, List.mem_cons, List.not_mem_nil, or_false, Option.some.injEq] at hm
    rcases hm with h | h | h | h | h | h | h | h
    · subst text; exact ⟨.source, rfl⟩
    · subst text; exact ⟨.scratch, rfl⟩
    · subst text; exact ⟨.waveform, rfl⟩
    · subst text; exact ⟨.sourceFeature, rfl⟩
    · subst text; exact ⟨.pooled, rfl⟩
    · subst text; exact ⟨.toeplitz, rfl⟩
    · subst text; exact ⟨.temporal, rfl⟩
    · subst text; exact ⟨.xdawn, rfl⟩
  | tuple values =>
    cases hh : allHashable values <;> simp [setOption, setContains, builtinHashable, hh, require] at h
  | null => simp [setOption, setContains, builtinHashable, initializationOptions, require] at h
  | boolean _ => simp [setOption, setContains, builtinHashable, require] at h
  | integer _ => simp [setOption, setContains, builtinHashable, require] at h
  | floating _ => simp [setOption, setContains, builtinHashable, require] at h
  | numpyInteger _ => simp [setOption, setContains, builtinHashable, require] at h
  | numpyFloat64 _ => simp [setOption, setContains, builtinHashable, require] at h
  | numpyFloat32 _ => simp [setOption, setContains, builtinHashable, require] at h
  | numpyBoolean _ => simp [setOption, setContains, builtinHashable, require] at h
  | list _ => simp [setOption, setContains, builtinHashable] at h
  | object _ => simp [setOption, setContains, builtinHashable] at h
  | nonstringKeyDict => simp [setOption, setContains, builtinHashable] at h

theorem unknown_recipe_keyword_precedes_adapter_creation (p : Primitives) (fields : Fields)
    (h : ¬fields.keys ⊆ keywordNames) :
    fromKwargs p fields = .error ⟨"TargetRecipe kwargs", .keywordConstruction, .typeError⟩ := by
  simp [fromKwargs, readArgs, h]

theorem bad_config_keyword_precedes_every_scalar (p : Primitives) (r : Raw) (fields : Fields)
    (ha : r.adapter = .raw (.object fields))
    (hk : ¬fields.keys ⊆ PersonalFeatureTransferConfig.names) :
    validate p r = .error ⟨"FeatureTransferConfig kwargs", .keywordConstruction, .typeError⟩ := by
  simp [validate, prepareAdapter, convertAdapter, ha,
    PersonalFeatureTransferConfig.unknown_config_keyword_precedes_every_numeric_operation p fields hk]

theorem indexed_existing_config_precedes_every_scalar (p : Primitives) (r : Raw) (config : FeatureConfig)
    (ha : r.adapter = .existing config) (hw : r.weights = .text "adapter")
    (hc : config.sourceCount ≠ 0 ∨ config.privateCount ≠ 0) :
    validate p r = .error ⟨"target residual unindexed FeatureTransferConfig", .validation, .valueError⟩ := by
  have hn : ¬(config.sourceCount = 0 ∧ config.privateCount = 0) := by tauto
  simp [validate, prepareAdapter, convertAdapter, ha, hw, requireUnindexed, require, hn]

theorem supplied_config_without_adapter_weights_precedes_every_scalar (p : Primitives) (r : Raw) (config : FeatureConfig)
    (ha : r.adapter = .existing config) (hw : r.weights ≠ .text "adapter") :
    validate p r = .error ⟨"target_adapter_config requires adapter weights", .validation, .valueError⟩ := by
  simp [validate, prepareAdapter, convertAdapter, ha, hw]

theorem numpy_temperature_exception_after_no_adapter (p : Primitives) (r : Raw) (exception : Exception)
    (ha : r.adapter = .raw .null) (hw : r.weights ≠ .text "adapter")
    (hl : r.lossWeight = .null) (ht : isBuiltinBool r.temperature = false ∧ recipeScalarTag r.temperature = true)
    (he : p.numpyFinite r.temperature = .error exception) :
    validate p r = .error ⟨"candidate_loss_temperature", .numpyFinite, exception⟩ := by
  simp [validate, prepareAdapter, convertAdapter, ha, hw, scalarPrefix, scalarField, hl,
    require, ht, atom, he]

theorem nonstring_name_raises_its_attribute_error_before_later_guards (p : Primitives) (number : ℤ) :
    checkName p (.integer number) = .error ⟨"recipe name.strip", .nameStrip, .attributeError⟩ := rfl

theorem numpy_integer_width_is_rejected (value : ℤ) :
    positiveBuiltinInt "erp_bin_width" (.numpyInteger value) =
      .error ⟨"erp_bin_width", .validation, .valueError⟩ := rfl

theorem bool_width_is_rejected (value : Bool) :
    positiveBuiltinInt "erp_bin_width" (.boolean value) =
      .error ⟨"erp_bin_width", .validation, .valueError⟩ := rfl

end
end N2P3.PersonalTargetRecipe
