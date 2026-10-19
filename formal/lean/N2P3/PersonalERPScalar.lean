import N2P3.PersonalERPRecipe

/-!
# Scalar operation boundaries and the actual ERP constructor order

Sources: recipes.py:39-106, subject_adapter.py:98-165, calibrated.py:546-659.
This extends, and does not rewrite, the frozen decoded-numeric component.
`Value.integer` keeps arbitrary Python integers; `Value.number` is a decoded
finite scalar, not an IEEE implementation. JSON parsing/digit limits, floats,
NumPy's universal representability range, tensor codecs and fitting are pending.

Only ONE scalar finite test or ONE named literal-bound comparison may be a
primitive. No primitive answers whole recipe, admission or prediction validity.
Atomic exceptions propagate with their operation and field. Unsupported atomic
cases are explicit `unrefined`, never silently accepted or converted to false.
Builtin JSON set membership first checks hashability: lists/dicts raise the
operation's TypeError, while a hashable absent enum yields source ValueError.
The finite table at the end is a logical witness for independently fixed probe
inputs, not a general NumPy specification. Direct API NaN/inf is outside Value.

ERP projection requires no residual adapter. Dict/adapter configurations are
explicitly unrefined before the scalar loop because FeatureTransferConfig has
its own constructor. Existing decoded structural readers retain their stated
boundaries and their LedgerFault, rather than inventing Python exception types.
All other recipe and adapter checks below are concrete fieldwise definitions.
The existing theorem `adapter_unchanged_default_checks` discharges their fixed
default numerical checks. Full P01 remains pending at the named boundaries.
-/

namespace N2P3.PersonalERPScalar
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe

noncomputable section
set_option exponentiation.threshold 1024
set_option maxRecDepth 4096
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive AtomicException where
  | typeError | overflowError
  | other (name : String)
  | unrefined (boundary : String)
  deriving DecidableEq

inductive Comparison where
  | ltZero | leZero | ltOne | zeroLe | zeroLt | neZero
  deriving DecidableEq

inductive Operation where
  | numpyFinite | mathFinite | compare (comparison : Comparison) | setMembership
  deriving DecidableEq

inductive Fault where
  | sourceValueError (field : String)
  | atomic (field : String) (operation : Operation) (exception : AtomicException)
  | inherited (fault : LedgerFault)
  | unrefined (boundary : String)
  deriving DecidableEq

abbrev Result (α : Type) := Except Fault α

structure Primitives where
  numpyIsFinite : Value → Except AtomicException Bool
  mathIsFinite : Value → Except AtomicException Bool
  compare : Comparison → Value → Except AtomicException Bool

structure Call where
  field : String
  operation : Operation
  deriving DecidableEq

structure Observation (α : Type) where
  calls : List Call
  result : Result α

def done (value : α) : Observation α := ⟨[], .ok value⟩
def stopped (fault : Fault) : Observation α := ⟨[], .error fault⟩

/-- The trace retains operations even on an exception and stops afterwards. -/
def andThen (first : Observation α) (next : α → Observation β) : Observation β :=
  match first.result with
  | .error fault => ⟨first.calls, .error fault⟩
  | .ok value => let second := next value
    ⟨first.calls ++ second.calls, second.result⟩

def atomicCall (field : String) (operation : Operation)
    (answer : Except AtomicException Bool) : Observation Bool :=
  ⟨[⟨field, operation⟩], answer.mapError (Fault.atomic field operation)⟩

def numpyCall (p : Primitives) (field : String) (value : Value) : Observation Bool :=
  atomicCall field .numpyFinite (p.numpyIsFinite value)

def compareCall (p : Primitives) (field : String) (op : Comparison)
    (value : Value) : Observation Bool :=
  atomicCall field (.compare op) (p.compare op value)

def numericNonBool : Value → Bool
  | .integer _ | .number _ => true
  | _ => false

def scalarField (p : Primitives) (field : String) (skipNone : Bool)
    (value : Value) : Observation Unit :=
  if skipNone && value = .null then done ()
  else if numericNonBool value then
    andThen (numpyCall p field value) fun finite =>
      if finite then
        andThen (compareCall p field .ltZero value) fun negative =>
          if negative then stopped (.sourceValueError field) else done ()
      else stopped (.sourceValueError field)
  else stopped (.sourceValueError field)

/-- The WHOLE loop precedes the strict-positive temperature test. -/
def scalarPrefix (p : Primitives) (r : RawRecipe) : Observation Unit :=
  andThen (scalarField p "candidate_loss_weight" true r.candidateLossWeight) fun _ =>
  andThen (scalarField p "candidate_loss_temperature" false r.candidateLossTemperature) fun _ =>
  andThen (scalarField p "source_prior_weight" false r.sourcePriorWeight) fun _ =>
  andThen (compareCall p "candidate_loss_temperature positive" .leZero r.candidateLossTemperature) fun bad =>
    if bad then stopped (.sourceValueError "candidate_loss_temperature positive") else done ()

def liftCheck (check : Check α) : Result α := check.mapError Fault.inherited

def require (condition : Prop) (field : String) : Result Unit :=
  if condition then .ok () else .error (.sourceValueError field)

/-- Exactly builtin JSON tags, not general Python user-defined hashability. -/
def builtinHashable : Value → Bool
  | .sequence _ | .object _ => false
  | _ => true

/-- A single actual set-membership operation. Scalar equality retains Python
finite numeric equality, although the concrete source option sets use text/None.
Lists and dicts fail before equality/membership, including empty containers. -/
def setContains (field : String) (allowed : List Value) (value : Value) : Result Bool :=
  if builtinHashable value then
    .ok (allowed.any (fun expected => decide (PersonalAdmission.PyEq value expected)))
  else .error (.atomic field .setMembership .typeError)

def setOption (field : String) (allowed : List Value) (value : Value) : Result Unit := do
  let member ← setContains field allowed value
  require (member = true) field

def recipeOptions (r : RawRecipe) : Result Unit := do
  setOption "recipe initialization"
    [.text "source", .text "scratch", .text "waveform_lda", .text "source_feature_lda",
      .text "pooled_lda", .text "toeplitz_lda", .text "temporal_lda", .text "xdawn_erp"] r.initialization
  setOption "recipe input_statistics" [.text "source", .text "target_prefix"] r.inputStatistics
  setOption "recipe bn_buffers" [.text "source", .text "calibration", .text "online"] r.bnBuffers
  setOption "recipe bn_affine" [.text "frozen", .text "trainable"] r.bnAffine
  setOption "recipe weights"
    [.text "none", .text "head", .text "non_bn", .text "spatial", .text "temporal", .text "adapter"] r.weights
  setOption "recipe dropout" [.text "off", .text "checkpoint"] r.dropout

/-- Valid non-ERP initializations have already passed the real option loop;
their later treatment branches remain explicit unrefined continuations. -/
def erpContinuationFamily (value : Value) : Result Family :=
  match readFamily value with
  | .ok family => .ok family
  | .error _ => .error (.unrefined "non-ERP recipe continuation")

def compareResult (p : Primitives) (field : String) (op : Comparison)
    (value : Value) : Result Bool := (compareCall p field op value).result

def finiteResult (p : Primitives) (field : String) (value : Value) : Result Bool :=
  (p.mathIsFinite value).mapError (Fault.atomic field .mathFinite)

def rejectWhen (p : Primitives) (field : String) (op : Comparison)
    (value : Value) : Result Unit := do
  let bad ← compareResult p field op value
  require (bad = false) field

def mathBound (p : Primitives) (field : String) (op : Comparison)
    (value : Value) : Result Unit := do
  let finite ← finiteResult p field value
  require (finite = true) field
  let valid ← compareResult p field op value
  require (valid = true) field

/-- Source OR order: epochs<1, batch_size<1, then lr<=0. No finite/type guard
is invented, so boolTrue and arbitrary positive integers may pass comparisons. -/
def optimization (p : Primitives) (r : RawRecipe) : Result Unit := do
  rejectWhen p "adapter epochs" .ltOne r.epochs
  rejectWhen p "adapter batch_size" .ltOne r.batchSize
  rejectWhen p "adapter lr" .leZero r.lr

def adapterPrecheck (r : RawRecipe) : Result Unit :=
  match r.targetAdapterConfig with
  | .object _ => .error (.unrefined "FeatureTransferConfig kwargs constructor")
  | _ =>
    if r.weights = .text "adapter" then .error (.unrefined "FeatureTransferConfig adapter branch")
    else require (r.targetAdapterConfig = .null) "target_adapter_config requires adapter weights"

/-- Literal variable checks of SubjectAdapterConfig on the actual projection.
The fixed default checks are explicit and independently proved in the imported
adapter_unchanged_default_checks theorem; they are never backend callbacks. -/
def adapterChecks (p : Primitives) (r : RawRecipe) : Result Unit := do
  let a := adapterProjection r
  mathBound p "adapter source_prior_weight" .zeroLe a.sourcePriorWeight
  setOption "adapter head_kind" [.text "linear", .text "mlp16", .text "classifier_fine", .text "full_fine", .text "adapter"]
    (.text a.headKind)
  optimization p r
  setOption "adapter batchnorm_buffers" [.null, .text "source", .text "calibration", .text "online"] a.batchnormBuffers
  setOption "adapter batchnorm_affine" [.null, .text "frozen", .text "trainable"] a.batchnormAffine
  setOption "adapter trainable_weights"
    [.null, .text "none", .text "head", .text "non_bn", .text "spatial", .text "temporal", .text "adapter"] a.trainableWeights
  setOption "adapter dropout_mode" [.text "legacy", .text "off", .text "checkpoint"] a.dropoutMode
  setOption "adapter validation_behavior"
    [.text "select_epoch", .text "select_epoch_full_trajectory", .text "diagnostics_only"] (.text a.validationBehavior)
  if a.maxUpdates ≠ .null then
    rejectWhen p "adapter max_updates positive" .ltOne a.maxUpdates
    require (ExactInt a.maxUpdates) "adapter max_updates integer"
  require (1 ≤ a.bnReestimationBatchSize) "adapter bn_reestimation_batch_size"
  require (a.trainableWeights ≠ .text "none" ∨
    (a.batchnormAffine ≠ .text "trainable" ∧ a.batchnormBuffers ≠ .text "online")) "adapter statistics-only"
  require (a.trainableWeights ∉ [.text "spatial", .text "temporal", .text "adapter"] ∨
    (a.batchnormBuffers ≠ .null ∧ a.batchnormAffine ≠ .null ∧ a.dropoutMode ≠ .text "legacy")) "adapter structural"
  require (a.trainableWeights ≠ .text "adapter" ∨ a.headKind = "adapter") "adapter head role"
  require (a.valGroupFraction = .null) "adapter fixed no validation fraction"
  require (1 ≤ a.valGroupsMin ∧ a.valGroupsMin ≤ a.valGroupsMax) "adapter validation groups"
  require (1 ≤ a.earlyStopPatience ∧ 0 ≤ a.earlyStopMinDelta) "adapter early stopping"
  require (ExactBool a.refitFullPrefix) "adapter refit_full_prefix"
  setOption "adapter input_statistics"
    [.text "auto", .text "source", .text "target_prefix", .text "shrinkage"] a.inputStatistics
  require (0 ≤ a.targetStatWeight ∧ a.targetStatWeight ≤ 1) "adapter target_stat_weight"
  mathBound p "adapter candidate_loss_weight" .zeroLe a.candidateLossWeight
  mathBound p "adapter candidate_loss_temperature" .zeroLt a.candidateLossTemperature
  require (0 ≤ a.asMMDLambda ∧ a.asMMDLambda = 0) "adapter fixed AS-MMD lambda"
  require (1 ≤ a.asMMDWarmup) "adapter AS-MMD warmup"
  require (0 < a.asMMDMin ∧ a.asMMDMin ≤ a.asMMDMax) "adapter AS-MMD bounds"

/-- ERP restriction after the scalar prefix. Non-ERP and residual-adapter
families are not claimed covered. Numeric prior!=0 is one atomic comparison. -/
def recipeRemainder (p : Primitives) (r : RawRecipe) : Result Recipe := do
  recipeOptions r
  let name ← liftCheck (readTextValue r.name "recipe name.strip")
  require (Nonblank name) "recipe name"
  let family ← erpContinuationFamily r.initialization
  require (r.weights = .text "none" ∧ r.bnAffine = .text "frozen" ∧ r.dropout = .text "off") "ERP frozen treatment"
  require (r.candidateLossWeight = .null) "ERP no optimized loss"
  rejectWhen p "ERP no source-prior loss" .neZero r.sourcePriorWeight
  let width ← liftCheck (positiveInt r.erpBinWidth "erp_bin_width")
  let (kind, rank) ← if family = .temporal then do
    setOption "temporal declaration" [.text "fixed_cosine", .text "source_supervised"] r.temporalBasisKind
    let rank ← liftCheck (positiveInt r.temporalRank "temporal_rank")
    require (width = 1) "temporal width"
    let kind ← liftCheck (readTextValue r.temporalBasisKind "temporal kind")
    pure (some kind, some rank)
  else do
    require (r.temporalBasisKind = .null ∧ r.temporalRank = .null) "non-temporal declaration"
    pure (none, none)
  require (r.bnBuffers ≠ .text "online") "statistics-only buffers"
  adapterChecks p r
  pure ⟨r, family, width, kind, rank, adapterProjection r⟩

def validateScalar (p : Primitives) (r : RawRecipe) : Result Recipe := do
  adapterPrecheck r
  (scalarPrefix p r).result
  recipeRemainder p r

def fromKwargsScalar (p : Primitives) (kwargs : Fields) : Result Recipe := do
  let r ← liftCheck (readArgs kwargs)
  validateScalar p r

/-- Actual placement: selectors and inner reader BEFORE recipe kwargs; bind
physical shape/role only AFTER the concrete constructor has succeeded. -/
def readBranchScalar (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (physical : Physical) (kind : String) (sourceFlag : Value)
    (kwargs record : Fields) (state : NamedState) : Result Bound := do
  liftCheck (ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind")
  let tensor ← liftCheck (tensorAt state "waveform_channels")
  let selected ← liftCheck (readSelected physical.channels tensor)
  let model ← liftCheck (CurrentERPState.fromState p ref record (erpFields state))
  let recipe ← fromKwargsScalar s kwargs
  liftCheck (bindERP recipe kind sourceFlag physical selected model)

theorem loss_none_skips_every_operation (p : Primitives) :
    scalarField p "candidate_loss_weight" true .null = done () := by simp [scalarField]

theorem bool_rejected_before_numpy (p : Primitives) (field : String) (skip : Bool) (b : Bool) :
    scalarField p field skip (.boolean b) = stopped (.sourceValueError field) := by
  simp [scalarField, numericNonBool]

theorem text_rejected_before_numpy (p : Primitives) (field text : String) (skip : Bool) :
    scalarField p field skip (.text text) = stopped (.sourceValueError field) := by
  simp [scalarField, numericNonBool]

theorem temperature_none_is_not_skipped (p : Primitives) :
    scalarField p "candidate_loss_temperature" false .null =
      stopped (.sourceValueError "candidate_loss_temperature") := by
  simp [scalarField, numericNonBool]

theorem numpy_exception_preserved (p : Primitives) (field : String) (skip : Bool)
    (value : Value) (exception : AtomicException) (hn : numericNonBool value = true)
    (h : p.numpyIsFinite value = .error exception) :
    scalarField p field skip value =
      ⟨[⟨field, .numpyFinite⟩], .error (.atomic field .numpyFinite exception)⟩ := by
  have hv : value ≠ .null := by intro hv; simp [hv, numericNonBool] at hn
  simp [scalarField, hv, hn, numpyCall, atomicCall, andThen, h]

theorem finite_false_stops_before_comparison (p : Primitives) (field : String) (skip : Bool)
    (value : Value) (hn : numericNonBool value = true) (h : p.numpyIsFinite value = .ok false) :
    scalarField p field skip value =
      ⟨[⟨field, .numpyFinite⟩], .error (.sourceValueError field)⟩ := by
  have hv : value ≠ .null := by intro hv; simp [hv, numericNonBool] at hn
  simp [scalarField, hv, hn, numpyCall, atomicCall, andThen, stopped, h]

theorem comparison_exception_preserved (p : Primitives) (field : String) (skip : Bool)
    (value : Value) (exception : AtomicException) (hn : numericNonBool value = true)
    (hf : p.numpyIsFinite value = .ok true) (hc : p.compare .ltZero value = .error exception) :
    scalarField p field skip value =
      ⟨[⟨field, .numpyFinite⟩, ⟨field, .compare .ltZero⟩],
        .error (.atomic field (.compare .ltZero) exception)⟩ := by
  have hv : value ≠ .null := by intro hv; simp [hv, numericNonBool] at hn
  simp [scalarField, hv, hn, numpyCall, compareCall, atomicCall, andThen, hf, hc]

theorem negative_rejected_after_exact_two_operations (p : Primitives) (field : String) (skip : Bool)
    (value : Value) (hn : numericNonBool value = true)
    (hf : p.numpyIsFinite value = .ok true) (hc : p.compare .ltZero value = .ok true) :
    scalarField p field skip value =
      ⟨[⟨field, .numpyFinite⟩, ⟨field, .compare .ltZero⟩], .error (.sourceValueError field)⟩ := by
  have hv : value ≠ .null := by intro hv; simp [hv, numericNonBool] at hn
  simp [scalarField, hv, hn, numpyCall, compareCall, atomicCall, andThen, stopped, hf, hc]

theorem scalar_field_succeeds (p : Primitives) (field : String) (skip : Bool)
    (value : Value) (hn : numericNonBool value = true)
    (hf : p.numpyIsFinite value = .ok true) (hc : p.compare .ltZero value = .ok false) :
    scalarField p field skip value =
      ⟨[⟨field, .numpyFinite⟩, ⟨field, .compare .ltZero⟩], .ok ()⟩ := by
  have hv : value ≠ .null := by intro hv; simp [hv, numericNonBool] at hn
  simp [scalarField, hv, hn, numpyCall, compareCall, atomicCall, andThen, done, hf, hc]

theorem first_field_failure_stops_loop (p : Primitives) (r : RawRecipe)
    (calls : List Call) (fault : Fault)
    (h : scalarField p "candidate_loss_weight" true r.candidateLossWeight = ⟨calls, .error fault⟩) :
    scalarPrefix p r = ⟨calls, .error fault⟩ := by simp [scalarPrefix, h, andThen]

theorem prior_failure_precedes_positive_temperature (p : Primitives) (r : RawRecipe)
    (lossCalls tempCalls priorCalls : List Call) (fault : Fault)
    (hl : scalarField p "candidate_loss_weight" true r.candidateLossWeight = ⟨lossCalls, .ok ()⟩)
    (ht : scalarField p "candidate_loss_temperature" false r.candidateLossTemperature = ⟨tempCalls, .ok ()⟩)
    (hp : scalarField p "source_prior_weight" false r.sourcePriorWeight = ⟨priorCalls, .error fault⟩) :
    scalarPrefix p r = ⟨lossCalls ++ (tempCalls ++ priorCalls), .error fault⟩ := by
  simp [scalarPrefix, hl, ht, hp, andThen]

theorem earlier_inner_failure_precedes_scalar (s : Primitives) (p : TemporalPrimitives)
    (ref : ReferencePrimitives) (physical : Physical) (kind : String) (flag : Value)
    (kwargs record : Fields) (state : NamedState) (tensor : Tensor) (selected : List ℕ)
    (fault : LedgerFault)
    (hk : kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"])
    (ht : tensorAt state "waveform_channels" = .ok tensor)
    (hs : readSelected physical.channels tensor = .ok selected)
    (hi : CurrentERPState.fromState p ref record (erpFields state) = .error fault) :
    readBranchScalar s p ref physical kind flag kwargs record state = .error (.inherited fault) := by
  simp [readBranchScalar, hk, ensure, ht, hs, hi, liftCheck]

theorem kwargs_reconstruct_before_scalar (p : Primitives) (r : RawRecipe) :
    fromKwargsScalar p (Fields.ofList (rawFields r)) = validateScalar p r := by
  simp [fromKwargsScalar, explicit_kwargs_reconstruct_all_fields, liftCheck]

theorem sequence_option_raises_before_membership (field : String) (allowed : List Value) (xs : Values) :
    setOption field allowed (.sequence xs) = .error (.atomic field .setMembership .typeError) := by
  simp [setOption, setContains, builtinHashable]

theorem object_option_raises_before_membership (field : String) (allowed : List Value) (fs : Fields) :
    setOption field allowed (.object fs) = .error (.atomic field .setMembership .typeError) := by
  simp [setOption, setContains, builtinHashable]

def huge100 : Value := .integer (10 ^ 100)
def huge400 : Value := .integer (10 ^ 400)

/-- Only independently fixed small finite inputs and two huge integer inputs
are assigned finite-test outcomes. Every other input remains unrefined. -/
def observedSmallFinite (v : Value) : Except AtomicException Bool :=
  match v with
  | .integer z => if z ∈ [-1, 0, 1, 64] then .ok true else .error (.unrefined "unobserved finite input")
  | .number x => if x ∈ [0, 1, 1 / 1000] then .ok true else .error (.unrefined "unobserved finite input")
  | .boolean _ => .ok true
  | _ => .error (.unrefined "unobserved finite input")

/-- Comparison with exact literal bounds on the declared finite decoded
numeric domain. This is not an IEEE/NaN comparison theorem. -/
def decodedCompare (op : Comparison) (v : Value) : Except AtomicException Bool :=
  match numericScalar v with
  | none => .error .typeError
  | some x => .ok (decide (match op with
      | .ltZero => x < 0 | .leZero => x ≤ 0 | .ltOne => x < 1
      | .zeroLe => 0 ≤ x | .zeroLt => 0 < x | .neZero => x ≠ 0))

def observedPrimitives : Primitives := {
  numpyIsFinite := fun v => match v with
    | .integer z => if z = 10 ^ 100 ∨ z = 10 ^ 400 then .error .typeError else observedSmallFinite v
    | _ => observedSmallFinite v
  mathIsFinite := fun v => match v with
    | .integer z => if z = 10 ^ 100 then .ok true
      else if z = 10 ^ 400 then .error .overflowError else observedSmallFinite v
    | _ => observedSmallFinite v
  compare := decodedCompare }

theorem observed_numpy_huge_typeerror : observedPrimitives.numpyIsFinite huge100 = .error .typeError := by
  change (if (10 : ℤ) ^ 100 = 10 ^ 100 ∨ (10 : ℤ) ^ 100 = 10 ^ 400 then
    Except.error AtomicException.typeError else observedSmallFinite huge100) = _
  rw [if_pos (Or.inl rfl)]

theorem observed_math_huge_finite : observedPrimitives.mathIsFinite huge100 = .ok true := by
  change (if (10 : ℤ) ^ 100 = 10 ^ 100 then Except.ok true else
    if (10 : ℤ) ^ 100 = 10 ^ 400 then Except.error AtomicException.overflowError else observedSmallFinite huge100) = _
  rw [if_pos rfl]

theorem observed_math_larger_overflow : observedPrimitives.mathIsFinite huge400 = .error .overflowError := by
  norm_num [observedPrimitives, huge100, huge400]

theorem observed_bool_finite_does_not_bypass_tag_guard :
    observedPrimitives.numpyIsFinite (.boolean true) = .ok true ∧
    scalarField observedPrimitives "candidate_loss_temperature" false (.boolean true) =
      stopped (.sourceValueError "candidate_loss_temperature") := by
  constructor
  · rfl
  · exact bool_rejected_before_numpy _ _ _ _

theorem huge_integer_not_treated_as_positive_real :
    scalarField observedPrimitives "candidate_loss_temperature" false huge100 =
      ⟨[⟨"candidate_loss_temperature", .numpyFinite⟩],
        .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError)⟩ := by
  exact numpy_exception_preserved _ _ _ _ _ rfl observed_numpy_huge_typeerror

theorem math_overflow_keeps_its_class :
    mathBound observedPrimitives "adapter source_prior_weight" .zeroLe huge400 =
      .error (.atomic "adapter source_prior_weight" .mathFinite .overflowError) := by
  simp [mathBound, finiteResult, observed_math_larger_overflow]

theorem zero_temperature_checks_prior_first :
    scalarPrefix observedPrimitives
      { (pooledFields 2).raw with candidateLossTemperature := .integer 0, sourcePriorWeight := huge100 } =
      ⟨[⟨"candidate_loss_temperature", .numpyFinite⟩,
        ⟨"candidate_loss_temperature", .compare .ltZero⟩, ⟨"source_prior_weight", .numpyFinite⟩],
        .error (.atomic "source_prior_weight" .numpyFinite .typeError)⟩ := by
  let r := { (pooledFields 2).raw with candidateLossTemperature := Value.integer 0, sourcePriorWeight := huge100 }
  have hl : scalarField observedPrimitives "candidate_loss_weight" true r.candidateLossWeight = ⟨[], .ok ()⟩ := by
    exact loss_none_skips_every_operation _
  have ht := scalar_field_succeeds observedPrimitives "candidate_loss_temperature" false (.integer 0) rfl
    (by norm_num [observedPrimitives, observedSmallFinite])
    (by norm_num [observedPrimitives, decodedCompare, numericScalar])
  have hp := numpy_exception_preserved observedPrimitives "source_prior_weight" false huge100
    .typeError rfl observed_numpy_huge_typeerror
  exact prior_failure_precedes_positive_temperature observedPrimitives r [] _ _ _ hl ht hp

theorem negative_loss_stops_before_huge_temperature :
    (scalarPrefix observedPrimitives
      { (pooledFields 2).raw with candidateLossWeight := .integer (-1), candidateLossTemperature := huge100 }).result =
      .error (.sourceValueError "candidate_loss_weight") := by
  have hl := negative_rejected_after_exact_two_operations observedPrimitives "candidate_loss_weight" true
    (.integer (-1)) rfl (by norm_num [observedPrimitives, observedSmallFinite])
    (by norm_num [observedPrimitives, decodedCompare, numericScalar])
  have hp := first_field_failure_stops_loop observedPrimitives
    { (pooledFields 2).raw with candidateLossWeight := .integer (-1), candidateLossTemperature := huge100 } _ _ hl
  exact congrArg Observation.result hp

def hugeTemperatureRaw : RawRecipe :=
  { (pooledFields 2).raw with candidateLossTemperature := huge100 }

theorem huge_temperature_concrete_constructor :
    validateScalar observedPrimitives hugeTemperatureRaw =
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  have hprefix : (scalarPrefix observedPrimitives hugeTemperatureRaw).result =
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
    simp [scalarPrefix, hugeTemperatureRaw, pooledFields, ERPFields.raw,
      loss_none_skips_every_operation, huge_integer_not_treated_as_positive_real, andThen, done]
  have hpre : adapterPrecheck hugeTemperatureRaw = .ok () := by
    simp [adapterPrecheck, hugeTemperatureRaw, pooledFields, ERPFields.raw, require]
  simp [validateScalar, hpre, hprefix]

theorem positive_bool_optimization_is_not_strict_integer_validation :
    optimization observedPrimitives
      { (pooledFields 2).raw with epochs := .boolean true, batchSize := .boolean true, lr := .boolean true } = .ok () := by
  norm_num [optimization, rejectWhen, compareResult, compareCall, atomicCall,
    observedPrimitives, decodedCompare, numericScalar, require]

theorem huge_positive_optimization_uses_no_finite_operation :
    optimization observedPrimitives
      { (pooledFields 2).raw with epochs := huge100, batchSize := huge100, lr := huge100 } = .ok () := by
  norm_num [optimization, rejectWhen, compareResult, compareCall, atomicCall,
    observedPrimitives, decodedCompare, huge100, numericScalar, require]

theorem baseline_recipe_prefix : (scalarPrefix observedPrimitives (pooledFields 2).raw).result = .ok () := by
  norm_num [scalarPrefix, scalarField, pooledFields, ERPFields.raw, andThen,
    numpyCall, compareCall, atomicCall, done, numericNonBool,
    observedPrimitives, observedSmallFinite, huge100, huge400, decodedCompare, numericScalar]

theorem list_recipe_option_preserves_typeerror :
    validateScalar observedPrimitives { (pooledFields 2).raw with inputStatistics := .sequence .nil } =
      .error (.atomic "recipe input_statistics" .setMembership .typeError) := by
  have hp : (scalarPrefix observedPrimitives
      { (pooledFields 2).raw with inputStatistics := .sequence .nil }).result = .ok () := baseline_recipe_prefix
  simp only [validateScalar, hp]
  simp [adapterPrecheck, recipeRemainder, recipeOptions,
    pooledFields, ERPFields.raw, Family.text, setOption, setContains, builtinHashable,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, require]

theorem dict_recipe_option_preserves_typeerror :
    validateScalar observedPrimitives { (pooledFields 2).raw with inputStatistics := .object .nil } =
      .error (.atomic "recipe input_statistics" .setMembership .typeError) := by
  have hp : (scalarPrefix observedPrimitives
      { (pooledFields 2).raw with inputStatistics := .object .nil }).result = .ok () := baseline_recipe_prefix
  simp only [validateScalar, hp]
  simp [adapterPrecheck, recipeRemainder, recipeOptions,
    pooledFields, ERPFields.raw, Family.text, setOption, setContains, builtinHashable,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, require]

theorem hashable_wrong_string_option_is_valueerror :
    validateScalar observedPrimitives
      { (pooledFields 2).raw with inputStatistics := .text "unsupported_probe_value" } =
      .error (.sourceValueError "recipe input_statistics") := by
  have hp : (scalarPrefix observedPrimitives
      { (pooledFields 2).raw with inputStatistics := .text "unsupported_probe_value" }).result = .ok () := baseline_recipe_prefix
  simp only [validateScalar, hp]
  simp [adapterPrecheck, recipeRemainder, recipeOptions,
    pooledFields, ERPFields.raw, Family.text, setOption, setContains, builtinHashable,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, require]

theorem temporal_kind_container_is_typeerror :
    recipeRemainder observedPrimitives
      { (temporalFields "fixed_cosine" 1).raw with temporalBasisKind := .sequence .nil } =
      .error (.atomic "temporal declaration" .setMembership .typeError) := by
  have hn : Nonblank "temporal_lda" := rfl
  norm_num [recipeRemainder, recipeOptions, setOption, setContains, builtinHashable,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, erpContinuationFamily,
    liftCheck, readFamily, Family.text, readTextValue, hn, temporalFields,
    pooledFields, ERPFields.raw, rejectWhen, compareResult, compareCall, atomicCall,
    observedPrimitives, decodedCompare, numericScalar, require, ensure, positiveInt]
  simp_all

theorem adapter_option_container_is_typeerror :
    adapterChecks observedPrimitives { (pooledFields 2).raw with bnBuffers := .sequence .nil } =
      .error (.atomic "adapter batchnorm_buffers" .setMembership .typeError) := by
  norm_num [adapterChecks, adapterProjection, optimization, mathBound, finiteResult,
    rejectWhen, compareResult, compareCall, atomicCall, observedPrimitives,
    observedSmallFinite, decodedCompare, numericScalar, require,
    pooledFields, ERPFields.raw, setOption, setContains, builtinHashable,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]
  simp_all

theorem baseline_concrete_remainder :
    recipeRemainder observedPrimitives (pooledFields 2).raw = .ok fixtureRecipe := by
  have hn : Nonblank "pooled_lda" := rfl
  norm_num [recipeRemainder, recipeOptions, setOption, setContains, builtinHashable,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, erpContinuationFamily,
    liftCheck, readFamily, Family.text, readTextValue, hn,
    pooledFields, ERPFields.raw, ERPFields.recipe, fixtureRecipe,
    adapterChecks, adapterProjection, optimization, mathBound, finiteResult,
    rejectWhen, compareResult, compareCall, atomicCall, observedPrimitives,
    observedSmallFinite, huge100, huge400, decodedCompare, numericScalar,
    require, ensure, positiveInt, ExactBool]
  simp_all

theorem baseline_concrete_constructor : validateScalar observedPrimitives (pooledFields 2).raw = .ok fixtureRecipe := by
  rw [validateScalar, show adapterPrecheck (pooledFields 2).raw = .ok () by
    simp [adapterPrecheck, pooledFields, ERPFields.raw, require]]
  simp only [baseline_recipe_prefix]
  exact baseline_concrete_remainder

theorem legal_scalar_branch_nonempty :
    readBranchScalar observedPrimitives fixturePrimitives fixtureReference fixturePhysical
      "target_only_structured_erp" (.boolean false)
      (Fields.ofList (rawFields (pooledFields 2).raw))
      (plainRecord .pooled 2 4 2 5 (.text "none")) fixtureState =
      .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by
  have hs := selected_indices_preserve_order 3 [0, 2] (by decide) (by simp) (by simp)
  have hb := bind_plain_from_fields fixtureRecipe fixturePhysical [0, 2] fixtureModel
    rfl rfl rfl rfl rfl rfl
  simp only [fixturePhysical] at hb
  simp [readBranchScalar, ensure, liftCheck, fixture_selected_field_reads,
    fixture_ERP_fields_read, fixture_inner_fields_construct, fixturePhysical, hs,
    kwargs_reconstruct_before_scalar, baseline_concrete_constructor, hb]

theorem huge_temperature_at_actual_branch_position :
    readBranchScalar observedPrimitives fixturePrimitives fixtureReference fixturePhysical
      "target_only_structured_erp" (.boolean false)
      (Fields.ofList (rawFields hugeTemperatureRaw))
      (plainRecord .pooled 2 4 2 5 (.text "none")) fixtureState =
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  have hs := selected_indices_preserve_order 3 [0, 2] (by decide) (by simp) (by simp)
  simp [readBranchScalar, ensure, liftCheck, fixture_selected_field_reads,
    fixture_ERP_fields_read, fixture_inner_fields_construct, fixturePhysical, hs,
    kwargs_reconstruct_before_scalar, huge_temperature_concrete_constructor]

end
end N2P3.PersonalERPScalar
