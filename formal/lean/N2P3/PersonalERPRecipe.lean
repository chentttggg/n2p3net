import N2P3.ERPState

/-!
# Actual ERP recipe kwargs and personal ERP branch binding

Sources: recipes.py:17-144, subject_adapter.py:62-165,
calibrated.py:455-456,546-659, calibration_candidates.py:311-332.
This is a decoded numeric domain after `_json_record(...allow_nan=False)`.
Source correspondence additionally requires supported scalar representations
on which the actual np.isfinite/numeric conversions are defined and finite.
Finite JSON alone does not ensure this for arbitrary-size Python integers.
Huge-integer TypeError/overflow and general numeric conversion exceptions are
outside the inherited scalar codec boundary, not proved absent or accepted.
Parsed mappings have unique keys and decoded tensors have shape/product and
dtype/content consistency. Torch bytes, NumPy conversion, copying/ownership,
the earlier personal decision/QC/physical-execution gates and genuine fitting
are independent obligations, not opaque success primitives in this file.

`readArgs` models actual dataclass keyword defaults, not a nonexistent
TargetRecipe.from_record. `validateERPRecipe` is the ERP-treatment restriction
of that constructor plus its actual SubjectAdapterConfig projection. Other
neural/adapter treatments are outside this restriction. Rejection labels group
source failures; no arbitrary invalid Python object's exact exception trace is
claimed. ERP requires no residual adapter, so no FeatureTransferConfig success
is assumed. Every relevant scalar check is explicit. In particular epochs,
batch_size and lr retain Python bool/numeric comparison behavior; seed and
weight_decay have no invented validation. The independently decoded inner
ERP reader is the previously frozen implementation, never an assumed codec.
-/

/-!
The following namespace is the explicit current-version refinement added by
science_contract_v2.json. The frozen ERPState module remains unchanged.
-/
namespace N2P3.CurrentERPState
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def allZero (tensor : Tensor) : Prop := ∀ value ∈ tensor.values, numericScalar value = some 0

/-- Literal prototypes == prototypes[:, :1] broadcasting, on the already
shape-checked finite float64 domain. This is exact equality, not allclose. -/
def constantRows (tensor : Tensor) : Prop :=
  match tensor.shape with
  | [rows, bins] => ∀ row < rows, ∀ time < bins,
      numericScalar (tensor.values[row * bins + time]?.getD .null) =
        numericScalar (tensor.values[row * bins]?.getD .null)
  | _ => False

def currentGuard (model : Model) : Check Model := do
  if model.header.family = .xdawn then
    let filters ← tensorAt model.state "filters"
    let prototypes ← tensorAt model.state "prototypes"
    ensure (model.times / model.header.binWidth ≠ 1 ∧ ¬(allZero filters ∧ constantRows prototypes))
      "current xDAWN universal zero energy"
  pure model

def fromState (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (record : Fields) (state : NamedState) : Check Model :=
  (ERPState.fromState p ref record state).bind currentGuard

theorem other_families_current_guard_preserves (model : Model) (h : model.header.family ≠ .xdawn) :
    currentGuard model = .ok model := by simp [currentGuard, h]

theorem earlier_reader_error_precedes_new_guard (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (record : Fields) (state : NamedState) (fault : LedgerFault)
    (h : ERPState.fromState p ref record state = .error fault) :
    fromState p ref record state = .error fault := by rw [fromState, h]; rfl

theorem one_bin_rejected_after_individual_lookups (model : Model) (filters prototypes : Tensor)
    (hf : model.header.family = .xdawn) (hb : model.times / model.header.binWidth = 1)
    (hfilters : lookupTensor "filters" model.state = some filters)
    (hprototypes : lookupTensor "prototypes" model.state = some prototypes) :
    currentGuard model = .error (.sourceReject "current xDAWN universal zero energy") := by
  simp [currentGuard, hf, tensorAt, hfilters, hprototypes, hb, ensure]

theorem zero_filters_constant_prototypes_rejected (model : Model) (filters prototypes : Tensor)
    (hf : model.header.family = .xdawn)
    (hfilters : lookupTensor "filters" model.state = some filters)
    (hprototypes : lookupTensor "prototypes" model.state = some prototypes)
    (hz : allZero filters) (hc : constantRows prototypes) :
    currentGuard model = .error (.sourceReject "current xDAWN universal zero energy") := by
  simp [currentGuard, hf, tensorAt, hfilters, hprototypes, hz, hc, ensure]

theorem nondegenerate_current_guard_accepts (model : Model) (filters prototypes : Tensor)
    (hf : model.header.family = .xdawn) (hb : model.times / model.header.binWidth ≠ 1)
    (hfilters : lookupTensor "filters" model.state = some filters)
    (hprototypes : lookupTensor "prototypes" model.state = some prototypes)
    (hn : ¬(allZero filters ∧ constantRows prototypes)) :
    currentGuard model = .ok model := by
  simp [currentGuard, hf, tensorAt, hfilters, hprototypes, hb, hn, ensure]

theorem previous_one_time_xdawn_fixture_now_rejects :
    fromState fixturePrimitives fixtureReference (xdawnRecord 1 1 1 1 1 1 (.text "none"))
      fixtureXDawnState = .error (.sourceReject "current xDAWN universal zero energy") := by
  rw [fromState, xdawn_nonempty_construct]
  simp [currentGuard, tensorAt, fixtureXDawnState, xdawnNumericState,
    lookupTensor, ensure]

def currentXDawnState : NamedState := xdawnNumericState 1 2 1 [1, 2] [0, 1, 1, 0]
  [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] (fun _ => 1) 0

def currentXDawnModel : Model :=
  ⟨⟨.xdawn, 1, 1, .text "none", none⟩, 1, 2, currentXDawnState, some 1, some 1⟩

/-- Current b=2, nonzero-filter numeric artifact. Reference numerical predicates
are still explicit inherited primitives evaluated on identity4 here; this is
not a future-covariance or actual source-fit correctness theorem. -/
theorem current_two_time_xdawn_nonempty :
    fromState fixturePrimitives fixtureReference (xdawnRecord 1 1 1 2 1 1 (.text "none"))
      currentXDawnState = .ok currentXDawnModel := by
  have h := xdawn_family_constructs fixturePrimitives fixtureReference 1 1 1 2 1 1 (.text "none")
    [1, 2] [0, 1, 1, 0] [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] (fun _ => 1) 0
    (by decide) (by decide) (by decide) (by decide) (by decide) ⟨by decide, by decide⟩
    ⟨by decide, by decide⟩ rfl rfl rfl rfl rfl
  have h' : ERPState.fromState fixturePrimitives fixtureReference
      (xdawnRecord 1 1 1 2 1 1 (.text "none")) currentXDawnState = .ok currentXDawnModel := h
  rw [fromState, h']
  simp [currentGuard, currentXDawnModel, currentXDawnState, tensorAt, xdawnNumericState,
    lookupTensor, allZero, realTensor, numericScalar, ensure]

inductive NumericalException where
  | linAlgError
  | other (name : String)
  deriving DecidableEq

inductive Operation where
  | oas | finiteAll | cholesky | tangent | affine
  deriving DecidableEq

inductive PredictionFault where
  | covarianceNonfinite
  | covarianceCholesky
  | propagated (operation : Operation) (exception : NumericalException)
  deriving DecidableEq

structure Observation (Scores : Type) where
  calls : List Operation
  result : Except PredictionFault Scores

/-- Independent numerical operations, each with its actual narrow source role.
Covariance rows/features/scores are backend carriers, not exact-real matrices.
`rowFinite` is per-row np.isfinite(...).all; `cholesky` is the actual batched
numerical call. No axiom identifies numerical success with exact-real SPD.
Tangent and BLAS affine may independently return nonfinite backend values or
raise. They are never replaced by a whole prediction-success primitive. -/
structure PredictionNumerics (Input CovarianceRow Features Scores : Type) where
  oas : Input → Except NumericalException (List CovarianceRow)
  rowFinite : CovarianceRow → Bool
  cholesky : List CovarianceRow → Except NumericalException Unit
  tangent : Tensor → List CovarianceRow → Except NumericalException Features
  affine : Tensor → Tensor → Features → Except NumericalException Scores

def predictAfterAugmentation {Input CovarianceRow Features Scores : Type}
    (p : PredictionNumerics Input CovarianceRow Features Scores)
    (input : Input) (reference coefficient intercept : Tensor) : Observation Scores :=
  match p.oas input with
  | .error exception => ⟨[.oas], .error (.propagated .oas exception)⟩
  | .ok covariance =>
    if covariance.all p.rowFinite then
      match p.cholesky covariance with
      | .error .linAlgError => ⟨[.oas, .finiteAll, .cholesky], .error .covarianceCholesky⟩
      | .error exception => ⟨[.oas, .finiteAll, .cholesky], .error (.propagated .cholesky exception)⟩
      | .ok () =>
        match p.tangent reference covariance with
        | .error exception => ⟨[.oas, .finiteAll, .cholesky, .tangent], .error (.propagated .tangent exception)⟩
        | .ok features =>
          match p.affine coefficient intercept features with
          | .error exception => ⟨[.oas, .finiteAll, .cholesky, .tangent, .affine], .error (.propagated .affine exception)⟩
          | .ok scores => ⟨[.oas, .finiteAll, .cholesky, .tangent, .affine], .ok scores⟩
    else ⟨[.oas, .finiteAll], .error .covarianceNonfinite⟩

theorem nonfinite_covariance_stops_before_cholesky {Input CovarianceRow Features Scores : Type}
    (p : PredictionNumerics Input CovarianceRow Features Scores) (input : Input)
    (reference coefficient intercept : Tensor) (covariance : List CovarianceRow)
    (ho : p.oas input = .ok covariance) (hf : covariance.all p.rowFinite = false) :
    predictAfterAugmentation p input reference coefficient intercept =
      ⟨[.oas, .finiteAll], .error .covarianceNonfinite⟩ := by
  simp [predictAfterAugmentation, ho, hf]

theorem one_nonfinite_row_prevents_batch_tangent {Input CovarianceRow Features Scores : Type}
    (p : PredictionNumerics Input CovarianceRow Features Scores) (input : Input)
    (reference coefficient intercept : Tensor) (before after : List CovarianceRow) (bad : CovarianceRow)
    (ho : p.oas input = .ok (before ++ bad :: after)) (hb : p.rowFinite bad = false) :
    predictAfterAugmentation p input reference coefficient intercept =
      ⟨[.oas, .finiteAll], .error .covarianceNonfinite⟩ := by
  apply nonfinite_covariance_stops_before_cholesky p input reference coefficient intercept _ ho
  simp [hb]

theorem cholesky_linalg_error_converted_before_tangent {Input CovarianceRow Features Scores : Type}
    (p : PredictionNumerics Input CovarianceRow Features Scores) (input : Input)
    (reference coefficient intercept : Tensor) (covariance : List CovarianceRow)
    (ho : p.oas input = .ok covariance) (hf : covariance.all p.rowFinite = true)
    (hc : p.cholesky covariance = .error .linAlgError) :
    predictAfterAugmentation p input reference coefficient intercept =
      ⟨[.oas, .finiteAll, .cholesky], .error .covarianceCholesky⟩ := by
  simp [predictAfterAugmentation, ho, hf, hc]

theorem other_cholesky_exception_propagates {Input CovarianceRow Features Scores : Type}
    (p : PredictionNumerics Input CovarianceRow Features Scores) (input : Input)
    (reference coefficient intercept : Tensor) (covariance : List CovarianceRow) (name : String)
    (ho : p.oas input = .ok covariance) (hf : covariance.all p.rowFinite = true)
    (hc : p.cholesky covariance = .error (.other name)) :
    predictAfterAugmentation p input reference coefficient intercept =
      ⟨[.oas, .finiteAll, .cholesky], .error (.propagated .cholesky (.other name))⟩ := by
  simp [predictAfterAugmentation, ho, hf, hc]

theorem tangent_error_is_not_cholesky_error {Input CovarianceRow Features Scores : Type}
    (p : PredictionNumerics Input CovarianceRow Features Scores) (input : Input)
    (reference coefficient intercept : Tensor) (covariance : List CovarianceRow) (exception : NumericalException)
    (ho : p.oas input = .ok covariance) (hf : covariance.all p.rowFinite = true)
    (hc : p.cholesky covariance = .ok ()) (ht : p.tangent reference covariance = .error exception) :
    predictAfterAugmentation p input reference coefficient intercept =
      ⟨[.oas, .finiteAll, .cholesky, .tangent], .error (.propagated .tangent exception)⟩ := by
  simp [predictAfterAugmentation, ho, hf, hc, ht]

/-- After source `_input` and temporal handling, an empty batch returns its
empty score array before OAS. A nonempty batch uses the literal spatial sum
and prototype-first augmentation from the frozen coordinate definitions. -/
def predictXDawnBatch {c bins effective : ℕ} {CovarianceRow Features Scores : Type}
    (p : PredictionNumerics (List (ERPFeatures.Epoch (4 * effective) bins)) CovarianceRow Features Scores)
    (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) bins)
    (reference coefficient intercept : Tensor) (emptyScores : Scores)
    (epochs : List (ERPFeatures.Epoch c bins)) : Observation Scores :=
  if epochs = [] then ⟨[], .ok emptyScores⟩
  else predictAfterAugmentation p
    (epochs.map (fun epoch => prototypeFirst prototypes (spatialTransform filters epoch)))
    reference coefficient intercept

theorem empty_batch_skips_all_numerical_operations {c bins effective : ℕ} {CovarianceRow Features Scores : Type}
    (p : PredictionNumerics (List (ERPFeatures.Epoch (4 * effective) bins)) CovarianceRow Features Scores)
    (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) bins)
    (reference coefficient intercept : Tensor) (emptyScores : Scores) :
    predictXDawnBatch p filters prototypes reference coefficient intercept emptyScores [] =
      ⟨[], .ok emptyScores⟩ := by simp [predictXDawnBatch]

theorem nonempty_batch_reaches_literal_augmentation {c bins effective : ℕ} {CovarianceRow Features Scores : Type}
    (p : PredictionNumerics (List (ERPFeatures.Epoch (4 * effective) bins)) CovarianceRow Features Scores)
    (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) bins)
    (reference coefficient intercept : Tensor) (emptyScores : Scores)
    (epochs : List (ERPFeatures.Epoch c bins)) (hn : epochs ≠ []) :
    predictXDawnBatch p filters prototypes reference coefficient intercept emptyScores epochs =
      predictAfterAugmentation p
        (epochs.map (fun epoch => prototypeFirst prototypes (spatialTransform filters epoch)))
        reference coefficient intercept := by simp [predictXDawnBatch, hn]

/-- Classification-only numerical-operation countermodel, not actual NumPy
matrix execution. The independently produced source probe owns real numerical
witnesses. The false-row result follows from finite-all before either backend. -/
def finiteRowControl : PredictionNumerics (List Bool) Bool Unit (List ℝ) := {
  oas := fun input => .ok input
  rowFinite := id
  cholesky := fun _ => .ok ()
  tangent := fun _ _ => .ok ()
  affine := fun _ _ _ => .ok [7] }

theorem finite_row_countermodel_rejects_whole_batch :
    predictAfterAugmentation finiteRowControl [true, false]
      (realTensor [1] [1]) (realTensor [1] [1]) (realTensor [] [0]) =
      ⟨[.oas, .finiteAll], .error .covarianceNonfinite⟩ := by
  simp [predictAfterAugmentation, finiteRowControl]

end
end N2P3.CurrentERPState

namespace N2P3.PersonalERPRecipe
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure RawRecipe where
  name : Value := .null
  initialization : Value := .text "source"
  inputStatistics : Value := .text "target_prefix"
  bnBuffers : Value := .text "online"
  bnAffine : Value := .text "trainable"
  weights : Value := .text "non_bn"
  dropout : Value := .text "checkpoint"
  batchSize : Value := .integer 32
  lr : Value := .number (1 / 1000)
  weightDecay : Value := .number (1 / 10000)
  epochs : Value := .integer 60
  maxUpdates : Value := .null
  seed : Value := .integer 0
  refitFullPrefix : Value := .boolean false
  targetAdapterConfig : Value := .null
  candidateLossWeight : Value := .null
  candidateLossTemperature : Value := .number 1
  sourcePriorWeight : Value := .number 0
  erpBinWidth : Value := .integer 1
  temporalBasisKind : Value := .null
  temporalRank : Value := .null

def rawFields (r : RawRecipe) : List (String × Value) :=
  [("name", r.name), ("initialization", r.initialization),
   ("input_statistics", r.inputStatistics), ("bn_buffers", r.bnBuffers),
   ("bn_affine", r.bnAffine), ("weights", r.weights), ("dropout", r.dropout),
   ("batch_size", r.batchSize), ("lr", r.lr), ("weight_decay", r.weightDecay),
   ("epochs", r.epochs), ("max_updates", r.maxUpdates), ("seed", r.seed),
   ("refit_full_prefix", r.refitFullPrefix), ("target_adapter_config", r.targetAdapterConfig),
   ("candidate_loss_weight", r.candidateLossWeight),
   ("candidate_loss_temperature", r.candidateLossTemperature),
   ("source_prior_weight", r.sourcePriorWeight), ("erp_bin_width", r.erpBinWidth),
   ("temporal_basis_kind", r.temporalBasisKind), ("temporal_rank", r.temporalRank)]

def keywordNames : List String := (rawFields {}).map Prod.fst
def getDefault (f : Fields) (key : String) (value : Value) : Value := (f.get key).getD value

/-- Literal kwargs reconstruction. The required name is the sole field with no default. -/
def readArgs (f : Fields) : Check RawRecipe := do
  ensure (fieldKeys f ⊆ keywordNames) "TargetRecipe unexpected keyword"
  let name ← required f "name"
  pure {
    name := name
    initialization := getDefault f "initialization" (.text "source")
    inputStatistics := getDefault f "input_statistics" (.text "target_prefix")
    bnBuffers := getDefault f "bn_buffers" (.text "online")
    bnAffine := getDefault f "bn_affine" (.text "trainable")
    weights := getDefault f "weights" (.text "non_bn")
    dropout := getDefault f "dropout" (.text "checkpoint")
    batchSize := getDefault f "batch_size" (.integer 32)
    lr := getDefault f "lr" (.number (1 / 1000))
    weightDecay := getDefault f "weight_decay" (.number (1 / 10000))
    epochs := getDefault f "epochs" (.integer 60)
    maxUpdates := getDefault f "max_updates" .null
    seed := getDefault f "seed" (.integer 0)
    refitFullPrefix := getDefault f "refit_full_prefix" (.boolean false)
    targetAdapterConfig := getDefault f "target_adapter_config" .null
    candidateLossWeight := getDefault f "candidate_loss_weight" .null
    candidateLossTemperature := getDefault f "candidate_loss_temperature" (.number 1)
    sourcePriorWeight := getDefault f "source_prior_weight" (.number 0)
    erpBinWidth := getDefault f "erp_bin_width" (.integer 1)
    temporalBasisKind := getDefault f "temporal_basis_kind" .null
    temporalRank := getDefault f "temporal_rank" .null }

def scalar (v : Value) (stage : String) : Check ℝ :=
  match numericScalar v with
  | some x => .ok x
  | none => .error (.fieldFailure stage)

def nonBoolScalar (v : Value) (stage : String) : Check ℝ := do
  ensure (¬ExactBool v) stage
  scalar v stage

theorem non_boolean_real_reads (value : ℝ) (stage : String) :
    nonBoolScalar (.number value) stage = .ok value := by
  simp [nonBoolScalar, scalar, ExactBool, numericScalar, ensure]

def readTextValue (v : Value) (stage : String) : Check String :=
  match v with | .text s => .ok s | _ => .error (.fieldFailure stage)

def validMaxUpdates (v : Value) : Prop :=
  v = .null ∨ ∃ n : ℤ, v = .integer n ∧ 1 ≤ n

/-- All fields passed by TargetRecipe.adapter_config, plus defaults retained
from SubjectAdapterConfig. This is a fieldwise projection, not a validator oracle. -/
structure AdapterProjection where
  headKind : String
  inputStatistics : Value
  inputMean : Value
  inputStd : Value
  batchnormBuffers : Value
  freezeBatchnormStats : Bool
  batchnormAffine : Value
  trainableWeights : Value
  dropoutMode : Value
  batchSize : Value
  lr : Value
  weightDecay : Value
  epochs : Value
  maxUpdates : Value
  seed : Value
  valGroupFraction : Value
  refitFullPrefix : Value
  candidateLossWeight : Value
  candidateLossTemperature : Value
  sourcePriorWeight : Value
  valGroupsMin : ℕ := 2
  valGroupsMax : ℕ := 6
  earlyStopMinDelta : ℝ := 1 / 1000000
  earlyStopPatience : ℕ := 6
  targetStatWeight : ℝ := 1 / 4
  asMMDLambda : ℝ := 0
  asMMDWarmup : ℕ := 1
  asMMDMin : ℝ := 1
  asMMDMax : ℝ := 6
  bnReestimationBatchSize : ℕ := 128
  validationBehavior : String := "select_epoch"

def adapterProjection (r : RawRecipe) : AdapterProjection := {
  headKind := if r.weights = .text "adapter" then "adapter"
    else if r.weights ∈ [.text "non_bn", .text "spatial", .text "temporal"] then "full_fine"
    else "classifier_fine"
  inputStatistics := r.inputStatistics
  inputMean := .null
  inputStd := .null
  batchnormBuffers := r.bnBuffers
  freezeBatchnormStats := r.bnBuffers != .text "online"
  batchnormAffine := r.bnAffine
  trainableWeights := r.weights
  dropoutMode := r.dropout
  batchSize := r.batchSize
  lr := r.lr
  weightDecay := r.weightDecay
  epochs := r.epochs
  maxUpdates := r.maxUpdates
  seed := r.seed
  valGroupFraction := .null
  refitFullPrefix := r.refitFullPrefix
  candidateLossWeight := if r.candidateLossWeight = .null then .number 0 else r.candidateLossWeight
  candidateLossTemperature := r.candidateLossTemperature
  sourcePriorWeight := r.sourcePriorWeight }

structure Recipe where
  raw : RawRecipe
  family : Family
  width : ℕ
  temporalKind : Option String
  temporalRank : Option ℕ
  adapter : AdapterProjection

def validateERPRecipe (r : RawRecipe) : Check Recipe := do
  ensure (r.targetAdapterConfig = .null) "ERP target_adapter_config"
  if r.candidateLossWeight ≠ .null then
    let loss ← nonBoolScalar r.candidateLossWeight "candidate_loss_weight"
    ensure (0 ≤ loss) "candidate_loss_weight"
  let temperature ← nonBoolScalar r.candidateLossTemperature "candidate_loss_temperature"
  ensure (0 < temperature) "candidate_loss_temperature"
  let prior ← nonBoolScalar r.sourcePriorWeight "source_prior_weight"
  ensure (0 ≤ prior) "source_prior_weight"
  let family ← readFamily r.initialization
  ensure (r.inputStatistics ∈ [.text "source", .text "target_prefix"]) "recipe input_statistics"
  ensure (r.bnBuffers ∈ [.text "source", .text "calibration", .text "online"]) "recipe bn_buffers"
  ensure (r.bnAffine ∈ [.text "frozen", .text "trainable"]) "recipe bn_affine"
  ensure (r.weights ∈ [.text "none", .text "head", .text "non_bn", .text "spatial", .text "temporal", .text "adapter"]) "recipe weights"
  ensure (r.dropout ∈ [.text "off", .text "checkpoint"]) "recipe dropout"
  let name ← readTextValue r.name "recipe name.strip"
  ensure (Nonblank name) "recipe name"
  ensure (r.weights = .text "none" ∧ r.bnAffine = .text "frozen" ∧ r.dropout = .text "off") "ERP frozen treatment"
  ensure (r.candidateLossWeight = .null ∧ prior = 0) "ERP no optimized loss"
  let width ← positiveInt r.erpBinWidth "erp_bin_width"
  let (kind, rank) ← if family = .temporal then do
    ensure (r.temporalBasisKind ∈ [.text "fixed_cosine", .text "source_supervised"]) "temporal declaration"
    let rank ← positiveInt r.temporalRank "temporal_rank"
    ensure (width = 1) "temporal width"
    let kind ← readTextValue r.temporalBasisKind "temporal kind"
    pure (some kind, some rank)
  else do
    ensure (r.temporalBasisKind = .null ∧ r.temporalRank = .null) "non-temporal declaration"
    pure (none, none)
  ensure (r.bnBuffers ≠ .text "online") "statistics-only buffers"
  -- The remaining nonconstant SubjectAdapterConfig checks are explicit.
  let epochs ← scalar r.epochs "adapter epochs comparison"
  let batch ← scalar r.batchSize "adapter batch_size comparison"
  let lr ← scalar r.lr "adapter lr comparison"
  ensure (1 ≤ epochs ∧ 1 ≤ batch ∧ 0 < lr) "adapter optimization"
  ensure (validMaxUpdates r.maxUpdates) "adapter max_updates"
  ensure (ExactBool r.refitFullPrefix) "adapter refit_full_prefix"
  pure ⟨r, family, width, kind, rank, adapterProjection r⟩

def fromKwargs (f : Fields) : Check Recipe := (readArgs f).bind validateERPRecipe

/-- Literal source record omissions. Python numeric equality controls scalar
omissions, so this writer does not promise to preserve integer-vs-real tags. -/
def recordFields (r : RawRecipe) : Fields := Fields.ofList ((rawFields r).filter fun pair =>
  if pair.1 = "target_adapter_config" then decide (r.targetAdapterConfig ≠ .null)
  else if pair.1 = "candidate_loss_weight" then decide (r.candidateLossWeight ≠ .null)
  else if pair.1 = "candidate_loss_temperature" then
    decide (r.candidateLossWeight ≠ .null ∨ ¬PersonalAdmission.PyEq r.candidateLossTemperature (.number 1))
  else if pair.1 = "source_prior_weight" then decide (¬PersonalAdmission.PyEq r.sourcePriorWeight (.number 0))
  else if pair.1 = "erp_bin_width" then decide (¬PersonalAdmission.PyEq r.erpBinWidth (.integer 1))
  else if pair.1 = "temporal_basis_kind" ∨ pair.1 = "temporal_rank" then decide (r.temporalBasisKind ≠ .null)
  else true)

structure ERPFields where
  name : String
  family : Family
  inputStatistics : String
  buffers : String
  epochs : Value
  batchSize : Value
  lr : Value
  weightDecay : Value
  maxUpdates : Value
  seed : Value
  refit : Bool
  temperature : ℝ
  width : ℕ
  kind : Option String
  rank : Option ℕ

def ERPFields.raw (f : ERPFields) : RawRecipe := {
  name := .text f.name, initialization := .text f.family.text,
  inputStatistics := .text f.inputStatistics, bnBuffers := .text f.buffers,
  bnAffine := .text "frozen", weights := .text "none", dropout := .text "off",
  epochs := f.epochs, batchSize := f.batchSize, lr := f.lr, weightDecay := f.weightDecay,
  maxUpdates := f.maxUpdates, seed := f.seed, refitFullPrefix := .boolean f.refit,
  candidateLossTemperature := .number f.temperature, erpBinWidth := .integer f.width,
  temporalBasisKind := match f.kind with | none => .null | some k => .text k,
  temporalRank := match f.rank with | none => .null | some k => .integer k }

structure Conditions (f : ERPFields) : Prop where
  name : Nonblank f.name
  statistics : f.inputStatistics = "source" ∨ f.inputStatistics = "target_prefix"
  buffers : f.buffers = "source" ∨ f.buffers = "calibration"
  temperature : 0 < f.temperature
  width : 0 < f.width
  optimization : ∃ epochs batch lr : ℝ, numericScalar f.epochs = some epochs ∧
    numericScalar f.batchSize = some batch ∧ numericScalar f.lr = some lr ∧
    1 ≤ epochs ∧ 1 ≤ batch ∧ 0 < lr
  maxUpdates : validMaxUpdates f.maxUpdates
  temporal : (f.family = .temporal ∧ f.width = 1 ∧
    ∃ kind rank, f.kind = some kind ∧ f.rank = some rank ∧
      (kind = "fixed_cosine" ∨ kind = "source_supervised") ∧ 0 < rank) ∨
    (f.family ≠ .temporal ∧ f.kind = none ∧ f.rank = none)

def ERPFields.recipe (f : ERPFields) : Recipe :=
  ⟨f.raw, f.family, f.width, f.kind, f.rank, adapterProjection f.raw⟩

theorem explicit_kwargs_reconstruct_all_fields (r : RawRecipe) :
    readArgs (Fields.ofList (rawFields r)) = .ok r := by
  cases r
  simp [readArgs, rawFields, keywordNames, fieldKeys, Fields.ofList, Fields.get,
    getDefault, required, ensure]

theorem missing_keywords_use_actual_defaults (name : Value) :
    readArgs (Fields.ofList [("name", name)]) = .ok { name := name } := by
  simp [readArgs, keywordNames, rawFields, fieldKeys, Fields.ofList, Fields.get, getDefault, required, ensure]

theorem ERP_constructor_from_individual_fields (f : ERPFields) (h : Conditions f) :
    validateERPRecipe f.raw = .ok f.recipe := by
  rcases h with ⟨hn, hs, hb, ht, hw, ho, hm, htemporal⟩
  rcases ho with ⟨epochs, batch, lr, he, hba, hl, hep, hbp, hlp⟩
  have he' : scalar f.epochs "adapter epochs comparison" = .ok epochs := by simp [scalar, he]
  have hba' : scalar f.batchSize "adapter batch_size comparison" = .ok batch := by simp [scalar, hba]
  have hl' : scalar f.lr "adapter lr comparison" = .ok lr := by simp [scalar, hl]
  have hfamily := family_text_decodes f.family
  rcases htemporal with ⟨hf, hwidth, kind, rank, hk, hr, hkind, hrank⟩ | ⟨hf, hk, hr⟩
  · rcases hkind with hkind | hkind <;> rcases hs with hs | hs <;> rcases hb with hb | hb <;>
      simp_all [validateERPRecipe, ERPFields.raw, ERPFields.recipe,
        non_boolean_real_reads, ExactBool, readTextValue, ensure, positiveInt]
  · rcases hs with hs | hs <;> rcases hb with hb | hb <;>
      simp_all [validateERPRecipe, ERPFields.raw, ERPFields.recipe,
        non_boolean_real_reads, ExactBool, readTextValue, ensure, positiveInt]

theorem kwargs_constructor_from_individual_fields (f : ERPFields) (h : Conditions f) :
    fromKwargs (Fields.ofList (rawFields f.raw)) = .ok f.recipe := by
  rw [fromKwargs, explicit_kwargs_reconstruct_all_fields]
  exact ERP_constructor_from_individual_fields f h

theorem canonical_record_reconstructs_fields (f : ERPFields)
    (hk : f.kind = none → f.rank = none) :
    readArgs (recordFields f.raw) = .ok f.raw := by
  by_cases ht : f.temperature = 1 <;> by_cases hw : f.width = 1 <;>
    cases hkind : f.kind with
    | none =>
      have hr := hk hkind
      simp [readArgs, recordFields, rawFields, ERPFields.raw, fieldKeys, keywordNames,
        Fields.ofList, Fields.get, getDefault, required, ensure, ht, hw, hkind, hr,
        PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]
    | some kind =>
      simp [readArgs, recordFields, rawFields, ERPFields.raw, fieldKeys, keywordNames,
        Fields.ofList, Fields.get, getDefault, required, ensure, ht, hw, hkind,
        PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem canonical_record_constructor (f : ERPFields) (h : Conditions f) :
    fromKwargs (recordFields f.raw) = .ok f.recipe := by
  have hk : f.kind = none → f.rank = none := by
    intro hn
    rcases h.temporal with ⟨_, _, kind, rank, he, _, _⟩ | ⟨_, _, hr⟩
    · simp [hn] at he
    · exact hr
  rw [fromKwargs, canonical_record_reconstructs_fields f hk]
  exact ERP_constructor_from_individual_fields f h

theorem adapter_projection_preserves_optimizer_fields (r : RawRecipe) :
    (adapterProjection r).batchSize = r.batchSize ∧
    (adapterProjection r).lr = r.lr ∧ (adapterProjection r).weightDecay = r.weightDecay ∧
    (adapterProjection r).epochs = r.epochs ∧ (adapterProjection r).maxUpdates = r.maxUpdates ∧
    (adapterProjection r).seed = r.seed ∧ (adapterProjection r).refitFullPrefix = r.refitFullPrefix := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem adapter_projection_preserves_treatment_fields (r : RawRecipe) :
    (adapterProjection r).inputStatistics = r.inputStatistics ∧
    (adapterProjection r).batchnormBuffers = r.bnBuffers ∧
    (adapterProjection r).batchnormAffine = r.bnAffine ∧
    (adapterProjection r).trainableWeights = r.weights ∧
    (adapterProjection r).dropoutMode = r.dropout ∧
    (adapterProjection r).candidateLossTemperature = r.candidateLossTemperature ∧
    (adapterProjection r).sourcePriorWeight = r.sourcePriorWeight := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem ERP_adapter_constant_checks (f : ERPFields) (h : Conditions f) :
    (adapterProjection f.raw).headKind = "classifier_fine" ∧
    (adapterProjection f.raw).inputMean = .null ∧ (adapterProjection f.raw).inputStd = .null ∧
    (adapterProjection f.raw).valGroupFraction = .null ∧
    (adapterProjection f.raw).candidateLossWeight = .number 0 ∧
    (adapterProjection f.raw).freezeBatchnormStats = true := by
  rcases h.buffers with hb | hb <;> simp [adapterProjection, ERPFields.raw, hb]

/-- These actual default checks cannot fail for any ERP recipe projection.
early_stop_patience=6 comes from train/early_stopping.py:8. -/
theorem adapter_unchanged_default_checks (r : RawRecipe) :
    1 ≤ (adapterProjection r).valGroupsMin ∧
    (adapterProjection r).valGroupsMin ≤ (adapterProjection r).valGroupsMax ∧
    1 ≤ (adapterProjection r).earlyStopPatience ∧
    0 ≤ (adapterProjection r).earlyStopMinDelta ∧
    0 ≤ (adapterProjection r).targetStatWeight ∧ (adapterProjection r).targetStatWeight ≤ 1 ∧
    (adapterProjection r).asMMDLambda = 0 ∧
    1 ≤ (adapterProjection r).asMMDWarmup ∧
    0 < (adapterProjection r).asMMDMin ∧ (adapterProjection r).asMMDMin ≤ (adapterProjection r).asMMDMax ∧
    1 ≤ (adapterProjection r).bnReestimationBatchSize ∧
    (adapterProjection r).validationBehavior = "select_epoch" := by
  norm_num [adapterProjection]

def selectedTensor (indices : List ℕ) : Tensor :=
  ⟨"torch.int64", [indices.length], valuesToList (encodeNats indices)⟩

def readSelected (channels : ℕ) (tensor : Tensor) : Check (List ℕ) := do
  ensure (tensor.dtype = "torch.int64") "ERP selected dtype"
  ensure (tensor.shape = [tensor.values.length] ∧ tensor.values ≠ []) "ERP selected rank/nonempty"
  let indices ← match decodeNats (valuesOfList tensor.values) with
    | some indices => .ok indices | none => .error (.sourceReject "ERP selected negative/noninteger")
  ensure (indices.Nodup ∧ ∀ i ∈ indices, i < channels) "ERP selected unique/range"
  pure indices

theorem natural_list_values (indices : List ℕ) :
    valuesToList (encodeNats indices) = indices.map (fun n : ℕ => Value.integer (n : ℤ)) := by
  induction indices with
  | nil => rfl
  | cons n ns ih => simp [encodeNats, valuesToList, ih]

theorem natural_values_decode (indices : List ℕ) :
    decodeNats (valuesOfList (valuesToList (encodeNats indices))) = some indices := by
  induction indices with
  | nil => rfl
  | cons n ns ih => simp [encodeNats, valuesToList, valuesOfList, decodeNats, ih]

theorem selected_indices_preserve_order (channels : ℕ) (indices : List ℕ)
    (hn : indices ≠ []) (hu : indices.Nodup) (hr : ∀ i ∈ indices, i < channels) :
    readSelected channels (selectedTensor indices) = .ok indices := by
  have hlen : (valuesToList (encodeNats indices)).length = indices.length := by
    simp [natural_list_values]
  have hnon : valuesToList (encodeNats indices) ≠ [] := by
    intro he
    have hz : indices.length = 0 := by
      simpa [natural_list_values] using congrArg List.length he
    exact hn (List.length_eq_zero_iff.mp hz)
  simp [readSelected, selectedTensor, hlen, hnon, natural_values_decode, ensure, hu]
  rw [if_pos hr]

def usesSource (model : Model) : Bool :=
  match model.header.temporal with | none => false | some basis => decide (basis.role = .source)

def modelKind (source : Bool) : String :=
  if source then "source_temporal_target_erp" else "target_only_structured_erp"

structure Physical where
  channels : ℕ
  times : ℕ
  baseline : Value

structure Bound where
  recipe : Recipe
  selected : List ℕ
  model : Model

/-- Actual numeric ERP branch after the independent earlier personal gates.
No whole physical or personal admission primitive is hidden here. -/
def bindERP (recipe : Recipe) (kind : String) (sourceFlag : Value) (physical : Physical)
    (selected : List ℕ) (model : Model) : Check Bound := do
  ensure (model.channels = selected.length ∧ model.times = physical.times ∧
    PersonalAdmission.PyEq model.header.baseline physical.baseline ∧
    recipe.family = model.header.family ∧ recipe.width = model.header.binWidth)
    "ERP state recipe physical contract"
  ensure (sourceFlag = .boolean (usesSource model) ∧ kind = modelKind (usesSource model)) "ERP strict source role"
  match model.header.temporal with
  | none => pure ()
  | some basis =>
    ensure (recipe.temporalKind = some (if usesSource model then "source_supervised" else "fixed_cosine") ∧
      recipe.temporalRank = some basis.rank) "ERP temporal kind/rank"
  pure ⟨recipe, selected, model⟩

def readBranch (p : TemporalPrimitives) (ref : ReferencePrimitives) (physical : Physical)
    (kind : String) (sourceFlag : Value) (kwargs record : Fields) (state : NamedState) : Check Bound := do
  ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind"
  let tensor ← tensorAt state "waveform_channels"
  let selected ← readSelected physical.channels tensor
  let model ← CurrentERPState.fromState p ref record (erpFields state)
  let recipe ← fromKwargs kwargs
  bindERP recipe kind sourceFlag physical selected model

theorem bind_plain_from_fields (recipe : Recipe) (physical : Physical) (selected : List ℕ) (model : Model)
    (ht : model.header.temporal = none) (hc : model.channels = selected.length)
    (hn : model.times = physical.times) (hb : PersonalAdmission.PyEq model.header.baseline physical.baseline)
    (hf : recipe.family = model.header.family) (hw : recipe.width = model.header.binWidth) :
    bindERP recipe "target_only_structured_erp" (.boolean false) physical selected model =
      .ok ⟨recipe, selected, model⟩ := by
  simp [bindERP, usesSource, ht, hc, hn, hb, hf, hw, modelKind, ensure]

theorem bind_temporal_from_fields (recipe : Recipe) (physical : Physical) (selected : List ℕ)
    (model : Model) (basis : TemporalModel) (ht : model.header.temporal = some basis)
    (hc : model.channels = selected.length) (hn : model.times = physical.times)
    (hb : PersonalAdmission.PyEq model.header.baseline physical.baseline)
    (hf : recipe.family = model.header.family) (hw : recipe.width = model.header.binWidth)
    (hk : recipe.temporalKind = some (if basis.role = .source then "source_supervised" else "fixed_cosine"))
    (hr : recipe.temporalRank = some basis.rank) :
    bindERP recipe (modelKind (decide (basis.role = .source))) (.boolean (decide (basis.role = .source)))
      physical selected model = .ok ⟨recipe, selected, model⟩ := by
  simp [bindERP, usesSource, ht, hc, hn, hb, hf, hw, hk, hr, ensure]

theorem integer_source_flags_rejected (recipe : Recipe) (physical : Physical) (selected : List ℕ)
    (model : Model) (kind : String) (flag : ℤ)
    (hc : model.channels = selected.length) (hn : model.times = physical.times)
    (hb : PersonalAdmission.PyEq model.header.baseline physical.baseline)
    (hf : recipe.family = model.header.family) (hw : recipe.width = model.header.binWidth) :
    bindERP recipe kind (.integer flag) physical selected model = .error (.sourceReject "ERP strict source role") := by
  simp [bindERP, hc, hn, hb, hf, hw, ensure]

theorem waveform_name_cannot_have_ERP_prefix (name : String) :
    "waveform_channels" ≠ "erp." ++ name := by
  intro h
  have hh := congrArg stripERP h
  have hnone : stripERP "waveform_channels" = none := rfl
  rw [hnone, prefixed_name_strips_once] at hh
  cases hh

theorem waveform_lookup_after_prefixed_fields (state : NamedState) (channels : Tensor) :
    lookupTensor "waveform_channels"
      (state.map (fun entry => ("erp." ++ entry.1, entry.2)) ++ [("waveform_channels", channels)]) = some channels := by
  induction state with
  | nil => simp [lookupTensor]
  | cons entry tail ih =>
    rcases entry with ⟨name, tensor⟩
    simp [lookupTensor, waveform_name_cannot_have_ERP_prefix, ih]

theorem personal_writer_waveform_field (model : Model) (channels : Tensor) :
    lookupTensor "waveform_channels" (writePersonalERP model channels []) = some channels := by
  simp only [writePersonalERP, copied_state_preserves_all_fields]
  exact waveform_lookup_after_prefixed_fields model.state channels

def plainModel (f : ERPFields) (c t nfilter : ℕ) (baseline : Value)
    (coefficient : Fin (c * (t / f.width)) → ℝ) (intercept : ℝ) : Model :=
  ⟨⟨f.family, f.width, nfilter, baseline, none⟩, c, t, linearState coefficient intercept, none, none⟩

theorem plain_personal_branch_from_individual_fields (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (h : Conditions f) (physical : Physical) (indices : List ℕ) (nfilter : ℕ)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (hn : indices ≠ []) (hu : indices.Nodup) (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / f.width) (hfilters : 0 < nfilter) :
    readBranch p ref physical "target_only_structured_erp" (.boolean false)
      (recordFields f.raw) (plainRecord f.family f.width nfilter indices.length physical.times physical.baseline)
      (writePersonalERP (plainModel f indices.length physical.times nfilter physical.baseline coefficient intercept)
        (selectedTensor indices) []) =
      .ok ⟨f.recipe, indices, plainModel f indices.length physical.times nfilter physical.baseline coefficient intercept⟩ := by
  have hreader := plain_family_constructs p ref f.family f.width nfilter indices.length physical.times
    physical.baseline coefficient intercept h.width hfilters (List.length_pos_iff.mpr hn) ht hbins hf hb
  have hcurrent : CurrentERPState.fromState p ref
      (plainRecord f.family f.width nfilter indices.length physical.times physical.baseline)
      (linearState coefficient intercept) =
      .ok (plainModel f indices.length physical.times nfilter physical.baseline coefficient intercept) := by
    rw [CurrentERPState.fromState, hreader]
    exact CurrentERPState.other_families_current_guard_preserves _ hf.2
  have hprefix := personal_prefix_parser_recovers_fields
    (plainModel f indices.length physical.times nfilter physical.baseline coefficient intercept)
    (selectedTensor indices) [] (by rfl)
  have hselected := selected_indices_preserve_order physical.channels indices hn hu hr
  have hrecipe := canonical_record_constructor f h
  have hbind := bind_plain_from_fields f.recipe physical indices
    (plainModel f indices.length physical.times nfilter physical.baseline coefficient intercept)
    rfl rfl rfl rfl rfl rfl
  simp only [plainModel] at hprefix hbind hcurrent ⊢
  simp [readBranch, ensure, tensorAt, personal_writer_waveform_field,
    hselected, hprefix, hcurrent, hrecipe, hbind]

/-- Actual erp_target_recipe pooled fields. No fitting or optimizer execution
is claimed; the numeric state is a complete from_state artifact. -/
def pooledFields (width : ℕ) : ERPFields := {
  name := "pooled_lda", family := .pooled, inputStatistics := "target_prefix", buffers := "source",
  epochs := .integer 1, batchSize := .integer 64, lr := .number (1 / 1000),
  weightDecay := .number (1 / 10000), maxUpdates := .null, seed := .integer 0,
  refit := false, temperature := 1, width := width, kind := none, rank := none }

theorem pooled_fields_satisfy_individual_checks (width : ℕ) (hw : 0 < width) :
    Conditions (pooledFields width) := by
  refine ⟨?_, Or.inr rfl, Or.inl rfl, ?_, hw, ?_, Or.inl rfl, Or.inr ⟨by simp [pooledFields], rfl, rfl⟩⟩
  · exact (show Nonblank "pooled_lda" from rfl)
  · norm_num [pooledFields]
  · exact ⟨1, 64, 1 / 1000, by norm_num [pooledFields, numericScalar],
      by norm_num [pooledFields, numericScalar], rfl, by norm_num, by norm_num, by norm_num⟩

def fixturePhysical : Physical := ⟨3, 5, .text "none"⟩
def fixtureCoefficient : Fin 4 → ℝ := fun i => (i.val : ℝ) + 1
def fixtureModel : Model := plainModel (pooledFields 2) 2 5 4 (.text "none") fixtureCoefficient (-1)
def fixtureState : NamedState := writePersonalERP fixtureModel (selectedTensor [0, 2]) []
def fixtureRecipe : Recipe := (pooledFields 2).recipe
def fixtureBranch (physical : Physical := fixturePhysical) (kwargs : Fields := recordFields (pooledFields 2).raw)
    (state : NamedState := fixtureState) (kind : String := "target_only_structured_erp")
    (sourceFlag : Value := .boolean false) : Check Bound :=
  readBranch fixturePrimitives fixtureReference physical kind sourceFlag kwargs
    (plainRecord .pooled 2 4 2 5 (.text "none")) state

theorem legal_actual_ERP_personal_branch_nonempty :
    fixtureBranch = .ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ := by
  exact plain_personal_branch_from_individual_fields fixturePrimitives fixtureReference
    (pooledFields 2) (pooled_fields_satisfy_individual_checks 2 (by norm_num)) fixturePhysical [0, 2] 4
    fixtureCoefficient (-1) (by decide) (by simp) (by simp [fixturePhysical])
    (by decide) (Or.inl (by decide)) (by norm_num [fixturePhysical])
    (by norm_num [fixturePhysical, pooledFields]) (by norm_num)

theorem actual_ERP_recipe_record_omits_default_width :
    (recordFields (pooledFields 1).raw).get "erp_bin_width" = none := by
  simp [recordFields, rawFields, ERPFields.raw, pooledFields, Fields.ofList, Fields.get,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem actual_ERP_recipe_record_preserves_nondefault_width :
    (recordFields (pooledFields 2).raw).get "erp_bin_width" = some (.integer 2) := by
  simp [recordFields, rawFields, ERPFields.raw, pooledFields, Fields.ofList, Fields.get,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

def defaultWidth : Check Value := do
  let r ← readArgs (Fields.ofList [("name", .text "pooled_lda")])
  pure r.erpBinWidth

theorem missing_width_reconstructs_one : defaultWidth = .ok (.integer 1) := by
  rw [defaultWidth, missing_keywords_use_actual_defaults]
  rfl

theorem boolean_width_rejected :
    validateERPRecipe { (pooledFields 2).raw with erpBinWidth := .boolean true } =
      .error (.sourceReject "erp_bin_width") := by
  have hn : Nonblank "pooled_lda" := rfl
  simp [validateERPRecipe, pooledFields, ERPFields.raw, non_boolean_real_reads,
    readFamily, Family.text, readTextValue, hn, ensure, positiveInt]

theorem enabled_dropout_rejected :
    validateERPRecipe { (pooledFields 2).raw with dropout := .text "checkpoint" } =
      .error (.sourceReject "ERP frozen treatment") := by
  have hn : Nonblank "pooled_lda" := rfl
  simp [validateERPRecipe, pooledFields, ERPFields.raw, non_boolean_real_reads,
    readFamily, Family.text, readTextValue, hn, ensure]

theorem duplicate_selection_rejected : readSelected 3 (selectedTensor [0, 0]) =
    .error (.sourceReject "ERP selected unique/range") := by
  simp [readSelected, selectedTensor, encodeNats, valuesToList, valuesOfList, decodeNats, ensure]

theorem selection_permutation_is_preserved :
    readSelected 3 (selectedTensor [2, 0]) = .ok [2, 0] := by
  exact selected_indices_preserve_order 3 [2, 0] (by decide) (by simp) (by simp)

theorem fixture_inner_fields_construct :
    CurrentERPState.fromState fixturePrimitives fixtureReference (plainRecord .pooled 2 4 2 5 (.text "none"))
      fixtureModel.state = .ok fixtureModel := by
  have h := plain_family_constructs fixturePrimitives fixtureReference .pooled 2 4 2 5 (.text "none")
    fixtureCoefficient (-1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by decide) (Or.inl (by decide))
  have h' : ERPState.fromState fixturePrimitives fixtureReference
      (plainRecord .pooled 2 4 2 5 (.text "none")) fixtureModel.state = .ok fixtureModel := h
  rw [CurrentERPState.fromState, h']
  exact CurrentERPState.other_families_current_guard_preserves fixtureModel
    (by simp [fixtureModel, plainModel, pooledFields])

theorem fixture_selected_field_reads :
    tensorAt fixtureState "waveform_channels" = .ok (selectedTensor [0, 2]) := by
  simp [fixtureState, tensorAt, personal_writer_waveform_field]

theorem fixture_ERP_fields_read : erpFields fixtureState = fixtureModel.state := by
  exact personal_prefix_parser_recovers_fields fixtureModel (selectedTensor [0, 2]) [] (by rfl)

theorem physical_time_mismatch_rejected :
    fixtureBranch ⟨3, 6, .text "none"⟩ = .error (.sourceReject "ERP state recipe physical contract") := by
  have hr := canonical_record_constructor (pooledFields 2) (pooled_fields_satisfy_individual_checks 2 (by norm_num))
  have hs := selected_indices_preserve_order 3 [0, 2] (by decide) (by simp) (by simp)
  simp only [fixtureBranch]
  simp [readBranch, ensure, fixture_selected_field_reads, fixture_ERP_fields_read,
    fixture_inner_fields_construct, hr, hs]
  simp [bindERP, fixtureModel, plainModel, ensure]

theorem recipe_width_mismatch_rejected :
    fixtureBranch fixturePhysical (recordFields (pooledFields 1).raw) =
      .error (.sourceReject "ERP state recipe physical contract") := by
  have hr := canonical_record_constructor (pooledFields 1) (pooled_fields_satisfy_individual_checks 1 (by norm_num))
  have hs := selected_indices_preserve_order 3 [0, 2] (by decide) (by simp) (by simp)
  simp only [fixtureBranch]
  simp [readBranch, ensure, fixturePhysical, fixture_selected_field_reads, fixture_ERP_fields_read,
    fixture_inner_fields_construct, hr, hs]
  simp [bindERP, fixtureModel, plainModel, ERPFields.recipe, pooledFields, ensure]

/-- A source-shaped decoded model for the role gate, not a claim that changing
an old fixed record creates genuine source fitting. Independent receipt/source
construction is a separate obligation; the role theorem above is general. -/
def roleModel (role : Role) : Model :=
  ⟨⟨.temporal, 1, 4, .text "none", some { (fixedTemporalModel fixturePrimitives fixtureFixedInput) with role := role }⟩,
    1, 1, fixtureTemporalState, none, none⟩

def temporalFields (kind : String) (rank : ℕ) : ERPFields :=
  { pooledFields 1 with name := "temporal_lda", family := .temporal, kind := some kind, rank := some rank }

def temporalRecipe (kind : String) (rank : ℕ) : Recipe := (temporalFields kind rank).recipe

theorem temporal_fields_satisfy_individual_checks (kind : String) (rank : ℕ)
    (hk : kind = "fixed_cosine" ∨ kind = "source_supervised") (hr : 0 < rank) :
    Conditions (temporalFields kind rank) := by
  refine ⟨?_, Or.inr rfl, Or.inl rfl, ?_, by norm_num [temporalFields, pooledFields], ?_,
    Or.inl rfl, Or.inl ⟨rfl, rfl, kind, rank, rfl, rfl, hk, hr⟩⟩
  · exact (show Nonblank "temporal_lda" from rfl)
  · norm_num [temporalFields, pooledFields]
  · exact ⟨1, 64, 1 / 1000, by norm_num [temporalFields, pooledFields, numericScalar],
      by norm_num [temporalFields, pooledFields, numericScalar], rfl,
      by norm_num, by norm_num, by norm_num⟩

theorem temporal_recipe_is_source_parsed (kind : String) (rank : ℕ)
    (hk : kind = "fixed_cosine" ∨ kind = "source_supervised") (hr : 0 < rank) :
    fromKwargs (recordFields (temporalFields kind rank).raw) = .ok (temporalRecipe kind rank) :=
  canonical_record_constructor _ (temporal_fields_satisfy_individual_checks kind rank hk hr)

def fixedTemporalModel : Model :=
  ⟨⟨.temporal, 1, 1, .text "none", some (ERPState.fixedTemporalModel fixturePrimitives fixtureFixedInput)⟩,
    1, 1, fixtureTemporalState, none, none⟩

theorem fixed_temporal_branch_nonempty :
    readBranch fixturePrimitives fixtureReference ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false)
      (recordFields (temporalFields "fixed_cosine" 1).raw)
      (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1)
      (writePersonalERP fixedTemporalModel (selectedTensor [0]) []) =
      .ok ⟨temporalRecipe "fixed_cosine" 1, [0], fixedTemporalModel⟩ := by
  have hi : CurrentERPState.fromState fixturePrimitives fixtureReference
      (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1) fixtureTemporalState = .ok fixedTemporalModel := by
    rw [CurrentERPState.fromState, temporal_nonempty_construct]
    exact CurrentERPState.other_families_current_guard_preserves fixedTemporalModel
      (by simp [fixedTemporalModel])
  have hp := personal_prefix_parser_recovers_fields fixedTemporalModel (selectedTensor [0]) [] (by rfl)
  have hs := selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  have hr := temporal_recipe_is_source_parsed "fixed_cosine" 1 (Or.inl rfl) (by norm_num)
  have hb := bind_temporal_from_fields (temporalRecipe "fixed_cosine" 1) ⟨1, 1, .text "none"⟩ [0]
    fixedTemporalModel (ERPState.fixedTemporalModel fixturePrimitives fixtureFixedInput)
    rfl rfl rfl rfl rfl rfl
    (by simp [temporalRecipe, temporalFields, ERPFields.recipe, ERPState.fixedTemporalModel]) rfl
  simp only [fixedTemporalModel] at hp hb hi ⊢
  simp [readBranch, ensure, tensorAt, personal_writer_waveform_field, hs, hp, hi, hr]
  simpa [ERPState.fixedTemporalModel, modelKind] using hb

def weakOptimizerFields : ERPFields :=
  { pooledFields 2 with
    epochs := .boolean true
    batchSize := .boolean true
    lr := .boolean true
    weightDecay := .text "unchecked_by_constructor"
    seed := .object .nil }

/-- This intentionally preserves the actual weak optimizer checks. It is a
constructor theorem only: later training/optimizer semantics are not inferred. -/
theorem numeric_boolean_optimization_is_source_accepted :
    validateERPRecipe weakOptimizerFields.raw = .ok weakOptimizerFields.recipe := by
  apply ERP_constructor_from_individual_fields
  refine ⟨?_, Or.inr rfl, Or.inl rfl, ?_, by norm_num [weakOptimizerFields, pooledFields], ?_,
    Or.inl rfl, Or.inr ⟨by simp [weakOptimizerFields, pooledFields], rfl, rfl⟩⟩
  · exact (show Nonblank "pooled_lda" from rfl)
  · norm_num [weakOptimizerFields, pooledFields]
  · exact ⟨1, 1, 1, rfl, rfl, rfl, by norm_num, by norm_num, by norm_num⟩

theorem source_one_is_not_true :
    bindERP (temporalRecipe "source_supervised" 1) "source_temporal_target_erp" (.integer 1)
      ⟨1, 1, .text "none"⟩ [0] (roleModel .source) = .error (.sourceReject "ERP strict source role") := by
  simp [bindERP, temporalRecipe, temporalFields, pooledFields, ERPFields.recipe, roleModel, ensure,
    usesSource, modelKind, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem source_free_zero_is_not_false :
    bindERP fixtureRecipe "target_only_structured_erp" (.integer 0) fixturePhysical [0, 2] fixtureModel =
      .error (.sourceReject "ERP strict source role") := by
  simp [bindERP, fixtureRecipe, fixturePhysical, fixtureModel, plainModel, ERPFields.recipe,
    pooledFields, ensure, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem wrong_model_kind_rejected :
    bindERP (temporalRecipe "source_supervised" 1) "target_only_structured_erp" (.boolean true)
      ⟨1, 1, .text "none"⟩ [0] (roleModel .source) = .error (.sourceReject "ERP strict source role") := by
  simp [bindERP, temporalRecipe, temporalFields, pooledFields, ERPFields.recipe, roleModel, ensure,
    usesSource, modelKind, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem wrong_temporal_recipe_kind_rejected :
    bindERP (temporalRecipe "fixed_cosine" 1) "source_temporal_target_erp" (.boolean true)
      ⟨1, 1, .text "none"⟩ [0] (roleModel .source) = .error (.sourceReject "ERP temporal kind/rank") := by
  simp [bindERP, temporalRecipe, temporalFields, pooledFields, ERPFields.recipe, roleModel, ensure,
    usesSource, modelKind, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem unknown_ERP_tensor_is_forwarded :
    erpFields [("erp.unrecognized", realTensor [1] [9])] = [("unrecognized", realTensor [1] [9])] := by
  rfl

theorem prefixed_unknown_key_reaches_inner_rejection :
    fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none"))
      (erpFields ((writePersonalERP (fixturePlain .pooled) (selectedTensor [0]) []) ++
        [("erp.unrecognized", realTensor [1] [9])])) = .error (.sourceReject "ERP state keys") := by
  rw [erp_fields_append, personal_prefix_parser_recovers_fields _ _ _ (by rfl), unknown_ERP_tensor_is_forwarded]
  simp [fromState, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, required, readFamily, Family.text, positiveInt, readGeometry, encodeNats,
    fixturePlain, fixtureLinearState, exactStateKeys, stateKeys, expectedKeys,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, ensure]

end
end N2P3.PersonalERPRecipe
