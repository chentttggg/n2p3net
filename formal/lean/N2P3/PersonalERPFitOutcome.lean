import N2P3.PersonalERPActualConsumer

/-!
# ERP fit outcomes including nonfinite arrays and their actual reader order

Sources: structured_erp.py:76-123,156-230; calibration_candidates.py:311-332;
calibrated.py:550-551,619-642,660-661. Unlike the retained finite-output Fit
slice, output arrays below contain finite representatives, +Inf, -Inf or NaN.
These are structural IEEE observations, not definitions of IEEE arithmetic or
IEEE equality (in particular the NaN token may be structurally copied).

Estimator and packing remain separate numerical operations. Neither has an
invented output-finiteness guard. Export/prefix/clone retain the observed cells.
The current inner reader checks metadata/geometry/keys, casts/copies ALL arrays,
then checks finiteness, then shapes/current xDAWN guards. The actual Scalar
kwargs constructor/binding follows inner success. The outer Torch finite gate
is last, after that branch, and also sees non-ERP state entries.

The module starts at already selected/aligned fit operands or the numeric
consumer subroute. Full physical ingress/identity/authority and estimator
formulas are still pending. Array casts/copies retain single-operation fault
boundaries; transport definitions are decoded observations, not Torch/NumPy
byte/storage/alias or resource-failure refinement.
-/
namespace N2P3.PersonalERPFitOutcome
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPTransport N2P3.PersonalERPScalar
open N2P3.PersonalERPFit (NumericBatch)
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Cell where
  | finite (value : ℝ)
  | positiveInfinity | negativeInfinity | nan

def Cell.isFinite : Cell → Bool
  | .finite _ => true
  | _ => false

/-- Used only after the explicit all-finite guard. Nonfinite fallback0 is
unreachable there and does not replace a source NaN/Inf with a finite number. -/
def Cell.finiteRepresentative : Cell → ℝ
  | .finite value => value
  | _ => 0

structure Array64 where
  shape : List ℕ
  cells : List Cell
  count : cells.length = shape.foldr (· * ·) 1

abbrev Arrays := List (String × Array64)

def arrayFinite (array : Array64) : Bool := array.cells.all Cell.isFinite
def arraysFinite (arrays : Arrays) : Bool := arrays.all (fun entry => arrayFinite entry.2)

def decodeArray (array : Array64) : Tensor :=
  realTensor array.shape (array.cells.map Cell.finiteRepresentative)

def decodeArrays (arrays : Arrays) : NamedState := arrays.map (fun entry => (entry.1, decodeArray entry.2))

def exactArrayKeys (arrays : Arrays) (expected : List String) : Prop :=
  arrays.map Prod.fst ⊆ expected ∧ expected ⊆ arrays.map Prod.fst

theorem all_finite_cells_reconstruct (cells : List Cell)
    (h : cells.all Cell.isFinite = true) :
    cells.map (fun cell => Cell.finite cell.finiteRepresentative) = cells := by
  induction cells with
  | nil => rfl
  | cons cell tail ih =>
    have ha : Cell.isFinite cell = true ∧ tail.all Cell.isFinite = true := by simpa using h
    cases cell <;> simp_all [Cell.isFinite, Cell.finiteRepresentative]

def fromFiniteArray (array : PersonalERPTransport.Array64) : Array64 :=
  ⟨array.shape, array.cells.map Cell.finite, by simpa using array.cellCount⟩

def fromFiniteArrays (arrays : PersonalERPTransport.ArrayFields) : Arrays :=
  arrays.map (fun entry => (entry.1, fromFiniteArray entry.2))

theorem decoded_finite_array_preserves_fields (array : PersonalERPTransport.Array64) :
    decodeArray (fromFiniteArray array) = numpyTensor array := by
  simp [decodeArray, fromFiniteArray, numpyTensor, Function.comp_def, Cell.finiteRepresentative]

structure Payload where
  arrays : Arrays
  observedRank : Option ℕ
  effectiveNfilter : Option ℕ

structure Numerics where
  castFloat64 : (rows channels times : ℕ) → NumericBatch rows channels times → Check (NumericBatch rows channels times)
  binMean : (rows channels times width : ℕ) → ERPFeatures.Batch rows channels times →
    Check (NumericBatch rows channels (times / width))
  temporalMatmul : (rows channels times rank : ℕ) → ERPFeatures.Batch rows channels times → Tensor →
    Check (NumericBatch rows channels rank)
  estimate : (rows channels times : ℕ) → Header → NumericBatch rows channels times → (Fin rows → Bool) → Check Payload
  packFloat64 : Payload → Check Payload

def inputStage (ops : Numerics) {R C T : ℕ} (header : Header)
    (values : ERPFeatures.Batch R C T) : Check (NumericBatch R C (T / header.binWidth)) := do
  let casted ← ops.castFloat64 R C T (.finite values)
  match casted with
  | .nonfinite => .error (.sourceReject "StructuredERP finite input geometry")
  | .finite values => do
    ensure (0 < C ∧ header.binWidth ≤ T) "StructuredERP finite input geometry"
    ops.binMean R C T header.binWidth values

def temporalStage (ops : Numerics) {R C T : ℕ} (temporal : TemporalModel)
    (values : NumericBatch R C T) : Check (NumericBatch R C temporal.rank) := do
  let casted ← ops.castFloat64 R C T values
  match casted with
  | .nonfinite => .error (.sourceReject "StructuredERP temporal projection geometry/finite")
  | .finite values => do
    ensure (T = temporal.times) "StructuredERP temporal projection geometry/finite"
    ops.temporalMatmul R C T temporal.rank values temporal.matrix

def fitStages (ops : Numerics) {R C T : ℕ} (header : Header)
    (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool) : Check Payload := do
  let binned ← inputStage ops header values
  PersonalERPFit.structuredLabelGate labels
  let estimated ← match header.temporal with
    | none => ops.estimate R C (T / header.binWidth) header binned labels
    | some temporal => do
      let projected ← temporalStage ops temporal binned
      ops.estimate R C temporal.rank header projected labels
  ops.packFloat64 estimated

theorem plain_fit_returns_any_packed_cells (ops : Numerics) {R C T : ℕ}
    (header : Header) (values casted : ERPFeatures.Batch R C T) (labels : Fin R → Bool)
    (binned : NumericBatch R C (T / header.binWidth)) (estimated packed : Payload)
    (ht : header.temporal = none)
    (hc : ops.castFloat64 R C T (.finite values) = .ok (.finite casted))
    (hg : 0 < C ∧ header.binWidth ≤ T)
    (hm : ops.binMean R C T header.binWidth casted = .ok binned)
    (hl : PersonalERPFit.bothClasses labels ∧ 2 ≤ PersonalERPFit.classCount labels false ∧
      2 ≤ PersonalERPFit.classCount labels true)
    (he : ops.estimate R C (T / header.binWidth) header binned labels = .ok estimated)
    (hp : ops.packFloat64 estimated = .ok packed) :
    fitStages ops header values labels = .ok packed := by
  simp [fitStages, inputStage, ht, hc, ensure, hg, hm, PersonalERPFit.structuredLabelGate, hl, he, hp]

structure Artifact where
  header : Header
  channels : ℕ
  times : ℕ
  numeric : Payload

/-- Array cells do not participate in StructuredERP.export_state's record. -/
def metadataModel (artifact : Artifact) : Model :=
  ⟨artifact.header, artifact.channels, artifact.times, [],
    artifact.numeric.observedRank, artifact.numeric.effectiveNfilter⟩

def artifactRecord (artifact : Artifact) : Fields := exportRecord (metadataModel artifact)

def fitArtifact (p : TemporalPrimitives) (ops : Numerics) {R C T : ℕ}
    (family width baseline : Value) (basis : Option TemporalModel)
    (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool) : Check Artifact := do
  let header ← commonHeader p (PersonalERPFit.rawConstructorFields family width (.integer 4) baseline) basis
  let numeric ← fitStages ops header values labels
  pure ⟨header, C, T, numeric⟩

inductive StateTensor where
  | numeric (array : Array64)
  | selector (tensor : Tensor)

abbrev State := List (String × StateTensor)

def producer (arrays : Arrays) (selected : Tensor) : State :=
  arrays.map (fun entry => ("erp." ++ entry.1, .numeric entry.2)) ++ [("waveform_channels", .selector selected)]

def cloneTensor : StateTensor → StateTensor
  | .numeric array => .numeric ⟨array.shape, array.cells.map id, by simpa using array.count⟩
  | .selector tensor => .selector (copyTensor tensor)

def cloneState (state : State) : State := state.map (fun entry => (entry.1, cloneTensor entry.2))

theorem clone_preserves_every_observed_cell (state : State) : cloneState state = state := by
  induction state with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name, tensor⟩
    cases tensor <;> simp [cloneState, cloneTensor, copyTensor_preserves] at ih ⊢ <;> exact ih

def lookupState (name : String) : State → Option StateTensor
  | [] => none
  | (key, tensor) :: tail => if name = key then some tensor else lookupState name tail

def selectedTensor (state : State) : Result Tensor :=
  match lookupState "waveform_channels" state with
  | none => .error (.inherited (.fieldFailure "ERP missing tensor"))
  | some (.selector tensor) => .ok tensor
  | some (.numeric _) => .error (.unrefined "numeric output represented as selector tensor")

def consumerArrays : State → Result Arrays
  | [] => .ok []
  | (name, tensor) :: tail =>
    match stripERP name with
    | none => consumerArrays tail
    | some key => do
      let array ← match tensor with
        | .numeric array => .ok array
        | .selector _ => .error (.unrefined "selector representation under ERP array key")
      let rest ← consumerArrays tail
      pure ((key, array) :: rest)

def castAll (castCopy : String → Array64 → Check Array64) : Arrays → Check Arrays
  | [] => .ok []
  | (name, array) :: tail => do
    let copied ← castCopy name array
    let rest ← castAll castCopy tail
    pure ((name, copied) :: rest)

def finiteStateTensor : StateTensor → Bool
  | .numeric array => arrayFinite array
  | .selector tensor => tensor.values.all (fun value => (numericScalar value).isSome)

def finalTorchFinite (state : State) : Check Unit :=
  ensure (state.all (fun entry => finiteStateTensor entry.2) = true) "calibrated predictor contains nonfinite state"

/-- Explicit source tail after all arrays are converted and finite-checked. -/
def finishFinite (ref : ReferencePrimitives) (record : Fields) (header : Header)
    (c t : ℕ) (state : NamedState) : Check Model := do
  let model ← match header.family with
    | .xdawn => do
      let rank ← positiveInt (fieldDefault record "observed_rank") "xDAWN rank/effective"
      let effective ← positiveInt (fieldDefault record "effective_nfilter") "xDAWN rank/effective"
      ensure (rank ≤ c ∧ effective = min header.nfilter rank) "xDAWN rank/effective"
      let v := 2 * effective
      let covariance ← tensorAt state "reference"
      if covariance.shape = [2 * v, 2 * v] then
        ensure (ref.symmetryAllclose covariance = true ∧ ref.eigenvaluesPositive covariance = true)
          "xDAWN reference symmetry/positive eigenvalues"
      else pure ()
      shapeAt state "filters" [v, c]
      shapeAt state "prototypes" [v, t / header.binWidth]
      shapeAt state "reference" [2 * v, 2 * v]
      shapeAt state "coefficient" [(2 * v) * (2 * v + 1) / 2]
      shapeAt state "intercept" []
      pure ⟨header, c, t, state, some rank, some effective⟩
    | _ => do
      match header.temporal with
      | some temporal =>
        let matrix ← tensorAt state "temporal_basis"
        ensure (temporal.times = t ∧ sameNumericArray matrix temporal.matrix) "temporal duplicate basis equality"
      | none => pure ()
      shapeAt state "coefficient" [featureWidth header c t]
      shapeAt state "intercept" []
      match header.temporal with
      | some temporal => shapeAt state "temporal_basis" temporal.matrix.shape
      | none => pure ()
      pure ⟨header, c, t, state, none, none⟩
  CurrentERPState.currentGuard model

def readOutcome (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (castCopy : String → Array64 → Check Array64) (record : Fields) (arrays : Arrays) : Check Model := do
  ensure (fieldDefault record "schema" = .text "n2p3_structured_erp/1" ∧
    fieldDefault record "feature_order" = .text "channel_prime") "ERP schema/order"
  let temporal ← optionalTemporal p record
  let header ← commonHeader p record temporal
  let (c, t) ← readGeometry record
  ensure (0 < t / header.binWidth ∧ PersonalAdmission.PyEq (fieldDefault record "discarded_tail_samples")
    (.integer (t % header.binWidth))) "ERP binning"
  ensure (exactArrayKeys arrays (expectedKeys header)) "ERP state keys"
  let copied ← castAll castCopy arrays
  ensure (arraysFinite copied = true) "ERP state contains nonfinite values"
  finishFinite ref record header c t (decodeArrays copied)

def consumeOutcome (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (castCopy : String → Array64 → Check Array64) (physical : Physical) (kind : String) (flag : Value)
    (kwargs record : Fields) (state : State) : Result Bound := do
  let restored := cloneState state
  liftCheck (ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind")
  let tensor ← selectedTensor restored
  let indices ← liftCheck (readSelected physical.channels tensor)
  let arrays ← consumerArrays restored
  let model ← liftCheck (readOutcome p ref castCopy (copyFields record) arrays)
  let recipe ← fromKwargsScalar s (copyFields kwargs)
  let bound ← liftCheck (bindERP recipe kind flag physical indices model)
  liftCheck (finalTorchFinite restored)
  pure bound

def identityCopy : String → Array64 → Check Array64 := fun _ array => .ok array

def boundaryCoefficient : Array64 := ⟨[1], [.positiveInfinity], rfl⟩
def boundaryIntercept : Array64 := ⟨[], [.negativeInfinity], rfl⟩
def boundaryPayload : Payload :=
  ⟨[("coefficient", boundaryCoefficient), ("intercept", boundaryIntercept)], none, none⟩
def boundaryHeader : Header := ⟨.pooled, 1, 4, .text "none", none⟩
def boundaryArtifact : Artifact := ⟨boundaryHeader, 1, 1, boundaryPayload⟩
def boundaryValues : ERPFeatures.Batch 4 1 1 :=
  fun i _ _ => ![-((2 : ℝ) ^ 530)⁻¹, ((2 : ℝ) ^ 530)⁻¹, ((2 : ℝ) ^ 20)⁻¹, ((2 : ℝ) ^ 20)⁻¹] i
def boundaryLabels : Fin 4 → Bool := ![false, false, true, true]

/-- Declared individual operation responses. The independent actual source
probe supplies the overflow witness; this definition computes no IEEE fit. -/
def boundaryNumerics : Numerics := {
  castFloat64 := fun _ _ _ values => .ok values
  binMean := fun _ _ _ width values => .ok (.finite (fun i => ERPFeatures.completeBins width (values i)))
  temporalMatmul := fun _ _ _ _ _ _ => .ok .nonfinite
  estimate := fun _ _ _ _ _ _ => .ok boundaryPayload
  packFloat64 := fun payload => .ok payload }

theorem boundary_has_two_rows_per_class :
    PersonalERPFit.bothClasses boundaryLabels ∧ 2 ≤ PersonalERPFit.classCount boundaryLabels false ∧
      2 ≤ PersonalERPFit.classCount boundaryLabels true := by
  refine ⟨⟨⟨0, rfl⟩, ⟨2, rfl⟩⟩, ?_, ?_⟩ <;> decide

theorem fit_can_return_nonfinite_packed_arrays :
    fitStages boundaryNumerics boundaryHeader boundaryValues boundaryLabels = .ok boundaryPayload ∧
      arraysFinite boundaryPayload.arrays = false := by
  constructor
  · apply plain_fit_returns_any_packed_cells
    · rfl
    · rfl
    · norm_num [boundaryHeader]
    · rfl
    · exact boundary_has_two_rows_per_class
    · rfl
    · rfl
  · rfl

theorem constructor_fit_returns_nonfinite_artifact :
    fitArtifact fixturePrimitives boundaryNumerics (.text "pooled_lda") (.integer 1) (.text "none") none
      boundaryValues boundaryLabels = .ok boundaryArtifact := by
  have hh := PersonalERPFit.common_header_from_raw_individual_fields fixturePrimitives
    (.text "pooled_lda") (.integer 1) (.integer 4) (.text "none") .pooled 1 4 .absent
    rfl rfl rfl (by norm_num) (by norm_num) (by simp)
    (by simp [PersonalERPFit.TemporalConstructorFields.Conditions])
  simp only [PersonalERPFit.TemporalConstructorFields.basis] at hh
  change commonHeader _ _ none = .ok boundaryHeader at hh
  simp [fitArtifact, hh, fit_can_return_nonfinite_packed_arrays.1, boundaryArtifact]

theorem boundary_current_reader_rejects_nonfinite_after_keys :
    readOutcome fixturePrimitives fixtureReference identityCopy (artifactRecord boundaryArtifact)
      boundaryPayload.arrays = .error (.sourceReject "ERP state contains nonfinite values") := by
  simp [readOutcome, artifactRecord, metadataModel, boundaryArtifact, boundaryHeader, boundaryPayload,
    exportRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal, commonHeader, readFamily,
    Family.text, readGeometry, positiveInt, required, ensure, exactArrayKeys, expectedKeys,
    castAll, identityCopy, arraysFinite, arrayFinite,
    boundaryCoefficient, boundaryIntercept, Cell.isFinite, PersonalAdmission.PyEq,
    PersonalAdmission.normalizeValue, encodeNats]

theorem producer_retains_selector_for_any_output_cells (arrays : Arrays) (selected : Tensor) :
    lookupState "waveform_channels" (producer arrays selected) = some (.selector selected) := by
  induction arrays with
  | nil => simp [producer, lookupState]
  | cons entry tail ih =>
    rcases entry with ⟨name, array⟩
    simp [producer, lookupState, waveform_name_cannot_have_ERP_prefix] at ih ⊢
    exact ih

theorem producer_consumer_retains_nonfinite_cells (arrays : Arrays) (selected : Tensor) :
    consumerArrays (producer arrays selected) = .ok arrays := by
  induction arrays with
  | nil => simp [producer, consumerArrays, stripERP]
  | cons entry tail ih =>
    rcases entry with ⟨name, array⟩
    simp only [producer, List.map_cons, List.cons_append, consumerArrays, prefixed_name_strips_once]
    change (do let rest ← consumerArrays (producer tail selected); pure ((name, array) :: rest)) = _
    rw [ih]
    rfl

/-- General continuation identity for arbitrary output cells and arbitrary
scalar kwargs; no inner-reader/recipe-success assumption or finite-output
domain restriction. Source error priority is determined by the explicit bind. -/
theorem producer_actual_consumer_continuation
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (castCopy : String → Array64 → Check Array64) (physical : Physical) (flag : Value)
    (kwargs record : Fields) (arrays : Arrays) (tensor : Tensor) :
    consumeOutcome s p ref castCopy physical "target_only_structured_erp" flag kwargs record (producer arrays tensor) =
      (liftCheck (readSelected physical.channels tensor)).bind (fun indices =>
        (liftCheck (readOutcome p ref castCopy record arrays)).bind (fun model =>
          (fromKwargsScalar s kwargs).bind (fun recipe =>
            (liftCheck (bindERP recipe "target_only_structured_erp" flag physical indices model)).bind (fun bound =>
              (liftCheck (finalTorchFinite (producer arrays tensor))).map (fun _ => bound))))) := by
  simp [consumeOutcome, clone_preserves_every_observed_cell, ensure, selectedTensor,
    producer_retains_selector_for_any_output_cells, producer_consumer_retains_nonfinite_cells,
    copyFields_preserves, liftCheck]

theorem nonfinite_ERP_output_rejected_before_any_scalar_kwargs (s : Primitives) (kwargs : Fields) :
    consumeOutcome s fixturePrimitives fixtureReference identityCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (artifactRecord boundaryArtifact)
      (producer boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.inherited (.sourceReject "ERP state contains nonfinite values")) := by
  rw [producer_actual_consumer_continuation]
  have hs := selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [liftCheck, hs, boundary_current_reader_rejects_nonfinite_after_keys]

def fitThenOutcome (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (ops : Numerics) (castCopy : String → Array64 → Check Array64) {R C T : ℕ}
    (recipe : Recipe) (baseline : Value) (basis : Option TemporalModel) (physical : Physical)
    (selected : Tensor) (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool) : Result Bound := do
  let artifact ← liftCheck (fitArtifact p ops recipe.raw.initialization recipe.raw.erpBinWidth baseline basis values labels)
  let temporal := temporalTransfer (metadataModel artifact)
  let transferred := decide (recipe.raw.initialization = .text "source")
  consumeOutcome s p ref castCopy physical (producedKind transferred false temporal true)
    (sourceFlag transferred temporal) (recordFields recipe.raw) (artifactRecord artifact)
    (producer artifact.numeric.arrays selected)

/-- General raw-field/individual-operation composition with ANY packed cells.
It neither assumes nor concludes packed finiteness or successful loading. -/
theorem raw_plain_fit_to_outcome_consumer_from_individual_operations
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives) (ops : Numerics)
    (castCopy : String → Array64 → Check Array64) {R C T : ℕ}
    (recipe : Recipe) (family : Family) (width : ℕ) (baseline : Value) (physical : Physical)
    (selected : Tensor) (values casted : ERPFeatures.Batch R C T) (labels : Fin R → Bool)
    (binned : NumericBatch R C (T / width)) (estimated packed : Payload)
    (hfamily : recipe.raw.initialization = .text family.text)
    (hwidth : recipe.raw.erpBinWidth = .integer width) (hw : 0 < width)
    (hf : family ≠ .temporal) (hb : family ≠ .toeplitz ∨ baseline = .text "none")
    (hc : ops.castFloat64 R C T (.finite values) = .ok (.finite casted))
    (hg : 0 < C ∧ width ≤ T) (hm : ops.binMean R C T width casted = .ok binned)
    (hl : PersonalERPFit.bothClasses labels ∧ 2 ≤ PersonalERPFit.classCount labels false ∧
      2 ≤ PersonalERPFit.classCount labels true)
    (he : ops.estimate R C (T / width) ⟨family, width, 4, baseline, none⟩ binned labels = .ok estimated)
    (hp : ops.packFloat64 estimated = .ok packed) :
    fitThenOutcome s p ref ops castCopy recipe baseline none physical selected values labels =
      consumeOutcome s p ref castCopy physical "target_only_structured_erp" (.boolean false)
        (recordFields recipe.raw) (artifactRecord ⟨⟨family, width, 4, baseline, none⟩, C, T, packed⟩)
        (producer packed.arrays selected) := by
  have hh := PersonalERPFit.common_header_from_raw_individual_fields p recipe.raw.initialization
    recipe.raw.erpBinWidth (.integer 4) baseline family width 4 .absent hfamily hwidth rfl hw
    (by norm_num) hb hf
  simp only [PersonalERPFit.TemporalConstructorFields.basis] at hh
  have hs := plain_fit_returns_any_packed_cells ops ⟨family, width, 4, baseline, none⟩
    values casted labels binned estimated packed rfl hc hg hm hl he hp
  have hn : recipe.raw.initialization ≠ .text "source" := by
    rw [hfamily]
    intro heq
    exact every_erp_family_is_not_source_initialization family (Value.text.inj heq)
  simp [fitThenOutcome, fitArtifact, hh, hs, liftCheck, hn, metadataModel,
    temporalTransfer, producedKind, sourceFlag]

theorem selector_refusal_precedes_nonfinite_ERP_output (s : Primitives) (kwargs : Fields) :
    consumeOutcome s fixturePrimitives fixtureReference identityCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (artifactRecord boundaryArtifact)
      (producer boundaryPayload.arrays PersonalERPCurrentConsumer.floatingSelector) =
      .error (.inherited (.sourceReject "ERP selected dtype")) := by
  rw [producer_actual_consumer_continuation]
  simp [liftCheck, readSelected, PersonalERPCurrentConsumer.floatingSelector, ensure]

theorem nonempty_fit_returns_but_actual_consumer_rejects :
    fitArtifact fixturePrimitives boundaryNumerics (.text "pooled_lda") (.integer 1) (.text "none") none
      boundaryValues boundaryLabels = .ok boundaryArtifact ∧
    fitThenOutcome observedPrimitives fixturePrimitives fixtureReference boundaryNumerics identityCopy
      (pooledFields 1).recipe (.text "none") none ⟨1, 1, .text "none"⟩
      (PersonalERPRecipe.selectedTensor [0]) boundaryValues boundaryLabels =
      .error (.inherited (.sourceReject "ERP state contains nonfinite values")) := by
  refine ⟨constructor_fit_returns_nonfinite_artifact, ?_⟩
  simp only [fitThenOutcome, ERPFields.recipe, ERPFields.raw, pooledFields, Family.text, Nat.cast_one]
  rw [constructor_fit_returns_nonfinite_artifact]
  simpa [liftCheck, metadataModel, boundaryArtifact, boundaryHeader, temporalTransfer,
    producedKind, sourceFlag, ERPFields.raw, pooledFields, Family.text] using
    (nonfinite_ERP_output_rejected_before_any_scalar_kwargs observedPrimitives (recordFields (pooledFields 1).raw))

def failLaterCopy (name : String) (array : Array64) : Check Array64 :=
  if name = "intercept" then .error (.fieldFailure "intercept float64 copy exception") else .ok array

theorem later_array_cast_exception_precedes_earlier_nonfinite :
    readOutcome fixturePrimitives fixtureReference failLaterCopy (artifactRecord boundaryArtifact)
      boundaryPayload.arrays = .error (.fieldFailure "intercept float64 copy exception") := by
  simp [readOutcome, artifactRecord, metadataModel, boundaryArtifact, boundaryHeader, boundaryPayload,
    exportRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal, commonHeader, readFamily,
    Family.text, readGeometry, positiveInt, required, ensure, exactArrayKeys, expectedKeys,
    castAll, failLaterCopy, PersonalAdmission.PyEq,
    PersonalAdmission.normalizeValue, encodeNats]

def badKeyArrays : Arrays := [("wrong_coefficient", boundaryCoefficient), ("intercept", boundaryIntercept)]

theorem inner_keys_precede_array_conversion_and_finite_guard :
    readOutcome fixturePrimitives fixtureReference failLaterCopy (artifactRecord boundaryArtifact)
      badKeyArrays = .error (.sourceReject "ERP state keys") := by
  simp [readOutcome, artifactRecord, metadataModel, boundaryArtifact, boundaryHeader, boundaryPayload, badKeyArrays,
    exportRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal, commonHeader, readFamily,
    Family.text, readGeometry, positiveInt, required, ensure, exactArrayKeys, expectedKeys,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

def finiteFixtureArrays : Arrays := fromFiniteArrays PersonalERPCurrentConsumer.fixtureArrays
def nonERPExtraState : State :=
  producer finiteFixtureArrays (PersonalERPRecipe.selectedTensor [0]) ++ [("diagnostic", .numeric boundaryIntercept)]

theorem finite_fixture_outcome_reader :
    readOutcome fixturePrimitives fixtureReference identityCopy PersonalERPCurrentConsumer.fixtureRecord
      finiteFixtureArrays = .ok PersonalERPCurrentConsumer.fixtureModel := by
  simp [readOutcome, PersonalERPCurrentConsumer.fixtureRecord, PersonalERPCurrentConsumer.fixtureModel,
    plainModel, pooledFields, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, readFamily, Family.text, readGeometry, positiveInt, required, ensure, exactArrayKeys,
    expectedKeys, decodeArrays, decodeArray, finiteFixtureArrays, fromFiniteArrays,
    fromFiniteArray, PersonalERPCurrentConsumer.fixtureArrays, linearArrays,
    vectorArray, scalarArray, castAll, identityCopy, arraysFinite, arrayFinite,
    Cell.isFinite, Cell.finiteRepresentative, Function.comp_def, finishFinite, shapeAt, tensorAt,
    lookupTensor, featureWidth, CurrentERPState.currentGuard, realTensor, linearState,
    coefficientTensor, interceptTensor, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

theorem nonERP_extra_is_excluded_from_inner_arrays : consumerArrays nonERPExtraState = .ok finiteFixtureArrays := by
  simp [consumerArrays, nonERPExtraState, producer, finiteFixtureArrays, fromFiniteArrays,
    PersonalERPCurrentConsumer.fixtureArrays, linearArrays, stripERP]

theorem nonERP_extra_fails_only_final_state_finite_gate :
    finalTorchFinite nonERPExtraState = .error (.sourceReject "calibrated predictor contains nonfinite state") := by
  simp [finalTorchFinite, nonERPExtraState, producer, finiteStateTensor, boundaryIntercept,
    arrayFinite, Cell.isFinite, ensure]

theorem nonERP_extra_scalar_then_final_formula (s : Primitives) (kwargs : Fields) :
    consumeOutcome s fixturePrimitives fixtureReference identityCopy PersonalERPCurrentConsumer.fixturePhysical
      "target_only_structured_erp" (.boolean false) kwargs PersonalERPCurrentConsumer.fixtureRecord nonERPExtraState =
      (fromKwargsScalar s kwargs).bind (fun recipe =>
        (liftCheck (bindERP recipe "target_only_structured_erp" (.boolean false)
          PersonalERPCurrentConsumer.fixturePhysical [0] PersonalERPCurrentConsumer.fixtureModel)).bind (fun bound =>
          (liftCheck (finalTorchFinite nonERPExtraState)).map (fun _ => bound))) := by
  have hs : selectedTensor nonERPExtraState = .ok (PersonalERPRecipe.selectedTensor [0]) := by
    simp [selectedTensor, nonERPExtraState, producer, finiteFixtureArrays, fromFiniteArrays,
      PersonalERPCurrentConsumer.fixtureArrays, linearArrays, lookupState]
  have hr := selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [consumeOutcome, clone_preserves_every_observed_cell, copyFields_preserves, ensure,
    hs, PersonalERPCurrentConsumer.fixturePhysical, hr, liftCheck,
    nonERP_extra_is_excluded_from_inner_arrays, finite_fixture_outcome_reader]

theorem huge_scalar_exception_precedes_nonERP_extra_final_gate :
    consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      (Fields.ofList (rawFields PersonalERPActualConsumer.hugeActualRaw)) PersonalERPCurrentConsumer.fixtureRecord
      nonERPExtraState = .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  have hh := PersonalERPScalarFields.temperature_exception_from_single_numpy_atom observedPrimitives
    PersonalERPActualConsumer.hugeActualRaw .typeError rfl
    (by simp [PersonalERPActualConsumer.hugeActualRaw, ERPFields.raw, pooledFields]) rfl rfl
    observed_numpy_huge_typeerror
  rw [nonERP_extra_scalar_then_final_formula, kwargs_reconstruct_before_scalar, hh]
  rfl

theorem legal_scalar_reaches_nonERP_extra_final_gate :
    consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      (recordFields (pooledFields 1).raw) PersonalERPCurrentConsumer.fixtureRecord nonERPExtraState =
      .error (.inherited (.sourceReject "calibrated predictor contains nonfinite state")) := by
  have hs := PersonalERPScalarFields.canonical_kwargs_from_independent_fields_and_atoms observedPrimitives
    (pooledFields 1) (pooled_fields_satisfy_individual_checks 1 (by norm_num)) (by decide)
    (PersonalERPScalarFields.pooled_fixture_scalar_atoms 1)
  have hb := bind_plain_from_fields (pooledFields 1).recipe PersonalERPCurrentConsumer.fixturePhysical [0]
    PersonalERPCurrentConsumer.fixtureModel rfl rfl rfl rfl rfl rfl
  rw [nonERP_extra_scalar_then_final_formula, hs]
  simp [liftCheck, hb, nonERP_extra_fails_only_final_state_finite_gate]

end
end N2P3.PersonalERPFitOutcome
