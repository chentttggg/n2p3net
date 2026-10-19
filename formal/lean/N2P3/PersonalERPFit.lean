import N2P3.PersonalERPRecipe
import N2P3.PersonalERPAvailability
import N2P3.PersonalERPTransport
import N2P3.PersonalERPIdentity

/-!
# Actual ERP fitting dispatch, call operands, components and exported state

Source: calibration_candidates.py:504-693, especially 579-586,593,650-692,
311-332; StructuredERP.__init__/_input/fit/export_state.

Fixed obligations before implementation: derive ERP-only components from the
literal three dispatch tests and None initialization, carry the ordered
trainingSelection/erpFitInvocation operands into the numerical call, preserve
the two different label gates and requested/executed digests, construct every
candidate field outside the numerical primitive, and connect its actual state
property to Transport's per-array codec. No NoSimultaneousComponents premise
or already-correct candidate-returning oracle is used.

Five independent false alternatives are fixed: ERP creates an adapter; ERP
creates LDA; ERP fitting consumes every original channel; ERP labels use an
unsliced row order; initial >2 rows and two classes bypass the internal
StructuredERP requirement of two rows in EACH class.

This core begins after public quality ingress/raw-scope and temporal-resolver
entry. It does NOT certify physical row authority, provenance, source fitting,
or epoch-QC numerical correctness. QC original-row maps are the independently
modeled Availability inputs; the recursive wrapper attaches the quality model
and its separate history evidence without changing fitted components or state.
Producing/authenticating that evidence remains separate. ArrayEncoders and hash
updates remain explicit byte/cryptographic boundaries. The requested digest
refers to the current core invocation's dataset, including when it is the
already-cleaned recursive dataset, not an invented pre-QC outer digest.

Numerical operations are separated into float64 cast, bin mean, temporal
matmul, estimator and final float64 packing. Their result cannot supply a
family, recipe, physical shape, identities or candidate. A mean/matmul may
return a nonfinite array without throwing: that value is retained until the
actual later guard/estimator, rather than replaced by an invented exception.
Numeric-array contents outside finite real representatives, genuine estimator
formulas and IEEE bounds remain unrefined. NumericalPayload represents only
finite decoded output arrays; a nonfinite estimator/packing result has no
correspondence claim in this success slice. No source refusal is inferred for
that unrepresented case. P12 authentication also remains separate.
-/

namespace N2P3.PersonalERPFit
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPAvailability N2P3.PersonalERPTransport

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def erpNames : List String := [.pooled, .toeplitz, .temporal, .xdawn].map Family.text

structure Dispatch where
  adapterBranch : Bool
  featureBranch : Bool
  erpFitBranch : Bool
  ldaFitBranch : Bool

/-- Literal tests at source lines593,650,674. The non-ERP estimator bodies
are outside this ERP restriction; their dispatch is not silently rewritten. -/
def dispatch (initialization : String) : Dispatch :=
  let adapter := initialization ∉ "waveform_lda" :: erpNames
  let features := initialization ∈ "waveform_lda" :: "source_feature_lda" :: erpNames
  let erp := initialization ∈ erpNames
  ⟨decide adapter, decide features, decide erp, decide (features ∧ ¬erp)⟩

theorem every_ERP_dispatch_skips_adapter_and_LDA (family : Family) :
    dispatch family.text = ⟨false, true, true, false⟩ := by
  cases family <;> simp [dispatch, erpNames, Family.text]

def classCount {R : ℕ} (labels : Fin R → Bool) (label : Bool) : ℕ :=
  ((List.finRange R).filter (fun i => labels i == label)).length

def bothClasses {R : ℕ} (labels : Fin R → Bool) : Prop :=
  (∃ i, labels i = false) ∧ (∃ i, labels i = true)

def candidateLabelGate {R : ℕ} (labels : Fin R → Bool) : Check Unit :=
  ensure (2 < R ∧ bothClasses labels) "candidate more-than-two rows and two classes"

def structuredLabelGate {R : ℕ} (labels : Fin R → Bool) : Check Unit :=
  ensure (bothClasses labels ∧ 2 ≤ classCount labels false ∧ 2 ≤ classCount labels true)
    "StructuredERP at least two rows in each class"

structure NumericalPayload where
  arrays : ArrayFields
  observedRank : Option ℕ
  effectiveNfilter : Option ℕ

/-- Nonfinite values are observed, not silently converted to source errors.
Their individual IEEE contents are outside this scheduling abstraction. -/
inductive NumericBatch (rows channels times : ℕ) where
  | finite (values : ERPFeatures.Batch rows channels times)
  | nonfinite

structure NumericalFit where
  castFloat64 : (rows channels times : ℕ) → NumericBatch rows channels times →
    Check (NumericBatch rows channels times)
  binMean : (rows channels times width : ℕ) → ERPFeatures.Batch rows channels times →
    Check (NumericBatch rows channels (times / width))
  temporalMatmul : (rows channels times rank : ℕ) → ERPFeatures.Batch rows channels times → Tensor →
    Check (NumericBatch rows channels rank)
  estimate : (rows channels times : ℕ) → Header → NumericBatch rows channels times →
    (Fin rows → Bool) → Check NumericalPayload
  packFloat64 : NumericalPayload → Check NumericalPayload

/-- Literal _input order: cast first; full finite/geometry guard; then mean.
No output-finiteness gate occurs after mean in this source method. -/
def inputStage (fit : NumericalFit) {R C T : ℕ} (header : Header)
    (values : ERPFeatures.Batch R C T) : Check (NumericBatch R C (T / header.binWidth)) := do
  let casted ← fit.castFloat64 R C T (.finite values)
  match casted with
  | .nonfinite => .error (.sourceReject "StructuredERP finite input geometry")
  | .finite values => do
    ensure (0 < C ∧ header.binWidth ≤ T) "StructuredERP finite input geometry"
    fit.binMean R C T header.binWidth values

def temporalStage (fit : NumericalFit) {R C T : ℕ} (temporal : TemporalModel)
    (values : NumericBatch R C T) : Check (NumericBatch R C temporal.rank) := do
  let casted ← fit.castFloat64 R C T values
  match casted with
  | .nonfinite => .error (.sourceReject "StructuredERP temporal projection geometry/finite")
  | .finite values => do
    ensure (T = temporal.times) "StructuredERP temporal projection geometry/finite"
    fit.temporalMatmul R C T temporal.rank values temporal.matrix

/-- Input/binning precedes the class gate; temporal projection and estimator
follow it. Every primitive exception propagates without reinterpretation. -/
def fitStages (fit : NumericalFit) {R C T : ℕ} (header : Header)
    (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool) : Check NumericalPayload := do
  let binned ← inputStage fit header values
  structuredLabelGate labels
  let estimated ← match header.temporal with
    | none => fit.estimate R C (T / header.binWidth) header binned labels
    | some temporal => do
      let projected ← temporalStage fit temporal binned
      fit.estimate R C temporal.rank header projected labels
  fit.packFloat64 estimated

/-- Every premise names one operation result or source guard, never fitStages
success. Finite binned values are required only by temporal.project. -/
def FitSteps (fit : NumericalFit) {R C T : ℕ} (header : Header)
    (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool) (numeric : NumericalPayload) : Prop :=
  ∃ (casted : ERPFeatures.Batch R C T) (binned : NumericBatch R C (T / header.binWidth))
    (estimated : NumericalPayload),
    fit.castFloat64 R C T (.finite values) = .ok (.finite casted) ∧
    (0 < C ∧ header.binWidth ≤ T) ∧
    fit.binMean R C T header.binWidth casted = .ok binned ∧
    (bothClasses labels ∧ 2 ≤ classCount labels false ∧ 2 ≤ classCount labels true) ∧
    (match header.temporal with
     | none => fit.estimate R C (T / header.binWidth) header binned labels = .ok estimated
     | some temporal => ∃ (temporalInput : ERPFeatures.Batch R C (T / header.binWidth))
         (projected : NumericBatch R C temporal.rank),
       fit.castFloat64 R C (T / header.binWidth) binned = .ok (.finite temporalInput) ∧
       T / header.binWidth = temporal.times ∧
       fit.temporalMatmul R C (T / header.binWidth) temporal.rank temporalInput temporal.matrix = .ok projected ∧
       fit.estimate R C temporal.rank header projected labels = .ok estimated) ∧
    fit.packFloat64 estimated = .ok numeric

theorem fit_stages_from_individual_operations (fit : NumericalFit) {R C T : ℕ}
    (header : Header) (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool)
    (numeric : NumericalPayload) (h : FitSteps fit header values labels numeric) :
    fitStages fit header values labels = .ok numeric := by
  rcases h with ⟨casted, binned, estimated, hc, hg, hm, hl, hp, hf⟩
  cases ht : header.temporal with
  | none =>
    simp only [ht] at hp
    simp [fitStages, inputStage, hc, ensure, hg, hm, structuredLabelGate, hl, ht, hp, hf]
  | some temporal =>
    simp only [ht] at hp
    rcases hp with ⟨temporalInput, projected, htc, htg, htm, he⟩
    simp [fitStages, inputStage, hc, ensure, hg, hm, structuredLabelGate, hl, ht,
      temporalStage, htc, htg, htm, he, hf]

inductive TemporalConstructorFields where
  | absent
  | fixed (fields : FixedTemporalInput)
  | source (fields : FixedTemporalInput) (receipts : List PersonalERPIdentity.ReceiptInput)
      (identities : List IdentityImage)

def TemporalConstructorFields.basis (p : TemporalPrimitives) : TemporalConstructorFields → Option TemporalModel
  | .absent => none
  | .fixed b => some (fixedTemporalModel p b)
  | .source b rs ids => some (PersonalERPIdentity.sourceModel p b rs (canonicalStudies ids))

def TemporalConstructorFields.Conditions (p : TemporalPrimitives) (family : Family)
    (width : ℕ) (baseline : Value) : TemporalConstructorFields → Prop
  | .absent => family ≠ .temporal
  | .fixed b => family = .temporal ∧ width = 1 ∧ FixedTemporalConditions p b ∧
      PersonalAdmission.PyEq b.baseline baseline
  | .source b rs ids => family = .temporal ∧ width = 1 ∧
      PersonalERPIdentity.SourceBasisConditions p b ∧ rs ≠ [] ∧
      List.Forall₂ (PersonalERPIdentity.ReceiptConditions p) rs ids ∧
      (ids.map IdentityImage.authority).Nodup ∧ PersonalAdmission.PyEq b.baseline baseline

def rawConstructorFields (family width nfilter baseline : Value) : Fields :=
  Fields.ofList [("family", family), ("bin_width", width), ("baseline_mode", baseline), ("nfilter", nfilter)]

/-- Raw constructor values and every temporal field/receipt guard derive the
header. Neither commonHeader nor readTemporal success is a premise. The hash,
preprocessing, array, cosine and identity primitives remain explicit boundaries. -/
theorem common_header_from_raw_individual_fields (p : TemporalPrimitives)
    (rawFamily rawWidth rawNfilter baseline : Value) (family : Family) (width nfilter : ℕ)
    (temporal : TemporalConstructorFields)
    (hf : rawFamily = .text family.text) (hw : rawWidth = .integer width)
    (hn : rawNfilter = .integer nfilter) (hwidth : 0 < width) (hnfilter : 0 < nfilter)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none")
    (ht : temporal.Conditions p family width baseline) :
    commonHeader p (rawConstructorFields rawFamily rawWidth rawNfilter baseline) (temporal.basis p) =
      .ok ⟨family, width, nfilter, baseline, temporal.basis p⟩ := by
  subst rawFamily; subst rawWidth; subst rawNfilter
  cases temporal with
  | absent =>
    change family ≠ .temporal at ht
    cases family <;> simp_all [commonHeader, rawConstructorFields, Fields.ofList, Fields.get,
      required, readFamily, Family.text, positiveInt, ensure, TemporalConstructorFields.basis]
  | fixed b =>
    rcases ht with ⟨rfl, rfl, hb, hbase⟩
    have hr := fixed_temporal_reader_from_explicit_fields p b hb
    simp [commonHeader, rawConstructorFields, Fields.ofList, Fields.get, required,
      readFamily, Family.text, positiveInt, ensure, hnfilter, TemporalConstructorFields.basis,
      fixedTemporalModel, hr, hbase]
  | source b rs ids =>
    rcases ht with ⟨rfl, rfl, hb, hrs, hreceipts, hu, hbase⟩
    have hr := PersonalERPIdentity.source_reader_with_derived_studies p b rs ids hb hrs hreceipts hu
    simp [commonHeader, rawConstructorFields, Fields.ofList, Fields.get, required,
      readFamily, Family.text, positiveInt, ensure, hnfilter, TemporalConstructorFields.basis,
      PersonalERPIdentity.sourceModel, hr, hbase]

structure Artifact where
  header : Header
  channels : ℕ
  times : ℕ
  numeric : NumericalPayload

structure Components where
  adapter : Option NamedState
  lda : Option NamedState
  erp : Option Artifact

def initialComponents : Components := ⟨none, none, none⟩

/-- Source None initialization followed by literal branch assignments.
Non-ERP numerical states are arbitrary arguments, never candidate oracles;
the ERP dispatch theorem derives that neither argument is consumed. -/
def componentAssignments (initialization : String) (artifact : Artifact)
    (adapterState ldaState : NamedState) : Components :=
  let branch := dispatch initialization
  let components := initialComponents
  let components := if branch.adapterBranch then { components with adapter := some adapterState } else components
  if branch.featureBranch then
    if branch.erpFitBranch then { components with erp := some artifact }
    else { components with lda := some ldaState }
  else components

theorem ERP_assignments_derive_only_ERP (family : Family) (artifact : Artifact)
    (adapterState ldaState : NamedState) :
    componentAssignments family.text artifact adapterState ldaState = ⟨none, none, some artifact⟩ := by
  rw [componentAssignments, every_ERP_dispatch_skips_adapter_and_LDA]
  rfl

def artifactModel (artifact : Artifact) : Model :=
  ⟨artifact.header, artifact.channels, artifact.times, numpyArrays artifact.numeric.arrays,
    artifact.numeric.observedRank, artifact.numeric.effectiveNfilter⟩

structure History (Digest : Type) where
  trainingRows : ℕ
  requestedTrainingRows : Option ℕ
  validationRows : ℕ
  prefixMaskExplicit : Bool
  requestedDataDigest : Digest
  executedDataDigest : Digest
  waveformChannels : List ℕ
  usedUnion : Bool
  erpArtifact : Fields
  epochQualityEvidence : Option Value := none

structure Candidate (Digest : Type) where
  recipe : PersonalERPRecipe.Recipe
  sfreq : ℝ
  adapter : Option NamedState
  lda : Option NamedState
  erp : Option Artifact
  selected : List ℕ
  history : History Digest
  epochQuality : Option Value
  channelQualityAvailability : Bool
  minimumUsableChannels : ℤ

structure Produced (N C T : ℕ) (Digest : Type) where
  family : Family
  selection : TrainingSelection N C
  invocation : ERPFitInvocation (availableRows selection.rows).length selection.channels.length T Digest
  artifact : Artifact
  minimum : ℤ
  localAvailability : Bool

def labelsAtRows {N : ℕ} (labels : Fin N → Bool) (rows : List (Fin N)) : Fin rows.length → Bool :=
  fun i => labels (rows.get i)

theorem class_count_on_original_rows {N : ℕ} (labels : Fin N → Bool)
    (rows : List (Fin N)) (label : Bool) :
    classCount (labelsAtRows labels rows) label = (rows.filter (fun i => labels i == label)).length := by
  have h := congrArg (fun xs : List (Fin N) => (xs.filter (fun i => labels i == label)).length)
    (List.finRange_map_get rows)
  simpa only [List.filter_map, List.length_map, Function.comp_def, classCount, labelsAtRows] using h

def produce {N C T : ℕ} {HashState Digest : Type}
    (p : TemporalPrimitives) (fit : NumericalFit) (ceilProduct : CeilProduct)
    (hash : HashOperations HashState Digest) (encode : ArrayEncoders)
    (recipe : PersonalERPRecipe.Recipe) (baseline : Value) (basis : Option TemporalModel)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : Mask N) (mask : Observed N C) : Check (Produced N C T Digest) := do
  ensure (∃ i, training i = true) "nonempty aligned training mask"
  candidateLabelGate (labelsAtRows labels (availableRows training))
  -- adapter, fitted_lda, fitted_erp and waveform_channels start as None.
  let family ← readFamily recipe.raw.initialization
  let branch := dispatch family.text
  ensure (branch.adapterBranch = false ∧ branch.featureBranch = true ∧
    branch.erpFitBranch = true ∧ branch.ldaFitBranch = false) "ERP dispatch restriction"
  let selection ← trainingSelection ceilProduct processing training mask
  if selection.usedUnion then
    candidateLabelGate (labelsAtRows labels (availableRows selection.rows))
  ensure (selection.channels ≠ []) "waveform channels nonempty"
  let invocation := erpFitInvocation hash encode values labels mask training selection
  -- These are the actual StructuredERP constructor defaults/arguments.
  let header ← commonHeader p
    (Fields.ofList [("family", recipe.raw.initialization), ("bin_width", recipe.raw.erpBinWidth),
      ("baseline_mode", baseline), ("nfilter", .integer 4)]) basis
  let numeric ← fitStages fit header invocation.values invocation.labels
  let minimum ← minimumCount ceilProduct processing C
  let localAvailability ← inferenceLocal processing
  pure ⟨family, selection, invocation, ⟨header, selection.channels.length, T, numeric⟩, minimum, localAvailability⟩

/-- Composition of explicit gates and already-defined small helper operations.
The only numerical fit result premise is an array/rank payload. No premise
asserts whole candidate, producer, state-export or prediction correctness. -/
theorem produce_from_individual_source_steps {N C T : ℕ} {HashState Digest : Type}
    (p : TemporalPrimitives) (fit : NumericalFit) (ceilProduct : CeilProduct)
    (hash : HashOperations HashState Digest) (encode : ArrayEncoders)
    (recipe : PersonalERPRecipe.Recipe) (baseline : Value) (basis : Option TemporalModel)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : Mask N) (mask : Observed N C) (family : Family)
    (selection : TrainingSelection N C) (header : Header) (numeric : NumericalPayload)
    (minimum : ℤ) (localAvailability : Bool)
    (hnonempty : ∃ i, training i = true)
    (hrequested : 2 < (availableRows training).length ∧ bothClasses (labelsAtRows labels (availableRows training)))
    (hfamily : recipe.raw.initialization = .text family.text)
    (hselection : trainingSelection ceilProduct processing training mask = .ok selection)
    (hfallback : selection.usedUnion = true →
      2 < (availableRows selection.rows).length ∧ bothClasses (labelsAtRows labels (availableRows selection.rows)))
    (hchannels : selection.channels ≠ [])
    (hconstructor : commonHeader p (Fields.ofList [("family", recipe.raw.initialization),
      ("bin_width", recipe.raw.erpBinWidth), ("baseline_mode", baseline), ("nfilter", .integer 4)]) basis = .ok header)
    (hsteps : FitSteps fit header
      (erpFitInvocation hash encode values labels mask training selection).values
      (erpFitInvocation hash encode values labels mask training selection).labels numeric)
    (hminimum : minimumCount ceilProduct processing C = .ok minimum)
    (hlocal : inferenceLocal processing = .ok localAvailability) :
    produce p fit ceilProduct hash encode recipe baseline basis processing values labels training mask =
      .ok ⟨family, selection, erpFitInvocation hash encode values labels mask training selection,
        ⟨header, selection.channels.length, T, numeric⟩, minimum, localAvailability⟩ := by
  have hr : readFamily recipe.raw.initialization = .ok family := by
    rw [hfamily, family_text_decodes]
  have hrequestedGate : candidateLabelGate (labelsAtRows labels (availableRows training)) = .ok () := by
    simp [candidateLabelGate, ensure, hrequested]
  have hfit := fit_stages_from_individual_operations fit header
    (erpFitInvocation hash encode values labels mask training selection).values
    (erpFitInvocation hash encode values labels mask training selection).labels numeric hsteps
  simp [produce, ensure, hnonempty, hrequestedGate, hr, every_ERP_dispatch_skips_adapter_and_LDA,
    hselection, hchannels, hconstructor, hfit, hminimum, hlocal]
  cases hu : selection.usedUnion <;> simp_all [candidateLabelGate, ensure]

/-- Public composition corollary: constructor header is derived from raw
family/bin/default-nfilter/baseline and fixed/source temporal fields. The
underlying source-step lemma's helper-result premise is discharged here. -/
theorem produce_from_independent_raw_fields {N C T : ℕ} {HashState Digest : Type}
    (p : TemporalPrimitives) (fit : NumericalFit) (ceilProduct : CeilProduct)
    (hash : HashOperations HashState Digest) (encode : ArrayEncoders)
    (recipe : PersonalERPRecipe.Recipe) (baseline : Value) (temporal : TemporalConstructorFields)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : Mask N) (mask : Observed N C) (family : Family) (width : ℕ)
    (selection : TrainingSelection N C) (numeric : NumericalPayload) (minimum : ℤ) (localAvailability : Bool)
    (hnonempty : ∃ i, training i = true)
    (hrequested : 2 < (availableRows training).length ∧ bothClasses (labelsAtRows labels (availableRows training)))
    (hfamily : recipe.raw.initialization = .text family.text)
    (hwidthRaw : recipe.raw.erpBinWidth = .integer width) (hwidth : 0 < width)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none")
    (htemporal : temporal.Conditions p family width baseline)
    (hselection : trainingSelection ceilProduct processing training mask = .ok selection)
    (hfallback : selection.usedUnion = true →
      2 < (availableRows selection.rows).length ∧ bothClasses (labelsAtRows labels (availableRows selection.rows)))
    (hchannels : selection.channels ≠ [])
    (hsteps : FitSteps fit ⟨family, width, 4, baseline, temporal.basis p⟩
      (erpFitInvocation hash encode values labels mask training selection).values
      (erpFitInvocation hash encode values labels mask training selection).labels numeric)
    (hminimum : minimumCount ceilProduct processing C = .ok minimum)
    (hlocal : inferenceLocal processing = .ok localAvailability) :
    produce p fit ceilProduct hash encode recipe baseline (temporal.basis p) processing values labels training mask =
      .ok ⟨family, selection, erpFitInvocation hash encode values labels mask training selection,
        ⟨⟨family, width, 4, baseline, temporal.basis p⟩, selection.channels.length, T, numeric⟩, minimum, localAvailability⟩ := by
  have hc := common_header_from_raw_individual_fields p recipe.raw.initialization recipe.raw.erpBinWidth
    (.integer 4) baseline family width 4 temporal hfamily hwidthRaw rfl hwidth (by norm_num) hbaseline htemporal
  exact produce_from_individual_source_steps p fit ceilProduct hash encode recipe baseline (temporal.basis p)
    processing values labels training mask family selection ⟨family, width, 4, baseline, temporal.basis p⟩
    numeric minimum localAvailability hnonempty hrequested hfamily hselection hfallback hchannels hc hsteps hminimum hlocal

def buildCandidate {N C T : ℕ} {Digest : Type} (recipe : PersonalERPRecipe.Recipe)
    (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) : Candidate Digest :=
  let selection := produced.selection
  let selected := selection.channels.map Fin.val
  let components := componentAssignments produced.family.text produced.artifact [] []
  { recipe := recipe, sfreq := sfreq,
    adapter := components.adapter, lda := components.lda, erp := components.erp, selected := selected,
    history := {
      trainingRows := if selection.usedUnion then (availableRows selection.rows).length else (availableRows training).length
      requestedTrainingRows := if selection.usedUnion then some (availableRows training).length else none
      validationRows := 0, prefixMaskExplicit := true,
      requestedDataDigest := produced.invocation.requestedDataDigest,
      executedDataDigest := produced.invocation.executedDataDigest,
      waveformChannels := selected, usedUnion := selection.usedUnion,
      erpArtifact := exportRecord (artifactModel produced.artifact) },
    epochQuality := none, channelQualityAvailability := produced.localAvailability,
    minimumUsableChannels := produced.minimum }

def fitCandidate {N C T : ℕ} {HashState Digest : Type}
    (p : TemporalPrimitives) (fit : NumericalFit) (ceilProduct : CeilProduct)
    (hash : HashOperations HashState Digest) (encode : ArrayEncoders)
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (baseline : Value) (basis : Option TemporalModel)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : Mask N) (mask : Observed N C) : Check (Candidate Digest) :=
  (produce p fit ceilProduct hash encode recipe baseline basis processing values labels training mask).map
    (buildCandidate recipe sfreq training)

theorem explicit_candidate_has_only_ERP {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) :
    (buildCandidate recipe sfreq training produced).adapter = none ∧
    (buildCandidate recipe sfreq training produced).lda = none ∧
    (buildCandidate recipe sfreq training produced).erp = some produced.artifact := by
  simp [buildCandidate, ERP_assignments_derive_only_ERP]

theorem successful_fit_derives_only_ERP {N C T : ℕ} {HashState Digest : Type}
    (p : TemporalPrimitives) (fit : NumericalFit) (ceilProduct : CeilProduct)
    (hash : HashOperations HashState Digest) (encode : ArrayEncoders)
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (baseline : Value) (basis : Option TemporalModel)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : Mask N) (mask : Observed N C) (candidate : Candidate Digest)
    (h : fitCandidate p fit ceilProduct hash encode recipe sfreq baseline basis processing values labels training mask = .ok candidate) :
    candidate.adapter = none ∧ candidate.lda = none ∧ candidate.erp.isSome = true := by
  unfold fitCandidate at h
  cases ho : produce p fit ceilProduct hash encode recipe baseline basis processing values labels training mask with
  | error fault => simp [ho] at h
  | ok produced =>
    simp only [ho, Except.map, Except.ok.injEq] at h
    subst candidate
    simp [buildCandidate, ERP_assignments_derive_only_ERP]

/-- Literal dictionary update for the optional LDA state block. ERP construction
derives that block absent; arbitrary simultaneous candidate objects are not
assumed safe. -/
def assignTensor (state : NamedState) (entry : String × Tensor) : NamedState :=
  if (state.map Prod.fst).contains entry.1 then
    state.map (fun old => if old.1 = entry.1 then entry else old)
  else state ++ [entry]

def candidateState {Digest : Type} (candidate : Candidate Digest) : NamedState :=
  let erpState := match candidate.erp with
    | none => []
    | some artifact => producerERP artifact.numeric.arrays (PersonalERPRecipe.selectedTensor candidate.selected)
  let state := afterAdapter erpState candidate.adapter
  match candidate.lda with
  | none => state
  | some ldaState => ldaState.foldl assignTensor state

theorem constructed_candidate_state_uses_actual_ERP_branch {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) :
    candidateState (buildCandidate recipe sfreq training produced) =
      producerERP produced.artifact.numeric.arrays
        (PersonalERPRecipe.selectedTensor (produced.selection.channels.map Fin.val)) := by
  simp [candidateState, buildCandidate, ERP_assignments_derive_only_ERP, afterAdapter]

theorem constructed_candidate_consumer_recovers_individual_arrays {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) :
    consumerERP (candidateState (buildCandidate recipe sfreq training produced)) =
      numpyArrays produced.artifact.numeric.arrays := by
  rw [constructed_candidate_state_uses_actual_ERP_branch]
  exact erp_only_producer_recovers_each_numpy_field _ _

def attachEpochQuality {Digest : Type} (candidate : Candidate Digest) (quality evidence : Value) : Candidate Digest :=
  { candidate with
    epochQuality := some quality
    history := { candidate.history with epochQualityEvidence := some evidence } }

theorem epoch_quality_wrapper_preserves_all_components {Digest : Type}
    (candidate : Candidate Digest) (quality evidence : Value) :
    (attachEpochQuality candidate quality evidence).adapter = candidate.adapter ∧
    (attachEpochQuality candidate quality evidence).lda = candidate.lda ∧
    (attachEpochQuality candidate quality evidence).erp = candidate.erp ∧
    candidateState (attachEpochQuality candidate quality evidence) = candidateState candidate := ⟨rfl, rfl, rfl, rfl⟩

theorem numerical_arrays_cannot_change_configured_geometry (header : Header) (channels times : ℕ)
    (numeric : NumericalPayload) :
    (artifactModel ⟨header, channels, times, numeric⟩).header = header ∧
    (artifactModel ⟨header, channels, times, numeric⟩).channels = channels ∧
    (artifactModel ⟨header, channels, times, numeric⟩).times = times := ⟨rfl, rfl, rfl⟩

theorem ERP_metadata_has_no_neural_source_initialization (family : Family) :
    family.text ≠ "source" := every_erp_family_is_not_source_initialization family

theorem constructed_metadata_role_uses_actual_basis {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) :
    producedKind false (buildCandidate recipe sfreq training produced).adapter.isSome
      (temporalTransfer (artifactModel produced.artifact)) (buildCandidate recipe sfreq training produced).erp.isSome =
      if temporalTransfer (artifactModel produced.artifact) then "source_temporal_target_erp" else "target_only_structured_erp" := by
  simp [buildCandidate, ERP_assignments_derive_only_ERP, producedKind]

def producedModelKind {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) : String :=
  let candidate := buildCandidate recipe sfreq training produced
  producedKind (decide (recipe.raw.initialization = .text "source")) candidate.adapter.isSome
    (temporalTransfer (artifactModel produced.artifact)) candidate.erp.isSome

def producedSourceFlag {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (produced : Produced N C T Digest) : Value :=
  sourceFlag (decide (recipe.raw.initialization = .text "source"))
    (temporalTransfer (artifactModel produced.artifact))

theorem metadata_kind_derived_from_initialization_and_component_assignments {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest)
    (hf : recipe.raw.initialization = .text produced.family.text) :
    producedModelKind recipe sfreq training produced =
      if temporalTransfer (artifactModel produced.artifact) then "source_temporal_target_erp" else "target_only_structured_erp" := by
  have hs : recipe.raw.initialization ≠ .text "source" := by
    rw [hf]
    intro h
    exact every_erp_family_is_not_source_initialization produced.family (Value.text.inj h)
  simp [producedModelKind, buildCandidate, ERP_assignments_derive_only_ERP, hs, producedKind]

theorem metadata_source_flag_derived_from_initialization_and_basis {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (produced : Produced N C T Digest)
    (hf : recipe.raw.initialization = .text produced.family.text) :
    producedSourceFlag recipe produced = .boolean (temporalTransfer (artifactModel produced.artifact)) := by
  have hs : recipe.raw.initialization ≠ .text "source" := by
    rw [hf]
    intro h
    exact every_erp_family_is_not_source_initialization produced.family (Value.text.inj h)
  simp [producedSourceFlag, hs, sourceFlag]

theorem constructed_history_preserves_two_digest_operands {N C T : ℕ} {Digest : Type}
    (recipe : PersonalERPRecipe.Recipe) (sfreq : ℝ) (training : Mask N) (produced : Produced N C T Digest) :
    (buildCandidate recipe sfreq training produced).history.requestedDataDigest = produced.invocation.requestedDataDigest ∧
    (buildCandidate recipe sfreq training produced).history.executedDataDigest = produced.invocation.executedDataDigest := ⟨rfl, rfl⟩

def threeLabels : Fin 3 → Bool := ![false, false, true]

theorem three_rows_pass_candidate_but_fail_structured_fit :
    candidateLabelGate threeLabels = .ok () ∧
    structuredLabelGate threeLabels = .error (.sourceReject "StructuredERP at least two rows in each class") := by
  have hb : bothClasses threeLabels := ⟨⟨0, rfl⟩, ⟨2, rfl⟩⟩
  have hc0 : classCount threeLabels false = 2 := rfl
  have hc1 : classCount threeLabels true = 1 := rfl
  simp [candidateLabelGate, structuredLabelGate, hb, hc0, hc1, ensure]

theorem cast_exception_precedes_input_and_class_guards (fit : NumericalFit) {R C T : ℕ}
    (header : Header) (values : ERPFeatures.Batch R C T) (labels : Fin R → Bool) (fault : LedgerFault)
    (hc : fit.castFloat64 R C T (.finite values) = .error fault) :
    fitStages fit header values labels = .error fault := by
  simp [fitStages, inputStage, hc]

/-- Even insufficient per-class counts do not supersede an earlier numerical
mean exception. This is the finite-large-input overflow order seen in source. -/
theorem bin_mean_exception_precedes_inner_class_refusal (fit : NumericalFit) {C T : ℕ}
    (header : Header) (values casted : ERPFeatures.Batch 3 C T) (fault : LedgerFault)
    (hc : fit.castFloat64 3 C T (.finite values) = .ok (.finite casted))
    (hg : 0 < C ∧ header.binWidth ≤ T)
    (hm : fit.binMean 3 C T header.binWidth casted = .error fault) :
    fitStages fit header values threeLabels = .error fault := by
  simp [fitStages, inputStage, hc, ensure, hg, hm]

/-- Default arithmetic may return nonfinite mean values. There is no new
post-mean finite guard: the actual inner label refusal still occurs next. -/
theorem nonfinite_mean_reaches_inner_class_refusal (fit : NumericalFit) {C T : ℕ}
    (header : Header) (values casted : ERPFeatures.Batch 3 C T)
    (hc : fit.castFloat64 3 C T (.finite values) = .ok (.finite casted))
    (hg : 0 < C ∧ header.binWidth ≤ T)
    (hm : fit.binMean 3 C T header.binWidth casted = .ok .nonfinite) :
    fitStages fit header values threeLabels =
      .error (.sourceReject "StructuredERP at least two rows in each class") := by
  simp [fitStages, inputStage, hc, ensure, hg, hm, three_rows_pass_candidate_but_fail_structured_fit.2]

theorem plain_nonfinite_mean_is_passed_to_estimator (fit : NumericalFit) {R C T : ℕ}
    (header : Header) (values casted : ERPFeatures.Batch R C T) (labels : Fin R → Bool)
    (fault : LedgerFault) (ht : header.temporal = none)
    (hc : fit.castFloat64 R C T (.finite values) = .ok (.finite casted))
    (hg : 0 < C ∧ header.binWidth ≤ T)
    (hm : fit.binMean R C T header.binWidth casted = .ok .nonfinite)
    (hl : bothClasses labels ∧ 2 ≤ classCount labels false ∧ 2 ≤ classCount labels true)
    (he : fit.estimate R C (T / header.binWidth) header .nonfinite labels = .error fault) :
    fitStages fit header values labels = .error fault := by
  simp [fitStages, inputStage, hc, ensure, hg, hm, structuredLabelGate, hl, ht, he]

theorem temporal_nonfinite_input_is_checked_before_matmul (fit : NumericalFit) {R C T : ℕ}
    (temporal : TemporalModel) (input : NumericBatch R C T)
    (hc : fit.castFloat64 R C T input = .ok .nonfinite) :
    temporalStage fit temporal input =
      .error (.sourceReject "StructuredERP temporal projection geometry/finite") := by
  simp [temporalStage, hc]

def meanOverflowModel : NumericalFit := {
  castFloat64 := fun _ _ _ input => .ok input
  binMean := fun _ _ _ _ _ => .error (.fieldFailure "numpy mean overflow")
  temporalMatmul := fun _ _ _ _ _ _ => .error (.fieldFailure "unused matmul")
  estimate := fun _ _ _ _ _ _ => .error (.fieldFailure "unused estimator")
  packFloat64 := fun _ => .error (.fieldFailure "unused packing") }

def meanNonfiniteModel : NumericalFit :=
  { meanOverflowModel with binMean := fun _ _ _ _ _ => .ok .nonfinite }

def orderFixtureValues : ERPFeatures.Batch 3 1 2 := fun _ _ _ => 10 ^ (308 : ℕ)

def orderFixtureHeader : Header := ⟨.pooled, 2, 4, .text "none", none⟩

/-- Declared arithmetic responses, not a claim Lean evaluated IEEE overflow.
The root source probe independently supplies the NumPy overflow witness. -/
theorem declared_mean_overflow_precedes_three_row_class_refusal :
    fitStages meanOverflowModel orderFixtureHeader orderFixtureValues threeLabels =
      .error (.fieldFailure "numpy mean overflow") := by
  apply bin_mean_exception_precedes_inner_class_refusal
  · rfl
  · norm_num [orderFixtureHeader]
  · rfl

theorem declared_nonfinite_mean_preserves_three_row_class_refusal :
    fitStages meanNonfiniteModel orderFixtureHeader orderFixtureValues threeLabels =
      .error (.sourceReject "StructuredERP at least two rows in each class") := by
  apply nonfinite_mean_reaches_inner_class_refusal
  · rfl
  · norm_num [orderFixtureHeader]
  · rfl

def fitFixtureProcessing : Fields := localProcessing (.text "applied") (.boolean true) (.number (1 / 2))
def fitFixtureCeil : CeilProduct := fun _ _ => .ok 1
def fitFixtureMask : Observed 6 2 := fun i c => decide (2 ≤ i.val ∧ c.val = 1)
def fitFixtureLabels : Fin 6 → Bool := ![true, false, false, false, true, true]
def fitFixtureValues : ERPFeatures.Batch 6 2 1 := fun i c _ => (10 * i.val + c.val : ℕ)
def fitFixtureSelection : TrainingSelection 6 2 := ⟨[1], fun i => decide (2 ≤ i.val), true⟩

theorem fit_fixture_selection_is_actual_fallback :
    trainingSelection fitFixtureCeil fitFixtureProcessing (fun _ => true) fitFixtureMask = .ok fitFixtureSelection := by
  have hi : trainingIntersection (fun _ => true) fitFixtureMask = [] := by
    simp [trainingIntersection, fitFixtureMask]
    decide
  have hu : trainingUnion (fun _ => true) fitFixtureMask = [1] := by
    simp [trainingUnion, fitFixtureMask]
    decide
  have hm : (fun i : Fin 6 => allSelected fitFixtureMask [1] i) = fun i => decide (2 ≤ i.val) := by
    funext i
    fin_cases i <;> simp [allSelected, fitFixtureMask]
  have hminimum : minimumCount fitFixtureCeil fitFixtureProcessing 2 = .ok 1 := by
    exact applied_true_minimum_calls_only_scalar_ceil fitFixtureCeil 2 (1 / 2)
  unfold trainingSelection
  rw [hminimum]
  simp [hi, hu, hm, fitFixtureProcessing, fitFixtureSelection, local_flag_is_exact_bool, ensure]

theorem fit_fixture_rows_preserve_original_positions :
    availableRows fitFixtureSelection.rows = [2, 3, 4, 5] := by
  simp [fitFixtureSelection, availableRows, List.finRange_succ]

theorem fit_fixture_rows_by_predicate :
    availableRows (fun i : Fin 6 => decide (2 ≤ i.val)) = [2, 3, 4, 5] :=
  fit_fixture_rows_preserve_original_positions

def firstFitRow : Fin (availableRows fitFixtureSelection.rows).length :=
  ⟨0, by rw [fit_fixture_rows_preserve_original_positions]; decide⟩
def thirdFitRow : Fin (availableRows fitFixtureSelection.rows).length :=
  ⟨2, by rw [fit_fixture_rows_preserve_original_positions]; decide⟩
def firstFitChannel : Fin fitFixtureSelection.channels.length := ⟨0, by simp [fitFixtureSelection]⟩

def fitFixtureHash : HashOperations (List UInt8) (List UInt8) := ⟨[], List.append, id⟩

/-- A transparent shape-byte codec countermodel only, not NumPy tobytes.
The general theorem keeps actual byte operations explicit and unrefined. -/
def fitFixtureEncoding : ArrayEncoders := {
  values := fun _ channels _ _ => [UInt8.ofNat channels]
  labels := fun rows _ => [UInt8.ofNat rows]
  observed := fun _ channels _ => [UInt8.ofNat channels] }

def fitFixturePayload : NumericalPayload :=
  ⟨linearArrays (fun _ : Fin 1 => (2 : ℝ)) (-1), none, none⟩
def fitFixtureHeader : Header := ⟨.pooled, 1, 4, .text "none", none⟩
def fitFixtureArtifact : Artifact := ⟨fitFixtureHeader, 1, 1, fitFixturePayload⟩
def fitFixtureNumericalOperation : NumericalFit := {
  castFloat64 := fun _ _ _ value => .ok value
  binMean := fun _ _ _ width values => .ok (.finite (fun i => ERPFeatures.completeBins width (values i)))
  temporalMatmul := fun _ _ _ _ _ _ => .ok .nonfinite
  estimate := fun _ _ _ _ _ _ => .ok fitFixturePayload
  packFloat64 := fun value => .ok value }
def fitFixtureInvocation :=
  erpFitInvocation fitFixtureHash fitFixtureEncoding fitFixtureValues fitFixtureLabels fitFixtureMask
    (fun _ => true) fitFixtureSelection
def fitFixtureProduced : Produced 6 2 1 (List UInt8) :=
  ⟨.pooled, fitFixtureSelection, fitFixtureInvocation, fitFixtureArtifact, 1, true⟩

theorem fit_fixture_uses_selected_channel_and_original_row :
    fitFixtureInvocation.values firstFitRow firstFitChannel 0 = 21 ∧ fitFixtureValues 2 0 0 = 20 := by
  simp [fitFixtureInvocation, erpFitInvocation, fitView, orderedView,
    fit_fixture_rows_by_predicate, fitFixtureSelection, fitFixtureValues, firstFitRow, firstFitChannel]

theorem fit_fixture_labels_are_sliced_with_values :
    fitFixtureInvocation.labels firstFitRow = false ∧ fitFixtureLabels 0 = true := by
  simp [fitFixtureInvocation, erpFitInvocation, fitView,
    fitFixtureSelection, fit_fixture_rows_by_predicate, fitFixtureLabels, firstFitRow]

theorem fit_fixture_initial_and_internal_labels_are_legal :
    (2 < (availableRows (N := 6) (fun _ => true)).length ∧
      bothClasses (labelsAtRows fitFixtureLabels (availableRows (fun _ => true)))) ∧
    (2 < (availableRows fitFixtureSelection.rows).length ∧ bothClasses fitFixtureInvocation.labels) ∧
    (2 ≤ classCount fitFixtureInvocation.labels false ∧ 2 ≤ classCount fitFixtureInvocation.labels true) := by
  have hrequested : bothClasses (labelsAtRows fitFixtureLabels (availableRows (fun _ => true))) :=
    ⟨⟨⟨1, by simp [availableRows]⟩, rfl⟩, ⟨⟨0, by simp [availableRows]⟩, rfl⟩⟩
  have hexecuted : bothClasses fitFixtureInvocation.labels := by
    constructor
    · exact ⟨firstFitRow, fit_fixture_labels_are_sliced_with_values.1⟩
    · exact ⟨thirdFitRow, by simp [fitFixtureInvocation, erpFitInvocation, fitView,
        fitFixtureSelection, fit_fixture_rows_by_predicate, fitFixtureLabels, thirdFitRow]⟩
  refine ⟨⟨?_, hrequested⟩, ⟨?_, hexecuted⟩, ?_, ?_⟩
  · simp [availableRows]
  · rw [fit_fixture_rows_preserve_original_positions]
    decide
  · change 2 ≤ classCount (labelsAtRows fitFixtureLabels (availableRows fitFixtureSelection.rows)) false
    rw [class_count_on_original_rows, fit_fixture_rows_preserve_original_positions]
    simp [fitFixtureLabels]
  · change 2 ≤ classCount (labelsAtRows fitFixtureLabels (availableRows fitFixtureSelection.rows)) true
    rw [class_count_on_original_rows, fit_fixture_rows_preserve_original_positions]
    simp [fitFixtureLabels]

/-- Source-shaped, nonempty core execution with a declared numeric-array
operation. This executes no actual fit and certifies no estimator or bytes. -/
theorem nonempty_fit_dispatch_with_independent_numeric_payload :
    produce fixturePrimitives fitFixtureNumericalOperation fitFixtureCeil fitFixtureHash fitFixtureEncoding
      (PersonalERPRecipe.pooledFields 1).recipe (.text "none") none fitFixtureProcessing
      fitFixtureValues fitFixtureLabels (fun _ => true) fitFixtureMask = .ok fitFixtureProduced := by
  have hlabels := fit_fixture_initial_and_internal_labels_are_legal
  apply produce_from_independent_raw_fields fixturePrimitives fitFixtureNumericalOperation fitFixtureCeil
    fitFixtureHash fitFixtureEncoding (PersonalERPRecipe.pooledFields 1).recipe (.text "none") .absent
    fitFixtureProcessing fitFixtureValues fitFixtureLabels (fun _ => true) fitFixtureMask .pooled 1
    fitFixtureSelection fitFixturePayload 1 true
  · exact ⟨0, rfl⟩
  · exact hlabels.1
  · rfl
  · rfl
  · norm_num
  · simp
  · simp [TemporalConstructorFields.Conditions]
  · exact fit_fixture_selection_is_actual_fallback
  · intro _; exact hlabels.2.1
  · simp [fitFixtureSelection]
  · refine ⟨fitFixtureInvocation.values,
      .finite (fun i => ERPFeatures.completeBins 1 (fitFixtureInvocation.values i)),
      fitFixturePayload, rfl, ?_, rfl, ?_, rfl, rfl⟩
    · simp [fitFixtureSelection]
    · exact ⟨hlabels.2.1.2, hlabels.2.2⟩
  · exact applied_true_minimum_calls_only_scalar_ceil fitFixtureCeil 2 (1 / 2)
  · exact applied_true_local_routes (.number (1 / 2))

theorem nonempty_fit_candidate_has_no_simultaneous_components :
    fitCandidate fixturePrimitives fitFixtureNumericalOperation fitFixtureCeil fitFixtureHash fitFixtureEncoding
      (PersonalERPRecipe.pooledFields 1).recipe 64 (.text "none") none fitFixtureProcessing
      fitFixtureValues fitFixtureLabels (fun _ => true) fitFixtureMask =
      .ok (buildCandidate (PersonalERPRecipe.pooledFields 1).recipe 64 (fun _ => true) fitFixtureProduced) := by
  rw [fitCandidate, nonempty_fit_dispatch_with_independent_numeric_payload]
  rfl

theorem ERP_fixture_adapter_is_absent :
    (buildCandidate (PersonalERPRecipe.pooledFields 1).recipe 64 (fun _ => true) fitFixtureProduced).adapter = none := by
  exact (explicit_candidate_has_only_ERP _ _ _ _).1

theorem ERP_fixture_LDA_is_absent :
    (buildCandidate (PersonalERPRecipe.pooledFields 1).recipe 64 (fun _ => true) fitFixtureProduced).lda = none := by
  exact (explicit_candidate_has_only_ERP _ _ _ _).2.1

end
end N2P3.PersonalERPFit
