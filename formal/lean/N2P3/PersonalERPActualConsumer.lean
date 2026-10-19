import N2P3.PersonalERPScalarFields
import N2P3.PersonalERPCurrentConsumer

/-!
# Actual scalar constructor on the ERP producer/current-consumer route

Sources: calibration_candidates.py:311-332, calibrated.py:550-551,619-642,
recipes.py:39-98, subject_adapter.py:98-165. This new Result-valued route keeps
the reviewed decoded CurrentConsumer19/2 unchanged. Clone and the Torch int64
selector precede per-ERP-array NumPy view/current reader; the actual Scalar
kwargs constructor follows them and precedes physical/role binding.

No new success oracle is introduced. Independent raw structural fields and
single finite/comparison responses derive constructor success. Huge integers
and JSON list/dict values remain possible inputs to this route and preserve
their scalar/set-membership exceptions. NumPy/BLAS/tensor bytes, actual numeric
array conversion, full physical ingress, temporal/xDAWN producer combinations,
and complete estimator/IEEE refinement remain separate unfinished obligations.
-/
namespace N2P3.PersonalERPActualConsumer
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPTransport
open N2P3.PersonalERPScalar N2P3.PersonalERPScalarFields
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def consumeActualScalar (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (physical : Physical) (kind : String) (sourceFlag : Value)
    (kwargs record : Fields) (state : NamedState) : Result Bound := do
  let restored := copyNamedState state
  liftCheck (ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind")
  let tensor ← liftCheck (tensorAt restored "waveform_channels")
  let selected ← liftCheck (readSelected physical.channels tensor)
  let model ← liftCheck (CurrentERPState.fromState p ref (copyFields record) (consumerERP restored))
  let recipe ← fromKwargsScalar s (copyFields kwargs)
  liftCheck (bindERP recipe kind sourceFlag physical selected model)

theorem scalar_selector_failure_precedes_kwargs (s : Primitives) (p : TemporalPrimitives)
    (ref : ReferencePrimitives) (physical : Physical) (flag : Value) (kwargs record : Fields)
    (arrays : ArrayFields) (selected : Tensor) (h : selected.dtype ≠ "torch.int64") :
    consumeActualScalar s p ref physical "target_only_structured_erp" flag kwargs record
      (producerERP arrays selected) = .error (.inherited (.sourceReject "ERP selected dtype")) := by
  simp [consumeActualScalar, liftCheck, ensure,
    PersonalERPCurrentConsumer.cloned_producer_selector_reads_before_NumPy, readSelected, h]

theorem plain_actual_consumer_from_independent_fields_and_atoms
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (h : Conditions f) (a : ScalarAtoms s f.raw) (physical : Physical) (indices : List ℕ)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (hn : indices ≠ []) (hu : indices.Nodup) (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / f.width) :
    consumeActualScalar s p ref physical "target_only_structured_erp" (.boolean false)
      (recordFields f.raw)
      (exportRecord (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))
      (producerERP (linearArrays coefficient intercept) (selectedTensor indices)) =
      .ok ⟨f.recipe, indices, plainModel f indices.length physical.times 4 physical.baseline coefficient intercept⟩ := by
  have hselected := selected_indices_preserve_order physical.channels indices hn hu hr
  have hreader := PersonalERPCurrentConsumer.independent_plain_fields_feed_current_reader p ref f physical indices
    coefficient intercept h.width hn ht hbins hf hb
  have hrecipe := canonical_kwargs_from_independent_fields_and_atoms s f h hf.1 a
  have hbind := bind_plain_from_fields f.recipe physical indices
    (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept)
    rfl rfl rfl rfl rfl rfl
  simp [consumeActualScalar, liftCheck, ensure, copyFields_preserves,
    PersonalERPCurrentConsumer.cloned_producer_selector_reads_before_NumPy,
    hselected, hreader, hrecipe, hbind]

/-- No constructor-result premise: arbitrary kwargs reach the actual Scalar
continuation only after the independently derived selector/current reader. -/
theorem plain_reader_then_actual_scalar_formula
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (physical : Physical) (indices : List ℕ) (kwargs : Fields) (flag : Value)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (hw : 0 < f.width) (hn : indices ≠ []) (hu : indices.Nodup)
    (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / f.width) :
    consumeActualScalar s p ref physical "target_only_structured_erp" flag kwargs
      (exportRecord (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))
      (producerERP (linearArrays coefficient intercept) (selectedTensor indices)) =
      (fromKwargsScalar s kwargs).bind (fun recipe => liftCheck (bindERP recipe
        "target_only_structured_erp" flag physical indices
        (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))) := by
  have hs := selected_indices_preserve_order physical.channels indices hn hu hr
  have hi := PersonalERPCurrentConsumer.independent_plain_fields_feed_current_reader p ref f physical indices
    coefficient intercept hw hn ht hbins hf hb
  simp [consumeActualScalar, copyFields_preserves, liftCheck, ensure,
    PersonalERPCurrentConsumer.cloned_producer_selector_reads_before_NumPy, hs, hi]

def fixtureActual (s : Primitives) (kwargs : Fields) : Result Bound :=
  consumeActualScalar s fixturePrimitives fixtureReference PersonalERPCurrentConsumer.fixturePhysical
    "target_only_structured_erp" (.boolean false) kwargs PersonalERPCurrentConsumer.fixtureRecord
    (producerERP PersonalERPCurrentConsumer.fixtureArrays PersonalERPCurrentConsumer.fixtureTensor)

theorem fixture_reader_then_arbitrary_raw_scalar (s : Primitives) (r : RawRecipe) :
    fixtureActual s (Fields.ofList (rawFields r)) =
      (validateScalar s r).bind (fun recipe => liftCheck (bindERP recipe "target_only_structured_erp"
        (.boolean false) PersonalERPCurrentConsumer.fixturePhysical [0] PersonalERPCurrentConsumer.fixtureModel)) := by
  have h := plain_reader_then_actual_scalar_formula s fixturePrimitives fixtureReference (pooledFields 1)
    PersonalERPCurrentConsumer.fixturePhysical [0] (Fields.ofList (rawFields r)) (.boolean false)
    ERPState.fixtureCoefficient (-1) (by norm_num [pooledFields]) (by decide) (by simp)
    (by simp [PersonalERPCurrentConsumer.fixturePhysical]) (by decide) (by simp [pooledFields])
    (by norm_num [PersonalERPCurrentConsumer.fixturePhysical])
    (by norm_num [PersonalERPCurrentConsumer.fixturePhysical, pooledFields])
  simpa [fixtureActual, PersonalERPCurrentConsumer.fixtureRecord, PersonalERPCurrentConsumer.fixtureModel,
    PersonalERPCurrentConsumer.fixtureArrays, PersonalERPCurrentConsumer.fixtureTensor,
    PersonalERPCurrentConsumer.exported_plain_record_is_actual_field_record,
    kwargs_reconstruct_before_scalar, PersonalERPCurrentConsumer.fixturePhysical, pooledFields] using h

theorem actual_temperature_exception_after_independent_reader (s : Primitives) (r : RawRecipe)
    (exception : AtomicException) (ha : r.targetAdapterConfig = .null)
    (hw : r.weights ≠ .text "adapter") (hl : r.candidateLossWeight = .null)
    (hn : numericNonBool r.candidateLossTemperature = true)
    (he : s.numpyIsFinite r.candidateLossTemperature = .error exception) :
    fixtureActual s (Fields.ofList (rawFields r)) =
      .error (.atomic "candidate_loss_temperature" .numpyFinite exception) := by
  rw [fixture_reader_then_arbitrary_raw_scalar,
    temperature_exception_from_single_numpy_atom s r exception ha hw hl hn he]
  rfl

def hugeActualRaw : RawRecipe := { (pooledFields 1).raw with candidateLossTemperature := huge100 }
def sequenceActualRaw : RawRecipe := { (pooledFields 1).raw with initialization := .sequence .nil }
def objectActualRaw : RawRecipe := { (pooledFields 1).raw with initialization := .object .nil }

theorem changing_initialization_preserves_individual_scalar_atoms (s : Primitives)
    (r : RawRecipe) (value : Value) (a : ScalarAtoms s r) :
    ScalarAtoms s { r with initialization := value } :=
  ⟨a.temperatureTag, a.priorTag, a.numpyTemperature, a.numpyPrior,
    a.temperatureNegative, a.priorNegative, a.temperatureNonpositive,
    a.priorNonzero, a.mathPrior, a.priorNonnegative, a.epochBound, a.batchBound,
    a.learningRateBound, a.updateBound, a.mathDefaultLoss, a.defaultLossNonnegative,
    a.mathTemperature, a.temperaturePositive⟩

theorem actual_huge_temperature_typeerror_is_preserved :
    fixtureActual observedPrimitives (Fields.ofList (rawFields hugeActualRaw)) =
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  apply actual_temperature_exception_after_independent_reader
  · rfl
  · simp [hugeActualRaw, ERPFields.raw, pooledFields]
  · rfl
  · rfl
  · exact observed_numpy_huge_typeerror

theorem actual_sequence_option_raises_after_scalar_prefix :
    fixtureActual observedPrimitives (Fields.ofList (rawFields sequenceActualRaw)) =
      .error (.atomic "recipe initialization" .setMembership .typeError) := by
  rw [fixture_reader_then_arbitrary_raw_scalar,
    sequence_initialization_from_hashability_after_atoms observedPrimitives sequenceActualRaw .nil
      rfl (by simp [sequenceActualRaw, ERPFields.raw, pooledFields]) rfl rfl
      (changing_initialization_preserves_individual_scalar_atoms observedPrimitives
        (pooledFields 1).raw (.sequence .nil) (pooled_fixture_scalar_atoms 1))]
  rfl

theorem actual_object_option_raises_after_scalar_prefix :
    fixtureActual observedPrimitives (Fields.ofList (rawFields objectActualRaw)) =
      .error (.atomic "recipe initialization" .setMembership .typeError) := by
  rw [fixture_reader_then_arbitrary_raw_scalar,
    object_initialization_from_hashability_after_atoms observedPrimitives objectActualRaw .nil
      rfl (by simp [objectActualRaw, ERPFields.raw, pooledFields]) rfl rfl
      (changing_initialization_preserves_individual_scalar_atoms observedPrimitives
        (pooledFields 1).raw (.object .nil) (pooled_fixture_scalar_atoms 1))]
  rfl

theorem actual_double_prefix_refusal_precedes_huge_scalar :
    consumeActualScalar observedPrimitives fixturePrimitives fixtureReference
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      (Fields.ofList (rawFields hugeActualRaw)) PersonalERPCurrentConsumer.fixtureRecord
      (producerERP PersonalERPCurrentConsumer.doublePrefixedFixtureArrays PersonalERPCurrentConsumer.fixtureTensor) =
      .error (.inherited (.sourceReject "ERP state keys")) := by
  have hs : readSelected PersonalERPCurrentConsumer.fixturePhysical.channels
      PersonalERPCurrentConsumer.fixtureTensor = .ok [0] :=
    selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [consumeActualScalar, liftCheck, ensure, copyFields_preserves,
    copyNamedState_preserves, tensorAt, PersonalERPCurrentConsumer.producer_selector_retains_Torch_tensor,
    hs, PersonalERPCurrentConsumer.double_prefix_reaches_current_inner_key_refusal]

def consumeCandidateScalar (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (physical : Physical) {Digest : Type} (candidate : PersonalERPFit.Candidate Digest) : Result Bound := do
  let artifact ← liftCheck (match candidate.erp with
    | none => .error (.fieldFailure "ERP candidate artifact")
    | some artifact => .ok artifact)
  let transferred := decide (candidate.recipe.raw.initialization = .text "source")
  let temporal := temporalTransfer (PersonalERPFit.artifactModel artifact)
  consumeActualScalar s p ref physical
    (producedKind transferred candidate.adapter.isSome temporal candidate.erp.isSome)
    (sourceFlag transferred temporal) (recordFields candidate.recipe.raw)
    (exportRecord (PersonalERPFit.artifactModel artifact)) (PersonalERPFit.candidateState candidate)

theorem constructed_fit_candidate_feeds_scalar_consumer
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives) (physical : Physical)
    {N C T : ℕ} {Digest : Type} (recipe : Recipe) (sfreq : ℝ)
    (training : PersonalERPAvailability.Mask N) (produced : PersonalERPFit.Produced N C T Digest) :
    consumeCandidateScalar s p ref physical (PersonalERPFit.buildCandidate recipe sfreq training produced) =
      consumeActualScalar s p ref physical
        (PersonalERPFit.producedModelKind recipe sfreq training produced)
        (PersonalERPFit.producedSourceFlag recipe produced)
        (recordFields recipe.raw) (exportRecord (PersonalERPFit.artifactModel produced.artifact))
        (producerERP produced.artifact.numeric.arrays (selectedTensor (produced.selection.channels.map Fin.val))) := by
  simp [consumeCandidateScalar, liftCheck, PersonalERPFit.buildCandidate,
    PersonalERPFit.ERP_assignments_derive_only_ERP, PersonalERPFit.producedModelKind,
    PersonalERPFit.producedSourceFlag, PersonalERPFit.candidateState, afterAdapter]

def fitThenConsumeScalar {N C T : ℕ} {HashState Digest : Type}
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives) (fit : PersonalERPFit.NumericalFit)
    (ceilProduct : PersonalERPAvailability.CeilProduct)
    (hash : PersonalERPAvailability.HashOperations HashState Digest) (encode : PersonalERPAvailability.ArrayEncoders)
    (recipe : Recipe) (sfreq : ℝ) (baseline : Value) (processing : Fields)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : PersonalERPAvailability.Mask N) (mask : PersonalERPAvailability.Observed N C) : Result Bound := do
  let candidate ← liftCheck (PersonalERPFit.fitCandidate p fit ceilProduct hash encode recipe sfreq baseline none
    processing values labels training mask)
  consumeCandidateScalar s p ref ⟨C, T, baseline⟩ candidate

theorem independent_fit_candidate_loads_actual_scalar
    {N C T : ℕ} {Digest : Type} (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (h : Conditions f) (a : ScalarAtoms s f.raw) (baseline : Value) (sfreq : ℝ)
    (training : PersonalERPAvailability.Mask N) (selection : PersonalERPAvailability.TrainingSelection N C)
    (invocation : PersonalERPAvailability.ERPFitInvocation
      (PersonalERPAvailability.availableRows selection.rows).length selection.channels.length T Digest)
    (minimum : ℤ) (localAvailability : Bool)
    (coefficient : Fin ((PersonalERPCurrentConsumer.selectedIndices selection).length * (T / f.width)) → ℝ)
    (intercept : ℝ) (hn : selection.channels ≠ []) (hu : selection.channels.Nodup)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (ht : 0 < T) (hbins : 0 < T / f.width) :
    consumeCandidateScalar s p ref ⟨C, T, baseline⟩
      (PersonalERPFit.buildCandidate f.recipe sfreq training
        (PersonalERPCurrentConsumer.plainProduced f baseline selection invocation minimum localAvailability coefficient intercept)) =
      .ok ⟨f.recipe, PersonalERPCurrentConsumer.selectedIndices selection,
        plainModel f (PersonalERPCurrentConsumer.selectedIndices selection).length T 4 baseline coefficient intercept⟩ := by
  have hi : PersonalERPCurrentConsumer.selectedIndices selection ≠ [] := by
    intro he
    have hz := congrArg List.length he
    simp [PersonalERPCurrentConsumer.selectedIndices] at hz
    exact hn hz
  have hcore := plain_actual_consumer_from_independent_fields_and_atoms s p ref f h a
    ⟨C, T, baseline⟩ (PersonalERPCurrentConsumer.selectedIndices selection) coefficient intercept hi
    (PersonalERPCurrentConsumer.selected_indices_unique_from_individual_channels selection hu)
    (PersonalERPCurrentConsumer.selected_indices_have_original_physical_bounds selection) hf hb ht hbins
  rw [constructed_fit_candidate_feeds_scalar_consumer]
  have hk := PersonalERPFit.metadata_kind_derived_from_initialization_and_component_assignments f.recipe sfreq training
    (PersonalERPCurrentConsumer.plainProduced f baseline selection invocation minimum localAvailability coefficient intercept) rfl
  have hs := PersonalERPFit.metadata_source_flag_derived_from_initialization_and_basis f.recipe
    (PersonalERPCurrentConsumer.plainProduced f baseline selection invocation minimum localAvailability coefficient intercept) rfl
  rw [hk, hs]
  exact hcore

/-- The producer header follows independent raw fields; FitSteps supplies only
single numerical operations/guards. ScalarAtoms supplies only single scalar
operations. Neither whole fit/reader/recipe success is an input premise. -/
theorem raw_fit_operations_to_actual_scalar_consumer
    {N C T : ℕ} {HashState Digest : Type} (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (fit : PersonalERPFit.NumericalFit) (ceilProduct : PersonalERPAvailability.CeilProduct)
    (hash : PersonalERPAvailability.HashOperations HashState Digest) (encode : PersonalERPAvailability.ArrayEncoders)
    (f : ERPFields) (h : Conditions f) (a : ScalarAtoms s f.raw) (sfreq : ℝ) (baseline : Value)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : PersonalERPAvailability.Mask N) (mask : PersonalERPAvailability.Observed N C)
    (selection : PersonalERPAvailability.TrainingSelection N C) (minimum : ℤ) (localAvailability : Bool)
    (coefficient : Fin ((PersonalERPCurrentConsumer.selectedIndices selection).length * (T / f.width)) → ℝ)
    (intercept : ℝ) (hnonempty : ∃ i, training i = true)
    (hrequested : 2 < (PersonalERPAvailability.availableRows training).length ∧
      PersonalERPFit.bothClasses (PersonalERPFit.labelsAtRows labels (PersonalERPAvailability.availableRows training)))
    (hselection : PersonalERPAvailability.trainingSelection ceilProduct processing training mask = .ok selection)
    (hfallback : selection.usedUnion = true →
      2 < (PersonalERPAvailability.availableRows selection.rows).length ∧
      PersonalERPFit.bothClasses (PersonalERPFit.labelsAtRows labels (PersonalERPAvailability.availableRows selection.rows)))
    (hchannels : selection.channels ≠ []) (hunique : selection.channels.Nodup) (hwidth : f.width ≤ T)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbaseline : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (hsteps : PersonalERPFit.FitSteps fit ⟨f.family, f.width, 4, baseline, none⟩
      (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection).values
      (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection).labels
      ⟨linearArrays coefficient intercept, none, none⟩)
    (hminimum : PersonalERPAvailability.minimumCount ceilProduct processing C = .ok minimum)
    (hlocal : PersonalERPAvailability.inferenceLocal processing = .ok localAvailability) :
    fitThenConsumeScalar s p ref fit ceilProduct hash encode f.recipe sfreq baseline processing values labels training mask =
      .ok ⟨f.recipe, PersonalERPCurrentConsumer.selectedIndices selection,
        plainModel f (PersonalERPCurrentConsumer.selectedIndices selection).length T 4 baseline coefficient intercept⟩ := by
  have hp := PersonalERPFit.produce_from_independent_raw_fields p fit ceilProduct hash encode
    f.recipe baseline .absent processing values labels training mask f.family f.width selection
    ⟨linearArrays coefficient intercept, none, none⟩ minimum localAvailability hnonempty hrequested
    rfl rfl h.width hbaseline hf.1 hselection hfallback hchannels hsteps hminimum hlocal
  have hproduced : PersonalERPFit.produce p fit ceilProduct hash encode f.recipe baseline none
      processing values labels training mask =
      .ok (PersonalERPCurrentConsumer.plainProduced f baseline selection
        (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection)
        minimum localAvailability coefficient intercept) := by
    simpa only [PersonalERPCurrentConsumer.plainProduced, PersonalERPCurrentConsumer.selectedIndices, List.length_map] using hp
  rw [fitThenConsumeScalar, PersonalERPFit.fitCandidate, hproduced]
  exact independent_fit_candidate_loads_actual_scalar s p ref f h a baseline sfreq training selection
    (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection)
    minimum localAvailability coefficient intercept hchannels hunique hf hbaseline
    (lt_of_lt_of_le h.width hwidth) (Nat.div_pos hwidth h.width)

theorem nonempty_fit_scalar_current_consumer :
    fitThenConsumeScalar observedPrimitives fixturePrimitives fixtureReference PersonalERPFit.fitFixtureNumericalOperation
      PersonalERPFit.fitFixtureCeil PersonalERPFit.fitFixtureHash PersonalERPFit.fitFixtureEncoding
      (pooledFields 1).recipe 64 (.text "none") PersonalERPFit.fitFixtureProcessing
      PersonalERPFit.fitFixtureValues PersonalERPFit.fitFixtureLabels (fun _ => true) PersonalERPFit.fitFixtureMask =
      .ok ⟨(pooledFields 1).recipe, [1], plainModel (pooledFields 1) 1 1 4 (.text "none") (fun _ => 2) (-1)⟩ := by
  rw [fitThenConsumeScalar, PersonalERPFit.nonempty_fit_candidate_has_no_simultaneous_components]
  have h := independent_fit_candidate_loads_actual_scalar observedPrimitives fixturePrimitives fixtureReference
    (pooledFields 1) (pooled_fields_satisfy_individual_checks 1 (by norm_num)) (pooled_fixture_scalar_atoms 1)
    (.text "none") 64 (fun _ => true) PersonalERPFit.fitFixtureSelection PersonalERPFit.fitFixtureInvocation 1 true
    (fun _ => 2) (-1) (by simp [PersonalERPFit.fitFixtureSelection])
    (by simp [PersonalERPFit.fitFixtureSelection]) (by decide) (by simp) (by norm_num) (by norm_num [pooledFields])
  simpa [PersonalERPCurrentConsumer.plainProduced, PersonalERPCurrentConsumer.selectedIndices,
    PersonalERPFit.fitFixtureProduced, PersonalERPFit.fitFixtureArtifact, PersonalERPFit.fitFixtureHeader,
    PersonalERPFit.fitFixturePayload, PersonalERPFit.fitFixtureSelection, plainModel, pooledFields] using h

end
end N2P3.PersonalERPActualConsumer
