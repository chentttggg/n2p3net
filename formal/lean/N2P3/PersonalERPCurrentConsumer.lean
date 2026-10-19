import N2P3.PersonalERPRecipe
import N2P3.PersonalERPTransport
import N2P3.PersonalERPFit

/-!
# ERP producer arrays through the actual current personal numeric consumer

Source: calibration_candidates.py:311-332, calibrated.py:550-551,619-641,
structured_erp.py:from_state. The public physical/identity/QC/execution entry
precedes this numeric subroute and is not asserted by this file. Supported
decoded finite float64 arrays and correctly represented Torch tensors are the
same explicit mathematical domain as Transport and Recipe. Actual library
bytes, casts, aliasing, storage, unsupported scalar conversion/failure and
floating-point score bounds remain unrefined.

The selector stays Torch int64 while ONLY ERP-prefixed tensors are individually
viewed as NumPy float64 arrays. Prefix stripping happens once. A current reader
result is derived from independently supplied coefficient/intercept fields,
shape/binning, ordered selector bounds and the actual recipe fields. No whole
load, roundtrip or prediction equality is an input assumption. These general
pooled/Toeplitz results do not replace temporal/xDAWN or exception obligations,
nor complete P10, the full physical constructor P06, or actual fitting P12.
-/

namespace N2P3.PersonalERPCurrentConsumer
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPTransport

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def consumeNumericBranch (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (physical : Physical) (kind : String) (sourceFlag : Value)
    (kwargs record : Fields) (state : NamedState) : Check Bound := do
  let restored := copyNamedState state
  ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind"
  let tensor ← tensorAt restored "waveform_channels"
  let selected ← readSelected physical.channels tensor
  let model ← CurrentERPState.fromState p ref (copyFields record) (consumerERP restored)
  let recipe ← fromKwargs (copyFields kwargs)
  bindERP recipe kind sourceFlag physical selected model

theorem producer_selector_retains_Torch_tensor (arrays : ArrayFields) (selected : Tensor) :
    lookupTensor "waveform_channels" (producerERP arrays selected) = some selected := by
  induction arrays with
  | nil => simp [producerERP, prefixArrays, lookupTensor]
  | cons entry tail ih =>
    rcases entry with ⟨name, array⟩
    simp [producerERP, prefixArrays, lookupTensor, waveform_name_cannot_have_ERP_prefix] at ih ⊢
    exact ih

theorem cloned_producer_selector_reads_before_NumPy (arrays : ArrayFields) (selected : Tensor) :
    tensorAt (copyNamedState (producerERP arrays selected)) "waveform_channels" = .ok selected := by
  rw [copyNamedState_preserves]
  simp [tensorAt, producer_selector_retains_Torch_tensor]

theorem actual_consumer_views_only_prefixed_arrays (arrays : ArrayFields) (selected : Tensor) :
    consumerERP (copyNamedState (producerERP arrays selected)) = numpyArrays arrays :=
  erp_only_clone_recovers_each_numpy_field arrays selected

theorem exported_plain_record_is_actual_field_record (f : ERPFields) (channels times : ℕ)
    (baseline : Value) (coefficient : Fin (channels * (times / f.width)) → ℝ) (intercept : ℝ) :
    exportRecord (plainModel f channels times 4 baseline coefficient intercept) =
      plainRecord f.family f.width 4 channels times baseline := by
  simp [exportRecord, plainModel, plainRecord]

theorem independent_plain_fields_feed_current_reader (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (physical : Physical) (indices : List ℕ)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (hw : 0 < f.width) (hn : indices ≠ []) (ht : 0 < physical.times)
    (hbins : 0 < physical.times / f.width) (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none") :
    CurrentERPState.fromState p ref
      (exportRecord (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))
      (consumerERP (copyNamedState (producerERP (linearArrays coefficient intercept) (selectedTensor indices)))) =
      .ok (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept) := by
  rw [exported_plain_record_is_actual_field_record, transported_linear_fields, CurrentERPState.fromState]
  rw [plain_family_constructs p ref f.family f.width 4 indices.length physical.times physical.baseline
    coefficient intercept hw (by norm_num) (List.length_pos_iff.mpr hn) ht hbins hf hb]
  exact CurrentERPState.other_families_current_guard_preserves _ hf.2

theorem plain_producer_clone_current_consumer_from_fields (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (h : Conditions f) (physical : Physical) (indices : List ℕ)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (hn : indices ≠ []) (hu : indices.Nodup) (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / f.width) :
    consumeNumericBranch p ref physical "target_only_structured_erp" (.boolean false)
      (recordFields f.raw)
      (exportRecord (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))
      (producerERP (linearArrays coefficient intercept) (selectedTensor indices)) =
      .ok ⟨f.recipe, indices, plainModel f indices.length physical.times 4 physical.baseline coefficient intercept⟩ := by
  have hselected := selected_indices_preserve_order physical.channels indices hn hu hr
  have hreader := independent_plain_fields_feed_current_reader p ref f physical indices
    coefficient intercept h.width hn ht hbins hf hb
  have hrecipe := canonical_record_constructor f h
  have hbind := bind_plain_from_fields f.recipe physical indices
    (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept)
    rfl rfl rfl rfl rfl rfl
  simp [consumeNumericBranch, copyFields_preserves, ensure,
    cloned_producer_selector_reads_before_NumPy, hselected, hreader, hrecipe, hbind]

/-- Exact-real numerical counterpart, after independently derived admission.
The epoch is the already-admitted ordered channel view; availability and
IEEE/BLAS behavior are separate obligations. -/
theorem plain_current_consumer_scores_from_independent_fields (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (h : Conditions f) (physical : Physical) (indices : List ℕ)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (epoch : ERPFeatures.Epoch indices.length physical.times)
    (hn : indices ≠ []) (hu : indices.Nodup) (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / f.width) :
    (consumeNumericBranch p ref physical "target_only_structured_erp" (.boolean false)
      (recordFields f.raw)
      (exportRecord (plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))
      (producerERP (linearArrays coefficient intercept) (selectedTensor indices))).bind
      (fun bound => affineScore bound.model.state (indices.length * (physical.times / f.width))
        (ERPFeatures.binFeatures f.width epoch)) =
      .ok (ERPFeatures.evaluate ⟨coefficient, intercept⟩ (ERPFeatures.binFeatures f.width epoch)) := by
  rw [plain_producer_clone_current_consumer_from_fields p ref f h physical indices coefficient intercept
    hn hu hr hf hb ht hbins]
  exact affine_score_from_independent_fields coefficient intercept _

def fixtureArrays : ArrayFields := linearArrays ERPState.fixtureCoefficient (-1)
def fixturePhysical : Physical := ⟨1, 1, .text "none"⟩
def fixtureRecord : Fields := plainRecord .pooled 1 4 1 1 (.text "none")
def fixtureKwargs : Fields := recordFields (pooledFields 1).raw
def fixtureTensor : Tensor := selectedTensor [0]
def fixtureModel : Model := plainModel (pooledFields 1) 1 1 4 (.text "none") ERPState.fixtureCoefficient (-1)

theorem selector_dtype_refusal_precedes_ERP_array_conversion (p : TemporalPrimitives)
    (ref : ReferencePrimitives) (physical : Physical) (sourceFlag : Value)
    (kwargs record : Fields) (arrays : ArrayFields) (selected : Tensor)
    (hdtype : selected.dtype ≠ "torch.int64") :
    consumeNumericBranch p ref physical "target_only_structured_erp" sourceFlag kwargs record
      (producerERP arrays selected) = .error (.sourceReject "ERP selected dtype") := by
  simp [consumeNumericBranch, ensure, cloned_producer_selector_reads_before_NumPy, readSelected, hdtype]

def floatingSelector : Tensor := ⟨"torch.float64", [1], [.number 0]⟩

theorem floating_selector_does_not_load_supported_ERP_fields :
    consumeNumericBranch fixturePrimitives fixtureReference fixturePhysical
      "target_only_structured_erp" (.boolean false) fixtureKwargs fixtureRecord
      (producerERP fixtureArrays floatingSelector) = .error (.sourceReject "ERP selected dtype") := by
  apply selector_dtype_refusal_precedes_ERP_array_conversion
  decide

def doublePrefixedFixtureArrays : ArrayFields :=
  fixtureArrays.map (fun entry => ("erp." ++ entry.1, entry.2))

theorem double_prefix_is_stripped_exactly_once :
    stateKeys (consumerERP (producerERP doublePrefixedFixtureArrays fixtureTensor)) =
      ["erp.coefficient", "erp.intercept"] := by
  rw [erp_only_producer_recovers_each_numpy_field]
  rfl

theorem double_prefix_reaches_current_inner_key_refusal :
    CurrentERPState.fromState fixturePrimitives fixtureReference fixtureRecord
      (consumerERP (producerERP doublePrefixedFixtureArrays fixtureTensor)) =
      .error (.sourceReject "ERP state keys") := by
  rw [erp_only_producer_recovers_each_numpy_field, CurrentERPState.fromState]
  simp [ERPState.fromState, fixtureRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get,
    optionalTemporal, commonHeader, readFamily, Family.text, readGeometry, positiveInt, required, ensure,
    exactStateKeys, expectedKeys, stateKeys, numpyArrays, doublePrefixedFixtureArrays, fixtureArrays,
    linearArrays, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

theorem nonempty_current_consumer_preserves_selector_and_scores_five :
    (consumeNumericBranch fixturePrimitives fixtureReference fixturePhysical
      "target_only_structured_erp" (.boolean false) fixtureKwargs fixtureRecord
      (producerERP fixtureArrays fixtureTensor)).bind
      (fun bound => affineScore bound.model.state 1 (fun _ => 3)) = .ok 5 := by
  have h := plain_current_consumer_scores_from_independent_fields fixturePrimitives fixtureReference
    (pooledFields 1) (pooled_fields_satisfy_individual_checks 1 (by norm_num)) fixturePhysical [0]
    ERPState.fixtureCoefficient (-1) (fun _ _ => 3) (by decide) (by simp) (by simp [fixturePhysical])
    (by decide) (Or.inl (by decide)) (by norm_num [fixturePhysical]) (by norm_num [fixturePhysical, pooledFields])
  have he : ERPFeatures.binFeatures 1 (fun _ : Fin 1 => fun _ : Fin 1 => (3 : ℝ)) = (fun _ => 3) := by
    funext i
    simp [ERPFeatures.binFeatures, ERPFeatures.flatten, ERPFeatures.complete_bin_formula]
  have hc : (3 : ℝ) * 2 + -1 = 5 := by norm_num
  simpa [fixtureArrays, fixtureTensor, fixturePhysical, fixtureRecord, fixtureKwargs,
    exported_plain_record_is_actual_field_record, pooledFields, he, ERPFeatures.evaluate,
    ERPState.fixtureCoefficient, dotProduct, Fin.sum_univ_one, hc] using h

def consumeCandidate (p : TemporalPrimitives) (ref : ReferencePrimitives) (physical : Physical)
    {Digest : Type} (candidate : PersonalERPFit.Candidate Digest) : Check Bound := do
  let artifact ← match candidate.erp with
    | none => .error (.fieldFailure "ERP candidate artifact")
    | some artifact => .ok artifact
  let transferred := decide (candidate.recipe.raw.initialization = .text "source")
  let temporal := temporalTransfer (PersonalERPFit.artifactModel artifact)
  consumeNumericBranch p ref physical
    (producedKind transferred candidate.adapter.isSome temporal candidate.erp.isSome)
    (sourceFlag transferred temporal) (recordFields candidate.recipe.raw)
    (exportRecord (PersonalERPFit.artifactModel artifact)) (PersonalERPFit.candidateState candidate)

theorem constructed_fit_candidate_feeds_actual_numeric_consumer
    (p : TemporalPrimitives) (ref : ReferencePrimitives) (physical : Physical)
    {N C T : ℕ} {Digest : Type} (recipe : Recipe) (sfreq : ℝ)
    (training : PersonalERPAvailability.Mask N) (produced : PersonalERPFit.Produced N C T Digest) :
    consumeCandidate p ref physical (PersonalERPFit.buildCandidate recipe sfreq training produced) =
      consumeNumericBranch p ref physical
        (PersonalERPFit.producedModelKind recipe sfreq training produced)
        (PersonalERPFit.producedSourceFlag recipe produced)
        (recordFields recipe.raw) (exportRecord (PersonalERPFit.artifactModel produced.artifact))
        (producerERP produced.artifact.numeric.arrays (selectedTensor (produced.selection.channels.map Fin.val))) := by
  simp [consumeCandidate, PersonalERPFit.buildCandidate, PersonalERPFit.ERP_assignments_derive_only_ERP,
    PersonalERPFit.producedModelKind, PersonalERPFit.producedSourceFlag,
    PersonalERPFit.candidateState, afterAdapter]

def selectedIndices {N C : ℕ} (selection : PersonalERPAvailability.TrainingSelection N C) : List ℕ :=
  selection.channels.map Fin.val

theorem selected_indices_have_original_physical_bounds {N C : ℕ}
    (selection : PersonalERPAvailability.TrainingSelection N C) :
    ∀ i ∈ selectedIndices selection, i < C := by
  intro i hi
  obtain ⟨channel, _, rfl⟩ := List.mem_map.mp hi
  exact channel.isLt

theorem selected_indices_unique_from_individual_channels {N C : ℕ}
    (selection : PersonalERPAvailability.TrainingSelection N C) (hu : selection.channels.Nodup) :
    (selectedIndices selection).Nodup := by
  exact hu.map Fin.val_injective

/-- Independent plain numerical fields and the already-defined fit invocation.
The map-length expression equals the source selected-channel count exactly.
This does not assert that any numerical estimator produced these fields. -/
def plainProduced {N C T : ℕ} {Digest : Type} (f : ERPFields) (baseline : Value)
    (selection : PersonalERPAvailability.TrainingSelection N C)
    (invocation : PersonalERPAvailability.ERPFitInvocation
      (PersonalERPAvailability.availableRows selection.rows).length selection.channels.length T Digest)
    (minimum : ℤ) (localAvailability : Bool)
    (coefficient : Fin ((selectedIndices selection).length * (T / f.width)) → ℝ) (intercept : ℝ) :
    PersonalERPFit.Produced N C T Digest :=
  ⟨f.family, selection, invocation,
    ⟨⟨f.family, f.width, 4, baseline, none⟩, (selectedIndices selection).length, T,
      ⟨linearArrays coefficient intercept, none, none⟩⟩, minimum, localAvailability⟩

theorem plain_produced_shape_is_source_selected_count {N C T : ℕ} {Digest : Type}
    (f : ERPFields) (baseline : Value) (selection : PersonalERPAvailability.TrainingSelection N C)
    (invocation : PersonalERPAvailability.ERPFitInvocation
      (PersonalERPAvailability.availableRows selection.rows).length selection.channels.length T Digest)
    (minimum : ℤ) (localAvailability : Bool)
    (coefficient : Fin ((selectedIndices selection).length * (T / f.width)) → ℝ) (intercept : ℝ) :
    (plainProduced f baseline selection invocation minimum localAvailability coefficient intercept).artifact.channels =
      selection.channels.length := by simp [plainProduced, selectedIndices]

theorem independently_constructed_plain_fit_candidate_loads_current
    {N C T : ℕ} {Digest : Type} (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : ERPFields) (h : Conditions f) (baseline : Value) (sfreq : ℝ)
    (training : PersonalERPAvailability.Mask N) (selection : PersonalERPAvailability.TrainingSelection N C)
    (invocation : PersonalERPAvailability.ERPFitInvocation
      (PersonalERPAvailability.availableRows selection.rows).length selection.channels.length T Digest)
    (minimum : ℤ) (localAvailability : Bool)
    (coefficient : Fin ((selectedIndices selection).length * (T / f.width)) → ℝ) (intercept : ℝ)
    (hn : selection.channels ≠ []) (hu : selection.channels.Nodup)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (ht : 0 < T) (hbins : 0 < T / f.width) :
    consumeCandidate p ref ⟨C, T, baseline⟩
      (PersonalERPFit.buildCandidate f.recipe sfreq training
        (plainProduced f baseline selection invocation minimum localAvailability coefficient intercept)) =
      .ok ⟨f.recipe, selectedIndices selection,
        plainModel f (selectedIndices selection).length T 4 baseline coefficient intercept⟩ := by
  have hindices : selectedIndices selection ≠ [] := by
    intro hi
    have hl := congrArg List.length hi
    have hz : selection.channels.length = 0 := by simpa [selectedIndices] using hl
    exact hn (List.length_eq_zero_iff.mp hz)
  have hcore := plain_producer_clone_current_consumer_from_fields p ref f h ⟨C, T, baseline⟩
    (selectedIndices selection) coefficient intercept hindices
    (selected_indices_unique_from_individual_channels selection hu)
    (selected_indices_have_original_physical_bounds selection) hf hb ht hbins
  rw [constructed_fit_candidate_feeds_actual_numeric_consumer]
  have hk := PersonalERPFit.metadata_kind_derived_from_initialization_and_component_assignments
    f.recipe sfreq training (plainProduced f baseline selection invocation minimum localAvailability coefficient intercept) rfl
  have hs := PersonalERPFit.metadata_source_flag_derived_from_initialization_and_basis
    f.recipe (plainProduced f baseline selection invocation minimum localAvailability coefficient intercept) rfl
  rw [hk, hs]
  change consumeNumericBranch p ref ⟨C, T, baseline⟩ "target_only_structured_erp" (.boolean false)
    (recordFields f.raw) (exportRecord (plainModel f (selectedIndices selection).length T 4 baseline coefficient intercept))
    (producerERP (linearArrays coefficient intercept) (selectedTensor (selectedIndices selection))) = _
  exact hcore

def fitThenConsume {N C T : ℕ} {HashState Digest : Type}
    (p : TemporalPrimitives) (ref : ReferencePrimitives) (fit : PersonalERPFit.NumericalFit)
    (ceilProduct : PersonalERPAvailability.CeilProduct)
    (hash : PersonalERPAvailability.HashOperations HashState Digest)
    (encode : PersonalERPAvailability.ArrayEncoders) (recipe : Recipe) (sfreq : ℝ) (baseline : Value)
    (processing : Fields) (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool)
    (training : PersonalERPAvailability.Mask N) (mask : PersonalERPAvailability.Observed N C) : Check Bound := do
  let candidate ← PersonalERPFit.fitCandidate p fit ceilProduct hash encode recipe sfreq baseline none
    processing values labels training mask
  consumeCandidate p ref ⟨C, T, baseline⟩ candidate

/-- Full composition of the explicit fit-core dispatch and current numeric
consumer, on the supported plain-family domain. All conditions describe an
individual field/guard/small operation. Numerical success supplies arrays and
rank scalars only, never an already-admitted candidate or desired score. -/
theorem fit_core_to_current_consumer_from_individual_operations
    {N C T : ℕ} {HashState Digest : Type} (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (fit : PersonalERPFit.NumericalFit) (ceilProduct : PersonalERPAvailability.CeilProduct)
    (hash : PersonalERPAvailability.HashOperations HashState Digest) (encode : PersonalERPAvailability.ArrayEncoders)
    (f : ERPFields) (h : Conditions f) (sfreq : ℝ) (baseline : Value) (processing : Fields)
    (values : ERPFeatures.Batch N C T) (labels : Fin N → Bool) (training : PersonalERPAvailability.Mask N)
    (mask : PersonalERPAvailability.Observed N C) (selection : PersonalERPAvailability.TrainingSelection N C)
    (minimum : ℤ) (localAvailability : Bool)
    (coefficient : Fin ((selectedIndices selection).length * (T / f.width)) → ℝ) (intercept : ℝ)
    (hnonempty : ∃ i, training i = true)
    (hrequested : 2 < (PersonalERPAvailability.availableRows training).length ∧
      PersonalERPFit.bothClasses (PersonalERPFit.labelsAtRows labels (PersonalERPAvailability.availableRows training)))
    (hselection : PersonalERPAvailability.trainingSelection ceilProduct processing training mask = .ok selection)
    (hfallback : selection.usedUnion = true →
      2 < (PersonalERPAvailability.availableRows selection.rows).length ∧
      PersonalERPFit.bothClasses (PersonalERPFit.labelsAtRows labels (PersonalERPAvailability.availableRows selection.rows)))
    (hchannels : selection.channels ≠ []) (hunique : selection.channels.Nodup)
    (hwidth : f.width ≤ T) (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbaseline : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (hsteps : PersonalERPFit.FitSteps fit ⟨f.family, f.width, 4, baseline, none⟩
      (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection).values
      (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection).labels
      ⟨linearArrays coefficient intercept, none, none⟩)
    (hminimum : PersonalERPAvailability.minimumCount ceilProduct processing C = .ok minimum)
    (hlocal : PersonalERPAvailability.inferenceLocal processing = .ok localAvailability) :
    fitThenConsume p ref fit ceilProduct hash encode f.recipe sfreq baseline processing values labels training mask =
      .ok ⟨f.recipe, selectedIndices selection,
        plainModel f (selectedIndices selection).length T 4 baseline coefficient intercept⟩ := by
  have hp := PersonalERPFit.produce_from_independent_raw_fields p fit ceilProduct hash encode
    f.recipe baseline .absent processing values labels training mask f.family f.width selection
    ⟨linearArrays coefficient intercept, none, none⟩ minimum localAvailability
    hnonempty hrequested rfl rfl h.width hbaseline hf.1 hselection hfallback hchannels hsteps hminimum hlocal
  have hproduced : PersonalERPFit.produce p fit ceilProduct hash encode f.recipe baseline none
      processing values labels training mask =
      .ok (plainProduced f baseline selection
        (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection)
        minimum localAvailability coefficient intercept) := by
    simpa only [plainProduced, selectedIndices, List.length_map] using hp
  rw [fitThenConsume, PersonalERPFit.fitCandidate, hproduced]
  exact independently_constructed_plain_fit_candidate_loads_current p ref f h baseline sfreq training selection
    (PersonalERPAvailability.erpFitInvocation hash encode values labels mask training selection)
    minimum localAvailability coefficient intercept hchannels hunique hf hbaseline
    (lt_of_lt_of_le h.width hwidth) (Nat.div_pos hwidth h.width)

/-- A nonempty declared-operation instance traverses selected original rows,
constructor fields, fit stages, component assignments, export/clone, Torch
selector, NumPy ERP fields and the current inner reader. This is not an
execution of the production estimator or an authentication of physical data. -/
theorem nonempty_fit_core_reaches_current_consumer :
    fitThenConsume fixturePrimitives fixtureReference PersonalERPFit.fitFixtureNumericalOperation
      PersonalERPFit.fitFixtureCeil PersonalERPFit.fitFixtureHash PersonalERPFit.fitFixtureEncoding
      (pooledFields 1).recipe 64 (.text "none") PersonalERPFit.fitFixtureProcessing
      PersonalERPFit.fitFixtureValues PersonalERPFit.fitFixtureLabels (fun _ => true)
      PersonalERPFit.fitFixtureMask =
      .ok ⟨(pooledFields 1).recipe, [1], plainModel (pooledFields 1) 1 1 4 (.text "none") (fun _ => 2) (-1)⟩ := by
  rw [fitThenConsume, PersonalERPFit.nonempty_fit_candidate_has_no_simultaneous_components]
  have h := independently_constructed_plain_fit_candidate_loads_current fixturePrimitives fixtureReference
    (pooledFields 1) (pooled_fields_satisfy_individual_checks 1 (by norm_num)) (.text "none") 64
    (fun _ => true) PersonalERPFit.fitFixtureSelection PersonalERPFit.fitFixtureInvocation 1 true
    (fun _ => 2) (-1) (by simp [PersonalERPFit.fitFixtureSelection])
    (by simp [PersonalERPFit.fitFixtureSelection]) (by decide) (by simp) (by norm_num) (by norm_num [pooledFields])
  simpa [plainProduced, selectedIndices, PersonalERPFit.fitFixtureProduced,
    PersonalERPFit.fitFixtureArtifact, PersonalERPFit.fitFixtureHeader, PersonalERPFit.fitFixturePayload,
    PersonalERPFit.fitFixtureSelection, plainModel, pooledFields] using h

end
end N2P3.PersonalERPCurrentConsumer
