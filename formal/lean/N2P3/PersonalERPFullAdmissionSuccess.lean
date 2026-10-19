import N2P3.PersonalERPFullAdmission
import N2P3.PersonalRecipeRecord

/-!
# Complete plain ERP admission from independent fields and atomic replies

The pooled/Toeplitz, no-QC and receipt-free subroute constructs actual saved
recipe fields, embeds them into the extended constructor domain, and derives
the complete successful Predictor. Tensor.numpy preserves an independent raw
native-array observation; float64 conversion and copy are separate operations.
No complete reader, recipe, copy-loop or admission-success premise is supplied.

These decoded field observations and individual native operation replies do
not certify native bytes, IEEE, arbitrary Python subclasses, other metadata
branches, earlier raw physical ingress, training or scientific efficacy.
-/
namespace N2P3.PersonalERPFullAdmissionSuccess
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData N2P3.PersonalState
open N2P3.PersonalERPAdmission N2P3.PersonalERPFullAdmission
noncomputable section
set_option maxHeartbeats 1000000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def initialization : Family → PersonalTargetRecipe.Initialization
  | .pooled => .pooled | .toeplitz => .toeplitz | .temporal => .temporal | .xdawn => .xdawn

theorem initialization_text_is_the_independent_family (family : Family) :
    (initialization family).text = family.text := by cases family <;> rfl

/-- Every variable JSON scalar comes from its own supplied field. The adapter
is absent on this plain LDA branch, and all remaining literal policies match
the source treatment rather than an ERP-only constructor oracle. -/
def arguments (f : PersonalERPRecipe.ERPFields) : PersonalTargetRecipeFields.Arguments := {
  name := f.name, initialization := initialization f.family,
  statistics := f.inputStatistics, buffers := f.buffers, affine := "frozen", weights := "none", dropout := "off",
  batchSize := PersonalRecipePython.embed f.batchSize, lr := PersonalRecipePython.embed f.lr,
  weightDecay := PersonalRecipePython.embed f.weightDecay, epochs := PersonalRecipePython.embed f.epochs,
  maxUpdates := PersonalRecipePython.embed f.maxUpdates, seed := PersonalRecipePython.embed f.seed,
  refit := .boolean f.refit, adapter := .raw .null, lossWeight := .null,
  temperature := .floating (.finite f.temperature), prior := .floating (.finite 0),
  width := f.width, kind := f.kind, rank := f.rank }

theorem complete_JSON_field_image_embeds_from_independent_fields (f : PersonalERPRecipe.ERPFields) :
    PersonalRecipePython.embedFields (Fields.ofList (PersonalERPRecipe.rawFields f.raw)) =
      PersonalTargetRecipe.allRecordFields (arguments f).raw := by
  cases hk : f.kind <;> cases hr : f.rank <;>
    simp [PersonalERPRecipe.rawFields, PersonalERPRecipe.ERPFields.raw, arguments,
      PersonalTargetRecipeFields.Arguments.raw, PersonalTargetRecipe.allRecordFields,
      PersonalTargetRecipe.adapterRecordValue, PersonalRecipePython.embedFields,
      PersonalRecipePython.embed, Fields.ofList, PersonalRecipePython.Fields.ofList,
      initialization_text_is_the_independent_family, hk, hr]

/-- This is the actual conditionally omitted saved record, not the complete
21-key dataclass image. The None and numeric-default decisions are explicit. -/
theorem actual_saved_JSON_fields_embed_as_the_conditional_record_image (f : PersonalERPRecipe.ERPFields)
    (hk : f.kind = none) (hr : f.rank = none) :
    PersonalRecipePython.embedFields (PersonalERPRecipe.recordFields f.raw) =
      PersonalRecipeRecord.image (arguments f).raw (decide (f.temperature = 1)) true (decide (f.width = 1)) := by
  rw [PersonalRecipeRecord.conditional_pop_image_is_the_actual_saved_field_list]
  by_cases ht : f.temperature = 1 <;> by_cases hw : f.width = 1 <;>
    simp +decide [PersonalERPRecipe.recordFields, PersonalERPRecipe.rawFields, PersonalERPRecipe.ERPFields.raw,
      PersonalRecipeRecord.explicitImage, arguments, PersonalTargetRecipeFields.Arguments.raw,
      PersonalRecipePython.embedFields, PersonalRecipePython.embed,
      Fields.ofList, PersonalRecipePython.Fields.ofList, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue,
      initialization_text_is_the_independent_family, hk, hr, ht, hw, Int.cast_natCast, Nat.cast_eq_one]

theorem actual_saved_default_restoration_recovers_the_independent_plain_fields (f : PersonalERPRecipe.ERPFields)
    (hk : f.kind = none) (hr : f.rank = none) :
    PersonalRecipeRecord.restored (arguments f).raw (decide (f.temperature = 1)) true (decide (f.width = 1)) =
      (arguments f).raw := by
  by_cases ht : f.temperature = 1 <;> by_cases hw : f.width = 1 <;>
    simp [PersonalRecipeRecord.restored, arguments, PersonalTargetRecipeFields.Arguments.raw, ht, hw, hk, hr]

theorem saved_JSON_kwargs_reconstruct_from_the_individual_source_fields (f : PersonalERPRecipe.ERPFields)
    (hk : f.kind = none) (hr : f.rank = none) :
    PersonalTargetRecipe.readArgs (PersonalRecipePython.embedFields (PersonalERPRecipe.recordFields f.raw)) =
      .ok (arguments f).raw := by
  rw [actual_saved_JSON_fields_embed_as_the_conditional_record_image f hk hr,
    PersonalRecipeRecord.omitted_defaults_reconstruct_the_explicit_normalized_raw,
    actual_saved_default_restoration_recovers_the_independent_plain_fields f hk hr]

theorem full_recipe_from_actual_saved_fields_and_each_scalar_operation
    (p : PersonalRecipePython.Primitives) (f : PersonalERPRecipe.ERPFields)
    (hk : f.kind = none) (hr : f.rank = none)
    (fields : PersonalTargetRecipeFields.Conditions (arguments f))
    (atoms : PersonalTargetRecipeFields.AtomicSuccess p (arguments f)) :
    PersonalTargetRecipe.fromKwargs p (PersonalRecipePython.embedFields (PersonalERPRecipe.recordFields f.raw)) =
      .ok ((arguments f).recipe none) := by
  rw [PersonalTargetRecipe.fromKwargs, saved_JSON_kwargs_reconstruct_from_the_individual_source_fields f hk hr]
  simp only [Except.bind, Bind.bind]
  exact PersonalTargetRecipeFields.complete_constructor_from_independent_fields p (arguments f) none fields atoms
    (.absent rfl (by simp [arguments, PersonalTargetRecipeFields.Arguments.raw]))

def selectedTorch (indices : List ℕ) : TorchTensor :=
  ⟨"torch.int64", ⟨.integer, [indices.length], indices.map (fun index : ℕ => Atom.integer (index : ℤ)), by simp⟩⟩

/-- Preserve the producer order: ERP coefficient, ERP intercept, selector. -/
def state (coefficient intercept : TorchTensor) (indices : List ℕ) : TorchState :=
  ⟨[("erp.coefficient", coefficient), ("erp.intercept", intercept), ("waveform_channels", selectedTorch indices)], by simp⟩

def contract (channels times : ℕ) (rawArgs : Fields) : Fields := Fields.ofList
  [("n_channels", .integer channels), ("n_times", .integer times), ("preprocessing", .object rawArgs)]

theorem selector_integer_cells_decode_in_order (indices : List ℕ) :
    ((selectedTorch indices).image.values.mapM (fun value => do
      let index ← PersonalERPScalar.liftCheck (PersonalERPData.strictInteger value)
      pure index.toNat)) = .ok indices := by
  change List.mapM (fun value => do
    let index ← PersonalERPScalar.liftCheck (PersonalERPData.strictInteger value)
    pure index.toNat) (indices.map (fun index : ℕ => Atom.integer (index : ℤ))) = Except.ok indices
  induction indices with
  | nil => rfl
  | cons index tail ih =>
      rw [List.map_cons, List.mapM_cons, ih]
      simp [strictInteger, PersonalERPScalar.liftCheck]

theorem actual_selector_from_each_native_operation (p : Operations) (coefficient intercept : TorchTensor)
    (indices : List ℕ) (channels times : ℕ) (rawArgs : Fields) (hn : indices ≠ [])
    (unique : p.admission.selectedUniqueCount (selectedTorch indices) = .ok indices.length)
    (negative : p.admission.selectedAnyNegative (selectedTorch indices) = .ok false)
    (upper : p.admission.selectedAnyTooHigh (selectedTorch indices) (.integer channels) = .ok false) :
    PersonalERPAdmissionRuntime.selected p.admission (state coefficient intercept indices).entries
      (contract channels times rawArgs) = .ok indices := by
  have found : requiredTorch (state coefficient intercept indices).entries "waveform_channels" =
      .ok (selectedTorch indices) := by simp [state, requiredTorch, lookupTorch]
  have dtype : (selectedTorch indices).dtype = "torch.int64" := rfl
  have length : (selectedTorch indices).image.values.length = indices.length := by simp [selectedTorch]
  have shape : (selectedTorch indices).image.shape = [(selectedTorch indices).image.values.length] ∧
      (selectedTorch indices).image.values ≠ [] := by simp [selectedTorch, hn]
  have channelsField : PersonalERPModelRuntime.required (contract channels times rawArgs) "n_channels" =
      .ok (.integer channels) := by
    simp [contract, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  simp only [PersonalERPAdmissionRuntime.selected, found, PersonalERPScalar.liftCheck,
    Except.mapError, Except.bind, Bind.bind, PersonalERPScalar.require, dtype, shape, length,
    unique, negative, upper, channelsField, if_true, Except.pure, Pure.pure]
  simp only [true_and]
  rw [if_pos shape.2]
  exact selector_integer_cells_decode_in_order indices

theorem numpy_arrays_from_each_original_dtype_tensor (p : Operations) (coefficient intercept : TorchTensor)
    (indices : List ℕ) (coefficientArray interceptArray : NativeArray)
    (hc : p.numpy coefficient = .ok coefficientArray) (hi : p.numpy intercept = .ok interceptArray) :
    numpyState p (state coefficient intercept indices).entries =
      .ok [("coefficient", coefficientArray), ("intercept", interceptArray)] := by
  simp [numpyState, state, ERPState.stripERP, liftNative, hc, hi]

def finiteCoefficient {d : ℕ} (coefficient : Fin d → ℝ) : PersonalERPFitOutcome.Array64 :=
  PersonalERPFitOutcome.fromFiniteArray (PersonalERPTransport.vectorArray coefficient)

def finiteIntercept (intercept : ℝ) : PersonalERPFitOutcome.Array64 :=
  PersonalERPFitOutcome.fromFiniteArray (PersonalERPTransport.scalarArray intercept)

theorem all_array_conversions_and_copies_from_separate_operations (p : Operations) {d : ℕ}
    (coefficient : Fin d → ℝ) (intercept : ℝ) (coefficientArray interceptArray : NativeArray)
    (hc : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient))
    (hcc : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient))
    (hi : p.asFloat64 interceptArray = .ok (finiteIntercept intercept))
    (hic : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept)) :
    castAll p [("coefficient", coefficientArray), ("intercept", interceptArray)] =
      .ok (PersonalERPFitOutcome.fromFiniteArrays (PersonalERPTransport.linearArrays coefficient intercept)) := by
  change castAll p [("coefficient", coefficientArray), ("intercept", interceptArray)] =
    .ok [("coefficient", finiteCoefficient coefficient), ("intercept", finiteIntercept intercept)]
  simp [castAll, liftNative, hc, hcc, hi, hic]

theorem independent_plain_array_cells_are_finite {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) :
    PersonalERPFitOutcome.arraysFinite
      (PersonalERPFitOutcome.fromFiniteArrays (PersonalERPTransport.linearArrays coefficient intercept)) = true := by
  simp [PersonalERPFitOutcome.arraysFinite, PersonalERPFitOutcome.arrayFinite,
    PersonalERPFitOutcome.fromFiniteArrays, PersonalERPFitOutcome.fromFiniteArray,
    PersonalERPTransport.linearArrays, PersonalERPTransport.vectorArray, PersonalERPTransport.scalarArray,
    PersonalERPFitOutcome.Cell.isFinite, Function.comp_def]

theorem independent_plain_array_cells_decode_to_the_actual_model_fields {d : ℕ}
    (coefficient : Fin d → ℝ) (intercept : ℝ) :
    PersonalERPFitOutcome.decodeArrays
      (PersonalERPFitOutcome.fromFiniteArrays (PersonalERPTransport.linearArrays coefficient intercept)) =
      ERPState.linearState coefficient intercept := by
  simp [PersonalERPFitOutcome.decodeArrays, PersonalERPFitOutcome.fromFiniteArrays,
    PersonalERPTransport.linearArrays, PersonalERPFitOutcome.decoded_finite_array_preserves_fields]
  rfl

theorem finite_plain_reader_tail_from_individual_numeric_fields
    (reference : ReferencePrimitives) (f : PersonalERPRecipe.ERPFields) (channels times : ℕ) (baseline : Value)
    (coefficient : Fin (channels * (times / f.width)) → ℝ) (intercept : ℝ)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn) :
    PersonalERPFitOutcome.finishFinite reference (plainRecord f.family f.width 4 channels times baseline)
      ⟨f.family, f.width, 4, baseline, none⟩ channels times (linearState coefficient intercept) =
      .ok (PersonalERPRecipe.plainModel f channels times 4 baseline coefficient intercept) := by
  cases he : f.family <;>
    simp_all [PersonalERPFitOutcome.finishFinite, shapeAt, tensorAt, lookupTensor, linearState,
      coefficientTensor, interceptTensor, realTensor, featureWidth, CurrentERPState.currentGuard,
      PersonalERPRecipe.plainModel, ensure]

theorem native_plain_reader_from_each_cast_and_copy (p : Operations) (f : PersonalERPRecipe.ERPFields)
    (channels times : ℕ) (baseline : Value) (coefficient : Fin (channels * (times / f.width)) → ℝ) (intercept : ℝ)
    (coefficientArray interceptArray : NativeArray) (hw : 0 < f.width) (hc : 0 < channels) (ht : 0 < times)
    (hbins : 0 < times / f.width) (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (castCoefficient : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient))
    (copyCoefficient : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient))
    (castIntercept : p.asFloat64 interceptArray = .ok (finiteIntercept intercept))
    (copyIntercept : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept)) :
    readNativeOutcome p (.object (plainRecord f.family f.width 4 channels times baseline))
      [("coefficient", coefficientArray), ("intercept", interceptArray)] =
      .ok (PersonalERPRecipe.plainModel f channels times 4 baseline coefficient intercept) := by
  have header := plain_common_header p.admission.original.temporal f.family f.width 4 channels times baseline
    hw (by norm_num) hf.1 hb
  have converted := all_array_conversions_and_copies_from_separate_operations p coefficient intercept
    coefficientArray interceptArray castCoefficient copyCoefficient castIntercept copyIntercept
  have finite := independent_plain_array_cells_are_finite coefficient intercept
  have decoded := independent_plain_array_cells_decode_to_the_actual_model_fields coefficient intercept
  have finished := finite_plain_reader_tail_from_individual_numeric_fields p.admission.original.reference f channels times baseline
    coefficient intercept hf
  have schema : fieldDefault (plainRecord f.family f.width 4 channels times baseline) "schema" =
      .text "n2p3_structured_erp/1" ∧
      fieldDefault (plainRecord f.family f.width 4 channels times baseline) "feature_order" = .text "channel_prime" := by
    simp [plainRecord, fieldDefault, Fields.ofList, Fields.get]
  have temporal : optionalTemporal p.admission.original.temporal
      (plainRecord f.family f.width 4 channels times baseline) = .ok none := by
    simp [optionalTemporal, plainRecord, Fields.ofList, Fields.get]
  have geometry : readGeometry (plainRecord f.family f.width 4 channels times baseline) = .ok (channels,times) := by
    simp [readGeometry, plainRecord, Fields.ofList, Fields.get, encodeNats, required, ensure, hc, ht]
  have tail : PersonalAdmission.PyEq
      (fieldDefault (plainRecord f.family f.width 4 channels times baseline) "discarded_tail_samples")
      (.integer (times % f.width)) := by
    simp [plainRecord, fieldDefault, Fields.ofList, Fields.get,
      PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]
  simp [readNativeOutcome, getMapping, liftCheck, schema, temporal, header, geometry, ensure, hbins,
    tail, expectedKeys, hf.2, converted, finite, decoded, finished]

theorem plain_actual_binding_from_independent_contract_fields (p : Operations) (authority : String)
    (f : PersonalERPRecipe.ERPFields) (indices : List ℕ) (channels times : ℕ) (rawArgs : Fields)
    (baseline : Value) (coefficient : Fin (indices.length * (times / f.width)) → ℝ) (intercept : ℝ)
    (hb : rawArgs.get "baseline_mode" = some baseline) :
    bindActual p (noQualityRawMetadata authority (contract channels times rawArgs)
        (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      (contract channels times rawArgs) indices
      (PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept) ((arguments f).recipe none) =
      .ok ⟨(arguments f).recipe none, indices, PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept⟩ := by
  simp +decide [bindActual, noQualityRawMetadata, contract, PersonalERPRecipe.plainModel,
    PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
    indexedMapping, require, liftRuntime, PersonalERPScalar.require, PersonalERPScalar.liftCheck,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, arguments,
    PersonalTargetRecipeFields.Arguments.recipe, initialization_text_is_the_independent_family,
    PersonalERPRecipe.usesSource, fieldDefault, hb]

theorem outer_finiteness_from_each_named_tensor (p : Operations) (coefficient intercept : TorchTensor)
    (indices : List ℕ)
    (hc : p.admission.original.tensorIsFinite coefficient = .ok true)
    (hi : p.admission.original.tensorIsFinite intercept = .ok true)
    (hs : p.admission.original.tensorIsFinite (selectedTorch indices) = .ok true) :
    allStateFinite p.admission.original (state coefficient intercept indices).entries = .ok () := by
  simp [allStateFinite, state, hc, hi, hs, ensure]

theorem full_plain_continuation_is_successful_from_independent_fields_and_atoms
    (p : Operations) (authority : String) (f : PersonalERPRecipe.ERPFields)
    (indices : List ℕ) (channels times : ℕ) (rawArgs : Fields) (baseline : Value) (device : String)
    (coefficient : Fin (indices.length * (times / f.width)) → ℝ) (intercept : ℝ)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (hn : indices ≠ []) (hw : 0 < f.width) (ht : 0 < times) (hbins : 0 < times / f.width)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbaseline : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (hk : f.kind = none) (hr : f.rank = none) (baselineField : rawArgs.get "baseline_mode" = some baseline)
    (fields : PersonalTargetRecipeFields.Conditions (arguments f))
    (atoms : PersonalTargetRecipeFields.AtomicSuccess p.recipe (arguments f))
    (unique : p.admission.selectedUniqueCount (selectedTorch indices) = .ok indices.length)
    (negative : p.admission.selectedAnyNegative (selectedTorch indices) = .ok false)
    (upper : p.admission.selectedAnyTooHigh (selectedTorch indices) (.integer channels) = .ok false)
    (numpyCoefficient : p.numpy coefficientTensor = .ok coefficientArray)
    (numpyIntercept : p.numpy interceptTensor = .ok interceptArray)
    (castCoefficient : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient))
    (copyCoefficient : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient))
    (castIntercept : p.asFloat64 interceptArray = .ok (finiteIntercept intercept))
    (copyIntercept : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept))
    (finiteCoefficientTensor : p.admission.original.tensorIsFinite coefficientTensor = .ok true)
    (finiteInterceptTensor : p.admission.original.tensorIsFinite interceptTensor = .ok true)
    (finiteSelector : p.admission.original.tensorIsFinite (selectedTorch indices) = .ok true) :
    finishERP p (noQualityRawMetadata authority (contract channels times rawArgs) (PersonalERPRecipe.recordFields f.raw)
        (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      (state coefficientTensor interceptTensor indices).entries device none (contract channels times rawArgs) =
      .ok ⟨noQualityRawMetadata authority (contract channels times rawArgs) (PersonalERPRecipe.recordFields f.raw)
          (.object (plainRecord f.family f.width 4 indices.length times baseline)),
        (state coefficientTensor interceptTensor indices).entries, device, none,
        ⟨(arguments f).recipe none, indices, PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept⟩⟩ := by
  have selected := actual_selector_from_each_native_operation p coefficientTensor interceptTensor indices channels times rawArgs hn
    unique negative upper
  have numpy := numpy_arrays_from_each_original_dtype_tensor p coefficientTensor interceptTensor indices coefficientArray interceptArray
    numpyCoefficient numpyIntercept
  have reader := native_plain_reader_from_each_cast_and_copy p f indices.length times baseline coefficient intercept
    coefficientArray interceptArray hw (List.length_pos_iff.mpr hn) ht hbins hf hbaseline
    castCoefficient copyCoefficient castIntercept copyIntercept
  have recipe := full_recipe_from_actual_saved_fields_and_each_scalar_operation p.recipe f hk hr fields atoms
  have bound := plain_actual_binding_from_independent_contract_fields p authority f indices channels times rawArgs baseline
    coefficient intercept baselineField
  have finite := outer_finiteness_from_each_named_tensor p coefficientTensor interceptTensor indices
    finiteCoefficientTensor finiteInterceptTensor finiteSelector
  have kind : PersonalERPModelRuntime.required (noQualityRawMetadata authority (contract channels times rawArgs)
      (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      "model_kind" = .ok (.text "target_only_structured_erp") := by
    simp [noQualityRawMetadata, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  have artifact : PersonalERPModelRuntime.required (noQualityRawMetadata authority (contract channels times rawArgs)
      (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      "erp_artifact" = .ok (.object (plainRecord f.family f.width 4 indices.length times baseline)) := by
    simp [noQualityRawMetadata, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  have kwargs : PersonalERPModelRuntime.required (noQualityRawMetadata authority (contract channels times rawArgs)
      (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      "recipe" = .ok (.object (PersonalERPRecipe.recordFields f.raw)) := by
    simp [noQualityRawMetadata, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  simp [finishERP, kind, erpKind, liftRuntime, selected, artifact, numpy, reader, kwargs, recipe,
    kwargsMapping, liftRecipe, bound, liftCheck, finite]

theorem constructERP_plain_noQC_receipt_free_is_completely_successful
    (p : Operations) (authority : String) (f : PersonalERPRecipe.ERPFields)
    (indices : List ℕ) (channels times : ℕ) (rawArgs : Fields) (baseline : Value) (requested device : String)
    (coefficient : Fin (indices.length * (times / f.width)) → ℝ) (intercept : ℝ)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (ha : authority ≠ "") (known : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (hn : indices ≠ []) (hw : 0 < f.width) (ht : 0 < times) (hbins : 0 < times / f.width)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbaseline : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (hk : f.kind = none) (hr : f.rank = none) (baselineField : rawArgs.get "baseline_mode" = some baseline)
    (fields : PersonalTargetRecipeFields.Conditions (arguments f))
    (atoms : PersonalTargetRecipeFields.AtomicSuccess p.recipe (arguments f))
    (json : p.admission.original.finiteJSON (.object (noQualityRawMetadata authority (contract channels times rawArgs)
        (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))) =
      .ok (.object (noQualityRawMetadata authority (contract channels times rawArgs)
        (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))))
    (cloneCoefficient : p.admission.original.detachCPUClone coefficientTensor = .ok coefficientTensor)
    (cloneIntercept : p.admission.original.detachCPUClone interceptTensor = .ok interceptTensor)
    (cloneSelector : p.admission.original.detachCPUClone (selectedTorch indices) = .ok (selectedTorch indices))
    (deviceResult : p.admission.original.device requested = .ok device)
    (unique : p.admission.selectedUniqueCount (selectedTorch indices) = .ok indices.length)
    (negative : p.admission.selectedAnyNegative (selectedTorch indices) = .ok false)
    (upper : p.admission.selectedAnyTooHigh (selectedTorch indices) (.integer channels) = .ok false)
    (numpyCoefficient : p.numpy coefficientTensor = .ok coefficientArray)
    (numpyIntercept : p.numpy interceptTensor = .ok interceptArray)
    (castCoefficient : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient))
    (copyCoefficient : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient))
    (castIntercept : p.asFloat64 interceptArray = .ok (finiteIntercept intercept))
    (copyIntercept : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept))
    (finiteCoefficientTensor : p.admission.original.tensorIsFinite coefficientTensor = .ok true)
    (finiteInterceptTensor : p.admission.original.tensorIsFinite interceptTensor = .ok true)
    (finiteSelector : p.admission.original.tensorIsFinite (selectedTorch indices) = .ok true) :
    constructERP p (.object (noQualityRawMetadata authority (contract channels times rawArgs) (PersonalERPRecipe.recordFields f.raw)
        (.object (plainRecord f.family f.width 4 indices.length times baseline))))
      (state coefficientTensor interceptTensor indices) requested =
      .ok ⟨noQualityRawMetadata authority (contract channels times rawArgs) (PersonalERPRecipe.recordFields f.raw)
          (.object (plainRecord f.family f.width 4 indices.length times baseline)),
        (state coefficientTensor interceptTensor indices).entries, device, none,
        ⟨(arguments f).recipe none, indices, PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept⟩⟩ := by
  have clones : ∀ entry ∈ (state coefficientTensor interceptTensor indices).entries,
      p.admission.original.detachCPUClone entry.2 = .ok entry.2 := by
    intro entry member
    simp only [state, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with he | he | he <;> subst entry <;> assumption
  rw [prefix_from_fields_and_each_copy_operation p authority (contract channels times rawArgs)
    (PersonalERPRecipe.recordFields f.raw) rawArgs (.object (plainRecord f.family f.width 4 indices.length times baseline))
    (state coefficientTensor interceptTensor indices) requested device ha
    (by simp [contract, Fields.ofList, Fields.get]) known json clones deviceResult]
  exact full_plain_continuation_is_successful_from_independent_fields_and_atoms p authority f indices channels times rawArgs
    baseline device coefficient intercept coefficientTensor interceptTensor coefficientArray interceptArray hn hw ht hbins hf hbaseline
    hk hr baselineField fields atoms unique negative upper numpyCoefficient numpyIntercept castCoefficient copyCoefficient castIntercept
    copyIntercept finiteCoefficientTensor finiteInterceptTensor finiteSelector

/-- Selector admission observes only this contract field; extra physical
contract keys retain their real values. -/
theorem selector_unread_contract_fields_do_not_interfere
    (p : Operations) (state : List (String × TorchTensor)) (first second : Fields)
    (channelsField : first.get "n_channels" = second.get "n_channels") :
    PersonalERPAdmissionRuntime.selected p.admission state first =
      PersonalERPAdmissionRuntime.selected p.admission state second := by
  simp only [PersonalERPAdmissionRuntime.selected, PersonalERPModelRuntime.required,
    StateLedger.required, channelsField]

theorem plain_actual_binding_for_any_contract_from_its_read_fields (p : Operations) (authority : String)
    (f : PersonalERPRecipe.ERPFields) (indices : List ℕ) (times : ℕ) (rawArgs physicalContract : Fields)
    (baseline : Value) (coefficient : Fin (indices.length * (times / f.width)) → ℝ) (intercept : ℝ)
    (timesField : physicalContract.get "n_times" = some (.integer times))
    (preprocessingField : physicalContract.get "preprocessing" = some (.object rawArgs))
    (hb : rawArgs.get "baseline_mode" = some baseline) :
    bindActual p (noQualityRawMetadata authority physicalContract
        (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      physicalContract indices
      (PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept) ((arguments f).recipe none) =
      .ok ⟨(arguments f).recipe none, indices, PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept⟩ := by
  simp +decide [bindActual, noQualityRawMetadata, timesField, preprocessingField, PersonalERPRecipe.plainModel,
    PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
    indexedMapping, require, liftRuntime, PersonalERPScalar.require, PersonalERPScalar.liftCheck,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, arguments,
    PersonalTargetRecipeFields.Arguments.recipe, initialization_text_is_the_independent_family,
    PersonalERPRecipe.usesSource, fieldDefault, hb]

theorem full_plain_continuation_for_any_contract_from_independent_fields_and_atoms
    (p : Operations) (authority : String) (f : PersonalERPRecipe.ERPFields)
    (indices : List ℕ) (channels times : ℕ) (rawArgs physicalContract : Fields) (baseline : Value) (device : String)
    (coefficient : Fin (indices.length * (times / f.width)) → ℝ) (intercept : ℝ)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (hn : indices ≠ []) (hw : 0 < f.width) (ht : 0 < times) (hbins : 0 < times / f.width)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbaseline : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (hk : f.kind = none) (hr : f.rank = none) (baselineField : rawArgs.get "baseline_mode" = some baseline)
    (channelsField : physicalContract.get "n_channels" = some (.integer channels))
    (timesField : physicalContract.get "n_times" = some (.integer times))
    (preprocessingField : physicalContract.get "preprocessing" = some (.object rawArgs))
    (fields : PersonalTargetRecipeFields.Conditions (arguments f))
    (atoms : PersonalTargetRecipeFields.AtomicSuccess p.recipe (arguments f))
    (unique : p.admission.selectedUniqueCount (selectedTorch indices) = .ok indices.length)
    (negative : p.admission.selectedAnyNegative (selectedTorch indices) = .ok false)
    (upper : p.admission.selectedAnyTooHigh (selectedTorch indices) (.integer channels) = .ok false)
    (numpyCoefficient : p.numpy coefficientTensor = .ok coefficientArray)
    (numpyIntercept : p.numpy interceptTensor = .ok interceptArray)
    (castCoefficient : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient))
    (copyCoefficient : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient))
    (castIntercept : p.asFloat64 interceptArray = .ok (finiteIntercept intercept))
    (copyIntercept : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept))
    (finiteCoefficientTensor : p.admission.original.tensorIsFinite coefficientTensor = .ok true)
    (finiteInterceptTensor : p.admission.original.tensorIsFinite interceptTensor = .ok true)
    (finiteSelector : p.admission.original.tensorIsFinite (selectedTorch indices) = .ok true) :
    finishERP p (noQualityRawMetadata authority physicalContract (PersonalERPRecipe.recordFields f.raw)
        (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      (state coefficientTensor interceptTensor indices).entries device none physicalContract =
      .ok ⟨noQualityRawMetadata authority physicalContract (PersonalERPRecipe.recordFields f.raw)
          (.object (plainRecord f.family f.width 4 indices.length times baseline)),
        (state coefficientTensor interceptTensor indices).entries, device, none,
        ⟨(arguments f).recipe none, indices, PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept⟩⟩ := by
  have selected : PersonalERPAdmissionRuntime.selected p.admission
      (state coefficientTensor interceptTensor indices).entries physicalContract = .ok indices := by
    rw [selector_unread_contract_fields_do_not_interfere p
      (state coefficientTensor interceptTensor indices).entries physicalContract
      (contract channels times rawArgs) (by simp [contract, Fields.ofList, Fields.get, channelsField])]
    exact actual_selector_from_each_native_operation p coefficientTensor interceptTensor indices channels times rawArgs hn
      unique negative upper
  have numpy := numpy_arrays_from_each_original_dtype_tensor p coefficientTensor interceptTensor indices coefficientArray interceptArray
    numpyCoefficient numpyIntercept
  have reader := native_plain_reader_from_each_cast_and_copy p f indices.length times baseline coefficient intercept
    coefficientArray interceptArray hw (List.length_pos_iff.mpr hn) ht hbins hf hbaseline
    castCoefficient copyCoefficient castIntercept copyIntercept
  have recipe := full_recipe_from_actual_saved_fields_and_each_scalar_operation p.recipe f hk hr fields atoms
  have bound := plain_actual_binding_for_any_contract_from_its_read_fields p authority f indices times rawArgs physicalContract baseline
    coefficient intercept timesField preprocessingField baselineField
  have finite := outer_finiteness_from_each_named_tensor p coefficientTensor interceptTensor indices
    finiteCoefficientTensor finiteInterceptTensor finiteSelector
  have kind : PersonalERPModelRuntime.required (noQualityRawMetadata authority physicalContract
      (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      "model_kind" = .ok (.text "target_only_structured_erp") := by
    simp [noQualityRawMetadata, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  have artifact : PersonalERPModelRuntime.required (noQualityRawMetadata authority physicalContract
      (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      "erp_artifact" = .ok (.object (plainRecord f.family f.width 4 indices.length times baseline)) := by
    simp [noQualityRawMetadata, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  have kwargs : PersonalERPModelRuntime.required (noQualityRawMetadata authority physicalContract
      (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))
      "recipe" = .ok (.object (PersonalERPRecipe.recordFields f.raw)) := by
    simp [noQualityRawMetadata, PersonalERPModelRuntime.required, StateLedger.required, Fields.ofList, Fields.get,
      PersonalERPScalar.liftCheck]
  simp [finishERP, kind, erpKind, liftRuntime, selected, artifact, numpy, reader, kwargs, recipe,
    kwargsMapping, liftRecipe, bound, liftCheck, finite]

theorem constructERP_plain_for_any_contract_from_its_read_fields_and_atomic_operations
    (p : Operations) (authority : String) (f : PersonalERPRecipe.ERPFields)
    (indices : List ℕ) (channels times : ℕ) (rawArgs physicalContract : Fields) (baseline : Value) (requested device : String)
    (coefficient : Fin (indices.length * (times / f.width)) → ℝ) (intercept : ℝ)
    (coefficientTensor interceptTensor : TorchTensor) (coefficientArray interceptArray : NativeArray)
    (ha : authority ≠ "") (known : fieldKeys rawArgs ⊆ preprocessingKeywords)
    (hn : indices ≠ []) (hw : 0 < f.width) (ht : 0 < times) (hbins : 0 < times / f.width)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hbaseline : f.family ≠ .toeplitz ∨ baseline = .text "none")
    (hk : f.kind = none) (hr : f.rank = none) (baselineField : rawArgs.get "baseline_mode" = some baseline)
    (channelsField : physicalContract.get "n_channels" = some (.integer channels))
    (timesField : physicalContract.get "n_times" = some (.integer times))
    (preprocessingField : physicalContract.get "preprocessing" = some (.object rawArgs))
    (fields : PersonalTargetRecipeFields.Conditions (arguments f))
    (atoms : PersonalTargetRecipeFields.AtomicSuccess p.recipe (arguments f))
    (json : p.admission.original.finiteJSON (.object (noQualityRawMetadata authority physicalContract
        (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))) =
      .ok (.object (noQualityRawMetadata authority physicalContract
        (PersonalERPRecipe.recordFields f.raw) (.object (plainRecord f.family f.width 4 indices.length times baseline)))))
    (cloneCoefficient : p.admission.original.detachCPUClone coefficientTensor = .ok coefficientTensor)
    (cloneIntercept : p.admission.original.detachCPUClone interceptTensor = .ok interceptTensor)
    (cloneSelector : p.admission.original.detachCPUClone (selectedTorch indices) = .ok (selectedTorch indices))
    (deviceResult : p.admission.original.device requested = .ok device)
    (unique : p.admission.selectedUniqueCount (selectedTorch indices) = .ok indices.length)
    (negative : p.admission.selectedAnyNegative (selectedTorch indices) = .ok false)
    (upper : p.admission.selectedAnyTooHigh (selectedTorch indices) (.integer channels) = .ok false)
    (numpyCoefficient : p.numpy coefficientTensor = .ok coefficientArray)
    (numpyIntercept : p.numpy interceptTensor = .ok interceptArray)
    (castCoefficient : p.asFloat64 coefficientArray = .ok (finiteCoefficient coefficient))
    (copyCoefficient : p.copyFloat64 (finiteCoefficient coefficient) = .ok (finiteCoefficient coefficient))
    (castIntercept : p.asFloat64 interceptArray = .ok (finiteIntercept intercept))
    (copyIntercept : p.copyFloat64 (finiteIntercept intercept) = .ok (finiteIntercept intercept))
    (finiteCoefficientTensor : p.admission.original.tensorIsFinite coefficientTensor = .ok true)
    (finiteInterceptTensor : p.admission.original.tensorIsFinite interceptTensor = .ok true)
    (finiteSelector : p.admission.original.tensorIsFinite (selectedTorch indices) = .ok true) :
    constructERP p (.object (noQualityRawMetadata authority physicalContract (PersonalERPRecipe.recordFields f.raw)
        (.object (plainRecord f.family f.width 4 indices.length times baseline))))
      (state coefficientTensor interceptTensor indices) requested =
      .ok ⟨noQualityRawMetadata authority physicalContract (PersonalERPRecipe.recordFields f.raw)
          (.object (plainRecord f.family f.width 4 indices.length times baseline)),
        (state coefficientTensor interceptTensor indices).entries, device, none,
        ⟨(arguments f).recipe none, indices, PersonalERPRecipe.plainModel f indices.length times 4 baseline coefficient intercept⟩⟩ := by
  have clones : ∀ entry ∈ (state coefficientTensor interceptTensor indices).entries,
      p.admission.original.detachCPUClone entry.2 = .ok entry.2 := by
    intro entry member
    simp only [state, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with he | he | he <;> subst entry <;> assumption
  rw [prefix_from_fields_and_each_copy_operation p authority physicalContract
    (PersonalERPRecipe.recordFields f.raw) rawArgs (.object (plainRecord f.family f.width 4 indices.length times baseline))
    (state coefficientTensor interceptTensor indices) requested device ha
    preprocessingField known json clones deviceResult]
  exact full_plain_continuation_for_any_contract_from_independent_fields_and_atoms p authority f indices channels times rawArgs physicalContract
    baseline device coefficient intercept coefficientTensor interceptTensor coefficientArray interceptArray hn hw ht hbins hf hbaseline
    hk hr baselineField channelsField timesField preprocessingField fields atoms unique negative upper numpyCoefficient numpyIntercept castCoefficient copyCoefficient castIntercept
    copyIntercept finiteCoefficientTensor finiteInterceptTensor finiteSelector

end
end N2P3.PersonalERPFullAdmissionSuccess
