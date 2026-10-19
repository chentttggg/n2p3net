import N2P3.ERPState

/-!
# Per-field transport on the actual ERP-only producer branch

Source: calibration_candidates.py:310-332 and calibrated.py:523-550,618-640,
1144-1187. StructuredERP.export_state emits float64 arrays; the candidate
prefixes each name with `erp.`, converts those arrays to Torch tensors and
writes int64 waveform_channels. CalibratedPredictor clones each tensor, then
its ERP branch strips the prefix once and obtains NumPy arrays independently.

Array64 is a decoded finite float64 producer array, retaining shape and ordered
real cells. The functions below model the separate dtype labels and explicit
value-copy/projection steps. They DO NOT prove Torch/NumPy byte, stride, device,
alias or IEEE semantics; they are not definitions of those libraries. No
NumPy-to-Torch operation is assumed to preserve an entire model or prediction.
Every derived equality starts with independently written fields.

The ordinary ERP producer has no simultaneous adapter or LDA component.
The source's adapter assignment would replace the ERP map. That qualification
is explicit, and the overwrite counterexample is retained. Deriving the
component invariant from the full authorized fit is a separate caller task.
Metadata selection below preserves actual priority: source initialization,
then adapter, then source-temporal ERP, then target ERP, then waveform LDA.
-/

namespace N2P3.PersonalERPTransport

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState

noncomputable section

structure Array64 where
  shape : List ℕ
  cells : List ℝ
  cellCount : cells.length = shape.foldr (· * ·) 1

abbrev ArrayFields := List (String × Array64)

def numpyTensor (array : Array64) : Tensor := ERPState.realTensor array.shape array.cells

def torchTensor (array : Array64) : Tensor :=
  ⟨"torch.float64", array.shape, array.cells.map Value.number⟩

/-- A dtype projection on decoded float64 producer values, not a general numpy API. -/
def numpy64View (tensor : Tensor) : Tensor :=
  ⟨"float64", tensor.shape, tensor.values⟩

theorem torch_tensor_records_all_fields (array : Array64) :
    (torchTensor array).dtype = "torch.float64" ∧
    (torchTensor array).shape = array.shape ∧
    (torchTensor array).values = array.cells.map Value.number := ⟨rfl, rfl, rfl⟩

theorem torch_tensor_well_formed (array : Array64) : (torchTensor array).WellFormed := by
  constructor
  · simpa [Tensor.numel, torchTensor] using array.cellCount
  · intro v hv
    obtain ⟨x, _, rfl⟩ := List.mem_map.mp hv
    rfl

theorem per_tensor_numpy_view_recovers_array (array : Array64) :
    numpy64View (torchTensor array) = numpyTensor array := rfl

theorem detached_cpu_clone_preserves_fields (array : Array64) :
    copyTensor (torchTensor array) = torchTensor array := copyTensor_preserves _

theorem cloned_tensor_numpy_view (array : Array64) :
    numpy64View (copyTensor (torchTensor array)) = numpyTensor array := by
  rw [copyTensor_preserves]
  rfl

def prefixArrays (arrays : ArrayFields) : NamedState :=
  arrays.map (fun entry => ("erp." ++ entry.1, torchTensor entry.2))

def numpyArrays (arrays : ArrayFields) : NamedState :=
  arrays.map (fun entry => (entry.1, numpyTensor entry.2))

def producerERP (arrays : ArrayFields) (selected : Tensor) : NamedState :=
  prefixArrays arrays ++ [("waveform_channels", selected)]

def consumerERP (state : NamedState) : NamedState :=
  (ERPState.erpFields state).map (fun entry => (entry.1, numpy64View entry.2))

theorem numpy_dtype_is_separate_from_torch (array : Array64) :
    (torchTensor array).dtype ≠ (numpyTensor array).dtype := by
  change ("torch.float64" : String) ≠ "float64"
  decide

theorem all_prefixed_fields_are_read (arrays : ArrayFields) :
    ERPState.erpFields (prefixArrays arrays) =
      arrays.map (fun entry => (entry.1, torchTensor entry.2)) := by
  induction arrays with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name, array⟩
    simp only [prefixArrays, List.map_cons, ERPState.erpFields,
      ERPState.prefixed_name_strips_once]
    exact congrArg (List.cons (name, torchTensor array)) ih

theorem waveform_selector_is_not_inner_state (selected : Tensor) :
    ERPState.erpFields [("waveform_channels", selected)] = [] := rfl

theorem erp_only_producer_recovers_each_numpy_field (arrays : ArrayFields) (selected : Tensor) :
    consumerERP (producerERP arrays selected) = numpyArrays arrays := by
  simp only [consumerERP, producerERP, ERPState.erp_fields_append,
    all_prefixed_fields_are_read, waveform_selector_is_not_inner_state, List.append_nil,
    List.map_map, numpyArrays]
  rfl

theorem erp_only_clone_recovers_each_numpy_field (arrays : ArrayFields) (selected : Tensor) :
    consumerERP (copyNamedState (producerERP arrays selected)) = numpyArrays arrays := by
  rw [copyNamedState_preserves]
  exact erp_only_producer_recovers_each_numpy_field _ _

theorem normalized_numpy_array (array : Array64) :
    ERPState.normalizedTensor (numpyTensor array) = .ok (numpyTensor array) :=
  ERPState.normalized_real_tensor _ _

theorem normalized_numpy_fields (arrays : ArrayFields) :
    ERPState.normalizedState (numpyArrays arrays) = .ok (numpyArrays arrays) := by
  induction arrays with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name, array⟩
    change ERPState.normalizedState ((name, numpyTensor array) :: numpyArrays tail) = _
    simp only [ERPState.normalizedState, normalized_numpy_array, ih,
      Bind.bind, Except.bind, Pure.pure, Except.pure]
    rfl

theorem transported_inner_normalization (arrays : ArrayFields) (selected : Tensor) :
    ERPState.normalizedState (consumerERP (copyNamedState (producerERP arrays selected))) =
      .ok (numpyArrays arrays) := by
  rw [erp_only_clone_recovers_each_numpy_field]
  exact normalized_numpy_fields _

theorem transported_lookup_each_name (arrays : ArrayFields) (selected : Tensor) (name : String) :
    lookupTensor name (consumerERP (copyNamedState (producerERP arrays selected))) =
      lookupTensor name (numpyArrays arrays) := by
  rw [erp_only_clone_recovers_each_numpy_field]

def vectorArray {d : ℕ} (values : Fin d → ℝ) : Array64 :=
  ⟨[d], List.ofFn values, by simp⟩

def scalarArray (value : ℝ) : Array64 := ⟨[], [value], rfl⟩

def linearArrays {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) : ArrayFields :=
  [("coefficient", vectorArray coefficient), ("intercept", scalarArray intercept)]

theorem independent_linear_arrays_match_numeric_fields {d : ℕ}
    (coefficient : Fin d → ℝ) (intercept : ℝ) :
    numpyArrays (linearArrays coefficient intercept) = ERPState.linearState coefficient intercept := rfl

theorem transported_linear_fields {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ)
    (selected : Tensor) :
    consumerERP (copyNamedState (producerERP (linearArrays coefficient intercept) selected)) =
      ERPState.linearState coefficient intercept := by
  rw [erp_only_clone_recovers_each_numpy_field]
  rfl

theorem transported_fields_feed_literal_reader (p : ERPState.TemporalPrimitives)
    (ref : ERPState.ReferencePrimitives) (record : Fields) (arrays : ArrayFields) (selected : Tensor) :
    ERPState.fromState p ref record
      (consumerERP (copyNamedState (producerERP arrays selected))) =
      ERPState.fromState p ref record (numpyArrays arrays) := by
  rw [erp_only_clone_recovers_each_numpy_field]

/-- Nonempty family construction follows from its independent field guards,
not from an assumed constructor or score result. The selector is parsed by
the later personal caller, rather than this inner-ERP transport theorem. -/
theorem canonical_plain_transport_loads (p : ERPState.TemporalPrimitives)
    (ref : ERPState.ReferencePrimitives) (family : ERPState.Family)
    (width nfilter c t : ℕ) (baseline : Value)
    (coefficient : Fin (c * (t / width)) → ℝ) (intercept : ℝ) (selected : Tensor)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hf : family ≠ .temporal ∧ family ≠ .xdawn)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none") :
    ERPState.fromState p ref (ERPState.plainRecord family width nfilter c t baseline)
      (consumerERP (copyNamedState (producerERP (linearArrays coefficient intercept) selected))) =
      .ok ⟨⟨family, width, nfilter, baseline, none⟩, c, t,
        ERPState.linearState coefficient intercept, none, none⟩ := by
  rw [transported_linear_fields]
  exact ERPState.plain_family_constructs p ref family width nfilter c t baseline
    coefficient intercept hw hn hc ht hb hf hbaseline

theorem canonical_plain_transport_scores_from_fields (p : ERPState.TemporalPrimitives)
    (ref : ERPState.ReferencePrimitives) (family : ERPState.Family)
    (width nfilter c t : ℕ) (baseline : Value)
    (coefficient : Fin (c * (t / width)) → ℝ) (intercept : ℝ) (selected : Tensor)
    (epoch : ERPFeatures.Epoch c t)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hf : family ≠ .temporal ∧ family ≠ .xdawn)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none") :
    (ERPState.fromState p ref (ERPState.plainRecord family width nfilter c t baseline)
      (consumerERP (copyNamedState (producerERP (linearArrays coefficient intercept) selected)))).bind
      (fun model => ERPState.affineScore model.state (c * (t / width)) (ERPFeatures.binFeatures width epoch)) =
      .ok (ERPFeatures.evaluate ⟨coefficient, intercept⟩ (ERPFeatures.binFeatures width epoch)) := by
  rw [transported_linear_fields]
  exact ERPState.loaded_plain_uses_actual_bin_features p ref family width nfilter c t baseline
    coefficient intercept epoch hw hn hc ht hb hf hbaseline

theorem prefix_is_stripped_exactly_once (suffix : String) :
    ERPState.stripERP ("erp.erp." ++ suffix) = some ("erp." ++ suffix) := by
  change ERPState.stripERP ("erp." ++ ("erp." ++ suffix)) = _
  exact ERPState.prefixed_name_strips_once _

/-- The source's `if adapter is not None: state = ...` replacement. -/
def afterAdapter (erpState : NamedState) (adapterState : Option NamedState) : NamedState :=
  adapterState.getD erpState

theorem no_adapter_preserves_erp_map (arrays : ArrayFields) (selected : Tensor) :
    afterAdapter (producerERP arrays selected) none = producerERP arrays selected := rfl

theorem adapter_replaces_entire_erp_map (arrays : ArrayFields) (selected : Tensor)
    (adapter : NamedState) : afterAdapter (producerERP arrays selected) (some adapter) = adapter := rfl

def temporalTransfer (model : ERPState.Model) : Bool :=
  match model.header.temporal with
  | none => false
  | some basis => decide (basis.role = .source)

def producedKind (sourceInitialized adapterPresent temporalSource erpPresent : Bool) : String :=
  if sourceInitialized then "source_initialized_n2p3"
  else if adapterPresent then "scratch_n2p3"
  else if temporalSource then "source_temporal_target_erp"
  else if erpPresent then "target_only_structured_erp"
  else "target_only_shrinkage_lda"

def sourceFlag (sourceInitialized temporalSource : Bool) : Value :=
  .boolean (sourceInitialized || temporalSource)

theorem every_erp_family_is_not_source_initialization (family : ERPState.Family) :
    family.text ≠ "source" := by
  cases family <;> decide

theorem source_initialization_has_first_priority (adapter temporal erp : Bool) :
    producedKind true adapter temporal erp = "source_initialized_n2p3" := rfl

theorem adapter_priority_precedes_temporal_source (temporal erp : Bool) :
    producedKind false true temporal erp = "scratch_n2p3" := rfl

theorem actual_erp_only_kind (model : ERPState.Model) :
    producedKind false false (temporalTransfer model) true =
      if temporalTransfer model then "source_temporal_target_erp" else "target_only_structured_erp" := by
  simp [producedKind]

theorem actual_erp_only_source_flag (model : ERPState.Model) :
    sourceFlag false (temporalTransfer model) = .boolean (temporalTransfer model) := rfl

theorem source_flag_is_not_numeric_one : sourceFlag false true ≠ .integer 1 := by
  intro h
  cases h

theorem source_flag_is_not_numeric_zero : sourceFlag false false ≠ .integer 0 := by
  intro h
  cases h

def fixtureArray : Array64 := ⟨[2], [2, 3], rfl⟩
def fixtureSelected : Tensor := ⟨"torch.int64", [1], [.integer 0]⟩
def fixtureExport : ArrayFields := [("coefficient", fixtureArray)]

theorem nonempty_transport_fixture :
    consumerERP (copyNamedState (producerERP fixtureExport fixtureSelected)) =
      [("coefficient", ⟨"float64", [2], [.number 2, .number 3]⟩)] := by
  rw [erp_only_clone_recovers_each_numpy_field]
  rfl

theorem nonempty_torch_dtype_fixture :
    lookupTensor "erp.coefficient" (producerERP fixtureExport fixtureSelected) =
      some ⟨"torch.float64", [2], [.number 2, .number 3]⟩ := rfl

theorem nonempty_selector_survives_clone :
    lookupTensor "waveform_channels" (copyNamedState (producerERP fixtureExport fixtureSelected)) =
      some fixtureSelected := by
  rw [copyNamedState_preserves]
  rfl

theorem adapter_can_remove_an_erp_key :
    lookupTensor "erp.coefficient" (afterAdapter (producerERP fixtureExport fixtureSelected)
      (some [("input_mean", fixtureSelected)])) = none := rfl

theorem full_nonempty_transported_family_scores_five :
    (ERPState.fromState ERPState.fixturePrimitives ERPState.fixtureReference
      (ERPState.plainRecord .pooled 1 1 1 1 (.text "none"))
      (consumerERP (copyNamedState (producerERP (linearArrays ERPState.fixtureCoefficient (-1))
        fixtureSelected)))).bind
      (fun model => ERPState.affineScore model.state 1 (fun _ => 3)) = .ok 5 := by
  rw [transported_linear_fields]
  change (ERPState.fromState ERPState.fixturePrimitives ERPState.fixtureReference
    (ERPState.plainRecord .pooled 1 1 1 1 (.text "none")) ERPState.fixtureLinearState).bind
      (fun model => ERPState.affineScore model.state 1 (fun _ => 3)) = .ok 5
  rw [ERPState.pooled_and_toeplitz_nonempty_construct .pooled (Or.inl rfl)]
  change ERPState.affineScore (ERPState.linearState ERPState.fixtureCoefficient (-1))
    1 (fun _ => 3) = .ok 5
  rw [ERPState.affine_score_from_independent_fields]
  norm_num [ERPState.fixtureCoefficient, dotProduct]

end

end N2P3.PersonalERPTransport
