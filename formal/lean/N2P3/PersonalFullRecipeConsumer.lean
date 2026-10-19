import N2P3.PersonalTargetRecipeFields
import N2P3.PersonalERPCurrentConsumer

/-!
# Full constructor on the current ERP metadata-consumer branch

Source calibrated.py:619-642. This replaces no frozen accepted module. It
composes the actual selector and per-array/current reader with the new full
constructor, then the physical, strict source-role and temporal checks. Valid
source/scratch/waveform/source-feature recipes reach the physical mismatch,
instead of being spuriously rejected by an ERP-only constructor.

The inherited current reader uses finite decoded arrays. This begins after
the earlier metadata JSON and physical/QC ingress gates; it does not certify
those gates, nonfinite fit outputs or native Torch/NumPy implementations.
PersonalFullRecipeOutcome supplies the separate nonfinite output graph.
-/
namespace N2P3.PersonalFullRecipeConsumer
open N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPTransport
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Failure where
  | constructor (fault : Fault)
  | inherited (fault : LedgerFault)
  deriving DecidableEq

abbrev ConsumerResult (α : Type) := Except Failure α

def liftCheck (check : Check α) : ConsumerResult α := check.mapError Failure.inherited
def liftConstructor (result : Result α) : ConsumerResult α := result.mapError Failure.constructor

structure Bound where
  recipe : Recipe
  selected : List ℕ
  model : Model

def bindERP (recipe : Recipe) (kind : String) (flag : N2P3.FrozenState.Value)
    (physical : N2P3.PersonalERPRecipe.Physical) (selected : List ℕ) (model : Model) : Check Bound := do
  ensure (model.channels = selected.length ∧ model.times = physical.times ∧
    N2P3.PersonalAdmission.PyEq model.header.baseline physical.baseline ∧
    recipe.initialization.text = model.header.family.text ∧ recipe.width = model.header.binWidth)
    "ERP state recipe physical contract"
  ensure (flag = .boolean (N2P3.PersonalERPRecipe.usesSource model) ∧
    kind = N2P3.PersonalERPRecipe.modelKind (N2P3.PersonalERPRecipe.usesSource model)) "ERP strict source role"
  match model.header.temporal with
  | none => pure ()
  | some basis =>
    ensure (recipe.temporal = some (.text (if N2P3.PersonalERPRecipe.usesSource model then
      "source_supervised" else "fixed_cosine"), basis.rank)) "ERP temporal kind/rank"
  pure ⟨recipe, selected, model⟩

def consume (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (physical : N2P3.PersonalERPRecipe.Physical) (kind : String) (flag : N2P3.FrozenState.Value)
    (kwargs : Fields) (record : N2P3.FrozenState.Fields) (state : NamedState) : ConsumerResult Bound := do
  let restored := copyNamedState state
  liftCheck (ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind")
  let tensor ← liftCheck (tensorAt restored "waveform_channels")
  let selected ← liftCheck (N2P3.PersonalERPRecipe.readSelected physical.channels tensor)
  let model ← liftCheck (N2P3.CurrentERPState.fromState p ref (N2P3.FrozenState.copyFields record) (consumerERP restored))
  let recipe ← liftConstructor (fromKwargs s kwargs)
  liftCheck (bindERP recipe kind flag physical selected model)

theorem selector_dtype_error_precedes_full_constructor (s : Primitives) (p : TemporalPrimitives)
    (ref : ReferencePrimitives) (physical : N2P3.PersonalERPRecipe.Physical)
    (flag : N2P3.FrozenState.Value) (kwargs : Fields) (record : N2P3.FrozenState.Fields)
    (arrays : ArrayFields) (selected : Tensor) (h : selected.dtype ≠ "torch.int64") :
    consume s p ref physical "target_only_structured_erp" flag kwargs record (producerERP arrays selected) =
      .error (.inherited (.sourceReject "ERP selected dtype")) := by
  simp [consume, liftCheck, ensure,
    N2P3.PersonalERPCurrentConsumer.cloned_producer_selector_reads_before_NumPy,
    N2P3.PersonalERPRecipe.readSelected, h]

theorem plain_reader_then_full_constructor_formula
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : N2P3.PersonalERPRecipe.ERPFields) (physical : N2P3.PersonalERPRecipe.Physical)
    (indices : List ℕ) (kwargs : Fields) (flag : N2P3.FrozenState.Value)
    (coefficient : Fin (indices.length * (physical.times / f.width)) → ℝ) (intercept : ℝ)
    (hw : 0 < f.width) (hn : indices ≠ []) (hu : indices.Nodup) (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : f.family ≠ .temporal ∧ f.family ≠ .xdawn)
    (hb : f.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / f.width) :
    consume s p ref physical "target_only_structured_erp" flag kwargs
      (exportRecord (N2P3.PersonalERPRecipe.plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))
      (producerERP (linearArrays coefficient intercept) (N2P3.PersonalERPRecipe.selectedTensor indices)) =
      (liftConstructor (fromKwargs s kwargs)).bind (fun recipe => liftCheck (bindERP recipe
        "target_only_structured_erp" flag physical indices
        (N2P3.PersonalERPRecipe.plainModel f indices.length physical.times 4 physical.baseline coefficient intercept))) := by
  have hs := N2P3.PersonalERPRecipe.selected_indices_preserve_order physical.channels indices hn hu hr
  have hi := N2P3.PersonalERPCurrentConsumer.independent_plain_fields_feed_current_reader p ref f physical indices
    coefficient intercept hw hn ht hbins hf hb
  simp [consume, N2P3.FrozenState.copyFields_preserves, liftCheck, ensure,
    N2P3.PersonalERPCurrentConsumer.cloned_producer_selector_reads_before_NumPy, hs, hi]

theorem bind_plain_from_independent_fields (recipe : Recipe) (physical : N2P3.PersonalERPRecipe.Physical)
    (indices : List ℕ) (model : Model) (ht : model.header.temporal = none)
    (hc : model.channels = indices.length) (hn : model.times = physical.times)
    (hb : N2P3.PersonalAdmission.PyEq model.header.baseline physical.baseline)
    (hf : recipe.initialization.text = model.header.family.text) (hw : recipe.width = model.header.binWidth) :
    bindERP recipe "target_only_structured_erp" (.boolean false) physical indices model = .ok ⟨recipe, indices, model⟩ := by
  simp [bindERP, N2P3.PersonalERPRecipe.usesSource, N2P3.PersonalERPRecipe.modelKind,
    ht, hc, hn, hb, hf, hw, ensure]

theorem full_constructor_after_independent_plain_reader
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (f : Arguments) (h : Conditions f) (a : AtomicSuccess s f) (adapter : Option FeatureConfig)
    (ha : AdapterFields s f.raw adapter) (value : Value) (hj : f.raw.adapter = .raw value)
    (modelFields : N2P3.PersonalERPRecipe.ERPFields) (physical : N2P3.PersonalERPRecipe.Physical)
    (indices : List ℕ) (coefficient : Fin (indices.length * (physical.times / modelFields.width)) → ℝ)
    (intercept : ℝ) (hw : 0 < modelFields.width) (hn : indices ≠ []) (hu : indices.Nodup)
    (hr : ∀ i ∈ indices, i < physical.channels)
    (hf : modelFields.family ≠ .temporal ∧ modelFields.family ≠ .xdawn)
    (hb : modelFields.family ≠ .toeplitz ∨ physical.baseline = .text "none")
    (ht : 0 < physical.times) (hbins : 0 < physical.times / modelFields.width)
    (hfamily : f.initialization.text = modelFields.family.text) (hwidth : f.width = modelFields.width) :
    consume s p ref physical "target_only_structured_erp" (.boolean false) (allRecordFields f.raw)
      (exportRecord (N2P3.PersonalERPRecipe.plainModel modelFields indices.length physical.times 4 physical.baseline coefficient intercept))
      (producerERP (linearArrays coefficient intercept) (N2P3.PersonalERPRecipe.selectedTensor indices)) =
      .ok ⟨f.recipe adapter, indices,
        N2P3.PersonalERPRecipe.plainModel modelFields indices.length physical.times 4 physical.baseline coefficient intercept⟩ := by
  rw [plain_reader_then_full_constructor_formula s p ref modelFields physical indices _ _ coefficient intercept
    hw hn hu hr hf hb ht hbins, complete_json_kwargs_constructor_from_fields s f adapter value h a ha hj]
  have hbind := bind_plain_from_independent_fields (f.recipe adapter) physical indices
    (N2P3.PersonalERPRecipe.plainModel modelFields indices.length physical.times 4 physical.baseline coefficient intercept)
    rfl rfl rfl rfl hfamily hwidth
  simp [liftConstructor, liftCheck, hbind]

theorem valid_nonERP_constructor_is_rejected_at_physical_binding (recipe : Recipe)
    (physical : N2P3.PersonalERPRecipe.Physical) (indices : List ℕ) (model : Model)
    (kind : String) (flag : N2P3.FrozenState.Value)
    (h : recipe.initialization = .source ∨ recipe.initialization = .scratch ∨
      recipe.initialization = .waveform ∨ recipe.initialization = .sourceFeature) :
    bindERP recipe kind flag physical indices model = .error (.sourceReject "ERP state recipe physical contract") := by
  rcases h with h | h | h | h <;> cases hf : model.header.family <;>
    simp [bindERP, h, hf, Initialization.text, Family.text, ensure]

end
end N2P3.PersonalFullRecipeConsumer
