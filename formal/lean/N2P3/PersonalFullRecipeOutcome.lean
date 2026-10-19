import N2P3.PersonalFullRecipeConsumer
import N2P3.PersonalRecipeControls
import N2P3.PersonalERPFitOutcome

/-!
# Nonfinite fit-output graph with the full TargetRecipe constructor

calibrated.py:550-551,619-642,660-661. Arbitrary observed +Inf, -Inf and NaN
producer cells survive the inherited explicit clone and transport. Its inner
reader checks/casts all arrays before finiteness. The full fieldwise recipe
constructor and physical/role binding follow inner success; the outer Torch
finite gate remains last, including unprefixed diagnostic state.

The frozen FitOutcome and finite CurrentConsumer are unchanged. This graph
does not assume complete recipe success or array finiteness. It begins after
the earlier metadata JSON/physical ingress gates and retains their independent
obligations, native numerical refinements and resource exceptions.
-/
namespace N2P3.PersonalFullRecipeOutcome
open N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Failure where
  | core (fault : PersonalFullRecipeConsumer.Failure)
  | transport (fault : PersonalERPScalar.Fault)
  deriving DecidableEq

abbrev OutcomeResult (α : Type) := Except Failure α

def liftCheck (check : Check α) : OutcomeResult α :=
  check.mapError (fun fault => .core (.inherited fault))

def liftConstructor (result : Result α) : OutcomeResult α :=
  result.mapError (fun fault => .core (.constructor fault))

def liftTransport (result : PersonalERPScalar.Result α) : OutcomeResult α :=
  result.mapError (fun fault => match fault with
    | .inherited ledger => .core (.inherited ledger)
    | fault => .transport fault)

def consume (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (castCopy : String → PersonalERPFitOutcome.Array64 → Check PersonalERPFitOutcome.Array64)
    (physical : PersonalERPRecipe.Physical) (kind : String) (flag : FrozenState.Value)
    (kwargs : Fields) (record : FrozenState.Fields) (state : PersonalERPFitOutcome.State) :
    OutcomeResult PersonalFullRecipeConsumer.Bound := do
  let restored := PersonalERPFitOutcome.cloneState state
  liftCheck (ensure (kind ∈ ["target_only_structured_erp", "source_temporal_target_erp"]) "personal ERP model kind")
  let tensor ← liftTransport (PersonalERPFitOutcome.selectedTensor restored)
  let indices ← liftCheck (PersonalERPRecipe.readSelected physical.channels tensor)
  let arrays ← liftTransport (PersonalERPFitOutcome.consumerArrays restored)
  let model ← liftCheck (PersonalERPFitOutcome.readOutcome p ref castCopy (FrozenState.copyFields record) arrays)
  let recipe ← liftConstructor (fromKwargs s kwargs)
  let bound ← liftCheck (PersonalFullRecipeConsumer.bindERP recipe kind flag physical indices model)
  liftCheck (PersonalERPFitOutcome.finalTorchFinite restored)
  pure bound

theorem producer_full_constructor_continuation
    (s : Primitives) (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (castCopy : String → PersonalERPFitOutcome.Array64 → Check PersonalERPFitOutcome.Array64)
    (physical : PersonalERPRecipe.Physical) (flag : FrozenState.Value) (kwargs : Fields)
    (record : FrozenState.Fields) (arrays : PersonalERPFitOutcome.Arrays) (tensor : Tensor) :
    consume s p ref castCopy physical "target_only_structured_erp" flag kwargs record
      (PersonalERPFitOutcome.producer arrays tensor) =
      (liftCheck (PersonalERPRecipe.readSelected physical.channels tensor)).bind (fun indices =>
        (liftCheck (PersonalERPFitOutcome.readOutcome p ref castCopy record arrays)).bind (fun model =>
          (liftConstructor (fromKwargs s kwargs)).bind (fun recipe =>
            (liftCheck (PersonalFullRecipeConsumer.bindERP recipe "target_only_structured_erp" flag physical indices model)).bind
              (fun bound => (liftCheck (PersonalERPFitOutcome.finalTorchFinite
                (PersonalERPFitOutcome.producer arrays tensor))).map (fun _ => bound))))) := by
  simp [consume, PersonalERPFitOutcome.clone_preserves_every_observed_cell, ensure,
    PersonalERPFitOutcome.selectedTensor, PersonalERPFitOutcome.producer_retains_selector_for_any_output_cells,
    PersonalERPFitOutcome.producer_consumer_retains_nonfinite_cells, FrozenState.copyFields_preserves,
    liftCheck, liftTransport]

theorem nonfinite_inner_output_precedes_all_full_constructor_fields (s : Primitives) (kwargs : Fields) :
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (PersonalERPFitOutcome.artifactRecord PersonalERPFitOutcome.boundaryArtifact)
      (PersonalERPFitOutcome.producer PersonalERPFitOutcome.boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.core (.inherited (.sourceReject "ERP state contains nonfinite values"))) := by
  rw [producer_full_constructor_continuation]
  have hs := PersonalERPRecipe.selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [liftCheck, hs, PersonalERPFitOutcome.boundary_current_reader_rejects_nonfinite_after_keys]

theorem selector_refusal_precedes_nonfinite_inner_and_full_constructor (s : Primitives) (kwargs : Fields) :
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (PersonalERPFitOutcome.artifactRecord PersonalERPFitOutcome.boundaryArtifact)
      (PersonalERPFitOutcome.producer PersonalERPFitOutcome.boundaryPayload.arrays PersonalERPCurrentConsumer.floatingSelector) =
      .error (.core (.inherited (.sourceReject "ERP selected dtype"))) := by
  rw [producer_full_constructor_continuation]
  simp [liftCheck, PersonalERPRecipe.readSelected, PersonalERPCurrentConsumer.floatingSelector, ensure]

theorem later_array_cast_error_precedes_inner_finite_and_full_constructor (s : Primitives) (kwargs : Fields) :
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.failLaterCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (PersonalERPFitOutcome.artifactRecord PersonalERPFitOutcome.boundaryArtifact)
      (PersonalERPFitOutcome.producer PersonalERPFitOutcome.boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.core (.inherited (.fieldFailure "intercept float64 copy exception"))) := by
  rw [producer_full_constructor_continuation]
  have hs := PersonalERPRecipe.selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [liftCheck, hs, PersonalERPFitOutcome.later_array_cast_exception_precedes_earlier_nonfinite]

theorem wrong_inner_keys_precede_cast_and_full_constructor (s : Primitives) (kwargs : Fields) :
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.failLaterCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (PersonalERPFitOutcome.artifactRecord PersonalERPFitOutcome.boundaryArtifact)
      (PersonalERPFitOutcome.producer PersonalERPFitOutcome.badKeyArrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.core (.inherited (.sourceReject "ERP state keys"))) := by
  rw [producer_full_constructor_continuation]
  have hs := PersonalERPRecipe.selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [liftCheck, hs, PersonalERPFitOutcome.inner_keys_precede_array_conversion_and_finite_guard]

theorem finite_inner_then_full_constructor_binding_then_outer_formula (s : Primitives) (kwargs : Fields) :
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy PersonalERPCurrentConsumer.fixturePhysical
      "target_only_structured_erp" (.boolean false) kwargs PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      (liftConstructor (fromKwargs s kwargs)).bind (fun recipe =>
        (liftCheck (PersonalFullRecipeConsumer.bindERP recipe "target_only_structured_erp" (.boolean false)
          PersonalERPCurrentConsumer.fixturePhysical [0] PersonalERPCurrentConsumer.fixtureModel)).bind (fun bound =>
            (liftCheck (PersonalERPFitOutcome.finalTorchFinite PersonalERPFitOutcome.nonERPExtraState)).map (fun _ => bound))) := by
  have hs : PersonalERPFitOutcome.selectedTensor PersonalERPFitOutcome.nonERPExtraState =
      .ok (PersonalERPRecipe.selectedTensor [0]) := by
    simp [PersonalERPFitOutcome.selectedTensor, PersonalERPFitOutcome.nonERPExtraState,
      PersonalERPFitOutcome.producer, PersonalERPFitOutcome.finiteFixtureArrays, PersonalERPFitOutcome.fromFiniteArrays,
      PersonalERPCurrentConsumer.fixtureArrays, PersonalERPTransport.linearArrays, PersonalERPFitOutcome.lookupState]
  have hr := PersonalERPRecipe.selected_indices_preserve_order 1 [0] (by decide) (by simp) (by simp)
  simp [consume, PersonalERPFitOutcome.clone_preserves_every_observed_cell, FrozenState.copyFields_preserves,
    ensure, hs, PersonalERPCurrentConsumer.fixturePhysical, hr, liftCheck, liftTransport,
    PersonalERPFitOutcome.nonERP_extra_is_excluded_from_inner_arrays, PersonalERPFitOutcome.finite_fixture_outcome_reader]

theorem raw_huge_temperature_precedes_outer_nonfinite_gate :
    consume PersonalRecipeObserved.primitives fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      (allRecordFields { PersonalRecipeControls.base.raw with temperature := .integer PersonalRecipeObserved.huge100 })
      PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.core (.constructor ⟨"candidate_loss_temperature", .numpyFinite, .typeError⟩)) := by
  rw [finite_inner_then_full_constructor_binding_then_outer_formula, fromKwargs,
    json_kwargs_reconstruct_independent_raw_fields _ .null rfl]
  have ht := numpy_temperature_exception_after_no_adapter PersonalRecipeObserved.primitives
    { PersonalRecipeControls.base.raw with temperature := .integer PersonalRecipeObserved.huge100 } .typeError
    rfl (by simp [PersonalRecipeControls.base, Arguments.raw]) rfl ⟨rfl, rfl⟩ PersonalRecipeObserved.huge_numpy_typeerror
  simp [ht, liftConstructor]

theorem derived_nonERP_recipe_reaches_binding_before_outer (s : Primitives) (f : Arguments)
    (adapter : Option FeatureConfig) (value : Value) (h : Conditions f) (a : AtomicSuccess s f)
    (ha : AdapterFields s f.raw adapter) (hj : f.raw.adapter = .raw value)
    (hi : f.initialization = .source ∨ f.initialization = .scratch ∨
      f.initialization = .waveform ∨ f.initialization = .sourceFeature) :
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy PersonalERPCurrentConsumer.fixturePhysical
      "target_only_structured_erp" (.boolean false) (allRecordFields f.raw)
      PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.core (.inherited (.sourceReject "ERP state recipe physical contract"))) := by
  rw [finite_inner_then_full_constructor_binding_then_outer_formula,
    complete_json_kwargs_constructor_from_fields s f adapter value h a ha hj]
  have hb := PersonalFullRecipeConsumer.valid_nonERP_constructor_is_rejected_at_physical_binding
    (f.recipe adapter) PersonalERPCurrentConsumer.fixturePhysical [0] PersonalERPCurrentConsumer.fixtureModel
    "target_only_structured_erp" (.boolean false) hi
  simp [liftConstructor, liftCheck, hb]

theorem legal_pooled_fields_reach_outer_nonfinite_gate :
    consume PersonalRecipeObserved.primitives fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      (allRecordFields (PersonalRecipeControls.fixture .pooled).raw)
      PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.core (.inherited (.sourceReject "calibrated predictor contains nonfinite state"))) := by
  have hc := complete_json_kwargs_constructor_from_fields PersonalRecipeObserved.primitives
    (PersonalRecipeControls.fixture .pooled) none .null
    (PersonalRecipeControls.all_eight_fixture_structural_fields .pooled)
    (PersonalRecipeControls.all_eight_fixture_single_operations .pooled)
    (.absent rfl (by simp [PersonalRecipeControls.fixture, PersonalRecipeControls.lda, Arguments.raw,
      Initialization.isLDA])) rfl
  have hb := PersonalFullRecipeConsumer.bind_plain_from_independent_fields
    ((PersonalRecipeControls.fixture .pooled).recipe none) PersonalERPCurrentConsumer.fixturePhysical [0]
    PersonalERPCurrentConsumer.fixtureModel rfl rfl rfl rfl rfl rfl
  rw [finite_inner_then_full_constructor_binding_then_outer_formula, hc]
  simp [liftConstructor, liftCheck, hb, PersonalERPFitOutcome.nonERP_extra_fails_only_final_state_finite_gate]

theorem source_constructs_then_physical_mismatch_precedes_outer :
    consume PersonalRecipeObserved.primitives fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      (allRecordFields PersonalRecipeControls.base.raw)
      PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.core (.inherited (.sourceReject "ERP state recipe physical contract"))) := by
  exact derived_nonERP_recipe_reaches_binding_before_outer PersonalRecipeObserved.primitives PersonalRecipeControls.base
    none .null (PersonalRecipeControls.all_eight_fixture_structural_fields .source)
    (PersonalRecipeControls.all_eight_fixture_single_operations .source)
    (.absent rfl (by simp [PersonalRecipeControls.base, Arguments.raw])) rfl (Or.inl rfl)

theorem fitted_nonfinite_outputs_remain_rejected_before_any_full_recipe (s : Primitives) (kwargs : Fields) :
    PersonalERPFitOutcome.fitArtifact fixturePrimitives PersonalERPFitOutcome.boundaryNumerics
      (.text "pooled_lda") (.integer 1) (.text "none") none PersonalERPFitOutcome.boundaryValues PersonalERPFitOutcome.boundaryLabels =
        .ok PersonalERPFitOutcome.boundaryArtifact ∧
    consume s fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy ⟨1, 1, .text "none"⟩
      "target_only_structured_erp" (.boolean false) kwargs (PersonalERPFitOutcome.artifactRecord PersonalERPFitOutcome.boundaryArtifact)
      (PersonalERPFitOutcome.producer PersonalERPFitOutcome.boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
        .error (.core (.inherited (.sourceReject "ERP state contains nonfinite values"))) :=
  ⟨PersonalERPFitOutcome.constructor_fit_returns_nonfinite_artifact,
    nonfinite_inner_output_precedes_all_full_constructor_fields s kwargs⟩

end
end N2P3.PersonalFullRecipeOutcome
