import N2P3.PersonalRecipeRecordProducer

/-! Actual saved recipe producer feeding the nonfinite-aware consumer.
This is the recipe component of metadata JSON; other metadata ingress fields
remain independent. The original constructor, per-field record/JSON, inner
array reader, reconstructed constructor, binding and outer finite gate retain
their separate source positions. -/
namespace N2P3.PersonalRecipeRecordConsumer
open N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalRecipePython N2P3.PersonalTargetRecipe N2P3.PersonalTargetRecipeFields
open N2P3.PersonalRecipeRecord N2P3.PersonalRecipeRecordJSON N2P3.PersonalRecipeRecordProducer
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

abbrev ConsumerResult (α : Type) := Except (Sum PersonalRecipeRecord.Fault PersonalFullRecipeOutcome.Failure) α

def consumeRecord (copy : CopyOperations) (p : Primitives) (r : Raw)
    (temporal : TemporalPrimitives) (reference : ReferencePrimitives)
    (cast : String → PersonalERPFitOutcome.Array64 → Check PersonalERPFitOutcome.Array64)
    (physical : PersonalERPRecipe.Physical) (kind : String) (flag : FrozenState.Value)
    (artifact : FrozenState.Fields) (state : PersonalERPFitOutcome.State) : ConsumerResult PersonalFullRecipeConsumer.Bound := do
  let kwargs ← (recordJSON copy p r).mapError Sum.inl
  (PersonalFullRecipeOutcome.consume p temporal reference cast physical kind flag kwargs artifact state).mapError Sum.inr

theorem actual_saved_field_image_feeds_the_nonfinite_consumer
    (copy : CopyOperations) (p : Primitives) (f : Arguments) (temperature prior width : Bool)
    (copies : CopyFieldsFacts copy (allRecordFields f.raw))
    (equalities : EqualityAtoms p f.raw temperature prior width)
    (json : JSONFieldsFacts p (allRecordFields f.raw))
    (temporal : TemporalPrimitives) (reference : ReferencePrimitives)
    (cast : String → PersonalERPFitOutcome.Array64 → Check PersonalERPFitOutcome.Array64)
    (physical : PersonalERPRecipe.Physical) (kind : String) (flag : FrozenState.Value)
    (artifact : FrozenState.Fields) (state : PersonalERPFitOutcome.State) :
    consumeRecord copy p f.raw temporal reference cast physical kind flag artifact state =
      (PersonalFullRecipeOutcome.consume p temporal reference cast physical kind flag
        (image (normalRaw f.raw) temperature prior width) artifact state).mapError Sum.inr := by
  rw [consumeRecord, saved_record_JSON_image_from_individual_operations copy p f.raw temperature prior width copies equalities json]
  simp

theorem actual_saved_record_after_inner_reader_reconstructs_the_derived_fields
    (copy : CopyOperations) (p : Primitives) (f : Arguments) (temperature prior width : Bool)
    (copies : CopyFieldsFacts copy (allRecordFields f.raw))
    (equalities : EqualityAtoms p f.raw temperature prior width)
    (json : JSONFieldsFacts p (allRecordFields f.raw)) :
    consumeRecord copy p f.raw fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      ((PersonalFullRecipeOutcome.liftConstructor (validate p (loaded f temperature prior width).raw)).bind (fun recipe =>
        (PersonalFullRecipeOutcome.liftCheck (PersonalFullRecipeConsumer.bindERP recipe "target_only_structured_erp" (.boolean false)
          PersonalERPCurrentConsumer.fixturePhysical [0] PersonalERPCurrentConsumer.fixtureModel)).bind (fun bound =>
            (PersonalFullRecipeOutcome.liftCheck (PersonalERPFitOutcome.finalTorchFinite PersonalERPFitOutcome.nonERPExtraState)).map
              (fun _ => bound)))).mapError Sum.inr := by
  rw [actual_saved_field_image_feeds_the_nonfinite_consumer copy p f temperature prior width copies equalities json,
    PersonalFullRecipeOutcome.finite_inner_then_full_constructor_binding_then_outer_formula,
    fromKwargs, omitted_defaults_reconstruct_the_explicit_normalized_raw,
    ← loaded_fields_are_derived_from_source_fields_and_defaults]
  rfl

theorem actual_saved_nonERP_record_reaches_physical_binding_before_outer
    (copy : CopyOperations) (p : Primitives) (f : Arguments) (temperature prior width : Bool)
    (copies : CopyFieldsFacts copy (allRecordFields f.raw))
    (equalities : EqualityAtoms p f.raw temperature prior width)
    (json : JSONFieldsFacts p (allRecordFields f.raw)) (adapter : Option FeatureConfig)
    (fields : Conditions (loaded f temperature prior width))
    (atoms : AtomicSuccess p (loaded f temperature prior width))
    (adapterFields : AdapterFields p (loaded f temperature prior width).raw adapter)
    (hi : f.initialization = .source ∨ f.initialization = .scratch ∨
      f.initialization = .waveform ∨ f.initialization = .sourceFeature) :
    consumeRecord copy p f.raw fixturePrimitives fixtureReference PersonalERPFitOutcome.identityCopy
      PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
      PersonalERPCurrentConsumer.fixtureRecord PersonalERPFitOutcome.nonERPExtraState =
      .error (.inr (.core (.inherited (.sourceReject "ERP state recipe physical contract")))) := by
  rw [actual_saved_record_after_inner_reader_reconstructs_the_derived_fields copy p f temperature prior width copies equalities json,
    complete_constructor_from_independent_fields p (loaded f temperature prior width) adapter fields atoms adapterFields]
  have hb := PersonalFullRecipeConsumer.valid_nonERP_constructor_is_rejected_at_physical_binding
    ((loaded f temperature prior width).recipe adapter) PersonalERPCurrentConsumer.fixturePhysical [0]
    PersonalERPCurrentConsumer.fixtureModel "target_only_structured_erp" (.boolean false) hi
  simp [PersonalFullRecipeOutcome.liftConstructor, PersonalFullRecipeOutcome.liftCheck, hb]

def constructSaveThenConsume (copy : CopyOperations) (p : Primitives) (r : Raw)
    (temporal : TemporalPrimitives) (reference : ReferencePrimitives)
    (cast : String → PersonalERPFitOutcome.Array64 → Check PersonalERPFitOutcome.Array64)
    (physical : PersonalERPRecipe.Physical) (kind : String) (flag : FrozenState.Value)
    (artifact : FrozenState.Fields) (state : PersonalERPFitOutcome.State) : ConsumerResult PersonalFullRecipeConsumer.Bound := do
  let original ← (validate p r).mapError (fun fault => Sum.inl (PersonalRecipeRecord.Fault.operation fault))
  consumeRecord copy p original.raw temporal reference cast physical kind flag artifact state

theorem actual_constructor_normalized_adapter_produces_the_saved_consumer_input
    (copy : CopyOperations) (p : Primitives) (f : Arguments) (adapter : Option FeatureConfig)
    (conditions : Conditions f) (atoms : AtomicSuccess p f) (adapterFields : AdapterFields p f.raw adapter)
    (temporal : TemporalPrimitives) (reference : ReferencePrimitives)
    (cast : String → PersonalERPFitOutcome.Array64 → Check PersonalERPFitOutcome.Array64)
    (physical : PersonalERPRecipe.Physical) (kind : String) (flag : FrozenState.Value)
    (artifact : FrozenState.Fields) (state : PersonalERPFitOutcome.State) :
    constructSaveThenConsume copy p f.raw temporal reference cast physical kind flag artifact state =
      consumeRecord copy p (constructed f adapter).raw temporal reference cast physical kind flag artifact state := by
  rw [constructSaveThenConsume, complete_constructor_from_independent_fields p f adapter conditions atoms adapterFields]
  simp only [Except.mapError, Bind.bind, Except.bind, constructed_fields_are_actual_constructor_output]

end
end N2P3.PersonalRecipeRecordConsumer
