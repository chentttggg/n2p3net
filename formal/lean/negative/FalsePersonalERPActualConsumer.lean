import N2P3.PersonalERPActualConsumer

/-! Fixed alternatives: huge temperature accepted via the old real decoder;
list membership mislabeled ValueError; dict option accepted; scalar failure
before inner state keys; scalar failure before Torch selector dtype. -/
namespace N2P3.NegativePersonalERPActualConsumer
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPTransport N2P3.PersonalERPScalar
open N2P3.PersonalERPActualConsumer

example : fixtureActual observedPrimitives (Fields.ofList (rawFields hugeActualRaw)) =
    .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by
  rw [actual_huge_temperature_typeerror_is_preserved]
  have hn : (Except.error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) : Result Bound) ≠
      .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : fixtureActual observedPrimitives (Fields.ofList (rawFields sequenceActualRaw)) =
    .error (.sourceValueError "recipe initialization") := by
  rw [actual_sequence_option_raises_after_scalar_prefix]
  have hn : (Except.error (.atomic "recipe initialization" .setMembership .typeError) : Result Bound) ≠
      .error (.sourceValueError "recipe initialization") := by intro h; cases h
  simp only [hn]

example : fixtureActual observedPrimitives (Fields.ofList (rawFields objectActualRaw)) =
    .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by
  rw [actual_object_option_raises_after_scalar_prefix]
  have hn : (Except.error (.atomic "recipe initialization" .setMembership .typeError) : Result Bound) ≠
      .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : consumeActualScalar observedPrimitives fixturePrimitives fixtureReference
    PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (Fields.ofList (rawFields hugeActualRaw)) PersonalERPCurrentConsumer.fixtureRecord
    (producerERP PersonalERPCurrentConsumer.doublePrefixedFixtureArrays PersonalERPCurrentConsumer.fixtureTensor) =
    .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  rw [actual_double_prefix_refusal_precedes_huge_scalar]
  have hn : (Except.error (.inherited (.sourceReject "ERP state keys")) : Result Bound) ≠
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by intro h; cases h
  simp only [hn]

example : consumeActualScalar observedPrimitives fixturePrimitives fixtureReference
    PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (Fields.ofList (rawFields hugeActualRaw)) PersonalERPCurrentConsumer.fixtureRecord
    (producerERP PersonalERPCurrentConsumer.fixtureArrays PersonalERPCurrentConsumer.floatingSelector) =
    .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  rw [scalar_selector_failure_precedes_kwargs _ _ _ _ _ _ _ _ _ (by decide)]
  have hn : (Except.error (.inherited (.sourceReject "ERP selected dtype")) : Result Bound) ≠
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by intro h; cases h
  simp only [hn]

end N2P3.NegativePersonalERPActualConsumer
