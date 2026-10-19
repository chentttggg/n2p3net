import N2P3.PersonalERPCurrentConsumer

/-! Two independently fixed false alternatives: accept a Torch float64
selector despite valid ERP arrays; recursively remove two ERP prefixes so
malformed array keys would be accepted. Only unsolved False is expected. -/
namespace N2P3.NegativePersonalERPCurrentConsumer
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPTransport N2P3.PersonalERPCurrentConsumer

example : consumeNumericBranch fixturePrimitives fixtureReference PersonalERPCurrentConsumer.fixturePhysical
    "target_only_structured_erp" (.boolean false) fixtureKwargs fixtureRecord
    (producerERP fixtureArrays floatingSelector) =
      .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by
  rw [floating_selector_does_not_load_supported_ERP_fields]
  have hn : (Except.error (.sourceReject "ERP selected dtype") : Check Bound) ≠
      .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : CurrentERPState.fromState fixturePrimitives fixtureReference fixtureRecord
    (consumerERP (producerERP doublePrefixedFixtureArrays fixtureTensor)) = .ok PersonalERPCurrentConsumer.fixtureModel := by
  rw [double_prefix_reaches_current_inner_key_refusal]
  have hn : (Except.error (.sourceReject "ERP state keys") : Check Model) ≠ .ok PersonalERPCurrentConsumer.fixtureModel := by
    intro h; cases h
  simp only [hn]

end N2P3.NegativePersonalERPCurrentConsumer
