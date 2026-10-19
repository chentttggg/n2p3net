import N2P3.PersonalERPFitOutcome

/-! Independently fixed false alternatives from actual source ordering:
post-fit finite refusal; inner acceptance; global-final before inner finite;
scalar before inner finite; global-final before scalar on non-ERP extras;
extra acceptance; finite before a later array cast exception; finite before
key rejection; finite before selector rejection. -/
namespace N2P3.NegativePersonalERPFitOutcome
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPScalar N2P3.PersonalERPFitOutcome

example : fitArtifact fixturePrimitives boundaryNumerics (.text "pooled_lda") (.integer 1) (.text "none") none
    boundaryValues boundaryLabels = .error (.sourceReject "invented post-fit finite guard") := by
  rw [constructor_fit_returns_nonfinite_artifact]
  have hn : (Except.ok boundaryArtifact : Check Artifact) ≠
      .error (.sourceReject "invented post-fit finite guard") := by intro h; cases h
  simp only [hn]

example : readOutcome fixturePrimitives fixtureReference identityCopy (artifactRecord boundaryArtifact)
    boundaryPayload.arrays = .ok (metadataModel boundaryArtifact) := by
  rw [boundary_current_reader_rejects_nonfinite_after_keys]
  have hn : (Except.error (.sourceReject "ERP state contains nonfinite values") : Check Model) ≠
      .ok (metadataModel boundaryArtifact) := by intro h; cases h
  simp only [hn]

example : consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy ⟨1, 1, .text "none"⟩
    "target_only_structured_erp" (.boolean false) (recordFields (pooledFields 1).raw) (artifactRecord boundaryArtifact)
    (producer boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.inherited (.sourceReject "calibrated predictor contains nonfinite state")) := by
  rw [nonfinite_ERP_output_rejected_before_any_scalar_kwargs]
  have hn : (Except.error (.inherited (.sourceReject "ERP state contains nonfinite values")) : Result Bound) ≠
      .error (.inherited (.sourceReject "calibrated predictor contains nonfinite state")) := by
    simp only [ne_eq, Except.error.injEq, Fault.inherited.injEq, LedgerFault.sourceReject.injEq]
    decide
  simp only [hn]

example : consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy ⟨1, 1, .text "none"⟩
    "target_only_structured_erp" (.boolean false)
    (Fields.ofList (rawFields PersonalERPActualConsumer.hugeActualRaw)) (artifactRecord boundaryArtifact)
    (producer boundaryPayload.arrays (PersonalERPRecipe.selectedTensor [0])) =
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  rw [nonfinite_ERP_output_rejected_before_any_scalar_kwargs]
  have hn : (Except.error (.inherited (.sourceReject "ERP state contains nonfinite values")) : Result Bound) ≠
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by intro h; cases h
  simp only [hn]

example : consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy
    PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (Fields.ofList (rawFields PersonalERPActualConsumer.hugeActualRaw)) PersonalERPCurrentConsumer.fixtureRecord
    nonERPExtraState = .error (.inherited (.sourceReject "calibrated predictor contains nonfinite state")) := by
  rw [huge_scalar_exception_precedes_nonERP_extra_final_gate]
  have hn : (Except.error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) : Result Bound) ≠
      .error (.inherited (.sourceReject "calibrated predictor contains nonfinite state")) := by intro h; cases h
  simp only [hn]

example : consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy
    PersonalERPCurrentConsumer.fixturePhysical "target_only_structured_erp" (.boolean false)
    (recordFields (pooledFields 1).raw) PersonalERPCurrentConsumer.fixtureRecord nonERPExtraState =
      .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by
  rw [legal_scalar_reaches_nonERP_extra_final_gate]
  have hn : (Except.error (.inherited (.sourceReject "calibrated predictor contains nonfinite state")) : Result Bound) ≠
      .ok ⟨(pooledFields 1).recipe, [0], PersonalERPCurrentConsumer.fixtureModel⟩ := by intro h; cases h
  simp only [hn]

example : readOutcome fixturePrimitives fixtureReference failLaterCopy (artifactRecord boundaryArtifact)
    boundaryPayload.arrays = .error (.sourceReject "ERP state contains nonfinite values") := by
  rw [later_array_cast_exception_precedes_earlier_nonfinite]
  have hn : (Except.error (.fieldFailure "intercept float64 copy exception") : Check Model) ≠
      .error (.sourceReject "ERP state contains nonfinite values") := by intro h; cases h
  simp only [hn]

example : readOutcome fixturePrimitives fixtureReference failLaterCopy (artifactRecord boundaryArtifact)
    badKeyArrays = .error (.sourceReject "ERP state contains nonfinite values") := by
  rw [inner_keys_precede_array_conversion_and_finite_guard]
  have hn : (Except.error (.sourceReject "ERP state keys") : Check Model) ≠
      .error (.sourceReject "ERP state contains nonfinite values") := by
    simp only [ne_eq, Except.error.injEq, LedgerFault.sourceReject.injEq]
    decide
  simp only [hn]

example : consumeOutcome observedPrimitives fixturePrimitives fixtureReference identityCopy ⟨1, 1, .text "none"⟩
    "target_only_structured_erp" (.boolean false) (recordFields (pooledFields 1).raw) (artifactRecord boundaryArtifact)
    (producer boundaryPayload.arrays PersonalERPCurrentConsumer.floatingSelector) =
      .error (.inherited (.sourceReject "ERP state contains nonfinite values")) := by
  rw [selector_refusal_precedes_nonfinite_ERP_output]
  have hn : (Except.error (.inherited (.sourceReject "ERP selected dtype")) : Result Bound) ≠
      .error (.inherited (.sourceReject "ERP state contains nonfinite values")) := by
    simp only [ne_eq, Except.error.injEq, Fault.inherited.injEq, LedgerFault.sourceReject.injEq]
    decide
  simp only [hn]

end N2P3.NegativePersonalERPFitOutcome
