import N2P3.PersonalERPNineInputContract

/-!
# Nine-event public input reaches the actual identity and participant tail

This is a finite admission witness. Individual identifier operations are
supplied by Leaves; no identity/participant/validator success is an input.
The returned identity cache is explicit and does not assert heap aliasing.
-/
namespace N2P3.PersonalERPNineInputWitness
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.ERPTemporalNPZ N2P3.PersonalERPArrayIngress

noncomputable section
set_option maxRecDepth 8192
set_option maxHeartbeats 1000000
set_option pp.maxSteps 20000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

theorem actual_subject_strings_are_the_nine_original_rows (p : PersonalERPExecutionRuntime.Primitives) :
    PersonalERPRawInputTail.subjectStrings (operations p) raw=.ok (List.replicate 9 "P001") := by
  simp [PersonalERPRawInputTail.subjectStrings,operations,nativeCast,raw,texts,cellText]

theorem source_identity_pairs_follow_the_actual_evidence_scatter
    (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    PersonalERPRawInputTail.sourcePairs (operations p) raw 9=
      .ok (List.replicate 9 ("P001",expectedAuthority)) := by
  simp only [PersonalERPRawInputTail.sourcePairs,nine_event_timeline_is_admitted p h]
  simp [operations,nativeCast,raw,timelineArrays,texts,integers,cellInteger,cellText,
    masked,PersonalERPRawInputTail.scatter,PersonalERPRawInputTail.subjectStrings,
    PersonalERPInput.cleanString,PersonalERPInput.originKey,expectedAuthority,
    require,liftCheck,ensure,h.strip,literalStrings,h.quoteSource,h.quoteSubject]

theorem source_rows_construct_one_original_participant :
    PersonalERPInput.constructSourceRows (List.replicate 9 ("P001",expectedAuthority))=.ok expectedIdentityRows := by
  simp [PersonalERPInput.constructSourceRows,PersonalERPInput.checkIdentityTableConstructor,
    expectedIdentityRows,ensure]

theorem actual_identity_materialization_carries_the_cache
    (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) (checked : Validated raw) :
    PersonalERPRawInputTail.materializeIdentity (operations p) raw ⟨checked,none⟩=
      .ok (expectedIdentityRows,⟨checked,some expectedIdentityRows⟩) := by
  have rows := (every_guarded_raw_result_has_nine_rows checked).1
  simp only [PersonalERPRawInputTail.materializeIdentity,rows,
    source_identity_pairs_follow_the_actual_evidence_scatter p h,
    source_rows_construct_one_original_participant,liftCheck,
    Except.bind,Except.pure,Except.mapError,Bind.bind,Pure.pure]

theorem actual_participant_resolves_the_source_authority
    (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) (checked : Validated raw) :
    PersonalERPRawInputTail.participant (operations p) raw ⟨checked,none⟩=
      .ok ("P001",expectedAuthority,⟨checked,some expectedIdentityRows⟩) := by
  simp only [PersonalERPRawInputTail.participant,actual_subject_strings_are_the_nine_original_rows,
    actual_identity_materialization_carries_the_cache p h checked,
    Except.bind,Except.pure,Bind.bind,Pure.pure]
  simp [operations,
    PersonalERPInput.recordFor,PersonalERPInput.cleanString,PersonalERPInput.authorityKey,
    expectedIdentityRows,liftCheck,ensure,h.strip,literalStrings]

def physicalMetadata : Fields := Fields.ofList
  [("input_contract",.object expectedContract),("participant_key",.text expectedAuthority)]

theorem complete_raw_input_from_individual_metadata_fields_and_scalar_operations
    (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) (metadata : Fields)
    (contractField : metadata.get "input_contract"=some (.object expectedContract))
    (participantField : metadata.get "participant_key"=some (.text expectedAuthority))
    (bindingField : metadata.get "preprocessing_execution_binding"=none) :
    ∃ state : PersonalERPRawInputTail.State raw,
      PersonalERPRawInputTail.validateInput (operations p) metadata none raw=.ok state ∧
      state.checked.tensor.N=9 ∧ state.identityCache=some expectedIdentityRows := by
  obtain ⟨checked,admitted,nine⟩ := actual_nine_row_validator_success_exists p h
  refine ⟨⟨checked,some expectedIdentityRows⟩,?_,nine,rfl⟩
  simp only [PersonalERPRawInputTail.validateInput,admitted,input_contract_from_declared_fields p h checked,
    actual_participant_resolves_the_source_authority p h checked,show raw.identityCache=none from rfl,
    Except.bind,Except.pure,Bind.bind,Pure.pure]
  simp [PersonalERPModelRuntime.objectAt,ERPState.objectAt,
    contractField,participantField,bindingField,
    PersonalAdmission.PyEq,PersonalAdmission.normalizeValue,fieldDefault,Fields.get,
    require,liftCheck]

theorem complete_raw_input_from_individual_fields_and_scalar_operations
    (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    ∃ state : PersonalERPRawInputTail.State raw,
      PersonalERPRawInputTail.validateInput (operations p) physicalMetadata none raw=.ok state ∧
      state.checked.tensor.N=9 ∧ state.identityCache=some expectedIdentityRows :=
  complete_raw_input_from_individual_metadata_fields_and_scalar_operations p h physicalMetadata rfl rfl rfl

end
end N2P3.PersonalERPNineInputWitness
