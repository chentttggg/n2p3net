import N2P3.PersonalERPNineIdentityWitness

namespace N2P3.FalsePersonalERPNineInputWitness
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData N2P3.PersonalERPScalar
open N2P3.PersonalERPArrayIngress N2P3.PersonalERPNineInputWitness
noncomputable section

theorem false_admitted_tensor_has_zero_rows (checked : Validated raw) : checked.tensor.N=0 := by
  simp [(every_guarded_raw_result_has_nine_rows checked).1]

theorem false_input_contract_uses_a_measured_Cz_reference :
    expectedContract.get "source_reference"=some (.text "measured_channel:CZ") := by
  simp [expectedContract,Fields.ofList,Fields.get]

theorem false_original_subject_rows_are_empty (p : N2P3.PersonalERPExecutionRuntime.Primitives) :
    N2P3.PersonalERPRawInputTail.subjectStrings (operations p) raw=.ok [] := by
  simp [actual_subject_strings_are_the_nine_original_rows]

theorem false_evidence_scatter_silently_changes_local_identity
    (p : N2P3.PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    N2P3.PersonalERPRawInputTail.sourcePairs (operations p) raw 9=
      .ok (List.replicate 9 ("P002",expectedAuthority)) := by
  simp [source_identity_pairs_follow_the_actual_evidence_scatter p h]

theorem false_participant_can_resolve_another_subject
    (p : N2P3.PersonalERPExecutionRuntime.Primitives) (h : Leaves p) (checked : Validated raw) :
    N2P3.PersonalERPRawInputTail.participant (operations p) raw ⟨checked,none⟩=
      .ok ("P002",expectedAuthority,⟨checked,some expectedIdentityRows⟩) := by
  simp [actual_participant_resolves_the_source_authority p h checked]

end
end N2P3.FalsePersonalERPNineInputWitness
