import N2P3.GeneralMembershipOperational

/-! Independent wrong alternatives from the unchanged source-first contract.
These are finite inhabited inputs; source-refused relational inputs test the
admission gate, and complete direct inputs test the operational scoring gate. -/
namespace N2P3.FalseGeneralMembership
open N2P3.GeneralMembershipOperational N2P3.RelativeCandidateTie
noncomputable section

def repeatedSlots : Ledger (Fin 2) (Fin 2) (Fin 2) 1 :=
  ⟨fun _ => 0,fun _ => 0,fun _ c => decide (c=0),fun e => some e.val,true⟩

theorem repeated_slots_have_unique_observed_evidence : uniqueEvidence repeatedSlots := by
  intro a b index left right
  apply Fin.ext
  simpa [repeatedSlots] using left.trans right.symm

theorem repeated_slots_really_duplicate : ¬Function.Injective (slot repeatedSlots) := by
  intro unique
  have same : (0 : Fin 2)=1 := unique (show slot repeatedSlots 0=slot repeatedSlots 1 from rfl)
  exact (by decide : (0 : Fin 2) ≠ 1) same

example : scheduleRelationCheck repeatedSlots=none := by
  rw [duplicate_slot_is_refused_after_evidence_identity_check repeatedSlots
    repeated_slots_have_unique_observed_evidence repeated_slots_really_duplicate]
  simp

example : Function.Surjective (slot repeatedSlots) := by
  have absent : ¬Function.Surjective (slot repeatedSlots) := by
    intro all
    obtain ⟨e,same⟩ := all (0,1)
    have second := congrArg Prod.snd same
    norm_num [slot,repeatedSlots] at second
  simp only [absent]

example : selected (fun _ : Fin 2 => none) (some (fun _ => true)) 0=true := by
  norm_num [selected,physicallyObserved]

example : availabilityAdmitted (fun _ : Fin 2 => none) (fun _ => true) := by
  have refused := supplied_mask_cannot_restore_physical_missing_event
    (fun _ : Fin 2 => none) (fun _ => true) 0 rfl rfl
  simp only [refused]

example : (operational (directLedger 5 9) (fun _ => false) (fun c => if c=0 then 10 else 0)).1=some 0 := by
  rw [no_usable_evidence_refusal_precedes_every_finite_prior_score]
  simp

def indistinguishable : Ledger (Fin 2) (Fin 2) (Fin 2) 1 :=
  ⟨fun _ => 0,id,fun _ _ => true,fun e => some e.val,true⟩

theorem indistinguishable_columns : ¬identifiable (fun _ : Fin 2 => true) indistinguishable.member := by
  exact duplicate_available_columns_are_unidentifiable _ _ 0 1 (by decide) (fun _ _ => rfl)

example : identifiable (fun _ : Fin 2 => true) indistinguishable.member := by
  simp only [indistinguishable_columns]

example : (operational indistinguishable (fun _ => true) (fun c => if c=0 then 10 else 0)).1=some 0 := by
  have actual := unidentifiable_columns_refuse_even_with_unique_prior_posterior indistinguishable
    (fun _ => true) (fun c => if c=0 then 10 else 0) ⟨0,rfl⟩
    (by simp [sourceComplete,scheduleComplete,indistinguishable]) indistinguishable_columns
  rw [actual]
  simp

-- A source-false timeline flag cannot be repaired by all observations being present.
example : sourceComplete {directLedger 5 9 with timelineComplete := false} (fun _ => true) := by
  norm_num [sourceComplete,scheduleComplete]

-- Exact source-relative margin, not just a strict argmax, controls operation.
example : prediction (fun j => fixture j+10) sourceAtol sourceRtol=some 1 := by
  rw [fixture_after_shift_abstains]
  simp

example : prediction fixture sourceAtol sourceRtol=
    prediction (fun j => fixture j+10) sourceAtol sourceRtol := by
  rw [fixture_before_shift_predicts,fixture_after_shift_abstains]
  simp

example : threshold (fun j => fixture j+10) sourceAtol sourceRtol=sourceAtol := by
  have maximum : N2P3.CandidateTie.IsMaximum (fun j => fixture j+10) 1 :=
    (N2P3.CandidateTie.maximum_add_constant_iff fixture 1 10).mpr fixture_actual_maximum
  rw [threshold_at_any_maximum _ sourceAtol sourceRtol 1 maximum]
  norm_num [fixture,sourceAtol,sourceRtol]

end
end N2P3.FalseGeneralMembership
