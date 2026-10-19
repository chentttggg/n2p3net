import N2P3.RequestedOutcomeCounts

/-! A participant/key view of the original accepted accounting. This proves
local denominators from the actual global key checks, not from a free vector
of participant counts. Source: analyzer lines1008-1028 and runner endpoints. -/

namespace N2P3.RequestedOutcomeRestriction

open N2P3.RequestedOutcomes

noncomputable section
attribute [local instance] Classical.propDecidable

theorem projected_restricted_keys {Key : Type} (outcomes : List (Outcome Key)) (keep : Key → Bool) :
    observedKeys (outcomes.filter (fun outcome => keep outcome.key))=
      (observedKeys outcomes).filter (fun pair => keep pair.1) := by
  induction outcomes with
  | nil => rfl
  | cons outcome tail ih =>
    cases kept : keep outcome.key <;>
      simpa [observedKeys,Outcome.pair,kept] using ih

theorem restriction_preserves_actual_requested_key_checks {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (accepted : validateKeys requested levels outcomes=.ok ()) (keep : Key → Bool) :
    validateKeys (requested.filter keep) levels (outcomes.filter (fun outcome => keep outcome.key))=.ok () := by
  obtain ⟨requestedUnique,validLevels,observedUnique,complete⟩ :=
    (validated_keys_iff_actual_checks requested levels outcomes).mp accepted
  apply (validated_keys_iff_actual_checks _ _ _).mpr
  refine ⟨requestedUnique.filter keep,validLevels,?_,?_⟩
  · rw [projected_restricted_keys]
    exact observedUnique.filter _
  · rw [projected_restricted_keys]
    ext pair
    simp only [List.mem_toFinset,List.mem_filter,Finset.product_eq_sprod,Finset.mem_product]
    rw [requested_pair_membership requested levels outcomes complete pair]
    constructor
    · rintro ⟨⟨key,level⟩,kept⟩
      exact ⟨⟨key,kept⟩,level⟩
    · rintro ⟨⟨key,kept⟩,level⟩
      exact ⟨⟨key,level⟩,kept⟩

theorem local_correct_is_original_filtered_status_count {Key : Type}
    (outcomes : List (Outcome Key)) (keep : Key → Bool) (level : ℕ) :
    countAt (outcomes.filter (fun outcome => keep outcome.key)) level .correct=
      (outcomes.filter (fun outcome => keep outcome.key &&
        decide (outcome.level=level ∧ outcome.status=.correct))).length := by
  simp [countAt,List.filter_filter,Bool.and_comm]

theorem checked_local_correct_never_exceeds_its_actual_request {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (accepted : validateKeys requested levels outcomes=.ok ()) (keep : Key → Bool)
    (level : ℕ) (member : level ∈ levels) :
    (countsAt (outcomes.filter (fun outcome => keep outcome.key)) level).correct ≤ (requested.filter keep).length := by
  have checked := restriction_preserves_actual_requested_key_checks requested levels outcomes accepted keep
  have total := checked_counts_preserve_the_requested_denominator
    (requested.filter keep) levels (outcomes.filter (fun outcome => keep outcome.key)) checked level member
  rw [←total]
  have bounds := natural_count_bounds (countsAt (outcomes.filter (fun outcome => keep outcome.key)) level)
  exact bounds.1.trans bounds.2

end
end N2P3.RequestedOutcomeRestriction
