import N2P3.RequestedOutcomes

/-! V05.a source Counter/partition correspondence and the actual count-loop
assertion. The nonempty six-status/two-level witness was fixed in source
probe P01 before these proofs. No source score or full fit is a primitive. -/

namespace N2P3.RequestedOutcomeCounts

open N2P3.RequestedOutcomes

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

theorem status_count_equals_actual_partition_card {Key : Type} [DecidableEq Key]
    (outcomes : List (Outcome Key)) (unique : (observedKeys outcomes).Nodup)
    (level : ℕ) (status : Status) :
    countAt outcomes level status=(terminalKeys outcomes level status).card := by
  let selected := outcomes.filter (fun outcome => decide (outcome.level=level ∧ outcome.status=status))
  have rowsUnique : selected.Nodup :=
    (List.Nodup.of_map Outcome.pair unique).filter _
  have keysUnique : (selected.map Outcome.key).Nodup := by
    apply List.Nodup.map_on ?_ rowsUnique
    intro a am b bm sameKey
    have af : a ∈ outcomes ∧ a.level=level ∧ a.status=status := by
      simpa only [selected,List.mem_filter,decide_eq_true_eq] using am
    have bf : b ∈ outcomes ∧ b.level=level ∧ b.status=status := by
      simpa only [selected,List.mem_filter,decide_eq_true_eq] using bm
    apply List.inj_on_of_nodup_map unique af.1 bf.1
    simp [Outcome.pair,sameKey,af.2.1,bf.2.1]
  have cardinal := List.toFinset_card_of_nodup keysUnique
  simpa only [terminalKeys,countAt,selected,List.length_map] using cardinal.symm

theorem entire_required_product_length {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (accepted : validateKeys requested levels outcomes=.ok ()) :
    outcomes.length=requested.length*levels.length := by
  obtain ⟨requestedUnique,validLevels,observedUnique,complete⟩ :=
    (validated_keys_iff_actual_checks requested levels outcomes).mp accepted
  have cardinal := congrArg Finset.card complete
  rw [List.toFinset_card_of_nodup observedUnique] at cardinal
  simp only [Finset.product_eq_sprod,Finset.card_product,
    List.toFinset_card_of_nodup requestedUnique,List.toFinset_card_of_nodup validLevels.2.2] at cardinal
  simpa only [observedKeys,List.length_map] using cardinal

theorem count_loop_when_generated_counts_match {Key : Type} (requestedCount : ℕ)
    (outcomes : List (Outcome Key)) (levels : List ℕ)
    (counts : ∀ level ∈ levels, (countsAt outcomes level).total=requestedCount) :
    countLevels requestedCount outcomes levels=
      .ok (levels.map (fun level => ⟨level,countsAt outcomes level⟩)) := by
  induction levels with
  | nil => rfl
  | cons level rest ih =>
    have head := counts level (by simp)
    have tail : ∀ value ∈ rest, (countsAt outcomes value).total=requestedCount :=
      fun value member => counts value (by simp [member])
    simp [countLevels,head,ih tail]

theorem checked_builder_assertion_cannot_fail {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (accepted : validateKeys requested levels outcomes=.ok ()) :
    buildAccounting requested levels outcomes=
      .ok ⟨requested.length,levels.map (fun level => ⟨level,countsAt outcomes level⟩)⟩ := by
  have counts : ∀ level ∈ levels, (countsAt outcomes level).total=requested.length :=
    fun level member => checked_counts_preserve_the_requested_denominator requested levels outcomes accepted level member
  simp [buildAccounting,accepted,count_loop_when_generated_counts_match requested.length outcomes levels counts]

/-- All three properties come from the same original sequence and its key
checks; the count vector is not supplied independently of the terminal sets. -/
theorem checked_full_requested_terminal_partition {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (accepted : validateKeys requested levels outcomes=.ok ()) (level : ℕ) (member : level ∈ levels) :
    (∀ left right, left ≠ right → Disjoint (terminalKeys outcomes level left) (terminalKeys outcomes level right)) ∧
      allStatuses.biUnion (terminalKeys outcomes level)=requested.toFinset ∧
      (∀ status, countAt outcomes level status=(terminalKeys outcomes level status).card) ∧
      (countsAt outcomes level).total=requested.length := by
  obtain ⟨_,_,unique,complete⟩ := (validated_keys_iff_actual_checks requested levels outcomes).mp accepted
  exact ⟨fun left right different => different_terminal_statuses_are_disjoint outcomes unique level left right different,
    terminal_union_is_entire_request requested levels outcomes complete level member,
    fun status => status_count_equals_actual_partition_card outcomes unique level status,
    checked_counts_preserve_the_requested_denominator requested levels outcomes accepted level member⟩

theorem count_at_invariant_under_outcome_permutation {Key : Type}
    (left right : List (Outcome Key)) (same : left.Perm right) (level : ℕ) (status : Status) :
    countAt left level status=countAt right level status := by
  exact (same.filter (fun outcome => decide (outcome.level=level ∧ outcome.status=status))).length_eq

def fixtureRequested : List ℕ := [0,1,2,3,4,5]

def fixtureLevels : List ℕ := [1,2]

def fixtureRows (level : ℕ) : List (Outcome ℕ) :=
  [⟨0,level,.correct⟩,⟨1,level,.incorrect⟩,⟨2,level,.tie⟩,
   ⟨3,level,.abstain⟩,⟨4,level,.incomplete⟩,⟨5,level,.fitFailure⟩]

def fixtureOutcomes : List (Outcome ℕ) := fixtureRows 1 ++ fixtureRows 2

theorem nonempty_six_status_two_level_keys_pass :
    validateKeys fixtureRequested fixtureLevels fixtureOutcomes=.ok () := by
  apply (validated_keys_iff_actual_checks _ _ _).mpr
  refine ⟨by simp [fixtureRequested],by simp [ValidLevels,fixtureLevels],?_,?_⟩
  · simp [observedKeys,fixtureOutcomes,fixtureRows,Outcome.pair]
  · ext pair
    rcases pair with ⟨key,level⟩
    simp [observedKeys,fixtureRequested,fixtureLevels,fixtureOutcomes,fixtureRows,
      Outcome.pair,Finset.product_eq_sprod,Finset.mem_product]
    omega

theorem nonempty_fixture_builds_every_terminal_count :
    buildAccounting fixtureRequested fixtureLevels fixtureOutcomes=
      .ok ⟨6,[⟨1,⟨1,1,1,1,1,1⟩⟩,⟨2,⟨1,1,1,1,1,1⟩⟩]⟩ := by
  rw [checked_builder_assertion_cannot_fail _ _ _ nonempty_six_status_two_level_keys_pass]
  simp [fixtureRequested,fixtureLevels,fixtureOutcomes,fixtureRows,countsAt,countAt]

end
end N2P3.RequestedOutcomeCounts
