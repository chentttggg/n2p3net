import N2P3.ArtifactFoldSupport

/-!
# Frozen masking and the distinct training/default-test/excluded-test ledgers

Source artifact.py:540-585,682-796 and evaluate.py:550-623. These equations
model mask/ledger stages after existing input and fit-scope gates. They do not
claim that the whole apply wrapper ignores heldout values: transformation,
test validity, reported metrics and diagnostics deliberately inspect them.
Only the trained-state interface in ArtifactFoldSupport isolates outer fit.
-/
namespace N2P3.ArtifactFoldLedger
noncomputable section
attribute [local instance] Classical.propDecidable

def effective (observed bad : Bool) : Bool := observed && !bad
def project (mask : Bool) (value : ℝ) : ℝ := if mask then value else 0

theorem frozen_local_masking_never_enables_an_unobserved_channel
    (observed bad : Bool) (retained : effective observed bad=true) : observed=true := by
  cases observed <;> cases bad <;> simp_all [effective]

theorem explicit_and_deferred_zero_fill_agree_after_the_actual_mask_projection
    (observed bad : Bool) (value : ℝ) :
    project (effective observed bad) (project (effective observed bad) value)=
      project (effective observed bad) value := by
  cases observed <;> cases bad <;> rfl

def effectiveTrain {R : Type*} [DecidableEq R] (training : Finset R) (drop : R → Bool) : Finset R :=
  training.filter (fun row => drop row=false)

def excludedTest {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) : Finset R :=
  test.filter (fun row => allBad row=true)

def effectiveTest {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) : Finset R :=
  test.filter (fun row => allBad row=false)

theorem explicit_test_retention_and_exclusion_partition_the_original_heldout_support
    {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) :
    Disjoint (effectiveTest test allBad) (excludedTest test allBad) ∧
      effectiveTest test allBad ∪ excludedTest test allBad=test := by
  constructor
  · apply Finset.disjoint_left.mpr
    intro row retained excluded
    have good := (Finset.mem_filter.mp retained).2
    have bad := (Finset.mem_filter.mp excluded).2
    rw [good] at bad
    cases bad
  · ext row
    cases value : allBad row <;> simp [effectiveTest,excludedTest,value]

theorem the_reported_heldout_denominator_is_retained_plus_explicitly_excluded
    {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) :
    (effectiveTest test allBad).card+(excludedTest test allBad).card=test.card := by
  have partition := explicit_test_retention_and_exclusion_partition_the_original_heldout_support test allBad
  rw [← Finset.card_union_of_disjoint partition.1,partition.2]

theorem train_drops_and_explicit_heldout_exclusion_cannot_create_a_train_test_overlap
    {R : Type*} [DecidableEq R] (training test : Finset R) (drop allBad : R → Bool)
    (separate : Disjoint training test) : Disjoint (effectiveTrain training drop) (effectiveTest test allBad) := by
  apply separate.mono (Finset.filter_subset _ _) (Finset.filter_subset _ _)

inductive HeldoutPolicy where
  | failClosed
  | explicitExclusion
  deriving DecidableEq

inductive Failure where
  | everyChannelMaskedOnSomeHeldoutEpoch (count : ℕ)
  | noHeldoutEvidence
  deriving DecidableEq

structure HeldoutLedger (R : Type*) where
  scoringRows : Finset R
  excludedCount : ℕ

def heldoutPhase {R : Type*} [DecidableEq R] (policy : HeldoutPolicy) (test : Finset R) (allBad : R → Bool) :
    Except Failure (HeldoutLedger R) :=
  match policy with
  | .failClosed =>
    if (excludedTest test allBad).Nonempty then
      .error (.everyChannelMaskedOnSomeHeldoutEpoch (excludedTest test allBad).card)
    else .ok ⟨test,0⟩
  | .explicitExclusion =>
    if (effectiveTest test allBad).Nonempty then
      .ok ⟨effectiveTest test allBad,(excludedTest test allBad).card⟩
    else .error .noHeldoutEvidence

theorem default_heldout_policy_either_errors_or_preserves_every_original_scoring_row
    {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) (ledger : HeldoutLedger R)
    (returned : heldoutPhase .failClosed test allBad=.ok ledger) :
    ledger.scoringRows=test ∧ ledger.excludedCount=0 := by
  by_cases bad : (excludedTest test allBad).Nonempty
  · simp only [heldoutPhase,if_pos bad] at returned
    cases returned
  · have same : (⟨test,0⟩ : HeldoutLedger R)=ledger := by
      simpa only [heldoutPhase,if_neg bad,Except.ok.injEq] using returned
    cases same
    exact ⟨rfl,rfl⟩

theorem an_all_masked_heldout_row_triggers_the_default_error_instead_of_a_hidden_denominator_change
    {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) (row : R)
    (member : row ∈ test) (masked : allBad row=true) :
    heldoutPhase .failClosed test allBad=
      .error (.everyChannelMaskedOnSomeHeldoutEpoch (excludedTest test allBad).card) := by
  have bad : (excludedTest test allBad).Nonempty := ⟨row,Finset.mem_filter.mpr ⟨member,masked⟩⟩
  simp only [heldoutPhase,if_pos bad]

theorem successful_explicit_exclusion_keeps_the_original_requested_denominator_visible
    {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool) (ledger : HeldoutLedger R)
    (returned : heldoutPhase .explicitExclusion test allBad=.ok ledger) :
    ledger.scoringRows=effectiveTest test allBad ∧
      ledger.scoringRows.card+ledger.excludedCount=test.card := by
  by_cases remain : (effectiveTest test allBad).Nonempty
  · have same : (⟨effectiveTest test allBad,(excludedTest test allBad).card⟩ : HeldoutLedger R)=ledger := by
      simpa only [heldoutPhase,if_pos remain,Except.ok.injEq] using returned
    cases same
    exact ⟨rfl,the_reported_heldout_denominator_is_retained_plus_explicitly_excluded test allBad⟩
  · simp only [heldoutPhase,if_neg remain] at returned
    cases returned

theorem explicit_exclusion_with_no_scoring_rows_requires_abstention
    {R : Type*} [DecidableEq R] (test : Finset R) (allBad : R → Bool)
    (noneRemain : effectiveTest test allBad=∅) :
    heldoutPhase .explicitExclusion test allBad=.error .noHeldoutEvidence := by
  simp [heldoutPhase,noneRemain]

def scoringView {R Input Label : Type*} [DecidableEq R]
    (order : List R) (rows : Finset R) (input : R → Input) (truth : R → Label) : List (Input × Label) :=
  ArtifactFoldSupport.trainingView order rows (fun row => (input row,truth row))

theorem scoring_inputs_and_truth_use_the_same_returned_original_row_order
    {R Input Label : Type*} [DecidableEq R] (order : List R) (rows : Finset R)
    (input : R → Input) (truth : R → Label) :
    (scoringView order rows input truth).map Prod.fst=ArtifactFoldSupport.trainingView order rows input ∧
      (scoringView order rows input truth).map Prod.snd=ArtifactFoldSupport.trainingView order rows truth := by
  simp [scoringView,ArtifactFoldSupport.trainingView,List.map_map,Function.comp_def]

end
end N2P3.ArtifactFoldLedger
