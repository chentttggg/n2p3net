import N2P3.PreprocessingFitScope

/-!
# Actual outer-training views and group-disjoint inner CV support

Source-first contract: lean_fold_artifact_20261018/slice_contract.json.
artifact.py:647-679 and evaluate.py:885-908 slice every fit input BEFORE
policy.fit. The learner interface below receives only that ordered training
view and fixed policy; it may fail. Noninterference follows from extraction,
not from a whole-fit correctness/noninterference premise.
Inner validation remains part of outer training and may affect the winner.
Group assignment models the unique-group array_split membership, not row-wise
random splitting. Computed/provided QC fields remain explicit input fields.
-/
namespace N2P3.ArtifactFoldSupport
noncomputable section
attribute [local instance] Classical.propDecidable

structure FitRow (C T : Type*) where
  signal : C → T → ℝ
  observed : C → Bool
  relativePTP : C → ℝ
  channelStdV : C → ℝ
  epochScaleV : ℝ
  subject : String
  group : String

abbrev Learner (Policy Input Error Model : Type*) := Policy → List Input → Except Error Model

def trainingView {R Input : Type*} [DecidableEq R]
    (order : List R) (training : Finset R) (data : R → Input) : List Input :=
  ((order.filter (fun row => row ∈ training)).map data)

theorem ordered_training_view_only_reads_selected_row_fields
    {R Input : Type*} [DecidableEq R] (order : List R) (training : Finset R) (left right : R → Input)
    (same : ∀ row ∈ training, left row=right row) :
    trainingView order training left=trainingView order training right := by
  unfold trainingView
  apply List.map_congr_left
  intro row member
  exact same row (by simpa only [decide_eq_true_eq] using (List.mem_filter.mp member).2)

def editRows {R Input : Type*} [DecidableEq R] (changed : Finset R) (edit : R → Input → Input) (data : R → Input) :
    R → Input := fun row => if row ∈ changed then edit row (data row) else data row

theorem arbitrary_outer_heldout_edits_preserve_the_actual_training_view
    {R Input : Type*} [DecidableEq R] (order : List R) (training heldout : Finset R)
    (edit : R → Input → Input) (data : R → Input) (separate : Disjoint training heldout) :
    trainingView order training (editRows heldout edit data)=trainingView order training data := by
  apply ordered_training_view_only_reads_selected_row_fields
  intro row member
  have outside : row ∉ heldout := fun bad => Finset.disjoint_left.mp separate member bad
  simp only [editRows,if_neg outside]

def fitOnTraining {R Policy Input Error Model : Type*} [DecidableEq R]
    (learner : Learner Policy Input Error Model) (policy : Policy) (order : List R)
    (training : Finset R) (data : R → Input) : Except Error Model :=
  learner policy (trainingView order training data)

theorem source_learner_result_is_isolated_from_arbitrary_outer_heldout_edits
    {R Policy Input Error Model : Type*} [DecidableEq R]
    (learner : Learner Policy Input Error Model) (policy : Policy) (order : List R)
    (training heldout : Finset R) (edit : R → Input → Input) (data : R → Input)
    (separate : Disjoint training heldout) :
    fitOnTraining learner policy order training (editRows heldout edit data)=
      fitOnTraining learner policy order training data := by
  unfold fitOnTraining
  rw [arbitrary_outer_heldout_edits_preserve_the_actual_training_view order training heldout edit data separate]

def innerValidation {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (heldoutFold : Fold) : Finset R :=
  outerTraining.filter (fun row => assignment (group row)=heldoutFold)

def innerTraining {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (heldoutFold : Fold) : Finset R :=
  outerTraining.filter (fun row => assignment (group row) ≠ heldoutFold)

theorem constructed_inner_supports_are_disjoint
    {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (heldoutFold : Fold) :
    Disjoint (innerTraining outerTraining group assignment heldoutFold)
      (innerValidation outerTraining group assignment heldoutFold) := by
  apply Finset.disjoint_left.mpr
  intro row train validation
  exact (Finset.mem_filter.mp train).2 (Finset.mem_filter.mp validation).2

theorem constructed_inner_supports_cover_exactly_outer_training
    {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (heldoutFold : Fold) :
    innerTraining outerTraining group assignment heldoutFold ∪
      innerValidation outerTraining group assignment heldoutFold=outerTraining := by
  ext row
  by_cases owned : assignment (group row)=heldoutFold <;>
    simp [innerTraining,innerValidation,owned]

theorem both_inner_sides_stay_inside_the_authorized_outer_training_support
    {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (heldoutFold : Fold) :
    innerTraining outerTraining group assignment heldoutFold ⊆ outerTraining ∧
      innerValidation outerTraining group assignment heldoutFold ⊆ outerTraining :=
  ⟨Finset.filter_subset _ _,Finset.filter_subset _ _⟩

theorem equal_groups_cannot_cross_training_and_validation
    {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (heldoutFold : Fold)
    (trainRow validationRow : R)
    (train : trainRow ∈ innerTraining outerTraining group assignment heldoutFold)
    (validation : validationRow ∈ innerValidation outerTraining group assignment heldoutFold) :
    group trainRow ≠ group validationRow := by
  intro same
  have notOwned := (Finset.mem_filter.mp train).2
  have owned := (Finset.mem_filter.mp validation).2
  exact notOwned (same ▸ owned)

theorem each_outer_training_row_has_exactly_one_validation_fold
    {R Group Fold : Type*} [DecidableEq R] [DecidableEq Fold]
    (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold) (row : R)
    (member : row ∈ outerTraining) :
    ∃! fold, row ∈ innerValidation outerTraining group assignment fold := by
  refine ⟨assignment (group row),?_,?_⟩
  · simp [innerValidation,member]
  · intro fold included
    exact (Finset.mem_filter.mp included).2.symm

def candidateInputs {R Group Fold Input : Type*} [DecidableEq R] [DecidableEq Fold]
    (order : List R) (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold)
    (heldoutFold : Fold) (clean : R → Bool) (data : R → Input) : List Input × List Input :=
  (trainingView order ((innerTraining outerTraining group assignment heldoutFold).filter (fun row => clean row)) data,
   trainingView order (innerValidation outerTraining group assignment heldoutFold) data)

theorem the_same_validation_target_operation_gets_the_same_unfiltered_argument
    {R Group Fold Input Eligible Target : Type*} [DecidableEq R] [DecidableEq Fold]
    (target : Eligible → List Input → Target) (eligible : Eligible)
    (order : List R) (outerTraining : Finset R) (group : R → Group) (assignment : Group → Fold)
    (heldoutFold : Fold) (first second : R → Bool) (data : R → Input) :
    target eligible (candidateInputs order outerTraining group assignment heldoutFold first data).2=
      target eligible (candidateInputs order outerTraining group assignment heldoutFold second data).2 := rfl

end
end N2P3.ArtifactFoldSupport
