import N2P3.PersonalERPArrayOrder

namespace N2P3.FalsePersonalERPArrayOrder
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.ERPTemporalNPZ
open N2P3.PersonalERPArrayIngress N2P3.PersonalERPArrayOrder
noncomputable section

theorem false_wrong_length_labels_reach_dtype (p : Operations) (labels : RawArray) (N : ℕ)
    (timeline : Fin 17 → RawArray) (required : Bool) (shape : labels.shape ≠ [N]) :
    labelChecks p (some labels) N timeline required=.error (.sourceValueError "nonbool integer labels") := by
  simp [label_shape_precedes_dtype p labels N timeline required shape]

theorem false_float_labels_reach_candidate_check (p : Operations) (labels : RawArray) (N : ℕ)
    (timeline : Fin 17 → RawArray) (required : Bool)
    (shape : labels.shape=[N]) (dtype : labels.kind.integer ≠ true) :
    labelChecks p (some labels) N timeline required=
      .error (.sourceValueError "binary labels agree with candidate targets") := by
  simp [label_dtype_follows_shape p labels N timeline required shape dtype]

theorem false_trial_dtype_precedes_static_support (N C : ℕ) (static : RawArray) (trial : Option RawArray)
    (dtype : static.kind=.boolean) (shape : static.shape=[C])
    (absent : true ∉ static.cells.map cellBool) :
    maskHeader N C static trial=.error (.sourceValueError "trial channel mask boolean dtype") := by
  simp [static_support_precedes_trial_arrays N C static trial dtype shape absent]

theorem false_wrong_trial_dtype_reaches_shape (N C : ℕ) (static trial : RawArray)
    (dtype : static.kind=.boolean) (shape : static.shape=[C])
    (observed : true ∈ static.cells.map cellBool) (wrong : trial.kind ≠ .boolean) :
    maskHeader N C static (some trial)=.error (.sourceValueError "trial channel mask array shape") := by
  simp [trial_dtype_precedes_trial_shape N C static trial dtype shape observed wrong]

theorem false_rank_one_implies_common_lengths :
    ¬∃ shape : Fin 17 → List ℕ, (∀ j, (shape j).length=1) ∧ shape 0 ≠ shape 9 := by
  simp [rank_one_vectors_need_not_have_common_lengths]

end
end N2P3.FalsePersonalERPArrayOrder
