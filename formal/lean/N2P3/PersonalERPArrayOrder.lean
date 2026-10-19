import N2P3.PersonalERPArrayIngress

namespace N2P3.PersonalERPArrayOrder
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.ERPTemporalNPZ
open N2P3.PersonalERPArrayIngress
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

theorem label_shape_precedes_dtype (p : Operations) (labels : RawArray) (N : ℕ)
    (timeline : Fin 17 → RawArray) (required : Bool) (shape : labels.shape ≠ [N]) :
    labelChecks p (some labels) N timeline required=
      .error (.sourceValueError "one dimensional labels aligned to epochs") := by
  simp [labelChecks,rawShape,shape]

theorem label_dtype_follows_shape (p : Operations) (labels : RawArray) (N : ℕ)
    (timeline : Fin 17 → RawArray) (required : Bool)
    (shape : labels.shape=[N]) (dtype : labels.kind.integer ≠ true) :
    labelChecks p (some labels) N timeline required=.error (.sourceValueError "nonbool integer labels") := by
  simp [labelChecks,rawShape,shape,dtype]

theorem static_support_precedes_trial_arrays (N C : ℕ) (static : RawArray) (trial : Option RawArray)
    (dtype : static.kind=.boolean) (shape : static.shape=[C])
    (absent : true ∉ static.cells.map cellBool) :
    maskHeader N C static trial=.error (.sourceValueError "at least one static observed channel") := by
  simp [maskHeader,rawShape,require,dtype,shape,absent]

theorem trial_dtype_precedes_trial_shape (N C : ℕ) (static trial : RawArray)
    (dtype : static.kind=.boolean) (shape : static.shape=[C])
    (observed : true ∈ static.cells.map cellBool) (wrong : trial.kind ≠ .boolean) :
    maskHeader N C static (some trial)=.error (.sourceValueError "trial channel mask boolean dtype") := by
  simp [maskHeader,rawShape,require,dtype,shape,observed,wrong]

theorem native_boolean_index_failure_occurs_at_use (raw : RawArray) (mask : List Bool) (site : String)
    (shape : raw.shape ≠ [mask.length]) :
    masked raw mask site=.error (.inherited (.fieldFailure (site ++ " boolean index shape mismatch"))) := by
  simp [masked,shape]

theorem rank_one_vectors_need_not_have_common_lengths :
    ∃ shape : Fin 17 → List ℕ,
      (∀ j, (shape j).length=1) ∧ shape 0 ≠ shape 9 := by
  refine ⟨fun j => if j=9 then [2] else [1],?_,?_⟩
  · intro j
    by_cases h : j=9 <;> simp [h]
  · simp

end
end N2P3.PersonalERPArrayOrder
