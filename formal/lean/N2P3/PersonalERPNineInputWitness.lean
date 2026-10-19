import N2P3.PersonalERPNineInputFields

namespace N2P3.PersonalERPNineInputWitness
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.ERPTemporalNPZ N2P3.PersonalERPArrayIngress

noncomputable section
set_option maxRecDepth 8192
set_option maxHeartbeats 500000
set_option pp.maxSteps 2000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

@[simp] theorem text_cast_is_one_identity_operation (p : PersonalERPExecutionRuntime.Primitives) (values : List String) :
    toTexts (operations p) (texts values)=.ok (texts values) := by
  simp [toTexts,operations,nativeCast,texts]

@[simp] theorem float_cast_is_one_identity_operation (p : PersonalERPExecutionRuntime.Primitives)
    (shape : List ℕ) (values : List ℝ) (length : values.length=shape.prod) :
    toFloats (operations p) (floats shape values length)=.ok (floats shape values length) := by
  simp [toFloats,operations,nativeCast,floats]

@[simp] theorem self_indexed_branch (f : Fin 17 → RawArray) (j k : Fin 17) :
    (if j=k then f k else f j)=f j := by
  by_cases h : j=k <;> simp [h]

theorem timeline_fields_have_actual_rank_one :
    ∀ j, (raw.timeline.arrays j).shape.length=1 := by decide

theorem timeline_event_count_is_nine : PersonalERPArrayIngress.arrayLength (raw.timeline.arrays 0)=.ok 9 := rfl

theorem string_timeline_cast_at_source_indices (p : PersonalERPExecutionRuntime.Primitives) (j : Fin 17)
    (member : j ∈ ([0,1,2,8,9,10,11,12,13,14,15] : List (Fin 17))) :
    toTexts (operations p) (raw.timeline.arrays j)=.ok (raw.timeline.arrays j) := by
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals exact text_cast_is_one_identity_operation p _

theorem float_timeline_cast_at_source_indices (p : PersonalERPExecutionRuntime.Primitives) (j : Fin 17)
    (member : j ∈ ([5,6] : List (Fin 17))) :
    toFloats (operations p) (raw.timeline.arrays j)=.ok (raw.timeline.arrays j) := by
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at member
  rcases member with rfl | rfl
  all_goals exact float_cast_is_one_identity_operation p _ _ rfl

theorem literal_candidate_sort :
    (["1","2","3","4","5","6","7","8","9"] : List String).mergeSort (fun a b => decide (a ≤ b))=
      ["1","2","3","4","5","6","7","8","9"] := by
  letI : IsTotal String (fun a b => a ≤ b) := ⟨String.le_total⟩
  letI : IsTrans String (fun a b => a ≤ b) := ⟨fun _ _ _ hab hbc => String.le_trans hab hbc⟩
  letI : IsAntisymm String (fun a b => a ≤ b) := ⟨fun _ _ hab hba => String.le_antisymm hab hba⟩
  apply List.mergeSort_eq_self (r := (fun a b : String => a ≤ b))
  simp [List.Sorted]

theorem decoded_timeline_preserves_all_actual_arrays (p : PersonalERPExecutionRuntime.Primitives) :
    decodeTimelineArrays (operations p) raw.timeline=.ok raw.timeline.arrays := by
  simp [decodeTimelineArrays,string_timeline_cast_at_source_indices,
    float_timeline_cast_at_source_indices]

theorem nine_event_timeline_is_admitted (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    timelineValidate (operations p) raw.timeline 9=.ok raw.timeline.arrays := by
  simp [timelineValidate,timelineValidateOptional,decoded_timeline_preserves_all_actual_arrays,
    validateDecodedTimeline,timeline_fields_have_actual_rank_one,
    timeline_event_count_is_nine,require,string_timeline_cast_at_source_indices]
  simp [operations,nativeFinite,nativeIntegers,nativeLess,nativeLessEqual,nativeGreaterEqual,
    nativeEqual,nativeSubtract,nativeAdd,comparedZero,assertFinite,masked,nativeRows,nthCell,
    timelineGroups,PersonalERPArrayIngress.groupRows,adjacent,homogeneous,
    raw,timelineArrays,texts,integers,floats,cellText,cellAtom,cellInteger,atomNumber,Kind.integer,
    finiteNumber,zeroCell,integerCell,require,h.strip,literalStrings,timelineStatuses,List.range_succ,
    literal_candidate_sort]
  all_goals norm_num

end
end N2P3.PersonalERPNineInputWitness
