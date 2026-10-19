import N2P3.PersonalERPNineInputWitness

namespace N2P3.PersonalERPNineInputWitness
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.ERPTemporalNPZ N2P3.PersonalERPArrayIngress

noncomputable section
set_option maxRecDepth 8192
set_option maxHeartbeats 500000
set_option pp.maxSteps 20000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def fixedTensor : TensorHeader raw.x := ⟨9,1,2,rfl,rfl⟩

def fixedMasks : {masks : MaskHeader 9 1 // masks.staticRaw=raw.staticMask ∧ masks.trialRaw=raw.trialMask} :=
  ⟨⟨raw.staticMask,rfl,rfl,none,by simp,by simp⟩,rfl,rfl⟩

theorem physical_fields_pass :
    checkPhysicalCoordinates raw.historical raw.coordinates
      (if raw.provenanceIsMapping then raw.provenance.get "data_alignment" else none)=.ok () := by
  simp [raw,checkPhysicalCoordinates,ensure,Fields.ofList,Fields.get]

theorem scheduled_fields_pass (p : PersonalERPExecutionRuntime.Primitives) :
    scheduledExclusions (operations p) raw.timeline=.ok () := by
  simp [scheduledExclusions,raw,timelineArrays,texts,
    require,cellText,operations,toTexts,nativeCast,PersonalERPInput.sourceParticipantExcluded,
    PersonalERPInput.startsWith]

theorem tensor_fields_pass : tensorHeader raw.x 2=.ok fixedTensor := by
  simp [tensorHeader,raw,floats,require,fixedTensor]

theorem mask_fields_pass : maskHeader 9 1 raw.staticMask raw.trialMask=.ok fixedMasks := by
  simp [maskHeader,rawShape,raw,booleans,cellBool,require,fixedMasks]

theorem fixed_static_mask_observes_the_channel (c : Fin 1) : staticAt fixedMasks.1 c=true := by
  have zero : c=0 := Subsingleton.elim _ _
  subst c
  rfl

theorem placeholder_fields_pass (p : PersonalERPExecutionRuntime.Primitives) :
    placeholders (operations p) raw.x 2 fixedMasks.1=.ok () := by
  have hidden : ¬∃ c, staticAt fixedMasks.1 c=false := by
    rintro ⟨c,bad⟩
    simp only [fixed_static_mask_observes_the_channel] at bad
    cases bad
  simp only [placeholders,if_neg hidden]
  rfl

theorem observed_coordinate_fields_pass (p : PersonalERPExecutionRuntime.Primitives) :
    observedCoordinates (operations p) raw.positions fixedMasks.1=.ok () := by
  have observed : staticAt fixedMasks.1=(fun _ => true) := funext fixed_static_mask_observes_the_channel
  simp [observedCoordinates,operations,nativeNorm,nativeFiniteRows,nativeLessEqual,nativeLess,
    observed,raw,floats,cellAtom,atomNumber,finiteNumber,
    zeroCell,integerCell,require,List.range_succ]
  all_goals norm_num

theorem subject_alignment_fields_pass (p : PersonalERPExecutionRuntime.Primitives) :
    subjectAlignment (operations p) raw.timeline raw.subjects 9=.ok () := by
  simp [subjectAlignment,toTexts,operations,nativeCast,raw,timelineArrays,texts,integers,
    cellInteger,masked,require]

theorem every_guarded_raw_result_has_nine_rows (checked : Validated raw) :
    checked.tensor.N=9 ∧ checked.tensor.C=1 ∧ checked.tensor.T=2 := by
  have shape := checked.tensor.shape
  change [9,1,2]=[checked.tensor.N,checked.tensor.C,checked.tensor.T] at shape
  obtain ⟨hn,rest⟩ := List.cons.inj shape
  obtain ⟨hc,rest⟩ := List.cons.inj rest
  obtain ⟨ht,_⟩ := List.cons.inj rest
  exact ⟨hn.symm,hc.symm,ht.symm⟩

/-- The success flag is computed from the actual full validator result. It
is a conclusion, never a supplied validity predicate or primitive. -/
theorem full_raw_admission_is_nonempty (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    (PersonalERPArrayIngress.validate (operations p) raw false).toOption.isSome=true := by
  have stripName := h.strip "BrainSync-GTN" (by simp [literalStrings])
  have stripSubject := h.strip "P001" (by simp [literalStrings])
  have spec := preprocessing_from_individual_scalar_atoms p h
  have timeline := nine_event_timeline_is_admitted p h
  have read : readPreprocessingArgs raw.preprocessingArgs=.ok resolvedPreprocessing := preprocessing_fields_are_known
  simp only [PersonalERPArrayIngress.validate,physical_fields_pass,scheduled_fields_pass,
    show raw.identityCache=none from rfl,
    show (operations p).input=p from rfl,
    read,spec,tensor_fields_pass,fixedTensor,mask_fields_pass,observed_coordinate_fields_pass,
    subject_alignment_fields_pass,timeline,placeholder_fields_pass,
    liftCheck,Except.mapError,Except.bind,Except.pure,Bind.bind,Pure.pure]
  simp [operations,
    raw,fixedMasks,canonicalNames,rawShape,labelChecks,
    checkDatasetTail,PersonalERPExecutionRuntime.executionBinding,
    PersonalERPModelRuntime.text,PersonalERPExecution.text,Kind.numeric,
    nativeCast,nativeFinite,texts,integers,floats,booleans,timelineArrays,cellText,cellAtom,
    finiteNumber,require,liftCheck,
    stripName,stripSubject,h.canonical,assertFinite,ensure,
    fieldDefault,Fields.ofList,Fields.get,
    preprocessing,resolvedPreprocessing,Except.toOption]

theorem actual_nine_row_validator_success_exists (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p) :
    ∃ checked : Validated raw, PersonalERPArrayIngress.validate (operations p) raw false=.ok checked ∧
      checked.tensor.N=9 := by
  have admitted := full_raw_admission_is_nonempty p h
  cases result : PersonalERPArrayIngress.validate (operations p) raw false with
  | error fault => simp [result,Except.toOption] at admitted
  | ok checked => exact ⟨checked,rfl,(every_guarded_raw_result_has_nine_rows checked).1⟩

end
end N2P3.PersonalERPNineInputWitness
