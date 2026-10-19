import N2P3.ERPTemporalRawPreprocessing

/- Independent wrong alternatives fixed in raw_validation_adapter_design_01.
These are declaration/field claims, not claims of full Dataset admission. -/

namespace N2P3.NegativeERPTemporalRawPreprocessing

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPScalar N2P3.ERPTemporalJSON
open N2P3.ERPTemporalRawPreprocessing

private theorem error_ne_success {ε α : Type} (error : ε) (value : α) :
    (Except.error error : Except ε α) ≠ .ok value := by intro equal; cases equal

private theorem success_ne_error {ε α : Type} (error : ε) (value : α) :
    (Except.ok value : Except ε α) ≠ .error error := by intro equal; cases equal

example : realType (.floating .nan) "numeric type"=.error (.sourceValueError "numeric type") := by
  have accepted : realType (.floating .nan) "numeric type"=.ok () := by
    simp [realType,ERPTemporalRawPreprocessing.numericNonBool,require]
  rw [accepted]
  simp only [success_ne_error]

example (p : Operations) (fields : JFields) (tmin tmax : ℝ)
    (center : ERPTemporalRawFields.fieldDefault fields "trial_reference_center"=.text "mean")
    (scale : ERPTemporalRawFields.fieldDefault fields "trial_reference_scale"=.text "none")
    (baseline : ¬pythonEqual (ERPTemporalRawFields.fieldDefault fields "baseline_mode") (.text "trial_reference")) :
    checkWindow p fields tmin tmax=.error (.sourceValueError "all unused windows must be finite") := by
  rw [nonreference_mode_ignores_unused_window p fields tmin tmax center scale baseline]
  simp only [success_ne_error]

example (p : Operations) (quality : JValue) :
    checkNotches p (.sequence (.cons (.boolean true) .nil)) quality=.ok ([],30) := by
  rw [Boolean_notch_refused_before_quality]
  simp only [error_ne_success]

end N2P3.NegativeERPTemporalRawPreprocessing
