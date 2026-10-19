import N2P3.RequestedOutcomes

/-! Status-dependent target/prediction/coverage/reason checks in
DecisionOutcome.__post_init__. Coverage is its helper's Boolean; the external
frozen candidate task is a separate check. This is the source truth-predicate
layer, not Python object coercion or timing/IEEE refinement. -/

namespace N2P3.RequestedConstructorFields

open N2P3.RequestedOutcomes

noncomputable section
attribute [local instance] Classical.propDecidable

structure Fields where
  status : Status
  completeCoverage : Bool
  target : Option ℕ
  prediction : Option ℕ
  nonemptyReason : Bool

def statusFieldsValid (fields : Fields) : Prop :=
  match fields.status with
  | .correct => fields.completeCoverage=true ∧ fields.target≠none ∧ fields.prediction≠none ∧
      fields.prediction=fields.target
  | .incorrect => fields.completeCoverage=true ∧ fields.target≠none ∧ fields.prediction≠none ∧
      fields.prediction≠fields.target
  | .tie => fields.completeCoverage=true ∧ fields.target≠none ∧ fields.prediction=none
  | .incomplete | .fitFailure => fields.nonemptyReason=true ∧ fields.prediction=none
  | .abstain => True

theorem correct_requires_complete_coverage_and_equal_present_prediction (fields : Fields)
    (accepted : statusFieldsValid fields) (correct : fields.status=.correct) :
    fields.completeCoverage=true ∧ fields.target≠none ∧ fields.prediction≠none ∧
      fields.prediction=fields.target := by
  simpa only [statusFieldsValid,correct] using accepted

theorem incorrect_requires_present_different_prediction (fields : Fields)
    (accepted : statusFieldsValid fields) (incorrect : fields.status=.incorrect) :
    fields.completeCoverage=true ∧ fields.target≠none ∧ fields.prediction≠none ∧
      fields.prediction≠fields.target := by
  simpa only [statusFieldsValid,incorrect] using accepted

theorem tie_requires_coverage_target_and_no_unique_prediction (fields : Fields)
    (accepted : statusFieldsValid fields) (tie : fields.status=.tie) :
    fields.completeCoverage=true ∧ fields.target≠none ∧ fields.prediction=none := by
  simpa only [statusFieldsValid,tie] using accepted

theorem failed_status_requires_reason_and_no_prediction (fields : Fields)
    (accepted : statusFieldsValid fields)
    (failed : fields.status=.incomplete ∨ fields.status=.fitFailure) :
    fields.nonemptyReason=true ∧ fields.prediction=none := by
  rcases failed with incomplete | fitFailed
  · simpa only [statusFieldsValid,incomplete] using accepted
  · simpa only [statusFieldsValid,fitFailed] using accepted

def matchingAbstention : Fields := ⟨.abstain,true,some 0,some 0,false⟩
def completeFailure : Fields := ⟨.incomplete,true,some 0,none,true⟩

theorem source_allows_matching_abstention_without_reclassifying_as_correct :
    statusFieldsValid matchingAbstention ∧ matchingAbstention.prediction=matchingAbstention.target ∧
      matchingAbstention.status≠.correct := by
  simp [statusFieldsValid,matchingAbstention]

theorem incomplete_status_does_not_require_incomplete_coverage :
    statusFieldsValid completeFailure ∧ completeFailure.completeCoverage=true := by
  simp [statusFieldsValid,completeFailure]

end
end N2P3.RequestedConstructorFields
