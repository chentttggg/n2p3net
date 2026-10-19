import N2P3.RequestedSlice

/- Fixed wrong alternatives C01-C15 and source near-zero finding.
Each example must reduce to an unsolved False, not an elaboration accident. -/
namespace N2P3.NegativeRequestedMetrics

open N2P3.RequestedOutcomes N2P3.RequestedOutcomeRates
open N2P3.RequestedCoverage N2P3.RequestedConstructorFields
open N2P3.RequestedNLL N2P3.RequestedStudyStatistics

noncomputable section
attribute [local instance] Classical.propDecidable

-- C01: equal row counts do not make the requested/observed key sets equal.
example : (([0,2] : List ℕ).toFinset)=([0,1] : List ℕ).toFinset := by
  have actual : (([0,2] : List ℕ).toFinset)≠([0,1] : List ℕ).toFinset := by
    intro same
    have member : 2 ∈ ([0,1] : List ℕ).toFinset := same ▸ (by simp)
    simp at member
  simp only [actual]

-- C02: duplicate observations are not made legal by counting their rows.
example : (observedKeys ([⟨0,1,.correct⟩,⟨0,1,.correct⟩] : List (Outcome ℕ))).Nodup := by
  simp [observedKeys,Outcome.pair]

-- C03: legal individual statuses do not supply a missing requested row.
example : (([0] : List ℕ).toFinset)=([0,1] : List ℕ).toFinset := by
  have actual : (([0] : List ℕ).toFinset)≠([0,1] : List ℕ).toFinset := by
    intro same
    have member : 1 ∈ ([0] : List ℕ).toFinset := same.symm ▸ (by simp)
    simp at member
  simp only [actual]

-- C04: two evidence prefixes are not one per-level numerator.
example : ratio0 2 1 ≤ 1 := by
  norm_num [ratio0]

-- C05: tie belongs to the valid denominator.
example : ratio0 (Counts.correct ⟨1,0,1,0,0,0⟩) (Counts.valid ⟨1,0,1,0,0,0⟩)=1 := by
  norm_num [Counts.valid,ratio0]

-- C06: all six actual terminal states stay in the requested denominator.
example : ratio0 1 6=1/3 := by
  norm_num [ratio0]

-- C07: a complete empty coverage helper is not a nonempty task.
example : 0 < emptyCoverage.expectedKeys.card := by
  simp [emptyCoverage]

-- C09: pooled decisions and participant macro differ on the source fixture.
example : requestedMacro N2P3.RequestedParticipantPlan.fixtureRequested
    (participantRate N2P3.RequestedParticipantPlan.fixtureEntries
      N2P3.RequestedParticipantPlan.fixtureFailures N2P3.RequestedOutcomeRates.fixtureOutcomes 1)=some (1/10) := by
  rw [source_fixed_macro_pool_and_failed_participant_counterexample.1]
  norm_num

-- C10: excluding a requested selection failure changes the macro.
example : requestedMacro N2P3.RequestedParticipantPlan.fixtureRequested
    (participantRate N2P3.RequestedParticipantPlan.fixtureEntries
      N2P3.RequestedParticipantPlan.fixtureFailures N2P3.RequestedOutcomeRates.fixtureOutcomes 1)=some (1/2) := by
  rw [source_fixed_macro_pool_and_failed_participant_counterexample.1]
  norm_num

-- C11: analyzer None cannot be silently identified with numeric zero.
example : partialRatio 0 0=some 0 := by
  simp [partialRatio]

-- C13: source-fixed subject AUCs are each 1, but pooled AUC is 3/4.
-- C12: no constructor/accounting law forces correct predictions to persist.
example : ratio0 (countsAt ([⟨0,1,.correct⟩,⟨0,2,.incorrect⟩] : List (Outcome ℕ)) 1).correct 1 ≤
    ratio0 (countsAt ([⟨0,1,.correct⟩,⟨0,2,.incorrect⟩] : List (Outcome ℕ)) 2).correct 1 := by
  norm_num [countsAt,countAt,ratio0]

-- C13: source-fixed subject AUCs are each 1, but pooled AUC is 3/4.
example : N2P3.RequestedAUC.auc [1,101] [0,100]=some 1 := by
  rw [N2P3.RequestedAUC.source_fixed_macro_auc_differs_from_pooled_auc.2.2]
  norm_num

-- C14: uniform filling is a different quantity from actual scored NLL.
example : (-Real.log (9/10)+Real.log 9)/2 ≤ -Real.log (9/10) := by
  have actual := fixed_nine_choice_uniform_missing_nll_differs_from_model_nll.2.2
  simp only [not_le.mpr actual]

-- C15: matching prediction/target on abstain does not change status to correct.
example : matchingAbstention.status=.correct := by
  simp [matchingAbstention]

-- Full-study availability does not mean all requested participants are complete.
example : missingCount [some 1,none]=0 := by
  simp [missingCount]

-- Source allows the negative residual; exact symmetric mass bound is false.
example : -positiveMass negativeResidualWeights ≤ contrast negativeResidualWeights extremalAUCs := by
  have actual := approximate_zero_sum_does_not_entail_exact_symmetric_bound.2
  simp only [not_le.mpr actual]

end
end N2P3.NegativeRequestedMetrics
