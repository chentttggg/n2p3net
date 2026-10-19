import N2P3.RequestedAggregationWitnesses

namespace N2P3.NegativeRequestedAggregationWeights

open N2P3.RequestedAggregationWeights N2P3.RequestedAggregationWitnesses
open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

-- W01: observed-only seed mean cannot promote an incomplete grid.
example : participantEstimate program 0 1=some 1 := by
  rw [actual_complete_grid_filter_excludes_only_the_missing_participant.1]
  simp

-- W02: an explicit diagonal declaration has2 pairs, not a fabricated4 grid.
example : program.seedPairs.card=4 := by
  rw [explicit_diagonal_declaration_has_two_pairs_instead_of_four.1]
  norm_num

-- W03: equal declared-study means differ from pooling all complete rows.
example : rowCoefficient program 0 0 (0,0)=1/8 := by
  rw [unequal_study_support_derives_nonpooled_primary_row_weights.1]
  norm_num

-- W04: a missing expected study does not become full result over remaining.
example : fullEstimate missingStudyProgram=some 1 := by
  rw [missing_entire_declared_study_has_no_full_point_estimate]
  simp

-- W05: conditional and requested missing-bound denominators are distinct.
example : (program.participants 0).card=(completeParticipants program 0).card := by
  rw [incomplete_requested_person_remains_in_missing_bound_population.1,
    incomplete_requested_person_remains_in_missing_bound_population.2.1]
  norm_num

-- Signed source AUC-cell coefficients do not inherit unit-delta mass1.
example : (∑ study ∈ program.studies,∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
    ∑ arm ∈ ({0,1} : Finset ℕ),rowCoefficient program study participant pair*(if arm=0 then 1 else -1))=1 := by
  rw [signed_auc_cell_mass_of_exact_zero_contrast_is_zero]
  norm_num

end
end N2P3.NegativeRequestedAggregationWeights
