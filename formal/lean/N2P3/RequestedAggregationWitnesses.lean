import N2P3.RequestedAggregationWeights

/-! Fixed W01-W05 nonempty witnesses from the source-before-proof contract,
plus the distinct signed source-cell mass from arm_mapping_requirements01.
They witness the general results; they do not replace the general proofs. -/

namespace N2P3.RequestedAggregationWitnesses

open N2P3.RequestedAggregationWeights
open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def diagonalPairs : Finset SeedPair := {(0,0),(1,1)}

def program : Program ℕ ℕ where
  studies := {0,1}
  participants := fun study => if study=0 then {0,1} else {0,1,2}
  seedPairs := diagonalPairs
  rows := fun study participant pair =>
    if study=0 ∧ participant=1 then if pair=(0,0) then some 1 else none
    else some (if study=0 then 1 else 0)

theorem explicit_diagonal_declaration_has_two_pairs_instead_of_four :
    program.seedPairs.card=2 ∧ (({0,1} : Finset ℤ).product {0,1}).card=4 := by
  norm_num [program,diagonalPairs]

theorem actual_complete_grid_filter_excludes_only_the_missing_participant :
    participantEstimate program 0 1=none ∧ completeParticipants program 0={0} ∧
      completeParticipants program 1={0,1,2} ∧ fullStudyFlag program := by
  simp [participantEstimate,completeParticipants,missingPairs,program,diagonalPairs,
    fullStudyFlag,estimableStudies,Finset.filter_insert,Finset.filter_singleton]

theorem unequal_study_support_derives_nonpooled_primary_row_weights :
    rowCoefficient program 0 0 (0,0)=1/4 ∧ rowCoefficient program 1 0 (0,0)=1/12 ∧
      rowCoefficient program 0 1 (0,0)=0 := by
  simp [effective_coefficient_is_derived_from_the_nested_operator,
    completeParticipants,missingPairs,program,diagonalPairs,Finset.filter_insert,Finset.filter_singleton]
  norm_num

theorem incomplete_requested_person_remains_in_missing_bound_population :
    (program.participants 0).card=2 ∧ (completeParticipants program 0).card=1 ∧
      (missingParticipants program 0).card=1 := by
  have requestCount : (program.participants 0).card=2 := by simp [program]
  have completeCount : (completeParticipants program 0).card=1 := by
    rw [actual_complete_grid_filter_excludes_only_the_missing_participant.2.1]
    simp
  have partition := conditional_and_missing_bound_populations_come_from_the_same_request_filter program 0
  exact ⟨requestCount,completeCount,by omega⟩

def missingStudyProgram : Program ℕ ℕ :=
  {program with rows := fun study participant pair =>
    if study=1 then none else program.rows study participant pair}

theorem missing_entire_declared_study_has_no_full_point_estimate :
    fullEstimate missingStudyProgram=none := by
  apply missing_expected_study_returns_none_instead_of_renormalizing missingStudyProgram 1
  · simp [missingStudyProgram,program]
  · norm_num [completeParticipants,missingPairs,missingStudyProgram,program,diagonalPairs,Finset.filter_insert]

theorem signed_auc_cell_mass_of_exact_zero_contrast_is_zero :
    (∑ study ∈ program.studies,∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
      ∑ arm ∈ ({0,1} : Finset ℕ),rowCoefficient program study participant pair*(if arm=0 then 1 else -1))=0 := by
  rw [signed_auc_cell_mass_preserves_the_declared_coefficient_residual program
    (fun _ => ({0,1} : Finset ℕ)) (fun _ arm => if arm=0 then 1 else -1)
    actual_complete_grid_filter_excludes_only_the_missing_participant.2.2.2 (by simp [program,diagonalPairs])]
  norm_num [mean,program]

end
end N2P3.RequestedAggregationWitnesses
