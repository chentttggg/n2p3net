import N2P3.RequestedBudgetPolicyWeights

/-! Fixed K01-K05 witnesses from the immutable source-before-proof policy
contract. Actual K counts differ by person; a whole policy person fails when
one of their actual K seed grids fails; unknown K is not invented. -/

namespace N2P3.RequestedBudgetPolicyWitnesses

open N2P3.RequestedBudgetPolicyWeights N2P3.RequestedNLLMacroWeights
open N2P3.RequestedAggregationWeights (SeedPair)

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def diagonalPairs : Finset SeedPair := {(0,0),(1,1)}

def program : PolicyProgram ℕ ℕ where
  studies := {0,1}
  outerParticipants := fun study => if study=0 then {0,1,2} else {0,1}
  budgets := fun study participant =>
    if study=0 then if participant=0 then {200} else if participant=1 then {200,800} else ∅
    else if participant=0 then {200,800} else if participant=1 then {200,800,2000} else ∅
  seedPairs := diagonalPairs
  rows := fun study participant budget pair =>
    if study=0 ∧ participant=1 ∧ budget=800 ∧ pair=(1,1) then none
    else some (if study=0 ∧ participant=0 then 1 else 0)

theorem actual_policy_domains_exclude_unknown_K_but_keep_known_failed_people :
    policyDomain program 0={0,1} ∧ policyDomain program 1={0,1} ∧
      (program.outerParticipants 0).card=3 ∧ (policyDomain program 0).card=2 := by
  simp [policyDomain,program,Finset.filter_insert,Finset.filter_singleton]

theorem actual_lookup_filters_out_the_person_with_one_failed_K :
    scored (studyMeans program) 0={0} ∧ scored (studyMeans program) 1={0,1} := by
  constructor
  · ext participant
    simp only [selected_policy_person_is_derived_from_actual_lookup,
      whole_person_policy_gate_requires_every_actual_K_and_declared_pair]
    by_cases zero : participant=0 <;> by_cases one : participant=1 <;>
      simp [program,diagonalPairs,zero,one]
  · ext participant
    simp only [selected_policy_person_is_derived_from_actual_lookup,
      whole_person_policy_gate_requires_every_actual_K_and_declared_pair]
    by_cases zero : participant=0 <;> by_cases one : participant=1 <;>
      simp [program,diagonalPairs,zero,one]

theorem source_fixed_policy_has_complete_declared_study_support : fullStudyFlag program := by
  apply (full_policy_gate_requires_each_declared_study_have_policy_people program).mpr
  intro study declared
  have choices : study=0 ∨ study=1 := by simpa [program] using declared
  rcases choices with zero | one
  · subst study
    rw [actual_lookup_filters_out_the_person_with_one_failed_K.1]
    simp
  · subst study
    rw [actual_lookup_filters_out_the_person_with_one_failed_K.2]
    simp

theorem actual_variable_K_counts_derive_different_primary_row_coefficients :
    policyCoefficient program 0 0 200 (0,0)=1/4 ∧
      policyCoefficient program 1 0 200 (0,0)=1/16 ∧
      policyCoefficient program 1 1 200 (0,0)=1/24 ∧
      policyCoefficient program 0 1 200 (0,0)=0 := by
  have hasPairs : program.seedPairs≠∅ := by simp [program,diagonalPairs]
  constructor
  · rw [full_policy_row_coefficient_has_the_derived_actual_K_factor program
      source_fixed_policy_has_complete_declared_study_support hasPairs,
      actual_lookup_filters_out_the_person_with_one_failed_K.1]
    norm_num [program,diagonalPairs]
  constructor
  · rw [full_policy_row_coefficient_has_the_derived_actual_K_factor program
      source_fixed_policy_has_complete_declared_study_support hasPairs,
      actual_lookup_filters_out_the_person_with_one_failed_K.2]
    norm_num [program,diagonalPairs]
  constructor
  · rw [full_policy_row_coefficient_has_the_derived_actual_K_factor program
      source_fixed_policy_has_complete_declared_study_support hasPairs,
      actual_lookup_filters_out_the_person_with_one_failed_K.2]
    norm_num [program,diagonalPairs]
  · apply excluded_policy_person_has_zero_coefficients_at_every_K
    rw [actual_lookup_filters_out_the_person_with_one_failed_K.1]
    simp

def missingStudyProgram : PolicyProgram ℕ ℕ :=
  {program with rows := fun study participant budget pair =>
    if study=1 then none else program.rows study participant budget pair}

theorem one_missing_study_closes_the_full_policy_point : fullPolicyEstimate missingStudyProgram=none := by
  apply missing_expected_study_has_no_full_policy_estimate missingStudyProgram 1
  · simp [missingStudyProgram,program]
  · ext participant
    simp only [selected_policy_person_is_derived_from_actual_lookup,
      whole_person_policy_gate_requires_every_actual_K_and_declared_pair]
    by_cases zero : participant=0 <;> by_cases one : participant=1 <;>
      simp [missingStudyProgram,program,diagonalPairs,zero,one]
    all_goals exact ⟨0,0,by simp⟩

end
end N2P3.RequestedBudgetPolicyWitnesses
