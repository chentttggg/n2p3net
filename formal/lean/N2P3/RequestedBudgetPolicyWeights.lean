import N2P3.RequestedAggregationWeights
import N2P3.RequestedNLLMacroWeights

/-! Source-before-proof budget_policy_weight_requirements01. Source collects
each person's actual positive-K rows, rejects the policy point if ANY K fails,
then averages that person's K rows before participant/study averaging.
We reuse the exact-real conditional-mean algebra: its completed Bool and nll
Option fields are admission/value interface slots here, not literal source
status strings or a claim that AUC deltas are NLL values. -/

namespace N2P3.RequestedBudgetPolicyWeights

open scoped BigOperators
open N2P3.RequestedNLLMacroWeights
open N2P3.RequestedAggregationWeights (SeedPair mean)

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

structure PolicyProgram (Study Participant : Type) where
  studies : Finset Study
  outerParticipants : Study → Finset Participant
  budgets : Study → Participant → Finset ℕ
  seedPairs : Finset SeedPair
  rows : Study → Participant → ℕ → SeedPair → Option ℝ

variable {Study Participant : Type}

def seedView (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) :
    N2P3.RequestedAggregationWeights.Program ℕ Unit where
  studies := program.budgets study participant
  participants := fun _ => {()}
  seedPairs := program.seedPairs
  rows := fun budget _ pair => program.rows study participant budget pair

def missingPairsAt (program : PolicyProgram Study Participant) (study : Study)
    (participant : Participant) (budget : ℕ) : Finset SeedPair :=
  N2P3.RequestedAggregationWeights.missingPairs (seedView program study participant) budget ()

def completeBudgets (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) : Finset ℕ :=
  (program.budgets study participant).filter (fun budget => missingPairsAt program study participant budget=∅)

def policyDomain (program : PolicyProgram Study Participant) (study : Study) : Finset Participant :=
  (program.outerParticipants study).filter (fun participant => program.budgets study participant≠∅)

def legalPolicy (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) : Prop :=
  completeBudgets program study participant=program.budgets study participant

def budgetMeans (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) : Program ℕ SeedPair where
  cells := program.budgets study participant
  decisions := fun _ => program.seedPairs
  completed := fun budget => decide (missingPairsAt program study participant budget=∅)
  nll := program.rows study participant

def primaryValue (program : PolicyProgram Study Participant) (study : Study)
    (participant : Participant) (budget : ℕ) (pair : SeedPair) : ℝ :=
  (program.rows study participant budget pair).getD 0

def policyMean (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) : ℝ :=
  linearMean (budgetMeans program study participant) (primaryValue program study participant)

def policyValue (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) : Option ℝ :=
  if legalPolicy program study participant then some (policyMean program study participant) else none

def studyMeans (program : PolicyProgram Study Participant) : Program Study Participant where
  cells := program.studies
  decisions := policyDomain program
  completed := fun _ => true
  nll := policyValue program

def fullStudyFlag (program : PolicyProgram Study Participant) : Prop :=
  (available (studyMeans program)).card=program.studies.card

def fullPolicyEstimate (program : PolicyProgram Study Participant) : Option ℝ :=
  if fullStudyFlag program then reported (studyMeans program) else none

def policyLinearMean (program : PolicyProgram Study Participant)
    (values : Study → Participant → ℕ → SeedPair → ℝ) : ℝ :=
  linearMean (studyMeans program) (fun study participant =>
    linearMean (budgetMeans program study participant) (values study participant))

def policyCoefficient (program : PolicyProgram Study Participant) (chosenStudy : Study)
    (chosenParticipant : Participant) (chosenBudget : ℕ) (chosenPair : SeedPair) : ℝ :=
  policyLinearMean program (fun study participant budget pair =>
    if study=chosenStudy then if participant=chosenParticipant then
      if budget=chosenBudget then if pair=chosenPair then 1 else 0 else 0 else 0 else 0)

theorem policy_requested_people_come_only_from_nonempty_actual_K_sets
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) :
    participant ∈ policyDomain program study ↔ participant ∈ program.outerParticipants study ∧
      program.budgets study participant≠∅ := by
  simp [policyDomain]

theorem whole_person_policy_gate_requires_every_actual_K_and_declared_pair
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) :
    legalPolicy program study participant ↔
      ∀ budget ∈ program.budgets study participant,∀ pair ∈ program.seedPairs,
        program.rows study participant budget pair≠none := by
  unfold legalPolicy completeBudgets
  rw [Finset.filter_eq_self]
  simp only [missingPairsAt,N2P3.RequestedAggregationWeights.missing_grid_empty_iff_every_declared_pair_available,
    seedView]

theorem completed_actual_K_uses_all_explicit_seed_pairs
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) (budget : ℕ)
    (complete : missingPairsAt program study participant budget=∅) :
    scored (budgetMeans program study participant) budget=program.seedPairs := by
  exact N2P3.RequestedAggregationWeights.completed_grid_observed_pairs_equal_the_explicit_declaration
    (seedView program study participant) budget () complete

theorem actual_K_mean_admission_is_exactly_the_complete_K_filter
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (hasPairs : program.seedPairs≠∅) :
    available (budgetMeans program study participant)=completeBudgets program study participant := by
  ext budget
  simp only [available,completeBudgets,Finset.mem_filter]
  change (budget ∈ program.budgets study participant ∧
      decide (missingPairsAt program study participant budget=∅)=true ∧
      scored (budgetMeans program study participant) budget≠∅) ↔
    (budget ∈ program.budgets study participant ∧ missingPairsAt program study participant budget=∅)
  simp only [decide_eq_true_eq]
  constructor
  · rintro ⟨declared,complete,_⟩
    exact ⟨declared,complete⟩
  · rintro ⟨declared,complete⟩
    exact ⟨declared,complete,by rw [completed_actual_K_uses_all_explicit_seed_pairs program study participant budget complete]; exact hasPairs⟩

theorem selected_policy_person_is_derived_from_actual_lookup
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) :
    participant ∈ scored (studyMeans program) study ↔
      participant ∈ program.outerParticipants study ∧ program.budgets study participant≠∅ ∧
        legalPolicy program study participant := by
  by_cases legal : legalPolicy program study participant <;>
    simp [scored,studyMeans,policyValue,policyDomain,legal]

theorem selected_policy_person_has_all_actual_K_means_available
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (hasPairs : program.seedPairs≠∅) (selected : participant ∈ scored (studyMeans program) study) :
    available (budgetMeans program study participant)=program.budgets study participant ∧
      available (budgetMeans program study participant)≠∅ := by
  have fields := (selected_policy_person_is_derived_from_actual_lookup program study participant).mp selected
  rw [actual_K_mean_admission_is_exactly_the_complete_K_filter program study participant hasPairs,fields.2.2]
  exact ⟨rfl,fields.2.1⟩

theorem one_failed_actual_K_excludes_every_policy_row_of_that_person
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (budget : ℕ) (declaredBudget : budget ∈ program.budgets study participant)
    (pair : SeedPair) (declaredPair : pair ∈ program.seedPairs)
    (missing : program.rows study participant budget pair=none) :
    policyValue program study participant=none ∧ participant ∉ scored (studyMeans program) study := by
  have illegal : ¬legalPolicy program study participant := by
    intro legal
    exact (whole_person_policy_gate_requires_every_actual_K_and_declared_pair program study participant).mp legal
      budget declaredBudget pair declaredPair missing
  exact ⟨by simp [policyValue,illegal],by simp [selected_policy_person_is_derived_from_actual_lookup,illegal]⟩

theorem full_policy_gate_requires_each_declared_study_have_policy_people
    (program : PolicyProgram Study Participant) :
    fullStudyFlag program ↔ ∀ study ∈ program.studies,scored (studyMeans program) study≠∅ := by
  have availableView : available (studyMeans program)=program.studies.filter
      (fun study => scored (studyMeans program) study≠∅) := by
    simp [available,studyMeans]
  unfold fullStudyFlag
  rw [availableView,Finset.card_filter_eq_iff]

theorem full_policy_gate_keeps_exactly_declared_studies
    (program : PolicyProgram Study Participant) (full : fullStudyFlag program) :
    available (studyMeans program)=program.studies := by
  unfold available
  apply Finset.filter_eq_self.mpr
  intro study declared
  exact ⟨rfl,(full_policy_gate_requires_each_declared_study_have_policy_people program).mp full study declared⟩

theorem missing_expected_study_has_no_full_policy_estimate
    (program : PolicyProgram Study Participant) (study : Study) (declared : study ∈ program.studies)
    (missing : scored (studyMeans program) study=∅) : fullPolicyEstimate program=none := by
  have noFull : ¬fullStudyFlag program := fun full =>
    (full_policy_gate_requires_each_declared_study_have_policy_people program).mp full study declared missing
  simp [fullPolicyEstimate,noFull]

theorem mean_operator_commutes_with_scalar_amplitude {Cell Decision : Type}
    (program : Program Cell Decision) (values : Cell → Decision → ℝ) (amplitude : ℝ) :
    linearMean program (fun cell decision => amplitude*values cell decision)=amplitude*linearMean program values := by
  unfold linearMean mean
  simp_rw [←Finset.mul_sum,mul_div_assoc]
  rw [←Finset.mul_sum,mul_div_assoc]

theorem mean_basis_response_scales_by_its_amplitude {Cell Decision : Type}
    (program : Program Cell Decision) (cell : Cell) (decision : Decision) (amplitude : ℝ) :
    linearMean program (fun c d => if c=cell then if d=decision then amplitude else 0 else 0)=
      amplitude*coefficient program cell decision := by
  unfold coefficient
  rw [←mean_operator_commutes_with_scalar_amplitude]
  congr 1
  funext c d
  split_ifs <;> simp_all

theorem four_layer_indicator_response_derives_the_product_of_two_mean_responses
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (budget : ℕ) (pair : SeedPair) :
    policyCoefficient program study participant budget pair=
      coefficient (studyMeans program) study participant*coefficient (budgetMeans program study participant) budget pair := by
  have reduction (s : Study) (p : Participant) :
      linearMean (budgetMeans program s p) (fun k r =>
        if s=study then if p=participant then if k=budget then if r=pair then 1 else 0 else 0 else 0 else 0)=
      if s=study then if p=participant then coefficient (budgetMeans program study participant) budget pair else 0 else 0 := by
    by_cases sameStudy : s=study
    · subst s
      by_cases sameParticipant : p=participant
      · subst p
        simp only [coefficient]
        congr 1
        funext k r
        split_ifs <;> simp_all
      · simp [sameParticipant,linearMean,mean]
    · simp [sameStudy,linearMean,mean]
  unfold policyCoefficient policyLinearMean
  simp_rw [reduction]
  rw [mean_basis_response_scales_by_its_amplitude]
  exact mul_comm _ _

theorem full_policy_row_coefficient_has_the_derived_actual_K_factor
    (program : PolicyProgram Study Participant) (full : fullStudyFlag program)
    (hasPairs : program.seedPairs≠∅) (study : Study) (participant : Participant) (budget : ℕ) (pair : SeedPair) :
    policyCoefficient program study participant budget pair=
      if study ∈ program.studies ∧ participant ∈ scored (studyMeans program) study ∧
          budget ∈ program.budgets study participant ∧ pair ∈ program.seedPairs then
        1/((program.studies.card : ℝ)*(scored (studyMeans program) study).card*
          (program.budgets study participant).card*program.seedPairs.card) else 0 := by
  have allStudies := full_policy_gate_keeps_exactly_declared_studies program full
  rw [four_layer_indicator_response_derives_the_product_of_two_mean_responses]
  by_cases studyDeclared : study ∈ program.studies
  · by_cases personSelected : participant ∈ scored (studyMeans program) study
    · have inner := selected_policy_person_has_all_actual_K_means_available program study participant hasPairs personSelected
      have legal := (selected_policy_person_is_derived_from_actual_lookup program study participant).mp personSelected |>.2.2
      rw [source_cell_and_row_denominators_can_vary_without_pooling,allStudies,
        if_pos ⟨studyDeclared,personSelected⟩,source_cell_and_row_denominators_can_vary_without_pooling,inner.1]
      by_cases budgetDeclared : budget ∈ program.budgets study participant
      · have complete : missingPairsAt program study participant budget=∅ :=
          (Finset.filter_eq_self.mp legal) budget budgetDeclared
        rw [completed_actual_K_uses_all_explicit_seed_pairs program study participant budget complete]
        by_cases pairDeclared : pair ∈ program.seedPairs
        · simp only [studyDeclared,personSelected,budgetDeclared,pairDeclared,and_self,if_true]
          simp only [div_eq_mul_inv,mul_inv_rev,one_mul]
          ring
        · simp [studyDeclared,personSelected,budgetDeclared,pairDeclared]
      · simp [studyDeclared,personSelected,budgetDeclared]
    · simp [source_cell_and_row_denominators_can_vary_without_pooling,allStudies,studyDeclared,personSelected]
  · simp [source_cell_and_row_denominators_can_vary_without_pooling,allStudies,studyDeclared]

theorem policy_row_coefficients_are_nonnegative
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant) (budget : ℕ) (pair : SeedPair) :
    0 ≤ policyCoefficient program study participant budget pair := by
  rw [four_layer_indicator_response_derives_the_product_of_two_mean_responses]
  exact mul_nonneg (computed_nll_coefficients_are_nonnegative _ _ _) (computed_nll_coefficients_are_nonnegative _ _ _)

theorem excluded_policy_person_has_zero_coefficients_at_every_K
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (excluded : participant ∉ scored (studyMeans program) study) (budget : ℕ) (pair : SeedPair) :
    policyCoefficient program study participant budget pair=0 := by
  rw [four_layer_indicator_response_derives_the_product_of_two_mean_responses,
    missing_or_unavailable_rows_have_zero_coefficient (studyMeans program) study participant (Or.inr excluded),zero_mul]

theorem unknown_actual_K_is_not_fabricated_into_policy_rows
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (unknown : program.budgets study participant=∅) (budget : ℕ) (pair : SeedPair) :
    participant ∉ policyDomain program study ∧ policyCoefficient program study participant budget pair=0 := by
  have notSelected : participant ∉ scored (studyMeans program) study := by
    intro selected
    exact (selected_policy_person_is_derived_from_actual_lookup program study participant).mp selected |>.2.1 unknown
  exact ⟨by simp [policyDomain,unknown],excluded_policy_person_has_zero_coefficients_at_every_K program study participant notSelected budget pair⟩

theorem each_person_policy_row_mass_is_the_derived_outer_coefficient
    (program : PolicyProgram Study Participant) (hasPairs : program.seedPairs≠∅) (study : Study) (participant : Participant) :
    (∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
      policyCoefficient program study participant budget pair)=coefficient (studyMeans program) study participant := by
  by_cases selected : participant ∈ scored (studyMeans program) study
  · have inner := selected_policy_person_has_all_actual_K_means_available program study participant hasPairs selected
    have innerMass := original_requested_nll_row_coefficient_mass_is_one (budgetMeans program study participant) inner.2
    change (∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
      coefficient (budgetMeans program study participant) budget pair)=1 at innerMass
    simp_rw [four_layer_indicator_response_derives_the_product_of_two_mean_responses]
    simp_rw [←Finset.mul_sum]
    rw [innerMass,mul_one]
  · simp [excluded_policy_person_has_zero_coefficients_at_every_K program study participant selected,
      missing_or_unavailable_rows_have_zero_coefficient (studyMeans program) study participant (Or.inr selected)]

theorem normalized_policy_unit_row_mass_is_derived_as_one
    (program : PolicyProgram Study Participant) (full : fullStudyFlag program)
    (hasStudies : program.studies≠∅) (hasPairs : program.seedPairs≠∅) :
    (∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
      ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
        policyCoefficient program study participant budget pair)=1 := by
  have nonempty : available (studyMeans program)≠∅ := by
    rw [full_policy_gate_keeps_exactly_declared_studies program full]
    exact hasStudies
  have outerMass := original_requested_nll_row_coefficient_mass_is_one (studyMeans program) nonempty
  change (∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
    coefficient (studyMeans program) study participant)=1 at outerMass
  simp_rw [each_person_policy_row_mass_is_the_derived_outer_coefficient program hasPairs]
  exact outerMass

theorem each_declared_study_policy_mass_is_one_over_the_declared_study_count
    (program : PolicyProgram Study Participant) (full : fullStudyFlag program)
    (hasPairs : program.seedPairs≠∅) (study : Study) (declared : study ∈ program.studies) :
    (∑ participant ∈ policyDomain program study,∑ budget ∈ program.budgets study participant,
      ∑ pair ∈ program.seedPairs,policyCoefficient program study participant budget pair)=1/(program.studies.card : ℝ) := by
  have selected : study ∈ available (studyMeans program) := by
    rw [full_policy_gate_keeps_exactly_declared_studies program full]
    exact declared
  have mass := every_retained_cell_gets_equal_total_mass (studyMeans program) study selected
  rw [full_policy_gate_keeps_exactly_declared_studies program full] at mass
  simp_rw [each_person_policy_row_mass_is_the_derived_outer_coefficient program hasPairs]
  calc
    (∑ participant ∈ policyDomain program study,coefficient (studyMeans program) study participant)=
      ∑ participant ∈ scored (studyMeans program) study,coefficient (studyMeans program) study participant := by
      symm
      apply Finset.sum_subset (Finset.filter_subset _ _)
      intro participant _ absent
      exact missing_or_unavailable_rows_have_zero_coefficient _ _ _ (Or.inr absent)
    _=1/(program.studies.card : ℝ) := mass

theorem conditional_mean_expansion_over_its_original_keys {Cell Decision : Type}
    (program : Program Cell Decision) (values : Cell → Decision → ℝ) :
    linearMean program values=∑ cell ∈ program.cells,∑ decision ∈ program.decisions cell,
      values cell decision*coefficient program cell decision := by
  rw [nested_macro_is_derived_weighted_sum_of_original_scored_rows,
    extending_to_original_requests_adds_only_zero_weight_terms]

theorem four_layer_policy_mean_has_the_original_row_expansion
    (program : PolicyProgram Study Participant) (values : Study → Participant → ℕ → SeedPair → ℝ) :
    policyLinearMean program values=
      ∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
        ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
          values study participant budget pair*policyCoefficient program study participant budget pair := by
  unfold policyLinearMean
  rw [conditional_mean_expansion_over_its_original_keys]
  change (∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
      linearMean (budgetMeans program study participant) (values study participant)*coefficient (studyMeans program) study participant)=_
  apply Finset.sum_congr rfl
  intro study _
  apply Finset.sum_congr rfl
  intro participant _
  rw [conditional_mean_expansion_over_its_original_keys]
  change ((∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
    values study participant budget pair*coefficient (budgetMeans program study participant) budget pair)*
      coefficient (studyMeans program) study participant)=_
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro budget _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro pair _
  rw [four_layer_indicator_response_derives_the_product_of_two_mean_responses]
  ring

theorem mean_operator_depends_only_on_its_actual_selected_support {Cell Decision : Type}
    (program : Program Cell Decision) (left right : Cell → Decision → ℝ)
    (same : ∀ cell ∈ available program,∀ decision ∈ scored program cell,left cell decision=right cell decision) :
    linearMean program left=linearMean program right := by
  unfold linearMean mean
  apply congrArg (fun numerator : ℝ => numerator/((available program).card : ℝ))
  apply Finset.sum_congr rfl
  intro cell cellSelected
  apply congrArg (fun numerator : ℝ => numerator/((scored program cell).card : ℝ))
  apply Finset.sum_congr rfl
  exact same cell cellSelected

theorem full_policy_return_value_is_the_source_four_layer_mean
    (program : PolicyProgram Study Participant) (full : fullStudyFlag program) (hasStudies : program.studies≠∅) :
    fullPolicyEstimate program=some (policyLinearMean program (primaryValue program)) := by
  have nonempty : available (studyMeans program)≠∅ := by
    rw [full_policy_gate_keeps_exactly_declared_studies program full]
    exact hasStudies
  have same : linearMean (studyMeans program) (value (studyMeans program))=
      policyLinearMean program (primaryValue program) := by
    apply mean_operator_depends_only_on_its_actual_selected_support
    intro study _ participant selected
    have legal := (selected_policy_person_is_derived_from_actual_lookup program study participant).mp selected |>.2.2
    simp [value,studyMeans,policyValue,legal,policyMean]
  simp only [fullPolicyEstimate,if_pos full,reported,if_neg nonempty]
  rw [same]

theorem returned_policy_point_has_the_computed_original_row_coefficients
    (program : PolicyProgram Study Participant) (full : fullStudyFlag program) (hasStudies : program.studies≠∅) :
    fullPolicyEstimate program=some
      (∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
        ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
          primaryValue program study participant budget pair*policyCoefficient program study participant budget pair) := by
  rw [full_policy_return_value_is_the_source_four_layer_mean program full hasStudies,
    four_layer_policy_mean_has_the_original_row_expansion]

def policyFailures (program : PolicyProgram Study Participant) (study : Study) : Finset Participant :=
  policyDomain program study \ scored (studyMeans program) study

theorem requested_policy_bound_count_keeps_failed_people_and_excludes_unknown_K
    (program : PolicyProgram Study Participant) (study : Study) :
    (scored (studyMeans program) study).card+(policyFailures program study).card=(policyDomain program study).card := by
  have subset : scored (studyMeans program) study ⊆ policyDomain program study := Finset.filter_subset _ _
  rw [policyFailures,Finset.card_sdiff_of_subset subset]
  have bounded := Finset.card_le_card subset
  omega

def completePeopleAtBudget (program : PolicyProgram Study Participant) (study : Study) (budget : ℕ) : Finset Participant :=
  (program.outerParticipants study).filter (fun participant =>
    budget ∈ program.budgets study participant ∧ missingPairsAt program study participant budget=∅)

def commonStandardCohort (program : PolicyProgram Study Participant) (study : Study) (standard : Finset ℕ) : Finset Participant :=
  (program.outerParticipants study).filter (fun participant =>
    ∀ budget ∈ standard,participant ∈ completePeopleAtBudget program study budget)

theorem standard_K_common_cohort_is_derived_from_actual_presence_and_full_seed_grids
    (program : PolicyProgram Study Participant) (study : Study) (standard : Finset ℕ) (participant : Participant) :
    participant ∈ commonStandardCohort program study standard ↔ participant ∈ program.outerParticipants study ∧
      ∀ budget ∈ standard,budget ∈ program.budgets study participant ∧
        ∀ pair ∈ program.seedPairs,program.rows study participant budget pair≠none := by
  simp only [commonStandardCohort,completePeopleAtBudget,Finset.mem_filter,missingPairsAt,
    N2P3.RequestedAggregationWeights.missing_grid_empty_iff_every_declared_pair_available,seedView]
  constructor
  · rintro ⟨person,complete⟩
    exact ⟨person,fun budget declared => (complete budget declared).2⟩
  · rintro ⟨person,complete⟩
    exact ⟨person,fun budget declared => ⟨person,complete budget declared⟩⟩

def policyArmMean {Arm : Type} (program : PolicyProgram Study Participant) (arms : Study → Finset Arm)
    (armCoefficient : Study → Arm → ℝ) (auc : Study → Participant → ℕ → SeedPair → Arm → ℝ) : ℝ :=
  policyLinearMean program (fun study participant budget pair =>
    ∑ arm ∈ arms study,armCoefficient study arm*auc study participant budget pair arm)

theorem original_policy_auc_cell_coefficient_includes_its_signed_arm_coefficient {Arm : Type}
    (program : PolicyProgram Study Participant) (arms : Study → Finset Arm) (armCoefficient : Study → Arm → ℝ)
    (auc : Study → Participant → ℕ → SeedPair → Arm → ℝ) :
    policyArmMean program arms armCoefficient auc=
      ∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
        ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
          ∑ arm ∈ arms study,auc study participant budget pair arm*
            (policyCoefficient program study participant budget pair*armCoefficient study arm) := by
  rw [policyArmMean,four_layer_policy_mean_has_the_original_row_expansion]
  apply Finset.sum_congr rfl
  intro study _
  apply Finset.sum_congr rfl
  intro participant _
  apply Finset.sum_congr rfl
  intro budget _
  apply Finset.sum_congr rfl
  intro pair _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro arm _
  ring

theorem policy_signed_AUC_cell_mass_retains_actual_coefficient_residual {Arm : Type}
    (program : PolicyProgram Study Participant) (arms : Study → Finset Arm) (armCoefficient : Study → Arm → ℝ)
    (full : fullStudyFlag program) (hasPairs : program.seedPairs≠∅) :
    (∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
      ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
        ∑ arm ∈ arms study,policyCoefficient program study participant budget pair*armCoefficient study arm)=
    mean program.studies (fun study => ∑ arm ∈ arms study,armCoefficient study arm) := by
  unfold mean
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro study declared
  calc
    (∑ participant ∈ policyDomain program study,
      ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
        ∑ arm ∈ arms study,policyCoefficient program study participant budget pair*armCoefficient study arm)=
        (∑ participant ∈ policyDomain program study,
          ∑ budget ∈ program.budgets study participant,∑ pair ∈ program.seedPairs,
            policyCoefficient program study participant budget pair)*(∑ arm ∈ arms study,armCoefficient study arm) := by
      simp_rw [←Finset.mul_sum,←Finset.sum_mul]
    _=(∑ arm ∈ arms study,armCoefficient study arm)/(program.studies.card : ℝ) := by
      rw [each_declared_study_policy_mass_is_one_over_the_declared_study_count program full hasPairs study declared]
      simp [div_eq_mul_inv,mul_comm]

def participantPolicyMetric (program : PolicyProgram Study Participant) (transform : ℝ → ℝ) : ℝ :=
  linearMean (studyMeans program) (fun study participant => transform (policyMean program study participant))

theorem participant_metric_transform_is_applied_after_both_inner_means
    (program : PolicyProgram Study Participant) (transform : ℝ → ℝ) :
    participantPolicyMetric program transform=
      ∑ study ∈ program.studies,∑ participant ∈ policyDomain program study,
        transform (policyMean program study participant)*coefficient (studyMeans program) study participant := by
  exact conditional_mean_expansion_over_its_original_keys _ _

theorem computed_conditional_mean_preserves_bounded_actual_rows {Cell Decision : Type}
    (program : Program Cell Decision) (values : Cell → Decision → ℝ)
    (defined : available program≠∅) (lower upper : ℝ)
    (bounded : ∀ cell ∈ available program,∀ decision ∈ scored program cell,
      lower ≤ values cell decision ∧ values cell decision ≤ upper) :
    lower ≤ linearMean program values ∧ linearMean program values ≤ upper := by
  have mass := conditional_macro_coefficient_mass_is_one program defined
  have constants (constant : ℝ) :
      (∑ cell ∈ available program,∑ decision ∈ scored program cell,constant*coefficient program cell decision)=constant := by
    simp_rw [←Finset.mul_sum]
    rw [mass,mul_one]
  have low := Finset.sum_le_sum (fun cell cellMember => Finset.sum_le_sum (fun decision rowMember =>
    mul_le_mul_of_nonneg_right (bounded cell cellMember decision rowMember).1
      (computed_nll_coefficients_are_nonnegative program cell decision)))
  have high := Finset.sum_le_sum (fun cell cellMember => Finset.sum_le_sum (fun decision rowMember =>
    mul_le_mul_of_nonneg_right (bounded cell cellMember decision rowMember).2
      (computed_nll_coefficients_are_nonnegative program cell decision)))
  rw [constants lower] at low
  rw [constants upper] at high
  rw [nested_macro_is_derived_weighted_sum_of_original_scored_rows]
  exact ⟨low,high⟩

theorem selected_person_policy_mean_preserves_its_study_contrast_bounds
    (program : PolicyProgram Study Participant) (study : Study) (participant : Participant)
    (hasPairs : program.seedPairs≠∅) (selected : participant ∈ scored (studyMeans program) study)
    (lower upper : ℝ) (bounded : ∀ budget ∈ program.budgets study participant,∀ pair ∈ program.seedPairs,
      lower ≤ primaryValue program study participant budget pair ∧ primaryValue program study participant budget pair ≤ upper) :
    lower ≤ policyMean program study participant ∧ policyMean program study participant ≤ upper := by
  have inner := selected_policy_person_has_all_actual_K_means_available program study participant hasPairs selected
  apply computed_conditional_mean_preserves_bounded_actual_rows _ _ inner.2 lower upper
  intro budget declared pair pairSelected
  have budgetMember : budget ∈ program.budgets study participant := by simpa only [inner.1] using declared
  have pairMember : pair ∈ program.seedPairs := (Finset.mem_filter.mp pairSelected).1
  exact bounded budget budgetMember pair pairMember

end
end N2P3.RequestedBudgetPolicyWeights
