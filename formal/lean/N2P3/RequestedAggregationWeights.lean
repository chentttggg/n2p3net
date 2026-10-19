import N2P3.RequestedStudyStatistics
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise

/-! Source-fixed composition of stats.transfer._participant_rows and _summary.
Requirements aggregation_weights_requirements_01 were frozen before this file.
Keys are the actual unique declarations after parsing/identity checks; the
seed-pair set is explicit and is not replaced by a Cartesian product.
The primary lookup projects actual complete_pair rows to Some difference and
all other rows to None. Completed support and all effective weights are
computed below; neither a balanced support nor final weights are inputs.
This finite exact-real view does not refine native lookup/order/IEEE behavior. -/

namespace N2P3.RequestedAggregationWeights

open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

abbrev SeedPair := ℤ × ℤ

structure Program (Study Participant : Type) where
  studies : Finset Study
  participants : Study → Finset Participant
  seedPairs : Finset SeedPair
  rows : Study → Participant → SeedPair → Option ℝ

variable {Study Participant : Type}

def missingPairs (program : Program Study Participant) (study : Study) (participant : Participant) : Finset SeedPair :=
  program.seedPairs.filter (fun pair => program.rows study participant pair=none)

def observedPairs (program : Program Study Participant) (study : Study) (participant : Participant) : Finset SeedPair :=
  program.seedPairs.filter (fun pair => (program.rows study participant pair).isSome)

def completeParticipants (program : Program Study Participant) (study : Study) : Finset Participant :=
  (program.participants study).filter (fun participant => missingPairs program study participant=∅)

def estimableStudies (program : Program Study Participant) : Finset Study :=
  program.studies.filter (fun study => completeParticipants program study≠∅)

def fullStudyFlag (program : Program Study Participant) : Prop :=
  (estimableStudies program).card=program.studies.card

def mean {Index : Type} (keys : Finset Index) (value : Index → ℝ) : ℝ :=
  (∑ key ∈ keys,value key)/(keys.card : ℝ)

def rowValue (program : Program Study Participant) (study : Study) (participant : Participant) (pair : SeedPair) : ℝ :=
  (program.rows study participant pair).getD 0

def participantMean (program : Program Study Participant) (study : Study) (participant : Participant) : ℝ :=
  mean (observedPairs program study participant) (rowValue program study participant)

def participantEstimate (program : Program Study Participant) (study : Study) (participant : Participant) : Option ℝ :=
  if missingPairs program study participant=∅ then some (participantMean program study participant) else none

def studyMean (program : Program Study Participant) (study : Study) : ℝ :=
  mean (completeParticipants program study) (participantMean program study)

def partialEstimate (program : Program Study Participant) : Option ℝ :=
  if estimableStudies program=∅ then none else some (mean (estimableStudies program) (studyMean program))

def fullEstimate (program : Program Study Participant) : Option ℝ :=
  if fullStudyFlag program then partialEstimate program else none

/-- The nested arithmetic of the source, with the support computed from the
original lookup. Applying this operator to row indicators derives weights. -/
def linearMean (program : Program Study Participant) (values : Study → Participant → SeedPair → ℝ) : ℝ :=
  mean program.studies (fun study =>
    mean (completeParticipants program study) (fun participant => mean program.seedPairs (values study participant)))

theorem missing_grid_empty_iff_every_declared_pair_available
    (program : Program Study Participant) (study : Study) (participant : Participant) :
    missingPairs program study participant=∅ ↔
      ∀ pair ∈ program.seedPairs, program.rows study participant pair≠none := by
  constructor
  · intro empty pair declared absent
    have member : pair ∈ missingPairs program study participant := Finset.mem_filter.mpr ⟨declared,absent⟩
    simp [empty] at member
  · intro available
    apply Finset.eq_empty_iff_forall_notMem.mpr
    intro pair member
    have fields := Finset.mem_filter.mp member
    exact available pair fields.1 fields.2

theorem complete_participant_is_derived_from_actual_lookup
    (program : Program Study Participant) (study : Study) (participant : Participant) :
    participant ∈ completeParticipants program study ↔ participant ∈ program.participants study ∧
      ∀ pair ∈ program.seedPairs, program.rows study participant pair≠none := by
  simp only [completeParticipants,Finset.mem_filter,missing_grid_empty_iff_every_declared_pair_available]

theorem completed_grid_observed_pairs_equal_the_explicit_declaration
    (program : Program Study Participant) (study : Study) (participant : Participant)
    (completed : missingPairs program study participant=∅) :
    observedPairs program study participant=program.seedPairs := by
  apply Finset.filter_eq_self.mpr
  intro pair declared
  have available := (missing_grid_empty_iff_every_declared_pair_available program study participant).mp completed pair declared
  cases present : program.rows study participant pair <;> simp_all

theorem incomplete_grid_cannot_use_only_observed_seed_mean
    (program : Program Study Participant) (study : Study) (participant : Participant)
    (missing : missingPairs program study participant≠∅) :
    participantEstimate program study participant=none := by
  simp [participantEstimate,missing]

theorem complete_grid_seed_denominator_is_declared_pair_count
    (program : Program Study Participant) (study : Study) (participant : Participant)
    (completed : missingPairs program study participant=∅) :
    participantEstimate program study participant=some (mean program.seedPairs (rowValue program study participant)) := by
  simp [participantEstimate,completed,participantMean,
    completed_grid_observed_pairs_equal_the_explicit_declaration program study participant completed]

theorem full_study_flag_iff_every_declared_study_has_complete_participants
    (program : Program Study Participant) :
    fullStudyFlag program ↔ ∀ study ∈ program.studies, completeParticipants program study≠∅ := by
  unfold fullStudyFlag estimableStudies
  exact Finset.card_filter_eq_iff

theorem full_flag_keeps_exactly_the_declared_studies (program : Program Study Participant)
    (full : fullStudyFlag program) : estimableStudies program=program.studies := by
  exact Finset.filter_eq_self.mpr
    ((full_study_flag_iff_every_declared_study_has_complete_participants program).mp full)

theorem missing_expected_study_returns_none_instead_of_renormalizing
    (program : Program Study Participant) (study : Study) (declared : study ∈ program.studies)
    (missing : completeParticipants program study=∅) : fullEstimate program=none := by
  have incomplete : ¬fullStudyFlag program := fun full =>
    (full_study_flag_iff_every_declared_study_has_complete_participants program).mp full study declared missing
  simp [fullEstimate,incomplete]

theorem actual_study_mean_uses_full_declared_pairs_for_each_selected_participant
    (program : Program Study Participant) (study : Study) :
    studyMean program study=mean (completeParticipants program study)
      (fun participant => mean program.seedPairs (rowValue program study participant)) := by
  unfold studyMean mean
  congr 1
  apply Finset.sum_congr rfl
  intro participant selected
  have complete := (Finset.mem_filter.mp selected).2
  simp only [participantMean,mean,
    completed_grid_observed_pairs_equal_the_explicit_declaration program study participant complete]

theorem full_point_estimate_is_the_actual_nested_mean (program : Program Study Participant)
    (full : fullStudyFlag program) (declared : program.studies≠∅) :
    fullEstimate program=some (linearMean program (rowValue program)) := by
  have allStudies := full_flag_keeps_exactly_the_declared_studies program full
  simp only [fullEstimate,if_pos full,partialEstimate,allStudies,if_neg declared]
  congr 1
  unfold linearMean mean
  congr 1
  apply Finset.sum_congr rfl
  intro study _
  exact actual_study_mean_uses_full_declared_pairs_for_each_selected_participant program study

def rowIndicator (study chosenStudy : Study) (participant chosenParticipant : Participant)
    (pair chosenPair : SeedPair) : ℝ :=
  if study=chosenStudy then if participant=chosenParticipant then if pair=chosenPair then 1 else 0 else 0 else 0

def rowCoefficient (program : Program Study Participant) (chosenStudy : Study)
    (chosenParticipant : Participant) (chosenPair : SeedPair) : ℝ :=
  linearMean program (fun study participant pair =>
    rowIndicator study chosenStudy participant chosenParticipant pair chosenPair)

theorem effective_coefficient_is_derived_from_the_nested_operator
    (program : Program Study Participant) (study : Study) (participant : Participant) (pair : SeedPair) :
    rowCoefficient program study participant pair=
      if study ∈ program.studies ∧ participant ∈ completeParticipants program study ∧ pair ∈ program.seedPairs then
        ((1/(program.seedPairs.card : ℝ))/(completeParticipants program study).card)/program.studies.card else 0 := by
  have seedReduction (s : Study) (p : Participant) :
      mean program.seedPairs (fun r => rowIndicator s study p participant r pair)=
        if s=study then if p=participant then
          (if pair ∈ program.seedPairs then 1/(program.seedPairs.card : ℝ) else 0) else 0 else 0 := by
    by_cases sameStudy : s=study <;> by_cases sameParticipant : p=participant <;>
      simp [mean,rowIndicator,sameStudy,sameParticipant,ite_div]
  have participantReduction (s : Study) :
      mean (completeParticipants program s) (fun p =>
        mean program.seedPairs (fun r => rowIndicator s study p participant r pair))=
      if s=study then if participant ∈ completeParticipants program s then
        (if pair ∈ program.seedPairs then 1/(program.seedPairs.card : ℝ) else 0)/
          (completeParticipants program s).card else 0 else 0 := by
    simp_rw [seedReduction]
    by_cases sameStudy : s=study <;> simp [mean,sameStudy,ite_div]
  unfold rowCoefficient linearMean
  simp_rw [participantReduction]
  by_cases studyDeclared : study ∈ program.studies <;>
    by_cases participantSelected : participant ∈ completeParticipants program study <;>
    by_cases pairDeclared : pair ∈ program.seedPairs <;>
    simp [mean,studyDeclared,participantSelected,pairDeclared]

theorem effective_coefficient_has_the_derived_reciprocal_product
    (program : Program Study Participant) (study : Study) (participant : Participant) (pair : SeedPair) :
    rowCoefficient program study participant pair=
      if study ∈ program.studies ∧ participant ∈ completeParticipants program study ∧ pair ∈ program.seedPairs then
        1/((program.studies.card : ℝ)*(completeParticipants program study).card*program.seedPairs.card) else 0 := by
  rw [effective_coefficient_is_derived_from_the_nested_operator]
  split_ifs
  · simp only [div_eq_mul_inv,mul_inv_rev,one_mul]
    ring
  · rfl

theorem effective_coefficients_are_nonnegative
    (program : Program Study Participant) (study : Study) (participant : Participant) (pair : SeedPair) :
    0 ≤ rowCoefficient program study participant pair := by
  rw [effective_coefficient_has_the_derived_reciprocal_product]
  split_ifs
  · exact div_nonneg zero_le_one
      (mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) (Nat.cast_nonneg _))
  · exact le_refl _

theorem excluded_participant_has_zero_coefficient_for_every_row
    (program : Program Study Participant) (study : Study) (participant : Participant) (pair : SeedPair)
    (excluded : participant ∉ completeParticipants program study) :
    rowCoefficient program study participant pair=0 := by
  simp [effective_coefficient_is_derived_from_the_nested_operator,excluded]

theorem one_missing_declared_pair_zeroes_all_participant_coefficients
    (program : Program Study Participant) (study : Study) (participant : Participant)
    (absentPair : SeedPair) (declared : absentPair ∈ program.seedPairs)
    (missing : program.rows study participant absentPair=none) (anyPair : SeedPair) :
    rowCoefficient program study participant anyPair=0 := by
  have excluded : participant ∉ completeParticipants program study := by
    intro selected
    exact (complete_participant_is_derived_from_actual_lookup program study participant).mp selected |>.2 absentPair declared missing
  exact excluded_participant_has_zero_coefficient_for_every_row program study participant anyPair excluded

theorem selected_participant_row_mass_is_derived_from_declared_seed_count
    (program : Program Study Participant) (study : Study) (participant : Participant)
    (studyDeclared : study ∈ program.studies) (selected : participant ∈ completeParticipants program study)
    (hasPairs : program.seedPairs≠∅) :
    (∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=
      (1/(completeParticipants program study).card)/(program.studies.card : ℝ) := by
  have pairsNonzero : (program.seedPairs.card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr hasPairs))
  calc
    (∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=
        (∑ _pair ∈ program.seedPairs,((1/(program.seedPairs.card : ℝ))/(completeParticipants program study).card)/
          program.studies.card) := by
      apply Finset.sum_congr rfl
      intro pair declared
      simp [effective_coefficient_is_derived_from_the_nested_operator,studyDeclared,selected,declared]
    _=(1/(completeParticipants program study).card)/(program.studies.card : ℝ) := by
      simp only [Finset.sum_const,nsmul_eq_mul]
      rw [←mul_div_assoc,←mul_div_assoc]
      simp [pairsNonzero]

theorem each_declared_study_has_one_over_declared_study_count_mass
    (program : Program Study Participant) (full : fullStudyFlag program)
    (hasPairs : program.seedPairs≠∅) (study : Study) (declared : study ∈ program.studies) :
    (∑ participant ∈ completeParticipants program study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=1/(program.studies.card : ℝ) := by
  have nonempty := (full_study_flag_iff_every_declared_study_has_complete_participants program).mp full study declared
  have participantsNonzero : ((completeParticipants program study).card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr nonempty))
  calc
    (∑ participant ∈ completeParticipants program study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=
        ∑ _participant ∈ completeParticipants program study,
          (1/(completeParticipants program study).card)/(program.studies.card : ℝ) := by
      apply Finset.sum_congr rfl
      intro participant selected
      exact selected_participant_row_mass_is_derived_from_declared_seed_count program study participant declared selected hasPairs
    _=1/(program.studies.card : ℝ) := by
      simp only [Finset.sum_const,nsmul_eq_mul]
      rw [←mul_div_assoc]
      simp [participantsNonzero]

theorem computed_coefficient_mass_over_surviving_rows_is_one
    (program : Program Study Participant) (full : fullStudyFlag program)
    (hasStudies : program.studies≠∅) (hasPairs : program.seedPairs≠∅) :
    (∑ study ∈ program.studies,∑ participant ∈ completeParticipants program study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=1 := by
  have studiesNonzero : (program.studies.card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr (Finset.nonempty_iff_ne_empty.mpr hasStudies))
  calc
    (∑ study ∈ program.studies,∑ participant ∈ completeParticipants program study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=
        ∑ _study ∈ program.studies,1/(program.studies.card : ℝ) := by
      apply Finset.sum_congr rfl
      intro study declared
      exact each_declared_study_has_one_over_declared_study_count_mass program full hasPairs study declared
    _=1 := by simp [studiesNonzero]

theorem sum_over_requested_adds_only_zero_coefficients_for_excluded_participants
    (program : Program Study Participant) (study : Study) (values : Participant → SeedPair → ℝ) :
    (∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
      values participant pair*rowCoefficient program study participant pair)=
    ∑ participant ∈ completeParticipants program study,∑ pair ∈ program.seedPairs,
      values participant pair*rowCoefficient program study participant pair := by
  symm
  apply Finset.sum_subset (Finset.filter_subset _ _)
  intro participant _ excluded
  apply Finset.sum_eq_zero
  intro pair _
  rw [excluded_participant_has_zero_coefficient_for_every_row program study participant pair excluded,mul_zero]

theorem computed_coefficient_mass_over_all_requested_rows_is_one
    (program : Program Study Participant) (full : fullStudyFlag program)
    (hasStudies : program.studies≠∅) (hasPairs : program.seedPairs≠∅) :
    (∑ study ∈ program.studies,∑ participant ∈ program.participants study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=1 := by
  have same : (∑ study ∈ program.studies,∑ participant ∈ program.participants study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=
    ∑ study ∈ program.studies,∑ participant ∈ completeParticipants program study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair := by
    apply Finset.sum_congr rfl
    intro study _
    simpa only [one_mul] using
      sum_over_requested_adds_only_zero_coefficients_for_excluded_participants program study (fun _ _ => 1)
  rw [same]
  exact computed_coefficient_mass_over_surviving_rows_is_one program full hasStudies hasPairs

theorem nested_mean_equals_original_rows_times_their_derived_coefficients
    (program : Program Study Participant) (values : Study → Participant → SeedPair → ℝ) :
    linearMean program values=
      ∑ study ∈ program.studies,∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
        values study participant pair*rowCoefficient program study participant pair := by
  have expansion : linearMean program values=
      ∑ study ∈ program.studies,∑ participant ∈ completeParticipants program study,∑ pair ∈ program.seedPairs,
        values study participant pair*rowCoefficient program study participant pair := by
    unfold linearMean mean
    simp_rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro study declared
    apply Finset.sum_congr rfl
    intro participant selected
    apply Finset.sum_congr rfl
    intro pair pairDeclared
    rw [effective_coefficient_is_derived_from_the_nested_operator]
    simp [declared,selected,pairDeclared,div_eq_mul_inv,mul_assoc]
  rw [expansion]
  apply Finset.sum_congr rfl
  intro study _
  exact (sum_over_requested_adds_only_zero_coefficients_for_excluded_participants program study (values study)).symm

theorem returned_full_point_has_the_derived_effective_row_expansion
    (program : Program Study Participant) (full : fullStudyFlag program) (hasStudies : program.studies≠∅) :
    fullEstimate program=some
      (∑ study ∈ program.studies,∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
        rowValue program study participant pair*rowCoefficient program study participant pair) := by
  rw [full_point_estimate_is_the_actual_nested_mean program full hasStudies,
    nested_mean_equals_original_rows_times_their_derived_coefficients]

def missingParticipants (program : Program Study Participant) (study : Study) : Finset Participant :=
  program.participants study \ completeParticipants program study

theorem conditional_and_missing_bound_populations_come_from_the_same_request_filter
    (program : Program Study Participant) (study : Study) :
    (completeParticipants program study).card+(missingParticipants program study).card=
      (program.participants study).card := by
  have subset : completeParticipants program study ⊆ program.participants study := Finset.filter_subset _ _
  rw [missingParticipants,Finset.card_sdiff_of_subset subset]
  have bounded := Finset.card_le_card subset
  omega

theorem declared_study_mass_over_the_whole_requested_population
    (program : Program Study Participant) (full : fullStudyFlag program)
    (hasPairs : program.seedPairs≠∅) (study : Study) (declared : study ∈ program.studies) :
    (∑ participant ∈ program.participants study,
      ∑ pair ∈ program.seedPairs,rowCoefficient program study participant pair)=1/(program.studies.card : ℝ) := by
  have same := sum_over_requested_adds_only_zero_coefficients_for_excluded_participants program study (fun _ _ => 1)
  simp only [one_mul] at same
  rw [same]
  exact each_declared_study_has_one_over_declared_study_count_mass program full hasPairs study declared

def aggregateDeclaredAUCContrast {Arm : Type} (program : Program Study Participant)
    (arms : Study → Finset Arm) (coefficient : Study → Arm → ℝ)
    (auc : Study → Participant → SeedPair → Arm → ℝ) : ℝ :=
  linearMean program (fun study participant pair =>
    ∑ arm ∈ arms study,coefficient study arm*auc study participant pair arm)

theorem actual_auc_cell_coefficient_is_unit_weight_times_declared_arm_coefficient {Arm : Type}
    (program : Program Study Participant) (arms : Study → Finset Arm) (coefficient : Study → Arm → ℝ)
    (auc : Study → Participant → SeedPair → Arm → ℝ) :
    aggregateDeclaredAUCContrast program arms coefficient auc=
      ∑ study ∈ program.studies,∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
        ∑ arm ∈ arms study,auc study participant pair arm*
          (rowCoefficient program study participant pair*coefficient study arm) := by
  rw [aggregateDeclaredAUCContrast,nested_mean_equals_original_rows_times_their_derived_coefficients]
  apply Finset.sum_congr rfl
  intro study _
  apply Finset.sum_congr rfl
  intro participant _
  apply Finset.sum_congr rfl
  intro pair _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro arm _
  ring

theorem signed_auc_cell_mass_preserves_the_declared_coefficient_residual {Arm : Type}
    (program : Program Study Participant) (arms : Study → Finset Arm) (coefficient : Study → Arm → ℝ)
    (full : fullStudyFlag program) (hasPairs : program.seedPairs≠∅) :
    (∑ study ∈ program.studies,∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
      ∑ arm ∈ arms study,rowCoefficient program study participant pair*coefficient study arm)=
      mean program.studies (fun study => ∑ arm ∈ arms study,coefficient study arm) := by
  unfold mean
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro study declared
  calc
    (∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
      ∑ arm ∈ arms study,rowCoefficient program study participant pair*coefficient study arm)=
        (∑ participant ∈ program.participants study,∑ pair ∈ program.seedPairs,
          rowCoefficient program study participant pair)*(∑ arm ∈ arms study,coefficient study arm) := by
      simp_rw [←Finset.mul_sum,←Finset.sum_mul]
    _=(∑ arm ∈ arms study,coefficient study arm)/(program.studies.card : ℝ) := by
      rw [declared_study_mass_over_the_whole_requested_population program full hasPairs study declared]
      simp [div_eq_mul_inv,mul_comm]

end
end N2P3.RequestedAggregationWeights
