import N2P3.CandidateCoverage
import N2P3.CandidateTie
import N2P3.CandidateProbability
import N2P3Pilot
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Fintype.EquivFin

/-! SelectionERP's actual LLR membership sum after its family returns finite
binary logits. No Gaussian density interpretation is substituted for its
declared overlapping-epoch independence approximation. The stored prior scalar
is used literally; its constructor's tolerance check is not strengthened to
exact log(count1/count0). Physical twelve-flash completeness and thirty-six
positive candidate columns are distinct accepted upstream definitions. -/
namespace N2P3.SelectionERPEvidence
open scoped BigOperators
open N2P3.CandidateProbability N2P3.CandidateTie
noncomputable section
variable {E H : Type*} [Fintype E]

def evidence (membership : E → H → ℤ) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (candidate : H) : ℝ :=
  N2P3Pilot.membershipScore (fun e c => (membership e c : ℝ)) (fun _ => true)
    binaryLogit (fun _ => priorLogOdds) candidate

theorem literal_llr_subtraction_then_membership_transpose_sum
    (membership : E → H → ℤ) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (candidate : H) :
    evidence membership binaryLogit priorLogOdds candidate=
      ∑ event,(membership event candidate:ℝ)*(binaryLogit event-priorLogOdds) := by
  simp [evidence,N2P3Pilot.membershipScore]

theorem prior_subtraction_is_multiplied_by_the_actual_candidate_membership_count
    (membership : E → H → ℤ) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (candidate : H) :
    evidence membership binaryLogit priorLogOdds candidate=
      evidence membership binaryLogit 0 candidate-
        priorLogOdds*(CandidateCoverage.columnCount membership candidate:ℝ) := by
  simpa only [evidence,if_true,mul_one,CandidateCoverage.columnCount,Int.cast_sum] using
    N2P3Pilot.scalar_offset_formula (fun e c => (membership e c : ℝ)) (fun _ => true)
      binaryLogit priorLogOdds candidate

theorem actual_equal_counts_preserve_all_pairwise_llr_differences
    (membership : E → H → ℤ) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (a b : H)
    (counts : CandidateCoverage.columnCount membership a=CandidateCoverage.columnCount membership b) :
    evidence membership binaryLogit priorLogOdds a-evidence membership binaryLogit priorLogOdds b=
      evidence membership binaryLogit 0 a-evidence membership binaryLogit 0 b := by
  rw [prior_subtraction_is_multiplied_by_the_actual_candidate_membership_count membership binaryLogit priorLogOdds a,
    prior_subtraction_is_multiplied_by_the_actual_candidate_membership_count membership binaryLogit priorLogOdds b,counts]
  ring

theorem identical_public_membership_columns_give_identical_actual_evidence
    (membership : E → H → ℤ) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (a b : H)
    (same : ∀ event,membership event a=membership event b) :
    evidence membership binaryLogit priorLogOdds a=evidence membership binaryLogit priorLogOdds b := by
  simp only [literal_llr_subtraction_then_membership_transpose_sum,same]

section Readout
variable [Fintype H] [Nonempty H] [LinearOrder H]

structure Readout where
  scores : H → ℝ
  probabilities : H → ℝ
  winnerIndex : Option H

def readout (scores : H → ℝ) : Readout (H := H) :=
  ⟨scores,probability scores,winner scores sourceTolerance⟩

theorem finite_llr_readout_normalizes_every_frozen_candidate (scores : H → ℝ) :
    (∀ c,0<(readout scores).probabilities c) ∧ ∑ c,(readout scores).probabilities c=1 :=
  ⟨probability_positive scores,probability_sum_one scores⟩

theorem candidate_common_shift_preserves_the_actual_absolute_tie_readout
    (scores : H → ℝ) (shift : ℝ) :
    (readout (fun c => scores c+shift)).probabilities=(readout scores).probabilities ∧
      (readout (fun c => scores c+shift)).winnerIndex=(readout scores).winnerIndex := by
  constructor
  · funext c
    exact probability_common_constant scores shift c
  · exact winner_add_constant scores sourceTolerance shift

theorem identical_maximal_membership_columns_force_actual_abstention
    (membership : E → H → ℤ) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (a b : H)
    (different : a ≠ b) (same : ∀ event,membership event a=membership event b)
    (maximal : IsMaximum (evidence membership binaryLogit priorLogOdds) a) :
    (readout (evidence membership binaryLogit priorLogOdds)).winnerIndex=none := by
  change winner _ sourceTolerance=none
  rw [winner_none_iff]
  apply (tie_independent_of_argmax _ _ a sourceTolerance (firstArgmax_isMaximum _) maximal).mpr
  apply (tied_iff_close_competitor _ a sourceTolerance_positive.le maximal).mpr
  refine ⟨b,different.symm,?_⟩
  rw [identical_public_membership_columns_give_identical_actual_evidence membership binaryLogit priorLogOdds a b same]
  simpa only [sub_self] using sourceTolerance_positive.le

theorem source_absolute_margin_is_required_even_for_unique_exact_argmax
    (scores : H → ℝ) (candidate : H) :
    (readout scores).winnerIndex=some candidate ↔
      IsMaximum scores candidate ∧ ∀ other,other ≠ candidate → sourceTolerance<scores candidate-scores other :=
  winner_some_iff_margin scores candidate sourceTolerance_positive.le

end Readout

section PhysicalSelection
open N2P3.CandidateCoverage

def physicalEvidence (schedule : E → Flash) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) (candidate : Fin 36) : ℝ :=
  evidence (physicalMatrix schedule) binaryLogit priorLogOdds (finProdFinEquiv.symm candidate)

def indexedPhysicalMatrix (schedule : E → Flash) (event : E) (candidate : Fin 36) : ℤ :=
  physicalMatrix schedule event (finProdFinEquiv.symm candidate)

theorem indexed_physical_transpose_sum_is_the_actual_ordered36_evidence
    (schedule : E → Flash) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) :
    evidence (indexedPhysicalMatrix schedule) binaryLogit priorLogOdds=physicalEvidence schedule binaryLogit priorLogOdds := rfl

theorem indexed_complete_physical_matrix_passes_the_actual36_position_gate
    (schedule : E → Flash) (complete : PhysicalComplete schedule) :
    membershipReason (Fintype.card E) 36 (indexedPhysicalMatrix schedule)=none := by
  rw [membership_success_iff]
  refine ⟨rfl,by decide,?_,?_,?_⟩
  · intro e c
    exact physical_binary schedule e (finProdFinEquiv.symm c)
  · intro e
    have same : rowCount (indexedPhysicalMatrix schedule) e=rowCount (physicalMatrix schedule) e := by
      unfold rowCount indexedPhysicalMatrix
      simpa only [Equiv.symm_apply_apply] using
        (Equiv.sum_comp (finProdFinEquiv : Cell ≃ Fin 36)
          (fun c => physicalMatrix schedule e (finProdFinEquiv.symm c))).symm
    rw [same,physical_each_event_has_six]
    norm_num
  · intro c
    change 0<columnCount (physicalMatrix schedule) (finProdFinEquiv.symm c)
    rw [physical_complete_gives_twenty_per_cell schedule complete]
    norm_num

theorem row_major_choice_index_is_the_original_frozen_word_position (cell : Cell) :
    (finProdFinEquiv cell).val=cell.2.val+6*cell.1.val := rfl

theorem physical_complete_llr_prior_is_one_common_twenty_event_offset
    (schedule : E → Flash) (complete : PhysicalComplete schedule) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) :
    physicalEvidence schedule binaryLogit priorLogOdds=
      fun c => physicalEvidence schedule binaryLogit 0 c-priorLogOdds*20 := by
  funext c
  unfold physicalEvidence
  rw [prior_subtraction_is_multiplied_by_the_actual_candidate_membership_count,
    physical_complete_gives_twenty_per_cell schedule complete]
  norm_num

theorem actual_complete_selection_prior_removal_preserves_all36_probabilities_and_absolute_winner
    (schedule : E → Flash) (complete : PhysicalComplete schedule) (binaryLogit : E → ℝ) (priorLogOdds : ℝ) :
    (readout (physicalEvidence schedule binaryLogit priorLogOdds)).probabilities=
      (readout (physicalEvidence schedule binaryLogit 0)).probabilities ∧
    (readout (physicalEvidence schedule binaryLogit priorLogOdds)).winnerIndex=
      (readout (physicalEvidence schedule binaryLogit 0)).winnerIndex := by
  rw [physical_complete_llr_prior_is_one_common_twenty_event_offset schedule complete]
  simpa only [sub_eq_add_neg] using candidate_common_shift_preserves_the_actual_absolute_tie_readout
    (physicalEvidence schedule binaryLogit 0) (-(priorLogOdds*20))

theorem actual_complete_selection_prior_removal_preserves_source_nll
    (schedule : E → Flash) (complete : PhysicalComplete schedule) (binaryLogit : E → ℝ)
    (priorLogOdds : ℝ) (truth : Fin 36) :
    nll (physicalEvidence schedule binaryLogit priorLogOdds) truth=
      nll (physicalEvidence schedule binaryLogit 0) truth := by
  rw [physical_complete_llr_prior_is_one_common_twenty_event_offset schedule complete]
  simpa only [sub_eq_add_neg] using nll_common_constant (physicalEvidence schedule binaryLogit 0)
    (-(priorLogOdds*20)) truth

theorem actual_complete_physical_selection_passes_the_source_membership_gate
    (schedule : E → Flash) (complete : PhysicalComplete schedule) :
    membershipReason (Fintype.card E) 36 (physicalMatrix schedule)=none :=
  physical_complete_implies_scorer_membership schedule complete

abbrev CompleteEvents := Fin 10 × Flash
def completeSchedule (event : CompleteEvents) : Flash := event.2

theorem constructed120_event_schedule_is_physically_complete : PhysicalComplete completeSchedule := by
  rw [physical_complete_iff_each_flash_ten]
  intro flash
  change (∑ event : Fin 10 × Flash,if event.2=flash then (1:ℕ) else 0)=10
  rw [Fintype.sum_prod_type]
  simp

theorem constructed_selection_has120_events_and20_observations_per_candidate :
    Fintype.card CompleteEvents=120 ∧
      ∀ c : Cell,columnCount (physicalMatrix completeSchedule) c=20 := by
  refine ⟨by decide,?_⟩
  exact physical_complete_gives_twenty_per_cell completeSchedule constructed120_event_schedule_is_physically_complete

def eventPositionEquiv : CompleteEvents ≃ Fin 120 := Fintype.equivFinOfCardEq (by decide)
def indexedCompleteSchedule (event : Fin 120) : Flash := completeSchedule (eventPositionEquiv.symm event)

theorem actual120_position_witness_is_physically_complete : PhysicalComplete indexedCompleteSchedule := by
  rw [physical_complete_iff_each_flash_ten]
  intro flash
  have reindex : flashCount indexedCompleteSchedule flash=flashCount completeSchedule flash := by
    unfold flashCount indexedCompleteSchedule
    simpa only [Equiv.symm_apply_apply] using
      (Equiv.sum_comp eventPositionEquiv (fun e => if completeSchedule (eventPositionEquiv.symm e)=flash then (1:ℕ) else 0)).symm
  rw [reindex]
  exact (physical_complete_iff_each_flash_ten completeSchedule).mp constructed120_event_schedule_is_physically_complete flash

theorem actual120_position_witness_passes_the_actual_fixed36_vocabulary_gate :
    membershipReason 120 36 (indexedPhysicalMatrix indexedCompleteSchedule)=none := by
  exact indexed_complete_physical_matrix_passes_the_actual36_position_gate indexedCompleteSchedule
    actual120_position_witness_is_physically_complete

theorem zero_llrs_give_actual_uniform36_probability_and_absolute_tie :
    (∀ c : Fin 36,(readout (fun _ => 0)).probabilities c=1/36) ∧
      (readout (fun _ : Fin 36 => 0)).winnerIndex=none := by
  constructor
  · intro c
    norm_num [readout,probability_eq_exp_div_partition,partition]
  · norm_num [readout,winner,IsTied,closeCandidates,sourceTolerance]

end PhysicalSelection
end
end N2P3.SelectionERPEvidence
