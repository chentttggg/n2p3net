import N2P3.FixedResampleMean
import N2P3.FixedResampleBatches
import N2P3.LinearQuantile
import N2P3.RequestedNLLMacroWeights
import Mathlib.Data.Fintype.EquivFin

/-! Exact source interface for stats.transfer._summary after participant rows
have been formed. completed/nll below are generic mean-kernel slots; actual
study admission is literal true and participant Some is complete_pair.
Units are the actual ordered source vectors of these complete participants.
No randomness/independence/CI coverage assertion follows from this interface. -/

namespace N2P3.FixedResampleInterface

open scoped BigOperators

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

structure SummaryInput (Study Participant : Type) where
  studies : Finset Study
  requested : Study → Finset Participant
  participantMean : Study → Participant → Option ℝ

variable {Study Participant : Type}

def meanProgram (input : SummaryInput Study Participant) : N2P3.RequestedNLLMacroWeights.Program Study Participant where
  cells := input.studies
  decisions := input.requested
  completed := fun _ => true
  nll := input.participantMean

def unitCount (input : SummaryInput Study Participant) (study : Study) : ℕ :=
  (N2P3.RequestedNLLMacroWeights.scored (meanProgram input) study).card

def estimable (input : SummaryInput Study Participant) : Finset Study :=
  N2P3.RequestedNLLMacroWeights.available (meanProgram input)

def fullGate (input : SummaryInput Study Participant) : Prop := (estimable input).card=input.studies.card

def bootstrapEnabled (input : SummaryInput Study Participant) (withIntervals : Bool) : Prop :=
  withIntervals=true ∧ ∃ study ∈ estimable input,1<unitCount input study

def fromRequestedProgram (program : N2P3.RequestedAggregationWeights.Program Study Participant) : SummaryInput Study Participant where
  studies := program.studies
  requested := program.participants
  participantMean := N2P3.RequestedAggregationWeights.participantEstimate program

theorem original_complete_participant_filter_is_the_resample_unit_population
    (program : N2P3.RequestedAggregationWeights.Program Study Participant) (study : Study) :
    N2P3.RequestedNLLMacroWeights.scored (meanProgram (fromRequestedProgram program)) study=
      N2P3.RequestedAggregationWeights.completeParticipants program study := by
  ext participant
  by_cases complete : N2P3.RequestedAggregationWeights.missingPairs program study participant=∅ <;>
    simp [N2P3.RequestedNLLMacroWeights.scored,meanProgram,fromRequestedProgram,
      N2P3.RequestedAggregationWeights.participantEstimate,N2P3.RequestedAggregationWeights.completeParticipants,complete]

theorem original_estimable_studies_are_the_resample_study_population
    (program : N2P3.RequestedAggregationWeights.Program Study Participant) :
    estimable (fromRequestedProgram program)=N2P3.RequestedAggregationWeights.estimableStudies program := by
  unfold estimable N2P3.RequestedNLLMacroWeights.available
  simp only [original_complete_participant_filter_is_the_resample_unit_population]
  simp [meanProgram,fromRequestedProgram,N2P3.RequestedAggregationWeights.estimableStudies]

theorem full_interval_gate_is_the_existing_full_mean_gate
    (program : N2P3.RequestedAggregationWeights.Program Study Participant) :
    fullGate (fromRequestedProgram program) ↔ N2P3.RequestedAggregationWeights.fullStudyFlag program := by
  unfold fullGate N2P3.RequestedAggregationWeights.fullStudyFlag
  rw [original_estimable_studies_are_the_resample_study_population]
  rfl

theorem every_estimable_study_has_a_positive_source_unit_count
    (input : SummaryInput Study Participant) (study : Study) (member : study ∈ estimable input) :
    0<unitCount input study := by
  have nonempty := (Finset.mem_filter.mp member).2.2
  exact Finset.card_pos.mpr (Finset.nonempty_iff_ne_empty.mpr nonempty)

theorem enabling_overall_resampling_guarantees_a_nonempty_study_population
    (input : SummaryInput Study Participant) (withIntervals : Bool) (enabled : bootstrapEnabled input withIntervals) :
    estimable input≠∅ := by
  obtain ⟨study,member,_⟩ := enabled.2
  exact Finset.nonempty_iff_ne_empty.mp ⟨study,member⟩

def drawValue {draws : ℕ} (input : SummaryInput Study Participant)
    (units : (study : Study) → Fin (unitCount input study) → ℝ)
    (indices : Fin draws → (study : Study) → Fin (unitCount input study) → Fin (unitCount input study))
    (draw : Fin draws) : ℝ :=
  N2P3.FixedResampleMean.studyDrawMean (estimable input) (unitCount input) units (indices draw)

def orderedDrawSample {draws : ℕ} (input : SummaryInput Study Participant)
    (units : (study : Study) → Fin (unitCount input study) → ℝ)
    (indices : Fin draws → (study : Study) → Fin (unitCount input study) → Fin (unitCount input study))
    (rank : Equiv.Perm (Fin draws)) (ordered : Monotone (fun draw => drawValue input units indices (rank draw))) :
    N2P3.LinearQuantile.OrderedSample draws :=
  ⟨drawValue input units indices,rank,ordered⟩

/- The ordered vector is a bijective enumeration of the ACTUAL complete
participant filter. Values are read from that participant's actual Option;
no arbitrary real vector is substituted for the source's participant means. -/
def SourceUnitOrder (input : SummaryInput Study Participant) :=
  (study : Study) → Fin (unitCount input study) ≃
    {participant // participant ∈ N2P3.RequestedNLLMacroWeights.scored (meanProgram input) study}

def canonicalSourceUnitOrder (input : SummaryInput Study Participant) : SourceUnitOrder input :=
  fun study => (N2P3.RequestedNLLMacroWeights.scored (meanProgram input) study).equivFin.symm

def participantValues (input : SummaryInput Study Participant) (order : SourceUnitOrder input)
    (study : Study) (position : Fin (unitCount input study)) : ℝ :=
  (input.participantMean study (order study position).val).getD 0

theorem every_actual_complete_participant_occurs_exactly_once_in_source_vector
    (input : SummaryInput Study Participant) (order : SourceUnitOrder input) (study : Study)
    (participant : {participant // participant ∈ N2P3.RequestedNLLMacroWeights.scored (meanProgram input) study}) :
    ∃! position,order study position=participant := by
  exact ⟨(order study).symm participant,(order study).apply_symm_apply participant,
    fun other same => (order study).injective (same.trans ((order study).apply_symm_apply participant).symm)⟩

theorem source_vector_uses_the_actual_available_participant_value
    (input : SummaryInput Study Participant) (order : SourceUnitOrder input)
    (study : Study) (position : Fin (unitCount input study)) :
    input.participantMean study (order study position).val=some (participantValues input order study position) := by
  have admitted := (Finset.mem_filter.mp (order study position).property).2
  change (input.participantMean study (order study position).val).isSome=true at admitted
  cases found : input.participantMean study (order study position).val with
  | none => simp [found] at admitted
  | some value => simp [participantValues,found]

def sourceDrawValue {draws : ℕ} (input : SummaryInput Study Participant) (order : SourceUnitOrder input)
    (indices : Fin draws → (study : Study) → Fin (unitCount input study) → Fin (unitCount input study))
    (draw : Fin draws) : ℝ := drawValue input (participantValues input order) indices draw

theorem ordered_source_sample_raw_values_are_the_complete_participant_draws {draws : ℕ}
    (input : SummaryInput Study Participant) (order : SourceUnitOrder input)
    (indices : Fin draws → (study : Study) → Fin (unitCount input study) → Fin (unitCount input study))
    (rank : Equiv.Perm (Fin draws))
    (ordered : Monotone (fun draw => sourceDrawValue input order indices (rank draw))) (draw : Fin draws) :
    (orderedDrawSample input (participantValues input order) indices rank ordered).raw draw=
      sourceDrawValue input order indices draw := by rfl

def computedInterval {draws : ℕ} (input : SummaryInput Study Participant) (withIntervals : Bool)
    (sample : N2P3.LinearQuantile.OrderedSample draws) (positive : 0<draws) : Option (ℝ × ℝ) :=
  if bootstrapEnabled input withIntervals then some (N2P3.LinearQuantile.interval sample positive (19/20) (by norm_num)) else none

def partialInterval {draws : ℕ} (input : SummaryInput Study Participant) (withIntervals : Bool)
    (sample : N2P3.LinearQuantile.OrderedSample draws) (positive : 0<draws) : Option (ℝ × ℝ) :=
  computedInterval input withIntervals sample positive

def fullInterval {draws : ℕ} (input : SummaryInput Study Participant) (withIntervals : Bool)
    (sample : N2P3.LinearQuantile.OrderedSample draws) (positive : 0<draws) : Option (ℝ × ℝ) :=
  if fullGate input then computedInterval input withIntervals sample positive else none

theorem each_source_draw_has_the_fixed_study_multiplicity_expansion {draws : ℕ}
    (input : SummaryInput Study Participant)
    (units : (study : Study) → Fin (unitCount input study) → ℝ)
    (indices : Fin draws → (study : Study) → Fin (unitCount input study) → Fin (unitCount input study)) (draw : Fin draws) :
    drawValue input units indices draw=
      ∑ study ∈ estimable input,∑ unit : Fin (unitCount input study),units study unit*
        (N2P3.FixedResampleMean.drawWeight (indices draw study) unit/((estimable input).card : ℝ)) := by
  exact N2P3.FixedResampleMean.each_study_draw_retains_its_own_unit_denominator _ _ _ _

theorem enabled_fixed_study_draw_has_total_original_unit_weight_one {draws : ℕ}
    (input : SummaryInput Study Participant) (withIntervals : Bool) (enabled : bootstrapEnabled input withIntervals)
    (indices : Fin draws → (study : Study) → Fin (unitCount input study) → Fin (unitCount input study)) (draw : Fin draws) :
    (∑ study ∈ estimable input,∑ unit : Fin (unitCount input study),
      N2P3.FixedResampleMean.drawWeight (indices draw study) unit/((estimable input).card : ℝ))=1 := by
  exact N2P3.FixedResampleMean.fixed_study_draw_coefficient_mass_is_one _ _ _
    (enabling_overall_resampling_guarantees_a_nonempty_study_population input withIntervals enabled)
    (every_estimable_study_has_a_positive_source_unit_count input)

theorem all_singleton_studies_do_not_trigger_an_overall_bootstrap
    (input : SummaryInput Study Participant) (withIntervals : Bool)
    (singletons : ∀ study ∈ estimable input,unitCount input study≤1) : ¬bootstrapEnabled input withIntervals := by
  rintro ⟨_,study,member,more⟩
  exact (not_lt_of_ge (singletons study member)) more

theorem intervals_disabled_means_no_full_or_partial_interval {draws : ℕ}
    (input : SummaryInput Study Participant) (sample : N2P3.LinearQuantile.OrderedSample draws) (positive : 0<draws) :
    fullInterval input false sample positive=none ∧ partialInterval input false sample positive=none := by
  simp [fullInterval,partialInterval,computedInterval,bootstrapEnabled]

theorem a_missing_expected_study_can_keep_only_the_explicit_partial_interval {draws : ℕ}
    (input : SummaryInput Study Participant) (withIntervals : Bool)
    (sample : N2P3.LinearQuantile.OrderedSample draws) (positive : 0<draws)
    (enabled : bootstrapEnabled input withIntervals) (partialOnly : ¬fullGate input) :
    fullInterval input withIntervals sample positive=none ∧ partialInterval input withIntervals sample positive=
      some (N2P3.LinearQuantile.interval sample positive (19/20) (by norm_num)) := by
  simp [fullInterval,partialInterval,computedInterval,enabled,partialOnly]

theorem full_and_partial_intervals_share_the_same_computed_quantiles_when_full {draws : ℕ}
    (input : SummaryInput Study Participant) (withIntervals : Bool)
    (sample : N2P3.LinearQuantile.OrderedSample draws) (positive : 0<draws) (full : fullGate input) :
    fullInterval input withIntervals sample positive=partialInterval input withIntervals sample positive := by
  simp [fullInterval,partialInterval,full]

end
end N2P3.FixedResampleInterface
