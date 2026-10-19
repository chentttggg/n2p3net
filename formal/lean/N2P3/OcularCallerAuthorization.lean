import N2P3.OcularReferenceRecord
import Mathlib.Data.Finset.Max
import Mathlib.Data.Finset.Lattice.Lemmas

/-!
EO1/EO2 actual caller gates, source-first slice_contract_02. Input sample IDs
carry their original session/native-row identity. The finite mask is the
realized QC output; no invariance of recomputed QC or physical identity is
assumed. Native timestamps are finite before these exact-real relations.
-/
namespace N2P3.OcularCallerAuthorization
open N2P3.OcularCalibrationChain
open N2P3.OcularReferenceRecord
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def actualMask {T : Type*} (intervals acquired excluded qcValid : Finset T) : Finset T :=
  ((intervals ∩ acquired) \ excluded) ∩ qcValid

theorem actual_selected_row_requires_interval_acquisition_exclusion_and_QC_authority
    {T : Type*} (intervals acquired excluded qcValid : Finset T) (row : T) :
    row ∈ actualMask intervals acquired excluded qcValid ↔
      row ∈ intervals ∧ row ∈ acquired ∧ row ∉ excluded ∧ row ∈ qcValid := by
  simp [actualMask,and_assoc]

theorem acquired_cutoff_exclusions_and_QC_can_only_remove_authorized_rows
    {T : Type*} (intervals acquired excluded qcValid : Finset T) :
    actualMask intervals acquired excluded qcValid ⊆ intervals := by
  intro row member
  exact ((actual_selected_row_requires_interval_acquisition_exclusion_and_QC_authority
    intervals acquired excluded qcValid row).mp member).1

theorem additional_exclusions_can_only_shrink_the_actual_fit_support
    {T : Type*} (intervals acquired excluded moreExcluded qcValid : Finset T)
    (more : excluded ⊆ moreExcluded) :
    actualMask intervals acquired moreExcluded qcValid ⊆ actualMask intervals acquired excluded qcValid := by
  intro row member
  rcases (actual_selected_row_requires_interval_acquisition_exclusion_and_QC_authority
    intervals acquired moreExcluded qcValid row).mp member with ⟨hi,ha,he,hq⟩
  exact (actual_selected_row_requires_interval_acquisition_exclusion_and_QC_authority
    intervals acquired excluded qcValid row).mpr ⟨hi,ha,fun h => he (more h),hq⟩

theorem retained_rest_is_selected_when_it_meets_the_actual_fit_caller_mask
    {T : Type*} (intervals acquired excluded qcValid rest : Finset T) (row : T)
    (_rest : row ∈ rest) (hi : row ∈ intervals) (ha : row ∈ acquired)
    (he : row ∉ excluded) (hq : row ∈ qcValid) :
    row ∈ actualMask intervals acquired excluded qcValid :=
  (actual_selected_row_requires_interval_acquisition_exclusion_and_QC_authority
    intervals acquired excluded qcValid row).mpr ⟨hi,ha,he,hq⟩

def pairedRows {S A : Type*} (session : S) (rows : List (A × Bool)) : List ((S × A) × Bool) :=
  rows.map (fun row => ((session,row.1),row.2))

def orderedConcatenation {S A : Type*} (sessions : List (S × List (A × Bool))) : List ((S × A) × Bool) :=
  sessions.flatMap (fun entry => pairedRows entry.1 entry.2)

theorem mask_selection_of_concatenated_sessions_preserves_actual_session_native_row_pairs
    {S A : Type*} (sessions : List (S × List (A × Bool))) :
    (orderedConcatenation sessions).filter (fun row => row.2)=
      sessions.flatMap (fun entry => (pairedRows entry.1 entry.2).filter (fun row => row.2)) := by
  simp [orderedConcatenation,List.filter_flatMap]

theorem source_selected_fit_ignores_all_edits_outside_the_realized_caller_support
    {T C E : Type*} [Fintype E] (intervals acquired excluded qcValid : Finset T)
    (eye otherEye : T → E → ℝ) (eeg otherEEG : T → C → ℝ) (strength : ℝ)
    (sameEye : ∀ row ∈ intervals, eye row=otherEye row)
    (sameEEG : ∀ row ∈ intervals, eeg row=otherEEG row) :
    fitFrozen (actualMask intervals acquired excluded qcValid) eye eeg strength=
      fitFrozen (actualMask intervals acquired excluded qcValid) otherEye otherEEG strength := by
  apply arbitrary_unselected_EEG_and_EOG_edits_preserve_the_complete_frozen_numeric_fit
  · intro row member
    exact sameEye row (acquired_cutoff_exclusions_and_QC_can_only_remove_authorized_rows _ _ _ _ member)
  · intro row member
    exact sameEEG row (acquired_cutoff_exclusions_and_QC_can_only_remove_authorized_rows _ _ _ _ member)

structure Ledger (P S : Type*) where
  participant : P
  sessions : Finset S

def budgetAuthorized {P S : Type*} (allowed : P → Finset S) (ledger : Ledger P S) : Prop :=
  ledger.sessions.Nonempty ∧ ledger.sessions ⊆ allowed ledger.participant

def singleParticipantBudget {P S : Type*} (participant : P) (supplied : Finset S) (candidate : P) : Finset S :=
  if candidate=participant then supplied else ∅

theorem nonempty_authorized_QC_ledger_derives_the_exact_producer_participant_and_session_subset
    {P S : Type*} (participant : P) (supplied : Finset S) (ledger : Ledger P S)
    (accepted : budgetAuthorized (singleParticipantBudget participant supplied) ledger) :
    ledger.participant=participant ∧ ledger.sessions ⊆ supplied := by
  by_cases same : ledger.participant=participant
  · exact ⟨same,by simpa only [singleParticipantBudget,if_pos same] using accepted.2⟩
  · obtain ⟨session,member⟩ := accepted.1
    have impossible := accepted.2 member
    simp only [singleParticipantBudget,if_neg same,Finset.notMem_empty] at impossible

def appliedBudget {P S : Type*} (applied digestMatches : Bool)
    (allowed : P → Finset S) (ledger : Ledger P S) : Prop :=
  applied=true → digestMatches=true ∧ budgetAuthorized allowed ledger

theorem nonapplied_status_bypasses_the_source_budget_helper
    {P S : Type*} (digestMatches : Bool) (allowed : P → Finset S) (ledger : Ledger P S) :
    appliedBudget false digestMatches allowed ledger := by simp [appliedBudget]

theorem unknown_participant_with_empty_allowed_support_cannot_authorize_applied_processing
    {P S : Type*} (allowed : P → Finset S) (ledger : Ledger P S)
    (unknown : allowed ledger.participant=∅) (digest : Bool) :
    ¬ appliedBudget true digest allowed ledger := by
  intro admitted
  rcases (admitted rfl).2.1 with ⟨session,member⟩
  have := (admitted rfl).2.2 member
  simp [unknown] at this

theorem any_out_of_budget_session_refuses_applied_processing
    {P S : Type*} (allowed : P → Finset S) (ledger : Ledger P S) (digest : Bool)
    (session : S) (present : session ∈ ledger.sessions) (outside : session ∉ allowed ledger.participant) :
    ¬ appliedBudget true digest allowed ledger := by
  intro admitted
  exact outside ((admitted rfl).2.2 present)

theorem nested_recovery_authority_is_transitive_through_the_decoded_QC_ledger
    {P S : Type*} (allowed : P → Finset S) (qc recovery : Ledger P S)
    (qcAccepted : budgetAuthorized allowed qc) (same : recovery.participant=qc.participant)
    (nonempty : recovery.sessions.Nonempty) (subset : recovery.sessions ⊆ qc.sessions) :
    budgetAuthorized allowed recovery := by
  refine ⟨nonempty,?_⟩
  rw [same]
  exact subset.trans qcAccepted.2

theorem producer_QC_and_recovery_cannot_exceed_the_actual_supplied_ocular_sessions
    {P S : Type*} (supplied : Finset S) (participant : P) (qc recovery : Ledger P S)
    (qcAccepted : budgetAuthorized (singleParticipantBudget participant supplied) qc)
    (recoverySame : recovery.participant=qc.participant) (recoveryWithin : recovery.sessions ⊆ qc.sessions) :
    recovery.participant=participant ∧ recovery.sessions ⊆ supplied := by
  have actual := nonempty_authorized_QC_ledger_derives_the_exact_producer_participant_and_session_subset
    participant supplied qc qcAccepted
  exact ⟨recoverySame.trans actual.1,recoveryWithin.trans actual.2⟩

def legacyRawGate {P S : Type*} (model : Ledger P S) (participant : P)
    (session : S) (fitEnd start : ℝ) : Prop :=
  model.participant=participant ∧ model.sessions.Nonempty ∧ (session ∈ model.sessions ∨ fitEnd ≤ start)

def independentGate {S : Type*} (calibration evaluation fit : Finset S) (session : S) : Prop :=
  calibration.Nonempty ∧ evaluation.Nonempty ∧ Disjoint calibration evaluation ∧
    session ∈ calibration ∪ evaluation ∧ fit ⊆ calibration

def prospectiveGate {S : Type*} (calibration : Finset S) (session : S) (available start : ℝ) : Prop :=
  session ∉ calibration ∧ available < start

theorem legacy_raw_replay_allows_a_fit_session_without_a_prospective_time_test
    {P S : Type*} (model : Ledger P S) (participant : P) (session : S)
    (fitEnd start : ℝ) (same : model.participant=participant) (fitSession : session ∈ model.sessions) :
    legacyRawGate model participant session fitEnd start :=
  ⟨same,⟨session,fitSession⟩,Or.inl fitSession⟩

theorem legacy_nonfit_session_accepts_exactly_at_fit_end
    {P S : Type*} (model : Ledger P S) (participant : P) (session : S) (fitEnd : ℝ)
    (same : model.participant=participant) (nonempty : model.sessions.Nonempty) :
    legacyRawGate model participant session fitEnd fitEnd := ⟨same,nonempty,Or.inr le_rfl⟩

theorem personalized_prospective_session_is_after_every_calibration_availability_bound
    {S : Type*} (calibration : Finset S) (session : S) (available start bound : ℝ)
    (accepted : prospectiveGate calibration session available start) (maximum : bound ≤ available) :
    bound < start := lt_of_le_of_lt maximum accepted.2

theorem equality_at_label_availability_is_refused_by_personalized_inference
    {S : Type*} (calibration : Finset S) (session : S) (available : ℝ) :
    ¬ prospectiveGate calibration session available available := by simp [prospectiveGate]

theorem personalized_inference_cannot_reuse_a_calibration_session
    {S : Type*} (calibration : Finset S) (session : S) (available start : ℝ)
    (inCalibration : session ∈ calibration) : ¬ prospectiveGate calibration session available start := by
  intro accepted; exact accepted.1 inCalibration

theorem independent_evaluation_recordings_are_disjoint_from_every_authorized_fit_session
    {S : Type*} (calibration evaluation fit : Finset S) (session fitted : S)
    (accepted : independentGate calibration evaluation fit session)
    (isEvaluation : session ∈ evaluation) (isFit : fitted ∈ fit) : session ≠ fitted := by
  intro same
  subst fitted
  exact Finset.disjoint_left.mp accepted.2.2.1 (accepted.2.2.2.2 isFit) isEvaluation

theorem producer_actual_end_maximum_bounds_every_supplied_session
    {S : Type*} (sessions : Finset S) (nonempty : sessions.Nonempty)
    (ended : S → ℝ) (session : S) (member : session ∈ sessions) :
    ended session ≤ (sessions.image ended).max' (nonempty.image ended) := by
  exact Finset.le_max' _ _ (Finset.mem_image.mpr ⟨session,member,rfl⟩)

theorem raw_new_session_after_actual_maximum_is_after_each_fit_end
    {S : Type*} (sessions : Finset S) (nonempty : sessions.Nonempty)
    (ended : S → ℝ) (session : S) (member : session ∈ sessions) (start : ℝ)
    (accepted : (sessions.image ended).max' (nonempty.image ended) ≤ start) : ended session ≤ start :=
  (producer_actual_end_maximum_bounds_every_supplied_session sessions nonempty ended session member).trans accepted

theorem actual_label_availability_bounds_every_session_end_before_new_inference
    {S : Type*} (sessions : Finset S) (session : S) (ended : S → ℝ) (available start : ℝ)
    (endBeforeLabelAvailable : ∀ calibration ∈ sessions, ended calibration ≤ available)
    (member : session ∈ sessions) (inference : available < start) : ended session < start :=
  lt_of_le_of_lt (endBeforeLabelAvailable session member) inference

end
end N2P3.OcularCallerAuthorization
