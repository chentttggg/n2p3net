import N2P3Pilot
import N2P3.CandidateProbability
import N2P3.RelativeCandidateTie
import Mathlib.Data.Fintype.Card
import Mathlib.Logic.Equiv.Fin.Basic

/-! MembershipDecision's finite schedule/observed/identifiable operations.
Canonical finite axes model the individually validated integer/string axes and
boolean matrix shape. There is no assumed whole constructor or scorer success.
No proper-subset row requirement is added: that belongs to the different
continuous scorer. Native decoding/coercion and physical authority are separate.
Source-first slice_contract_01 was frozen before this module. -/
namespace N2P3.GeneralMembershipOperational
open scoped BigOperators
noncomputable section
attribute [local instance] Classical.propDecidable

structure Ledger (E S C : Type*) (repetitions : ℕ) where
  repetition : E → Fin repetitions
  stimulus : E → S
  member : E → C → Bool
  evidence : E → Option ℕ
  timelineComplete : Bool

def slot {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (e : E) : Fin r × S :=
  (ledger.repetition e,ledger.stimulus e)

def scheduleComplete {E S C : Type*} [Fintype E] [Fintype S] {r : ℕ}
    (ledger : Ledger E S C r) : Prop :=
  ledger.timelineComplete=true ∧ Fintype.card E=r*Fintype.card S

def uniqueEvidence {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) : Prop :=
  ∀ e f index, ledger.evidence e=some index → ledger.evidence f=some index → e=f

def consistentStimulusPatterns {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) : Prop :=
  ∀ e f, ledger.stimulus e=ledger.stimulus f → ∀ c, ledger.member e c=ledger.member f c

inductive ScheduleRefusal where
  | reusedEvidence | duplicateSlot | inconsistentPattern
  deriving DecidableEq

/-- The three relational checks, in their literal source order, after aligned
canonical axes, names, shape and scalar-index bounds have been validated. -/
def scheduleRelationCheck {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) : Option ScheduleRefusal :=
  if ¬uniqueEvidence ledger then some .reusedEvidence
  else if ¬Function.Injective (slot ledger) then some .duplicateSlot
  else if ¬consistentStimulusPatterns ledger then some .inconsistentPattern else none

theorem schedule_relational_acceptance_iff_individual_source_checks
    {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) :
    scheduleRelationCheck ledger=none ↔ uniqueEvidence ledger ∧
      Function.Injective (slot ledger) ∧ consistentStimulusPatterns ledger := by
  unfold scheduleRelationCheck
  split_ifs <;> simp_all

theorem reused_evidence_is_refused_before_duplicate_slot_or_pattern
    {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (reused : ¬uniqueEvidence ledger) :
    scheduleRelationCheck ledger=some .reusedEvidence := by simp [scheduleRelationCheck,reused]

theorem duplicate_slot_is_refused_after_evidence_identity_check
    {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (unique : uniqueEvidence ledger)
    (duplicate : ¬Function.Injective (slot ledger)) :
    scheduleRelationCheck ledger=some .duplicateSlot := by simp [scheduleRelationCheck,unique,duplicate]

theorem inconsistent_stimulus_pattern_is_refused_after_both_uniqueness_checks
    {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (unique : uniqueEvidence ledger)
    (slots : Function.Injective (slot ledger)) (inconsistent : ¬consistentStimulusPatterns ledger) :
    scheduleRelationCheck ledger=some .inconsistentPattern := by
  simp [scheduleRelationCheck,unique,slots,inconsistent]

theorem unique_slots_and_source_count_give_every_requested_slot
    {E S C : Type*} [Fintype E] [Fintype S] {r : ℕ} (ledger : Ledger E S C r)
    (unique : Function.Injective (slot ledger)) (complete : scheduleComplete ledger) :
    Function.Bijective (slot ledger) := by
  refine ⟨unique,?_⟩
  by_contra missing
  have fewer := Fintype.card_lt_of_injective_not_surjective (slot ledger) unique missing
  have equal : Fintype.card E=Fintype.card (Fin r × S) := by
    simpa only [Fintype.card_prod,Fintype.card_fin] using complete.2
  omega

theorem a_missing_requested_slot_prevents_source_count_completeness
    {E S C : Type*} [Fintype E] [Fintype S] {r : ℕ} (ledger : Ledger E S C r)
    (unique : Function.Injective (slot ledger)) (requested : Fin r × S)
    (missing : ∀ e, slot ledger e ≠ requested) : ¬scheduleComplete ledger := by
  intro complete
  obtain ⟨e,same⟩ := (unique_slots_and_source_count_give_every_requested_slot ledger unique complete).2 requested
  exact missing e same

def prefixLedger {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (length : ℕ) :
    Ledger {e // (ledger.repetition e).val < length} S C length :=
  ⟨fun e => ⟨(ledger.repetition e.val).val,e.property⟩,
   fun e => ledger.stimulus e.val,fun e => ledger.member e.val,
   fun e => ledger.evidence e.val,ledger.timelineComplete⟩

theorem prefix_keeps_original_evidence_indices_membership_and_timeline_flag
    {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (length : ℕ)
    (e : {e // (ledger.repetition e).val < length}) :
    (prefixLedger ledger length).evidence e=ledger.evidence e.val ∧
      (prefixLedger ledger length).member e=ledger.member e.val ∧
      (prefixLedger ledger length).timelineComplete=ledger.timelineComplete := ⟨rfl,rfl,rfl⟩

theorem prefix_preserves_all_three_relational_checks
    {E S C : Type*} {r : ℕ} (ledger : Ledger E S C r) (length : ℕ)
    (unique : uniqueEvidence ledger) (slots : Function.Injective (slot ledger))
    (patterns : consistentStimulusPatterns ledger) :
    uniqueEvidence (prefixLedger ledger length) ∧
      Function.Injective (slot (prefixLedger ledger length)) ∧
      consistentStimulusPatterns (prefixLedger ledger length) := by
  refine ⟨?_,?_,?_⟩
  · intro e f index left right
    exact Subtype.ext (unique e.val f.val index left right)
  · intro e f same
    apply Subtype.ext
    apply slots
    apply Prod.ext
    · apply Fin.ext
      exact congrArg (fun pair => pair.1.val) same
    · exact congrArg (fun pair : Fin length × S => pair.2) same
  · intro e f same c
    exact patterns e.val f.val same c

theorem prefix_of_a_complete_unique_schedule_has_every_prefix_slot
    {E S C : Type*} [Fintype E] [Fintype S] {r : ℕ} (ledger : Ledger E S C r)
    (length : ℕ) (bound : length ≤ r) (unique : Function.Injective (slot ledger))
    (complete : scheduleComplete ledger) :
    Function.Bijective (slot (prefixLedger ledger length)) := by
  refine ⟨?_,?_⟩
  · intro e f same
    apply Subtype.ext
    apply unique
    apply Prod.ext
    · apply Fin.ext
      exact congrArg (fun pair => pair.1.val) same
    · exact congrArg (fun pair : Fin length × S => pair.2) same
  · intro target
    obtain ⟨e,same⟩ := (unique_slots_and_source_count_give_every_requested_slot ledger unique complete).2
      (⟨target.1.val,lt_of_lt_of_le target.1.isLt bound⟩,target.2)
    have index : (ledger.repetition e).val=target.1.val := congrArg (fun pair => pair.1.val) same
    refine ⟨⟨e,by rw [index]; exact target.1.isLt⟩,?_⟩
    apply Prod.ext
    · exact Fin.ext index
    · exact congrArg (fun pair : Fin r × S => pair.2) same

theorem prefix_of_complete_schedule_has_the_actual_prefix_count
    {E S C : Type*} [Fintype E] [Fintype S] {r : ℕ} (ledger : Ledger E S C r)
    (length : ℕ) (bound : length ≤ r) (unique : Function.Injective (slot ledger))
    (complete : scheduleComplete ledger) : scheduleComplete (prefixLedger ledger length) := by
  refine ⟨complete.1,?_⟩
  have count := Fintype.card_of_bijective
    (prefix_of_a_complete_unique_schedule_has_every_prefix_slot ledger length bound unique complete)
  simpa only [Fintype.card_prod,Fintype.card_fin] using count

def physicallyObserved {E : Type*} (evidence : E → Option ℕ) (e : E) : Bool :=
  (evidence e).isSome

def selected {E : Type*} (evidence : E → Option ℕ) (supplied : Option (E → Bool)) (e : E) : Bool :=
  match supplied with
  | none => physicallyObserved evidence e
  | some available => physicallyObserved evidence e && available e

def availabilityAdmitted {E : Type*} (evidence : E → Option ℕ) (supplied : E → Bool) : Prop :=
  ∀ e, supplied e=true → physicallyObserved evidence e=true

theorem default_selection_is_physical_observation {E : Type*} (evidence : E → Option ℕ) :
    selected evidence none=physicallyObserved evidence := rfl

theorem selected_event_is_always_physically_observed {E : Type*} (evidence : E → Option ℕ)
    (supplied : Option (E → Bool)) (e : E) (kept : selected evidence supplied e=true) :
    physicallyObserved evidence e=true := by
  cases supplied with
  | none => exact kept
  | some mask => cases h : physicallyObserved evidence e <;> simp_all [selected]

theorem physical_missing_event_cannot_be_selected {E : Type*} (evidence : E → Option ℕ)
    (supplied : Option (E → Bool)) (e : E) (missing : evidence e=none) :
    selected evidence supplied e=false := by
  cases supplied <;> simp [selected,physicallyObserved,missing]

theorem supplied_mask_cannot_restore_physical_missing_event {E : Type*} (evidence : E → Option ℕ)
    (supplied : E → Bool) (e : E) (missing : evidence e=none) (restore : supplied e=true) :
    ¬availabilityAdmitted evidence supplied := by
  intro accepted
  have actual := accepted e restore
  simp [physicallyObserved,missing] at actual

theorem admitted_supplied_mask_equals_actual_selection {E : Type*} (evidence : E → Option ℕ)
    (supplied : E → Bool) (accepted : availabilityAdmitted evidence supplied) :
    selected evidence (some supplied)=supplied := by
  funext e
  cases flag : supplied e with
  | false => simp [selected,flag]
  | true => simp [selected,flag,accepted e flag]

def sourceComplete {E S C : Type*} [Fintype E] [Fintype S] {r : ℕ}
    (ledger : Ledger E S C r) (mask : E → Bool) : Prop :=
  scheduleComplete ledger ∧ ∀ e, mask e=true

def pattern {E C : Type*} (mask : E → Bool) (member : E → C → Bool) (candidate : C) :
    {e // mask e=true} → Bool := fun e => member e.val candidate

def identifiable {E C : Type*} [Fintype C] (mask : E → Bool) (member : E → C → Bool) : Prop := by
  classical
  exact (Finset.univ.image (pattern mask member)).card=Fintype.card C

theorem actual_unique_column_count_iff_pattern_injective {E C : Type*} [Fintype C]
    (mask : E → Bool) (member : E → C → Bool) :
    identifiable mask member ↔ Function.Injective (pattern mask member) := by
  classical
  rw [identifiable, ← Finset.card_univ, Finset.card_image_iff]
  simp only [Set.InjOn, Finset.coe_univ, Set.mem_univ, forall_const]
  rfl

theorem duplicate_available_columns_are_unidentifiable {E C : Type*} [Fintype C]
    (mask : E → Bool) (member : E → C → Bool) (a b : C) (different : a ≠ b)
    (same : ∀ e, mask e=true → member e a=member e b) : ¬identifiable mask member := by
  intro identified
  apply different
  apply (actual_unique_column_count_iff_pattern_injective mask member).mp identified
  funext e
  exact same e.val e.property

def scores {E C : Type*} [Fintype E] (mask : E → Bool) (member : E → C → Bool)
    (evidence offset : E → ℝ) (temperature : ℝ) (prior : C → ℝ) (candidate : C) : ℝ :=
  N2P3Pilot.membershipScore (fun e c => if member e c then 1 else 0)
    mask evidence offset candidate / temperature + Real.log (prior candidate)

theorem scores_ignore_values_of_unavailable_events {E C : Type*} [Fintype E]
    (mask : E → Bool) (member : E → C → Bool) (left right offset : E → ℝ)
    (temperature : ℝ) (prior : C → ℝ) (same : ∀ e, mask e=true → left e=right e) :
    scores mask member left offset temperature prior=scores mask member right offset temperature prior := by
  funext candidate
  unfold scores
  rw [N2P3Pilot.unavailable_evidence_irrelevant _ mask left right offset same candidate]

theorem all_unavailable_scores_are_log_prior {E C : Type*} [Fintype E]
    (member : E → C → Bool) (evidence offset : E → ℝ) (temperature : ℝ) (prior : C → ℝ) :
    scores (fun _ => false) member evidence offset temperature prior=fun c => Real.log (prior c) := by
  funext c
  simp [scores,N2P3Pilot.membershipScore]

theorem positive_normalized_prior_is_the_actual_no_evidence_posterior
    {C : Type*} [Fintype C] [Nonempty C] (prior : C → ℝ)
    (positive : ∀ c, 0 < prior c) (normalized : ∑ c, prior c=1) (c : C) :
    CandidateProbability.probability (fun j => Real.log (prior j)) c=prior c := by
  rw [CandidateProbability.probability_eq_exp_div_partition]
  simp only [CandidateProbability.partition,Real.exp_log (positive _)]
  rw [normalized,div_one]

def scalarParametersAccepted {C : Type*} [Fintype C] (temperature : ℝ) (prior : C → ℝ) : Prop :=
  0 < temperature ∧ (∀ c, 0 < prior c) ∧ |(∑ c, prior c)-1| ≤ (1/100000000:ℝ)

/-- The source admits a tolerance around prior mass one. It is the normalized
ratio, not literally the supplied vector, unless its mass is exactly one. -/
theorem actual_no_evidence_posterior_normalizes_positive_prior_mass
    {C : Type*} [Fintype C] [Nonempty C] (prior : C → ℝ)
    (positive : ∀ c, 0 < prior c) (c : C) :
    CandidateProbability.probability (fun j => Real.log (prior j)) c=prior c/(∑ j,prior j) := by
  rw [CandidateProbability.probability_eq_exp_div_partition]
  simp only [CandidateProbability.partition,Real.exp_log (positive _)]

theorem source_accepted_prior_has_strictly_positive_total_mass
    {C : Type*} [Fintype C] [Nonempty C] (temperature : ℝ) (prior : C → ℝ)
    (accepted : scalarParametersAccepted temperature prior) : 0 < ∑ c, prior c :=
  Finset.sum_pos (fun c _ => accepted.2.1 c) Finset.univ_nonempty

theorem no_evidence_posterior_still_sums_to_one_for_tolerance_admitted_prior
    {C : Type*} [Fintype C] [Nonempty C] (temperature : ℝ) (prior : C → ℝ)
    (accepted : scalarParametersAccepted temperature prior) :
    (∑ c, prior c/(∑ j,prior j))=1 := by
  simp_rw [← actual_no_evidence_posterior_normalizes_positive_prior_mass prior accepted.2.1]
  exact CandidateProbability.probability_sum_one _

inductive Refusal where
  | noUsableEvidence | incompleteCoverage | unidentifiable | tie
  deriving DecidableEq

def operational {E S C : Type*} [Fintype E] [Fintype S] [Fintype C] [Nonempty C] [LinearOrder C]
    {r : ℕ} (ledger : Ledger E S C r) (mask : E → Bool) (score : C → ℝ) : Option C × Option Refusal := by
  classical
  exact if ¬∃ e, mask e=true then (none,some .noUsableEvidence)
    else if ¬sourceComplete ledger mask then (none,some .incompleteCoverage)
    else if ¬identifiable mask ledger.member then (none,some .unidentifiable)
    else
      let guessed := RelativeCandidateTie.prediction score RelativeCandidateTie.sourceAtol RelativeCandidateTie.sourceRtol
      (guessed,if guessed=none then some .tie else none)

theorem no_usable_evidence_refusal_precedes_every_finite_prior_score
    {E S C : Type*} [Fintype E] [Fintype S] [Fintype C] [Nonempty C] [LinearOrder C]
    {r : ℕ} (ledger : Ledger E S C r) (score : C → ℝ) :
    operational ledger (fun _ => false) score=(none,some .noUsableEvidence) := by
  simp [operational]

theorem incomplete_schedule_or_missing_evidence_has_priority_over_tie
    {E S C : Type*} [Fintype E] [Fintype S] [Fintype C] [Nonempty C] [LinearOrder C]
    {r : ℕ} (ledger : Ledger E S C r) (mask : E → Bool) (score : C → ℝ)
    (someEvidence : ∃ e, mask e=true) (incomplete : ¬sourceComplete ledger mask) :
    operational ledger mask score=(none,some .incompleteCoverage) := by
  simp only [operational,if_neg (not_not.mpr someEvidence),if_pos incomplete]

theorem unidentifiable_columns_refuse_even_with_unique_prior_posterior
    {E S C : Type*} [Fintype E] [Fintype S] [Fintype C] [Nonempty C] [LinearOrder C]
    {r : ℕ} (ledger : Ledger E S C r) (mask : E → Bool) (score : C → ℝ)
    (someEvidence : ∃ e, mask e=true) (complete : sourceComplete ledger mask)
    (duplicate : ¬identifiable mask ledger.member) :
    operational ledger mask score=(none,some .unidentifiable) := by
  simp only [operational,if_neg (not_not.mpr someEvidence),if_neg (not_not.mpr complete),if_pos duplicate]

theorem complete_identifiable_route_uses_the_actual_relative_tie
    {E S C : Type*} [Fintype E] [Fintype S] [Fintype C] [Nonempty C] [LinearOrder C]
    {r : ℕ} (ledger : Ledger E S C r) (mask : E → Bool) (score : C → ℝ)
    (someEvidence : ∃ e, mask e=true) (complete : sourceComplete ledger mask)
    (identified : identifiable mask ledger.member) :
    (operational ledger mask score).1=RelativeCandidateTie.prediction score
      RelativeCandidateTie.sourceAtol RelativeCandidateTie.sourceRtol := by
  simp only [operational,if_neg (not_not.mpr someEvidence),if_neg (not_not.mpr complete),
    if_neg (not_not.mpr identified)]

theorem operational_prediction_requires_actual_complete_identifiable_evidence
    {E S C : Type*} [Fintype E] [Fintype S] [Fintype C] [Nonempty C] [LinearOrder C]
    {r : ℕ} (ledger : Ledger E S C r) (mask : E → Bool) (score : C → ℝ) (c : C)
    (returned : (operational ledger mask score).1=some c) :
    (∃ e, mask e=true) ∧ sourceComplete ledger mask ∧ identifiable mask ledger.member := by
  classical
  by_cases usable : ∃ e, mask e=true
  · by_cases complete : sourceComplete ledger mask
    · by_cases identified : identifiable mask ledger.member
      · exact ⟨usable,complete,identified⟩
      · simp [operational,usable,complete,identified] at returned
    · simp [operational,usable,complete] at returned
  · simp [operational,usable] at returned

def directLedger (r k : ℕ) : Ledger (Fin r × Fin k) (Fin k) (Fin k) r :=
  ⟨Prod.fst,Prod.snd,fun e c => decide (e.2=c),fun e => some (e.1.val*k+e.2.val),true⟩

theorem direct_ledger_slots_are_the_full_cartesian_schedule (r k : ℕ) :
    Function.Bijective (slot (directLedger r k)) ∧ scheduleComplete (directLedger r k) := by
  refine ⟨?_,?_⟩
  · exact Function.bijective_id
  · simp [scheduleComplete,directLedger,Fintype.card_prod]

theorem direct_ledger_observed_rows_are_unique (r k : ℕ) (a b : Fin r × Fin k)
    (same : (directLedger r k).evidence a=(directLedger r k).evidence b) : a=b := by
  have equation : a.1.val*k+a.2.val=b.1.val*k+b.2.val := by
    simpa only [directLedger,Option.some.injEq] using same
  apply finProdFinEquiv.injective
  apply Fin.ext
  change a.2.val+k*a.1.val=b.2.val+k*b.1.val
  calc
    _ = a.1.val*k+a.2.val := by ac_rfl
    _ = b.1.val*k+b.2.val := equation
    _ = _ := by ac_rfl

theorem direct_ledger_passes_actual_relational_checks (r k : ℕ) :
    scheduleRelationCheck (directLedger r k)=none := by
  rw [schedule_relational_acceptance_iff_individual_source_checks]
  refine ⟨?_,(direct_ledger_slots_are_the_full_cartesian_schedule r k).1.1,?_⟩
  · intro a b index left right
    exact direct_ledger_observed_rows_are_unique r k a b (left.trans right.symm)
  · intro a b same c
    change decide (a.2=c)=decide (b.2=c)
    rw [show a.2=b.2 from same]

theorem complete_direct_columns_are_identifiable (r k : ℕ) (positive : 0 < r) :
    identifiable (fun _ : Fin r × Fin k => true) (directLedger r k).member := by
  apply (actual_unique_column_count_iff_pattern_injective _ _).mpr
  intro a b same
  have row := congrFun same (⟨(⟨0,positive⟩,a),rfl⟩ : {e : Fin r × Fin k // true=true})
  simpa [pattern,directLedger] using row

theorem adult_nine_R5_schedule_and_identifiability_are_nonvacuous :
    scheduleComplete (directLedger 5 9) ∧
      identifiable (fun _ : Fin 5 × Fin 9 => true) (directLedger 5 9).member :=
  ⟨(direct_ledger_slots_are_the_full_cartesian_schedule 5 9).2,
    complete_direct_columns_are_identifiable 5 9 (by decide)⟩

end
end N2P3.GeneralMembershipOperational
