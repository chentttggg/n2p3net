import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Prod
import Mathlib.Data.Finset.Union
import Mathlib.Data.List.Nodup
import Lean.Elab.Tactic.Omega

/-!
# General requested decision/evidence accounting

Source: transfer/outcomes.py:280-327. Requirements and wrong alternatives
were fixed in lean_outcomes_20261018/source_requirements_01 before this file.
Inputs are after DecisionKey/DecisionStatus construction and int(level)
normalization. This is the natural mathematical/source-audit layer, not a
Python coercion, Counter implementation or IEEE division refinement.

The program checks actual requested/level/observed keys before counting. No
partition or already-balanced free count vector is an input. Counts are the
six filters of the supplied primary outcomes, preserving source level order.
Participant selection failures and AUC/NLL/aggregation follow separately.
-/

namespace N2P3.RequestedOutcomes

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Status where
  | correct | incorrect | tie | abstain | incomplete | fitFailure
  deriving DecidableEq

def allStatuses : Finset Status :=
  {.correct,.incorrect,.tie,.abstain,.incomplete,.fitFailure}

theorem status_is_one_of_the_six (status : Status) : status ∈ allStatuses := by
  cases status <;> simp [allStatuses]

structure Outcome (Key : Type) where
  key : Key
  level : ℕ
  status : Status
  deriving DecidableEq

def Outcome.pair {Key : Type} (outcome : Outcome Key) : Key × ℕ := (outcome.key,outcome.level)

def observedKeys {Key : Type} (outcomes : List (Outcome Key)) : List (Key × ℕ) :=
  outcomes.map Outcome.pair

def ValidLevels (levels : List ℕ) : Prop :=
  levels ≠ [] ∧ (∀ level ∈ levels, 0 < level) ∧ levels.Nodup

inductive Fault where
  | duplicateRequested | invalidLevels | duplicateOutcomes | missingOrUnexpected
  | invariantCount
  deriving DecidableEq

def validateKeys {Key : Type} [DecidableEq Key] (requested : List Key) (levels : List ℕ)
    (outcomes : List (Outcome Key)) : Except Fault Unit :=
  if requested.Nodup then
    if ValidLevels levels then
      if (observedKeys outcomes).Nodup then
        if (observedKeys outcomes).toFinset=requested.toFinset.product levels.toFinset then .ok ()
        else .error .missingOrUnexpected
      else .error .duplicateOutcomes
    else .error .invalidLevels
  else .error .duplicateRequested

theorem validated_keys_iff_actual_checks {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key)) :
    validateKeys requested levels outcomes=.ok () ↔
      requested.Nodup ∧ ValidLevels levels ∧ (observedKeys outcomes).Nodup ∧
        (observedKeys outcomes).toFinset=requested.toFinset.product levels.toFinset := by
  unfold validateKeys
  split_ifs <;> simp_all

def countAt {Key : Type} (outcomes : List (Outcome Key)) (level : ℕ) (status : Status) : ℕ :=
  (outcomes.filter (fun outcome => decide (outcome.level=level ∧ outcome.status=status))).length

structure Counts where
  correct : ℕ
  incorrect : ℕ
  tie : ℕ
  abstain : ℕ
  incomplete : ℕ
  fitFailure : ℕ
  deriving DecidableEq

def Counts.total (counts : Counts) : ℕ :=
  counts.correct+counts.incorrect+counts.tie+counts.abstain+counts.incomplete+counts.fitFailure

def Counts.valid (counts : Counts) : ℕ := counts.correct+counts.incorrect+counts.tie

def countsAt {Key : Type} (outcomes : List (Outcome Key)) (level : ℕ) : Counts :=
  ⟨countAt outcomes level .correct,countAt outcomes level .incorrect,countAt outcomes level .tie,
   countAt outcomes level .abstain,countAt outcomes level .incomplete,countAt outcomes level .fitFailure⟩

theorem total_counts_every_primary_status {Key : Type} (outcomes : List (Outcome Key)) (level : ℕ) :
    (countsAt outcomes level).total=
      (outcomes.filter (fun outcome => decide (outcome.level=level))).length := by
  induction outcomes with
  | nil => simp [countsAt,countAt,Counts.total]
  | cons outcome tail ih =>
    by_cases same : outcome.level=level
    · cases status : outcome.status <;>
        simp [countsAt,countAt,Counts.total,same,status] at * <;> omega
    · simp [countsAt,countAt,Counts.total,same] at *
      exact ih

theorem projected_level_filter {Key : Type} (outcomes : List (Outcome Key)) (level : ℕ) :
    (outcomes.filter (fun outcome => decide (outcome.level=level))).map Outcome.pair=
      (observedKeys outcomes).filter (fun pair => decide (pair.2=level)) := by
  induction outcomes with
  | nil => rfl
  | cons outcome tail ih =>
    by_cases same : outcome.level=level <;>
      simp [observedKeys,Outcome.pair,same] at * <;> exact ih

theorem requested_pair_membership {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (complete : (observedKeys outcomes).toFinset=requested.toFinset.product levels.toFinset)
    (pair : Key × ℕ) :
    pair ∈ observedKeys outcomes ↔ pair.1 ∈ requested ∧ pair.2 ∈ levels := by
  constructor
  · intro member
    have member : pair ∈ (observedKeys outcomes).toFinset := List.mem_toFinset.mpr member
    rw [complete] at member
    simpa using member
  · intro member
    apply List.mem_toFinset.mp
    rw [complete]
    simpa using member

theorem filtered_key_set_is_requested_single_level {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (complete : (observedKeys outcomes).toFinset=requested.toFinset.product levels.toFinset)
    (level : ℕ) (member : level ∈ levels) :
    ((observedKeys outcomes).filter (fun pair => decide (pair.2=level))).toFinset=
      requested.toFinset.product {level} := by
  ext pair
  simp only [List.mem_toFinset,List.mem_filter,decide_eq_true_eq,
    Finset.product_eq_sprod,Finset.mem_product,Finset.mem_singleton]
  rw [requested_pair_membership requested levels outcomes complete pair]
  constructor
  · rintro ⟨⟨key,_⟩,atLevel⟩
    exact ⟨key,atLevel⟩
  · rintro ⟨key,atLevel⟩
    exact ⟨⟨key,atLevel.symm ▸ member⟩,atLevel⟩

theorem each_checked_level_has_full_requested_length {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (requestedUnique : requested.Nodup) (observedUnique : (observedKeys outcomes).Nodup)
    (complete : (observedKeys outcomes).toFinset=requested.toFinset.product levels.toFinset)
    (level : ℕ) (member : level ∈ levels) :
    (outcomes.filter (fun outcome => decide (outcome.level=level))).length=requested.length := by
  have unique := observedUnique.filter (fun pair => decide (pair.2=level))
  have cardinal := congrArg Finset.card
    (filtered_key_set_is_requested_single_level requested levels outcomes complete level member)
  rw [List.toFinset_card_of_nodup unique] at cardinal
  simp only [Finset.product_eq_sprod,Finset.card_product,Finset.card_singleton,
    Nat.mul_one,List.toFinset_card_of_nodup requestedUnique] at cardinal
  have projected := congrArg List.length (projected_level_filter outcomes level)
  simpa only [List.length_map,cardinal] using projected

theorem checked_counts_preserve_the_requested_denominator {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (accepted : validateKeys requested levels outcomes=.ok ()) (level : ℕ) (member : level ∈ levels) :
    (countsAt outcomes level).total=requested.length := by
  obtain ⟨requestedUnique,_,observedUnique,complete⟩ :=
    (validated_keys_iff_actual_checks requested levels outcomes).mp accepted
  rw [total_counts_every_primary_status]
  exact each_checked_level_has_full_requested_length requested levels outcomes
    requestedUnique observedUnique complete level member

def terminalKeys {Key : Type} [DecidableEq Key] (outcomes : List (Outcome Key))
    (level : ℕ) (status : Status) : Finset Key :=
  ((outcomes.filter (fun outcome => decide (outcome.level=level ∧ outcome.status=status))).map Outcome.key).toFinset

theorem terminal_membership {Key : Type} [DecidableEq Key] (outcomes : List (Outcome Key))
    (level : ℕ) (status : Status) (key : Key) :
    key ∈ terminalKeys outcomes level status ↔
      ∃ outcome ∈ outcomes, outcome.key=key ∧ outcome.level=level ∧ outcome.status=status := by
  simp only [terminalKeys,List.mem_toFinset,List.mem_map,List.mem_filter,decide_eq_true_eq]
  constructor
  · rintro ⟨outcome,⟨member,atLevel,atStatus⟩,atKey⟩
    exact ⟨outcome,member,atKey,atLevel,atStatus⟩
  · rintro ⟨outcome,member,atKey,atLevel,atStatus⟩
    exact ⟨outcome,⟨member,atLevel,atStatus⟩,atKey⟩

theorem different_terminal_statuses_are_disjoint {Key : Type} [DecidableEq Key]
    (outcomes : List (Outcome Key)) (unique : (observedKeys outcomes).Nodup)
    (level : ℕ) (left right : Status) (different : left ≠ right) :
    Disjoint (terminalKeys outcomes level left) (terminalKeys outcomes level right) := by
  apply Finset.disjoint_left.mpr
  intro key leftMember rightMember
  obtain ⟨a,am,ak,al,ast⟩ := (terminal_membership outcomes level left key).mp leftMember
  obtain ⟨b,bm,bk,bl,bst⟩ := (terminal_membership outcomes level right key).mp rightMember
  have pairs : a.pair=b.pair := by simp [Outcome.pair,ak,bk,al,bl]
  have equal := List.inj_on_of_nodup_map unique am bm pairs
  exact different (ast.symm.trans ((congrArg Outcome.status equal).trans bst))

theorem terminal_union_is_entire_request {Key : Type} [DecidableEq Key]
    (requested : List Key) (levels : List ℕ) (outcomes : List (Outcome Key))
    (complete : (observedKeys outcomes).toFinset=requested.toFinset.product levels.toFinset)
    (level : ℕ) (member : level ∈ levels) :
    allStatuses.biUnion (terminalKeys outcomes level)=requested.toFinset := by
  ext key
  constructor
  · intro terminal
    obtain ⟨status,_,keyMember⟩ := Finset.mem_biUnion.mp terminal
    obtain ⟨outcome,present,atKey,atLevel,_⟩ := (terminal_membership outcomes level status key).mp keyMember
    have paired : (key,level) ∈ observedKeys outcomes :=
      List.mem_map.mpr ⟨outcome,present,by simp [Outcome.pair,atKey,atLevel]⟩
    exact List.mem_toFinset.mpr ((requested_pair_membership requested levels outcomes complete _).mp paired).1
  · intro requestedMember
    have paired : (key,level) ∈ observedKeys outcomes :=
      (requested_pair_membership requested levels outcomes complete _).mpr ⟨List.mem_toFinset.mp requestedMember,member⟩
    obtain ⟨outcome,present,equal⟩ := List.mem_map.mp paired
    apply Finset.mem_biUnion.mpr
    refine ⟨outcome.status,status_is_one_of_the_six _,?_⟩
    apply (terminal_membership outcomes level outcome.status key).mpr
    exact ⟨outcome,present,congrArg Prod.fst equal,congrArg Prod.snd equal,rfl⟩

structure LevelCounts where
  level : ℕ
  counts : Counts
  deriving DecidableEq

def countLevels {Key : Type} (requestedCount : ℕ) (outcomes : List (Outcome Key)) :
    List ℕ → Except Fault (List LevelCounts)
  | [] => .ok []
  | level :: rest => do
    let counts := countsAt outcomes level
    if counts.total=requestedCount then
      let rest ← countLevels requestedCount outcomes rest
      pure (⟨level,counts⟩ :: rest)
    else .error .invariantCount

structure Accounting where
  requested : ℕ
  byLevel : List LevelCounts
  deriving DecidableEq

def buildAccounting {Key : Type} [DecidableEq Key] (requested : List Key) (levels : List ℕ)
    (outcomes : List (Outcome Key)) : Except Fault Accounting := do
  validateKeys requested levels outcomes
  let counts ← countLevels requested.length outcomes levels
  pure ⟨requested.length,counts⟩

theorem valid_includes_ties (counts : Counts) :
    counts.valid=counts.correct+counts.incorrect+counts.tie := rfl

theorem natural_count_bounds (counts : Counts) :
    counts.correct ≤ counts.valid ∧ counts.valid ≤ counts.total := by
  simp only [Counts.valid,Counts.total]
  omega

end
end N2P3.RequestedOutcomes
