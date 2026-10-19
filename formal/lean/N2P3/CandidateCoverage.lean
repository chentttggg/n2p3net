import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Order.Interval.Set.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Sum.Basic
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum
import Lean.Elab.Tactic.Omega
import Mathlib.Tactic.Linarith

/-!
# Source candidate gates, physical coverage, refusal priority and availability

Source extraction: continuous_decision.py:301-306; outcomes.py:73-113;
raw_continuous.py:115-121,206-240. Integers model exact finite membership values;
array shape is the cardinality of the two aligned finite coordinate types.
NaN/array decoding, earlier frozen-model identity checks and numerical scoring
are outside this gate specification.

Coverage inputs already passed canonical-count validation: unique expected and
observed keys, nonnegative counts and no unexpected observed key. The finite
key type is precisely that validated expected vocabulary; `none` denotes an
omitted observed key and is read as zero, as in observed.get(candidate, 0).

The raw builder counts twelve PHYSICAL flash sets, each expected ten times.
The scorer has thirty-six cell hypotheses, each only required to have positive
membership evidence. The two gates are proved distinct using actual finite
row/column schedules, not inferred from successful examples.

Refusal order models the four explicit if/elif branches before sample slicing.
Construction also has separate raw decoder/filter and reused-sample exceptions;
theorems about returned successful views are conditional on that construction
returning normally. `available` is the complete filtered test-scope stop for
every view, including refused views. There is no online current-time gate.
-/

namespace N2P3.CandidateCoverage

open scoped BigOperators

noncomputable section

section Membership

variable {E H : Type*} [Fintype E] [Fintype H]

def Binary (M : E → H → ℤ) : Prop := ∀ e h, M e h = 0 ∨ M e h = 1

def rowCount (M : E → H → ℤ) (e : E) : ℤ := ∑ h, M e h

def columnCount (M : E → H → ℤ) (h : H) : ℤ := ∑ e, M e h

inductive MembershipRefusal where
  | shapeOrNonbinary
  | improperEventSubset
  | missingCandidate
  deriving DecidableEq, Repr

/-- Exact membership branch order after the source's earlier model/data guards. -/
def membershipReason (expectedEvents expectedHypotheses : ℕ) (M : E → H → ℤ) :
    Option MembershipRefusal := by
  classical
  exact if Fintype.card E ≠ expectedEvents ∨ Fintype.card H ≠ expectedHypotheses ∨ ¬Binary M
    then some .shapeOrNonbinary
    else if ∃ e, rowCount M e = 0 ∨ rowCount M e = Fintype.card H
      then some .improperEventSubset
      else if ∃ h, columnCount M h = 0 then some .missingCandidate else none

omit [Fintype E] in
theorem binary_row_bounds (M : E → H → ℤ) (hM : Binary M) (e : E) :
    0 ≤ rowCount M e ∧ rowCount M e ≤ Fintype.card H := by
  constructor
  · apply Finset.sum_nonneg
    intro h _
    rcases hM e h with h | h <;> omega
  · calc
      rowCount M e ≤ ∑ _ : H, (1 : ℤ) := by
        apply Finset.sum_le_sum
        intro h _
        rcases hM e h with h | h <;> omega
      _ = Fintype.card H := by simp

omit [Fintype H] in
theorem binary_column_nonnegative (M : E → H → ℤ) (hM : Binary M) (h : H) :
    0 ≤ columnCount M h := by
  apply Finset.sum_nonneg
  intro e _
  rcases hM e h with hh | hh <;> omega

/-- Binary, nonempty proper event subsets and positive evidence for every
frozen hypothesis are exactly the scorer's membership admission conditions. -/
theorem membership_success_iff (events hypotheses : ℕ) (M : E → H → ℤ) :
    membershipReason events hypotheses M = none ↔
      Fintype.card E = events ∧ Fintype.card H = hypotheses ∧ Binary M ∧
        (∀ e, 0 < rowCount M e ∧ rowCount M e < Fintype.card H) ∧
          ∀ h, 0 < columnCount M h := by
  classical
  unfold membershipReason
  split_ifs with hs hr hc
  · simp only [false_iff]
    intro h
    rcases hs with hs | hs | hs
    · exact hs h.1
    · exact hs h.2.1
    · exact hs h.2.2.1
  · simp only [false_iff]
    intro h
    obtain ⟨e, he⟩ := hr
    have hp := h.2.2.2.1 e
    rcases he with he | he <;> omega
  · simp only [false_iff]
    intro h
    obtain ⟨j, hj⟩ := hc
    have hp := h.2.2.2.2 j
    omega
  · simp only [true_iff]
    have hf := not_or.mp hs
    have hb := not_or.mp hf.2
    have hbinary : Binary M := not_not.mp hb.2
    refine ⟨not_not.mp hf.1, not_not.mp hb.1, hbinary, ?_, ?_⟩
    · intro e
      have hn := binary_row_bounds M hbinary e
      have hx : rowCount M e ≠ 0 ∧ rowCount M e ≠ Fintype.card H := by
        simpa only [not_or] using (not_exists.mp hr e)
      omega
    · intro h
      have hn := binary_column_nonnegative M hbinary h
      have hx : columnCount M h ≠ 0 := not_exists.mp hc h
      omega

theorem accepted_has_shape (events hypotheses : ℕ) (M : E → H → ℤ)
    (h : membershipReason events hypotheses M = none) :
    Fintype.card E = events ∧ Fintype.card H = hypotheses := by
  have hp := (membership_success_iff events hypotheses M).mp h
  exact ⟨hp.1, hp.2.1⟩

end Membership

section Counts

variable {K : Type*} [Fintype K]

def observedCount (observed : K → Option ℕ) (k : K) : ℕ := (observed k).getD 0

def missingKeys (expected : K → ℕ) (observed : K → Option ℕ) : Finset K :=
  Finset.univ.filter (fun k => observedCount observed k < expected k)

def excessKeys (expected : K → ℕ) (observed : K → Option ℕ) : Finset K :=
  Finset.univ.filter (fun k => expected k < observedCount observed k)

def Complete (expected : K → ℕ) (observed : K → Option ℕ) : Prop :=
  missingKeys expected observed = ∅ ∧ excessKeys expected observed = ∅

theorem complete_iff_exact_counts (expected : K → ℕ) (observed : K → Option ℕ) :
    Complete expected observed ↔ ∀ k, observedCount observed k = expected k := by
  simp only [Complete, missingKeys, excessKeys, Finset.filter_eq_empty_iff,
    Finset.mem_univ, forall_true_left]
  constructor
  · intro h k
    have hlo := h.1 (x := k)
    have hhi := h.2 (x := k)
    omega
  · intro h
    exact ⟨fun k => by rw [h k]; omega, fun k => by rw [h k]; omega⟩

omit [Fintype K] in
theorem missing_key_read_as_zero (observed : K → Option ℕ) (k : K) (h : observed k = none) :
    observedCount observed k = 0 := by simp [observedCount, h]

theorem missing_positive_expected_prevents_complete (expected : K → ℕ)
    (observed : K → Option ℕ) (k : K) (hk : 0 < expected k) (ho : observed k = none) :
    ¬Complete expected observed := by
  intro hc
  have he := (complete_iff_exact_counts expected observed).mp hc k
  rw [missing_key_read_as_zero observed k ho] at he
  omega

theorem exact_present_counts_complete (expected : K → ℕ) :
    Complete expected (fun k => some (expected k)) := by
  rw [complete_iff_exact_counts]
  intro k
  rfl

theorem zero_expected_can_be_complete_without_observations :
    Complete (fun _ : K => 0) (fun _ => none) := by
  rw [complete_iff_exact_counts]
  intro k
  rfl

end Counts

section PhysicalRowColumn

abbrev Axis := Fin 6
abbrev Flash := Axis ⊕ Axis
abbrev Cell := Axis × Axis

def physicalCode : Flash → ℕ := Sum.elim Fin.val (fun j => 6 + j.val)

def physicalMembership (f : Flash) (cell : Cell) : ℤ :=
  match f with
  | .inl r => if cell.1 = r then 1 else 0
  | .inr c => if cell.2 = c then 1 else 0

theorem physical_code_in_range (f : Flash) : physicalCode f < 12 := by
  cases f with
  | inl r => have := r.isLt; simp only [physicalCode, Sum.elim_inl]; omega
  | inr c => have := c.isLt; simp only [physicalCode, Sum.elim_inr]; omega

/-- Literal source code convention: row 0..5, column 6..11, cell (row,column). -/
theorem physical_membership_source_formula (f : Flash) (cell : Cell) :
    physicalMembership f cell =
      if physicalCode f = cell.1.val ∨ physicalCode f = 6 + cell.2.val then 1 else 0 := by
  cases f with
  | inl r =>
    have hn : ¬r.val = 6 + cell.2.val := by have := r.isLt; omega
    simp [physicalMembership, physicalCode, hn, Fin.ext_iff, eq_comm]
  | inr c =>
    have hn : ¬6 + c.val = cell.1.val := by have := cell.1.isLt; omega
    simp [physicalMembership, physicalCode, hn, Fin.ext_iff, eq_comm]

variable {E : Type*} [Fintype E]

def physicalMatrix (schedule : E → Flash) : E → Cell → ℤ :=
  fun e => physicalMembership (schedule e)

def flashCount (schedule : E → Flash) (f : Flash) : ℕ :=
  ∑ e, if schedule e = f then 1 else 0

def physicalObserved (schedule : E → Flash) : Flash → Option ℕ :=
  fun f => some (flashCount schedule f)

def PhysicalComplete (schedule : E → Flash) : Prop := Complete (fun _ : Flash => 10) (physicalObserved schedule)

omit [Fintype E] in
theorem physical_binary (schedule : E → Flash) : Binary (physicalMatrix schedule) := by
  intro e cell
  cases hs : schedule e <;> simp only [physicalMatrix, hs, physicalMembership] <;>
    split_ifs <;> simp

omit [Fintype E] in
theorem physical_each_event_has_six (schedule : E → Flash) (e : E) :
    rowCount (physicalMatrix schedule) e = 6 := by
  cases hs : schedule e with
  | inl r =>
    simp only [rowCount, physicalMatrix, hs, physicalMembership]
    rw [Fintype.sum_prod_type]
    have hin (i : Axis) : (∑ _ : Axis, if i = r then (1 : ℤ) else 0) =
        if i = r then 6 else 0 := by
      by_cases h : i = r <;> simp [h]
    simp_rw [hin]
    simp
  | inr c =>
    simp only [rowCount, physicalMatrix, hs, physicalMembership]
    rw [Fintype.sum_prod_type]
    simp

theorem physical_candidate_count (schedule : E → Flash) (cell : Cell) :
    columnCount (physicalMatrix schedule) cell =
      (flashCount schedule (.inl cell.1) + flashCount schedule (.inr cell.2) : ℕ) := by
  simp only [columnCount, flashCount, Nat.cast_add, Nat.cast_sum, Nat.cast_ite,
    Nat.cast_one, Nat.cast_zero, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro e _
  cases hs : schedule e <;> simp [physicalMatrix, hs, physicalMembership, eq_comm]

theorem physical_complete_iff_each_flash_ten (schedule : E → Flash) :
    PhysicalComplete schedule ↔ ∀ f, flashCount schedule f = 10 := by
  exact complete_iff_exact_counts _ _

theorem physical_complete_gives_twenty_per_cell (schedule : E → Flash)
    (h : PhysicalComplete schedule) (cell : Cell) :
    columnCount (physicalMatrix schedule) cell = 20 := by
  rw [physical_candidate_count]
  have he := (physical_complete_iff_each_flash_ten schedule).mp h
  rw [he, he]
  norm_num

theorem physical_complete_implies_scorer_membership (schedule : E → Flash)
    (h : PhysicalComplete schedule) :
    membershipReason (Fintype.card E) 36 (physicalMatrix schedule) = none := by
  rw [membership_success_iff]
  refine ⟨rfl, by decide, physical_binary schedule, ?_, ?_⟩
  · intro e
    rw [physical_each_event_has_six]
    norm_num [Fintype.card_prod]
  · intro cell
    rw [physical_complete_gives_twenty_per_cell schedule h cell]
    norm_num

def onlyRows : Axis → Flash := Sum.inl

theorem only_rows_column_count (cell : Cell) : columnCount (physicalMatrix onlyRows) cell = 1 := by
  rw [physical_candidate_count]
  simp [flashCount, onlyRows]

theorem only_rows_scorer_accepts : membershipReason 6 36 (physicalMatrix onlyRows) = none := by
  rw [membership_success_iff]
  refine ⟨by decide, by decide, physical_binary _, ?_, ?_⟩
  · intro e
    rw [physical_each_event_has_six]
    norm_num [Fintype.card_prod]
  · intro cell
    rw [only_rows_column_count]
    norm_num

theorem only_rows_physically_incomplete : ¬PhysicalComplete onlyRows := by
  rw [physical_complete_iff_each_flash_ten]
  intro h
  have hc := h (.inr (0 : Axis))
  have hz : flashCount onlyRows (.inr (0 : Axis)) = 0 := by
    unfold flashCount
    apply Finset.sum_eq_zero
    intro e _
    rfl
  omega

abbrev BiasedEvents := (Fin 9 ⊕ Fin 11) × Axis

def biasedSchedule : BiasedEvents → Flash :=
  fun e => match e.1 with | .inl _ => .inl e.2 | .inr _ => .inr e.2

theorem biased_flash_counts (r c : Axis) :
    flashCount biasedSchedule (.inl r) = 9 ∧ flashCount biasedSchedule (.inr c) = 11 := by
  constructor <;> unfold flashCount <;> rw [Fintype.sum_prod_type] <;>
    rw [Fintype.sum_sum_type] <;> simp [biasedSchedule]

theorem biased_twenty_per_cell (cell : Cell) :
    columnCount (physicalMatrix biasedSchedule) cell = 20 := by
  rw [physical_candidate_count, (biased_flash_counts cell.1 cell.2).1,
    (biased_flash_counts cell.1 cell.2).2]
  norm_num

theorem biased_scorer_accepts : membershipReason 120 36 (physicalMatrix biasedSchedule) = none := by
  rw [membership_success_iff]
  refine ⟨by decide, by decide, physical_binary _, ?_, ?_⟩
  · intro e
    rw [physical_each_event_has_six]
    norm_num [Fintype.card_prod]
  · intro cell
    rw [biased_twenty_per_cell]
    norm_num

theorem biased_physical_incomplete : ¬PhysicalComplete biasedSchedule := by
  rw [physical_complete_iff_each_flash_ten]
  intro h
  have hc := h (.inl (0 : Axis))
  rw [(biased_flash_counts 0 0).1] at hc
  norm_num at hc

theorem all_eleven_has_no_missing :
    missingKeys (fun _ : Flash => 10) (fun _ => some 11) = ∅ := by
  ext f
  simp [missingKeys, observedCount]

theorem all_eleven_is_incomplete : ¬Complete (fun _ : Flash => 10) (fun _ => some 11) := by
  rw [complete_iff_exact_counts]
  intro h
  have hc := h (.inl (0 : Axis))
  norm_num [observedCount] at hc

end PhysicalRowColumn

section DecisionViews

inductive ViewRefusal where
  | partialCoverage
  | outsideScope
  | previousUnknownOverlap
  | nextUnknownOverlap
  deriving DecidableEq, Repr

structure Geometry where
  first : ℤ
  last : ℤ
  width : ℤ
  preroll : ℤ
  testStart : ℤ
  testStop : ℤ
  previousLast : Option ℤ
  nextFirst : Option ℤ
  deriving Repr

namespace Geometry

def start (g : Geometry) : ℤ := g.first - g.preroll
def stop (g : Geometry) : ℤ := g.last + g.width
def available (g : Geometry) : ℤ := g.testStop

def outside (g : Geometry) : Prop := g.start < g.testStart ∨ g.testStop < g.stop

def previousOverlap (g : Geometry) : Prop :=
  match g.previousLast with | none => False | some prev => g.first - prev < g.width + g.preroll

def nextOverlap (g : Geometry) : Prop :=
  match g.nextFirst with | none => False | some next => next < g.stop

def reason (g : Geometry) (observed : Flash → Option ℕ) : Option ViewRefusal := by
  classical
  exact if ¬Complete (fun _ : Flash => 10) observed then some .partialCoverage
    else if g.outside then some .outsideScope
      else if g.previousOverlap then some .previousUnknownOverlap
        else if g.nextOverlap then some .nextUnknownOverlap else none

theorem partial_priority (g : Geometry) (observed : Flash → Option ℕ) :
    g.reason observed = some .partialCoverage ↔ ¬Complete (fun _ : Flash => 10) observed := by
  classical
  unfold reason
  split_ifs <;> simp_all

theorem outside_priority (g : Geometry) (observed : Flash → Option ℕ) :
    g.reason observed = some .outsideScope ↔ Complete (fun _ : Flash => 10) observed ∧ g.outside := by
  classical
  unfold reason
  split_ifs <;> simp_all

theorem previous_priority (g : Geometry) (observed : Flash → Option ℕ) :
    g.reason observed = some .previousUnknownOverlap ↔
      Complete (fun _ : Flash => 10) observed ∧ ¬g.outside ∧ g.previousOverlap := by
  classical
  unfold reason
  split_ifs <;> simp_all

theorem next_priority (g : Geometry) (observed : Flash → Option ℕ) :
    g.reason observed = some .nextUnknownOverlap ↔
      Complete (fun _ : Flash => 10) observed ∧ ¬g.outside ∧ ¬g.previousOverlap ∧ g.nextOverlap := by
  classical
  unfold reason
  split_ifs <;> simp_all

theorem success_iff (g : Geometry) (observed : Flash → Option ℕ) :
    g.reason observed = none ↔ Complete (fun _ : Flash => 10) observed ∧
      g.testStart ≤ g.start ∧ g.stop ≤ g.testStop ∧ ¬g.previousOverlap ∧ ¬g.nextOverlap := by
  classical
  unfold reason
  split_ifs <;> simp_all [outside]
  omega

theorem successful_scope_and_availability (g : Geometry) (observed : Flash → Option ℕ)
    (h : g.reason observed = none) :
    g.testStart ≤ g.start ∧ g.stop ≤ g.available ∧ g.available = g.testStop := by
  have hp := (success_iff g observed).mp h
  exact ⟨hp.2.1, hp.2.2.1, rfl⟩

theorem successful_sample_support (g : Geometry) (observed : Flash → Option ℕ)
    (h : g.reason observed = none) : Set.Ico g.start g.stop ⊆ Set.Ico g.testStart g.testStop := by
  have hp := (success_iff g observed).mp h
  intro x hx
  exact ⟨le_trans hp.2.1 hx.1, lt_of_lt_of_le hx.2 hp.2.2.1⟩

theorem no_previous_overlap_orders_response (g : Geometry) (prev : ℤ)
    (hp : g.previousLast = some prev) (h : ¬g.previousOverlap) :
    prev + g.width ≤ g.start := by
  simp only [previousOverlap, hp] at h
  unfold start
  omega

theorem no_next_overlap_orders_response (g : Geometry) (next : ℤ)
    (hp : g.nextFirst = some next) (h : ¬g.nextOverlap) : g.stop ≤ next := by
  simpa only [nextOverlap, hp, not_lt] using h

end Geometry

/-- Equality at all support/neighbor endpoints is allowed: all source guards
are strict inequalities. All four fail conditions can also be simultaneously
true, but incomplete physical coverage has the first refusal priority. -/
def boundaryGeometry : Geometry := ⟨10, 20, 2, 1, 9, 22, some 7, some 22⟩

theorem support_boundary_equalities_are_accepted :
    boundaryGeometry.reason (fun _ => some 10) = none := by
  rw [Geometry.success_iff]
  refine ⟨exact_present_counts_complete _, ?_⟩
  norm_num [boundaryGeometry, Geometry.start, Geometry.stop,
    Geometry.previousOverlap, Geometry.nextOverlap]

def delayedGeometry : Geometry := ⟨10, 20, 2, 1, 0, 100, none, none⟩

def allFailureGeometry : Geometry := ⟨10, 20, 2, 1, 0, 21, some 9, some 21⟩

theorem all_failures_keep_coverage_priority :
    allFailureGeometry.outside ∧ allFailureGeometry.previousOverlap ∧ allFailureGeometry.nextOverlap ∧
      allFailureGeometry.reason (fun _ => some 0) = some .partialCoverage := by
  have hc : ¬Complete (fun _ : Flash => 10) (fun _ => some 0) := by
    rw [complete_iff_exact_counts]
    intro h
    have he := h (.inl (0 : Axis))
    norm_num [observedCount] at he
  refine ⟨?_, ?_, ?_, (Geometry.partial_priority _ _).mpr hc⟩ <;>
    norm_num [allFailureGeometry, Geometry.outside, Geometry.start, Geometry.stop,
      Geometry.previousOverlap, Geometry.nextOverlap]

/-- A view may pass every source gate long before its full filtered input is
available. The source records available=whole test stop; it has no now gate. -/
theorem accepted_view_can_have_later_availability :
    delayedGeometry.reason (fun _ => some 10) = none ∧
      delayedGeometry.stop < delayedGeometry.available := by
  constructor
  · rw [Geometry.success_iff]
    refine ⟨exact_present_counts_complete _, ?_⟩
    norm_num [delayedGeometry, Geometry.start, Geometry.stop,
      Geometry.previousOverlap, Geometry.nextOverlap]
  · norm_num [delayedGeometry, Geometry.stop, Geometry.available]

end DecisionViews

end

end N2P3.CandidateCoverage
