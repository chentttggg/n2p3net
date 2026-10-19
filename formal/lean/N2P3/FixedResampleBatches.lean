import Mathlib.Data.Finset.Prod
import Mathlib.Logic.Function.Iterate
import Mathlib.Data.Nat.Init
import Lean.Elab.Tactic.Omega

/-! Deterministic source batching for paired_interval (cap4000) and each
fixed study's additive draw updates (stride1000). A draw slot has one batch
owner per writer. Multiple studies intentionally update the same slot once
each; global uniqueness of an additive update is not asserted. -/

namespace N2P3.FixedResampleBatches

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local instance] Classical.decEq

def takeCount (total width offset : ℕ) : ℕ := min width (total-offset)

def nextCursor (total width offset : ℕ) : ℕ := offset+takeCount total width offset

def blockContains (total width block position : ℕ) : Prop :=
  block*width ≤ position ∧ position < block*width+takeCount total width (block*width)

def activeBlock (total width block : ℕ) : Prop := block*width<total

theorem source_cursor_step_equals_clamped_stride (total width offset : ℕ) (inside : offset ≤ total) :
    nextCursor total width offset=min (offset+width) total := by
  unfold nextCursor takeCount
  omega

theorem every_active_source_step_is_positive_and_stays_inside_output
    (total width offset : ℕ) (positive : 0<width) (active : offset<total) :
    offset<nextCursor total width offset ∧ nextCursor total width offset≤total := by
  rw [source_cursor_step_equals_clamped_stride total width offset (le_of_lt active)]
  omega

theorem actual_cursor_after_steps_is_the_clamped_block_start (total width steps : ℕ) :
    (nextCursor total width)^[steps] 0=min (steps*width) total := by
  induction steps with
  | zero => simp
  | succ steps ih =>
    rw [Function.iterate_succ_apply',ih,
      source_cursor_step_equals_clamped_stride total width _ (Nat.min_le_right _ _),Nat.succ_mul]
    omega

theorem source_while_cursor_reaches_terminal_output_in_finitely_many_steps
    (total width : ℕ) (positive : 0<width) :
    (nextCursor total width)^[total] 0=total := by
  rw [actual_cursor_after_steps_is_the_clamped_block_start]
  have bound : total ≤ total*width := by simpa using Nat.mul_le_mul_left total positive
  exact Nat.min_eq_right bound

theorem active_while_and_range_schedules_have_the_same_block_start
    (total width block : ℕ) (active : activeBlock total width block) :
    (nextCursor total width)^[block] 0=block*width := by
  rw [actual_cursor_after_steps_is_the_clamped_block_start]
  exact Nat.min_eq_left (le_of_lt active)

theorem actual_tail_count_and_stop_are_bounded (total width offset : ℕ) (inside : offset ≤ total) :
    takeCount total width offset≤width ∧ nextCursor total width offset≤total := by
  exact ⟨Nat.min_le_left _ _,by rw [source_cursor_step_equals_clamped_stride total width offset inside]; exact Nat.min_le_right _ _⟩

theorem every_output_position_has_its_actual_division_block
    (total width position : ℕ) (positive : 0<width) (inside : position<total) :
    activeBlock total width (position/width) ∧ blockContains total width (position/width) position := by
  have startLower := Nat.div_mul_le_self position width
  have remainder := Nat.mod_lt position positive
  have split : position=(position/width)*width+position%width := by
    simpa only [Nat.mul_comm,Nat.add_comm] using (Nat.mod_add_div position width).symm
  have nextUpper : position<(position/width)*width+width := by omega
  refine ⟨by unfold activeBlock; omega,?_⟩
  change (position/width)*width ≤ position ∧ position<nextCursor total width ((position/width)*width)
  rw [source_cursor_step_equals_clamped_stride total width _ (by omega)]
  exact ⟨startLower,lt_min nextUpper inside⟩

theorem distinct_source_blocks_cannot_write_the_same_position
    (total width block position : ℕ) (positive : 0<width)
    (written : blockContains total width block position) : block=position/width := by
  have limited : takeCount total width (block*width)≤width := Nat.min_le_left _ _
  have nextUpper : position<block*width+width := by
    unfold blockContains at written
    omega
  symm
  apply (Nat.div_eq_iff positive).mpr
  exact ⟨written.1,by omega⟩

theorem source_batches_cover_every_slot_exactly_once
    (total width position : ℕ) (positive : 0<width) (inside : position<total) :
    ∃! block,activeBlock total width block ∧ blockContains total width block position := by
  refine ⟨position/width,every_output_position_has_its_actual_division_block total width position positive inside,?_⟩
  intro other written
  exact distinct_source_blocks_cannot_write_the_same_position total width other position positive written.2

def studyUpdateKeys {Study : Type} (studies : Finset Study) (total width position : ℕ) : Finset (Study × ℕ) :=
  studies.product ((Finset.range total).filter (fun block => blockContains total width block position))

theorem per_study_update_keys_are_one_owner_for_each_actual_study {Study : Type}
    (studies : Finset Study) (total width position : ℕ) (positive : 0<width) (inside : position<total) :
    studyUpdateKeys studies total width position=studies.product {position/width} := by
  have blocks : ((Finset.range total).filter (fun block => blockContains total width block position))={position/width} := by
    ext block
    simp only [Finset.mem_filter,Finset.mem_range,Finset.mem_singleton]
    constructor
    · rintro ⟨_,written⟩
      exact distinct_source_blocks_cannot_write_the_same_position total width block position positive written
    · intro owner
      subst block
      exact ⟨lt_of_le_of_lt (Nat.div_le_self _ _) inside,
        (every_output_position_has_its_actual_division_block total width position positive inside).2⟩
  rw [studyUpdateKeys,blocks]

theorem overall_draw_slot_has_one_additive_update_per_study {Study : Type}
    (studies : Finset Study) (total width position : ℕ) (positive : 0<width) (inside : position<total) :
    (studyUpdateKeys studies total width position).card=studies.card := by
  rw [per_study_update_keys_are_one_owner_for_each_actual_study studies total width position positive inside]
  simp

end
end N2P3.FixedResampleBatches
