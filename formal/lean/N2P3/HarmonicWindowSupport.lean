import N2P3.PreprocessingFitScope
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Positivity
import Lean.Elab.Tactic.Omega

/-!
# Coverage and strictly positive denominator of source harmonic overlap-add

harmonic.py:170-188 takes regular grid starts, appends the final tail when
needed, uses hanning(width+2)[1:-1], and normalizes the weighted sum. This
finite support representation forgets start order but preserves exact start
membership and tail coverage. The mathematical Hann expression is separated
from NumPy bytes/rounding. The integer hop bounds are supplied from the source
window/overlap recipe, not an assumed completed denominator-success result.
-/
namespace N2P3.HarmonicWindowSupport
open scoped BigOperators
noncomputable section

def windowStarts (length width hop : ℕ) : Finset ℕ :=
  insert (length-width) ((Finset.range (length-width+1)).filter (fun start => start % hop=0))

theorem regular_grid_plus_final_tail_covers_every_source_sample
    (length width hop sample : ℕ) (widthFits : width ≤ length)
    (hopPositive : 0 < hop) (noGap : hop ≤ width) (inside : sample < length) :
    ∃ start ∈ windowStarts length width hop, start ≤ sample ∧ sample < start+width := by
  have baseBefore : sample/hop*hop ≤ sample := Nat.div_mul_le_self sample hop
  have remainder : sample % hop < hop := Nat.mod_lt sample hopPositive
  have decomposition : sample % hop+sample/hop*hop=sample := by
    simpa only [Nat.mul_comm] using Nat.mod_add_div sample hop
  by_cases regular : sample/hop*hop ≤ length-width
  · refine ⟨sample/hop*hop,?_,baseBefore,?_⟩
    · simp only [windowStarts,Finset.mem_insert,Finset.mem_filter,Finset.mem_range]
      exact Or.inr ⟨by omega,by simp⟩
    · omega
  · refine ⟨length-width,?_,?_,?_⟩
    · exact Finset.mem_insert_self _ _
    · omega
    · omega

def interiorHann (width offset : ℕ) : ℝ :=
  Real.sin (Real.pi*((offset : ℝ)+1)/((width : ℝ)+1))^2

theorem interior_Hann_is_the_source_cosine_window (width offset : ℕ) :
    interiorHann width offset=
      1/2-1/2*Real.cos (2*(Real.pi*((offset : ℝ)+1)/((width : ℝ)+1))) := by
  unfold interiorHann
  rw [Real.cos_two_mul]
  nlinarith [Real.sin_sq_add_cos_sq (Real.pi*((offset : ℝ)+1)/((width : ℝ)+1))]

theorem every_interior_Hann_weight_is_strictly_positive
    (width offset : ℕ) (inside : offset < width) : 0 < interiorHann width offset := by
  have denominator : (0 : ℝ) < (width : ℝ)+1 := by positivity
  have numerator : (0 : ℝ) < (offset : ℝ)+1 := by positivity
  have relative : (offset : ℝ)+1 < (width : ℝ)+1 := by exact_mod_cast Nat.add_lt_add_right inside 1
  apply sq_pos_of_pos
  apply Real.sin_pos_of_pos_of_lt_pi
  · exact div_pos (mul_pos Real.pi_pos numerator) denominator
  · exact (div_lt_iff₀ denominator).2 (mul_lt_mul_of_pos_left relative Real.pi_pos)

def weightAt (width sample start : ℕ) : ℝ :=
  if start ≤ sample ∧ sample < start+width then interiorHann width (sample-start) else 0

theorem source_window_weights_are_nonnegative (width sample start : ℕ) : 0 ≤ weightAt width sample start := by
  unfold weightAt
  split_ifs
  · exact sq_nonneg _
  · exact le_refl _

theorem source_coverage_gives_a_strictly_positive_overlap_denominator
    (length width hop sample : ℕ) (widthFits : width ≤ length)
    (hopPositive : 0 < hop) (noGap : hop ≤ width) (inside : sample < length) :
    0 < ∑ start ∈ windowStarts length width hop, weightAt width sample start := by
  obtain ⟨start,member,before,after⟩ := regular_grid_plus_final_tail_covers_every_source_sample
    length width hop sample widthFits hopPositive noGap inside
  apply Finset.sum_pos'
  · intro s _
    exact source_window_weights_are_nonnegative width sample s
  · refine ⟨start,member,?_⟩
    rw [weightAt,if_pos ⟨before,after⟩]
    exact every_interior_Hann_weight_is_strictly_positive width (sample-start) (by omega)

theorem the_fixed_source_tail_really_is_needed_for_the_last_sample :
    19 ∈ windowStarts 35 16 16 ∧ 19 ≤ 34 ∧ 34 < 19+16 ∧ ¬(34 < 16+16) := by
  norm_num [windowStarts]

end
end N2P3.HarmonicWindowSupport
