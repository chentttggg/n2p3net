import N2P3.ObservedStatistics
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Lean.Elab.Tactic.Omega

/-!
# The literal calibration helper population-moment merge

Source: splitbn.py:638-714. A site's observation count counts all batch and
nonfeature coordinates. The helper uses local population variance, then the
literal delta*delta_after*new_count update, and divides by total observations.

This proves moments of the activations actually captured. It does not identify
that population with a separately masked caller forward or with activations
after upstream target buffers have been installed. Exact tensor reductions and
casts remain per-operation interfaces. The source-first contract is
v07_calibration_population_contract_01.json.

The source-first contract retains its original read identity; current checks
separately pin the active source set. No production change is part of this file.
-/
namespace N2P3.CalibrationPopulation
open scoped BigOperators
noncomputable section

structure Summary where
  count : ℕ
  mean : ℝ
  m2 : ℝ

def rawSum (summary : Summary) : ℝ := summary.count * summary.mean
def rawSquares (summary : Summary) : ℝ := summary.m2 + summary.count * summary.mean ^ 2
def populationVariance (summary : Summary) : ℝ := summary.m2 / summary.count

/-- The exact source arithmetic after a previous block has been recorded. -/
def merge (previous block : Summary) : Summary :=
  let total := previous.count + block.count
  let delta := block.mean - previous.mean
  let updatedMean := previous.mean + delta * ((block.count : ℝ) / total)
  let deltaAfter := block.mean - updatedMean
  ⟨total, updatedMean, previous.m2 + block.m2 + delta * deltaAfter * block.count⟩

def sourceBlock (count : ℕ) (mean variance : ℝ) : Summary :=
  ⟨count, mean, variance * count⟩

def append (previous : Option Summary) (block : Summary) : Summary :=
  match previous with
  | none => block
  | some summary => merge summary block

theorem first_block_keeps_its_local_population_m2 (count : ℕ) (mean variance : ℝ) :
    append none (sourceBlock count mean variance) = ⟨count, mean, variance * count⟩ := rfl

theorem merged_count_is_the_sum_of_observation_counts (previous block : Summary) :
    (merge previous block).count = previous.count + block.count := rfl

theorem literal_mean_merge_preserves_actual_raw_sum (previous block : Summary)
    (positiveTotal : 0 < previous.count + block.count) :
    rawSum (merge previous block) = rawSum previous + rawSum block := by
  have nonzero : (previous.count : ℝ) + block.count ≠ 0 := by
    exact_mod_cast (ne_of_gt positiveTotal)
  dsimp [rawSum, merge]
  simp only [Nat.cast_add]
  field_simp [nonzero]
  ring

theorem literal_m2_merge_preserves_actual_raw_squared_sum (previous block : Summary)
    (positiveTotal : 0 < previous.count + block.count) :
    rawSquares (merge previous block) = rawSquares previous + rawSquares block := by
  have nonzero : (previous.count : ℝ) + block.count ≠ 0 := by
    exact_mod_cast (ne_of_gt positiveTotal)
  dsimp [rawSquares, merge]
  simp only [Nat.cast_add]
  field_simp [nonzero]
  ring

theorem source_m2_is_within_plus_between_population_dispersion (previous block : Summary)
    (positiveTotal : 0 < previous.count + block.count) :
    (merge previous block).m2 = previous.m2 + block.m2 +
      (block.mean - previous.mean) ^ 2 * previous.count * block.count /
        ((previous.count : ℝ) + block.count) := by
  have nonzero : (previous.count : ℝ) + block.count ≠ 0 := by
    exact_mod_cast (ne_of_gt positiveTotal)
  dsimp [merge]
  simp only [Nat.cast_add]
  field_simp [nonzero]
  ring

theorem nonnegative_local_m2_gives_nonnegative_merged_m2 (previous block : Summary)
    (positiveTotal : 0 < previous.count + block.count)
    (previousNonnegative : 0 ≤ previous.m2) (blockNonnegative : 0 ≤ block.m2) :
    0 ≤ (merge previous block).m2 := by
  rw [source_m2_is_within_plus_between_population_dispersion previous block positiveTotal]
  exact add_nonneg (add_nonneg previousNonnegative blockNonnegative) (by positivity)

theorem observation_count_and_raw_moments_determine_a_unique_summary
    (first second : Summary) (positiveCount : 0 < second.count)
    (sameCount : first.count = second.count) (sameSum : rawSum first = rawSum second)
    (sameSquares : rawSquares first = rawSquares second) : first = second := by
  have nonzero : (second.count : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt positiveCount)
  have sameMean : first.mean = second.mean := by
    apply mul_left_cancel₀ nonzero
    simpa only [rawSum, sameCount] using sameSum
  have sameM2 : first.m2 = second.m2 := by
    dsimp only [rawSquares] at sameSquares
    rw [sameCount, sameMean] at sameSquares
    exact add_right_cancel sameSquares
  cases first
  cases second
  simp_all

theorem nonempty_block_order_does_not_change_the_merged_population
    (first second : Summary) (firstPositive : 0 < first.count) (secondPositive : 0 < second.count) :
    merge first second = merge second first := by
  have firstTotal : 0 < first.count + second.count := by omega
  have secondTotal : 0 < second.count + first.count := by omega
  apply observation_count_and_raw_moments_determine_a_unique_summary _ _ secondTotal
  · simp only [merged_count_is_the_sum_of_observation_counts, Nat.add_comm]
  · rw [literal_mean_merge_preserves_actual_raw_sum first second firstTotal,
      literal_mean_merge_preserves_actual_raw_sum second first secondTotal]
    exact add_comm _ _
  · rw [literal_m2_merge_preserves_actual_raw_squared_sum first second firstTotal,
      literal_m2_merge_preserves_actual_raw_squared_sum second first secondTotal]
    exact add_comm _ _

theorem nonempty_block_parenthesization_does_not_change_the_merged_population
    (first second third : Summary) (firstPositive : 0 < first.count)
    (secondPositive : 0 < second.count) (thirdPositive : 0 < third.count) :
    merge (merge first second) third = merge first (merge second third) := by
  have ab : 0 < first.count + second.count := by omega
  have bc : 0 < second.count + third.count := by omega
  have leftTotal : 0 < (merge first second).count + third.count := by
    simp only [merged_count_is_the_sum_of_observation_counts]
    omega
  have rightTotal : 0 < first.count + (merge second third).count := by
    simp only [merged_count_is_the_sum_of_observation_counts]
    omega
  apply observation_count_and_raw_moments_determine_a_unique_summary _ _ rightTotal
  · simp only [merged_count_is_the_sum_of_observation_counts, Nat.add_assoc]
  · rw [literal_mean_merge_preserves_actual_raw_sum _ _ leftTotal,
      literal_mean_merge_preserves_actual_raw_sum _ _ rightTotal,
      literal_mean_merge_preserves_actual_raw_sum _ _ ab,
      literal_mean_merge_preserves_actual_raw_sum _ _ bc]
    exact add_assoc _ _ _
  · rw [literal_m2_merge_preserves_actual_raw_squared_sum _ _ leftTotal,
      literal_m2_merge_preserves_actual_raw_squared_sum _ _ rightTotal,
      literal_m2_merge_preserves_actual_raw_squared_sum _ _ ab,
      literal_m2_merge_preserves_actual_raw_squared_sum _ _ bc]
    exact add_assoc _ _ _

theorem final_population_variance_is_the_raw_second_moment_minus_squared_mean
    (summary : Summary) (positiveCount : 0 < summary.count) :
    populationVariance summary = rawSquares summary / summary.count - summary.mean ^ 2 := by
  have nonzero : (summary.count : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt positiveCount)
  dsimp [populationVariance, rawSquares]
  field_simp [nonzero]
  ring

def runChunks (first : Summary) (remaining : List Summary) : Summary :=
  remaining.foldl merge first

theorem every_nonempty_chunk_stream_preserves_count_and_both_raw_moments
    (first : Summary) (remaining : List Summary) (firstPositive : 0 < first.count)
    (blocksPositive : ∀ block ∈ remaining, 0 < block.count) :
    (runChunks first remaining).count = first.count + (remaining.map Summary.count).sum ∧
      rawSum (runChunks first remaining) = rawSum first + (remaining.map rawSum).sum ∧
      rawSquares (runChunks first remaining) = rawSquares first + (remaining.map rawSquares).sum := by
  induction remaining generalizing first with
  | nil => simp [runChunks]
  | cons block remaining ih =>
    have blockPositive : 0 < block.count := blocksPositive block (by simp)
    have totalPositive : 0 < first.count + block.count := by omega
    have remainingPositive : ∀ other ∈ remaining, 0 < other.count := by
      intro other membership
      exact blocksPositive other (by simp [membership])
    obtain ⟨countIdentity, sumIdentity, squaresIdentity⟩ :=
      ih (merge first block) totalPositive remainingPositive
    refine ⟨?_, ?_, ?_⟩
    · simpa [runChunks, merged_count_is_the_sum_of_observation_counts, Nat.add_assoc] using countIdentity
    · rw [literal_mean_merge_preserves_actual_raw_sum first block totalPositive] at sumIdentity
      simpa [runChunks, add_assoc] using sumIdentity
    · rw [literal_m2_merge_preserves_actual_raw_squared_sum first block totalPositive] at squaresIdentity
      simpa [runChunks, add_assoc] using squaresIdentity

theorem summary_mean_is_its_actual_raw_sum_divided_by_observation_count
    (summary : Summary) (positiveCount : 0 < summary.count) :
    summary.mean = rawSum summary / summary.count := by
  have nonzero : (summary.count : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt positiveCount)
  dsimp [rawSum]
  field_simp [nonzero]

theorem final_stream_population_variance_uses_all_observations_and_raw_moments
    (first : Summary) (remaining : List Summary) (firstPositive : 0 < first.count)
    (blocksPositive : ∀ block ∈ remaining, 0 < block.count) :
    populationVariance (runChunks first remaining) =
      (rawSquares first + (remaining.map rawSquares).sum) /
          (first.count + (remaining.map Summary.count).sum : ℕ) -
        ((rawSum first + (remaining.map rawSum).sum) /
          (first.count + (remaining.map Summary.count).sum : ℕ)) ^ 2 := by
  obtain ⟨countIdentity, sumIdentity, squaresIdentity⟩ :=
    every_nonempty_chunk_stream_preserves_count_and_both_raw_moments first remaining firstPositive blocksPositive
  have finalPositive : 0 < (runChunks first remaining).count := by
    rw [countIdentity]
    omega
  rw [final_population_variance_is_the_raw_second_moment_minus_squared_mean _ finalPositive,
    summary_mean_is_its_actual_raw_sum_divided_by_observation_count _ finalPositive,
    countIdentity, sumIdentity, squaresIdentity]

section ActualLocalPopulation
variable {ι : Type*} [Fintype ι] [Nonempty ι]

def actualMean (values : ι → ℝ) : ℝ := (∑ index, values index) / Fintype.card ι
def actualCenteredM2 (values : ι → ℝ) (center : ℝ) : ℝ :=
  ∑ index, (values index - center) ^ 2
def actualVariance (values : ι → ℝ) : ℝ :=
  actualCenteredM2 values (actualMean values) / Fintype.card ι
def actualSourceBlock (values : ι → ℝ) : Summary :=
  sourceBlock (Fintype.card ι) (actualMean values) (actualVariance values)

omit [Nonempty ι] in
theorem centered_m2_expansion_for_actual_finite_coordinates (values : ι → ℝ) (center : ℝ) :
    actualCenteredM2 values center = (∑ index, (values index) ^ 2) -
      2 * center * (∑ index, values index) + Fintype.card ι * center ^ 2 := by
  unfold actualCenteredM2
  calc
    _ = ∑ index, ((values index) ^ 2 - (2 * center) * values index + center ^ 2) := by
      apply Finset.sum_congr rfl
      intro index _
      ring
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
      simp

theorem actual_local_mean_recovers_the_observation_sum (values : ι → ℝ) :
    rawSum (actualSourceBlock values) = ∑ index, values index := by
  have nonzero : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  dsimp [rawSum, actualSourceBlock, sourceBlock, actualMean]
  field_simp [nonzero]

theorem actual_first_population_variance_recovers_centered_m2 (values : ι → ℝ) :
    (actualSourceBlock values).m2 = actualCenteredM2 values (actualMean values) := by
  have nonzero : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  dsimp [actualSourceBlock, sourceBlock, actualVariance]
  exact div_mul_cancel₀ _ nonzero

theorem actual_local_mean_and_variance_recover_the_squared_observation_sum (values : ι → ℝ) :
    rawSquares (actualSourceBlock values) = ∑ index, (values index) ^ 2 := by
  have sumIdentity := actual_local_mean_recovers_the_observation_sum values
  change (Fintype.card ι : ℝ) * actualMean values = ∑ index, values index at sumIdentity
  unfold rawSquares
  rw [actual_first_population_variance_recovers_centered_m2,
    centered_m2_expansion_for_actual_finite_coordinates, ← sumIdentity]
  change (∑ index, values index ^ 2) -
      2 * actualMean values * ((Fintype.card ι : ℝ) * actualMean values) +
      Fintype.card ι * actualMean values ^ 2 + Fintype.card ι * actualMean values ^ 2 = _
  ring

theorem actual_local_population_m2_is_nonnegative (values : ι → ℝ) :
    0 ≤ (actualSourceBlock values).m2 := by
  rw [actual_first_population_variance_recovers_centered_m2]
  exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)

end ActualLocalPopulation

def separatedFirst : Summary := sourceBlock 2 1 1
def separatedSecond : Summary := sourceBlock 2 9 1

theorem separated_block_means_create_between_population_variance :
    (merge separatedFirst separatedSecond).mean = 5 ∧
      populationVariance (merge separatedFirst separatedSecond) = 17 ∧
      ((1 + 1) / 2 : ℝ) = 1 := by
  norm_num [merge, sourceBlock, separatedFirst, separatedSecond, populationVariance]

theorem unequal_blocks_require_observation_weighted_mean_and_population_denominator :
    (merge (sourceBlock 2 1 1) (sourceBlock 6 5 1)).mean = 4 ∧
      populationVariance (merge (sourceBlock 2 1 1) (sourceBlock 6 5 1)) = 4 ∧
      ((1 + 5) / 2 : ℝ) = 3 ∧ ((32 : ℝ) / (8 - 1)) = 32 / 7 := by
  norm_num [merge, sourceBlock, populationVariance]

/-- The absent trace has already been zero-filled by the caller. Counting it
as observed still changes the population. This is the first temporal BN
identity-kernel fixture; it is not a runtime claim about an executed helper. -/
theorem omitting_the_observed_trace_mask_changes_the_population_variance :
    (((-1 : ℝ) ^ 2 + 1 ^ 2) / 2) = 1 ∧
      (((-1 : ℝ) ^ 2 + 1 ^ 2 + 0 ^ 2 + 0 ^ 2) / 4) = 1 / 2 ∧
      (1 : ℝ) ≠ 1 / 2 := by norm_num

end
end N2P3.CalibrationPopulation
