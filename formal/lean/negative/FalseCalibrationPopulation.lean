import N2P3.CalibrationPopulation

namespace N2P3.NegativeCalibrationPopulation
open N2P3.CalibrationPopulation

-- Dropping between-block mean dispersion is not the source population law.
example : populationVariance (merge separatedFirst separatedSecond) = 1 := by
  rw [separated_block_means_create_between_population_variance.2.1]
  norm_num

-- Batches of unequal size do not receive equal mean weight.
example : (merge (sourceBlock 2 1 1) (sourceBlock 6 5 1)).mean = 3 := by
  rw [unequal_blocks_require_observation_weighted_mean_and_population_denominator.1]
  norm_num

-- The source returns the population denominator, not n-1.
example : populationVariance (merge (sourceBlock 2 1 1) (sourceBlock 6 5 1)) = 32 / 7 := by
  rw [unequal_blocks_require_observation_weighted_mean_and_population_denominator.2.1]
  norm_num

-- Two captured traces with eight time samples are sixteen observations.
def actualTracePopulation : Fin 2 × Fin 8 → ℝ :=
  fun coordinate => if coordinate.2.val % 2 = 0 then -1 else 1

example : (actualSourceBlock actualTracePopulation).count = 2 := by
  norm_num [actualSourceBlock, sourceBlock]

end N2P3.NegativeCalibrationPopulation
