import N2P3.GaussianPredictive

/-!
Two deliberate false claims: silently discarding cross-block noise correlation
would give mean 1 and row variance 3/2 for the legal correlation-1/2 witness.
Both proofs must end with an unsolved `False` goal.
-/

namespace N2P3.NegativePredictive

open Matrix N2P3.GaussianPredictive

-- Wrong mean: the corrected value is 3/2.
example :
    predictiveMean (regression (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)))
      (1 : Matrix One One ℝ) 1 (2 • 1) (1 : Matrix One One ℝ) = 1 := by
  rw [half_cross_corrected_mean]
  ext i j
  fin_cases i
  fin_cases j
  norm_num

-- Wrong covariance: the corrected value is 7/8.
example :
    predictiveRows (1 : Matrix One One ℝ) ((1 / 2 : ℝ) • (1 : Matrix One One ℝ)) 1
      (1 : Matrix One One ℝ) 1 ((1 / 2 : ℝ) • 1) = (3 / 2 : ℝ) • 1 := by
  rw [half_cross_corrected_rows]
  ext i j
  fin_cases i
  fin_cases j
  norm_num

end N2P3.NegativePredictive
