import N2P3.GaussianCandidates

open N2P3.GaussianCandidates

/-- False: a legal normalized Gaussian posterior produces a nonpositive
actual predictive likelihood. The concrete model meets every factor premise. -/
example : zeroDesignModel.density 0 ≤ 0 := by
  have hp := zeroDesignModel_actual_likelihood_positive 0
  simp only [not_le.mpr hp]
