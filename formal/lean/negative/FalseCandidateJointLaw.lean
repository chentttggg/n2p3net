import N2P3.CandidateJointLaw
import Mathlib.Tactic.FinCases

/-!
The concrete Gaussian model supplies real normalized observation laws. The
proper candidate prior puts all mass on index zero. A candidate excluded by
that prior cannot gain posterior mass merely because its likelihood is positive,
and the source's uniform-prior softmax is not this nonuniform-prior conditional.
Each false claim must reduce to exactly one False goal without auxiliary errors.
-/

namespace N2P3.CandidateJointLaw.Negative

open MeasureTheory ProbabilityTheory
open N2P3.MatrixNormalCoordinates

private noncomputable def pointPrior (i : Fin 2) : ℝ := if i = 0 then 1 else 0

private theorem pointPrior_is_legal : CandidateProbability.IsPrior pointPrior := by
  constructor
  · intro i
    fin_cases i <;> norm_num [pointPrior]
  · norm_num [pointPrior, Fin.sum_univ_succ]

private theorem actual_posterior_is_markov :
    IsMarkovKernel (posteriorKernel GaussianCandidates.zeroDesignModel pointPrior) :=
  posteriorKernel_markov GaussianCandidates.zeroDesignModel pointPrior pointPrior_is_legal

private theorem excluded_candidate_zero :
    posteriorKernel GaussianCandidates.zeroDesignModel pointPrior
      (vectorize GaussianCandidates.zeroDesignModel.Y) {1} = 0 := by
  apply zero_prior_posterior_singleton
  norm_num [pointPrior]

/-- False: positive Gaussian likelihood forces posterior mass despite a zero prior. -/
example : 0 < posteriorKernel GaussianCandidates.zeroDesignModel pointPrior
    (vectorize GaussianCandidates.zeroDesignModel.Y) {1} := by
  rw [excluded_candidate_zero]
  norm_num

/-- False: the source softmax is the conditional law for every proper candidate prior. -/
example : ENNReal.ofReal (CandidateProbability.probability GaussianCandidates.zeroDesignModel.evidence 1) =
    posteriorKernel GaussianCandidates.zeroDesignModel pointPrior
      (vectorize GaussianCandidates.zeroDesignModel.Y) {1} := by
  rw [excluded_candidate_zero]
  have hp := ENNReal.ofReal_pos.mpr
    (CandidateProbability.probability_positive GaussianCandidates.zeroDesignModel.evidence 1)
  simp only [(ne_of_gt hp)]

end N2P3.CandidateJointLaw.Negative
