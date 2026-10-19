import N2P3.CandidateDecisionRisk

open N2P3.CandidateDecisionRisk N2P3.CandidateTie

/-- False: a largest posterior weight guarantees error zero. Both equal
candidates maximize the weights, but their conditional error is one-half. -/
example : scoreError equalScores 0 = 0 := by
  rw [equalScores_error_half]
  norm_num

/-- False: candidates inside the source tolerance necessarily have equal
conditional error. The accepted near-tie witness has a strict score gap. -/
example : scoreError nearTieScores 0 = scoreError nearTieScores 1 := by
  simp only [nearTie_errors_differ]
