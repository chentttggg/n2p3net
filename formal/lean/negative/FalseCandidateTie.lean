import N2P3.CandidateTie

open N2P3.CandidateTie

/-- False: a strict mathematical maximum always yields the source prediction. -/
example : winner nearTieScores sourceTolerance = some 1 := by
  rw [nearTie_abstains]
  simp

/-- False: the absolute-tolerance test checks only exact equality of scores. -/
example : ¬ IsTied nearTieScores 1 sourceTolerance := by
  have hb : IsMaximum nearTieScores 1 := by
    intro j
    fin_cases j <;> norm_num [nearTieScores]
  have ht := (tied_iff_close_competitor nearTieScores 1 sourceTolerance_positive.le hb).mpr
    ⟨0, by decide, by norm_num [nearTieScores, sourceTolerance]⟩
  simp only [ht, not_true_eq_false]
