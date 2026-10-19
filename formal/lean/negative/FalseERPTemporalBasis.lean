import N2P3.ERPTemporalBasis

/-! Independent false claims about tolerance, normalization, weighting and ties. -/

namespace N2P3.NegativeERPTemporalBasis

open N2P3.ERPTemporalBasis

-- The actual tolerance predicate admits a basis whose Gram is not exactly I.
example : approximateOrthonormal nearlyUnitBasis →
    gram nearlyUnitBasis = (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  intro _
  simp only [nearly_unit_not_exactly_orthogonal]

-- Trace normalization removes the raw contrast energy; it does not preserve 4.
example : (normalizedGram (fixtureContrast 0)).trace = 4 := by
  rw [normalized_gram_trace _ (fixture_positive_energy 0)]
  have hn : (1 : ℝ) ≠ 4 := by norm_num
  simp only [hn]

-- Source weighting is per participant; normalizing pooled raw Grams gives 4/22.
example : participantAverage fixtureChannels fixtureContrast 0 0 = 2/11 := by
  rw [fixture_source_average]
  change (1/2 : ℝ) = 2/11
  have hn : (1/2 : ℝ) ≠ 2/11 := by norm_num
  simp only [hn]

-- Positive equal eigenvalues and fixed positive signs do not choose a unique axis.
example : unitAxis 0 = unitAxis 1 := by
  simp only [tied_axes_are_distinct]

end N2P3.NegativeERPTemporalBasis
