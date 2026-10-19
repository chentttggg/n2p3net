import N2P3.ERPFeatures

/-! Six discriminating errors in mean normalization, complete-bin support,
channel-prime coordinates, affine intercept and actual temporal projection. -/

namespace N2P3.NegativeERPFeatures

open N2P3.ERPFeatures

-- Channel-major flatten gives 50; the source's time-major flatten gives 53.
example : evaluate fixtureState (binFeatures 2 fixtureEpoch)=50 := by
  rw [fixture_saved_score_53]
  have hn : (53 : ℝ) ≠ 50 := by norm_num
  simp only [hn]

-- The discarded fifth sample never becomes a shorter third bin.
example : retainedSamples 5 2=5 := by
  rw [fixture_discards_one_sample.1]
  have hn : (4 : ℕ) ≠ 5 := by decide
  simp only [hn]

-- Both complete finite inputs pass the same shape; editing their unused tail is irrelevant.
example : evaluate fixtureState (binFeatures 2 changedFiniteTail) ≠
    evaluate fixtureState (binFeatures 2 fixtureEpoch) := by
  rw [fixture_finite_tail_change_keeps_score,fixture_saved_score_53]
  simp only [ne_eq,not_true_eq_false]

-- Bin mean is not the unnormalized sum of its two retained samples.
example : completeBins 2 fixtureEpoch 0 0=4 := by
  rw [fixture_bin_values]
  change (2 : ℝ)=4
  have hn : (2 : ℝ) ≠ 4 := by norm_num
  simp only [hn]

-- A nonzero intercept prevents linear scaling of the complete score.
example : evaluate fixtureState (binFeatures 2 (combineEpoch 1 1 fixtureEpoch fixtureEpoch))=
    2*evaluate fixtureState (binFeatures 2 fixtureEpoch) := by
  rw [fixture_doubled_signal_score_107,fixture_saved_score_53]
  have hn : (107 : ℝ) ≠ 2*53 := by norm_num
  simp only [hn]

-- This exact orthonormal basis selects the second time sample, not the first.
example : evaluate temporalFixtureState (temporalFeatures temporalFixtureBasis temporalFixtureEpoch)=12 := by
  rw [nonempty_temporal_projection_score_17]
  have hn : (17 : ℝ) ≠ 12 := by norm_num
  simp only [hn]

end N2P3.NegativeERPFeatures
