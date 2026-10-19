import N2P3.TemporalWhitening

/-! Six source-derived errors: retained rows, diagnostic scale, fresh startup,
and the actual simultaneous forward/backward lattice recursion. -/

namespace N2P3.NegativeTemporalWhitening

open N2P3.TemporalWhitening

-- Conditional mode excludes exactly the observed p-sample history prefix.
example : (conditional [3/5] 8 fixtureOnes).length = 8 := by
  rw [conditional_output_length]
  have hn : (8-([3/5] : List ℝ).length) ≠ 8 := by decide
  simp only [hn]

-- Stationary innovations retains every prefix row; it is not conditional mode.
example : (lattice [1/2,1/4] 5 twoStageFixture).length = 3 := by
  rw [lattice_output_length _ _ _ (by norm_num)]
  have hn : (5 : ℕ) ≠ 3 := by decide
  simp only [hn]

-- Changing a positive diagnostic variance does not rescale the observations.
example : whiten .stationary [3/5] [-3/5] fixtureRoot 2 8 fixtureOnes ≠
    whiten .stationary [3/5] [-3/5] fixtureRoot 19 8 fixtureOnes := by
  have he := diagnostic_variance_does_not_rescale .stationary
    [3/5] [-3/5] fixtureRoot 2 19 8 fixtureOnes
  simp only [he, ne_eq, not_true_eq_false]

-- Nonempty stationary startup uses the unit-innovation root, not the raw row.
example : rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 0 0 = 1 := by
  rw [nonempty_stationary_first_row]
  have hn : (4/5 : ℝ) ≠ 1 := by norm_num
  simp only [hn]

-- Joining a previous record changes startup into a continuation innovation.
example : rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 0 0 =
    rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 4 0 := by
  simp only [fresh_start_differs_from_record_continuation]

-- Omitting k*forward from next_backward loses the stage interaction term.
example : rowAt (lattice [1/2,1/4] 5 twoStageFixture) 2 0 = 17/4 := by
  rw [nonempty_two_stage_tail_value]
  have hn : (9/2 : ℝ) ≠ 17/4 := by norm_num
  simp only [hn]

end N2P3.NegativeTemporalWhitening
