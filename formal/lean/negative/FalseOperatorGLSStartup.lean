import N2P3.OperatorGLSStartup

/-! Six source-fixed wrong alternatives. All record and channel sets are
inhabited; no impossible input hypotheses or extra computation axioms. -/
namespace N2P3.NegativeOperatorGLSStartup
open N2P3.OperatorGLSStartup N2P3.TemporalWhitening
noncomputable section

example : rows conditionalFixture 4 = 4 := by
  rw [conditional_fixture_retains_real_history_and_shortens_rows.1]
  norm_num

example : white conditionalFixture 4 historySignal fixtureRow 0 = 3 := by
  rw [conditional_fixture_retains_real_history_and_shortens_rows.2]
  norm_num

example : originalNuisance 4 1 1 = originalNuisance 3 0 1 := by
  rw [original_nuisance_grid_differs_from_rebuilt_conditional_grid.1,
    original_nuisance_grid_differs_from_rebuilt_conditional_grid.2]
  norm_num

example : white {conditionalFixture with diagnosticVariance := 4} 4 historySignal fixtureRow 0 = 1/2 := by
  have unchanged : white {conditionalFixture with diagnosticVariance := 4} 4 historySignal fixtureRow 0 =
      white conditionalFixture 4 historySignal fixtureRow 0 := rfl
  rw [unchanged, conditional_fixture_retains_real_history_and_shortens_rows.2]
  norm_num

example : effectiveLambda .innovation priorFixture (25/8) = effectiveLambda .marginal priorFixture (25/8) := by
  rw [changing_prior_convention_changes_lambda_with_same_noise.1,
    changing_prior_convention_changes_lambda_with_same_noise.2]
  norm_num

example : rowAt (stationary [3/5] fixtureRoot 4 fixtureOnes) 0 0 =
    rowAt (stationary [3/5] fixtureRoot 8 fixtureOnes) 4 0 := by
  have first : rowAt (stationary [3/5] fixtureRoot 4 fixtureOnes) 0 0 = 4/5 := by
    rw [stationary_prefix_coordinate _ _ _ _ _ _ (by norm_num), prefixSolve_zero]
    norm_num [fixtureRoot, fixtureOnes]
  rw [first, nonempty_stationary_interior_row]
  norm_num

end
end N2P3.NegativeOperatorGLSStartup
