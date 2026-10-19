import N2P3.ObservedStatistics

/- Wrong alternatives are derived from the measured-only and absent-channel
source equations fixed before the positive module, not from a source run. -/
namespace N2P3.NegativeObservedStatistics
open N2P3.ObservedStatistics

example : maskedMean exampleMask exampleValues = unmaskedMean exampleValues := by
  rw [measured_mean_is_two_for_both_absent_fill_values.1,
    unmasked_mean_changes_with_the_absent_fill_value.1]
  norm_num

example : preparedCell (· - ·) (· / ·) (0 : ℝ) 0 2 1 false = -2 := by
  rw [final_overwrite_removes_any_absent_cell]
  norm_num

example : maskedStd (1 / 2) (fun _ : Fin 1 => false) (fun _ => 7) = 1 / 2 := by
  have count : observedCount (fun _ : Fin 1 => false) = 0 :=
    all_absent_has_zero_count _ (by intro _; rfl)
  rw [(zero_observation_mean_and_scale_conventions (1 / 2) _ (fun _ => 7) count).2]
  norm_num

example : maskedMean (fun _ : Fin 1 => false) (fun _ => 7) = 1 := by
  have count : observedCount (fun _ : Fin 1 => false) = 0 :=
    all_absent_has_zero_count _ (by intro _; rfl)
  rw [(zero_observation_mean_and_scale_conventions (1 / 2) _ (fun _ => 7) count).1]
  norm_num

end N2P3.NegativeObservedStatistics
