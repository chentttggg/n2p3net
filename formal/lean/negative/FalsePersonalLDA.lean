import N2P3.PersonalLDA

/-! Six deliberate failures, fixed from source guards and evaluator coordinates. -/

namespace N2P3.NegativePersonalLDA

open N2P3.PersonalLDA

-- Missing waveform_channels means every channel, including the missing second one.
example : availabilityGate .rejectBatch baseAvailable partialObserved none = .ok baseAvailable := by
  rw [default_missing_channel_rejects]
  have hn : (Except.error () : Except Unit (Fin 1 → Bool)) ≠ .ok baseAvailable := by
    intro h
    cases h
  simp only [hn]

-- Soft policy masks a row that lacks a selected channel.
example : availabilityGate .maskRows baseAvailable partialObserved none = .ok baseAvailable := by
  rw [soft_missing_channel_masks_row]
  have hn : baseUnavailable ≠ baseAvailable := by
    intro h
    have hf := congrFun h 0
    change false = true at hf
    cases hf
  simp only [Except.ok.injEq, hn]

-- A fully observed first row cannot bypass another row's hard batch rejection.
example : availabilityGate .rejectBatch twoRowBase twoRowObserved none = .ok twoRowBase := by
  rw [hard_partial_batch_rejects_fully_observed_row]
  have hn : (Except.error () : Except Unit (Fin 2 → Bool)) ≠ .ok twoRowBase := by
    intro h
    cases h
  simp only [hn]

-- C-order flatten keeps time samples contiguous within each selected channel.
example : waveformFeatures identityTrace orderedChannels orderedEpoch secondCoordinate = 3 := by
  have hn : (2 : ℝ) ≠ 3 := by norm_num
  simp only [channel_major_second_feature, hn]

-- Recovering mean, scale and intercept cannot compensate for a changed coefficient.
example : evaluate changedCoefficient legalFeatures = evaluate legalState legalFeatures := by
  have hn : (47 : ℝ) ≠ 45 := by norm_num
  simp only [changed_coefficient_changes_score, legal_saved_score, hn]

-- The real-domain constructor still enforces strictly positive scale.
example : loadState (writeState zeroScaleState) = some zeroScaleState := by
  rw [zero_scale_rejected]
  have hn : (none : Option (State 2)) ≠ some zeroScaleState := by
    intro h
    cases h
  simp only [hn]

end N2P3.NegativePersonalLDA
