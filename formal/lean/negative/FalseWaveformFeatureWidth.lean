import N2P3.WaveformFeatureWidth

/-! Physical shape, selected count, no-upsampling, rank, scale and arithmetic
failure alternatives fixed by the source before execution of these controls.
The rational arithmetic fixture is restricted to integral quotients.
-/
namespace N2P3.NegativeWaveformFeatureWidth
open N2P3.StateLedger N2P3.WaveformFeatureWidth

example : downsampleLength integralFixtureArithmetic 2 10 = .ok 4 := by
  rw [nonempty_no_upsampling_length]
  have h : (Except.ok 2 : Check ℤ) ≠ .ok 4 := by
    intro he
    have hn := Except.ok.inj he
    norm_num at hn
  simp only [h]

example : waveformWidthGuard integralFixtureArithmetic 3 25 250 none obsoleteTwoFields = .ok () := by
  rw [matching_old_shape_fields_do_not_suffice]
  have h : (Except.error (.sourceReject "calibrated LDA feature state has invalid shape or scale") : Check Unit) ≠ .ok () := by
    intro he; cases he
  simp only [h]

example : waveformWidthGuard integralFixtureArithmetic 3 25 250 (some [2,0]) sixFields = .ok () := by
  rw [selected_subset_cannot_reuse_all_channel_width]
  have h : (Except.error (.sourceReject "calibrated LDA feature state has invalid shape or scale") : Check Unit) ≠ .ok () := by
    intro he; cases he
  simp only [h]

example : featureGuard {sixFields with scaleAllPositive := false} 6 = .ok () := by
  rw [nonpositive_scale_refuses _ _ rfl]
  have h : (Except.error (.sourceReject "calibrated LDA feature state has invalid shape or scale") : Check Unit) ≠ .ok () := by
    intro he; cases he
  simp only [h]

example : featureGuard ⟨[2,3], [2,3], [2,3], true, 1⟩ 6 = .ok () := by
  rw [equal_numel_does_not_replace_coefficient_rank]
  have h : (Except.error (.sourceReject "calibrated LDA feature state has invalid shape or scale") : Check Unit) ≠ .ok () := by
    intro he; cases he
  simp only [h]

example : waveformWidthGuard integralFixtureArithmetic 3 25 0 none sixFields = .ok () := by
  rw [zero_sfreq_operation_failure_is_not_constructor_success]
  have h : (Except.error (.fieldFailure "division:ZeroDivisionError") : Check Unit) ≠ .ok () := by
    intro he; cases he
  simp only [h]

end N2P3.NegativeWaveformFeatureWidth
