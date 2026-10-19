import N2P3.PersonalERPTransport

/-! False alternatives distinguish decoded dtype projection, prefix handling,
selector filtering, adapter overwrite and the actual metadata priorities.
They do not claim to exercise Torch/NumPy floating-point or ownership semantics.
-/

namespace N2P3.NegativePersonalERPTransport
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.PersonalERPTransport

example : (torchTensor fixtureArray).dtype = (numpyTensor fixtureArray).dtype := by
  have h := numpy_dtype_is_separate_from_torch fixtureArray
  simp only [h]

example : N2P3.ERPState.stripERP "erp.erp.coefficient" = some "coefficient" := by
  have hstrip : N2P3.ERPState.stripERP "erp.erp.coefficient" = some "erp.coefficient" := rfl
  rw [hstrip]
  have h : (some "erp.coefficient" : Option String) ≠ some "coefficient" := by decide
  simp only [h]

example : N2P3.ERPState.erpFields [("waveform_channels", fixtureSelected)] =
    [("waveform_channels", fixtureSelected)] := by
  rw [waveform_selector_is_not_inner_state]
  have h : ([] : NamedState) ≠ [("waveform_channels", fixtureSelected)] := by simp
  simp only [h]

example : lookupTensor "erp.coefficient"
    (afterAdapter (producerERP fixtureExport fixtureSelected)
      (some [("input_mean", fixtureSelected)])) = some (torchTensor fixtureArray) := by
  rw [adapter_can_remove_an_erp_key]
  have h : (none : Option Tensor) ≠ some (torchTensor fixtureArray) := by simp
  simp only [h]

example : sourceFlag false true = .integer 1 := by
  have h := source_flag_is_not_numeric_one
  simp only [h]

example : sourceFlag false false = .integer 0 := by
  have h := source_flag_is_not_numeric_zero
  simp only [h]

example : producedKind false true true true = "source_temporal_target_erp" := by
  rw [adapter_priority_precedes_temporal_source]
  have h : ("scratch_n2p3" : String) ≠ "source_temporal_target_erp" := by decide
  simp only [h]

example : producedKind true true true true = "scratch_n2p3" := by
  rw [source_initialization_has_first_priority]
  have h : ("source_initialized_n2p3" : String) ≠ "scratch_n2p3" := by decide
  simp only [h]

end N2P3.NegativePersonalERPTransport
