import N2P3.OperatorRowSupport

/-! The v3 normalized-row guard cannot inherit the historical batch dilution;
zero rows and a zero-width reduction are distinct source operation domains. -/

namespace N2P3.NegativeOperatorRowSupport
open N2P3.StateLedger N2P3.OperatorRowSupport

example : validateSupport exactNorm axisProjector [unsupportedRow, largeSupportedRow] = .ok () := by
  rw [unsupported_plus_large_supported_refuses]
  have hn : (Except.error (.sourceReject "frozen covariance identifiable support") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

example : validateSupport exactNorm axisProjector [] =
    .error (.sourceReject "frozen covariance identifiable support") := by
  rw [positive_width_empty_batch_passes 2 (by decide)]
  have hn : (Except.ok () : Check Unit) ≠ .error (.sourceReject "frozen covariance identifiable support") := by
    intro h; cases h
  simp only [hn]

example : validateSupport (exactNorm (d := 0)) (fun _ _ => 0) [] = .ok () := by
  rw [zero_width_is_empty_reduction_failure]
  have hn : (Except.error (.fieldFailure "np.max empty last axis") : Check Unit) ≠ .ok () := by
    intro h; cases h
  simp only [hn]

end N2P3.NegativeOperatorRowSupport
