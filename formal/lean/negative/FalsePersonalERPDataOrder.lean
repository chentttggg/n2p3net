import N2P3.PersonalERPData

/-! Four wrong source-order alternatives. Each must leave exactly one
unsolved False; their proposition text is independent of proof repair. -/

namespace N2P3.NegativePersonalERPDataOrder
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
noncomputable section

-- An integer rank-one ndarray does not report dtype before rank.
example : checkTensorGeometry ⟨.integer,[0],[],by rfl⟩ 0 0=
    .error (.sourceReject "floating EEG tensor dtype") := by
  rw [rank_and_dtype_fault_reports_rank_first]
  have hn : (Except.error (.fieldFailure "EEG tensor rank three") : Check ℕ) ≠
      .error (.sourceReject "floating EEG tensor dtype") := by
    intro he
    cases he
  simp only [hn]

-- Invalid position geometry is checked before an invalid static mask dtype.
example (norm : CoordinateNorm) {N C T M : ℕ} (data : DatasetImage N C T M)
    (h : data.positions.shape ≠ [C,3] ∧ data.staticImage.dtype ≠ .boolean) :
    checkPositionAndMasks norm data=.error (.sourceReject "static channel mask array") := by
  rw [position_header_fault_precedes_mask_fault norm data h]
  have hn : (Except.error (.sourceReject "physical channel positions geometry") : Check Unit) ≠
      .error (.sourceReject "static channel mask array") := by
    intro he
    have hs : "physical channel positions geometry"="static channel mask array" := LedgerFault.sourceReject.inj (Except.error.inj he)
    have hn : "physical channel positions geometry" ≠ "static channel mask array" := by decide
    exact hn hs
  simp only [hn]

-- Static absence cannot be hidden behind a later trial-array fault.
example {N C T M : ℕ} (data : DatasetImage N C T M)
    (h : data.staticImage.dtype=.boolean ∧ data.staticImage.shape=[C] ∧ ¬∃ c, data.static c=true) :
    checkArrayMasks data=.error (.sourceReject "trial channel mask array") := by
  rw [static_absence_precedes_trial_array_checks data h]
  have hn : (Except.error (.sourceReject "at least one static observed channel") : Check Unit) ≠
      .error (.sourceReject "trial channel mask array") := by
    intro he
    have hs : "at least one static observed channel"="trial channel mask array" := LedgerFault.sourceReject.inj (Except.error.inj he)
    have hn : "at least one static observed channel" ≠ "trial channel mask array" := by decide
    exact hn hs
  simp only [hn]

-- Every observed finite check precedes any coordinate norm failure.
example (norm : CoordinateNorm) {N C T M : ℕ} (data : DatasetImage N C T M)
    (h : ¬∀ c ∈ (List.finRange C).filter data.static, ∀ coordinate,
      finiteNumber (data.coordinates c coordinate)=true) :
    checkObservedPositions norm data=.error (.sourceReject "registered metre head frame") := by
  rw [all_observed_finite_checks_precede_any_norm norm data h]
  have hn : (Except.error (.sourceReject "finite observed coordinates") : Check Unit) ≠
      .error (.sourceReject "registered metre head frame") := by
    intro he
    have hs : "finite observed coordinates"="registered metre head frame" := LedgerFault.sourceReject.inj (Except.error.inj he)
    have hn : "finite observed coordinates" ≠ "registered metre head frame" := by decide
    exact hn hs
  simp only [hn]

end
end N2P3.NegativePersonalERPDataOrder
