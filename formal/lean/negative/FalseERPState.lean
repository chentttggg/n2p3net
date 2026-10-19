import N2P3.ERPState

/-! Eight independent false claims, fixed from actual ERP source guards. -/

namespace N2P3.NegativeERPState
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState

example : fromState fixturePrimitives fixtureReference
    (Fields.ofList [("schema", .text "wrong"), ("feature_order", .text "channel_prime"), ("temporal_basis", .null)]) [] =
      .error (.fieldFailure "temporal basis record.get") := by
  rw [bad_schema_precedes_temporal_read]
  have hn : LedgerFault.sourceReject "ERP schema/order" ≠ .fieldFailure "temporal basis record.get" := by
    intro h; cases h
  simp only [Except.error.injEq, hn]

example : fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none")) badScalarState =
    .ok (fixturePlain .pooled) := by
  rw [rank_one_intercept_full_reader_rejects]
  have hn : (Except.error (.sourceReject "ERP state shape") : Check Model) ≠ .ok (fixturePlain .pooled) := by
    intro h; cases h
  simp only [hn]

example : fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none")) badWidthState =
    .ok (fixturePlain .pooled) := by
  rw [incorrect_coefficient_width_full_reader_rejects]
  have hn : (Except.error (.sourceReject "ERP state shape") : Check Model) ≠ .ok (fixturePlain .pooled) := by
    intro h; cases h
  simp only [hn]

example : fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none")) extraERPState =
    .ok (fixturePlain .pooled) := by
  rw [extra_erp_key_full_reader_rejects]
  have hn : (Except.error (.sourceReject "ERP state keys") : Check Model) ≠ .ok (fixturePlain .pooled) := by
    intro h; cases h
  simp only [hn]

example : erpFields (writePersonalERP (fixturePlain .pooled) ⟨"torch.int64", [1], [.integer 0]⟩
    [("unused", realTensor [1] [9])]) ≠ fixtureLinearState := by
  rw [personal_prefix_parser_recovers_fields _ _ _ (by rfl)]
  simp only [fixturePlain, ne_eq, not_true_eq_false]

example : fromState fixturePrimitives fixtureReference
    (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1) changedBasisState =
      fromState fixturePrimitives fixtureReference
        (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1) fixtureTemporalState := by
  rw [inconsistent_duplicate_temporal_basis_rejected, temporal_nonempty_construct]
  have hn : (Except.error (.sourceReject "temporal duplicate basis equality") : Check Model) ≠
      .ok ⟨⟨.temporal, 1, 1, .text "none", some (fixedTemporalModel fixturePrimitives fixtureFixedInput)⟩,
        1, 1, fixtureTemporalState, none, none⟩ := by intro h; cases h
  simp only [hn]

example : fromState fixturePrimitives fixtureReference
    (xdawnRecord 1 1 1 1 1 2 (.text "none")) fixtureXDawnState =
      fromState fixturePrimitives fixtureReference
        (xdawnRecord 1 1 1 1 1 1 (.text "none")) fixtureXDawnState := by
  rw [invalid_xdawn_effective_count_precedes_reference, xdawn_nonempty_construct]
  have hn : (Except.error (.sourceReject "xDAWN rank/effective") : Check Model) ≠
      .ok ⟨⟨.xdawn, 1, 1, .text "none", none⟩, 1, 1, fixtureXDawnState, some 1, some 1⟩ := by intro h; cases h
  simp only [hn]

example : fromState fixturePrimitives fixtureReference
    (plainRecord .toeplitz 1 1 1 1 (.text "mean_only")) fixtureLinearState = .ok (fixturePlain .toeplitz) := by
  rw [toeplitz_baseline_is_enforced]
  have hn : (Except.error (.sourceReject "Toeplitz baseline") : Check Model) ≠ .ok (fixturePlain .toeplitz) := by
    intro h; cases h
  simp only [hn]

end N2P3.NegativeERPState
