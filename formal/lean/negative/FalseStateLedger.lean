import N2P3.StateLedger

/-! Five deliberate failures at the actual constructor-ledger boundaries. -/

namespace N2P3.NegativeStateLedger

open N2P3.StateLedger N2P3.FrozenState

-- Constructor acceptance of falsy None records does not license record traversal.
example : checkContinuous 3 (analyticLedger (.boolean true) .null) = .ok () →
    RecordIdTraversalReady (analyticLedger (.boolean true) .null) := by
  intro _
  have hn := none_false_zero_fail_record_iteration.1
  simp only [hn]

-- Python bool is excluded by the exact integer count guard.
example : checkRecord 3 booleanCountRecord = .ok () := by
  rw [boolean_event_count_rejected]
  have hn : (Except.error (.sourceReject "continuous_record_counts") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- The Continuous scalar ridge path accepts True as numeric one.
example : checkContinuous 3 booleanRidgeTraining ≠ .ok () := by
  simp only [boolean_ridge_training_accepted, ne_eq, not_true_eq_false]

-- Joint rejects boolean shrinkage even when every preceding ledger guard passes.
example : checkJoint booleanShrinkageJoint = .ok () := by
  rw [boolean_shrinkage_full_ledger_rejected]
  have hn : (Except.error (.sourceReject "joint_shrinkage") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- Full ID collection precedes the per-record exact-key check.
example : checkContinuous 3 wrongKeysThenMissingId =
    .error (.sourceReject "continuous_record_keys") := by
  rw [record_id_collection_precedes_record_key_rejection]
  have hn : (LedgerFault.fieldFailure "missing:record_id") ≠
      LedgerFault.sourceReject "continuous_record_keys" := by
    intro h
    cases h
  simp only [Except.error.injEq, hn]

end N2P3.NegativeStateLedger
