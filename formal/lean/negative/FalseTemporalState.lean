import N2P3.TemporalState

/-! Eight deliberate failures at the actual temporal constructor/record gates.
Fixtures are nonempty AR(1) models or the explicit complete 33-sample ledger. -/

namespace N2P3.NegativeTemporalState

open N2P3.FrozenState N2P3.StateLedger N2P3.TemporalState

-- True is a valid numeric coefficient coercion, but never a variance here.
example : readVariance (.boolean true) = .ok 1 := by
  rw [boolean_variance_rejected]
  have hn : (Except.error (.sourceReject "positive innovation variance") : Check ℝ) ≠ .ok 1 := by
    intro h
    cases h
  simp only [hn]

-- Temporal analytic reference IDs must be textual and nonblank.
example : checkProvenance implementationDigest 1
    (analyticLedger (.boolean true) (.sequence .nil)) = .ok () := by
  rw [boolean_reference_rejected]
  have hn : (Except.error (.sourceReject "temporal analytic reference") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- A valid fitted ledger does not replace the actual Burg reflection sequence.
example : selectReflections fixturePrimitives {fixtureInput .stationaryInnovations with
    provenance:=legalTrainingLedger} = .ok [-3/5] := by
  rw [fitted_innovations_requires_actual_reflections]
  have hn : (Except.error (.sourceReject "actual fitted PARCOR required") : Check (List ℝ)) ≠
      .ok [-3/5] := by intro h; cases h
  simp only [hn]

-- A correctly formatted, otherwise valid ledger must match the running implementation.
example : checkProvenance implementationDigest 1 foreignTrainingLedger = .ok () := by
  rw [valid_foreign_ledger_fails_implementation_identity]
  have hn : (Except.error (.sourceReject "temporal implementation identity") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- /1 defaults to stationary; an extra mode field is still refused by exact replay keys.
example : checkReplay fixturePrimitives (extraModeState fixturePrimitives (fixtureCore .stationary))
    (fixtureCore .stationary) = .ok () := by
  rw [schema_one_extra_mode_fails_exact_keys]
  have hn : (Except.error (.sourceReject "temporal numeric replay") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- Recomputed radius diagnostics cannot be replaced by a caller's resealed value.
example : checkReplay fixturePrimitives (changedRadiusState fixturePrimitives (fixtureCore .stationary))
    (fixtureCore .stationary) = .ok () := by
  rw [changed_radius_fails_recomputed_diagnostic]
  have hn : (Except.error (.sourceReject "temporal numeric replay") : Check Unit) ≠ .ok () := by
    intro h
    cases h
  simp only [hn]

-- The raw float-ready domain accepts numeric strings and normalizes their tags.
example : floatVector fixturePrimitives (.sequence (valuesOfList [.text "0.6"])) ≠ .ok [3/5] := by
  rw [finite_string_coefficients_normalize]
  simp only [ne_eq,not_true_eq_false]

-- A corrupted outer seal is refused before an unsupported/missing contract.
example : fromRecord fixturePrimitives (.object (.cons "artifact_sha256" (.text "wrong") .nil)) =
    .error (.sourceReject "temporal contract") := by
  rw [bad_record_seal_precedes_contract _ _ _ rfl (by simp [fixturePrimitives])]
  have hn : LedgerFault.sourceReject "temporal artifact digest" ≠ .sourceReject "temporal contract" := by decide
  simp only [Except.error.injEq,hn]

end N2P3.NegativeTemporalState
