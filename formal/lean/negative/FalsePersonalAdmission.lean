import N2P3.PersonalAdmission

namespace N2P3.PersonalAdmission.Negative

open N2P3.FrozenState

/-- False: direct timestamp comparison rejects NaN without the official finite check. -/
example : sessionReason exampleScope "later" .nan ≠ none := by
  rw [direct_nan_comparison_passes exampleScope "later" (by simp [exampleScope])]
  simp only [ne_eq, not_true_eq_false]

/-- False: an independent session starting exactly at label availability passes. -/
example : sessionReason exampleScope "later" (.finite 100) = none := by
  have h := equality_is_not_prospective exampleScope "later" (by simp [exampleScope])
  change sessionReason exampleScope "later" (.finite 100) = some .beforeAvailable at h
  rw [h]
  simp only [Option.some_ne_none]

private def missingRow : TrialRow := ⟨none, .nan⟩

private theorem missing_row_scope_passes :
    scopeReason exampleScope ⟨true, true, [missingRow]⟩ = none := by
  apply all_missing_ids_pass_scope
  intro row hr
  have he := List.mem_singleton.mp hr
  subst row
  rfl

/-- False: successful scope checking proves a finite time on skipped missing-ID rows. -/
example : scopeReason exampleScope ⟨true, true, [missingRow]⟩ = none →
    ∃ t, missingRow.started = .finite t := by
  rw [missing_row_scope_passes]
  simp [missingRow]

/-- False: Python finite scalar equality distinguishes True and numeric one. -/
example : ¬PyEq (.boolean true) (.number 1) := by
  simp only [python_true_equals_numeric_one, not_true_eq_false]

private theorem missing_field_differs_from_present_null :
    ¬PyEq (.object .nil) (.object (Fields.ofList [("x", .null)])) := by
  intro h
  have hv := congrArg (fun v : PyValue => match v with | .object f => f "x" | _ => none) h
  simp [normalizeValue, normalizeFields, Fields.ofList] at hv

/-- False: replacing an absent key by present-null preserves Python dictionary equality. -/
example : PyEq (.object .nil) (.object (Fields.ofList [("x", .null)])) := by
  simp only [missing_field_differs_from_present_null]

/-- False: the no-usable-evidence CLI branch still returns a classifier evaluation. -/
example (contract : InputContract) : inferOutcome exampleScope contract
    .null false 36 (.error .noUsableEvidence) (fun _ => .ok ()) = .ok (.evaluated ()) := by
  simp [inferOutcome]

end N2P3.PersonalAdmission.Negative
