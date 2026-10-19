import N2P3.ERPTemporalProducer

/-! Fixed source alternatives from P12 preparation: sorting requested rows,
global montage intersection, compressed event IDs, reversed subtraction,
reduced observed-mask hashes, wrong dtype, repeated authority, insufficient
spectral rank, ascending retained columns and a magically refreshed parent.
The decoded fixture is not a physically admitted EpochDataset. -/

namespace N2P3.NegativeERPTemporalProducer

open Matrix N2P3.FrozenState N2P3.StateLedger N2P3.ERPTemporalProducer

/- Pure JSON comparison alternatives only. These are not whole artifacts,
and no illegal Boolean rank is claimed to pass any actual reader. -/
example : ¬ replayRecordEqual (Fields.ofList [("cell", .integer 1)])
    (Fields.ofList [("cell", .number 1)]) := by
  simp only [replay_comparison_integer_float_tags, not_true_eq_false]

example : ¬ replayRecordEqual (Fields.ofList [("cell", .boolean true)])
    (Fields.ofList [("cell", .integer 1)]) := by
  simp only [replay_comparison_boolean_numeric_tags, not_true_eq_false]

example : ¬ replayRecordEqual (Fields.ofList [("a", .text "x"),("b", .text "y")])
    (Fields.ofList [("b", .text "y"),("a", .text "x")]) := by
  have same := replay_comparison_ignores_dictionary_key_order "a" "b" (.text "x") (.text "y") (by decide)
  simp only [same, not_true_eq_false]

example : selectedRows fixture fixtureRequested "A" = [0,1,2,3] := by
  rw [fixture_selected_rows.1]
  have hn : ([3,0,2,1] : List (Fin 8)) ≠ [0,1,2,3] := by decide
  simp only [hn]

example : commonChannels fixture [7,4,6,5] = [0] := by
  rw [fixture_different_local_montages.2.1]
  have hn : ([1,2] : List (Fin 3)) ≠ [0] := by decide
  simp only [hn]

/- Replacing the participant-local B montage by the actual global
intersection, which is empty, is a distinct false alternative. -/
example : commonChannels fixture [7,4,6,5] = [] := by
  rw [fixture_different_local_montages.2.1]
  have hn : ([1,2] : List (Fin 3)) ≠ [] := by decide
  simp only [hn]

example : (0 : Fin 3) ∈ commonChannels fixtureVariableMask [3,0,2,1] := by
  rw [all_observed_is_not_any_observed_on_varying_rows.1]
  simp only [List.not_mem_nil]

example (initial : Fin 8 → Option (Fin 10)) :
    originalEventIds fixture initial [3,0,2,1] = ["e3","e0","e2","e1"] := by
  rw [(fixture_original_event_names initial).1]
  have hn : (["e3","e2","e0","e4"] : List String) ≠ ["e3","e0","e2","e1"] := by decide
  simp only [hn]

example : sourceContrast fixture [3,0,2,1] [0] (0 : Fin 1) (0 : Fin 3) = -2 := by
  rw [fixture_signed_contrasts.1]
  norm_num

example : (observedImage fixture [3,0,2,1]).shape = [4,1] := by
  change ([4,3] : List ℕ) = [4,1]
  norm_num

example : (valuesImage fixture [3,0,2,1] [0]).dtype = "<f4" := by
  change "<f8" = "<f4"
  have hn : "<f8" ≠ "<f4" := by decide
  simp only [hn]

example (p : NumericHashPrimitives) (initial : Fin 8 → Option (Fin 10)) :
    buildContribution p fixture fixtureRequested "A" initial ["A"] =
      .ok (literalContribution p fixture fixtureRequested "A" initial) := by
  rw [duplicate_authority_refused_before_numeric_work p fixture fixtureRequested "A" initial
    ["A"] (by decide)]
  have hn : (Except.error (.sourceReject "source temporal duplicate authority") : Check (Contribution 3)) ≠
      .ok (literalContribution p fixture fixtureRequested "A" initial) := by
    intro h
    cases h
  simp only [hn]

example : spectralSelect 2 (by decide) (show 2 ≤ 3 by decide)
    (⟨![0,0,2], 1⟩ : EighImage 3) = .ok (fun _ _ => 0) := by
  rw [failed_relative_cutoff_refuses_even_with_positive_trace 2 (by decide) (by decide)
    (⟨![0,0,2], 1⟩ : EighImage 3) (by norm_num)]
  have hn : (Except.error (.sourceReject "source temporal identifiable rank") :
      Check (Matrix (Fin 3) (Fin 2) ℝ)) ≠ .ok (fun _ _ => 0) := by
    intro h
    cases h
  simp only [hn]

example : (retainedIndex 2 (show 2 ≤ 3 by decide) (0 : Fin 2)).val = 0 := by
  change (2 : ℕ) = 0
  norm_num

example (p : NumericHashPrimitives) :
    (receipt p {fixture with
      storedParent := some (.text "stored")
      values := fun row c t => fixture.values row c t + 1}
      fixtureRequested "A" (fun _ => none)).get "parent_cache_attestation" = some (.text "fresh") := by
  rw [receipt_parent_is_stored_identity]
  change (some (.text "stored") : Option Value) = some (.text "fresh")
  have hn : (some (.text "stored") : Option Value) ≠ some (.text "fresh") := by
    intro h
    have he : ("stored" : String) = "fresh" := Value.text.inj (Option.some.inj h)
    have ht : ("stored" : String) ≠ "fresh" := by decide
    exact ht he
  simp only [hn]

end N2P3.NegativeERPTemporalProducer
