import N2P3.ERPTemporalCLI
import N2P3.ERPTemporalMultiScope

/- Fixed P12 C01-C09 alternatives. Helpers are tested at their stated raw
field boundary. No malformed rank/artifact is claimed to pass a full reader. -/

namespace N2P3.NegativeERPTemporalCLI

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.ERPTemporalCLI

private theorem error_ne_success {ε α : Type} (e : ε) (a : α) :
    (Except.error e : Except ε α) ≠ .ok a := by intro h; cases h

private theorem success_ne_error {ε α : Type} (e : ε) (a : α) :
    (Except.ok a : Except ε α) ≠ .error e := by intro h; cases h

example : excludedStrings [.text ""] = .ok [""] := by
  rw [excluded_empty_string_refused]
  simp only [error_ne_success]

example : excludedStrings [.text " ",.text " "] =
    .error (.sourceReject "trimmed or duplicated excluded study") := by
  rw [excluded_whitespace_and_duplicates_are_not_newly_forbidden]
  simp only [success_ne_error]

example : sourceItems (.object .nil)=.error (.sourceReject "sources must be a list") := by
  rw [empty_object_sources_iterates_to_empty]
  simp only [success_ne_error]

example : sourceItems (.integer 0)=.ok [] := by
  rw [scalar_sources_is_operation_failure]
  simp only [error_ne_success]

example (p : PathPrimitives) (output : String) (header : ERPTemporalCLI.Header)
    (occupied : p.pathExists output=.ok true) : prepareAfterHeader p output header=.ok [] := by
  rw [existing_output_precedes_sources_subscription p header output occupied]
  simp only [error_ne_success]

example (p : PathPrimitives) (planPath path : String) (item : Fields)
    (field : item.get "cache"=some (.text path)) (converted : p.path (.text path)=.ok path)
    (absolute : p.isAbsolute path=true) : cachePath p planPath (.object item)=
      .error (.fieldFailure "resolve parent of missing plan") := by
  rw [absolute_cache_does_not_resolve_plan_parent p planPath item (.text path) path field converted absolute]
  simp only [success_ne_error]

example : (checkCLIRows 2 ⟨1,.signed,[0,0]⟩).map (fun selected => selected.indices.map Fin.val)=
    .error (.sourceReject "CLI refuses duplicates") := by
  rw [CLI_duplicate_rows_pass_but_producer_refuses.1]
  simp only [success_ne_error]

example : (checkProducerRows 2 ⟨1,.signed,[0,0]⟩).map (fun selected => selected.indices.map Fin.val)=
    .ok [0,0] := by
  rw [CLI_duplicate_rows_pass_but_producer_refuses.2]
  change (Except.error (.sourceReject "source temporal explicit unique in-range rows") : Check (List ℕ))=.ok [0,0]
  simp only [error_ne_success]

example (coerce : ArrayCoercion) : rowArray coerce 2 (.text "all")=
    .error (.fieldFailure "coercion failed") := by
  rw [all_token_constructs_actual_complete_range]
  simp only [success_ne_error]

example : (N2P3.ERPTemporalMultiScope.readRank (.boolean true) 3)=.ok 1 := by
  rw [N2P3.ERPTemporalMultiScope.Boolean_rank_is_not_an_integer]
  simp only [error_ne_success]

example : (N2P3.ERPTemporalMultiScope.readRank (.number 1) 3)=.ok 1 := by
  rw [N2P3.ERPTemporalMultiScope.float_rank_is_not_an_integer]
  simp only [error_ne_success]

end N2P3.NegativeERPTemporalCLI
