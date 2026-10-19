import N2P3.ERPTemporalRawExecution

/- Source-fixed absent/bundle/hash alternatives in raw_execution_obligations01. -/
namespace N2P3.NegativeERPTemporalRawExecution

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPScalar N2P3.ERPTemporalJSON
open N2P3.ERPTemporalRawExecution

private theorem error_ne_success {ε α : Type} (error : ε) (value : α) :
    (Except.error error : Except ε α) ≠ .ok value := by intro equal; cases equal

private theorem success_ne_error {ε α : Type} (error : ε) (value : α) :
    (Except.ok value : Except ε α) ≠ .error error := by intro equal; cases equal

example (p : Operations) (expected : JFields) :
    executionBinding p expected none=.error (.sourceValueError "absence validates expected spec") := by
  rw [absent_record_skips_all_declaration_and_hash_operations]
  simp only [success_ne_error]

example (p : Operations) (expected : JFields) :
    checkRecord p expected (.object rawUnknownBundle)=
      .error (.sourceValueError "reject all nonfinite bundle fields before children") := by
  rw [(bundle_headers_do_not_imply_all_fields_finite p expected).1]
  simp only [success_ne_error]

example (p : Operations) (expected : JFields) (fault : Fault)
    (failed : p.semanticHash (.object rawUnknownBundle)=.error fault) :
    executionBinding p expected (some (.object rawUnknownBundle))=.ok none := by
  rw [bundle_nonfinite_data_reaches_actual_final_hash p expected fault failed]
  simp only [error_ne_success]

end N2P3.NegativeERPTemporalRawExecution
