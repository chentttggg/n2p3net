import N2P3.ERPTemporalRawCLIProperties

/- Fixed wrong alternatives from the source-derived raw CLI obligations.
They test the pre-fit boundaries; no complete raw admission is assumed. -/

namespace N2P3.NegativeERPTemporalRawCLI

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPTemporalNPZ
open N2P3.ERPTemporalJSON N2P3.ERPTemporalRawCLI N2P3.ERPTemporalRawCLIProperties

private theorem error_ne_success {ε α : Type} (error : ε) (value : α) :
    (Except.error error : Except ε α) ≠ .ok value := by intro equal; cases equal

private theorem success_ne_error {ε α : Type} (error : ε) (value : α) :
    (Except.ok value : Except ε α) ≠ .error error := by intro equal; cases equal

example (dataset : N2P3.ERPTemporalRawCacheDecode.Dataset) (scalar : dataset.X.1.shape=[]) :
    epochCount dataset=.ok 0 := by
  rw [scalar_tensor_has_no_invented_zero_epoch_count dataset scalar]
  simp only [error_ne_success]

example (rows : RawArray) (rank : rows.shape.length=1) (kind : rows.kind=.boolean) :
    checkRowPrefix rows=.ok () := by
  rw [Boolean_row_dtype_refused rows rank kind]
  simp only [error_ne_success]

example : (checkHeader (.object rawNaNRankPlan)).map Header.raw=
    .error (.sourceReject "all decoded plan fields must be finite before the header") := by
  rw [raw_nan_rank_is_retained_by_header]
  simp only [success_ne_error]

end N2P3.NegativeERPTemporalRawCLI
