import N2P3.ERPTemporalMultiScope

namespace N2P3.NegativeERPTemporalMultiScope

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPTemporalMultiScope

example (reference : ℕ) (dataset : Dataset) (state : State) :
    (match storeDataset reference dataset state with
      | .ok _ after | .error _ after => after.heap reference)=none := by
  simp [storeDataset]

example (reference : ℕ) (dataset : Dataset) (state : State) (fault : LedgerFault) :
    (match ((storeDataset reference dataset >>= fun _ => attempt (.error fault : Check Unit)) state) with
      | .ok _ after | .error _ after => after.heap reference)=none := by
  rw [heap_write_survives_later_failure]
  simp only [ite_true,Option.some_ne_none]

example (p : HashOperations) (image contiguous : HashImage) (header : List UInt8)
    (fault : LedgerFault) (coerced : p.asContiguous image=.ok contiguous)
    (encoded : p.dtypeShapeHeader contiguous.dtype contiguous.shape=.ok header)
    (failed : p.startHash header=.error fault) : numericHash p image=.ok "claimed digest" := by
  rw [hash_initialization_failure_precedes_bytes p image contiguous header fault coerced encoded failed]
  have h : (Except.error fault : Check String) ≠ .ok "claimed digest" := by intro he; cases he
  simp only [h]

example (frame : Frame) (rows : List (Fin frame.dataset.epochs))
    (wide : frame.dataset.channels ≠ 1) : (observedImage frame rows).shape=[rows.length,1] := by
  simp only [observed_digest_keeps_full_original_width,List.cons.injEq,
    true_and,and_true,wide]

end N2P3.NegativeERPTemporalMultiScope
