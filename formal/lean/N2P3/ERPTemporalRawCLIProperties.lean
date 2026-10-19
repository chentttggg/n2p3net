import N2P3.ERPTemporalRawCLI

/-! Source-fixed raw CLI obligations in p12_raw_cli_preparation_obligations_01.
These claims concern the actual pre-fit field/row boundaries. They do not
assume that a malformed Dataset has passed full validation, nor assert full
fit_plan success, a numerical fit or native-library refinement. -/

namespace N2P3.ERPTemporalRawCLIProperties

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPTemporalCache
open N2P3.ERPTemporalSnapshot N2P3.ERPTemporalNPZ N2P3.ERPTemporalJSON
open N2P3.ERPTemporalRawCLI

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

theorem malformed_row_rank_refused (rows : RawArray) (rank : rows.shape.length ≠ 1) :
    checkRowPrefix rows=.error (.sourceReject "source temporal explicit valid rows") := by
  simp [checkRowPrefix,ensure,rank]

theorem Boolean_row_dtype_refused (rows : RawArray) (rank : rows.shape.length=1)
    (kind : rows.kind=.boolean) :
    checkRowPrefix rows=.error (.sourceReject "source temporal explicit valid rows") := by
  simp [checkRowPrefix,ensure,rank,kind,Kind.integer]

theorem floating_row_dtype_refused (rows : RawArray) (rank : rows.shape.length=1)
    (kind : rows.kind=.floating) :
    checkRowPrefix rows=.error (.sourceReject "source temporal explicit valid rows") := by
  simp [checkRowPrefix,ensure,rank,kind,Kind.integer]

/-- The RHS makes no reference to the tensor shape: even scalar X cannot
move its late shape[0] exception ahead of a failed row-prefix check. -/
theorem row_prefix_fault_precedes_tensor_shape {World : Type}
    (dataset : ERPTemporalRawCacheDecode.Dataset) (rows : RawArray)
    (fault : LedgerFault) (failed : checkRowPrefix rows=.error fault)
    (world : World) (result : EStateM.Result Fault World Unit) :
    checkRows dataset rows world result ↔ result=.error (.check fault) world := by
  simp [checkRows,checked,small,traceBind,failed]

theorem scalar_tensor_has_no_invented_zero_epoch_count
    (dataset : ERPTemporalRawCacheDecode.Dataset) (scalar : dataset.X.1.shape=[]) :
    epochCount dataset=.error (.fieldFailure "dataset.n_epochs X.shape[0]") := by
  simp [epochCount,scalar]

def rawNaNRankPlan : JFields := JFields.ofList
  [("schema",.text "n2p3_source_temporal_plan/1"),
   ("excluded_target_studies",.sequence (.cons (.text "heldout") .nil)),
   ("sources",.sequence .nil),("rank",.floating .nan)]

theorem raw_nan_rank_is_retained_by_header :
    (checkHeader (.object rawNaNRankPlan)).map Header.raw=.ok rawNaNRankPlan := by
  simp [checkHeader,rawNaNRankPlan,ERPTemporalRawFields.objectAt,
    ERPTemporalRawFields.fieldDefault,ERPTemporalRawFields.required,JFields.ofList,JFields.get,
    pythonEqual,scalarNaN,excludedStrings,ensure,JValues.toList]

theorem retained_raw_plan_has_no_blanket_finite_projection :
    finiteFields rawNaNRankPlan=none := by
  simp [rawNaNRankPlan,JFields.ofList,finiteFields,finiteView,finiteValues]

theorem raw_noniterable_nan_sources_is_an_iteration_fault :
    sourceItems (.floating .nan)=.error (.fieldFailure "iterate source temporal sources") := rfl

end
end N2P3.ERPTemporalRawCLIProperties
